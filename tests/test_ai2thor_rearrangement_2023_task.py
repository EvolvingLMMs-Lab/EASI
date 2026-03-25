"""Tests for AI2THORRearrangement2023Task (offline, no simulator)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from easi.core.episode import EpisodeRecord, Observation, StepResult


def _make_task():
    from easi.tasks.ai2thor_rearrangement_2023.task import AI2THORRearrangement2023Task
    return AI2THORRearrangement2023Task(
        split_yaml_path=Path(__file__).resolve().parent.parent
        / "easi" / "tasks" / "ai2thor_rearrangement_2023" / "ai2thor_rearrangement_2023_val.yaml"
    )


class TestTaskRegistration:
    def test_all_splits_registered(self):
        from easi.tasks.registry import list_tasks
        tasks = list_tasks()
        for split in ("train", "eval_train", "val", "test", "combined"):
            assert f"ai2thor_rearrangement_2023_{split}" in tasks

    def test_simulator_key(self):
        from easi.tasks.registry import get_task_entry
        entry = get_task_entry("ai2thor_rearrangement_2023_val")
        assert entry.simulator_key == "ai2thor:v5_0_0"

    def test_max_steps(self):
        from easi.tasks.registry import get_task_entry
        entry = get_task_entry("ai2thor_rearrangement_2023_val")
        assert entry.max_steps == 500


class TestActionSpace:
    def test_action_count(self):
        task = _make_task()
        assert len(task.action_space) == 84

    def test_no_duplicates(self):
        task = _make_task()
        assert len(task.action_space) == len(set(task.action_space))

    def test_key_actions_present(self):
        task = _make_task()
        actions = set(task.action_space)
        assert "done" in actions
        assert "move_ahead" in actions
        assert "drop_held_object_with_snap" in actions
        assert "pickup_bowl" in actions
        assert "open_by_type_fridge" in actions
        assert "open_by_type_cabinet" in actions


class TestFormatResetConfig:
    def test_basic_format(self):
        task = _make_task()
        episode = {
            "id": "FloorPlan21__0",
            "scene": "FloorPlan21",
            "agent_position": json.dumps({"x": -1.0, "y": 0.87, "z": -1.0}),
            "agent_rotation": 30,
            "starting_poses": json.dumps([{"name": "Bowl_123", "position": {"x": 0, "y": 1, "z": 0}}]),
            "target_poses": json.dumps([{"name": "Bowl_123", "position": {"x": 1, "y": 1, "z": 1}}]),
            "openable_data": json.dumps([]),
            "instruction": "Move the Bowl back.",
        }
        config = task.format_reset_config(episode)
        assert config["scene"] == "FloorPlan21"
        assert config["agent_rotation"] == 30
        # JSON strings passed through to bridge
        assert isinstance(config["starting_poses"], str)
        assert isinstance(config["target_poses"], str)


class TestEvaluateEpisode:
    def test_success_metrics(self):
        task = _make_task()
        episode = {"id": "test"}
        obs = Observation(rgb_path="/tmp/fake.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False, info={
                "action_name": "move_ahead", "action_success": True,
            }),
            StepResult(observation=obs, reward=0.0, done=False, info={
                "action_name": "pickup_bowl", "action_success": True,
            }),
            StepResult(observation=obs, reward=0.0, done=True, info={
                "success": 1.0, "prop_fixed_strict": 1.0, "energy_prop": 0.0,
                "num_initially_misplaced": 1, "num_fixed": 1,
                "num_newly_misplaced": 0, "num_broken": 0,
                "action_name": "done", "action_success": True,
            }),
        ]
        metrics = task.evaluate_episode(episode, trajectory)
        assert metrics["success"] == 1.0
        assert metrics["prop_fixed_strict"] == 1.0
        assert metrics["num_steps"] == 3.0
        # PuSR: 1 successful pickup / 1 total pickup = 1.0
        assert metrics["pickup_success_rate"] == 1.0
        # PuLen: 1 pickup action
        assert metrics["num_pickup_actions"] == 1.0
        # SuLen: episode succeeded, so success_length = 3
        assert metrics["success_length"] == 3.0

    def test_failed_episode(self):
        task = _make_task()
        obs = Observation(rgb_path="/tmp/fake.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=True, info={
                "success": 0.0, "prop_fixed_strict": 0.0, "energy_prop": 0.8,
                "num_initially_misplaced": 2, "num_fixed": 0,
                "num_newly_misplaced": 0, "num_broken": 0,
                "action_name": "done", "action_success": True,
            }),
        ]
        metrics = task.evaluate_episode({"id": "test"}, trajectory)
        assert metrics["success"] == 0.0
        # SuLen should be NaN for failed episodes
        assert math.isnan(metrics["success_length"])

    def test_empty_trajectory(self):
        task = _make_task()
        metrics = task.evaluate_episode({"id": "test"}, [])
        assert metrics["success"] == 0.0
        assert metrics["num_steps"] == 0.0

    def test_pickup_success_rate_partial(self):
        task = _make_task()
        obs = Observation(rgb_path="/tmp/fake.png")
        trajectory = [
            StepResult(observation=obs, info={
                "action_name": "pickup_bowl", "action_success": False,
            }),
            StepResult(observation=obs, info={
                "action_name": "pickup_bowl", "action_success": True,
            }),
            StepResult(observation=obs, done=True, info={
                "success": 0.0, "prop_fixed_strict": 0.0,
                "action_name": "done", "action_success": True,
            }),
        ]
        metrics = task.evaluate_episode({"id": "test"}, trajectory)
        # PuSR: 1 success / 2 total = 0.5
        assert metrics["pickup_success_rate"] == 0.5
        assert metrics["num_pickup_actions"] == 2.0


class TestGetInstruction:
    def test_uses_episode_instruction(self):
        task = _make_task()
        episode = {"instruction": "Move the Bowl back to the counter."}
        assert task.get_instruction(episode) == "Move the Bowl back to the counter."


class TestBuiltinEpisodes:
    def test_has_builtin(self):
        task = _make_task()
        episodes = task._get_builtin_episodes()
        assert len(episodes) >= 1
        assert "scene" in episodes[0]
        assert "instruction" in episodes[0]


class TestAggregateResults:
    def test_aggregate_filters_nan_success_length(self):
        """success_length NaN values should not affect the average."""
        task = _make_task()

        records = []
        # Successful episode (3 steps)
        records.append(EpisodeRecord(
            episode={"id": "ep1"}, trajectory=[],
            episode_results={
                "success": 1.0, "prop_fixed_strict": 1.0,
                "num_steps": 3.0, "success_length": 3.0,
                "pickup_success_rate": 1.0, "num_pickup_actions": 1.0,
            },
        ))
        # Failed episode (5 steps, NaN success_length)
        records.append(EpisodeRecord(
            episode={"id": "ep2"}, trajectory=[],
            episode_results={
                "success": 0.0, "prop_fixed_strict": 0.0,
                "num_steps": 5.0, "success_length": float("nan"),
                "pickup_success_rate": 0.0, "num_pickup_actions": 2.0,
            },
        ))

        agg = task.aggregate_results(records)
        assert agg["success"] == 0.5  # 1/2
        assert agg["num_steps"] == 4.0  # (3+5)/2
        # success_length should average only non-NaN: 3.0
        assert agg["success_length"] == 3.0

    def test_aggregate_only_official_metrics_and_runtime_stats(self):
        """Aggregation outputs 6 paper metrics + runtime stats, nothing else."""
        task = _make_task()

        records = [
            EpisodeRecord(
                episode={"id": "ep1"}, trajectory=[],
                episode_results={
                    "success": 1.0, "prop_fixed_strict": 1.0,
                    "num_steps": 3.0, "success_length": 3.0,
                    "pickup_success_rate": 1.0, "num_pickup_actions": 1.0,
                    # Runner-injected fields
                    "episode_id": "FloorPlan21__0",
                    "instruction": "Move the bowl.",
                    "elapsed_seconds": 5.0,
                    "llm_usage": {"total_calls": 2, "total_tokens": 100},
                },
            ),
            EpisodeRecord(
                episode={"id": "ep2"}, trajectory=[],
                episode_results={
                    "success": 0.0, "prop_fixed_strict": 0.0,
                    "num_steps": 7.0, "success_length": float("nan"),
                    "pickup_success_rate": 0.5, "num_pickup_actions": 2.0,
                    "episode_id": "FloorPlan22__0",
                    "instruction": "Move the cup.",
                    "elapsed_seconds": 9.0,
                    "llm_usage": {"total_calls": 4, "total_tokens": 200},
                },
            ),
        ]
        agg = task.aggregate_results(records)

        # Official paper metrics
        assert agg["success"] == 0.5
        assert agg["num_steps"] == 5.0
        assert agg["success_length"] == 3.0  # only successful ep

        # Runtime stats
        assert agg["avg_elapsed_seconds"] == 7.0
        assert agg["total_elapsed_seconds"] == 14.0
        assert agg["avg_llm_calls"] == 3.0

        # Junk keys excluded
        assert "episode_id" not in agg
        assert "instruction" not in agg
        assert "elapsed_seconds" not in agg
        assert "llm_usage" not in agg


class TestPromptBuilder:
    def _make_builder(self):
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        builder = AI2THORRearrangement2023PromptBuilder()
        builder.set_action_space(["done", "move_ahead", "pickup_bowl"])
        return builder

    def test_set_action_space(self):
        builder = self._make_builder()
        assert "done" in builder._action_name_set
        assert "move_ahead" in builder._action_name_set
        assert "pickup_bowl" in builder._action_name_set

    def test_lazy_init_action_space(self):
        """build_messages should lazy-init action space from memory.action_space."""
        from easi.core.memory import AgentMemory
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder

        builder = AI2THORRearrangement2023PromptBuilder()
        # Action space NOT set via set_action_space — should be empty
        assert builder._action_list_str == ""

        memory = AgentMemory(action_space=["done", "move_ahead", "pickup_bowl"])
        memory.current_observation = Observation(rgb_path="/tmp/fake.png")
        memory.task_description = "Rearrange objects."

        messages = builder.build_messages(memory)
        # After build_messages, action list should be populated from memory
        assert "done" in builder._action_list_str
        assert "move_ahead" in builder._action_list_str
        # System message should contain action list
        system_content = messages[0]["content"]
        assert "move_ahead" in system_content

    def test_gps_from_observation_metadata(self):
        """GPS data should come from step.observation.metadata (not step.info)."""
        from easi.core.memory import AgentMemory
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder

        builder = AI2THORRearrangement2023PromptBuilder()
        builder.set_action_space(["done", "move_ahead"])

        memory = AgentMemory(action_space=["done", "move_ahead"])
        # Record a step with GPS in observation metadata
        obs_with_gps = Observation(
            rgb_path="/tmp/fake.png",
            metadata={
                "agent_x": 1.5, "agent_y": 0.87, "agent_z": -2.0,
                "agent_rotation": 90.0, "agent_horizon": 0.0,
                "held_object": "Bowl",
            },
        )
        from easi.core.episode import Action
        memory.record_step(obs_with_gps, Action(action_name="move_ahead"), llm_response="test")
        memory.record_feedback("success")

        memory.current_observation = Observation(rgb_path="/tmp/fake2.png")
        memory.task_description = "Rearrange objects."

        messages = builder.build_messages(memory)
        # The current turn message should include GPS and held object
        last_user = messages[-1]
        text_parts = [p["text"] for p in last_user["content"] if p.get("type") == "text"]
        text = " ".join(text_parts)
        assert "1.50" in text  # agent_x
        assert "Holding: Bowl" in text

    def test_goal_image_in_content(self):
        """When goal_rgb_path is in metadata, build_messages should include it."""
        from easi.core.memory import AgentMemory
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        import tempfile, os
        from PIL import Image
        import numpy as np

        builder = AI2THORRearrangement2023PromptBuilder()
        builder.set_action_space(["done", "move_ahead"])

        # Create real temp images so _encode_image_base64 works
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
            obs_path = os.path.join(tmpdir, "obs.png")
            goal_path = os.path.join(tmpdir, "goal.png")
            img.save(obs_path)
            img.save(goal_path)

            memory = AgentMemory(action_space=["done", "move_ahead"])
            memory.current_observation = Observation(
                rgb_path=obs_path,
                metadata={"goal_rgb_path": goal_path},
            )
            memory.task_description = "Rearrange objects."

            messages = builder.build_messages(memory)
            last_user = messages[-1]
            # Should have 2 images (current + goal) + 1 text
            image_parts = [p for p in last_user["content"] if p.get("type") == "image_url"]
            assert len(image_parts) == 2
            # Text should label images explicitly
            text_parts = [p["text"] for p in last_user["content"] if p.get("type") == "text"]
            text = " ".join(text_parts)
            assert "Image 1: RGB current" in text
            assert "Image 2: RGB goal" in text

    def test_depth_image_in_content(self):
        """When use_depth=True and depth_path in metadata, include depth image."""
        from easi.core.memory import AgentMemory
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        import tempfile, os
        from PIL import Image
        import numpy as np

        builder = AI2THORRearrangement2023PromptBuilder(use_depth=True)
        builder.set_action_space(["done", "move_ahead"])

        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
            depth_img = Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
            obs_path = os.path.join(tmpdir, "obs.png")
            depth_path = os.path.join(tmpdir, "depth.png")
            goal_path = os.path.join(tmpdir, "goal.png")
            img.save(obs_path)
            depth_img.save(depth_path)
            img.save(goal_path)

            memory = AgentMemory(action_space=["done", "move_ahead"])
            memory.current_observation = Observation(
                rgb_path=obs_path,
                metadata={
                    "depth_path": depth_path,
                    "goal_rgb_path": goal_path,
                },
            )
            memory.task_description = "Rearrange objects."

            messages = builder.build_messages(memory)
            last_user = messages[-1]
            # Should have 3 images (RGB + Depth + Goal)
            image_parts = [p for p in last_user["content"] if p.get("type") == "image_url"]
            assert len(image_parts) == 3
            # Text labels should reflect all three
            text_parts = [p["text"] for p in last_user["content"] if p.get("type") == "text"]
            text = " ".join(text_parts)
            assert "Image 1: RGB current" in text
            assert "Image 2: Depth" in text
            assert "Image 3: RGB goal" in text

    def test_depth_excluded_when_toggled_off(self):
        """When use_depth=False (default), depth_path in metadata is ignored."""
        from easi.core.memory import AgentMemory
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        import tempfile, os
        from PIL import Image
        import numpy as np

        builder = AI2THORRearrangement2023PromptBuilder(use_depth=False)
        builder.set_action_space(["done", "move_ahead"])

        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
            obs_path = os.path.join(tmpdir, "obs.png")
            depth_path = os.path.join(tmpdir, "depth.png")
            img.save(obs_path)
            img.save(depth_path)

            memory = AgentMemory(action_space=["done", "move_ahead"])
            memory.current_observation = Observation(
                rgb_path=obs_path,
                metadata={"depth_path": depth_path},
            )
            memory.task_description = "Rearrange objects."

            messages = builder.build_messages(memory)
            last_user = messages[-1]
            # Should have only 1 image (RGB current), depth excluded
            image_parts = [p for p in last_user["content"] if p.get("type") == "image_url"]
            assert len(image_parts) == 1

    def test_dynamic_system_prompt_all_sensors(self):
        """System prompt should describe all active sensors."""
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        from easi.core.memory import AgentMemory

        builder = AI2THORRearrangement2023PromptBuilder(
            use_rgb=True, use_depth=True, use_gps=True, use_goal_image=True,
        )
        builder.set_action_space(["done"])
        memory = AgentMemory(action_space=["done"])
        memory.current_observation = Observation(rgb_path="/tmp/fake.png")
        memory.task_description = "Test."

        messages = builder.build_messages(memory)
        system = messages[0]["content"]
        assert "RGB Image — Current Scene" in system
        assert "Depth Image" in system
        assert "RGB Image — Goal Scene" in system
        assert "GPS Data" in system

    def test_dynamic_system_prompt_rgb_only(self):
        """System prompt should only mention RGB when other sensors are off."""
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        from easi.core.memory import AgentMemory

        builder = AI2THORRearrangement2023PromptBuilder(
            use_rgb=True, use_depth=False, use_gps=False, use_goal_image=False,
        )
        builder.set_action_space(["done"])
        memory = AgentMemory(action_space=["done"])
        memory.current_observation = Observation(rgb_path="/tmp/fake.png")
        memory.task_description = "Test."

        messages = builder.build_messages(memory)
        system = messages[0]["content"]
        assert "RGB Image — Current Scene" in system
        assert "Depth Image" not in system
        assert "RGB Image — Goal Scene" not in system
        assert "GPS Data" not in system

    def test_strategy_adapts_to_goal_image_toggle(self):
        """Strategy section changes based on whether goal images are available."""
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder
        from easi.core.memory import AgentMemory

        with_goal = AI2THORRearrangement2023PromptBuilder(use_goal_image=True)
        with_goal.set_action_space(["done"])
        mem = AgentMemory(action_space=["done"])
        mem.current_observation = Observation(rgb_path="/tmp/fake.png")
        mem.task_description = "Test."
        system_with = with_goal.build_messages(mem)[0]["content"]

        without_goal = AI2THORRearrangement2023PromptBuilder(use_goal_image=False)
        without_goal.set_action_space(["done"])
        system_without = without_goal.build_messages(mem)[0]["content"]

        assert "Compare the current and goal images" in system_with
        assert "Compare the current and goal images" not in system_without
        assert "Explore the environment" in system_without

    def test_gps_excluded_when_toggled_off(self):
        """GPS text should not appear when use_gps=False."""
        from easi.core.memory import AgentMemory
        from easi.core.episode import Action
        from easi.tasks.ai2thor_rearrangement_2023.prompts import AI2THORRearrangement2023PromptBuilder

        builder = AI2THORRearrangement2023PromptBuilder(use_gps=False)
        builder.set_action_space(["done", "move_ahead"])

        memory = AgentMemory(action_space=["done", "move_ahead"])
        obs_with_gps = Observation(
            rgb_path="/tmp/fake.png",
            metadata={
                "agent_x": 1.5, "agent_y": 0.87, "agent_z": -2.0,
                "agent_rotation": 90.0, "agent_horizon": 0.0,
                "held_object": "Bowl",
            },
        )
        memory.record_step(obs_with_gps, Action(action_name="move_ahead"), llm_response="test")
        memory.record_feedback("success")
        memory.current_observation = Observation(rgb_path="/tmp/fake2.png")
        memory.task_description = "Rearrange objects."

        messages = builder.build_messages(memory)
        last_user = messages[-1]
        text_parts = [p["text"] for p in last_user["content"] if p.get("type") == "text"]
        text = " ".join(text_parts)
        assert "GPS:" not in text
        assert "Position:" not in text

    def test_parse_valid_response(self):
        from easi.core.memory import AgentMemory
        builder = self._make_builder()
        memory = AgentMemory()
        response = json.dumps({
            "observation": "I see a kitchen",
            "reasoning": "Need to find bowl",
            "plan": [
                {"action_name": "move_ahead"},
                {"action_name": "pickup_bowl"},
            ]
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "move_ahead"
        assert actions[1].action_name == "pickup_bowl"

    def test_parse_invalid_response_returns_done(self):
        from easi.core.memory import AgentMemory
        builder = self._make_builder()
        memory = AgentMemory()
        actions = builder.parse_response("not json", memory)
        assert len(actions) == 1
        assert actions[0].action_name == "done"

    def test_parse_unknown_action_stops_plan(self):
        from easi.core.memory import AgentMemory
        builder = self._make_builder()
        memory = AgentMemory()
        response = json.dumps({
            "plan": [
                {"action_name": "move_ahead"},
                {"action_name": "fly_away"},
            ]
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1
        assert actions[0].action_name == "move_ahead"


class TestBridgeScript:
    def test_bridge_path_exists(self):
        task = _make_task()
        bridge_path = task.get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"
