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
