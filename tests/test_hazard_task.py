"""Tests for the HAZARD task (offline, no simulator needed)."""
import json
import pytest
from pathlib import Path

from easi.tasks.hazard.actions import (
    get_action_space, ACTION_TYPES, SCENARIO_MAX_STEPS,
    HIGH_VALUE, LOW_VALUE,
)


class TestHAZARDActions:
    def test_action_types(self):
        assert len(ACTION_TYPES) == 5
        assert "walk_to" in ACTION_TYPES
        assert "explore" in ACTION_TYPES
        assert "stop" in ACTION_TYPES

    def test_get_action_space(self):
        space = get_action_space()
        assert isinstance(space, list)
        assert len(space) == 5

    def test_scenario_max_steps(self):
        assert SCENARIO_MAX_STEPS["fire"] == 1500
        assert SCENARIO_MAX_STEPS["flood"] == 1500
        assert SCENARIO_MAX_STEPS["wind"] == 3000

    def test_values(self):
        assert HIGH_VALUE == 5
        assert LOW_VALUE == 1


class TestHAZARDConfig:
    def test_value_json_exists(self):
        path = Path(__file__).parent.parent / "easi/tasks/hazard/config/value.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_fire_json_exists(self):
        path = Path(__file__).parent.parent / "easi/tasks/hazard/config/fire.json"
        assert path.exists()

    def test_fluid_json_exists(self):
        path = Path(__file__).parent.parent / "easi/tasks/hazard/config/fluid.json"
        assert path.exists()

    def test_prompts_csv_exists(self):
        path = Path(__file__).parent.parent / "easi/tasks/hazard/config/prompts.csv"
        assert path.exists()


class TestHAZARDTask:
    @pytest.fixture
    def fire_task(self):
        from easi.tasks.hazard.task import HAZARDTask
        return HAZARDTask()

    def test_name(self, fire_task):
        assert "hazard" in fire_task.name.lower()

    def test_simulator_key(self, fire_task):
        assert fire_task.simulator_key == "tdw:v1_11_23"

    def test_max_steps(self, fire_task):
        assert fire_task.max_steps >= 1500

    def test_bridge_script_path(self, fire_task):
        path = fire_task.get_bridge_script_path()
        assert path.exists()
        assert path.name == "bridge.py"

    def test_get_instruction(self, fire_task):
        episode = {"instruction": "Rescue objects from fire"}
        assert fire_task.get_instruction(episode) == "Rescue objects from fire"

    def test_format_reset_config(self, fire_task):
        episode = {
            "id": 0,
            "episode_id": "test-1",
            "task": "fire",
            "instruction": "Rescue objects",
            "scene_name": "mm_kitchen_3a",
            "agent_position": [0.0, 0.0, 0.0],
            "target_categories": ["hairbrush"],
            "target_object_ids": [123],
            "hazard_params": '{"fire_positions": [[0, 0, 0]]}',
            "containers": [],
            "objects": "[]",
            "max_steps": 1500,
            "source_dir": "room_setup_fire/test_set/test-1",
            "_data_dir": "/tmp/test",
        }
        config = fire_task.format_reset_config(episode)
        assert "source_dir" in config
        assert "simulator_data" in config["source_dir"]
        assert config["target_categories"] == ["hairbrush"]
        assert config["target_object_ids"] == [123]

    def test_evaluate_empty_trajectory(self, fire_task):
        episode = {"target_categories": ["hairbrush"], "target_object_ids": [123]}
        result = fire_task.evaluate_episode(episode, [])
        assert result["value_score"] == 0.0
        assert result["value_rate"] == 0.0
        assert result["damage_rate"] == 0.0
        assert result["num_steps"] == 0.0

    def test_evaluate_episode_value_and_damage(self, fire_task):
        """Test both Value and Damage metrics from the HAZARD paper."""
        from easi.core.episode import StepResult, Observation
        episode = {"target_categories": ["hairbrush"], "target_object_ids": [123, 456]}
        trajectory = [
            StepResult(
                observation=Observation(rgb_path="/dev/null"),
                reward=0.0,
                done=True,
                info={
                    "value_score": 1.5,       # e.g., one at full value (1.0), one damaged (0.5)
                    "max_value": 2.0,
                    "rescued_count": 2.0,
                    "damaged_count": 1.0,      # one object was damaged when rescued
                    "targets_rescued": 2.0,
                    "targets_total": 2.0,
                },
            )
        ]
        result = fire_task.evaluate_episode(episode, trajectory)
        assert result["value_score"] == 1.5
        assert result["value_rate"] == 0.75    # 1.5 / 2.0 = 0.75 (paper "Value")
        assert result["damage_rate"] == 0.5    # 1 damaged / 2 rescued = 0.5 (paper "Damage")
        assert result["rescued_count"] == 2.0
        assert result["damaged_count"] == 1.0

    def test_evaluate_episode_no_damage(self, fire_task):
        """All objects rescued undamaged -> value_rate=1.0, damage_rate=0.0."""
        from easi.core.episode import StepResult, Observation
        episode = {"target_categories": ["hairbrush"], "target_object_ids": [123]}
        trajectory = [
            StepResult(
                observation=Observation(rgb_path="/dev/null"),
                reward=0.0,
                done=True,
                info={
                    "value_score": 1.0,
                    "max_value": 1.0,
                    "rescued_count": 1.0,
                    "damaged_count": 0.0,
                    "targets_rescued": 1.0,
                    "targets_total": 1.0,
                },
            )
        ]
        result = fire_task.evaluate_episode(episode, trajectory)
        assert result["value_rate"] == 1.0
        assert result["damage_rate"] == 0.0

    def test_aggregate_results_value_and_damage(self, fire_task):
        """Test that aggregate_results produces paper-matching metrics."""
        from easi.core.episode import EpisodeRecord
        records = [
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "value_rate": 0.8, "damage_rate": 0.0,
                    "rescued_count": 3.0, "damaged_count": 0.0,
                    "value_score": 8.0, "max_value": 10.0,
                    "num_steps": 50.0, "targets_rescued": 3.0, "targets_total": 4.0,
                    "max_rescue_frame": 900.0,
                },
            ),
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "value_rate": 0.5, "damage_rate": 0.0,
                    "rescued_count": 2.0, "damaged_count": 1.0,
                    "value_score": 4.0, "max_value": 8.0,
                    "num_steps": 100.0, "targets_rescued": 2.0, "targets_total": 3.0,
                    "max_rescue_frame": 600.0,
                },
            ),
        ]
        agg = fire_task.aggregate_results(records)
        # Value: average of per-episode value_rate = (0.8 + 0.5) / 2 = 0.65
        assert abs(agg["value_rate"] - 0.65) < 1e-6
        # Damage: cumulative = 1 damaged / 5 rescued = 0.2
        assert abs(agg["damage_rate"] - 0.2) < 1e-6
        # Step (paper): sum(max_rescue_frame) / sum(rescued) = (900+600) / 5 = 300
        assert abs(agg["avg_step"] - 300.0) < 1e-6
        # LLM steps: average = (50 + 100) / 2 = 75
        assert agg["avg_llm_steps"] == 75.0

    def test_builtin_episodes(self, fire_task):
        episodes = fire_task._get_builtin_episodes()
        assert len(episodes) >= 1
        assert "instruction" in episodes[0]
        assert "source_dir" in episodes[0]


class TestHAZARDTaskRegistry:
    def test_registry_discovers_hazard_tasks(self):
        from easi.tasks.registry import list_tasks, refresh
        refresh()
        tasks = list_tasks()
        hazard_tasks = [t for t in tasks if t.startswith("hazard")]
        assert len(hazard_tasks) == 3, f"Expected 3 HAZARD tasks, got {hazard_tasks}"


class TestHAZARDPromptBuilder:
    @pytest.fixture
    def builder(self):
        from easi.tasks.hazard.prompts import HAZARDPromptBuilder
        return HAZARDPromptBuilder(scenario="fire")

    def test_build_messages_returns_list(self, builder):
        from easi.core.episode import Observation
        from easi.core.memory import AgentMemory
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["look around"]',
                "holding_objects": "[]",
                "target_categories": '["hairbrush"]',
                "object_list": '[{"name": "hairbrush_1", "category": "hairbrush", "id": "123"}]',
                "current_seen_objects_id": '["123"]',
                "object_distances": '{"123": 3.45}',
                "env_change_record": "{}",
                "feedback": "",
                "frame_count": "0",
                "targets_rescued": "0",
                "targets_total": "1",
            },
        )
        memory = AgentMemory(
            task_description="Rescue hairbrush",
            action_space=["walk_to", "explore"],
            current_observation=obs,
        )
        msgs = builder.build_messages(memory)
        assert isinstance(msgs, list)
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "user"
        # Verify prompt content matches original format
        text = msgs[0]["content"][-1]["text"]
        assert "fire" in text.lower()  # scenario preamble
        # Target uses raw numeric value (not "high"/"low")
        assert "value: 1" in text
        assert "attribute: None" in text  # fire uses "None", not "fireproof"
        # State section has per-object details
        assert "Target objects currently seen:" in text
        assert "distance: 3.45 m" in text

    def test_parse_response_option_letter(self, builder):
        from easi.core.episode import Observation
        from easi.core.memory import AgentMemory
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["go pick up object <hairbrush> (123)", "look around"]',
                "holding_objects": "[]",
            },
        )
        memory = AgentMemory(
            task_description="Rescue",
            action_space=["walk_to", "explore"],
            current_observation=obs,
        )
        actions = builder.parse_response("A", memory)
        assert len(actions) == 1
        assert actions[0].action_name == "go pick up object <hairbrush> (123)"

    def test_parse_response_full_text(self, builder):
        from easi.core.episode import Observation
        from easi.core.memory import AgentMemory
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["go pick up object <hairbrush> (123)", "look around"]',
                "holding_objects": "[]",
            },
        )
        memory = AgentMemory(
            task_description="Rescue",
            action_space=["walk_to", "explore"],
            current_observation=obs,
        )
        actions = builder.parse_response("look around", memory)
        assert len(actions) == 1
        assert actions[0].action_name == "look around"

    def test_parse_response_fallback(self, builder):
        from easi.core.episode import Observation
        from easi.core.memory import AgentMemory
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["go pick up object <hairbrush> (123)"]',
                "holding_objects": "[]",
            },
        )
        memory = AgentMemory(
            task_description="Rescue",
            action_space=["walk_to", "explore"],
            current_observation=obs,
        )
        actions = builder.parse_response("gibberish nonsense", memory)
        assert len(actions) == 1
        assert actions[0].action_name == "look around"  # fallback

    def test_conforms_to_protocol(self, builder):
        from easi.agents.prompt_builder import PromptBuilderProtocol
        assert isinstance(builder, PromptBuilderProtocol)

    def test_set_action_space_noop(self, builder):
        # Should not raise
        builder.set_action_space(["walk_to", "explore"])

    def test_flood_builder_attributes(self):
        from easi.tasks.hazard.prompts import HAZARDPromptBuilder
        builder = HAZARDPromptBuilder(scenario="flood")
        from easi.core.episode import Observation
        from easi.core.memory import AgentMemory
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["look around"]',
                "holding_objects": "[]",
                "target_categories": '["hairbrush"]',
                "object_list": '[{"name": "hairbrush_1", "category": "hairbrush", "id": "123"}]',
                "current_seen_objects_id": '["123"]',
                "object_distances": '{"123": 2.0}',
                "env_change_record": "{}",
            },
        )
        memory = AgentMemory(
            task_description="Rescue",
            action_space=[],
            current_observation=obs,
        )
        msgs = builder.build_messages(memory)
        text = msgs[0]["content"][-1]["text"]
        # Flood shows waterproof attribute
        assert "non-waterproof" in text or "waterproof" in text

    def test_action_history_format(self, builder):
        from easi.core.episode import Observation, Action
        from easi.core.memory import AgentMemory, StepRecord
        dummy_obs = Observation(rgb_path="/dev/null")
        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "available_plans": '["look around"]',
                "holding_objects": "[]",
                "target_categories": '["hairbrush"]',
                "object_list": "[]",
                "current_seen_objects_id": "[]",
                "object_distances": "{}",
                "env_change_record": "{}",
            },
        )
        memory = AgentMemory(
            task_description="Rescue",
            action_space=[],
            current_observation=obs,
            steps=[
                StepRecord(
                    observation=dummy_obs,
                    action=Action(action_name="look around"),
                    feedback="success",
                ),
                StepRecord(
                    observation=dummy_obs,
                    action=Action(action_name="go pick up hairbrush"),
                    feedback="max steps reached",
                ),
            ],
        )
        msgs = builder.build_messages(memory)
        text = msgs[0]["content"][-1]["text"]
        assert "look around (success)" in text
        assert "paused after taking 100 steps" in text
