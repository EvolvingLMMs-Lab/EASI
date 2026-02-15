"""Tests for the EB-Navigation task (offline, no simulator needed)."""
import json

import pytest

from easi.core.episode import Action, Observation, StepResult
from easi.tasks.ebnavigation.actions import (
    ACTION_NAME_TO_ID,
    DISCRETE_ACTIONS,
    get_action_space,
)
from easi.tasks.ebnavigation.task import EBNavigationTask


class TestEBNavigationActionSpace:
    def test_action_space_has_8_actions(self):
        actions = get_action_space()
        assert len(actions) == 8

    def test_action_space_contents(self):
        actions = get_action_space()
        assert "Move forward by 0.25" in actions
        assert "Move backward by 0.25" in actions
        assert "Move rightward by 0.25" in actions
        assert "Move leftward by 0.25" in actions
        assert "Rotate to the right by 90 degrees." in actions
        assert "Rotate to the left by 90 degrees." in actions
        assert "Tilt the camera upward by 30 degrees." in actions
        assert "Tilt the camera downward by 30 degrees." in actions

    def test_action_name_to_id_mapping(self):
        assert ACTION_NAME_TO_ID["Move forward by 0.25"] == 0
        assert ACTION_NAME_TO_ID["Move backward by 0.25"] == 1
        assert ACTION_NAME_TO_ID["Move rightward by 0.25"] == 2
        assert ACTION_NAME_TO_ID["Move leftward by 0.25"] == 3
        assert ACTION_NAME_TO_ID["Rotate to the right by 90 degrees."] == 4
        assert ACTION_NAME_TO_ID["Rotate to the left by 90 degrees."] == 5
        assert ACTION_NAME_TO_ID["Tilt the camera upward by 30 degrees."] == 6
        assert ACTION_NAME_TO_ID["Tilt the camera downward by 30 degrees."] == 7
        assert len(ACTION_NAME_TO_ID) == 8

    def test_action_space_returns_copy(self):
        a1 = get_action_space()
        a2 = get_action_space()
        assert a1 == a2
        assert a1 is not a2

    def test_discrete_actions_matches_source(self):
        """Verify action text matches EBNavEnv.DISCRETE_SKILLSET exactly."""
        assert DISCRETE_ACTIONS[0] == "Move forward by 0.25"
        assert DISCRETE_ACTIONS[7] == "Tilt the camera downward by 30 degrees."


class TestEBNavigationTask:
    @pytest.fixture
    def task(self):
        return EBNavigationTask()

    def test_name(self, task):
        assert task.name == "ebnavigation_base"

    def test_simulator_key(self, task):
        assert task.simulator_key == "ai2thor:v5_0_0"

    def test_action_space_loaded(self, task):
        assert len(task.action_space) == 8

    def test_max_steps(self, task):
        assert task.max_steps == 20

    def test_format_reset_config(self, task):
        """Test the adapter from EB-Navigation HF row to bridge reset config."""
        episode = {
            "id": 0,
            "scene": "FloorPlan11",
            "instruction": "navigate to the Bread",
            "target_object_type": "Bread",
            "target_object_id": "Bread|+01.30|+00.98|-01.53",
            "target_position": {"x": 1.3, "y": 0.98, "z": -1.53},
            "agent_pose": {
                "position": {"x": -0.75, "y": 0.9009992, "z": -1.75},
                "rotation": 90.0,
                "horizon": 0.0,
            },
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == 0
        assert config["scene"] == "FloorPlan11"
        assert config["instruction"] == "navigate to the Bread"
        assert config["target_object_type"] == "Bread"
        assert config["target_object_id"] == "Bread|+01.30|+00.98|-01.53"
        assert config["target_position"]["x"] == 1.3
        assert config["agent_pose"]["rotation"] == 90.0

    def test_evaluate_episode(self, task):
        """Test metric computation from trajectory."""
        episode = {
            "id": 0, "scene": "FloorPlan11",
            "instruction": "test",
            "target_position": {"x": 0, "y": 0, "z": 0},
            "agent_pose": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": 0, "horizon": 0},
        }
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False,
                       info={"task_success": 0.0, "distance": 2.5}),
            StepResult(observation=obs, reward=1.0, done=True,
                       info={"task_success": 1.0, "distance": 0.8}),
        ]
        metrics = task.evaluate_episode(episode, trajectory)
        assert metrics["task_success"] == 1.0
        assert metrics["distance_to_target"] == 0.8
        assert metrics["num_steps"] == 2

    def test_evaluate_empty_trajectory(self, task):
        episode = {
            "id": 0, "scene": "FloorPlan11",
            "instruction": "test",
            "target_position": {"x": 0, "y": 0, "z": 0},
            "agent_pose": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": 0, "horizon": 0},
        }
        metrics = task.evaluate_episode(episode, [])
        assert metrics["task_success"] == 0.0
        assert metrics["distance_to_target"] == -1.0
        assert metrics["num_steps"] == 0.0

    def test_get_instruction(self, task):
        episode = {"instruction": "navigate to the Bread"}
        assert task.get_instruction(episode) == "navigate to the Bread"

    def test_builtin_episodes(self, task):
        episodes = task._get_builtin_episodes()
        assert len(episodes) == 1
        assert episodes[0]["scene"] == "FloorPlan11"
        assert episodes[0]["target_object_type"] == "Bread"
        assert "agent_pose" in episodes[0]
        assert "target_position" in episodes[0]

    def test_task_registry_discovers_all_splits(self):
        """Registry should discover all EB-Navigation split yamls."""
        from easi.tasks.registry import list_tasks, refresh
        refresh()
        tasks = list_tasks()
        assert "ebnavigation_base" in tasks
        assert "ebnavigation_common_sense" in tasks
        assert "ebnavigation_complex_instruction" in tasks
        assert "ebnavigation_long_horizon" in tasks
        assert "ebnavigation_visual_appearance" in tasks

    def test_split_specific_task_loading(self):
        """Loading ebnavigation_base should give a task with correct config."""
        from easi.tasks.registry import get_task_entry, refresh
        refresh()
        entry = get_task_entry("ebnavigation_base")
        assert entry.simulator_key == "ai2thor:v5_0_0"
        assert entry.config_path.name == "ebnavigation_base.yaml"

    def test_bridge_script_path(self, task):
        bridge_path = task.get_bridge_script_path()
        assert bridge_path is not None
        assert bridge_path.name == "bridge.py"
        assert bridge_path.exists()


class TestEBNavigationPromptBuilder:
    def test_build_messages_first_turn(self):
        """First turn: full system prompt in user message."""
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        actions = get_action_space()
        memory = AgentMemory(
            task_description="Navigate to the Bread.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        text_content = messages[0]["content"][-1]["text"]
        assert "robot operating in a home" in text_content
        assert "Move forward by 0.25" in text_content
        assert "Strategy" in text_content
        assert "1-2 actions" in text_content
        assert builder.action_name_to_id("Move forward by 0.25") == 0

    def test_build_messages_with_history(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        actions = get_action_space()
        memory = AgentMemory(
            task_description="Navigate to the Bread.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="Move forward by 0.25"), llm_response="r1")
        memory.record_feedback("success")
        memory.record_step(obs, Action(action_name="Move leftward by 0.25"), llm_response="r2")
        memory.record_feedback("failed: blocked")
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        # With chat_history=True, we get accumulated messages
        # Last message should be user
        assert messages[-1]["role"] == "user"
        text_content = ""
        for part in messages[-1]["content"]:
            if part["type"] == "text":
                text_content += part["text"]
        assert "Move forward by 0.25" in text_content
        assert "5-6 actions" in text_content

    def test_build_messages_stateless(self):
        """Stateless mode (chat_history=False) produces single user message each turn."""
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder(chat_history=False)
        actions = get_action_space()
        memory = AgentMemory(
            task_description="Navigate to the Bread.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="Move forward by 0.25"), llm_response="r1")
        memory.record_feedback("success")
        messages = builder.build_messages(memory)
        # Stateless: single user message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_parse_response_valid(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        actions = get_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        builder.set_action_space(actions)
        response = json.dumps({
            "visual_state_description": "I see a room.",
            "reasoning_and_reflection": "I should move forward.",
            "language_plan": "Step 1: Move forward",
            "executable_plan": [
                {"action_id": 0, "action_name": "Move forward by 0.25"},
                {"action_id": 0, "action_name": "Move forward by 0.25"},
            ],
        })
        parsed = builder.parse_response(response, memory)
        assert len(parsed) == 2
        assert parsed[0].action_name == "Move forward by 0.25"
        assert parsed[1].action_name == "Move forward by 0.25"

    def test_parse_response_invalid_json(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        actions = get_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        builder.set_action_space(actions)
        parsed = builder.parse_response("not json at all {{}", memory)
        assert parsed == []

    def test_parse_response_invalid_action_id(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        actions = get_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        builder.set_action_space(actions)
        response = json.dumps({
            "visual_state_description": "test",
            "reasoning_and_reflection": "test",
            "language_plan": "test",
            "executable_plan": [
                {"action_id": 0, "action_name": "Move forward by 0.25"},
                {"action_id": 99, "action_name": "invalid action"},
            ],
        })
        parsed = builder.parse_response(response, memory)
        # Should get first valid action, then break on invalid
        assert len(parsed) == 1
        assert parsed[0].action_name == "Move forward by 0.25"

    def test_conforms_to_protocol(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.agents.prompt_builder import PromptBuilderProtocol
        builder = EBNavigationPromptBuilder()
        assert isinstance(builder, PromptBuilderProtocol)

    def test_response_format(self):
        from easi.tasks.ebnavigation.prompts import EBNavigationPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBNavigationPromptBuilder()
        memory = AgentMemory(
            task_description="test",
            action_space=get_action_space(),
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        schema = builder.get_response_format(memory)
        assert schema["type"] == "json_schema"
        assert "executable_plan" in schema["json_schema"]["schema"]["properties"]
