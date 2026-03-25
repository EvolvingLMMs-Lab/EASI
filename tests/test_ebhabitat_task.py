"""Tests for the EB-Habitat task (offline, no simulator needed)."""
import json

import pytest
from pathlib import Path

from easi.core.episode import Action, Observation, StepResult
from easi.tasks.ebhabitat.task import EBHabitatTask
from easi.tasks.ebhabitat.actions import get_placeholder_action_space, PLACEHOLDER_ACTIONS


class TestEBHabitatActionSpace:
    def test_placeholder_action_space_not_empty(self):
        actions = get_placeholder_action_space()
        assert len(actions) > 0

    def test_placeholder_has_navigate(self):
        actions = get_placeholder_action_space()
        nav_actions = [a for a in actions if a.startswith("navigate")]
        assert len(nav_actions) >= 5

    def test_placeholder_has_pick(self):
        actions = get_placeholder_action_space()
        pick_actions = [a for a in actions if a.startswith("pick up")]
        assert len(pick_actions) >= 1

    def test_placeholder_has_place(self):
        actions = get_placeholder_action_space()
        place_actions = [a for a in actions if a.startswith("place at")]
        assert len(place_actions) >= 1

    def test_placeholder_has_open_close(self):
        actions = get_placeholder_action_space()
        assert any("open" in a for a in actions)
        assert any("close" in a for a in actions)

    def test_placeholder_returns_copy(self):
        a1 = get_placeholder_action_space()
        a2 = get_placeholder_action_space()
        assert a1 is not a2
        assert a1 == a2


class TestEBHabitatTask:
    @pytest.fixture
    def task(self):
        return EBHabitatTask()

    def test_name(self, task):
        assert task.name == "ebhabitat_base"

    def test_simulator_key(self, task):
        assert task.simulator_key == "habitat_sim:v0_3_0"

    def test_action_space_loaded(self, task):
        assert len(task.action_space) == len(PLACEHOLDER_ACTIONS)

    def test_max_steps(self, task):
        assert task.max_steps == 50

    def test_format_reset_config(self, task):
        """Test the adapter from EB-Habitat HF row to bridge reset config."""
        episode = {
            "id": 0,
            "episode_id": "140",
            "instruction": "Find a toy airplane and move it to the right counter.",
            "instruct_id": "f0917e29",
            "scene_id": "data/replica_cad/configs/scenes/v3_sc3_staging_02.scene_instance.json",
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == 0
        assert config["eval_set"] == "base"
        assert config["instruction"] == episode["instruction"]

    def test_evaluate_episode(self, task):
        """Test metric computation from trajectory."""
        episode = {"id": 0, "instruction": "test"}
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False,
                       info={"task_success": 0.0, "task_progress": 0.33, "subgoal_reward": 0.1}),
            StepResult(observation=obs, reward=0.5, done=True,
                       info={"task_success": 1.0, "task_progress": 1.0, "subgoal_reward": 0.8}),
        ]
        metrics = task.evaluate_episode(episode, trajectory)
        assert metrics["task_success"] == 1.0
        assert metrics["task_progress"] == 1.0
        assert metrics["subgoal_reward"] == 0.8
        assert metrics["num_steps"] == 2

    def test_evaluate_empty_trajectory(self, task):
        episode = {"id": 0, "instruction": "test"}
        metrics = task.evaluate_episode(episode, [])
        assert metrics["task_success"] == 0.0
        assert metrics["task_progress"] == 0.0
        assert metrics["subgoal_reward"] == 0.0
        assert metrics["num_steps"] == 0.0

    def test_get_instruction(self, task):
        episode = {"instruction": "Find a toy airplane and move it to the right counter."}
        assert task.get_instruction(episode) == "Find a toy airplane and move it to the right counter."

    def test_builtin_episodes(self, task):
        episodes = task._get_builtin_episodes()
        assert len(episodes) >= 1
        assert "instruction" in episodes[0]
        assert "episode_id" in episodes[0]

    def test_task_registry_discovers_all_splits(self):
        """Registry should discover all EB-Habitat split yamls."""
        from easi.tasks.registry import list_tasks, refresh
        refresh()
        tasks = list_tasks()
        expected = [
            "ebhabitat_base",
            "ebhabitat_common_sense",
            "ebhabitat_complex_instruction",
            "ebhabitat_spatial_relationship",
            "ebhabitat_visual_appearance",
            "ebhabitat_long_horizon",
        ]
        for name in expected:
            assert name in tasks, f"{name} not found in task registry"

    def test_split_specific_task_loading(self):
        """Loading ebhabitat_base should give a task with correct config."""
        from easi.tasks.registry import get_task_entry, refresh
        refresh()
        entry = get_task_entry("ebhabitat_base")
        assert entry.simulator_key == "habitat_sim:v0_3_0"
        assert entry.config_path.name == "ebhabitat_base.yaml"

    def test_bridge_script_path(self, task):
        bridge_path = task.get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"


class TestEBHabitatPromptBuilder:
    def test_build_messages_first_turn(self):
        """First turn: full system prompt in user message."""
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBHabitatPromptBuilder()
        actions = get_placeholder_action_space()
        memory = AgentMemory(
            task_description="Find a toy airplane and move it to the right counter.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        text_content = messages[0]["content"][-1]["text"]
        assert "You are a robot operating in a home" in text_content
        assert "Navigation:" in text_content
        assert "Pick:" in text_content
        assert "Place:" in text_content

    def test_build_messages_with_history(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBHabitatPromptBuilder()
        actions = get_placeholder_action_space()
        memory = AgentMemory(
            task_description="Find a toy airplane and move it to the right counter.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="navigate to the sofa"), llm_response="r1")
        memory.record_feedback("Last action executed successfully.")
        memory.record_step(obs, Action(action_name="pick up the toy airplane"), llm_response="r2")
        memory.record_feedback("Last action is invalid.")
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        text_content = ""
        for part in messages[0]["content"]:
            if part["type"] == "text":
                text_content += part["text"]
        assert "navigate to the sofa" in text_content
        assert "Last action is invalid." in text_content

    def test_parse_response_valid(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBHabitatPromptBuilder()
        actions = get_placeholder_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        builder.set_action_space(actions)
        response = json.dumps({
            "visual_state_description": "I see a room",
            "reasoning_and_reflection": "I should navigate",
            "language_plan": "1. Navigate to sofa",
            "executable_plan": [
                {"action_id": 6, "action_name": actions[6]},
                {"action_id": 0, "action_name": actions[0]},
            ],
        })
        parsed = builder.parse_response(response, memory)
        assert len(parsed) == 2
        assert parsed[0].action_name == actions[6]
        assert parsed[1].action_name == actions[0]

    def test_parse_response_invalid_json(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBHabitatPromptBuilder()
        actions = get_placeholder_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        parsed = builder.parse_response("not json at all", memory)
        assert parsed == []

    def test_parse_response_empty_plan(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBHabitatPromptBuilder()
        actions = get_placeholder_action_space()
        memory = AgentMemory(
            task_description="test",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        response = json.dumps({
            "visual_state_description": "test",
            "reasoning_and_reflection": "test",
            "language_plan": "test",
            "executable_plan": [],
        })
        parsed = builder.parse_response(response, memory)
        assert parsed == []

    def test_conforms_to_protocol(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        from easi.agents.prompt_builder import PromptBuilderProtocol
        builder = EBHabitatPromptBuilder()
        assert isinstance(builder, PromptBuilderProtocol)

    def test_system_prompt_matches_source(self):
        """Verify the system prompt text matches EmbodiedBench source."""
        from easi.tasks.ebhabitat.prompts import HABITAT_SYSTEM_PROMPT
        assert "Navigation:" in HABITAT_SYSTEM_PROMPT
        assert "Pick:" in HABITAT_SYSTEM_PROMPT
        assert "Place:" in HABITAT_SYSTEM_PROMPT
        assert "Open:" in HABITAT_SYSTEM_PROMPT
        assert "Close:" in HABITAT_SYSTEM_PROMPT
        # Should NOT contain Alfred-specific actions
        assert "Find:" not in HABITAT_SYSTEM_PROMPT
        assert "Slice:" not in HABITAT_SYSTEM_PROMPT
        assert "Turn on:" not in HABITAT_SYSTEM_PROMPT

    def test_examples_loaded(self):
        from easi.tasks.ebhabitat.prompts import EBHabitatPromptBuilder
        builder = EBHabitatPromptBuilder(n_shot=10)
        assert len(builder._examples) >= 10
