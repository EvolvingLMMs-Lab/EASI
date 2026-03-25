"""Tests for the EB-Alfred task (offline, no simulator needed)."""
import pytest
from pathlib import Path

from easi.core.episode import Observation, StepResult
from easi.tasks.ebalfred.task import EBAlfredTask
from easi.tasks.ebalfred.actions import get_global_action_space


class TestEBAlfredActionSpace:
    def test_action_space_not_empty(self):
        actions = get_global_action_space()
        assert len(actions) > 100  # EB-Alfred has ~133 actions

    def test_action_space_has_find(self):
        actions = get_global_action_space()
        find_actions = [a for a in actions if a.startswith("find")]
        assert len(find_actions) > 20

    def test_action_space_has_pick(self):
        actions = get_global_action_space()
        pick_actions = [a for a in actions if a.startswith("pick up")]
        assert len(pick_actions) > 20

    def test_action_space_has_put_and_drop(self):
        actions = get_global_action_space()
        assert "put down the object in hand" in actions
        assert "drop the object in hand" in actions

    def test_action_space_has_open_close(self):
        actions = get_global_action_space()
        assert "open the Fridge" in actions
        assert "close the Fridge" in actions

    def test_action_space_has_toggle(self):
        actions = get_global_action_space()
        assert "turn on the Faucet" in actions
        assert "turn off the Faucet" in actions

    def test_action_space_has_slice(self):
        actions = get_global_action_space()
        assert "slice the Potato" in actions


class TestEBAlfredTask:
    @pytest.fixture
    def task(self):
        return EBAlfredTask()

    def test_name(self, task):
        assert task.name == "ebalfred_base"

    def test_simulator_key(self, task):
        assert task.simulator_key == "ai2thor:v2_1_0"

    def test_action_space_loaded(self, task):
        assert len(task.action_space) > 100

    def test_format_reset_config(self, task):
        """Test the adapter from EB-Alfred HF row to THOR reset config."""
        episode = {
            "id": 0,
            "task": "pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001",
            "repeat_idx": 0,
            "instruction": "Put a mug on the shelf.",
            "task_type": "pick_and_place_simple",
            "trial_id": "trial_T20190001",
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == 0
        assert config["task"] == episode["task"]
        assert config["repeat_idx"] == 0
        assert config["instruction"] == "Put a mug on the shelf."

    def test_evaluate_episode(self, task):
        """Test metric computation from trajectory."""
        episode = {"id": 0, "task": "test", "repeat_idx": 0, "instruction": "test",
                   "task_type": "test", "trial_id": "trial_T00000000"}
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False,
                      info={"task_success": 0.0, "task_progress": 0.33}),
            StepResult(observation=obs, reward=0.5, done=True,
                      info={"task_success": 1.0, "task_progress": 1.0}),
        ]
        metrics = task.evaluate_episode(episode, trajectory)
        assert metrics["task_success"] == 1.0
        assert metrics["task_progress"] == 1.0
        assert metrics["num_steps"] == 2

    def test_evaluate_empty_trajectory(self, task):
        episode = {"id": 0, "task": "test", "repeat_idx": 0, "instruction": "test",
                   "task_type": "test", "trial_id": "trial_T00000000"}
        metrics = task.evaluate_episode(episode, [])
        assert metrics["task_success"] == 0.0
        assert metrics["num_steps"] == 0.0

    def test_get_instruction(self, task):
        episode = {"instruction": "Put a mug on the shelf."}
        assert task.get_instruction(episode) == "Put a mug on the shelf."

    def test_builtin_episodes(self, task):
        episodes = task._get_builtin_episodes()
        assert len(episodes) == 2
        assert episodes[0]["task_type"] == "pick_and_place_simple"

    def test_task_registry_discovers_all_splits(self):
        """Registry should discover all EB-Alfred split yamls."""
        from easi.tasks.registry import list_tasks, refresh
        refresh()
        tasks = list_tasks()
        assert "ebalfred_base" in tasks
        assert "ebalfred_long_horizon" in tasks
        assert "ebalfred_common_sense" in tasks
        assert "ebalfred_complex_instruction" in tasks
        assert "ebalfred_spatial" in tasks
        assert "ebalfred_visual_appearance" in tasks

    def test_split_specific_task_loading(self):
        """Loading ebalfred_base should give a task with correct config."""
        from easi.tasks.registry import get_task_entry, refresh
        refresh()
        entry = get_task_entry("ebalfred_base")
        assert entry.simulator_key == "ai2thor:v2_1_0"
        assert entry.config_path.name == "ebalfred_base.yaml"


class TestEBAlfredPromptBuilder:
    def test_build_messages_first_turn(self):
        """First turn: full system prompt in user message."""
        from easi.tasks.ebalfred.prompts import EBAlfredPromptBuilder
        from easi.core.memory import AgentMemory
        builder = EBAlfredPromptBuilder()
        actions = get_global_action_space()
        memory = AgentMemory(
            task_description="Put a mug on the shelf.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        text_content = messages[0]["content"][-1]["text"]
        assert "You are a robot operating in a home" in text_content
        # Action space should be initialized internally
        assert builder.action_name_to_id("find a Cart") == 0

    def test_build_messages_with_history(self):
        from easi.tasks.ebalfred.prompts import EBAlfredPromptBuilder
        from easi.core.memory import AgentMemory
        from easi.core.episode import Action
        builder = EBAlfredPromptBuilder()
        actions = get_global_action_space()
        memory = AgentMemory(
            task_description="Put a mug on the shelf.",
            action_space=actions,
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="find a Mug"), llm_response="r1")
        memory.record_feedback("success")
        memory.record_step(obs, Action(action_name="pick up the Mug"), llm_response="r2")
        memory.record_feedback("failed: not close enough")
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        text_content = ""
        for part in messages[0]["content"]:
            if part["type"] == "text":
                text_content += part["text"]
        assert "find a Mug" in text_content
        assert "failed: not close enough" in text_content

    def test_conforms_to_protocol(self):
        from easi.tasks.ebalfred.prompts import EBAlfredPromptBuilder
        from easi.agents.prompt_builder import PromptBuilderProtocol
        builder = EBAlfredPromptBuilder()
        assert isinstance(builder, PromptBuilderProtocol)
