"""Tests for REVERIE-CE task, action space, and prompt builder."""
import json

import pytest
from unittest.mock import MagicMock

from easi.core.episode import Action, EpisodeRecord, Observation, StepResult


class TestActionSpace:
    def test_has_four_actions(self):
        from easi.tasks.reverie_ce.actions import get_action_space
        actions = get_action_space()
        assert len(actions) == 4
        assert "move_forward" in actions
        assert "turn_left" in actions
        assert "turn_right" in actions
        assert "stop" in actions


class TestReverieCETask:
    @pytest.fixture
    def task(self):
        from easi.tasks.reverie_ce.task import ReverieCETask
        mock_config = {
            "name": "reverie_ce_val_unseen",
            "display_name": "REVERIE-CE Val Unseen",
            "simulator": "habitat_sim:v0_1_7",
            "task_class": "easi.tasks.reverie_ce.task.ReverieCETask",
            "max_steps": 500,
            "dataset": {"source": "huggingface", "repo_id": "oscarqjh/REVERIE-CE_easi", "split": "val_unseen"},
            "simulator_configs": {},
            "agent": {"prompt_builder": "easi.tasks.reverie_ce.prompts.ReverieCEPromptBuilder"},
        }
        task = ReverieCETask.__new__(ReverieCETask)
        task._config = mock_config
        task._yaml_path = None
        task._action_space = None
        return task

    def test_action_space(self, task):
        actions = task._build_action_space()
        assert actions == ["move_forward", "turn_left", "turn_right", "stop"]

    def test_format_reset_config(self, task):
        episode = {
            "episode_id": "50001",
            "scene_id": "cV4RVeZvu5T",
            "instruction": "Go to the laundry room and get the cushion",
            "start_position": [1.0, 0.5, -2.0],
            "start_rotation": [0, 0.707, 0, 0.707],
            "goal_position": [4.5, 0.5, 1.2],
            "geodesic_distance": 10.5,
            "gt_locations": [[1.0, 0.5, -2.0], [2.0, 0.5, -1.0]],
            "_data_dir": "/data/reverie_ce",
        }
        config = task.format_reset_config(episode)
        assert config["scene_id"] == "cV4RVeZvu5T"
        assert config["data_dir"] == "/data/reverie_ce"
        assert json.loads(config["start_position"]) == [1.0, 0.5, -2.0]

    def test_evaluate_episode_success(self, task):
        info = {
            "success": 1.0, "oracle_success": 1.0, "spl": 0.8,
            "navigation_error": 1.5, "ndtw": 0.9, "sdtw": 0.85,
            "path_length": 8.0,
        }
        obs = Observation(rgb_path="/tmp/step.png")
        step = StepResult(observation=obs, done=True, info=info)
        result = task.evaluate_episode({}, [step])
        assert result["success"] == 1.0
        assert result["spl"] == 0.8

    def test_evaluate_episode_empty(self, task):
        result = task.evaluate_episode({}, [])
        assert result["success"] is None
        assert result["path_length"] == 0.0

    def test_aggregate_results(self, task):
        records = [
            EpisodeRecord(episode={}, trajectory=[], episode_results={
                "success": 1.0, "oracle_success": 1.0, "spl": 0.8,
                "navigation_error": 1.5, "ndtw": 0.9, "sdtw": 0.85,
                "path_length": 8.0, "steps_taken": 30.0,
            }),
            EpisodeRecord(episode={}, trajectory=[], episode_results={
                "success": 0.0, "oracle_success": 0.0, "spl": 0.0,
                "navigation_error": 6.0, "ndtw": 0.3, "sdtw": 0.0,
                "path_length": 12.0, "steps_taken": 50.0,
            }),
        ]
        summary = task.aggregate_results(records)
        assert summary["num_episodes"] == 2
        assert summary["SR"] == 0.5
        assert summary["SPL"] == 0.4

    def test_bridge_script_path(self, task):
        path = task.get_bridge_script_path()
        assert path.name == "bridge.py"
        assert "reverie_ce" in str(path)


class TestReverieCEPromptBuilder:
    @pytest.fixture
    def mock_encode(self):
        # Must patch in vlnce_r2r.prompts where the function is actually
        # called (super().build_messages() resolves it there, not in
        # reverie_ce.prompts).
        import easi.tasks.vlnce_r2r.prompts as prompts_mod
        original = prompts_mod._encode_image_base64
        prompts_mod._encode_image_base64 = lambda x: "data:image/png;base64,AAAA"
        yield
        prompts_mod._encode_image_base64 = original

    def _make_memory(self, action_history=None):
        memory = MagicMock()
        memory.task_description = "Go to the laundry room and bring me the blue cushion"
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        memory.current_observation = Observation(
            rgb_path="/tmp/test.png",
            metadata={"geo_distance": "5.3"},
        )
        memory.action_history = action_history or []
        memory.steps = []
        return memory

    def test_system_prompt_mentions_high_level(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        system_msg = messages[0]["content"]
        assert "high-level" in system_msg.lower() or "described location" in system_msg.lower()

    def test_build_messages_has_image(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        user_content = messages[1]["content"]
        image_blocks = [b for b in user_content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_build_messages_has_instruction(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text_blocks = [b["text"] for b in messages[1]["content"] if b.get("type") == "text"]
        full_text = "\n".join(text_blocks)
        assert "laundry room" in full_text

    def test_build_messages_has_distance(self, mock_encode):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text_blocks = [b["text"] for b in messages[1]["content"] if b.get("type") == "text"]
        full_text = "\n".join(text_blocks)
        assert "5.3" in full_text

    def test_parse_response_valid(self):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        response = json.dumps({
            "visual_state_description": "I see a hallway",
            "reasoning_and_reflection": "Need to find the laundry room",
            "language_plan": "Move forward",
            "executable_plan": [{"action": "move_forward"}],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1
        assert actions[0].action_name == "move_forward"

    def test_parse_response_invalid_json(self):
        from easi.tasks.reverie_ce.prompts import ReverieCEPromptBuilder
        builder = ReverieCEPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        actions = builder.parse_response("not json", memory)
        assert actions == []
