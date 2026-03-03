"""Tests for LHPR-VLN task integration."""
import json
import pytest
from unittest.mock import MagicMock

from easi.core.episode import Action, EpisodeRecord, Observation, StepResult
from easi.tasks.lhpr_vln.actions import get_action_space
from easi.tasks.lhpr_vln.vendor.metrics import NavigationMetrics


class TestActionSpace:
    def test_has_four_actions(self):
        actions = get_action_space()
        assert len(actions) == 4
        assert "move_forward" in actions
        assert "turn_left" in actions
        assert "turn_right" in actions
        assert "stop" in actions


class TestNavigationMetrics:
    """Verify vendored metrics compute correctly."""

    def test_perfect_episode(self):
        m = NavigationMetrics()
        m.add_sample(
            success=1, gt_step=10, path_step=10, oracle_success=1,
            navigation_error=0.5, subtask_successes=[1, 1],
            subtask_path_step=[5, 5], gt_length=[3.0, 4.0],
            error_length=[0.3, 0.5],
        )
        result = m.compute()
        assert result["success_rate"] == 1.0
        assert result["oracle_success_rate"] == 1.0
        assert result["spl"] == 1.0
        assert result["independent_success_rate"] == 1.0

    def test_failed_episode(self):
        m = NavigationMetrics()
        m.add_sample(
            success=0, gt_step=10, path_step=50, oracle_success=0,
            navigation_error=5.0, subtask_successes=[0, 0],
            subtask_path_step=[5, 5], gt_length=[3.0, 4.0],
            error_length=[3.0, 5.0],
        )
        result = m.compute()
        assert result["success_rate"] == 0.0
        assert result["spl"] == 0.0
        assert result["independent_success_rate"] == 0.0

    def test_partial_success(self):
        m = NavigationMetrics()
        m.add_sample(
            success=0, gt_step=30, path_step=45, oracle_success=0,
            navigation_error=2.5, subtask_successes=[1, 0, 0],
            subtask_path_step=[10, 10, 10], gt_length=[3.0, 5.0, 4.0],
            error_length=[0.5, 3.0, 2.5],
        )
        result = m.compute()
        assert result["success_rate"] == 0.0
        assert result["independent_success_rate"] == pytest.approx(1/3, abs=0.01)

    def test_contest_score_formula(self):
        m = NavigationMetrics()
        m.add_sample(
            success=1, gt_step=20, path_step=25, oracle_success=1,
            navigation_error=0.5, subtask_successes=[1, 1],
            subtask_path_step=[10, 10], gt_length=[5.0, 5.0],
            error_length=[0.3, 0.5],
        )
        result = m.compute()
        score = 0.4 * result["tar"] + 0.2 * result["independent_success_rate"] + \
                0.2 * result["conditional_success_rate"] + 0.2 * result["conditional_path_length"]
        assert score > 0


class TestLHPRVLNTask:
    """Test task class without simulator."""

    @pytest.fixture
    def task(self):
        """Create task with mocked config loading."""
        from easi.tasks.lhpr_vln.task import LHPRVLNTask

        mock_config = {
            "name": "lhpr_vln_val",
            "display_name": "LHPR-VLN Val",
            "simulator": "habitat_sim:v0_3_0",
            "task_class": "easi.tasks.lhpr_vln.task.LHPRVLNTask",
            "max_steps": 500,
            "dataset": {"source": "huggingface", "repo_id": "oscarqjh/LHPR-VLN_easi", "split": "val"},
            "simulator_configs": {},
            "agent": {"prompt_builder": "easi.tasks.lhpr_vln.prompts.LHPRVLNPromptBuilder"},
        }
        task = LHPRVLNTask.__new__(LHPRVLNTask)
        task._config = mock_config
        task._yaml_path = None
        task._action_space = None
        return task

    def test_format_reset_config(self, task):
        episode = {
            "id": "batch_6_episode_0",
            "instruction": "Find the chair then the table.",
            "scene": "00706-abcdef",
            "robot": "spot",
            "objects": ["chair", "table"],
            "regions": ["3", "5"],
            "rooms": ["living room", "kitchen"],
            "gt_steps": [40, 55],
            "subtask_list": ["Move_to('chair_3')", "Move_to('table_5')"],
            "num_targets": 2,
            "batch": "batch_6",
            "_data_dir": "/data/lhpr",
        }
        config = task.format_reset_config(episode)
        assert config["scene_id"] == "00706-abcdef"
        assert config["robot"] == "spot"
        assert config["targets"] == ["chair", "table"]
        assert config["regions"] == ["3", "5"]
        assert config["gt_step"] == [40, 55]

    def test_evaluate_episode_empty_trajectory(self, task):
        result = task.evaluate_episode({}, [])
        assert result["task_success"] == 0.0
        assert result["num_steps"] == 0.0

    def test_evaluate_episode_successful(self, task):
        last_info = {
            "task_success": 1.0,
            "subtask_successes": "[1, 1]",
            "subtask_oracle_successes": "[1, 1]",
            "subtask_nav_errors": "[0.5, 0.3]",
            "subtask_nav_steps": "[40, 55]",
            "gt_steps": "[40, 55]",
            "gt_paths": "[5.0, 8.0]",
        }
        obs = Observation(rgb_path="/tmp/step.png")
        step = StepResult(observation=obs, done=True, info=last_info)
        result = task.evaluate_episode({}, [step])
        assert result["task_success"] == 1.0
        assert result["spl"] == 1.0
        assert result["num_subtasks"] == 2.0

    def test_aggregate_results_all_metrics(self, task):
        records = [
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "task_success": 1.0,
                    "_subtask_successes": "[1, 1]",
                    "_subtask_oracle_successes": "[1, 1]",
                    "_subtask_nav_errors": "[0.5, 0.3]",
                    "_subtask_nav_steps": "[40, 55]",
                    "_gt_steps": "[40, 55]",
                    "_gt_paths": "[5.0, 8.0]",
                },
            ),
        ]
        summary = task.aggregate_results(records)
        # All 8 metrics present
        for key in ["SR", "OSR", "SPL", "NE", "ISR", "CSR", "CGT", "TAR"]:
            assert key in summary, f"Missing metric: {key}"
        assert "contest_score" in summary
        assert "num_episodes" in summary
        assert summary["num_episodes"] == 1
        assert summary["SR"] == 1.0


class TestPromptBuilder:
    def test_build_messages_has_system_and_user(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.task_description = "Find the chair then the table."
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        memory.current_observation = Observation(
            rgb_path="/tmp/test_front.png",
            metadata={
                "subtask_stage": "0", "subtask_total": "2", "current_geo_distance": "5.3",
                "left_rgb_path": "/tmp/test_left.png",
                "front_rgb_path": "/tmp/test_front.png",
                "right_rgb_path": "/tmp/test_right.png",
            },
        )
        memory.action_history = [("move_forward", "Subtask 1/2: navigate to chair")]
        # Mock the image encoding to avoid file I/O
        import easi.tasks.lhpr_vln.prompts as prompts_mod
        original_encode = prompts_mod._encode_image_base64
        prompts_mod._encode_image_base64 = lambda x: "data:image/png;base64,AAAA"
        try:
            messages = builder.build_messages(memory)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            # Should have 3 image blocks (left, front, right) + text labels
            user_content = messages[1]["content"]
            image_blocks = [b for b in user_content if b.get("type") == "image_url"]
            assert len(image_blocks) == 3
        finally:
            prompts_mod._encode_image_base64 = original_encode

    def test_parse_response_json(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        actions = builder.parse_response('{"action": "turn_left"}', memory)
        assert len(actions) == 1
        assert actions[0].action_name == "turn_left"

    def test_parse_response_fallback(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        actions = builder.parse_response("I should stop here.", memory)
        assert actions[0].action_name == "stop"
