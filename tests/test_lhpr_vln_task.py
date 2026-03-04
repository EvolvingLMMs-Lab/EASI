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
                episode={"robot": "spot"}, trajectory=[],
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
        # Nested structure with "base" group
        assert "base" in summary
        base = summary["base"]
        for key in ["SR", "OSR", "SPL", "NE", "ISR", "CSR", "CGT", "TAR"]:
            assert key in base, f"Missing metric: {key}"
        assert "contest_score" in base
        assert base["num_episodes"] == 1
        assert base["SR"] == 1.0

    def test_aggregate_results_grouped_by_robot(self, task):
        """Metrics should be nested by robot type (spot, stretch)."""
        spot_results = {
            "_subtask_successes": "[1, 1]",
            "_subtask_oracle_successes": "[1, 1]",
            "_subtask_nav_errors": "[0.5, 0.3]",
            "_subtask_nav_steps": "[40, 55]",
            "_gt_steps": "[40, 55]",
            "_gt_paths": "[5.0, 8.0]",
        }
        stretch_results = {
            "_subtask_successes": "[0, 0]",
            "_subtask_oracle_successes": "[0, 0]",
            "_subtask_nav_errors": "[5.0, 4.0]",
            "_subtask_nav_steps": "[100, 100]",
            "_gt_steps": "[40, 55]",
            "_gt_paths": "[5.0, 8.0]",
        }
        records = [
            EpisodeRecord(episode={"robot": "spot"}, trajectory=[], episode_results=spot_results),
            EpisodeRecord(episode={"robot": "stretch"}, trajectory=[], episode_results=stretch_results),
        ]
        summary = task.aggregate_results(records)

        # Base group (all episodes)
        assert summary["base"]["num_episodes"] == 2

        # Robot-specific nested groups
        assert summary["spot"]["num_episodes"] == 1
        assert summary["stretch"]["num_episodes"] == 1
        assert summary["spot"]["SR"] == 1.0
        assert summary["stretch"]["SR"] == 0.0

        # All 8 metrics present in each group
        for group in ["base", "spot", "stretch"]:
            for key in ["SR", "OSR", "SPL", "NE", "ISR", "CSR", "CGT", "TAR", "contest_score"]:
                assert key in summary[group], f"Missing {group}.{key}"


class TestPromptBuilder:
    @pytest.fixture
    def mock_encode(self):
        """Mock image encoding to avoid file I/O."""
        import easi.tasks.lhpr_vln.prompts as prompts_mod
        original = prompts_mod._encode_image_base64
        prompts_mod._encode_image_base64 = lambda x: "data:image/png;base64,AAAA"
        yield
        prompts_mod._encode_image_base64 = original

    def _make_memory(self, action_history=None):
        memory = MagicMock()
        memory.task_description = "Find the chair then the table."
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        memory.is_first_turn = (action_history is None or len(action_history) == 0)
        memory.current_observation = Observation(
            rgb_path="/tmp/test_front.png",
            metadata={
                "subtask_stage": "0", "subtask_total": "2",
                "current_geo_distance": "5.3",
                "current_target": "chair",
                "left_rgb_path": "/tmp/test_left.png",
                "front_rgb_path": "/tmp/test_front.png",
                "right_rgb_path": "/tmp/test_right.png",
                "agent_position": "[1.5, 0.0, -3.2]",
                "target_coordinate": "[4.1, 0.0, -1.8]",
            },
        )
        memory.action_history = action_history or []
        memory.steps = []
        return memory

    def test_build_messages_single_user_message(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # Should have 3 image blocks + 3 text labels + 1 prompt text
        user_content = messages[0]["content"]
        image_blocks = [b for b in user_content if b.get("type") == "image_url"]
        assert len(image_blocks) == 3

    def test_prompt_contains_env_feedback(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text_blocks = [b["text"] for b in messages[0]["content"] if b.get("type") == "text"]
        prompt_text = "\n".join(text_blocks)
        assert "Current subtask: 1/2" in prompt_text
        assert "target object: chair" in prompt_text
        assert "Geodesic distance" in prompt_text
        assert "5.30m" in prompt_text

    def test_agent_position_toggle(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        # Off by default
        builder = LHPRVLNPromptBuilder(use_agent_position=False)
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Agent position" not in text
        # On
        builder = LHPRVLNPromptBuilder(use_agent_position=True)
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Agent position" in text
        assert "1.50" in text

    def test_target_coordinate_toggle(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        # Off by default
        builder = LHPRVLNPromptBuilder(use_target_coordinate=False)
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Target coordinate" not in text
        # On
        builder = LHPRVLNPromptBuilder(use_target_coordinate=True)
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Target coordinate" in text
        assert "4.10" in text

    def test_geo_distance_toggle(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder(use_geo_distance=False)
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Geodesic distance" not in text

    def test_subtask_progress_toggle(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder(use_subtask_progress=False)
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "Current subtask" not in text

    def test_action_history_with_feedback(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = self._make_memory(
            action_history=[("move_forward", "Subtask 1/2: navigate to chair. Distance: 5.3m")]
        )
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "action history" in text
        assert "env feedback:" in text
        assert "move_forward" in text

    def test_action_history_without_feedback(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder(use_feedback=False)
        memory = self._make_memory(
            action_history=[("move_forward", "Subtask 1/2: navigate to chair")]
        )
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "action history" in text
        assert "env feedback:" not in text

    def test_parse_response_executable_plan(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        response = json.dumps({
            "visual_state_description": "I see a hallway",
            "reasoning_and_reflection": "Need to move forward",
            "language_plan": "Step 1: move forward",
            "executable_plan": [
                {"action_id": 0, "action_name": "move_forward"},
                {"action_id": 1, "action_name": "turn_left"},
            ],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "move_forward"
        assert actions[1].action_name == "turn_left"

    def test_parse_response_action_id_lookup(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        builder.set_action_space(memory.action_space)
        # action_id 3 = "stop"
        response = json.dumps({
            "visual_state_description": "Close to target",
            "reasoning_and_reflection": "Within range",
            "language_plan": "Step 1: stop",
            "executable_plan": [{"action_id": 3, "action_name": "stop"}],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1
        assert actions[0].action_name == "stop"

    def test_parse_response_invalid_json(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        memory.action_space = ["move_forward", "turn_left", "turn_right", "stop"]
        actions = builder.parse_response("not valid json at all", memory)
        assert actions == []

    def test_get_response_format_returns_schema(self):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        memory = MagicMock()
        schema = builder.get_response_format(memory)
        assert schema["type"] == "json_schema"
        props = schema["json_schema"]["schema"]["properties"]
        assert "visual_state_description" in props
        assert "reasoning_and_reflection" in props
        assert "language_plan" in props
        assert "executable_plan" in props

    def test_first_turn_vs_subsequent_action_count(self, mock_encode):
        from easi.tasks.lhpr_vln.prompts import LHPRVLNPromptBuilder
        builder = LHPRVLNPromptBuilder()
        # First turn: "1-3 actions"
        memory = self._make_memory()
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "1-3 actions" in text
        # Subsequent turn: "3-5 actions"
        memory = self._make_memory(
            action_history=[("move_forward", "ok")]
        )
        messages = builder.build_messages(memory)
        text = "\n".join(b["text"] for b in messages[0]["content"] if b.get("type") == "text")
        assert "3-5 actions" in text
