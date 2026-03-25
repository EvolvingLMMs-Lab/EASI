"""Tests for ManipulaTHOR task integration (offline, no AI2-THOR)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from easi.core.episode import Action, EpisodeRecord, Observation, StepResult


# ── Task class tests ────────────────────────────────────────────────────────

class TestManipulaTHORTask:
    """Test task class methods."""

    @pytest.fixture
    def task(self):
        from easi.tasks.manipulathor.task import ManipulaTHORTask
        return ManipulaTHORTask()

    def test_action_space(self, task):
        assert len(task.action_space) == 13
        assert "MoveArmHeightP" in task.action_space
        assert "PickUpMidLevel" in task.action_space
        assert "DoneMidLevel" in task.action_space
        assert "MoveAheadContinuous" in task.action_space

    def test_max_steps(self, task):
        assert task.max_steps == 200

    def test_simulator_key(self, task):
        assert task.simulator_key == "ai2thor:v3_3_5"

    def test_bridge_script_path(self, task):
        bridge = task.get_bridge_script_path()
        assert bridge is not None
        assert bridge.name == "bridge.py"
        assert bridge.exists()

    def test_get_instruction(self, task):
        episode = {"object_type": "Apple"}
        assert task.get_instruction(episode) == "Pick up the Apple and move it to the goal location."

    def test_get_instruction_default(self, task):
        episode = {}
        assert task.get_instruction(episode) == "Pick up the object and move it to the goal location."

    def test_format_reset_config(self, task):
        episode = {
            "id": 42,
            "scene": "FloorPlan2_physics",
            "object_type": "Apple",
            "object_id": "Apple|+01.98|+00.77|-00.04",
            "source_position": '{"x": 1.0, "y": 0.5, "z": -0.3}',
            "target_position": '{"x": 2.0, "y": 0.8, "z": 0.1}',
            "initial_agent_pose": '{"x": 0.0, "y": 0.9, "z": 0.0, "rotation": 90, "horizon": 10}',
            "arm_starting_pose": '{"x": 0.5, "y": 0.9, "z": 0.2, "rotation": 0, "horizon": 30}',
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == "42"
        assert config["scene"] == "FloorPlan2_physics"
        assert config["object_id"] == "Apple|+01.98|+00.77|-00.04"
        assert config["source_position"] == '{"x": 1.0, "y": 0.5, "z": -0.3}'
        assert config["target_position"] == '{"x": 2.0, "y": 0.8, "z": 0.1}'

    def test_evaluate_episode_success(self, task):
        """Successful episode with pickup and success."""
        trajectory = [
            StepResult(
                observation=Observation(rgb_path="/tmp/test.png"),
                info={"episode_success": 0.0, "pickup_success": 0.0},
            ),
            StepResult(
                observation=Observation(rgb_path="/tmp/test.png"),
                info={
                    "episode_success": 1.0,
                    "pickup_success": 1.0,
                    "success_wo_disturb": 1.0,
                    "eplen_pickup": 3.0,
                    "eplen_success": 5.0,
                },
            ),
        ]
        metrics = task.evaluate_episode({}, trajectory)
        assert metrics["episode_success"] == 1.0
        assert metrics["pickup_success"] == 1.0
        assert metrics["success_wo_disturb"] == 1.0
        assert metrics["eplen_pickup"] == 3.0
        assert metrics["eplen_success"] == 5.0
        assert metrics["num_steps"] == 2.0

    def test_evaluate_episode_failure(self, task):
        """Failed episode — no pickup, no success."""
        trajectory = [
            StepResult(
                observation=Observation(rgb_path="/tmp/test.png"),
                info={"episode_success": 0.0, "pickup_success": 0.0},
            ),
        ]
        metrics = task.evaluate_episode({}, trajectory)
        assert metrics["episode_success"] == 0.0
        assert metrics["pickup_success"] == 0.0
        assert metrics["success_wo_disturb"] == 0.0
        assert metrics["num_steps"] == 1.0

    def test_evaluate_episode_empty(self, task):
        metrics = task.evaluate_episode({}, [])
        assert metrics["episode_success"] == 0.0
        assert metrics["num_steps"] == 0.0

    def test_aggregate_results(self, task):
        """Test custom aggregation with conditional averages."""
        records = [
            # Episode 1: success, no disturbance, pickup at step 5, 10 total steps
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "episode_success": 1.0,
                    "pickup_success": 1.0,
                    "success_wo_disturb": 1.0,
                    "eplen_pickup": 5.0,
                    "eplen_success": 10.0,
                    "num_steps": 10.0,
                },
            ),
            # Episode 2: success with disturbance, pickup at step 8, 15 total steps
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "episode_success": 1.0,
                    "pickup_success": 1.0,
                    "success_wo_disturb": 0.0,
                    "eplen_pickup": 8.0,
                    "eplen_success": 15.0,
                    "num_steps": 15.0,
                },
            ),
            # Episode 3: pickup only, no success, 20 steps
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "episode_success": 0.0,
                    "pickup_success": 1.0,
                    "success_wo_disturb": 0.0,
                    "eplen_pickup": 12.0,
                    "eplen_success": 0.0,
                    "num_steps": 20.0,
                },
            ),
            # Episode 4: total failure, 25 steps
            EpisodeRecord(
                episode={}, trajectory=[],
                episode_results={
                    "episode_success": 0.0,
                    "pickup_success": 0.0,
                    "success_wo_disturb": 0.0,
                    "eplen_pickup": 0.0,
                    "eplen_success": 0.0,
                    "num_steps": 25.0,
                },
            ),
        ]
        summary = task.aggregate_results(records)

        # Unconditional averages
        assert summary["episode_success_rate"] == 0.5      # 2/4
        assert summary["pickup_success_rate"] == 0.75       # 3/4
        assert summary["success_wo_disturb_rate"] == 0.25   # 1/4
        assert summary["avg_eplen"] == 17.5                 # (10+15+20+25)/4

        # Conditional: only episodes where pickup occurred (3 episodes)
        assert summary["avg_eplen_pickup"] == pytest.approx(
            (5.0 + 8.0 + 12.0) / 3, abs=0.1
        )

        # Conditional: only successful episodes (2 episodes)
        assert summary["avg_eplen_success"] == pytest.approx(
            (10.0 + 15.0) / 2, abs=0.1
        )

    def test_aggregate_results_empty(self, task):
        assert task.aggregate_results([]) == {}


# ── Actions tests ───────────────────────────────────────────────────────────

class TestActions:
    def test_action_space_length(self):
        from easi.tasks.manipulathor.actions import get_action_space
        actions = get_action_space()
        assert len(actions) == 13

    def test_action_names(self):
        from easi.tasks.manipulathor.actions import ACTION_SPACE
        expected = {
            "MoveArmHeightP", "MoveArmHeightM",
            "MoveArmXP", "MoveArmXM",
            "MoveArmYP", "MoveArmYM",
            "MoveArmZP", "MoveArmZM",
            "MoveAheadContinuous",
            "RotateRightContinuous", "RotateLeftContinuous",
            "PickUpMidLevel", "DoneMidLevel",
        }
        assert set(ACTION_SPACE) == expected

    def test_constants(self):
        from easi.tasks.manipulathor.actions import (
            MOVE_ARM_CONSTANT, MOVE_THR, MAX_STEPS, MANIPULATHOR_COMMIT_ID,
        )
        assert MOVE_ARM_CONSTANT == 0.05
        assert MOVE_THR == 0.01
        assert MAX_STEPS == 200
        assert len(MANIPULATHOR_COMMIT_ID) == 40  # SHA-1 hash


# ── Prompt builder tests ────────────────────────────────────────────────────

class TestPromptBuilder:
    @pytest.fixture
    def builder(self):
        from easi.tasks.manipulathor.prompts import ManipulaTHORPromptBuilder
        return ManipulaTHORPromptBuilder(n_shot=0, use_rgb=True, use_gps=True, use_depth=False)

    def test_build_messages_first_turn(self, builder, tmp_path):
        """First turn: image + system prompt + GPS state."""
        # Create a dummy image
        img_path = tmp_path / "step_0000.png"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(str(img_path))

        memory = MagicMock()
        memory.task_description = "Pick up the Apple and move it to the target location."
        memory.action_history = []
        memory.current_observation = Observation(
            rgb_path=str(img_path),
            metadata={
                "relative_current_obj_state": "[0.3, 0.1, 0.5, 0.0, 0.0, 0.0]",
                "relative_obj_to_goal": "[0.8, 0.2, 0.1]",
                "relative_agent_arm_to_obj": "[0.2, 0.0, 0.3]",
                "pickedup_object": "0.0",
            },
        )

        messages = builder.build_messages(memory)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        # Should have image + text
        assert any(c["type"] == "image_url" for c in content)
        assert any(c["type"] == "text" for c in content)

        text = next(c["text"] for c in content if c["type"] == "text")
        assert "Pick up the Apple" in text
        assert "Environment Feedback" in text
        assert "Object Held: No" in text
        # Strategy is now in system prompt
        system_text = messages[0]["content"]
        assert "Phase 1" in system_text

    def test_build_messages_gps_disabled(self, tmp_path):
        """When use_gps=False, no GPS section in prompt."""
        from easi.tasks.manipulathor.prompts import ManipulaTHORPromptBuilder
        builder = ManipulaTHORPromptBuilder(use_gps=False)

        img_path = tmp_path / "step_0000.png"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(str(img_path))

        memory = MagicMock()
        memory.task_description = "Pick up the Apple."
        memory.action_history = []
        memory.current_observation = Observation(
            rgb_path=str(img_path),
            metadata={"pickedup_object": "0.0"},
        )
        memory.steps = []
        messages = builder.build_messages(memory)
        text = next(c["text"] for c in messages[1]["content"] if c["type"] == "text")
        assert "Object Position" not in text
        assert "Arm-to-Object" not in text

    def test_build_messages_with_history(self, builder, tmp_path):
        """Subsequent turn: includes action history."""
        img_path = tmp_path / "step_0001.png"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(str(img_path))

        memory = MagicMock()
        memory.task_description = "Pick up the Apple."
        memory.action_history = [
            ("MoveArmZP", "Action 'MoveArmZP' succeeded."),
            ("MoveArmZP", "Action 'MoveArmZP' failed."),
        ]
        memory.current_observation = Observation(
            rgb_path=str(img_path),
            metadata={
                "relative_current_obj_state": "[0.1, 0.1, 0.2, 0.0, 0.0, 0.0]",
                "relative_obj_to_goal": "[0.6, 0.2, 0.1]",
                "relative_agent_arm_to_obj": "[0.05, 0.0, 0.1]",
                "pickedup_object": "0.0",
            },
        )

        messages = builder.build_messages(memory)
        text = next(c["text"] for c in messages[1]["content"] if c["type"] == "text")
        assert "Action History" in text
        assert "MoveArmZP" in text
        assert "succeeded" in text

    def test_parse_response_valid(self, builder):
        """Parse valid JSON response."""
        memory = MagicMock()
        response = json.dumps({
            "visual_state_description": "I see a kitchen.",
            "reasoning_and_reflection": "Need to reach the apple.",
            "language_plan": "Move arm forward, then pick up.",
            "executable_plan": [
                {"action_id": 6, "action_name": "MoveArmZP"},
                {"action_id": 11, "action_name": "PickUpMidLevel"},
            ],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "MoveArmZP"
        assert actions[1].action_name == "PickUpMidLevel"

    def test_parse_response_by_id_only(self, builder):
        """Parse response with action_id but no action_name."""
        memory = MagicMock()
        response = json.dumps({
            "executable_plan": [
                {"action_id": 8},  # MoveAheadContinuous
                {"action_id": 12},  # DoneMidLevel
            ],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "MoveAheadContinuous"
        assert actions[1].action_name == "DoneMidLevel"

    def test_parse_response_invalid_json(self, builder):
        """Malformed JSON returns empty list."""
        memory = MagicMock()
        actions = builder.parse_response("not json", memory)
        assert actions == []

    def test_parse_response_invalid_action(self, builder):
        """Invalid action name stops parsing."""
        memory = MagicMock()
        response = json.dumps({
            "executable_plan": [
                {"action_id": 6, "action_name": "MoveArmZP"},
                {"action_id": 99, "action_name": "InvalidAction"},
                {"action_id": 8, "action_name": "MoveAheadContinuous"},
            ],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1  # Stops at invalid action
        assert actions[0].action_name == "MoveArmZP"


# ── YAML and registry tests ────────────────────────────────────────────────

class TestYAMLConfig:
    def test_base_yaml_loads(self):
        from easi.tasks.yaml_utils import resolve_task_yaml
        base = Path(__file__).parent.parent / "easi" / "tasks" / "manipulathor" / "_base.yaml"
        config = resolve_task_yaml(base)
        assert config["simulator"] == "ai2thor:v3_3_5"
        assert config["max_steps"] == 200
        assert "ManipulaTHOR" in config["display_name"]

    def test_split_yaml_inherits(self):
        from easi.tasks.yaml_utils import resolve_task_yaml
        split = Path(__file__).parent.parent / "easi" / "tasks" / "manipulathor" / "manipulathor_test_seen.yaml"
        config = resolve_task_yaml(split)
        assert config["name"] == "manipulathor_test_seen"
        assert config["simulator"] == "ai2thor:v3_3_5"  # inherited
        assert config["dataset"]["split"] == "test_seen_obj"

    def test_all_splits_registered(self):
        """All 6 ManipulaTHOR splits should be discoverable."""
        from easi.tasks.registry import list_tasks
        all_tasks = list_tasks()
        manipulathor_tasks = [t for t in all_tasks if t.startswith("manipulathor_")]
        assert len(manipulathor_tasks) == 6
        expected = {
            "manipulathor_train",
            "manipulathor_val_seen",
            "manipulathor_val_novel",
            "manipulathor_test_seen",
            "manipulathor_test_novel",
            "manipulathor_seen_scenes_novel",
        }
        assert set(manipulathor_tasks) == expected
