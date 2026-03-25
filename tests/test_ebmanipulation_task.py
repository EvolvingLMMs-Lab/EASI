"""Tests for the EB-Manipulation task (offline, no simulator needed)."""
import json

import pytest
from pathlib import Path


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False

from easi.core.episode import Action, Observation, StepResult
from easi.tasks.ebmanipulation.task import EBManipulationTask
from easi.tasks.ebmanipulation.actions import (
    get_action_space,
    serialize_action,
    deserialize_action,
    extract_pose_list,
    EVAL_SETS,
    VALID_EVAL_SETS,
    DEFAULT_VOXEL_SIZE,
    DEFAULT_ROTATION_RESOLUTION,
)


class TestEBManipulationActions:
    def test_get_action_space_empty(self):
        actions = get_action_space()
        assert actions == []

    def test_serialize_action(self):
        action = [50, 42, 17, 6, 61, 36, 1]
        s = serialize_action(action)
        assert s == "[50, 42, 17, 6, 61, 36, 1]"

    def test_deserialize_action(self):
        s = "[50, 42, 17, 6, 61, 36, 1]"
        result = deserialize_action(s)
        assert result == [50, 42, 17, 6, 61, 36, 1]

    def test_deserialize_action_roundtrip(self):
        action = [50, 42, 17, 6, 61, 36, 1]
        s = serialize_action(action)
        result = deserialize_action(s)
        assert result == action

    def test_deserialize_invalid(self):
        assert deserialize_action("not a list") == []
        assert deserialize_action("[1, 2, 3]") == []  # Not 7 elements
        assert deserialize_action("") == []

    def test_extract_pose_list(self):
        text = "[[50, 42, 17, 6, 61, 36, 1], [28, 32, 26, 0, 60, 94, 0]]"
        poses = extract_pose_list(text)
        assert len(poses) == 2
        assert poses[0] == [50, 42, 17, 6, 61, 36, 1]
        assert poses[1] == [28, 32, 26, 0, 60, 94, 0]

    def test_extract_pose_list_skip_non_7d(self):
        text = "[[50, 42, 17], [28, 32, 26, 0, 60, 94, 0]]"
        poses = extract_pose_list(text)
        assert len(poses) == 1
        assert poses[0] == [28, 32, 26, 0, 60, 94, 0]

    def test_eval_sets_have_correct_splits(self):
        assert set(EVAL_SETS.keys()) == {"base", "common_sense", "complex", "spatial", "visual"}

    def test_valid_eval_sets(self):
        assert len(VALID_EVAL_SETS) == 5

    def test_defaults(self):
        assert DEFAULT_VOXEL_SIZE == 100
        assert DEFAULT_ROTATION_RESOLUTION == 3


class TestEBManipulationUtils:
    def test_point_to_voxel_index(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
            point_to_voxel_index,
        )
        import numpy as np

        # Center of scene should map to ~50
        point = np.array([0.2, 0.0, 1.1])
        voxel = point_to_voxel_index(point)
        assert len(voxel) == 3
        assert all(0 <= v <= 100 for v in voxel)

    def test_task_handlers_exist(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import TASK_HANDLERS

        assert "pick" in TASK_HANDLERS
        assert "stack" in TASK_HANDLERS
        assert "place" in TASK_HANDLERS
        assert "wipe" in TASK_HANDLERS

    def test_task_handler_has_sim_name_mapping(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import TASK_HANDLERS

        for task_type, handler_cls in TASK_HANDLERS.items():
            handler = handler_cls()
            assert hasattr(handler, "sim_name_to_real_name")
            assert isinstance(handler.sim_name_to_real_name, dict)
            assert len(handler.sim_name_to_real_name) > 0

    @pytest.mark.skipif(
        not _has_scipy(), reason="scipy not installed (simulator-only dep)"
    )
    def test_get_continuous_action_defaults(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
            get_continuous_action_from_discrete,
        )
        import numpy as np

        action = [50, 50, 50, 60, 60, 60, 1.0]
        result = get_continuous_action_from_discrete(action)
        assert isinstance(result, np.ndarray)
        assert len(result) == 8  # xyz(3) + quat(4) + gripper(1)
        assert result[-1] == 1.0  # gripper state preserved

    @pytest.mark.skipif(
        not _has_scipy(), reason="scipy not installed (simulator-only dep)"
    )
    def test_get_continuous_action_custom_params(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
            get_continuous_action_from_discrete,
        )
        import numpy as np

        action = [50, 50, 50, 60, 60, 60, 0.0]
        result = get_continuous_action_from_discrete(
            action,
            scene_bounds=np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]),
            voxel_size=100,
            rotation_resolution=3,
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == 8
        assert result[-1] == 0.0

    @pytest.mark.skipif(
        not _has_scipy(), reason="scipy not installed (simulator-only dep)"
    )
    def test_discrete_euler_to_quaternion(self):
        from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
            discrete_euler_to_quaternion,
        )
        import numpy as np

        quat = discrete_euler_to_quaternion(np.array([60, 60, 60]))
        assert len(quat) == 4
        # Quaternion norm should be ~1
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-6


class TestEBManipulationTask:
    @pytest.fixture
    def task(self):
        return EBManipulationTask()

    def test_name(self, task):
        assert task.name == "ebmanipulation_base"

    def test_simulator_key(self, task):
        assert task.simulator_key == "coppeliasim:v4_1_0"

    def test_action_space_empty(self, task):
        assert task.action_space == []

    def test_max_steps(self, task):
        assert task.max_steps == 15

    def test_format_reset_config(self, task):
        episode = {
            "id": 0,
            "task_name": "pick_cube_shape",
            "variation": 0,
            "episode_num": 0,
            "instruction": "Pick up the star and place it into the yellow container.",
            "task_type": "pick",
            "_data_dir": "/tmp/test_data",
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == 0
        assert config["data_dir"] == "/tmp/test_data"
        assert config["split"] == "base"
        assert config["task_name"] == "pick_cube_shape"
        assert config["variation"] == 0
        assert config["episode_num"] == 0
        assert config["instruction"] == "Pick up the star and place it into the yellow container."
        assert config["task_type"] == "pick"

    def test_format_reset_config_uses_data_dir(self, task):
        episode = {
            "id": 1,
            "task_name": "stack_cubes_color",
            "variation": 2,
            "episode_num": 1,
            "instruction": "Stack the cubes.",
            "task_type": "stack",
            "_data_dir": "/some/path",
        }
        config = task.format_reset_config(episode)
        assert config["data_dir"] == "/some/path"

    def test_evaluate_episode(self, task):
        episode = {
            "id": 0,
            "task_name": "pick_cube_shape",
            "variation": 0,
            "episode_num": 0,
            "instruction": "test",
            "task_type": "pick",
        }
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(
                observation=obs,
                reward=0.0,
                done=False,
                info={"task_success": 0.0, "action_success": 1.0},
            ),
            StepResult(
                observation=obs,
                reward=1.0,
                done=True,
                info={"task_success": 1.0, "action_success": 1.0},
            ),
        ]
        metrics = task.evaluate_episode(episode, trajectory)
        assert metrics["task_success"] == 1.0
        assert metrics["num_steps"] == 2.0
        assert metrics["action_success_rate"] == 1.0

    def test_evaluate_empty_trajectory(self, task):
        episode = {
            "id": 0,
            "task_name": "pick_cube_shape",
            "variation": 0,
            "episode_num": 0,
            "instruction": "test",
            "task_type": "pick",
        }
        metrics = task.evaluate_episode(episode, [])
        assert metrics["task_success"] == 0.0
        assert metrics["num_steps"] == 0.0
        assert metrics["action_success_rate"] == 0.0

    def test_get_instruction(self, task):
        episode = {"instruction": "Pick up the star."}
        assert task.get_instruction(episode) == "Pick up the star."

    def test_builtin_episodes(self, task):
        episodes = task._get_builtin_episodes()
        assert len(episodes) >= 1
        ep = episodes[0]
        assert "id" in ep
        assert "task_name" in ep
        assert "variation" in ep
        assert "episode_num" in ep
        assert "instruction" in ep
        assert "task_type" in ep

    def test_task_registry_discovers_all_splits(self):
        from easi.tasks.registry import list_tasks, refresh

        refresh()
        tasks = list_tasks()
        manip_tasks = [t for t in tasks if t.startswith("ebmanipulation")]
        assert len(manip_tasks) == 5
        expected = {
            "ebmanipulation_base",
            "ebmanipulation_common_sense",
            "ebmanipulation_complex",
            "ebmanipulation_spatial",
            "ebmanipulation_visual",
        }
        assert set(manip_tasks) == expected

    def test_bridge_script_path(self, task):
        bridge_path = task.get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"

    def test_yaml_has_zip_files(self, task):
        config = task._config
        dataset_config = config.get("dataset", {})
        zip_files = dataset_config.get("zip_files", [])
        assert "simulator_data.zip" in zip_files


class TestEBManipulationPromptBuilder:
    @pytest.fixture
    def builder(self):
        from easi.tasks.ebmanipulation.prompts import EBManipulationPromptBuilder

        return EBManipulationPromptBuilder(n_shot=2, split="base")

    @pytest.fixture
    def memory(self):
        from easi.core.memory import AgentMemory

        obs = Observation(
            rgb_path="/dev/null",
            metadata={
                "task_variation": "pick_cube_shape",
                "avg_obj_coord": "{'object 1': [45, 13, 18], 'object 2': [72, 20, 18]}",
            },
        )
        return AgentMemory(
            task_description="Pick up the star and place it into the yellow container.",
            action_space=[],
            current_observation=obs,
        )

    def test_build_messages_first_turn(self, builder, memory):
        msgs = builder.build_messages(memory)
        assert len(msgs) >= 1
        assert msgs[0]["role"] == "user"
        text = msgs[0]["content"][0]["text"]
        assert "Franka Panda robot" in text
        assert "7D discrete gripper action" in text

    def test_system_prompt_format_placeholders(self, builder, memory):
        msgs = builder.build_messages(memory)
        text = msgs[0]["content"][0]["text"]
        # Should have voxel_size and rotation params filled in
        assert "[0, 100]" in text
        assert "3 degrees" in text

    def test_per_task_examples(self, builder):
        assert "pick" in builder._examples
        assert "stack" in builder._examples
        assert "place" in builder._examples
        assert "wipe" in builder._examples
        assert len(builder._examples["pick"]) >= 2
        assert len(builder._examples["stack"]) >= 4
        assert len(builder._examples["place"]) >= 4
        assert len(builder._examples["wipe"]) >= 8

    def test_output_template_is_manip(self, builder, memory):
        msgs = builder.build_messages(memory)
        text = msgs[0]["content"][0]["text"]
        # Should contain template_manip content
        assert "visual_state_description" in text
        assert "executable_plan" in text

    def test_parse_response_7d_arrays(self, builder, memory):
        response = json.dumps(
            {
                "visual_state_description": "I see objects.",
                "reasoning_and_reflection": "Pick up the star.",
                "language_plan": "1. Move to star.",
                "executable_plan": "[[50, 42, 17, 6, 61, 36, 1], [50, 42, 17, 6, 61, 36, 0]]",
            }
        )
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "[50, 42, 17, 6, 61, 36, 1]"
        assert actions[1].action_name == "[50, 42, 17, 6, 61, 36, 0]"

    def test_parse_response_list_format(self, builder, memory):
        response = json.dumps(
            {
                "visual_state_description": "test",
                "reasoning_and_reflection": "test",
                "language_plan": "test",
                "executable_plan": [[50, 42, 17, 6, 61, 36, 1]],
            }
        )
        actions = builder.parse_response(response, memory)
        assert len(actions) == 1

    def test_parse_response_invalid_json(self, builder, memory):
        actions = builder.parse_response("not json", memory)
        assert actions == []

    def test_parse_response_empty_plan(self, builder, memory):
        response = json.dumps(
            {
                "visual_state_description": "test",
                "reasoning_and_reflection": "test",
                "language_plan": "test",
                "executable_plan": "[]",
            }
        )
        actions = builder.parse_response(response, memory)
        assert actions == []

    def test_response_schema_executable_plan_is_string(self, builder, memory):
        schema = builder.get_response_format(memory)
        props = schema["json_schema"]["schema"]["properties"]
        assert props["executable_plan"]["type"] == "string"

    def test_conforms_to_protocol(self, builder):
        assert hasattr(builder, "build_messages")
        assert hasattr(builder, "parse_response")
        assert hasattr(builder, "get_response_format")
        assert hasattr(builder, "set_action_space")

    def test_set_action_space_noop(self, builder):
        # Should not raise
        builder.set_action_space(["a", "b", "c"])
        builder.set_action_space([])
