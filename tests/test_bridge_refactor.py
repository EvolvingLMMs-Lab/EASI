"""Tests for the bridge architecture refactor.

Verifies:
- AI2ThorBridge is importable and has generic interface (unchanged)
- EBAlfredBridge subclasses BaseBridge (not AI2ThorBridge)
- Vendor code is importable
- get_bridge_script_path() works on BaseTask and EBAlfredTask
- simulator_configs / additional_deps / simulator_kwargs properties work
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --- Generic AI2ThorBridge tests ---

class TestAI2ThorBridgeImport:
    """Test that the generic AI2ThorBridge is properly structured."""

    def test_importable(self):
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        assert AI2ThorBridge is not None

    def test_has_generic_methods(self):
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        bridge_cls = AI2ThorBridge
        assert hasattr(bridge_cls, "start")
        assert hasattr(bridge_cls, "stop")
        assert hasattr(bridge_cls, "reset")
        assert hasattr(bridge_cls, "step")
        assert hasattr(bridge_cls, "run")
        assert hasattr(bridge_cls, "_step")
        assert hasattr(bridge_cls, "_cache_reachable_positions")
        assert hasattr(bridge_cls, "_make_observation_response")
        assert hasattr(bridge_cls, "_find_close_reachable_position")
        assert hasattr(bridge_cls, "_angle_diff")

    def test_no_ebalfred_methods(self):
        """Generic bridge should NOT have EB-Alfred-specific methods."""
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        bridge_cls = AI2ThorBridge
        assert not hasattr(bridge_cls, "_execute_skill")
        assert not hasattr(bridge_cls, "_restore_scene")
        assert not hasattr(bridge_cls, "_update_states")

    def test_no_ebalfred_state(self):
        """Generic bridge __init__ should NOT have EB-Alfred state."""
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AI2ThorBridge(workspace=tmpdir)
            assert not hasattr(bridge, "traj_data")
            assert not hasattr(bridge, "cleaned_objects")
            assert not hasattr(bridge, "cooled_objects")
            assert not hasattr(bridge, "heated_objects")

    def test_accepts_simulator_kwargs(self):
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {"quality": "Low", "screen_height": 300}
            bridge = AI2ThorBridge(workspace=tmpdir, simulator_kwargs=kwargs)
            assert bridge.simulator_kwargs == kwargs

    def test_simulator_kwargs_default_empty(self):
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AI2ThorBridge(workspace=tmpdir)
            assert bridge.simulator_kwargs == {}


# --- EBAlfredBridge tests ---

class TestEBAlfredBridgeImport:
    """Test that EBAlfredBridge properly extends BaseBridge."""

    def test_importable(self):
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert EBAlfredBridge is not None

    def test_subclasses_base_bridge(self):
        """EBAlfredBridge should extend BaseBridge, not AI2ThorBridge."""
        from easi.simulators.base_bridge import BaseBridge
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert issubclass(EBAlfredBridge, BaseBridge)

    def test_not_subclass_of_ai2thor_bridge(self):
        """EBAlfredBridge should NOT extend AI2ThorBridge anymore."""
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert not issubclass(EBAlfredBridge, AI2ThorBridge)

    def test_has_base_bridge_overrides(self):
        """EBAlfredBridge should override BaseBridge methods."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        bridge_cls = EBAlfredBridge
        assert hasattr(bridge_cls, "_create_env")
        assert hasattr(bridge_cls, "_on_reset")
        assert hasattr(bridge_cls, "_on_step")
        assert hasattr(bridge_cls, "_extract_image")
        assert hasattr(bridge_cls, "_extract_info")

    def test_no_skill_methods_on_bridge(self):
        """Skill methods now live in vendor's ThorConnector, not bridge."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        bridge_cls = EBAlfredBridge
        assert not hasattr(bridge_cls, "_execute_skill")
        assert not hasattr(bridge_cls, "_restore_scene")
        assert not hasattr(bridge_cls, "_update_states")
        assert not hasattr(bridge_cls, "_nav_obj")
        assert not hasattr(bridge_cls, "_pick")
        assert not hasattr(bridge_cls, "_put")

    def test_instantiable(self):
        """EBAlfredBridge can be instantiated with workspace."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = EBAlfredBridge(workspace=tmpdir)
            assert bridge.env is None
            assert bridge.step_count == 0

    def test_accepts_simulator_kwargs(self):
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {"screen_height": 500}
            bridge = EBAlfredBridge(workspace=tmpdir, simulator_kwargs=kwargs)
            assert bridge.simulator_kwargs == kwargs


# --- Vendor import tests ---

class TestVendorImports:
    """Test that vendor files are properly importable."""

    def test_vendor_package_importable(self):
        import easi.tasks.ebalfred.vendor
        assert easi.tasks.ebalfred.vendor is not None

    def test_vendor_utils_importable(self):
        from easi.tasks.ebalfred.vendor.utils import (
            alfred_objs,
            alfred_pick_obj,
            alfred_open_obj,
            alfred_slice_obj,
            alfred_toggle_obj,
            alfred_recep,
            load_task_json,
            natural_word_to_ithor_name,
            dotdict,
        )
        assert len(alfred_objs) > 0
        assert callable(load_task_json)
        assert callable(natural_word_to_ithor_name)

    def test_vendor_constants_importable(self):
        from easi.tasks.ebalfred.vendor.gen import constants
        assert hasattr(constants, "AGENT_STEP_SIZE")
        assert hasattr(constants, "CAMERA_HEIGHT_OFFSET")
        assert hasattr(constants, "VISIBILITY_DISTANCE")
        assert hasattr(constants, "GOALS")

    def test_vendor_no_embodiedbench_imports(self):
        """No vendor file should import from embodiedbench."""
        import ast
        from pathlib import Path
        vendor_dir = Path("easi/tasks/ebalfred/vendor")
        for py_file in vendor_dir.rglob("*.py"):
            source = py_file.read_text()
            assert "embodiedbench" not in source, (
                f"{py_file} still contains 'embodiedbench' import"
            )

    def test_vendor_no_splits_json(self):
        """splits.json should be removed — episode data comes from HF dataset."""
        from pathlib import Path
        splits_path = Path("easi/tasks/ebalfred/vendor/data/splits/splits.json")
        assert not splits_path.exists()

    def test_vendor_rewards_json_exists(self):
        from pathlib import Path
        rewards_path = Path("easi/tasks/ebalfred/vendor/models/config/rewards.json")
        assert rewards_path.exists()

    def test_vendor_layouts_exist(self):
        from pathlib import Path
        layouts_dir = Path("easi/tasks/ebalfred/vendor/gen/layouts")
        npy_files = list(layouts_dir.glob("*.npy"))
        assert len(npy_files) > 0, "No layout .npy files found"


# --- Generic thor_utils tests (unchanged) ---

class TestGenericThorUtils:
    """Test that generic thor_utils in simulators/ is still intact."""

    def test_generic_has_constants(self):
        from easi.simulators.ai2thor.v2_1_0 import thor_utils
        assert hasattr(thor_utils, "SCREEN_WIDTH")
        assert hasattr(thor_utils, "SCREEN_HEIGHT")
        assert hasattr(thor_utils, "CAMERA_HEIGHT_OFFSET")
        assert hasattr(thor_utils, "VISIBILITY_DISTANCE")
        assert hasattr(thor_utils, "AGENT_STEP_SIZE")

    def test_generic_has_object_helpers(self):
        from easi.simulators.ai2thor.v2_1_0 import thor_utils
        assert hasattr(thor_utils, "natural_word_to_ithor_name")
        assert hasattr(thor_utils, "get_objects_of_type")
        assert hasattr(thor_utils, "get_objects_with_name_and_prop")
        assert hasattr(thor_utils, "get_obj_of_type_closest_to_obj")

    def test_generic_has_no_goal_evaluators(self):
        from easi.simulators.ai2thor.v2_1_0 import thor_utils
        assert not hasattr(thor_utils, "GOALS")
        assert not hasattr(thor_utils, "GOAL_EVALUATORS")
        assert not hasattr(thor_utils, "evaluate_goal_conditions")


# --- get_bridge_script_path tests ---

class TestGetBridgeScriptPath:
    """Test get_bridge_script_path on various task classes."""

    def test_ebalfred_task_returns_path(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        bridge_path = task.get_bridge_script_path()
        assert bridge_path is not None
        assert isinstance(bridge_path, Path)
        assert bridge_path.name == "bridge.py"
        assert "ebalfred" in str(bridge_path)

    def test_ebalfred_bridge_path_exists(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        bridge_path = task.get_bridge_script_path()
        assert bridge_path.exists(), f"Bridge script not found at {bridge_path}"

    def test_dummy_task_returns_none(self):
        from easi.tasks.dummy_task.task import DummyTask
        task = DummyTask()
        assert task.get_bridge_script_path() is None

    def test_ebalfred_bridge_path_different_from_simulator(self):
        """Task bridge should point to easi/tasks/ebalfred/bridge.py,
        not easi/simulators/ai2thor/v2_1_0/bridge.py."""
        from easi.simulators.ai2thor.v2_1_0.simulator import AI2ThorSimulatorV210
        from easi.tasks.ebalfred.task import EBAlfredTask

        task = EBAlfredTask()
        sim = AI2ThorSimulatorV210()

        task_bridge = task.get_bridge_script_path()
        sim_bridge = sim._get_bridge_script_path()

        assert task_bridge != sim_bridge
        assert "tasks" in str(task_bridge)
        assert "simulators" in str(sim_bridge)


# --- simulator_configs tests ---

class TestSimulatorConfigs:
    """Test simulator_configs and additional_deps on BaseTask."""

    def test_ebalfred_has_simulator_configs(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        configs = task.simulator_configs
        assert isinstance(configs, dict)
        assert configs.get("quality") == "MediumCloseFitShadows"
        assert configs.get("screen_height") == 500

    def test_ebalfred_has_additional_deps(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        deps = task.additional_deps
        assert isinstance(deps, list)
        assert "gym" in deps
        assert "networkx" in deps
        assert "opencv-python" in deps

    def test_ebalfred_simulator_kwargs_excludes_deps(self):
        """simulator_kwargs should NOT contain additional_deps."""
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        kwargs = task.simulator_kwargs
        assert "additional_deps" not in kwargs
        assert kwargs.get("quality") == "MediumCloseFitShadows"
        assert kwargs.get("screen_height") == 500

    def test_ebalfred_no_eval_set(self):
        """eval_set removed -- episode data comes from HF dataset directly."""
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        kwargs = task.simulator_kwargs
        assert "eval_set" not in kwargs

    def test_dummy_task_empty_simulator_configs(self):
        from easi.tasks.dummy_task.task import DummyTask
        task = DummyTask()
        assert task.simulator_configs == {}
        assert task.additional_deps == []
        assert task.simulator_kwargs == {"max_steps": 100}

    def test_all_ebalfred_splits_have_configs(self):
        """All EB-Alfred split YAMLs should have simulator_configs."""
        from easi.tasks.registry import get_task_entry, load_task_class

        ebalfred_names = [
            "ebalfred_base",
            "ebalfred_long_horizon",
            "ebalfred_common_sense",
            "ebalfred_complex_instruction",
            "ebalfred_spatial",
            "ebalfred_visual_appearance",
        ]
        for name in ebalfred_names:
            entry = get_task_entry(name)
            TaskClass = load_task_class(name)
            task = TaskClass(split_yaml_path=entry.config_path)
            configs = task.simulator_configs
            assert isinstance(configs, dict), f"{name} simulator_configs is not a dict"
            assert "quality" in configs, f"{name} missing quality in simulator_configs"
            assert "additional_deps" in configs, f"{name} missing additional_deps"


# --- Protocol tests ---

class TestTaskProtocol:
    """Test that TaskProtocol includes new methods."""

    def test_protocol_has_get_bridge_script_path(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "get_bridge_script_path")

    def test_protocol_has_simulator_kwargs(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "simulator_kwargs")

    def test_protocol_has_simulator_configs(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "simulator_configs")

    def test_protocol_has_additional_deps(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "additional_deps")


# --- Task simplification tests ---

class TestTaskSimplification:
    """Test that EBAlfredTask is properly simplified."""

    def test_format_reset_config_has_episode_data(self):
        """reset_config should pass episode data directly to bridge."""
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        episode = {
            "id": 42,
            "task": "pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001",
            "repeat_idx": 0,
            "instruction": "Put a mug on the shelf.",
        }
        config = task.format_reset_config(episode)
        assert config["episode_id"] == 42
        assert config["task"] == episode["task"]
        assert config["repeat_idx"] == 0
        assert config["instruction"] == "Put a mug on the shelf."

    def test_format_reset_config_no_episode_idx(self):
        """No episode_idx needed — episode data is passed directly."""
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        episode = {
            "id": 0,
            "task": "test/trial_T00000001",
            "repeat_idx": 0,
            "instruction": "test",
        }
        config = task.format_reset_config(episode)
        assert "episode_idx" not in config
        assert "task_path" not in config

    def test_evaluate_episode_empty_trajectory(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        metrics = task.evaluate_episode({}, [])
        assert metrics["task_success"] == 0.0
        assert metrics["num_steps"] == 0.0

    def test_builtin_episodes_have_required_fields(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        episodes = task._get_builtin_episodes()
        for ep in episodes:
            assert "task" in ep
            assert "repeat_idx" in ep
            assert "instruction" in ep

    def test_no_thor_utils_in_ebalfred_tasks(self):
        """thor_utils.py should be removed from easi/tasks/ebalfred/."""
        thor_utils_path = Path("easi/tasks/ebalfred/thor_utils.py")
        assert not thor_utils_path.exists(), "thor_utils.py should be deleted"
