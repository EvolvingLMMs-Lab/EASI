"""Tests for the bridge architecture refactor.

Verifies:
- AI2ThorBridge is importable and has generic interface
- EBAlfredBridge subclasses AI2ThorBridge with task-specific methods
- get_bridge_script_path() works on BaseTask and EBAlfredTask
- simulator_kwargs property works on BaseTask
- Generic thor_utils has no goal evaluators
- EB-Alfred thor_utils has goal evaluators
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
        assert not hasattr(bridge_cls, "_nav_obj")
        assert not hasattr(bridge_cls, "_pick")
        assert not hasattr(bridge_cls, "_put")
        assert not hasattr(bridge_cls, "_open")
        assert not hasattr(bridge_cls, "_close")
        assert not hasattr(bridge_cls, "_toggleon")
        assert not hasattr(bridge_cls, "_toggleoff")
        assert not hasattr(bridge_cls, "_slice")
        assert not hasattr(bridge_cls, "_drop")
        assert not hasattr(bridge_cls, "_get_obj_id_from_name")
        assert not hasattr(bridge_cls, "_get_object_prop")

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
            assert not hasattr(bridge, "cur_receptacle")
            assert not hasattr(bridge, "put_count_dict")
            assert not hasattr(bridge, "sliced")

    def test_accepts_simulator_kwargs(self):
        """Generic bridge should accept simulator_kwargs."""
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
    """Test that EBAlfredBridge properly extends AI2ThorBridge."""

    def test_importable(self):
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert EBAlfredBridge is not None

    def test_subclasses_ai2thor_bridge(self):
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert issubclass(EBAlfredBridge, AI2ThorBridge)

    def test_has_skill_methods(self):
        """EB-Alfred bridge should have all skill execution methods."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        bridge_cls = EBAlfredBridge
        assert hasattr(bridge_cls, "_execute_skill")
        assert hasattr(bridge_cls, "_restore_scene")
        assert hasattr(bridge_cls, "_update_states")
        assert hasattr(bridge_cls, "_nav_obj")
        assert hasattr(bridge_cls, "_pick")
        assert hasattr(bridge_cls, "_put")
        assert hasattr(bridge_cls, "_open")
        assert hasattr(bridge_cls, "_close")
        assert hasattr(bridge_cls, "_toggleon")
        assert hasattr(bridge_cls, "_toggleoff")
        assert hasattr(bridge_cls, "_slice")
        assert hasattr(bridge_cls, "_drop")
        assert hasattr(bridge_cls, "_get_obj_id_from_name")
        assert hasattr(bridge_cls, "_get_object_prop")

    def test_has_ebalfred_state(self):
        """EB-Alfred bridge should have task-specific state."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = EBAlfredBridge(workspace=tmpdir, data_dir=tmpdir)
            assert hasattr(bridge, "traj_data")
            assert hasattr(bridge, "cleaned_objects")
            assert hasattr(bridge, "cooled_objects")
            assert hasattr(bridge, "heated_objects")
            assert hasattr(bridge, "cur_receptacle")
            assert hasattr(bridge, "put_count_dict")
            assert hasattr(bridge, "sliced")

    def test_inherits_generic_methods(self):
        """EB-Alfred bridge should inherit generic methods from AI2ThorBridge."""
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        bridge_cls = EBAlfredBridge
        assert hasattr(bridge_cls, "start")
        assert hasattr(bridge_cls, "stop")
        assert hasattr(bridge_cls, "run")
        assert hasattr(bridge_cls, "_step")
        assert hasattr(bridge_cls, "_cache_reachable_positions")
        assert hasattr(bridge_cls, "_make_observation_response")
        assert hasattr(bridge_cls, "_find_close_reachable_position")
        assert hasattr(bridge_cls, "_angle_diff")

    def test_overrides_reset_and_step(self):
        """EB-Alfred bridge should override reset and step."""
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        # The methods should be overridden (different from parent)
        assert EBAlfredBridge.reset is not AI2ThorBridge.reset
        assert EBAlfredBridge.step is not AI2ThorBridge.step

    def test_run_is_inherited(self):
        """EB-Alfred bridge should inherit run() from AI2ThorBridge (not override)."""
        from easi.simulators.ai2thor.v2_1_0.bridge import AI2ThorBridge
        from easi.tasks.ebalfred.bridge import EBAlfredBridge
        assert EBAlfredBridge.run is AI2ThorBridge.run


# --- Thor utils separation tests ---

class TestThorUtilsSeparation:
    """Test that thor_utils is properly split between generic and EB-Alfred."""

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
        """Generic thor_utils should NOT have goal evaluation functions."""
        from easi.simulators.ai2thor.v2_1_0 import thor_utils
        assert not hasattr(thor_utils, "GOALS")
        assert not hasattr(thor_utils, "GOAL_EVALUATORS")
        assert not hasattr(thor_utils, "evaluate_goal_conditions")
        assert not hasattr(thor_utils, "get_targets_from_traj")
        assert not hasattr(thor_utils, "load_task_json")
        assert not hasattr(thor_utils, "load_task_json_with_repeat")

    def test_ebalfred_has_goal_evaluators(self):
        from easi.tasks.ebalfred import thor_utils
        assert hasattr(thor_utils, "GOALS")
        assert hasattr(thor_utils, "GOAL_EVALUATORS")
        assert hasattr(thor_utils, "evaluate_goal_conditions")
        assert hasattr(thor_utils, "get_targets_from_traj")
        assert hasattr(thor_utils, "load_task_json")
        assert hasattr(thor_utils, "load_task_json_with_repeat")

    def test_ebalfred_goals_list(self):
        from easi.tasks.ebalfred.thor_utils import GOALS
        assert len(GOALS) == 7
        assert "pick_and_place_simple" in GOALS
        assert "pick_two_obj_and_place" in GOALS

    def test_ebalfred_evaluators_dict(self):
        from easi.tasks.ebalfred.thor_utils import GOAL_EVALUATORS
        assert len(GOAL_EVALUATORS) == 7
        assert "pick_and_place_simple" in GOAL_EVALUATORS
        assert "pick_heat_then_place_in_recep" in GOAL_EVALUATORS

    def test_ebalfred_imports_from_generic(self):
        """EB-Alfred thor_utils should import from generic thor_utils."""
        from easi.tasks.ebalfred.thor_utils import evaluate_goal_conditions
        # The function should be callable
        assert callable(evaluate_goal_conditions)


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


# --- simulator_kwargs tests ---

class TestSimulatorKwargs:
    """Test simulator_kwargs property on BaseTask."""

    def test_ebalfred_has_simulator_kwargs(self):
        from easi.tasks.ebalfred.task import EBAlfredTask
        task = EBAlfredTask()
        kwargs = task.simulator_kwargs
        assert isinstance(kwargs, dict)
        assert kwargs.get("quality") == "MediumCloseFitShadows"
        assert kwargs.get("screen_height") == 500
        assert kwargs.get("screen_width") == 500

    def test_dummy_task_empty_simulator_kwargs(self):
        from easi.tasks.dummy_task.task import DummyTask
        task = DummyTask()
        kwargs = task.simulator_kwargs
        assert isinstance(kwargs, dict)
        assert kwargs == {}

    def test_all_ebalfred_splits_have_kwargs(self):
        """All EB-Alfred split YAMLs should have simulator_kwargs."""
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
            kwargs = task.simulator_kwargs
            assert isinstance(kwargs, dict), f"{name} simulator_kwargs is not a dict"
            assert "quality" in kwargs, f"{name} missing quality in simulator_kwargs"


# --- Protocol tests ---

class TestTaskProtocol:
    """Test that TaskProtocol includes new methods."""

    def test_protocol_has_get_bridge_script_path(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "get_bridge_script_path")

    def test_protocol_has_simulator_kwargs(self):
        from easi.core.protocols import TaskProtocol
        assert hasattr(TaskProtocol, "simulator_kwargs")
