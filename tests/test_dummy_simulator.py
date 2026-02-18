"""End-to-end test for the dummy simulator.

Tests the full vertical slice: SubprocessRunner → bridge.py → filesystem IPC.
"""

import pytest

from easi.core.episode import Action
from easi.simulators.dummy.v1.env_manager import DummyEnvManager
from easi.simulators.dummy.v1.simulator import DummySimulator
from easi.simulators.subprocess_runner import SubprocessRunner


@pytest.fixture
def dummy_simulator():
    """Create and start a dummy simulator, cleaning up after the test."""
    env_manager = DummyEnvManager()
    sim = DummySimulator()

    runner = SubprocessRunner(
        python_executable=env_manager.get_python_executable(),
        bridge_script_path=sim._get_bridge_script_path(),
        startup_timeout=10.0,
        command_timeout=10.0,
    )
    runner.launch()
    sim.set_runner(runner)

    yield sim

    sim.close()


def test_reset(dummy_simulator):
    """Test that reset returns a valid observation."""
    obs = dummy_simulator.reset("test_episode_001")
    assert obs.rgb_path.endswith(".png")
    assert obs.agent_pose is not None


def test_step(dummy_simulator):
    """Test that step returns a valid StepResult."""
    dummy_simulator.reset("test_episode_002")

    action = Action(action_name="MoveAhead")
    result = dummy_simulator.step(action)

    assert result.observation.rgb_path.endswith(".png")
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_multiple_steps(dummy_simulator):
    """Test running 5 steps and verifying the results."""
    dummy_simulator.reset("test_episode_003")

    for i in range(5):
        action = Action(action_name="MoveAhead")
        result = dummy_simulator.step(action)
        assert result.observation.rgb_path.endswith(".png")
        assert not result.done  # dummy is done at step 10

    # Step 6-9 should still not be done
    for i in range(4):
        result = dummy_simulator.step(Action(action_name="TurnLeft"))
        assert not result.done

    # Step 10 should be done
    result = dummy_simulator.step(Action(action_name="MoveAhead"))
    assert result.done


def test_stop_action(dummy_simulator):
    """Test that the Stop action ends the episode."""
    dummy_simulator.reset("test_episode_004")

    result = dummy_simulator.step(Action(action_name="Stop"))
    assert result.done


def test_is_running(dummy_simulator):
    """Test is_running reports correctly."""
    assert dummy_simulator.is_running()


def test_dummy_simulator_with_env_vars_wiring():
    """Verify DummyEnvManager.get_env_vars() returns {} and doesn't break launch."""
    env_manager = DummyEnvManager()
    assert env_manager.get_env_vars() == {}

    sim = DummySimulator()
    runner = SubprocessRunner(
        python_executable=env_manager.get_python_executable(),
        bridge_script_path=sim._get_bridge_script_path(),
        startup_timeout=10.0,
        command_timeout=10.0,
        extra_env=env_manager.get_env_vars() or None,
    )
    runner.launch()
    try:
        assert runner.is_alive()
    finally:
        runner.shutdown()


def test_registry_discovery():
    """Test that the dummy simulator is discoverable via the registry."""
    from easi.simulators.registry import get_simulator_entry, list_simulators

    sims = list_simulators()
    assert "dummy" in sims
    assert "dummy:v1" in sims

    entry = get_simulator_entry("dummy")
    assert entry.name == "dummy"
    assert entry.version == "v1"

    entry_explicit = get_simulator_entry("dummy:v1")
    assert entry_explicit.name == "dummy"
