"""Tests for BaseBridge — wraps Gym-like envs in EASI IPC."""
import json
import threading

import numpy as np
from pathlib import Path

from easi.simulators.base_bridge import BaseBridge


class FakeGymEnv:
    """Minimal Gym-like env for testing."""

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action):
        self.step_count += 1
        obs = {"rgb": np.ones((8, 8, 3), dtype=np.uint8) * self.step_count}
        reward = 1.0 if action == "Stop" else 0.0
        done = action == "Stop" or self.step_count >= 5
        info = {
            "task_success": 1.0 if action == "Stop" else 0.0,
            "feedback": "ok",
        }
        return obs, reward, done, info

    def close(self):
        pass


class ConcreteBridge(BaseBridge):
    """Test bridge wrapping FakeGymEnv."""

    def _create_env(self, reset_config, simulator_kwargs):
        return FakeGymEnv()

    def _extract_image(self, obs):
        return obs["rgb"]


class TestBaseBridgeUnit:

    def test_bridge_creates_env_on_reset(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        response = bridge.reset({})
        assert response["status"] == "ok"
        assert "rgb_path" in response["observation"]
        assert Path(response["observation"]["rgb_path"]).exists()

    def test_bridge_step_returns_observation(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        response = bridge.step("MoveAhead")
        assert response["status"] == "ok"
        assert response["done"] is False

    def test_bridge_step_done_on_stop(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        response = bridge.step("Stop")
        assert response["done"] is True
        assert response["info"]["task_success"] == 1.0

    def test_bridge_saves_image_to_episode_output_dir(self, tmp_path):
        episode_dir = tmp_path / "episode_out"
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.episode_output_dir = str(episode_dir)
        bridge.reset({})
        assert (episode_dir / "step_0000.png").exists()

    def test_bridge_saves_image_to_workspace_by_default(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        assert (tmp_path / "step_0000.png").exists()

    def test_bridge_step_increments_count(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        bridge.step("MoveAhead")
        bridge.step("MoveAhead")
        assert (tmp_path / "step_0001.png").exists()
        assert (tmp_path / "step_0002.png").exists()

    def test_reset_clears_step_count(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        bridge.step("MoveAhead")
        bridge.reset({})  # Second reset
        bridge.step("MoveAhead")
        # After second reset, step count restarts at 1
        assert (tmp_path / "step_0001.png").exists()

    def test_close_sets_env_to_none(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        bridge.reset({})
        assert bridge.env is not None
        bridge.close()
        assert bridge.env is None

    def test_extract_info_filters_non_serializable(self, tmp_path):
        bridge = ConcreteBridge(workspace=tmp_path)
        info = {
            "score": 1.0,
            "label": "good",
            "flag": True,
            "count": 5,
            "array": np.array([1, 2]),  # should be filtered out
        }
        result = bridge._extract_info(info)
        assert result == {"score": 1.0, "label": "good", "flag": True, "count": 5}


class TestBaseBridgeIPC:
    """Test the full IPC loop using BaseBridge."""

    def test_ipc_reset_step_close(self, tmp_path):
        """Simulate the IPC protocol manually."""
        from easi.communication.filesystem import (
            poll_for_response,
            poll_for_status,
            write_command,
        )
        from easi.communication.schemas import make_reset_command, make_step_command
        from easi.core.episode import Action

        bridge = ConcreteBridge(workspace=tmp_path)

        # Run bridge in background thread
        thread = threading.Thread(target=bridge.run, daemon=True)
        thread.start()

        # Wait for status.json
        status = poll_for_status(tmp_path, timeout=5.0)
        assert status["ready"] is True

        # Send reset
        reset_cmd = make_reset_command("ep_001", reset_config={})
        write_command(tmp_path, reset_cmd)
        response = poll_for_response(tmp_path, timeout=5.0)
        assert response["status"] == "ok"
        assert "rgb_path" in response["observation"]

        # Send step
        step_cmd = make_step_command(Action(action_name="MoveAhead"))
        write_command(tmp_path, step_cmd)
        response = poll_for_response(tmp_path, timeout=5.0)
        assert response["status"] == "ok"
        assert response["done"] is False

        # Send close
        write_command(tmp_path, {"type": "close"})
        response = poll_for_response(tmp_path, timeout=5.0)
        assert response["status"] == "ok"

        thread.join(timeout=5)
