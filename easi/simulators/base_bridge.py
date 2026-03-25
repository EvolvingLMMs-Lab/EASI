"""Base bridge that wraps any Gym-like env in EASI's IPC protocol.

Subclass this to integrate external benchmark envs with minimal code.
Override _create_env() and _extract_image() — the base handles IPC,
image saving, and observation formatting.

Two extension patterns:
- Pattern A: Override _create_env() + _extract_image() for Gym-like envs
- Pattern B: Override reset() + step() directly for raw simulator control

Usage in a task-specific bridge.py:

    from easi.simulators.base_bridge import BaseBridge

    class MyBenchmarkBridge(BaseBridge):
        def _create_env(self, reset_config, simulator_kwargs):
            # simulator_kwargs comes from task YAML's simulator_configs
            # (with additional_deps stripped — bridge only gets runtime config)
            from external_package import MyEnv
            return MyEnv(**reset_config)

        def _extract_image(self, obs):
            return obs["head_rgb"]  # np.ndarray (H, W, 3)

    if __name__ == "__main__":
        MyBenchmarkBridge.main()
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable (for bridge subprocess)
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.communication.filesystem import (
    poll_for_command,
    write_response,
    write_status,
)
from easi.communication.schemas import (
    make_error_response,
    make_observation_response,
)
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class BaseBridge:
    """Wraps a Gym-like env (reset/step/close) in EASI's filesystem IPC.

    Subclasses must implement:
        _create_env(reset_config, simulator_kwargs) -> env object
        _extract_image(obs) -> np.ndarray (H, W, 3) uint8

    Optional overrides:
        _extract_info(info) -> dict  (filter/transform env info)
        _on_reset(env, reset_config) -> obs  (custom reset logic)
        _on_step(env, action_text) -> (obs, reward, done, info)
    """

    def __init__(self, workspace, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.simulator_kwargs = simulator_kwargs or {}
        self.env = None
        self.step_count = 0
        self.episode_output_dir = None

    # --- Override these ---

    def _create_env(self, reset_config, simulator_kwargs):
        """Create and return a Gym-like env. Called on first reset."""
        raise NotImplementedError

    def _extract_image(self, obs):
        """Extract RGB numpy array (H, W, 3) from env observation."""
        raise NotImplementedError

    def _extract_info(self, info):
        """Transform env info dict to EASI-compatible info. Override to filter."""
        return {k: v for k, v in info.items()
                if isinstance(v, (int, float, str, bool))}

    def _on_reset(self, env, reset_config):
        """Custom reset logic. Default: call env.reset(). Override if env
        needs episode data passed differently."""
        return env.reset()

    def _on_step(self, env, action_text):
        """Custom step logic. Default: call env.step(action_text)."""
        return env.step(action_text)

    def _get_topdown_map(self):
        """Return (rgb_array, metadata_dict) or None.

        Called once after _on_reset() inside the bridge subprocess.
        Override in subclasses that support top-down map rendering.
        """
        return None

    def _get_episode_meta(self):
        """Return episode metadata dict or None.

        Called once after _on_reset(). Override to persist gt_locations,
        goal_position, start_position, etc. for post-processing.
        """
        return None

    # --- Image saving ---

    def _save_image(self, image_array):
        """Save RGB numpy array as PNG, return path string."""
        from PIL import Image

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / ("step_%04d.png" % self.step_count)
        Image.fromarray(image_array).save(str(rgb_path))
        return str(rgb_path)

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Build IPC observation response from env output."""
        image = self._extract_image(obs)
        rgb_path = self._save_image(image)
        clean_info = self._extract_info(info or {})
        clean_info["step"] = str(self.step_count)
        return make_observation_response(
            rgb_path=rgb_path,
            agent_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata={"step": str(self.step_count)},
            reward=reward,
            done=done,
            info=clean_info,
        )

    # --- Reset / Step ---

    def reset(self, reset_config):
        """Reset env (create if first call) and return observation response."""
        if self.env is None:
            self.env = self._create_env(reset_config, self.simulator_kwargs)
        self.step_count = 0
        obs = self._on_reset(self.env, reset_config)

        # Save topdown map and episode metadata (optional, non-fatal)
        if self.episode_output_dir:
            save_dir = Path(self.episode_output_dir)
            try:
                map_result = self._get_topdown_map()
                if map_result is not None:
                    map_image, map_meta = map_result
                    from PIL import Image
                    Image.fromarray(map_image).save(str(save_dir / "topdown_map.png"))
                    with open(save_dir / "topdown_map_meta.json", "w") as f:
                        json.dump(map_meta, f)
            except Exception:
                logger.warning("Failed to save topdown map, continuing without it")

            try:
                ep_meta = self._get_episode_meta()
                if ep_meta is not None:
                    with open(save_dir / "episode_meta.json", "w") as f:
                        json.dump(ep_meta, f)
            except Exception:
                logger.warning("Failed to save episode metadata, continuing without it")

        return self._make_response(obs)

    def step(self, action_text):
        """Execute one action and return observation response."""
        self.step_count += 1
        obs, reward, done, info = self._on_step(self.env, action_text)
        return self._make_response(obs, reward=reward, done=done, info=info)

    def close(self):
        """Shut down the env."""
        if self.env is not None:
            if hasattr(self.env, "close"):
                self.env.close()
            self.env = None

    # --- IPC loop ---

    def run(self):
        """Main IPC loop. Polls commands, dispatches to reset/step/close."""
        write_status(self.workspace, ready=True)

        while True:
            try:
                command = poll_for_command(self.workspace, timeout=300.0)
            except Exception as e:
                logger.error("Failed to read command: %s", e)
                break

            cmd_type = command.get("type")

            if cmd_type == "reset":
                reset_config = command.get("reset_config", {})
                raw_output_dir = command.get("episode_output_dir")
                if raw_output_dir:
                    self.episode_output_dir = raw_output_dir
                    Path(raw_output_dir).mkdir(parents=True, exist_ok=True)
                else:
                    self.episode_output_dir = None

                try:
                    response = self.reset(reset_config)
                    write_response(self.workspace, response)
                except Exception as e:
                    logger.exception("Reset failed")
                    write_response(self.workspace, make_error_response(str(e)))

            elif cmd_type == "step":
                action_text = command.get("action", {}).get("action_name", "")
                try:
                    response = self.step(action_text)
                    write_response(self.workspace, response)
                except Exception as e:
                    logger.exception("Step failed")
                    write_response(self.workspace, make_error_response(str(e)))

            elif cmd_type == "close":
                logger.info("Close command received")
                self.close()
                write_response(self.workspace, {"status": "ok"})
                break

            else:
                write_response(
                    self.workspace,
                    make_error_response("Unknown command: %s" % cmd_type),
                )

    # --- CLI entry point ---

    @classmethod
    def main(cls):
        """Standard CLI entry point for bridge subclasses."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--workspace", type=Path, required=True)
        parser.add_argument("--data-dir", type=Path, default=None)
        parser.add_argument("--simulator-kwargs", type=str, default=None)
        args, _ = parser.parse_known_args()

        setup_logging("TRACE")

        sim_kwargs = json.loads(args.simulator_kwargs) if args.simulator_kwargs else {}
        if args.data_dir:
            sim_kwargs["data_dir"] = str(args.data_dir)

        bridge = cls(workspace=args.workspace, simulator_kwargs=sim_kwargs)
        bridge.run()
