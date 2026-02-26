"""ManipulaTHOR bridge for EASI's filesystem IPC protocol.

Runs in the easi_ai2thor_v3_3_5 conda env (Python 3.8).
Extends BaseBridge to wrap ManipulaTHOREnv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class ManipulaTHORBridge(BaseBridge):
    """Bridge for ManipulaTHOR Arm Point Navigation task.

    Wraps ManipulaTHOREnv in EASI's IPC protocol. Adds GPS state
    sensors and depth information to observations.
    """

    def _create_env(self, reset_config, simulator_kwargs):
        from easi.tasks.manipulathor.vendor.manipulathor_env import ManipulaTHOREnv

        # Pass any controller overrides from simulator_kwargs
        controller_kwargs = {}
        for k in ("width", "height", "gridSize", "fieldOfView",
                   "visibilityDistance", "renderDepthImage"):
            if k in simulator_kwargs:
                controller_kwargs[k] = simulator_kwargs[k]

        # Enable depth rendering for GPS-like observations
        controller_kwargs.setdefault("renderDepthImage", True)

        verbose_feedback = simulator_kwargs.get("verbose_feedback", True)
        env = ManipulaTHOREnv(
            controller_kwargs=controller_kwargs,
            verbose_feedback=verbose_feedback,
        )
        return env

    def _on_reset(self, env, reset_config):
        """Parse reset_config and delegate to ManipulaTHOREnv.reset()."""
        # Parse JSON strings from HF dataset back to dicts
        config = dict(reset_config)
        for key in ("source_position", "target_position",
                     "initial_agent_pose", "arm_starting_pose"):
            if isinstance(config.get(key), str):
                config[key] = json.loads(config[key])

        return env.reset(config)

    def _on_step(self, env, action_text):
        """Execute named action via ManipulaTHOREnv.step()."""
        return env.step(action_text)

    def _extract_image(self, obs):
        """Extract RGB frame from observation dict."""
        return obs["frame"]

    def _extract_info(self, info):
        """Pass through all metric fields from ManipulaTHOREnv."""
        return {k: v for k, v in info.items()
                if isinstance(v, (int, float, str, bool))}

    def _save_images(self, obs):
        """Save RGB and depth images with <id>_<type>.png naming."""
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)
        step_id = "%04d" % self.step_count

        # RGB
        rgb_path = save_dir / f"{step_id}_rgb.png"
        Image.fromarray(obs["frame"]).save(str(rgb_path))

        # Depth (turbo colormap PNG)
        depth_path = None
        if obs.get("depth_frame") is not None:
            depth = obs["depth_frame"]
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth)
            colormap = cm.get_cmap("turbo")
            depth_colored = (colormap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            depth_path = save_dir / f"{step_id}_depth.png"
            Image.fromarray(depth_colored).save(str(depth_path))
            depth_path = str(depth_path)

        return str(rgb_path), depth_path

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to add GPS state + depth to observation metadata."""
        rgb_path, depth_path = self._save_images(obs)

        # Build metadata with GPS state
        metadata = {"step": str(self.step_count)}
        if self.env is not None:
            gps = self.env.get_gps_state()
            metadata["relative_current_obj_state"] = json.dumps(gps["relative_current_obj_state"])
            metadata["relative_obj_to_goal"] = json.dumps(gps["relative_obj_to_goal"])
            metadata["relative_agent_arm_to_obj"] = json.dumps(gps["relative_agent_arm_to_obj"])
            metadata["pickedup_object"] = str(gps["pickedup_object"])

        # Build info
        clean_info = self._extract_info(info or {})
        clean_info["step"] = str(self.step_count)

        from easi.communication.schemas import make_observation_response
        return make_observation_response(
            rgb_path=rgb_path,
            depth_path=depth_path,
            agent_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata=metadata,
            reward=reward,
            done=done,
            info=clean_info,
        )


if __name__ == "__main__":
    ManipulaTHORBridge.main()
