"""EB-Navigation bridge -- wraps vendored EBNavEnv via BaseBridge.

This script runs inside the easi_ai2thor_v5_0_0 conda env (Python 3.10).
Communicates with parent process via filesystem IPC.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


class EBNavigationBridge(BaseBridge):
    """Thin BaseBridge wrapper around vendored EBNavEnv."""

    # Action text -> integer ID mapping (populated on env creation)
    _action_map: dict[str, int] = {}

    def _create_env(self, reset_config, simulator_kwargs):
        from easi.tasks.ebnavigation.actions import ACTION_NAME_TO_ID
        from easi.tasks.ebnavigation.vendor.EBNavEnv import EBNavEnv

        self._action_map = ACTION_NAME_TO_ID
        resolution = simulator_kwargs.get("screen_height", 500)
        max_steps = simulator_kwargs.get("max_steps", 20)
        fov = simulator_kwargs.get("fov", 100)
        success_threshold = simulator_kwargs.get("success_threshold", 1.0)
        grid_size = simulator_kwargs.get("grid_size", 0.1)
        visibility_distance = simulator_kwargs.get("visibility_distance", 10.0)
        boundingbox = simulator_kwargs.get("boundingbox", False)
        return EBNavEnv(
            resolution=resolution,
            fov=fov,
            max_steps=max_steps,
            success_threshold=success_threshold,
            grid_size=grid_size,
            visibility_distance=visibility_distance,
            boundingbox=boundingbox,
        )

    def _on_reset(self, env, reset_config):
        episode = {
            "scene": reset_config["scene"],
            "agent_pose": json.loads(reset_config["agent_pose"])
            if isinstance(reset_config["agent_pose"], str)
            else reset_config["agent_pose"],
            "target_object_id": reset_config["target_object_id"],
            "target_position": json.loads(reset_config["target_position"])
            if isinstance(reset_config["target_position"], str)
            else reset_config["target_position"],
            "instruction": reset_config["instruction"],
        }
        return env.reset(episode=episode)

    def _on_step(self, env, action_text):
        action_id = self._action_map.get(action_text, -1)
        if action_id < 0:
            # Try case-insensitive match
            for name, idx in self._action_map.items():
                if name.lower() == action_text.lower():
                    action_id = idx
                    break
        return env.step(action_id)

    def _extract_image(self, obs):
        return obs["head_rgb"]

    def _extract_info(self, info):
        return {
            "task_success": float(info.get("task_success", 0.0)),
            "distance": float(info.get("distance", 0.0)),
            "last_action_success": float(info.get("last_action_success", 0.0)),
            "feedback": str(info.get("env_feedback", "")),
            "action_id": int(info.get("action_id", -1)),
        }


if __name__ == "__main__":
    EBNavigationBridge.main()
