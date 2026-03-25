"""EB-Alfred bridge — wraps EmbodiedBench's EBAlfEnv via BaseBridge.

This script runs inside the easi_ai2thor_v2_1_0 conda env (Python 3.8).
It communicates with the parent process via filesystem IPC.

The vendor/ directory contains EBAlfEnv (Gym env) copied from EmbodiedBench
with minimal path changes. This bridge is a thin wrapper that delegates all
skill execution, scene restoration, and goal evaluation to EBAlfEnv.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--data-dir /path/to/datasets] [--simulator-kwargs '{}']
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


class EBAlfredBridge(BaseBridge):
    """Thin BaseBridge wrapper around EmbodiedBench's EBAlfEnv."""

    def _create_env(self, reset_config, simulator_kwargs):
        import os
        from easi.tasks.ebalfred.vendor.EBAlfEnv import EBAlfEnv

        # data_dir from reset_config has the correct task-level path
        # (e.g. datasets/oscarqjh_EB-Alfred_easi/tasks)
        data_dir = reset_config.get("data_dir") or simulator_kwargs.get("data_dir")
        resolution = simulator_kwargs.get("screen_height", 500)
        max_steps = simulator_kwargs.get("max_steps", 30)
        max_invalid_actions = simulator_kwargs.get("max_invalid_actions", 10)
        feedback_verbosity = simulator_kwargs.get("feedback_verbosity", 0)
        detection_box = simulator_kwargs.get("detection_box", False)
        # x_display from YAML, falling back to DISPLAY env var (set by xvfb-run)
        x_display = simulator_kwargs.get(
            "x_display", os.environ.get("DISPLAY", ":0").lstrip(":")
        )
        return EBAlfEnv(
            resolution=resolution,
            data_dir=data_dir,
            x_display=x_display,
            max_steps=max_steps,
            max_invalid_actions=max_invalid_actions,
            feedback_verbosity=feedback_verbosity,
            detection_box=detection_box,
        )

    def _on_reset(self, env, reset_config):
        episode = {
            "task": reset_config["task"],
            "repeat_idx": reset_config["repeat_idx"],
            "instruction": reset_config["instruction"],
        }
        return env.reset(episode=episode)

    def _on_step(self, env, action_text):
        return env.step(action_text)

    def _extract_image(self, obs):
        return obs["head_rgb"]

    def _extract_info(self, info):
        return {
            "task_success": float(info.get("task_success", 0.0)),
            "task_progress": float(info.get("task_progress", 0.0)),
            "last_action_success": float(info.get("last_action_success", 0.0)),
            "feedback": str(info.get("env_feedback", "")),
        }

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to include dynamic action space in reset response metadata."""
        response = super()._make_response(obs, reward, done, info)
        # After reset, env.language_skill_set contains the per-episode
        # dynamic action space (global actions + scene-specific instances).
        if self.env is not None and hasattr(self.env, 'language_skill_set'):
            response["observation"]["metadata"]["dynamic_action_space"] = json.dumps(
                self.env.language_skill_set
            )
        return response


if __name__ == "__main__":
    EBAlfredBridge.main()
