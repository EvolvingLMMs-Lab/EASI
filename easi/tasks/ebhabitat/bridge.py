"""EB-Habitat bridge — wraps vendored EBHabEnv via BaseBridge.

This script runs inside the easi_habitat_sim_v0_3_0 conda env (Python 3.10).
Communicates with parent process via filesystem IPC.

The bridge handles:
- Creating EBHabEnv with data_dir/dataset_dir from extracted zip files
- Mapping action text -> integer ID for env.step()
- Sending dynamic action space in reset response metadata
- Extracting task_success, task_progress, subgoal_reward from env info

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--data-dir /path/to/data] [--simulator-kwargs '{}']
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


class EBHabitatBridge(BaseBridge):
    """BaseBridge wrapper around vendored EBHabEnv."""

    _action_map: dict[str, int] = {}
    _language_skill_set: list[str] = []

    def _create_env(self, reset_config, simulator_kwargs):
        from easi.tasks.ebhabitat.vendor.EBHabEnv import EBHabEnv

        # Resolve data directories from extracted zip files
        data_dir = simulator_kwargs.get("data_dir")
        dataset_dir = None
        if data_dir:
            # datasets.zip extracts pickle files to data_dir/datasets/
            candidate = Path(data_dir) / "datasets"
            if candidate.exists():
                dataset_dir = str(candidate)

        eval_set = reset_config.get("eval_set", "base")
        resolution = simulator_kwargs.get("screen_height", 500)

        env = EBHabEnv(
            eval_set=eval_set,
            data_dir=data_dir,
            dataset_dir=dataset_dir,
            resolution=resolution,
        )

        # Extract dynamic action space
        self._language_skill_set = env.language_skill_set
        self._action_map = {name: i for i, name in enumerate(self._language_skill_set)}

        return env

    def _on_reset(self, env, reset_config):
        return env.reset()

    def _on_step(self, env, action_text):
        # Map action text to integer ID
        action_id = self._action_map.get(action_text, -1)
        if action_id < 0:
            # Try case-insensitive match
            for name, idx in self._action_map.items():
                if name.lower() == action_text.lower():
                    action_id = idx
                    break
        if action_id < 0:
            # Random action as fallback (matches EBHabEnv behavior)
            action_id = np.random.randint(len(self._language_skill_set))

        return env.step(action_id)

    def _extract_image(self, obs):
        from easi.tasks.ebhabitat.vendor.utils import observations_to_image
        return observations_to_image(obs, "head_rgb")

    def _extract_info(self, info):
        return {
            "task_success": float(info.get("task_success", 0.0)),
            "task_progress": float(info.get("task_progress", 0.0)),
            "subgoal_reward": float(info.get("subgoal_reward", 0.0)),
            "last_action_success": float(info.get("last_action_success", 0.0)),
            "feedback": str(info.get("env_feedback", "")),
            "action_id": int(info.get("action_id", -1)),
        }

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to include dynamic action space in metadata."""
        response = super()._make_response(obs, reward, done, info)
        if self._language_skill_set:
            response["observation"]["metadata"]["dynamic_action_space"] = json.dumps(
                self._language_skill_set
            )
        return response


if __name__ == "__main__":
    EBHabitatBridge.main()
