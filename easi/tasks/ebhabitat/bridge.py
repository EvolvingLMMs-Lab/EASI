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
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBHabitatBridge(BaseBridge):
    """BaseBridge wrapper around vendored EBHabEnv."""

    _action_map: dict[str, int] = {}
    _language_skill_set: list[str] = []

    def _create_env(self, reset_config, simulator_kwargs):
        from easi.tasks.ebhabitat.vendor.EBHabEnv import EBHabEnv

        # Resolve data directories from HF dataset cache (passed via reset_config)
        # or from --data-dir CLI arg (passed via simulator_kwargs)
        data_dir = reset_config.get("data_dir") or simulator_kwargs.get("data_dir")
        dataset_dir = None
        if data_dir:
            # HF dataset structure: data_dir/datasets/*.pickle, data_dir/data/...
            candidate = Path(data_dir) / "datasets"
            if candidate.exists():
                dataset_dir = str(candidate)

        eval_set = reset_config.get("eval_set", "base")
        resolution = simulator_kwargs.get("screen_height", 500)
        max_steps = simulator_kwargs.get("max_steps", 30)
        max_invalid_actions = simulator_kwargs.get("max_invalid_actions", 10)
        feedback_verbosity = simulator_kwargs.get("feedback_verbosity", 1)
        device = simulator_kwargs.get("device", "cpu")
        if device == "gpu":
            gpu_device_id = simulator_kwargs.get("_assigned_gpu_id", 0)
        else:
            gpu_device_id = None

        env = EBHabEnv(
            eval_set=eval_set,
            data_dir=data_dir,
            dataset_dir=dataset_dir,
            resolution=resolution,
            max_steps=max_steps,
            max_invalid_actions=max_invalid_actions,
            feedback_verbosity=feedback_verbosity,
            gpu_device_id=gpu_device_id,
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

    def _get_sim(self):
        """Access the underlying habitat-sim Simulator instance.

        Chain: bridge.env (EBHabEnv) -> .env (gym wrapper) -> .env -> .env
               -> ._env (HabitatEnv) -> .task._sim
        """
        try:
            return self.env.env.env.env._env.task._sim
        except AttributeError:
            return None

    def _extract_info(self, info):
        clean = {
            "task_success": float(info.get("task_success", 0.0)),
            "task_progress": float(info.get("task_progress", 0.0)),
            "subgoal_reward": float(info.get("subgoal_reward", 0.0)),
            "last_action_success": float(info.get("last_action_success", 0.0)),
            "feedback": str(info.get("env_feedback", "")),
            "action_id": int(info.get("action_id", -1)),
        }
        # Add agent position for trajectory visualization
        sim = self._get_sim()
        if sim is not None:
            try:
                agent = sim.get_agent_data(0).articulated_agent
                pos = agent.base_pos
                clean["agent_position"] = json.dumps([float(pos[0]), float(pos[1]), float(pos[2])])
            except Exception:
                pass
        return clean

    def _get_topdown_map(self):
        """Render habitat-sim navmesh as top-down RGB map."""
        sim = self._get_sim()
        if sim is None or not sim.pathfinder.is_loaded:
            return None

        meters_per_pixel = 0.1
        try:
            agent = sim.get_agent_data(0).articulated_agent
            height = float(agent.base_pos[1])
            topdown = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        except Exception:
            logger.warning("Failed to get topdown map from pathfinder")
            return None

        # Colorize: navigable=light gray, obstacle=dark gray
        rgb = np.full((*topdown.shape, 3), 60, dtype=np.uint8)
        rgb[topdown == 1] = [240, 240, 240]

        bounds = sim.pathfinder.get_bounds()
        return rgb, {
            "bounds_lower": [float(x) for x in bounds[0]],
            "bounds_upper": [float(x) for x in bounds[1]],
            "meters_per_pixel": meters_per_pixel,
            "height": height,
        }

    def _get_episode_meta(self):
        """Persist start position for trajectory visualization."""
        sim = self._get_sim()
        if sim is None:
            return None
        try:
            agent = sim.get_agent_data(0).articulated_agent
            pos = agent.base_pos
            return {"start_position": [float(pos[0]), float(pos[1]), float(pos[2])]}
        except Exception:
            return None

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
