"""VLN-CE R2R bridge — wraps vendored SceneSimulator via BaseBridge.

This script runs inside the easi_habitat_sim_v0_1_7 conda env (Python 3.8).
Creates a NEW SceneSimulator per episode (Habitat-Sim ties one instance to one scene).

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--data-dir /path/to/data] [--simulator-kwargs '{}']
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class VLNCEBridge(BaseBridge):
    """BaseBridge wrapper for VLN-CE R2R SceneSimulator."""

    _scene_sim = None

    def _create_env(self, reset_config, simulator_kwargs):
        """Store simulator_kwargs; real sim created in _on_reset."""
        self._sim_kwargs = simulator_kwargs
        return object()  # placeholder

    def _on_reset(self, env, reset_config):
        """Create a new SceneSimulator for this episode's scene."""
        from easi.tasks.vlnce_r2r.vendor.scene_simulator import SceneSimulator

        # Close previous simulator if any
        if self._scene_sim is not None:
            self._scene_sim.close()
            self._scene_sim = None

        data_dir = reset_config.get("data_dir") or self._sim_kwargs.get("data_dir", "")
        scene_id = reset_config["scene_id"]
        scene_path = str(Path(data_dir) / "mp3d" / scene_id / f"{scene_id}.glb")

        # Parse position/rotation (may be JSON strings from IPC)
        start_position = json.loads(reset_config["start_position"]) \
            if isinstance(reset_config["start_position"], str) else reset_config["start_position"]
        start_rotation = json.loads(reset_config["start_rotation"]) \
            if isinstance(reset_config["start_rotation"], str) else reset_config["start_rotation"]

        goal_position = reset_config.get("goal_position")
        if isinstance(goal_position, str):
            goal_position = json.loads(goal_position) if goal_position != "null" else None

        geodesic_distance = reset_config.get("geodesic_distance")
        if isinstance(geodesic_distance, str):
            geodesic_distance = float(geodesic_distance) if geodesic_distance != "null" else None

        gt_locations = reset_config.get("gt_locations")
        if isinstance(gt_locations, str):
            gt_locations = json.loads(gt_locations) if gt_locations != "null" else None

        # Log test split warning
        if goal_position is None:
            logger.warning(
                "Episode %s has no goal_position (test split). "
                "Only path_length and steps_taken will be computed.",
                reset_config.get("episode_id", "unknown"),
            )

        self._scene_sim = SceneSimulator(
            scene_path=scene_path,
            start_position=start_position,
            start_rotation=start_rotation,
            goal_position=goal_position,
            gt_locations=gt_locations,
            geodesic_distance=geodesic_distance,
            success_distance=self._sim_kwargs.get("success_distance", 3.0),
            max_steps=self._sim_kwargs.get("max_steps", 500),
            gpu_device_id=self._sim_kwargs.get("gpu_device_id", -1),
            width=self._sim_kwargs.get("screen_width", 480),
            height=self._sim_kwargs.get("screen_height", 480),
            hfov=self._sim_kwargs.get("hfov", 90),
            sensor_height=self._sim_kwargs.get("sensor_height", 1.25),
            forward_step_size=self._sim_kwargs.get("forward_step_size", 0.25),
            turn_angle=self._sim_kwargs.get("turn_angle", 15.0),
            allow_sliding=self._sim_kwargs.get("allow_sliding", True),
        )

        return self._scene_sim.get_observation()

    def _on_step(self, env, action_text):
        """Step the SceneSimulator and return (obs, reward, done, info)."""
        obs, done, info = self._scene_sim.step(action_text)
        return obs, 0.0, done, info

    def _extract_image(self, obs):
        """Extract front RGB from observation (RGBA -> RGB)."""
        return obs["color_sensor"][:, :, :3]

    def _extract_info(self, info):
        """Filter info to serializable types (int/float/str/bool/None)."""
        clean = {}
        for k, v in info.items():
            if v is None:
                clean[k] = "null"  # IPC encodes None as "null" string
            elif isinstance(v, (int, float, str, bool)):
                clean[k] = v
            elif isinstance(v, (list, np.ndarray)):
                clean[k] = json.dumps([float(x) for x in v] if hasattr(v, '__iter__') else str(v))
        return clean

    def _get_topdown_map(self):
        """Render habitat-sim navmesh as top-down RGB map."""
        if self._scene_sim is None:
            return None
        sim = self._scene_sim._sim
        if not sim.pathfinder.is_loaded:
            return None

        meters_per_pixel = 0.1
        height = float(sim.get_agent(0).get_state().position[1])
        try:
            # habitat-sim 0.1.7 API: get_topdown_view(meters_per_pixel, height)
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
            "height": float(sim.get_agent(0).get_state().position[1]),
        }

    def _get_episode_meta(self):
        """Persist start position, goal, and ground truth path."""
        if self._scene_sim is None:
            return None
        meta = {}
        agent_state = self._scene_sim._sim.get_agent(0).get_state()
        meta["start_position"] = [float(x) for x in agent_state.position]
        if self._scene_sim._gt_locations:
            meta["gt_locations"] = self._scene_sim._gt_locations
        if self._scene_sim._goal_position is not None:
            meta["goal_position"] = [float(x) for x in self._scene_sim._goal_position]
        return meta

    def close(self):
        """Close the SceneSimulator."""
        if self._scene_sim is not None:
            self._scene_sim.close()
            self._scene_sim = None
        super().close()


if __name__ == "__main__":
    VLNCEBridge.main()
