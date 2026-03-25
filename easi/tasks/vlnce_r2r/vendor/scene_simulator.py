"""Habitat-Sim 0.1.7 scene simulator for VLN-CE R2R.

Wraps habitat_sim.Simulator for single-goal navigation with path tracking.
Creates one simulator per episode (Habitat-Sim binds one instance to one scene).

Adapted from LHPR-VLN's SceneSimulator, simplified for single-goal VLN-CE:
- No subtask tracking (single goal)
- Specific start position/rotation (not random)
- 15° turns, single front RGB camera
- Path tracking for DTW metrics
- Null-safe for test split (no goal_position)
"""
from __future__ import annotations

import numpy as np

import habitat_sim

from easi.tasks.vlnce_r2r.vendor.scene_config import make_cfg
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class SceneSimulator:
    """Single-goal navigation simulator for VLN-CE R2R."""

    def __init__(
        self,
        *,
        scene_path: str,
        start_position: list[float],
        start_rotation: list[float],
        goal_position: list[float] | None = None,
        gt_locations: list[list[float]] | None = None,
        geodesic_distance: float | None = None,
        success_distance: float = 3.0,
        max_steps: int = 500,
        gpu_device_id: int = -1,
        width: int = 480,
        height: int = 480,
        hfov: int = 90,
        sensor_height: float = 1.25,
        forward_step_size: float = 0.25,
        turn_angle: float = 15.0,
        allow_sliding: bool = True,
    ):
        self._goal_position = np.array(goal_position) if goal_position else None
        self._gt_locations = gt_locations
        self._geodesic_distance = geodesic_distance
        self._success_distance = success_distance
        self._max_steps = max_steps

        # Create habitat-sim instance
        cfg = make_cfg(
            scene_path=scene_path,
            gpu_device_id=gpu_device_id,
            width=width,
            height=height,
            hfov=hfov,
            sensor_height=sensor_height,
            forward_step_size=forward_step_size,
            turn_angle=turn_angle,
            allow_sliding=allow_sliding,
        )
        self._sim = habitat_sim.Simulator(cfg)

        # Set agent to start position/rotation
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = np.array(start_position, dtype=np.float32)
        state.rotation = self._list_to_quaternion(start_rotation)
        agent.set_state(state)

        # Tracking state
        self._step_count = 0
        self._done = False
        self._stopped = False
        self._agent_positions = [list(state.position)]
        self._path_length = 0.0
        self._min_geodesic_to_goal = float("inf")

        # Compute initial geodesic distance
        if self._goal_position is not None:
            geo = self._compute_geodesic(state.position)
            if geo is not None:
                self._min_geodesic_to_goal = geo

    def _list_to_quaternion(self, rot):
        """Convert [x, y, z, w] list to habitat quaternion."""
        import quaternion
        if len(rot) == 4:
            return np.quaternion(rot[3], rot[0], rot[1], rot[2])
        return np.quaternion(1, 0, 0, 0)

    def get_observation(self):
        """Get current observation from the simulator."""
        return self._sim.get_sensor_observations()

    def step(self, action: str) -> tuple[dict, bool, dict]:
        """Execute an action and return (obs, done, info).

        Args:
            action: One of "move_forward", "turn_left", "turn_right", "stop".

        Returns:
            obs: Sensor observations dict with "color_sensor" key.
            done: Whether episode is over.
            info: Step info dict with metrics and feedback.
        """
        if self._done:
            return self.get_observation(), True, self._build_info()

        if action == "stop":
            self._stopped = True
            self._done = True
            obs = self.get_observation()
            return obs, True, self._build_info()

        # Execute movement action
        if action in ("move_forward", "turn_left", "turn_right"):
            obs = self._sim.step(action)
        else:
            logger.warning("Unknown action '%s', treating as no-op", action)
            obs = self.get_observation()

        self._step_count += 1

        # Update tracking
        pos = self._sim.get_agent(0).get_state().position
        prev_pos = np.array(self._agent_positions[-1])
        self._path_length += float(np.linalg.norm(pos - prev_pos))
        self._agent_positions.append(list(pos))

        # Update min geodesic (for oracle success)
        if self._goal_position is not None:
            geo = self._compute_geodesic(pos)
            if geo is not None and geo < self._min_geodesic_to_goal:
                self._min_geodesic_to_goal = geo

        # Check max steps
        if self._step_count >= self._max_steps:
            self._done = True

        return obs, self._done, self._build_info()

    def _compute_geodesic(self, position) -> float | None:
        """Compute geodesic distance from position to goal."""
        if self._goal_position is None:
            return None
        try:
            return self._sim.geodesic_distance(position, self._goal_position)
        except Exception:
            # Fallback to Euclidean if pathfinder fails
            return float(np.linalg.norm(position - self._goal_position))

    def _build_info(self) -> dict:
        """Build step info dict with metrics and feedback."""
        pos = self._sim.get_agent(0).get_state().position

        info = {
            "step": self._step_count,
            "path_length": self._path_length,
            "agent_position": list(pos),
        }

        if self._goal_position is not None:
            geo = self._compute_geodesic(pos)
            ne = geo if geo is not None else float(np.linalg.norm(pos - self._goal_position))
            info["navigation_error"] = ne
            info["geo_distance"] = ne  # For agent feedback

            success = 1.0 if ne <= self._success_distance and self._stopped else 0.0
            oracle_success = 1.0 if self._min_geodesic_to_goal <= self._success_distance else 0.0

            info["success"] = success
            info["oracle_success"] = oracle_success

            # SPL
            if self._geodesic_distance is not None and self._geodesic_distance > 0:
                info["spl"] = success * (
                    self._geodesic_distance / max(self._path_length, self._geodesic_distance)
                )
            else:
                info["spl"] = 0.0

            # Feedback string for agent
            info["feedback"] = f"Distance to goal: {ne:.1f}m"
        else:
            # Test split: no goal
            info["navigation_error"] = None
            info["success"] = None
            info["oracle_success"] = None
            info["spl"] = None
            info["feedback"] = "Navigating (no distance feedback available)"

        # Compute NDTW/SDTW on episode end
        if self._done:
            info.update(self._compute_dtw_metrics())

        return info

    def _compute_dtw_metrics(self) -> dict:
        """Compute NDTW and SDTW at episode end."""
        if self._gt_locations is None or len(self._gt_locations) == 0:
            return {"ndtw": None, "sdtw": None}

        try:
            from easi.tasks.vlnce_r2r.vendor.dtw import compute_ndtw, compute_sdtw

            ndtw = compute_ndtw(
                self._agent_positions, self._gt_locations, self._success_distance
            )
            # Compute success inline to avoid circular call to _build_info
            success = 0.0
            if self._goal_position is not None and self._stopped:
                pos = self._sim.get_agent(0).get_state().position
                geo = self._compute_geodesic(pos)
                ne = geo if geo is not None else float(np.linalg.norm(pos - self._goal_position))
                if ne <= self._success_distance:
                    success = 1.0
            sdtw = compute_sdtw(ndtw, success)
            return {"ndtw": float(ndtw), "sdtw": float(sdtw)}
        except ImportError:
            logger.warning("fastdtw not installed — NDTW/SDTW unavailable")
            return {"ndtw": None, "sdtw": None}
        except Exception as e:
            logger.warning("DTW computation failed: %s", e)
            return {"ndtw": None, "sdtw": None}

    def close(self):
        """Close the habitat-sim instance."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
