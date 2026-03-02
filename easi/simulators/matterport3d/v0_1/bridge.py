"""Matterport3D bridge for EASI's filesystem IPC protocol.

Runs inside the easi_matterport3d_v0_1 Docker container.
Wraps MatterSim in EASI's BaseBridge IPC protocol.

PYTHONPATH is set by the Dockerfile and docker run env vars to include
both /opt/MatterSim/build (for MatterSim) and /opt/easi (for easi.*).
No manual sys.path manipulation needed here — BaseBridge also handles
repo-root discovery via its own module path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easi.simulators.base_bridge import BaseBridge
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class Matterport3DBridge(BaseBridge):
    """Bridge wrapping MatterSim.Simulator for EASI IPC."""

    def _create_env(self, reset_config, simulator_kwargs):
        import MatterSim

        sim = MatterSim.Simulator()
        sim.setDatasetPath(simulator_kwargs.get("data_dir", "/data/v1/scans"))
        sim.setNavGraphPath(
            simulator_kwargs.get("nav_graph_path", "/opt/MatterSim/connectivity")
        )
        width = simulator_kwargs.get("width", 640)
        height = simulator_kwargs.get("height", 480)
        sim.setCameraResolution(width, height)
        sim.setCameraVFOV(simulator_kwargs.get("vfov", 0.8))
        sim.setDepthEnabled(simulator_kwargs.get("depth_enabled", True))
        sim.setDiscretizedViewingAngles(
            simulator_kwargs.get("discretized_views", True)
        )
        sim.setRenderingEnabled(True)
        sim.setBatchSize(1)
        sim.initialize()

        self._sim = sim
        self._width = width
        self._height = height
        return sim

    def _on_reset(self, env, reset_config):
        scan_id = reset_config.get("scan_id", "")
        viewpoint_id = reset_config.get("viewpoint_id", "")
        heading = float(reset_config.get("heading", 0.0))
        elevation = float(reset_config.get("elevation", 0.0))

        env.newEpisode([scan_id], [viewpoint_id], [heading], [elevation])
        state = env.getState()[0]
        return self._state_to_obs(state)

    def _on_step(self, env, action_text):
        """Parse action and execute navigation.

        Action format: JSON string with keys:
          - location_index: int (index into navigableLocations)
          - heading: float (heading change in radians)
          - elevation: float (elevation change in radians)
        """
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            action = {"location_index": 0, "heading": 0.0, "elevation": 0.0}

        env.makeAction(
            [int(action.get("location_index", 0))],
            [float(action.get("heading", 0.0))],
            [float(action.get("elevation", 0.0))],
        )
        state = env.getState()[0]
        obs = self._state_to_obs(state)

        done = bool(action.get("done", False))
        info = {
            "viewpoint_id": state.location.viewpointId,
            "heading": state.heading,
            "elevation": state.elevation,
            "step": state.step,
        }
        return obs, 0.0, done, info

    def _state_to_obs(self, state):
        """Convert MatterSim.SimState to observation dict."""
        obs = {"frame": state.rgb}
        if hasattr(state, "depth") and state.depth is not None:
            obs["depth_frame"] = state.depth
        obs["navigable_locations"] = [
            {
                "viewpointId": loc.viewpointId,
                "rel_heading": loc.rel_heading,
                "rel_elevation": loc.rel_elevation,
                "rel_distance": loc.rel_distance,
            }
            for loc in state.navigableLocations
        ]
        return obs

    def _extract_image(self, obs):
        return obs["frame"]


if __name__ == "__main__":
    Matterport3DBridge.main()
