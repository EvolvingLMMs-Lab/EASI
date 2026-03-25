"""Generic bridge subprocess for AI2-THOR v2.1.0.

This script runs inside the easi_ai2thor_v2_1_0 conda env (Python 3.8).
It communicates with the parent process via filesystem IPC.

Provides a generic AI2ThorBridge that handles:
- Controller startup/shutdown
- Scene reset with configurable parameters
- Raw THOR action execution
- Observation capture (RGB frames)
- Main IPC loop

Task-specific bridges (e.g., EBAlfredBridge) subclass this and override
reset() and step() for skill-based execution and goal evaluation.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--data-dir /path/to/datasets] [--simulator-kwargs '{}']
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# numpy and scipy are available in the ai2thor conda env (Python 3.8)
# but may not be in the host venv — imports are optional for importability
try:
    import numpy as np
    from scipy import spatial
except ImportError:
    np = None  # type: ignore[assignment]
    spatial = None  # type: ignore[assignment]

# Add repo root to path for easi imports
_repo_root = Path(__file__).resolve().parents[4]
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
from easi.simulators.ai2thor.v2_1_0.thor_utils import (
    AGENT_STEP_SIZE,
    CAMERA_HEIGHT_OFFSET,
    RECORD_SMOOTHING_FACTOR,
    RENDER_CLASS_IMAGE,
    RENDER_DEPTH_IMAGE,
    RENDER_IMAGE,
    RENDER_OBJECT_IMAGE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    VISIBILITY_DISTANCE,
)
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class AI2ThorBridge:
    """Generic AI2-THOR bridge managing controller lifecycle and raw actions.

    Subclass this for task-specific behavior (skill execution, goal evaluation).
    """

    def __init__(self, workspace, data_dir=None, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.data_dir = Path(data_dir) if data_dir else None
        self.simulator_kwargs = simulator_kwargs or {}
        self.controller = None
        self.last_event = None
        self.step_count = 0

        # Navigation cache
        self.reachable_positions = None
        self.reachable_position_kdtree = None

        # Agent height (set after reset from init_action)
        self.agent_height = 0.9009992

        # Output directory for saving images (set per-episode from reset command)
        self.episode_output_dir = None

    def start(self):
        """Initialize AI2-THOR controller with configurable parameters."""
        from ai2thor.controller import Controller

        quality = self.simulator_kwargs.get("quality", "MediumCloseFitShadows")
        screen_h = self.simulator_kwargs.get("screen_height", SCREEN_HEIGHT)
        screen_w = self.simulator_kwargs.get("screen_width", SCREEN_WIDTH)

        # ai2thor Controller.start() prepends ':' to x_display, so we must
        # strip any leading ':' from DISPLAY to avoid "::99" double-colon bug.
        x_display = os.environ.get("DISPLAY", ":0").lstrip(":")

        logger.info("Starting AI2-THOR controller (display=%s)...", x_display)
        self.controller = Controller(quality=quality)
        self.controller.start(
            x_display=x_display,
            player_screen_height=screen_h,
            player_screen_width=screen_w,
        )
        logger.info("AI2-THOR controller started.")

    def stop(self):
        """Stop the AI2-THOR controller."""
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
            self.controller = None

    def _step(self, action_dict):
        """Execute a raw THOR action and update last_event."""
        self.last_event = self.controller.step(action_dict)
        return self.last_event

    # --- Reset ---

    def reset(self, reset_config):
        """Reset to a default scene for smoke tests.

        Override in subclasses for task-specific resets (episode loading, etc.).
        """
        self.step_count = 0
        scene = reset_config.get("scene", "FloorPlan10")

        logger.info("Resetting to scene: %s", scene)
        self.controller.reset(scene)
        self.last_event = self.controller.step(dict(
            action="Initialize",
            gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
            cameraY=CAMERA_HEIGHT_OFFSET,
            renderImage=RENDER_IMAGE,
            renderDepthImage=RENDER_DEPTH_IMAGE,
            renderClassImage=RENDER_CLASS_IMAGE,
            renderObjectImage=RENDER_OBJECT_IMAGE,
            visibility_distance=VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        self.agent_height = self.last_event.metadata["agent"]["position"]["y"]
        self._cache_reachable_positions()
        return self._make_observation_response()

    # --- Step ---

    def step(self, action_text):
        """Execute a raw THOR action and return observation.

        Override in subclasses for skill-based execution.
        """
        self.step_count += 1
        self._step(dict(action=action_text))
        return self._make_observation_response(
            reward=0.0,
            done=False,
            info={
                "last_action_success": 1.0 if self.last_event.metadata["lastActionSuccess"] else 0.0,
                "env_step": float(self.step_count),
                "feedback": "success" if self.last_event.metadata["lastActionSuccess"] else "action failed",
            },
        )

    # --- Main IPC Loop ---

    def run(self):
        """Main bridge loop — subclasses inherit automatically."""
        logger.info("AI2-THOR bridge starting (workspace: %s)", self.workspace)
        self.start()

        write_status(self.workspace, ready=True)

        while True:
            try:
                command = poll_for_command(self.workspace, timeout=300.0)
            except Exception as e:
                logger.error("Failed to read command: %s", e)
                break

            cmd_type = command.get("type")

            if cmd_type == "reset":
                episode_id = command.get("episode_id", "unknown")
                reset_config = command.get("reset_config", {})
                logger.info("Reset: episode_id=%s", episode_id)

                # Set episode output directory for image saving
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
                action_data = command.get("action", {})
                action_text = action_data.get("action_name", "")
                logger.trace("Step %d: action=%s", self.step_count + 1, action_text)

                try:
                    response = self.step(action_text)
                    write_response(self.workspace, response)
                except Exception as e:
                    logger.exception("Step failed")
                    write_response(self.workspace, make_error_response(str(e)))

            elif cmd_type == "close":
                logger.info("Close command received")
                self.stop()
                write_response(self.workspace, {"status": "ok"})
                break

            else:
                write_response(
                    self.workspace, make_error_response("Unknown command: %s" % cmd_type)
                )

    # --- Navigation Helpers ---

    def _cache_reachable_positions(self):
        """Cache reachable positions + KD-tree for navigation."""
        event = self._step(dict(action="GetReachablePositions"))
        free_positions = event.metadata["actionReturn"]
        self.reachable_positions = np.array(
            [[p["x"], p["y"], p["z"]] for p in free_positions]
        )
        self.reachable_position_kdtree = spatial.KDTree(self.reachable_positions)

    def _find_close_reachable_position(self, loc, nth=1):
        """Find the nth closest reachable position to a location."""
        if self.reachable_position_kdtree is None:
            return None
        n_positions = len(self.reachable_positions)
        k = min(nth + 1, n_positions)
        if k == 0:
            return None
        d, idx = self.reachable_position_kdtree.query(loc, k=k)
        selected = min(nth - 1, k - 1)
        return self.reachable_positions[idx[selected]] if k > 1 else self.reachable_positions[idx]

    @staticmethod
    def _angle_diff(x, y):
        """Calculate angle difference in degrees."""
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))

    # --- Observation ---

    def _make_observation_response(self, reward=0.0, done=False, info=None):
        """Save RGB frame and return IPC response."""
        from PIL import Image

        event = self.last_event

        # Save to episode_output_dir if set, else IPC workspace
        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        rgb_path = save_dir / ("rgb_%04d.png" % self.step_count)

        # Save frame as PNG
        Image.fromarray(event.frame).save(str(rgb_path))

        agent = event.metadata["agent"]
        pose = [
            agent["position"]["x"],
            agent["position"]["y"],
            agent["position"]["z"],
            agent["rotation"]["y"],
            agent.get("cameraHorizon", 0),
            0,
        ]

        return make_observation_response(
            rgb_path=str(rgb_path),
            agent_pose=pose,
            metadata={"step": str(self.step_count)},
            reward=reward,
            done=done,
            info=info or {},
        )


def main():
    parser = argparse.ArgumentParser(description="AI2-THOR v2.1.0 generic bridge")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--simulator-kwargs", type=str, default=None,
                        help="JSON string of simulator configuration")
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")

    simulator_kwargs = json.loads(args.simulator_kwargs) if args.simulator_kwargs else {}
    bridge = AI2ThorBridge(
        workspace=args.workspace,
        data_dir=args.data_dir,
        simulator_kwargs=simulator_kwargs,
    )
    bridge.run()


if __name__ == "__main__":
    main()
