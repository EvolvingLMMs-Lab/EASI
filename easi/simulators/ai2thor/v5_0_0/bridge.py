"""Bridge subprocess for AI2-THOR v5.0.0 (CloudRendering API).

This script runs inside the easi_ai2thor_v5_0_0 conda environment.
It starts a real AI2-THOR controller using the CloudRendering platform
and handles reset/step/close commands via filesystem IPC.

Task-specific bridges (e.g., EBNavigationBridge) use the vendored env
directly and extend BaseBridge. This generic bridge is used by
`easi sim test ai2thor:v5_0_0` for smoke testing the simulator install.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    parse_action_from_command,
)
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Default v5 configuration
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500
DEFAULT_FOV = 100
DEFAULT_GRID_SIZE = 0.25
DEFAULT_VISIBILITY_DISTANCE = 10
DEFAULT_SCENE = "FloorPlan1"


class AI2ThorV5Bridge:
    """Generic AI2-THOR v5.0.0 bridge for smoke testing.

    Uses Linux64 platform (headless GPU rendering).
    """

    def __init__(self, workspace, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.simulator_kwargs = simulator_kwargs or {}
        self.controller = None
        self.step_count = 0
        self.episode_output_dir = None

    def start(self):
        """Initialize AI2-THOR v5 controller with Linux64."""
        from ai2thor.controller import Controller
        from ai2thor.platform import Linux64

        width = self.simulator_kwargs.get("screen_width", DEFAULT_WIDTH)
        height = self.simulator_kwargs.get("screen_height", DEFAULT_HEIGHT)
        fov = self.simulator_kwargs.get("fov", DEFAULT_FOV)
        grid_size = self.simulator_kwargs.get("grid_size", DEFAULT_GRID_SIZE)

        logger.info(
            "Starting AI2-THOR v5.0.0 controller (Linux64, %dx%d)...",
            width, height,
        )
        self.controller = Controller(
            agentMode="default",
            gridSize=grid_size,
            visibilityDistance=DEFAULT_VISIBILITY_DISTANCE,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=width,
            height=height,
            fieldOfView=fov,
            platform=Linux64,
        )
        logger.info("AI2-THOR v5.0.0 controller started.")

    def stop(self):
        """Stop the AI2-THOR controller."""
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
            self.controller = None

    def reset(self, reset_config):
        """Reset to a scene for smoke testing."""
        self.step_count = 0
        scene = reset_config.get("scene", DEFAULT_SCENE)

        logger.info("Resetting to scene: %s", scene)
        event = self.controller.reset(scene=scene)

        return self._make_observation_response(event)

    def step(self, action_name):
        """Execute one action and return observation."""
        self.step_count += 1

        # Map common action names to THOR API calls
        event = self._dispatch_action(action_name)

        success = event.metadata.get("lastActionSuccess", False)
        error = event.metadata.get("errorMessage", "")
        done = action_name == "Stop"

        info = {
            "last_action_success": 1.0 if success else 0.0,
            "env_step": float(self.step_count),
        }
        if not success and error:
            info["error"] = error

        return self._make_observation_response(
            event, reward=0.0, done=done, info=info,
        )

    def _dispatch_action(self, action_name):
        """Execute a THOR action, handling common navigation actions."""
        # Standard navigation actions with default magnitudes
        action_params = {}

        if action_name in ("MoveAhead", "MoveBack", "MoveRight", "MoveLeft"):
            action_params = {"moveMagnitude": DEFAULT_GRID_SIZE}
        elif action_name in ("RotateRight", "RotateLeft"):
            action_params = {"degrees": 90}
        elif action_name in ("LookUp", "LookDown"):
            action_params = {"degrees": 30}

        return self.controller.step(action=action_name, **action_params)

    def _make_observation_response(self, event, reward=0.0, done=False, info=None):
        """Save RGB frame and return IPC response."""
        from PIL import Image

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / ("rgb_%04d.png" % self.step_count)

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

    def run(self):
        """Main IPC loop."""
        logger.info("AI2-THOR v5.0.0 bridge starting (workspace: %s)", self.workspace)
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
                action = parse_action_from_command(command)
                logger.trace("Step %d: action=%s", self.step_count + 1, action.action_name)

                try:
                    response = self.step(action.action_name)
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
                write_response(self.workspace, make_error_response("Unknown command: %s" % cmd_type))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI2-THOR v5.0.0 bridge")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--simulator-kwargs", type=str, default=None,
                        help="JSON string of simulator configuration")
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")

    sim_kwargs = json.loads(args.simulator_kwargs) if args.simulator_kwargs else {}
    bridge = AI2ThorV5Bridge(workspace=args.workspace, simulator_kwargs=sim_kwargs)
    bridge.run()


if __name__ == "__main__":
    main()
