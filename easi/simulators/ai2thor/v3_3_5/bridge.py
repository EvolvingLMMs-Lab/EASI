"""Generic bridge for AI2-THOR v3.3.5 (arm-mode).

Runs inside the easi_ai2thor_v3_3_5 conda env (Python 3.8).
Configures the controller with agentMode='arm' and FifoServer.
Task-specific bridges should subclass BaseBridge in the task layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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

# Exact ai2thor commit used by ManipulaTHOR — tells the Controller which
# build to download instead of trying `git log` in site-packages.
COMMIT_ID = "a84dd29471ec2201f583de00257d84fac1a03de2"

# Default controller kwargs for arm-mode AI2-THOR
DEFAULT_CONTROLLER_KWARGS = {
    "gridSize": 0.25,
    "width": 224,
    "height": 224,
    "visibilityDistance": 1.0,
    "agentMode": "arm",
    "fieldOfView": 100,
    "agentControllerType": "mid-level",
    "useMassThreshold": True,
    "massThreshold": 10,
    "autoSimulation": False,
    "autoSyncTransforms": True,
}


class AI2ThorV335Bridge:
    """Generic bridge for AI2-THOR v3.3.5 arm-mode."""

    def __init__(self, workspace, data_dir=None, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.data_dir = Path(data_dir) if data_dir else None
        self.simulator_kwargs = simulator_kwargs or {}
        self.controller = None
        self.step_count = 0
        self.episode_output_dir = None

    def start(self):
        """Initialize the AI2-THOR controller."""
        import ai2thor.controller
        import ai2thor.fifo_server

        kwargs = dict(DEFAULT_CONTROLLER_KWARGS)
        # Allow simulator_kwargs to override defaults
        for k in (
            "width", "height", "gridSize", "fieldOfView",
            "visibilityDistance", "agentMode", "agentControllerType",
            "renderDepthImage",
        ):
            if k in self.simulator_kwargs:
                kwargs[k] = self.simulator_kwargs[k]

        kwargs["server_class"] = ai2thor.fifo_server.FifoServer
        kwargs["commit_id"] = COMMIT_ID
        self.controller = ai2thor.controller.Controller(**kwargs)
        logger.info("AI2-THOR v3.3.5 controller started (arm mode)")

    def stop(self):
        """Shut down the controller."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def reset(self, reset_config):
        """Reset to a scene. Returns IPC response dict."""
        scene = reset_config.get("scene", "FloorPlan1")
        self.step_count = 0

        self.controller.reset(scene)
        # Standard arm-mode setup commands
        self.controller.step(action="MakeAllObjectsMoveable")
        self.controller.step(
            action="MakeObjectsStaticKinematicMassThreshold"
        )

        self.episode_output_dir = reset_config.get("episode_output_dir")
        return self._make_observation_response(
            self.controller.last_event,
        )

    def step(self, action_name):
        """Execute an action. Returns IPC response dict."""
        self.step_count += 1

        success = False
        done = False

        if action_name in ("done", "DoneMidLevel"):
            done = True
            success = True
        else:
            event = self.controller.step(action=action_name)
            success = event.metadata["lastActionSuccess"]

        info = {
            "action_name": action_name,
            "action_success": success,
            "feedback": "success" if success else "action failed",
        }

        return self._make_observation_response(
            self.controller.last_event,
            reward=0.0,
            done=done,
            info=info,
        )

    def _make_observation_response(
        self, event, reward=0.0, done=False, info=None,
    ):
        """Save RGB frame and build IPC response dict."""
        from PIL import Image

        save_dir = (
            Path(self.episode_output_dir)
            if self.episode_output_dir
            else self.workspace
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        rgb = (
            event.frame.copy()
            if event.frame is not None
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        rgb_path = save_dir / f"step_{self.step_count:04d}.png"
        Image.fromarray(rgb).save(str(rgb_path))

        agent = event.metadata.get("agent", {})
        pos = agent.get("position", {})
        rot = agent.get("rotation", {})
        pose = [
            pos.get("x", 0),
            pos.get("y", 0),
            pos.get("z", 0),
            rot.get("y", 0),
            agent.get("cameraHorizon", 0),
            0,
        ]

        return make_observation_response(
            rgb_path=str(rgb_path),
            agent_pose=pose,
            metadata={"step": str(self.step_count)},
            reward=reward,
            done=done,
            info=info,
        )

    def run(self):
        """Main IPC loop."""
        self.start()
        write_status(self.workspace, ready=True)

        while True:
            command = poll_for_command(self.workspace, timeout=300.0)
            cmd_type = command.get("type")

            if cmd_type == "reset":
                reset_config = command.get("config", {})
                self.episode_output_dir = reset_config.get(
                    "episode_output_dir"
                )

                try:
                    response = self.reset(reset_config)
                    write_response(self.workspace, response)
                except Exception as e:
                    logger.exception("Reset failed")
                    write_response(
                        self.workspace, make_error_response(str(e))
                    )

            elif cmd_type == "step":
                action = parse_action_from_command(command)
                logger.trace(
                    "Step %d: action=%s",
                    self.step_count + 1,
                    action.action_name,
                )

                try:
                    response = self.step(action.action_name)
                    write_response(self.workspace, response)
                except Exception as e:
                    logger.exception("Step failed")
                    write_response(
                        self.workspace, make_error_response(str(e))
                    )

            elif cmd_type == "close":
                logger.info("Close command received")
                self.stop()
                write_response(self.workspace, {"status": "ok"})
                break

            else:
                write_response(
                    self.workspace,
                    make_error_response("Unknown command: %s" % cmd_type),
                )


def main():
    parser = argparse.ArgumentParser(
        description="AI2-THOR v3.3.5 generic bridge (arm mode)"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--simulator-kwargs", type=str, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")
    sim_kwargs = (
        json.loads(args.simulator_kwargs)
        if args.simulator_kwargs
        else {}
    )
    bridge = AI2ThorV335Bridge(
        workspace=args.workspace,
        data_dir=args.data_dir,
        simulator_kwargs=sim_kwargs,
    )
    bridge.run()


if __name__ == "__main__":
    main()
