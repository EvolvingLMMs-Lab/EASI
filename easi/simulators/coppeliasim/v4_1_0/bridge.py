"""Smoke-test bridge for CoppeliaSim V4.1.0.

This script runs inside the easi_coppeliasim_v4_1_0 conda environment.
It verifies PyRep + CoppeliaSim work by:
1. Launching PyRep with the task_design.ttt scene
2. Loading a Panda robot arm
3. Handling reset (return camera image) and step (advance sim) commands
4. Reporting success via IPC

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
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

# Default scene file (shipped with this simulator)
DEFAULT_TTT_FILE = str(Path(__file__).parent / "task_design.ttt")


class CoppeliaSimBridge:
    """Smoke test bridge for CoppeliaSim V4.1.0 + PyRep.

    Loads PyRep, creates a Panda arm, captures images from the front camera.
    Supports reset (arm to home pose) and step (advance simulation).
    """

    def __init__(self, workspace, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.simulator_kwargs = simulator_kwargs or {}
        self.pyrep = None
        self.robot_arm = None
        self.gripper = None
        self.camera = None
        self.step_count = 0
        self.episode_output_dir = None

    def start(self):
        """Initialize PyRep with CoppeliaSim and load Panda robot."""
        from pyrep import PyRep
        from pyrep.robots.arms.panda import Panda
        from pyrep.robots.end_effectors.panda_gripper import PandaGripper

        ttt_file = self.simulator_kwargs.get("ttt_file", DEFAULT_TTT_FILE)
        headless = self.simulator_kwargs.get("headless", True)

        logger.info("Launching PyRep with scene: %s (headless=%s)", ttt_file, headless)
        self.pyrep = PyRep()
        self.pyrep.launch(ttt_file, headless=headless)
        self.pyrep.start()

        # Load robot
        self.robot_arm = Panda()
        self.gripper = PandaGripper()

        # Get the front camera (defined in task_design.ttt)
        from pyrep.objects.vision_sensor import VisionSensor
        try:
            self.camera = VisionSensor("cam_front")
        except Exception:
            logger.warning("cam_front not found, will generate blank images")
            self.camera = None

        logger.info("CoppeliaSim bridge ready (Panda arm loaded)")

    def stop(self):
        """Shut down PyRep and CoppeliaSim."""
        if self.pyrep is not None:
            try:
                self.pyrep.stop()
                self.pyrep.shutdown()
            except Exception:
                pass
            self.pyrep = None

    def reset(self, reset_config):
        """Reset robot to home position, return camera image."""
        self.step_count = 0
        num_joints = len(self.robot_arm.get_joint_positions())
        self.robot_arm.set_joint_positions([0.0] * num_joints)
        self.pyrep.step()
        return self._make_observation_response()

    def step(self, action_name):
        """Execute one step. For smoke test, just advance simulation."""
        self.step_count += 1

        if action_name == "Stop":
            return self._make_observation_response(done=True)

        self.pyrep.step()

        return self._make_observation_response(
            info={
                "env_step": float(self.step_count),
                "last_action_success": 1.0,
            }
        )

    def _capture_image(self):
        """Capture RGB image from front camera (or generate placeholder)."""
        if self.camera is not None:
            try:
                image = self.camera.capture_rgb()
                return (image * 255).astype(np.uint8)
            except Exception as e:
                logger.warning("Camera capture failed: %s", e)
        return np.full((64, 64, 3), 128, dtype=np.uint8)

    def _make_observation_response(self, reward=0.0, done=False, info=None):
        """Capture image, build IPC response."""
        from PIL import Image

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / ("rgb_%04d.png" % self.step_count)

        image = self._capture_image()
        Image.fromarray(image).save(str(rgb_path))

        try:
            ee_pos = self.robot_arm.get_tip().get_position()
            ee_ori = self.robot_arm.get_tip().get_orientation()
            pose = list(ee_pos) + list(ee_ori)
        except Exception:
            pose = [0.0] * 6

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
        logger.info("CoppeliaSim bridge starting (workspace: %s)", self.workspace)
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
                reset_config = command.get("reset_config", {})
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
                write_response(
                    self.workspace,
                    make_error_response("Unknown command: %s" % cmd_type),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="CoppeliaSim V4.1.0 bridge")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--simulator-kwargs", type=str, default=None)
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")

    sim_kwargs = json.loads(args.simulator_kwargs) if args.simulator_kwargs else {}
    bridge = CoppeliaSimBridge(workspace=args.workspace, simulator_kwargs=sim_kwargs)
    bridge.run()


if __name__ == "__main__":
    main()
