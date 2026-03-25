"""Bridge subprocess for TDW v1.11.23.

This script runs inside the easi_tdw_v1_11_23 conda env (Python 3.10).
It communicates with the parent process via filesystem IPC.

Provides a TDWBridge that handles:
- Controller startup/shutdown (auto-launches Unity build)
- Scene loading and image capture
- Main IPC loop (reset/step/close)

Task-specific bridges (e.g., HAZARD) will subclass this for
scenario-specific scene setup (LogPlayback, hazard controllers).

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Default TDW settings
DEFAULT_PORT = 1071
DEFAULT_SCREEN_SIZE = 512
DEFAULT_SCENE = "tdw_room"


class TDWBridge:
    """Generic TDW bridge managing Controller lifecycle and basic actions.

    Subclass this for task-specific behavior (HAZARD scenario controllers).
    """

    def __init__(self, workspace, simulator_kwargs=None):
        self.workspace = Path(workspace)
        self.simulator_kwargs = simulator_kwargs or {}
        self.controller = None
        self.step_count = 0
        self.episode_output_dir = None
        self._screen_size = self.simulator_kwargs.get(
            "screen_size", DEFAULT_SCREEN_SIZE
        )

    def start(self):
        """Initialize TDW Controller (auto-launches Unity build)."""
        from tdw.controller import Controller

        port = self.simulator_kwargs.get("port", DEFAULT_PORT)
        launch_build = self.simulator_kwargs.get("launch_build", True)

        logger.info(
            "Starting TDW Controller (port=%d, screen_size=%d, launch_build=%s)...",
            port, self._screen_size, launch_build,
        )
        self.controller = Controller(
            port=port,
            check_version=True,
            launch_build=launch_build,
        )
        logger.info("TDW Controller started.")

    def stop(self):
        """Shutdown the TDW Controller and Unity process."""
        if self.controller is not None:
            try:
                self.controller.communicate({"$type": "terminate"})
            except Exception:
                pass
            self.controller = None

    def reset(self, reset_config):
        """Reset to a scene for smoke tests.

        Override in subclasses for task-specific resets (LogPlayback, etc.).
        """
        self.step_count = 0
        scene = reset_config.get("scene", DEFAULT_SCENE)

        logger.info("Resetting to scene: %s", scene)

        # Load scene and configure rendering
        commands = [
            self.controller.get_add_scene(scene),
            {"$type": "set_screen_size",
             "width": self._screen_size, "height": self._screen_size},
            {"$type": "set_render_quality", "render_quality": 5},
        ]

        # Create a third-person camera
        commands.extend([
            {"$type": "create_avatar", "type": "A_Img_Caps_Kinematic",
             "id": "a"},
            {"$type": "teleport_avatar_to",
             "position": {"x": 0, "y": 1.5, "z": -2}, "avatar_id": "a"},
            {"$type": "look_at_position",
             "position": {"x": 0, "y": 0.5, "z": 0}, "avatar_id": "a"},
            {"$type": "set_pass_masks",
             "pass_masks": ["_img"], "avatar_id": "a"},
            {"$type": "send_images", "frequency": "always"},
        ])

        resp = self.controller.communicate(commands)
        return self._make_observation_response(resp)

    def step(self, action_text):
        """Execute a basic action and return observation.

        For the smoke test, steps just advance physics.
        Override in subclasses for task-specific actions.
        """
        self.step_count += 1

        # Default: step physics forward
        resp = self.controller.communicate([
            {"$type": "step_physics", "frames": 10},
        ])
        return self._make_observation_response(
            resp,
            reward=0.0,
            done=False,
            info={"env_step": float(self.step_count)},
        )

    # --- Main IPC Loop ---

    def run(self):
        """Main bridge loop — reads IPC commands, dispatches to handlers."""
        logger.info("TDW bridge starting (workspace: %s)", self.workspace)
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
                action_data = command.get("action", {})
                action_text = action_data.get("action_name", "")
                logger.trace(
                    "Step %d: action=%s", self.step_count + 1, action_text
                )

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
                    self.workspace,
                    make_error_response("Unknown command: %s" % cmd_type),
                )

    # --- Image extraction ---

    def _extract_rgb_from_response(self, resp):
        """Parse TDW response bytes to extract RGB image as numpy array.

        TDW returns OutputData in resp. We look for Images data
        (type id "imag") and extract the RGB pass.

        Returns:
            numpy array (H, W, 3) uint8, or None if no image found.
        """
        from tdw.output_data import OutputData, Images

        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "imag":
                images = Images(resp[i])
                # Get the first image pass (should be _img / RGB)
                for j in range(images.get_num_passes()):
                    if images.get_pass_mask(j) == "_img":
                        # TDW Images are PNG-encoded bytes
                        img_bytes = images.get_image(j)
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        return np.array(pil_img)[:, :, :3]  # Drop alpha
        return None

    def _make_observation_response(self, resp, reward=0.0, done=False, info=None):
        """Extract image from TDW response and build IPC response."""
        rgb_array = self._extract_rgb_from_response(resp)

        save_dir = (
            Path(self.episode_output_dir)
            if self.episode_output_dir
            else self.workspace
        )
        rgb_path = save_dir / ("rgb_%04d.png" % self.step_count)

        if rgb_array is not None:
            Image.fromarray(rgb_array).save(str(rgb_path))
        else:
            # Fallback: generate a minimal placeholder image
            logger.warning("No image in TDW response, generating placeholder")
            placeholder = np.zeros(
                (self._screen_size, self._screen_size, 3), dtype=np.uint8
            )
            Image.fromarray(placeholder).save(str(rgb_path))

        return make_observation_response(
            rgb_path=str(rgb_path),
            agent_pose=[0.0, 1.5, -2.0, 0.0, 0.0, 0.0],
            metadata={"step": str(self.step_count)},
            reward=reward,
            done=done,
            info=info or {},
        )


def main():
    parser = argparse.ArgumentParser(description="TDW v1.11.23 bridge")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--simulator-kwargs", type=str, default=None,
        help="JSON string of simulator configuration",
    )
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")

    simulator_kwargs = (
        json.loads(args.simulator_kwargs) if args.simulator_kwargs else {}
    )
    bridge = TDWBridge(
        workspace=args.workspace,
        simulator_kwargs=simulator_kwargs,
    )
    bridge.run()


if __name__ == "__main__":
    main()
