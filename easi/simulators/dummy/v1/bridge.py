"""Bridge subprocess for the dummy simulator.

This script is launched by SubprocessRunner in a separate process.
It communicates with the parent via filesystem IPC (command.json / response.json).

The dummy simulator generates placeholder RGB images and responds to
reset/step/close commands with configurable delay.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--step-delay 0.1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add the repo root to sys.path so we can import easi modules
# This is needed when the bridge runs in a separate conda env
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


def _generate_dummy_image(workspace: Path, step: int, output_dir: Path | None = None) -> str:
    """Generate a small placeholder PNG image.

    Creates a minimal valid PNG file (1x1 pixel, red) to simulate
    an observation image without requiring Pillow.

    Args:
        workspace: IPC workspace directory (fallback for image saving).
        step: Current step number (used for filename and color).
        output_dir: If provided, save image here instead of workspace.
    """
    save_dir = output_dir if output_dir is not None else workspace
    rgb_path = save_dir / f"rgb_{step:04d}.png"

    # Minimal valid 1x1 red PNG (67 bytes)
    # This avoids requiring Pillow in the dummy simulator
    import struct
    import zlib

    def _create_minimal_png(width: int = 8, height: int = 8) -> bytes:
        """Create a minimal valid PNG with a solid color."""

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            crc = zlib.crc32(c) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

        # Generate raw pixel data (red-ish gradient based on step)
        r = (step * 37) % 256
        g = (step * 73) % 256
        b = (step * 113) % 256
        raw_data = b""
        for _ in range(height):
            raw_data += b"\x00"  # filter byte
            raw_data += bytes([r, g, b]) * width

        idat = _chunk(b"IDAT", zlib.compress(raw_data))
        iend = _chunk(b"IEND", b"")

        return header + ihdr + idat + iend

    rgb_path.write_bytes(_create_minimal_png())
    return str(rgb_path)


def run_bridge(workspace: Path, step_delay: float = 0.0) -> None:
    """Main bridge loop: read commands, process them, write responses."""
    logger.info("Dummy bridge starting (workspace: %s, step_delay: %.2f)", workspace, step_delay)

    # Signal readiness
    write_status(workspace, ready=True)

    step_count = 0
    episode_output_dir = None  # Set per-episode from reset command

    while True:
        try:
            command = poll_for_command(workspace, timeout=300.0)
        except Exception as e:
            logger.error("Failed to read command: %s", e)
            break

        cmd_type = command.get("type")

        if cmd_type == "reset":
            episode_id = command.get("episode_id", "unknown")
            logger.info("Reset: episode_id=%s", episode_id)

            # Read episode output directory (None for smoke tests)
            raw_output_dir = command.get("episode_output_dir")
            if raw_output_dir:
                episode_output_dir = Path(raw_output_dir)
                episode_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                episode_output_dir = None

            if step_delay > 0:
                time.sleep(step_delay)

            step_count = 0
            rgb_path = _generate_dummy_image(workspace, step_count, episode_output_dir)

            response = make_observation_response(
                rgb_path=rgb_path,
                agent_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata={"episode_id": episode_id, "step": "0"},
            )
            write_response(workspace, response)

        elif cmd_type == "step":
            step_count += 1
            action = parse_action_from_command(command)
            logger.trace("Step %d: action=%s", step_count, action.action_name)

            if step_delay > 0:
                time.sleep(step_delay)

            rgb_path = _generate_dummy_image(workspace, step_count, episode_output_dir)

            # Dummy: done after 10 steps or on Stop action
            done = step_count >= 10 or action.action_name == "Stop"

            response = make_observation_response(
                rgb_path=rgb_path,
                agent_pose=[float(step_count), 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata={"step": str(step_count)},
                reward=1.0 if done else 0.0,
                done=done,
            )
            write_response(workspace, response)

        elif cmd_type == "close":
            logger.info("Close command received, shutting down")
            write_response(workspace, {"status": "ok"})
            break

        else:
            logger.warning("Unknown command type: %s", cmd_type)
            write_response(workspace, make_error_response(f"Unknown command: {cmd_type}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy simulator bridge")
    parser.add_argument("--workspace", type=Path, required=True, help="IPC workspace directory")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Delay per step in seconds")
    args, _ = parser.parse_known_args()

    setup_logging("DEBUG")

    run_bridge(workspace=args.workspace, step_delay=args.step_delay)


if __name__ == "__main__":
    main()
