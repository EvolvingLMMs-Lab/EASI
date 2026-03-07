"""Generic Habitat v0.1.7 bridge for smoke testing.

This script runs inside the easi_habitat_sim_v0_1_7 conda environment (Python 3.8).
Task-specific bridges (e.g., VLNCEBridge) extend BaseBridge directly.
This generic bridge is used by `easi sim test habitat_sim:v0_1_7`.

Usage:
    python bridge.py --workspace /tmp/easi_xxx
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.communication.filesystem import poll_for_command, write_response, write_status
from easi.communication.schemas import make_error_response, make_observation_response
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _generate_placeholder_image(directory: Path, step: int) -> str:
    """Generate a minimal 8x8 placeholder PNG."""
    rgb_path = directory / f"rgb_{step:04d}.png"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    width, height = 8, 8
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + bytes([128, 128, 128]) * width
    idat = _chunk(b"IDAT", zlib.compress(raw_data))
    iend = _chunk(b"IEND", b"")
    rgb_path.write_bytes(header + ihdr + idat + iend)
    return str(rgb_path)


class HabitatV017Bridge:
    """Smoke test bridge -- verifies habitat-sim imports work."""

    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.step_count = 0

    def run(self):
        import habitat_sim

        logger.info("habitat-sim %s loaded successfully", habitat_sim.__version__)
        write_status(self.workspace, ready=True)

        while True:
            command = poll_for_command(self.workspace, timeout=60.0)
            cmd_type = command.get("type")

            if cmd_type == "reset":
                episode_id = command.get("episode_id", "unknown")
                logger.info("Reset: episode_id=%s", episode_id)
                self.step_count = 0
                rgb_path = _generate_placeholder_image(self.workspace, self.step_count)
                write_response(self.workspace, make_observation_response(
                    rgb_path=rgb_path,
                    agent_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    metadata={"episode_id": episode_id, "step": "0"},
                ))

            elif cmd_type == "step":
                self.step_count += 1
                rgb_path = _generate_placeholder_image(self.workspace, self.step_count)
                done = self.step_count >= 5
                write_response(self.workspace, make_observation_response(
                    rgb_path=rgb_path,
                    agent_pose=[float(self.step_count), 0.0, 0.0, 0.0, 0.0, 0.0],
                    metadata={"step": str(self.step_count)},
                    reward=0.0,
                    done=done,
                ))

            elif cmd_type == "close":
                logger.info("Close command received, shutting down")
                write_response(self.workspace, {"status": "ok"})
                break

            else:
                logger.warning("Unknown command type: %s", cmd_type)
                write_response(self.workspace, make_error_response(
                    f"Unknown command: {cmd_type}"
                ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--simulator-kwargs", type=str, default=None)
    args, _ = parser.parse_known_args()
    setup_logging("DEBUG")
    bridge = HabitatV017Bridge(workspace=args.workspace)
    bridge.run()


if __name__ == "__main__":
    main()
