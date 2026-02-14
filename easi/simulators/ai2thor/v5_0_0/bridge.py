"""Bridge subprocess for AI2-THOR v5.0.0 (modern API).

This script runs inside the ai2thor v5.0.0 conda environment.
The v5 API is significantly different from v2.1.0.

NOTE: This is a stub. Structure demonstrates how a real bridge works.

Usage:
    python bridge.py --workspace /tmp/easi_xxx
"""

from __future__ import annotations

import argparse
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


def run_bridge(workspace: Path) -> None:
    """Main bridge loop for AI2-THOR v5.0.0."""
    logger.info("AI2-THOR v5.0.0 bridge starting (workspace: %s)", workspace)

    # In a real implementation:
    # from ai2thor.controller import Controller
    # controller = Controller(width=1280, height=720, renderDepthImage=True)

    write_status(workspace, ready=True)

    step_count = 0

    while True:
        try:
            command = poll_for_command(workspace, timeout=300.0)
        except Exception as e:
            logger.error("Failed to read command: %s", e)
            break

        cmd_type = command.get("type")

        if cmd_type == "reset":
            episode_id = command.get("episode_id", "unknown")
            reset_config = command.get("reset_config", {})
            logger.info("Reset: episode_id=%s", episode_id)

            # Stub: controller.reset(scene=scene_id)
            # v5 API: controller.step(action="Initialize", ...)
            step_count = 0

            rgb_path = workspace / "rgb_0000.png"
            depth_path = workspace / "depth_0000.png"
            rgb_path.write_bytes(b"STUB_IMAGE")
            depth_path.write_bytes(b"STUB_DEPTH")

            response = make_observation_response(
                rgb_path=str(rgb_path),
                depth_path=str(depth_path),
                agent_pose=[0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
                metadata={"episode_id": episode_id, "step": "0"},
            )
            write_response(workspace, response)

        elif cmd_type == "step":
            step_count += 1
            action = parse_action_from_command(command)
            logger.trace("Step %d: action=%s", step_count, action.action_name)

            # Stub: event = controller.step(action=action_name, **params)
            # v5 API returns event with frame, depth, metadata

            rgb_name = f"rgb_{step_count:04d}.png"
            depth_name = f"depth_{step_count:04d}.png"
            (workspace / rgb_name).write_bytes(b"STUB_IMAGE")
            (workspace / depth_name).write_bytes(b"STUB_DEPTH")

            response = make_observation_response(
                rgb_path=str(workspace / rgb_name),
                depth_path=str(workspace / depth_name),
                agent_pose=[float(step_count) * 0.25, 0.9, 0.0, 0.0, 0.0, 0.0],
                metadata={"step": str(step_count)},
                done=action.action_name == "Stop",
            )
            write_response(workspace, response)

        elif cmd_type == "close":
            logger.info("Close command received")
            # Stub: controller.stop()
            write_response(workspace, {"status": "ok"})
            break

        else:
            write_response(workspace, make_error_response(f"Unknown command: {cmd_type}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI2-THOR v5.0.0 bridge")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    setup_logging("DEBUG")
    run_bridge(workspace=args.workspace)


if __name__ == "__main__":
    main()
