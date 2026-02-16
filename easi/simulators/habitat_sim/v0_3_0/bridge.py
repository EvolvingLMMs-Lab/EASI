"""Generic Habitat v0.3.0 bridge for smoke testing.

This script runs inside the easi_habitat_sim_v0_3_0 conda environment.
Task-specific bridges (e.g., EBHabitatBridge) extend BaseBridge directly.
This generic bridge is used by `easi sim test habitat_sim:v0_3_0`.

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

from easi.communication.filesystem import poll_for_command, write_response, write_status
from easi.communication.schemas import make_error_response
from easi.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class HabitatV030Bridge:
    """Smoke test bridge — verifies habitat-sim imports work."""

    def __init__(self, workspace):
        self.workspace = Path(workspace)

    def run(self):
        import habitat_sim

        logger.info("habitat-sim %s loaded successfully", habitat_sim.__version__)
        write_status(self.workspace, ready=True)

        while True:
            command = poll_for_command(self.workspace, timeout=60.0)
            if command.get("type") == "close":
                write_response(self.workspace, {"status": "ok"})
                break
            write_response(
                self.workspace,
                make_error_response(
                    "Smoke test bridge: only 'close' supported"
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--simulator-kwargs", type=str, default=None)
    args, _ = parser.parse_known_args()
    setup_logging("DEBUG")
    bridge = HabitatV030Bridge(workspace=args.workspace)
    bridge.run()


if __name__ == "__main__":
    main()
