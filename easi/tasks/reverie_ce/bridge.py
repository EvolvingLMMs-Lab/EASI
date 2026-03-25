"""REVERIE-CE bridge — inherits VLN-CE R2R bridge.

Runs inside the easi_habitat_sim_v0_1_7 conda env (Python 3.8).
Inherits all reset/step/extract logic from VLNCEBridge.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.tasks.vlnce_r2r.bridge import VLNCEBridge  # noqa: E402


class ReverieCEBridge(VLNCEBridge):
    """Bridge for REVERIE-CE. Identical to VLN-CE R2R for now."""
    pass


if __name__ == "__main__":
    ReverieCEBridge.main()
