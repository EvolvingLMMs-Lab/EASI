"""OmniGibson v3.7.2 simulator.

Stub implementation — the bridge.py handles the actual OmniGibson interaction.
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class OmniGibsonSimulator(BaseSimulator):
    """OmniGibson 3.7.2 + Isaac Sim 4.5.0 simulator for BEHAVIOR-1K."""

    @property
    def name(self) -> str:
        return "omnigibson"

    @property
    def version(self) -> str:
        return "v3_7_2"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
