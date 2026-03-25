"""TDW v1.11.23 simulator.

Stub implementation — the bridge.py handles the actual TDW interaction.
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class TDWSimulator(BaseSimulator):
    """TDW 1.11.23 simulator for HAZARD benchmark."""

    @property
    def name(self) -> str:
        return "tdw"

    @property
    def version(self) -> str:
        return "v1_11_23"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
