"""CoppeliaSim V4.1.0 simulator entry point."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class CoppeliaSimSimulatorV410(BaseSimulator):
    """CoppeliaSim V4.1.0 with PyRep for robotic manipulation."""

    @property
    def name(self) -> str:
        return "coppeliasim"

    @property
    def version(self) -> str:
        return "v4_1_0"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
