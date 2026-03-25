"""AI2-THOR v5.0.0 simulator (modern API)."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class AI2ThorSimulatorV500(BaseSimulator):
    """AI2-THOR 5.0.0 simulator with modern API."""

    @property
    def name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v5_0_0"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
