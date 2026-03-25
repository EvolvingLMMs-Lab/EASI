"""AI2-THOR v3.3.5 simulator (arm-mode, ManipulaTHOR)."""
from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class AI2ThorSimulatorV335(BaseSimulator):
    """AI2-THOR v3.3.5 simulator stub."""

    @property
    def name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v3_3_5"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
