"""AI2-THOR v2.1.0 simulator (legacy API).

Stub implementation — the bridge.py handles the actual AI2-THOR interaction.
"""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class AI2ThorSimulatorV210(BaseSimulator):
    """AI2-THOR 2.1.0 simulator for embodiedbench ALFRED track."""

    @property
    def name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v2_1_0"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
