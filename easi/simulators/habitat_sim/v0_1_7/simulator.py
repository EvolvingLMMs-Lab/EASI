"""Habitat simulator v0.1.7."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class HabitatSimulatorV017(BaseSimulator):
    """Habitat 0.1.7 simulator (VLN-CE R2R)."""

    @property
    def name(self) -> str:
        return "habitat_sim"

    @property
    def version(self) -> str:
        return "v0_1_7"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
