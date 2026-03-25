"""Habitat simulator v0.3.0."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class HabitatSimulatorV030(BaseSimulator):
    """Habitat 0.3.0 simulator (ReplicaCAD rearrangement)."""

    @property
    def name(self) -> str:
        return "habitat_sim"

    @property
    def version(self) -> str:
        return "v0_3_0"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
