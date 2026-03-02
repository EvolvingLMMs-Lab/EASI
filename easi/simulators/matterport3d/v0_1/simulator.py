"""Matterport3D simulator class."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class Matterport3DSimulator(BaseSimulator):
    """Matterport3D simulator (Docker-isolated)."""

    @property
    def name(self) -> str:
        return "matterport3d"

    @property
    def version(self) -> str:
        return "v0_1"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
