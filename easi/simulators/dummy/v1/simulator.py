"""Dummy simulator implementation.

A simple simulator for testing the full pipeline end-to-end without
requiring any real simulator dependencies.
"""

from __future__ import annotations

from pathlib import Path

from easi.core.base_simulator import BaseSimulator


class DummySimulator(BaseSimulator):
    """Dummy simulator that generates placeholder observations."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "v1"

    def _get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"
