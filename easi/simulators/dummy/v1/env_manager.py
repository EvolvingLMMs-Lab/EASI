"""Environment manager for the dummy simulator.

The dummy simulator runs in the host Python environment and has no
real dependencies, making it useful for testing the full pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class DummyEnvManager(BaseEnvironmentManager):
    """Environment manager for the dummy simulator (no real deps)."""

    @property
    def simulator_name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "v1"

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return []  # no system deps needed

    def get_validation_import(self) -> str:
        return "import json; print('dummy env ok')"

    def get_python_executable(self) -> str:
        """Use the current Python interpreter (no conda env needed for dummy)."""
        return sys.executable

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["headless", "dummy_custom"]

    def env_is_ready(self) -> bool:
        """Dummy env is always ready (uses host Python)."""
        return True

    def install(self) -> None:
        """No-op for dummy simulator."""
        pass
