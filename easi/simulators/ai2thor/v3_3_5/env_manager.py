"""Environment manager for AI2-THOR v3.3.5 (arm-mode API).

Used by ManipulaTHOR. Requires Python 3.8 and Xvfb.
Installs AI2-THOR from git commit a84dd29 (~v3.3.5).
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class AI2ThorEnvManagerV335(BaseEnvironmentManager):
    """Environment manager for AI2-THOR 3.3.5."""

    @property
    def simulator_name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v3_3_5"

    @property
    def default_render_platform(self) -> str:
        return "auto"

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["auto", "xvfb", "native", "xorg"]

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb"]

    def get_validation_import(self) -> str:
        return "import ai2thor; print(f'ai2thor {ai2thor.__version__}')"
