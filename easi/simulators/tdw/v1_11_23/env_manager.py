"""Environment manager for TDW v1.11.23.

Used by HAZARD benchmark. Requires Python 3.10 and Xvfb (Unity build needs X11).
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class TDWEnvManager(BaseEnvironmentManager):
    """Environment manager for TDW 1.11.23."""

    @property
    def simulator_name(self) -> str:
        return "tdw"

    @property
    def version(self) -> str:
        return "v1_11_23"

    @property
    def needs_display(self) -> bool:
        return True  # TDW Unity build requires X11

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb"]

    def get_validation_import(self) -> str:
        return "from tdw.controller import Controller; print('tdw ok')"
