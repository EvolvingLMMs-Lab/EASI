"""Environment manager for AI2-THOR v2.1.0 (legacy API).

Used by embodiedbench ALFRED track. Requires Python 3.8 and Xvfb.
"""

from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class AI2ThorEnvManagerV210(BaseEnvironmentManager):
    """Environment manager for AI2-THOR 2.1.0."""

    @property
    def simulator_name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v2_1_0"

    @property
    def default_render_platform(self) -> str:
        return "auto"

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["auto", "xvfb", "native"]

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb"]

    def get_validation_import(self) -> str:
        return "import ai2thor; assert ai2thor.__version__.startswith('2.1')"
