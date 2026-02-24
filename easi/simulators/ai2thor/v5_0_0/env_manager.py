"""Environment manager for AI2-THOR v5.0.0 (modern API)."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class AI2ThorEnvManagerV500(BaseEnvironmentManager):
    """Environment manager for AI2-THOR 5.0.0."""

    @property
    def simulator_name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v5_0_0"

    @property
    def default_render_platform(self) -> str:
        return "auto"

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["auto", "xvfb", "native"]

    @property
    def screen_config(self) -> str:
        return "1280x720x24"

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda"]

    def get_validation_import(self) -> str:
        return "import ai2thor; assert ai2thor.__version__.startswith('5.')"
