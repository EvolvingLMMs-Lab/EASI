"""Environment manager for Habitat v0.1.7."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class HabitatEnvManagerV017(BaseEnvironmentManager):
    """Environment manager for Habitat 0.1.7."""

    @property
    def simulator_name(self) -> str:
        return "habitat_sim"

    @property
    def version(self) -> str:
        return "v0_1_7"

    @property
    def default_render_platform(self) -> str:
        return "auto"

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["auto", "xvfb", "native", "egl"]

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb", "egl"]

    def get_validation_import(self) -> str:
        return "import habitat_sim; print('habitat-sim', habitat_sim.__version__)"
