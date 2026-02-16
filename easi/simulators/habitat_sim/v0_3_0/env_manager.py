"""Environment manager for Habitat v0.3.0."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class HabitatEnvManagerV030(BaseEnvironmentManager):
    """Environment manager for Habitat 0.3.0."""

    @property
    def simulator_name(self) -> str:
        return "habitat_sim"

    @property
    def version(self) -> str:
        return "v0_3_0"

    @property
    def needs_display(self) -> bool:
        return True  # habitat-sim needs a display (xvfb on headless servers)

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb", "egl"]

    def get_validation_import(self) -> str:
        return "import habitat_sim"
