"""Environment manager for TDW v1.11.23.

Used by HAZARD benchmark. Requires Python 3.10 and Xvfb (Unity build needs X11).

Handles:
1. Conda env creation (Python 3.10)
2. Pip deps via uv (requirements.txt)
3. TDW Unity build download + extraction (post_install)
4. Runtime env vars (TDW_BUILD_PATH) for bridge subprocess
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager
from easi.utils.logging import get_logger

logger = get_logger(__name__)


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

    def get_env_vars(self) -> dict[str, str]:
        """Return TDW env vars for bridge subprocess."""
        build_dir = self.installation_kwargs.get("build_dir_name", "")
        if not build_dir:
            return {}
        t = self._get_template_variables()
        build_path = self._resolve_template("{extras_dir}/" + build_dir, t)
        return {"TDW_BUILD_PATH": build_path}

    def post_install(self, context: dict) -> None:
        """Download and extract TDW Unity build.

        Args:
            context: Dict with env_dir, extras_dir, env_vars keys.
        """
        extras_dir = Path(context["extras_dir"])
        build_url = self.installation_kwargs.get("build_url")
        build_filename = self.installation_kwargs.get("build_filename")

        if build_url and build_filename:
            logger.info("Downloading TDW build from %s", build_url)
            self._download_and_extract(
                url=build_url,
                filename=build_filename,
                dest_dir=extras_dir,
            )
            # Make binary executable
            build_dir = self.installation_kwargs.get("build_dir_name", "TDW")
            binary = extras_dir / build_dir / "TDW.x86_64"
            if binary.exists():
                binary.chmod(binary.stat().st_mode | 0o755)
                logger.info("TDW build binary ready at %s", binary)
