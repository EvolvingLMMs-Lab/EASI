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
from easi.core.render_platforms import EnvVars
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
        return "from tdw.controller import Controller; print('tdw ok')"

    def env_is_ready(self) -> bool:
        """Check conda env + TDW build binary exist."""
        if not super().env_is_ready():
            return False
        # Also verify the TDW Unity build binary was downloaded
        build_dir = self.installation_kwargs.get("build_dir_name", "")
        if build_dir:
            binary = self.get_extras_dir() / build_dir / "TDW.x86_64"
            if not binary.exists():
                logger.info(
                    "TDW conda env is ready but build binary missing at %s, "
                    "will re-run install to download it.",
                    binary,
                )
                return False
        return True

    def get_env_vars(self, render_platform_name: str | None = None) -> EnvVars:
        """Return TDW env vars for bridge subprocess."""
        build_dir = self.installation_kwargs.get("build_dir_name", "")
        if not build_dir:
            return EnvVars()
        t = self._get_template_variables()
        build_path = self._resolve_template("{extras_dir}/" + build_dir, t)
        return EnvVars(replace={"TDW_BUILD_PATH": build_path})

    def post_install(self, context: dict) -> None:
        """Download and extract TDW Unity build.

        Args:
            context: Dict with env_dir, extras_dir, env_vars keys.
        """
        extras_dir = Path(context["extras_dir"])
        build_url = self.installation_kwargs.get("build_url")
        build_filename = self.installation_kwargs.get("build_filename")

        logger.trace("post_install: extras_dir=%s, build_url=%s", extras_dir, build_url)

        if build_url and build_filename:
            logger.info("Downloading TDW build from %s", build_url)
            self._download_and_extract(
                url=build_url,
                filename=build_filename,
                dest_dir=extras_dir,
            )

            # Log what was extracted
            build_dir = self.installation_kwargs.get("build_dir_name", "TDW")
            build_path = extras_dir / build_dir
            logger.trace("Expected build dir: %s (exists=%s)", build_path, build_path.exists())
            if build_path.exists():
                items = sorted(p.name for p in build_path.iterdir())
                logger.trace("Build dir contents: %s", items)
            else:
                # List extras_dir to help debug incorrect dir name
                items = sorted(p.name for p in extras_dir.iterdir() if not p.name.startswith("."))
                logger.trace("extras_dir contents (build dir missing): %s", items)

            # Make binary executable
            binary = build_path / "TDW.x86_64"
            logger.trace("Expected binary: %s (exists=%s)", binary, binary.exists())
            if binary.exists():
                binary.chmod(binary.stat().st_mode | 0o755)
                logger.info("TDW build binary ready at %s", binary)
            else:
                logger.warning("TDW binary not found at %s after extraction", binary)
