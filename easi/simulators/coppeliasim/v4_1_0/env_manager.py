"""Environment manager for CoppeliaSim with PyRep.

Handles:
1. Conda env creation (Python 3.10)
2. Pip deps via uv (requirements.txt)
3. CoppeliaSim binary download + extraction (post_install)
4. PyRep build with COPPELIASIM_ROOT set (post_install)
5. Lua addon script copy (post_install)
6. Runtime env vars for bridge subprocess
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager
from easi.core.render_platform import EnvVars
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class CoppeliaSimEnvManagerV410(BaseEnvironmentManager):
    """Environment manager for CoppeliaSim V4.1.0 (PyRep)."""

    @property
    def simulator_name(self) -> str:
        return "coppeliasim"

    @property
    def version(self) -> str:
        return "v4_1_0"

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
        return ["conda", "xvfb"]

    def get_validation_import(self) -> str:
        return "from pyrep import PyRep; print('PyRep OK')"

    def get_env_vars(self, render_platform_name: str | None = None) -> EnvVars:
        """Return platform-agnostic CoppeliaSim env vars for bridge subprocess.

        Platform-specific vars (QT_QPA_PLATFORM_PLUGIN_PATH, __EGL_VENDOR_LIBRARY_FILENAMES)
        are handled by custom render platform classes in render_platforms.py.
        """
        binary_dir_name = self.installation_kwargs.get("binary_dir_name", "")
        if not binary_dir_name:
            return EnvVars()
        t = self._get_template_variables()
        coppeliasim_root = self._resolve_template(
            "{extras_dir}/" + binary_dir_name, t
        )
        # Include conda env lib dir so fontconfig/freetype/Qt deps resolve
        conda_lib = self._resolve_template("{env_dir}/lib", t)
        ld_path = f"{coppeliasim_root}:{conda_lib}"
        return EnvVars(
            replace={"COPPELIASIM_ROOT": coppeliasim_root},
            prepend={"LD_LIBRARY_PATH": ld_path},
        )

    def post_install(self, context: dict) -> None:
        """Download CoppeliaSim, build PyRep, copy lua addon.

        All URLs and filenames come from self.installation_kwargs (set
        by manifest.yaml). Order matters:
        1. Download + extract CoppeliaSim binary
        2. pip install PyRep (needs COPPELIASIM_ROOT at build time)
        3. Copy simAddOnScript_PyRep.lua into CoppeliaSim root
        """
        extras_dir = Path(context["extras_dir"])
        env_vars = context["env_vars"]
        python = self.get_python_executable()

        binary_url = self.installation_kwargs.get("binary_url")
        binary_filename = self.installation_kwargs.get("binary_filename")
        pyrep_git_url = self.installation_kwargs.get("pyrep_git_url")
        lua_addon_script = self.installation_kwargs.get("lua_addon_script")

        # Download + extract CoppeliaSim binary
        if binary_url and binary_filename:
            logger.info("Downloading CoppeliaSim from %s", binary_url)
            self._download_and_extract(
                url=binary_url,
                filename=binary_filename,
                dest_dir=extras_dir,
            )

        # Create versioned symlinks (PyRep links against libcoppeliaSim.so.1)
        if "COPPELIASIM_ROOT" in env_vars:
            coppeliasim_root = Path(env_vars["COPPELIASIM_ROOT"])
            for lib_name in ["libcoppeliaSim.so", "libcoppeliaSimHeadless.so"]:
                lib_path = coppeliasim_root / lib_name
                symlink = coppeliasim_root / f"{lib_name}.1"
                if lib_path.exists() and not symlink.exists():
                    symlink.symlink_to(lib_path.name)
                    logger.info("Created symlink %s -> %s", symlink.name, lib_path.name)

        # Build env dict with COPPELIASIM_ROOT for PyRep's native build
        build_env = os.environ.copy()
        build_env.update(env_vars)

        # pip install PyRep (C++ extensions link against CoppeliaSim)
        if pyrep_git_url:
            logger.info("Installing PyRep from %s", pyrep_git_url)
            self._run_command(
                [python, "-m", "uv", "pip", "install", pyrep_git_url],
                "pip install PyRep",
                env=build_env,
            )

        # Copy PyRep addon lua script into CoppeliaSim root
        if lua_addon_script and "COPPELIASIM_ROOT" in env_vars:
            coppeliasim_root = Path(env_vars["COPPELIASIM_ROOT"])
            lua_src = Path(__file__).parent / lua_addon_script
            if lua_src.exists():
                logger.info("Copying %s to %s", lua_addon_script, coppeliasim_root)
                shutil.copy(str(lua_src), str(coppeliasim_root / lua_addon_script))
            else:
                logger.warning(
                    "Lua addon script not found at %s", lua_src
                )
