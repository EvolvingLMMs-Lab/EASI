"""Custom render platforms for CoppeliaSim V4.1.0.

CoppeliaSim needs simulator-specific env vars (QT_QPA_PLATFORM_PLUGIN_PATH,
__EGL_VENDOR_LIBRARY_FILENAMES) that depend on the CoppeliaSim binary location.
These custom platforms compute the correct paths from the env_manager.
"""

from __future__ import annotations

import os
from pathlib import Path

from easi.core.render_platform import (
    AutoPlatform,
    EnvVars,
    NativePlatform,
    XvfbPlatform,
)


def _coppeliasim_xvfb_env_vars(env_manager) -> EnvVars:
    """Compute CoppeliaSim env vars needed when running under Xvfb.

    Returns EnvVars with:
    - QT_QPA_PLATFORM_PLUGIN_PATH prepended to coppeliasim_root
      (bundled Qt plugins needed when no system display)
    - __EGL_VENDOR_LIBRARY_FILENAMES set to Mesa vendor
      (NVIDIA EGL crashes Xvfb on some systems)
    """
    if env_manager is None:
        return EnvVars()
    binary_dir_name = env_manager.installation_kwargs.get("binary_dir_name", "")
    if not binary_dir_name:
        return EnvVars()
    t = env_manager._get_template_variables()
    coppeliasim_root = env_manager._resolve_template(
        "{extras_dir}/" + binary_dir_name, t
    )
    replace: dict[str, str] = {}
    mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if mesa_vendor.exists():
        replace["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_vendor)
    return EnvVars(
        replace=replace,
        prepend={"QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim_root},
    )


class CoppeliaSimNativePlatform(NativePlatform):
    """Native display for CoppeliaSim — no bundled Qt plugins, no Mesa override."""

    @property
    def name(self) -> str:
        return "native"


class CoppeliaSimXvfbPlatform(XvfbPlatform):
    """Xvfb platform for CoppeliaSim — sets Qt plugin path + Mesa EGL vendor."""

    @property
    def name(self) -> str:
        return "xvfb"

    def get_env_vars(self) -> EnvVars:
        return _coppeliasim_xvfb_env_vars(self._env_manager)


class CoppeliaSimAutoPlatform(AutoPlatform):
    """Auto-detect for CoppeliaSim — native if DISPLAY, xvfb otherwise.

    Env vars follow the same split: native mode skips Qt plugin path,
    xvfb mode prepends it.
    """

    @property
    def name(self) -> str:
        return "auto"

    def get_env_vars(self) -> EnvVars:
        if os.environ.get("DISPLAY", ""):
            return EnvVars()
        return _coppeliasim_xvfb_env_vars(self._env_manager)
