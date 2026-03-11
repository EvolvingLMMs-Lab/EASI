"""Custom render platforms for CoppeliaSim V4.1.0.

CoppeliaSim needs simulator-specific env vars (QT_QPA_PLATFORM_PLUGIN_PATH,
__EGL_VENDOR_LIBRARY_FILENAMES) that depend on the CoppeliaSim binary location.
These custom platforms compute the correct paths from the env_manager.

Additionally, each platform sets COPPELIASIM_HEADLESS so the bridge can
start CoppeliaSim in the correct mode:
- native: headless=False (user has a real display and wants GUI)
- xvfb:   headless=True  (xvfb is only for Qt plugins, not CoppeliaSim rendering)
- auto:   headless=False if DISPLAY, True otherwise
"""

from __future__ import annotations

import os
from pathlib import Path

from easi.core.render_platforms import (
    AutoPlatform,
    EnvVars,
    NativePlatform,
    XorgPlatform,
    XvfbPlatform,
    _XorgWorkerPlatform,
)


def _coppeliasim_qt_env_vars(
    env_manager,
    *,
    include_mesa_egl: bool = False,
    headless: bool = True,
) -> EnvVars:
    """Compute CoppeliaSim Qt env vars.

    CoppeliaSim always needs QT_QPA_PLATFORM_PLUGIN_PATH to find its bundled
    Qt plugins (libqxcb.so, etc.), regardless of display mode.

    Args:
        env_manager: The CoppeliaSimEnvManager instance.
        include_mesa_egl: If True, also set __EGL_VENDOR_LIBRARY_FILENAMES
            to Mesa vendor (needed for Xvfb; NVIDIA EGL crashes Xvfb).
        headless: Whether CoppeliaSim should use its headless renderer.

    Returns:
        EnvVars with QT_QPA_PLATFORM_PLUGIN_PATH, COPPELIASIM_HEADLESS,
        and optionally Mesa EGL.
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
    replace: dict[str, str] = {
        "COPPELIASIM_HEADLESS": "1" if headless else "0",
    }
    if include_mesa_egl:
        mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa_vendor.exists():
            replace["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_vendor)
    return EnvVars(
        replace=replace,
        prepend={"QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim_root},
    )


class CoppeliaSimNativePlatform(NativePlatform):
    """Native display for CoppeliaSim — Qt plugin path, no Mesa override, headless=False."""

    @property
    def name(self) -> str:
        return "native"

    def get_env_vars(self) -> EnvVars:
        return _coppeliasim_qt_env_vars(
            self._env_manager,
            include_mesa_egl=False,
            headless=False,
        )


class CoppeliaSimXvfbPlatform(XvfbPlatform):
    """Xvfb platform for CoppeliaSim — Qt plugin path + Mesa EGL, headless=True.

    Xvfb provides a virtual X11 display for Qt plugin loading, but CoppeliaSim
    itself should use its native headless renderer (libcoppeliaSimHeadless.so).
    """

    @property
    def name(self) -> str:
        return "xvfb"

    def get_env_vars(self) -> EnvVars:
        return _coppeliasim_qt_env_vars(
            self._env_manager,
            include_mesa_egl=True,
            headless=True,
        )


class CoppeliaSimAutoPlatform(AutoPlatform):
    """Auto-detect for CoppeliaSim — native if DISPLAY, xvfb otherwise.

    Both modes need Qt plugin path. Only xvfb needs Mesa EGL override.
    Headless follows display: False with real display, True without.
    """

    @property
    def name(self) -> str:
        return "auto"

    def get_env_vars(self) -> EnvVars:
        has_display = bool(os.environ.get("DISPLAY", ""))
        return _coppeliasim_qt_env_vars(
            self._env_manager,
            include_mesa_egl=not has_display,
            headless=not has_display,
        )


class _CoppeliaSimXorgWorkerPlatform(_XorgWorkerPlatform):
    """Per-worker Xorg platform with CoppeliaSim Qt env vars merged in."""

    def __init__(self, display_num: int, gpu_id: int, coppeliasim_env: EnvVars):
        super().__init__(display_num=display_num, gpu_id=gpu_id)
        self._coppeliasim_env = coppeliasim_env

    def get_env_vars(self) -> EnvVars:
        return EnvVars.merge(super().get_env_vars(), self._coppeliasim_env)


class CoppeliaSimXorgPlatform(XorgPlatform):
    """Xorg platform for CoppeliaSim — Qt plugin path, no Mesa, headless=False.

    Xorg provides a real GPU-accelerated X11 display, so CoppeliaSim renders
    with its GUI library (not headless). Per-worker instances include the
    CoppeliaSim Qt env vars.
    """

    @property
    def name(self) -> str:
        return "xorg"

    def for_worker(self, worker_id: int) -> _CoppeliaSimXorgWorkerPlatform:
        if not self._instances:
            raise RuntimeError(
                "XorgPlatform.setup() must be called before for_worker()"
            )
        inst = self._instances[worker_id % len(self._instances)]
        coppeliasim_env = _coppeliasim_qt_env_vars(
            self._env_manager,
            include_mesa_egl=False,
            headless=False,
        )
        return _CoppeliaSimXorgWorkerPlatform(
            display_num=inst.display,
            gpu_id=inst.gpu_id,
            coppeliasim_env=coppeliasim_env,
        )
