from __future__ import annotations

import os
from pathlib import Path

from easi.core.render_platforms import (
    EnvVars,
    SimulatorRenderAdapter,
    WorkerBinding,
)


def _coppeliasim_qt_env_vars(
    env_manager,
    *,
    include_mesa_egl: bool = False,
    headless: bool = True,
) -> EnvVars:
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


class CoppeliaSimRenderAdapter(SimulatorRenderAdapter):
    def __init__(self, env_manager=None):
        self._env_manager = env_manager

    def get_env_vars(self, binding: WorkerBinding) -> EnvVars:
        backend = binding.metadata.get("backend", "")
        if backend == "xvfb":
            return _coppeliasim_qt_env_vars(
                self._env_manager, include_mesa_egl=True, headless=True
            )
        if backend in ("native", "xorg") or binding.cuda_visible_devices is not None:
            return _coppeliasim_qt_env_vars(
                self._env_manager, include_mesa_egl=False, headless=False
            )
        has_display = bool(os.environ.get("DISPLAY", ""))
        return _coppeliasim_qt_env_vars(
            self._env_manager,
            include_mesa_egl=not has_display,
            headless=not has_display,
        )
