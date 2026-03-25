"""EGL render platform — GPU-accelerated headless rendering via EGL (no X11 needed)."""

from __future__ import annotations

from pathlib import Path

from .base import EnvVars, RenderPlatform


class EGLPlatform(RenderPlatform):
    """GPU-accelerated headless rendering via EGL (no X11 needed)."""

    @property
    def name(self) -> str:
        return "egl"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        replace: dict[str, str] = {"PYOPENGL_PLATFORM": "egl"}
        # Don't set __EGL_VENDOR_LIBRARY_FILENAMES — let glvnd discover
        # vendors from its default search path (/usr/share/glvnd/egl_vendor.d).
        # Setting it explicitly restricts discovery to a single vendor file,
        # which breaks GPU rendering when NVIDIA is available but Mesa is forced.
        return EnvVars(replace=replace)

    def get_system_deps(self) -> list[str]:
        return ["egl"]
