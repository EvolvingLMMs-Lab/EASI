"""Pluggable render platform abstraction.

Each render platform encapsulates how to launch a bridge subprocess with
the correct display/rendering environment.  Simulators declare a default
platform; users can override via CLI (--render-platform) or task YAML.

Built-in platforms:
    auto     — use native DISPLAY if available, fall back to xvfb
    native   — require existing DISPLAY
    xvfb     — always wrap with xvfb-run
    egl      — GPU-accelerated headless via EGL
    headless — no display (simulator has native headless support)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from easi.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnvVars:
    """Structured environment variables with replace/prepend semantics.

    ``replace`` vars overwrite any existing value.
    ``prepend`` vars are prepended with ':' to any existing value (for PATH-like vars).
    """

    replace: dict[str, str] = field(default_factory=dict)
    prepend: dict[str, str] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, str]:
        """Combine into single dict (for internal use like post_install)."""
        return {**self.replace, **self.prepend}

    def apply_to_env(self, base: dict[str, str]) -> dict[str, str]:
        """Merge into a base env dict (e.g. os.environ.copy())."""
        env = dict(base)
        for k, v in self.replace.items():
            env[k] = v
        for k, v in self.prepend.items():
            env[k] = f"{v}:{env[k]}" if k in env else v
        return env

    def __bool__(self) -> bool:
        return bool(self.replace) or bool(self.prepend)

    @classmethod
    def merge(cls, *env_vars: EnvVars) -> EnvVars:
        """Merge multiple EnvVars. Later values win for replace; prepend values concatenate."""
        replace: dict[str, str] = {}
        prepend: dict[str, str] = {}
        for ev in env_vars:
            replace.update(ev.replace)
            for k, v in ev.prepend.items():
                prepend[k] = f"{v}:{prepend[k]}" if k in prepend else v
        return cls(replace=replace, prepend=prepend)


class RenderPlatform(ABC):
    """Strategy interface for display/rendering backends."""

    def __init__(self, env_manager=None):
        self._env_manager = env_manager

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'xvfb', 'egl')."""
        ...

    @abstractmethod
    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        """Optionally wrap the bridge launch command.

        Args:
            cmd: The original command ``[python, bridge.py, ...]``.
            screen_config: Screen resolution string, e.g. ``"1024x768x24"``.

        Returns:
            The (possibly wrapped) command.
        """
        ...

    def get_env_vars(self) -> EnvVars:
        """Extra env vars needed by this platform (merged into subprocess)."""
        return EnvVars()

    def get_system_deps(self) -> list[str]:
        """System dependency names required by this platform."""
        return []

    def is_available(self) -> bool:
        """Whether this platform can run in the current environment."""
        return True


# -- Built-in implementations ------------------------------------------------


class HeadlessPlatform(RenderPlatform):
    """No display at all -- for simulators with native headless support."""

    @property
    def name(self) -> str:
        return "headless"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd


class NativePlatform(RenderPlatform):
    """Use the existing DISPLAY. Fails at validation if none is set."""

    @property
    def name(self) -> str:
        return "native"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def is_available(self) -> bool:
        return bool(os.environ.get("DISPLAY", ""))


class XvfbPlatform(RenderPlatform):
    """Always wrap with ``xvfb-run``."""

    @property
    def name(self) -> str:
        return "xvfb"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return [
            "xvfb-run", "-a",
            "-s", f"-screen 0 {screen_config}",
        ] + cmd

    def get_system_deps(self) -> list[str]:
        return ["xvfb"]


class EGLPlatform(RenderPlatform):
    """GPU-accelerated headless rendering via EGL (no X11 needed)."""

    @property
    def name(self) -> str:
        return "egl"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        replace: dict[str, str] = {"PYOPENGL_PLATFORM": "egl"}
        mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa_vendor.exists():
            replace["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_vendor)
        return EnvVars(replace=replace)

    def get_system_deps(self) -> list[str]:
        return ["egl"]


class AutoPlatform(RenderPlatform):
    """Detect native display; fall back to xvfb if unavailable."""

    @property
    def name(self) -> str:
        return "auto"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        if os.environ.get("DISPLAY", ""):
            return cmd
        return XvfbPlatform().wrap_command(cmd, screen_config)


# -- Registry -----------------------------------------------------------------

_BUILTIN: dict[str, type[RenderPlatform]] = {
    "auto": AutoPlatform,
    "native": NativePlatform,
    "xvfb": XvfbPlatform,
    "egl": EGLPlatform,
    "headless": HeadlessPlatform,
}


def get_render_platform(name: str) -> RenderPlatform:
    """Instantiate a render platform by name.

    Raises:
        ValueError: If name is not recognised.
    """
    cls = _BUILTIN.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown render platform '{name}'. "
            f"Available: {', '.join(sorted(_BUILTIN))}"
        )
    return cls()


def available_platforms() -> list[str]:
    """Return sorted list of registered platform names."""
    return sorted(_BUILTIN)
