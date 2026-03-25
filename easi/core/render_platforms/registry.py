"""Render platform registry — maps names to platform classes."""

from __future__ import annotations

from .auto import AutoPlatform
from .base import RenderPlatform
from .egl import EGLPlatform
from .headless import HeadlessPlatform
from .native import NativePlatform
from .xorg import XorgPlatform
from .xvfb import XvfbPlatform

_BUILTIN: dict[str, type[RenderPlatform]] = {
    "auto": AutoPlatform,
    "native": NativePlatform,
    "xvfb": XvfbPlatform,
    "egl": EGLPlatform,
    "headless": HeadlessPlatform,
    "xorg": XorgPlatform,
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
