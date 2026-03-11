"""Render platform package — pluggable display/rendering backends.

Each render platform encapsulates how to launch a bridge subprocess with
the correct display/rendering environment. Simulators declare a default
platform; users can override via CLI (--render-platform) or task YAML.
Simulator-specific render quirks belong in a ``SimulatorRenderAdapter``
registered from the simulator manifest, not in backend-specific subclasses.

Built-in platforms:
    auto     — use native DISPLAY if available, fall back to xvfb
    native   — require existing DISPLAY
    xvfb     — always wrap with xvfb-run
    egl      — GPU-accelerated headless via EGL
    headless — no display (simulator has native headless support)
    xorg     — GPU-accelerated X11 via auto-managed Xorg servers
"""

from .auto import AutoPlatform
from .base import EnvVars, RenderPlatform, SimulatorRenderAdapter, WorkerBinding
from .egl import EGLPlatform
from .headless import HeadlessPlatform
from .native import NativePlatform
from .registry import available_platforms, get_render_platform
from .xorg import XorgPlatform
from .xorg_manager import XorgInstance, XorgManager
from .xvfb import XvfbPlatform

__all__ = [
    # Base types
    "RenderPlatform",
    "EnvVars",
    "WorkerBinding",
    "SimulatorRenderAdapter",
    # Built-in platforms
    "AutoPlatform",
    "NativePlatform",
    "XvfbPlatform",
    "EGLPlatform",
    "HeadlessPlatform",
    "XorgPlatform",
    # Xorg internals
    "XorgManager",
    "XorgInstance",
    # Registry
    "get_render_platform",
    "available_platforms",
]
