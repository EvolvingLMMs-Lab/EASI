"""Auto render platform — native display if available, falls back to xvfb."""

from __future__ import annotations

import os

from .base import RenderPlatform
from .xvfb import XvfbPlatform


class AutoPlatform(RenderPlatform):
    """Detect native display; fall back to xvfb if unavailable."""

    @property
    def name(self) -> str:
        return "auto"

    @property
    def resolved_name(self) -> str:
        return "native" if os.environ.get("DISPLAY", "") else "xvfb"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        if os.environ.get("DISPLAY", ""):
            return cmd
        return XvfbPlatform().wrap_command(cmd, screen_config)
