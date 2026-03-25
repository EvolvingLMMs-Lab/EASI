"""Native render platform — use existing DISPLAY environment variable."""

from __future__ import annotations

import os

from .base import RenderPlatform


class NativePlatform(RenderPlatform):
    """Use the existing DISPLAY. Fails at validation if none is set."""

    @property
    def name(self) -> str:
        return "native"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def is_available(self) -> bool:
        return bool(os.environ.get("DISPLAY", ""))
