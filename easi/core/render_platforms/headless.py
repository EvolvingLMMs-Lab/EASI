"""Headless render platform — no display, for simulators with native headless support."""

from __future__ import annotations

from .base import RenderPlatform


class HeadlessPlatform(RenderPlatform):
    """No display at all -- for simulators with native headless support."""

    @property
    def name(self) -> str:
        return "headless"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd
