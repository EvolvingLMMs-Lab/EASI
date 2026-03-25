"""Xvfb render platform — virtual framebuffer via xvfb-run."""

from __future__ import annotations

from .base import RenderPlatform


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
