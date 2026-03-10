"""Xorg render platform — GPU-accelerated X11 display managed by EASI."""

from __future__ import annotations

from easi.core.render_platform import EnvVars, RenderPlatform


class XorgPlatform(RenderPlatform):
    """Render platform backed by a managed Xorg server on a specific GPU.

    Each worker gets its own ``XorgPlatform`` instance with a dedicated
    display number and GPU ID, set by the ``XorgManager``.
    """

    def __init__(self, display_num: int, gpu_id: int):
        super().__init__()
        self.display_num = display_num
        self.gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "xorg"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        return EnvVars(replace={
            "DISPLAY": f":{self.display_num}",
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "EASI_GPU_DISPLAY": "1",
        })

    def is_available(self) -> bool:
        return True
