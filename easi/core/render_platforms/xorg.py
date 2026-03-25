"""Xorg render platform — GPU-accelerated X11 display managed by EASI.

``XorgPlatform`` owns the ``XorgManager`` lifecycle and resolves per-worker
bindings for adapter-driven simulator launch wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from easi.utils.logging import get_logger

from .base import EnvVars, RenderPlatform, WorkerBinding
from .xorg_manager import XorgManager

if TYPE_CHECKING:
    from .xorg_manager import XorgInstance

logger = get_logger(__name__)


class XorgPlatform(RenderPlatform):
    """Render platform backed by auto-managed Xorg servers.

    Call ``setup(gpu_ids=...)`` to start Xorg servers, then
    ``for_worker(worker_id)`` to get per-worker bindings.
    ``teardown()`` stops all servers.
    """

    def __init__(self, env_manager=None):
        super().__init__(env_manager=env_manager)
        self._xorg_mgr: XorgManager | None = None
        self._instances: list[XorgInstance] = []

    @property
    def name(self) -> str:
        return "xorg"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        return EnvVars()

    def is_available(self) -> bool:
        return True

    def setup(self, gpu_ids: list[int] | None = None) -> None:
        """Start one Xorg server per GPU."""
        self._xorg_mgr = XorgManager(gpu_ids=gpu_ids or [0])
        self._instances = self._xorg_mgr.start()

    def teardown(self) -> None:
        """Stop all Xorg servers."""
        if self._xorg_mgr is not None:
            self._xorg_mgr.stop()
            self._xorg_mgr = None
            self._instances = []

    def for_worker(self, worker_id: int) -> WorkerBinding:
        """Resolve a per-worker binding for a specific Xorg instance."""
        if not self._instances:
            raise RuntimeError(
                "XorgPlatform.setup() must be called before for_worker()"
            )
        inst = self._instances[worker_id % len(self._instances)]
        return _build_worker_binding(inst.display, inst.gpu_id)


def _build_worker_binding(display_num: int, gpu_id: int) -> WorkerBinding:
    return WorkerBinding(
        display=f":{display_num}",
        cuda_visible_devices=str(gpu_id),
        extra_env=EnvVars(replace={"EASI_GPU_DISPLAY": "1"}),
        metadata={"backend": "xorg", "display_num": display_num, "gpu_id": gpu_id},
    )
