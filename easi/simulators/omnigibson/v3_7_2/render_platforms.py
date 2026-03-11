"""Render adapter for OmniGibson v3.7.2.

OmniGibson-specific render behavior is expressed through
``OmniGibsonRenderAdapter`` so contributors can reuse core render backends
without adding backend-specific platform subclasses.
"""

from __future__ import annotations

import os

from easi.core.render_platforms import (
    EnvVars,
    SimulatorRenderAdapter,
    WorkerBinding,
)


class OmniGibsonRenderAdapter(SimulatorRenderAdapter):
    """Adapter for OmniGibson — sets OMNIGIBSON_HEADLESS from a WorkerBinding.

    Dispatches based on binding metadata ``backend`` key, then falls back to
    ``cuda_visible_devices`` (present on xorg bindings) as a headless heuristic,
    and finally reads ``$DISPLAY`` for the auto path.
    """

    def get_env_vars(self, binding: WorkerBinding) -> EnvVars:
        backend = binding.metadata.get("backend", "")
        if backend in ("native", "xorg"):
            return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0"})
        if backend == "xvfb":
            return EnvVars(replace={"OMNIGIBSON_HEADLESS": "1"})
        # xorg bindings always carry a GPU assignment; treat as display-attached
        if binding.cuda_visible_devices is not None:
            return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0"})
        # Auto path: check runtime display availability
        has_display = bool(os.environ.get("DISPLAY", ""))
        return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0" if has_display else "1"})
