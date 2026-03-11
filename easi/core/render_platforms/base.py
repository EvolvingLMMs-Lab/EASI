"""Base classes for render platforms.

Defines the RenderPlatform ABC and EnvVars dataclass used by all built-in
render backends plus simulator render adapters.

Also defines WorkerBinding (resolved per-worker render facts) and
SimulatorRenderAdapter (optional simulator-specific launch adjustments).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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
            if ev is None:
                continue
            replace.update(ev.replace)
            for k, v in ev.prepend.items():
                prepend[k] = f"{v}:{prepend[k]}" if k in prepend else v
        return cls(replace=replace, prepend=prepend)


@dataclass
class WorkerBinding:
    """Resolved per-worker render facts produced by a render backend.

    Carries the concrete display and GPU assignment for one worker subprocess,
    plus any extra env vars and arbitrary metadata the backend wants to pass
    downstream (e.g. to a SimulatorRenderAdapter).

    Fields:
        display: X display string (e.g. ":10"), or None for headless/EGL.
        cuda_visible_devices: GPU id(s) string (e.g. "0" or "0,1"), or None.
        extra_env: Additional env vars contributed by the render backend.
        metadata: Arbitrary backend-specific data for adapter consumption.
    """

    display: str | None = None
    cuda_visible_devices: str | None = None
    extra_env: EnvVars = field(default_factory=EnvVars)
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulatorRenderAdapter(ABC):
    """Extension point for simulator-specific render launch adjustments.

    Simulators that need to inject render-related env vars or wrap the launch
    command beyond what the core render backend provides should subclass this
    and register it via the simulator manifest.

    Default implementations are no-ops so simulators that need no adjustments
    do not have to implement anything.
    """

    def get_env_vars(self, binding: WorkerBinding) -> EnvVars:
        return EnvVars()

    def wrap_command(self, cmd: list[str], binding: WorkerBinding) -> list[str]:
        return cmd


class RenderPlatform(ABC):
    """Strategy interface for display/rendering backends.

    Lifecycle hooks (``setup`` / ``teardown``) allow platforms that manage
    external services (e.g. Xorg) to start and stop them without
    if/else logic in the callers. ``for_worker`` always returns a
    ``WorkerBinding`` so callers have a uniform interface; backends that
    need per-worker GPU/display assignment (e.g. ``XorgPlatform``) override it.
    """

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

    # -- Lifecycle hooks (override in platforms that manage services) ----------

    def setup(self, gpu_ids: list[int] | None = None) -> None:
        """Called once before any simulator is created. Start external services."""

    def teardown(self) -> None:
        """Called once after all simulators are done. Stop external services."""

    def for_worker(self, worker_id: int) -> WorkerBinding:
        """Return the per-worker render binding for this platform.

        The default returns a ``WorkerBinding`` carrying the backend name in
        ``metadata``.  Platforms that assign per-worker displays or GPUs
        (e.g. ``XorgPlatform``) override this to return a populated binding.
        """
        return WorkerBinding(metadata={"backend": self.name})
