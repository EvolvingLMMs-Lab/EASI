"""Abstract base class for simulators.

Concrete simulators subclass this and implement:
- name/version properties
- _get_bridge_script_path() — path to the bridge.py for subprocess execution
- _parse_observation() — convert bridge output into Observation
- _format_action() — convert Action into dict the bridge understands

The shared template methods (reset, step, close) handle subprocess communication
via the filesystem IPC layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from easi.communication import filesystem, schemas
from easi.core.episode import Action, Observation, StepResult
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class BaseSimulator(ABC):
    """Abstract base for all simulators. Manages subprocess lifecycle via IPC."""

    def __init__(self, workspace_dir: Path | None = None):
        self._workspace_dir = workspace_dir
        self._runner = None  # Set when start() is called

    @property
    @abstractmethod
    def name(self) -> str:
        """Simulator name (e.g., 'ai2thor')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier (e.g., 'v2_1_0')."""
        ...

    @abstractmethod
    def _get_bridge_script_path(self) -> Path:
        """Return the absolute path to the bridge.py script for subprocess execution."""
        ...

    def _parse_observation(self, data: dict) -> Observation:
        """Parse observation from response data. Override for custom parsing."""
        return schemas.parse_observation(data)

    def _format_action(self, action: Action) -> dict:
        """Format an Action into the command dict. Override for custom formatting."""
        return schemas.make_step_command(action)

    def reset(
        self,
        episode_id: str,
        reset_config: dict | None = None,
        episode_output_dir: str | None = None,
    ) -> Observation:
        """Reset the simulator for a new episode.

        Sends a reset command to the bridge subprocess and waits for the
        observation response.

        Args:
            episode_id: Unique identifier for this episode.
            reset_config: Task-specific configuration for the episode.
            episode_output_dir: Directory where the bridge should save
                observation images. If None, the bridge uses its IPC workspace.
        """
        if self._runner is None:
            raise RuntimeError("Simulator not started. Call start() first.")

        command = schemas.make_reset_command(episode_id, reset_config, episode_output_dir)
        response = self._runner.send_command(command)

        if response.get("status") == "error":
            from easi.core.exceptions import SimulatorError
            raise SimulatorError(f"Reset failed: {response.get('error', 'unknown')}")

        return self._parse_observation(response)

    def step(self, action: Action) -> StepResult:
        """Execute one action in the simulator.

        Sends a step command and returns the StepResult.
        """
        if self._runner is None:
            raise RuntimeError("Simulator not started. Call start() first.")

        command = self._format_action(action)
        response = self._runner.send_command(command)

        if response.get("status") == "error":
            from easi.core.exceptions import SimulatorError
            raise SimulatorError(f"Step failed: {response.get('error', 'unknown')}")

        return schemas.parse_step_result(response)

    def close(self) -> None:
        """Shut down the simulator subprocess."""
        if self._runner is not None:
            command = schemas.make_close_command()
            try:
                self._runner.send_command(command, timeout=10.0)
            except Exception:
                logger.warning("Close command failed, force-killing subprocess")
            self._runner.shutdown()
            self._runner = None

    def is_running(self) -> bool:
        """Check if the bridge subprocess is alive."""
        if self._runner is None:
            return False
        return self._runner.is_alive()

    def set_runner(self, runner: object) -> None:
        """Attach a SubprocessRunner instance (called by orchestration code)."""
        self._runner = runner
