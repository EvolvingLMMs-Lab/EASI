"""Protocol classes defining interfaces for EASI components."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from easi.core.episode import Action, Observation, StepResult


@runtime_checkable
class SimulatorProtocol(Protocol):
    """Any simulator must satisfy this interface."""

    name: str
    version: str

    def reset(self, episode_id: str, reset_config: dict | None = None) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def close(self) -> None: ...
    def is_running(self) -> bool: ...


@runtime_checkable
class EnvironmentManagerProtocol(Protocol):
    """Manages conda+uv environment for a specific simulator version."""

    simulator_name: str
    version: str

    def check_system_deps(self) -> list[str]: ...
    def install(self) -> None: ...
    def env_is_ready(self) -> bool: ...
    def get_python_executable(self) -> str: ...
    def get_env_name(self) -> str: ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Calls an LLM inference server."""

    def generate(self, messages: list[dict]) -> str: ...


@runtime_checkable
class AgentProtocol(Protocol):
    """An agent that decides actions given observations."""

    def reset(self) -> None: ...
    def act(self, observation: Observation, task_description: str) -> Action: ...
    def add_feedback(self, action_name: str, feedback: str) -> None: ...


@runtime_checkable
class TaskProtocol(Protocol):
    """A benchmark task that maps dataset episodes to simulator configs."""

    name: str
    simulator_key: str
    action_space: list[str]
    max_steps: int

    def download_dataset(self) -> Path: ...
    def load_episodes(self) -> list[dict]: ...
    def get_episode(self, index: int) -> dict: ...
    def format_reset_config(self, episode: dict) -> dict: ...
    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]: ...
    def get_bridge_script_path(self) -> Path | None: ...
    @property
    def simulator_configs(self) -> dict: ...
    @property
    def additional_deps(self) -> list[str]: ...
    @property
    def simulator_kwargs(self) -> dict: ...
    def __len__(self) -> int: ...
