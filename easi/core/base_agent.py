"""Abstract base class for agents."""
from __future__ import annotations

from abc import ABC, abstractmethod

from easi.core.episode import Action, Observation
from easi.core.protocols import LLMClientProtocol
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base for agents that bridge LLM inference and simulator actions."""

    def __init__(self, llm_client: LLMClientProtocol | None, action_space: list[str]):
        self.llm_client = llm_client
        self.action_space = action_space
        self._step_count: int = 0

    @abstractmethod
    def act(self, observation: Observation, task_description: str) -> Action:
        """Return the next action given current observation and task."""
        ...

    def add_feedback(self, action_name: str, feedback: str) -> None:
        """Record action feedback from the environment.

        Default: no-op. Subclasses (e.g., ReActAgent) override to track
        action history and clear action buffer on failure.
        """

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._step_count = 0
