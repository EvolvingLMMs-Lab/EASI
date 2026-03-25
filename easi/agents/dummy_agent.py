"""Dummy agent for testing — returns random actions without calling an LLM."""
from __future__ import annotations

import random

from easi.core.base_agent import BaseAgent
from easi.core.episode import Action, Observation


class DummyAgent(BaseAgent):
    """Agent that picks random actions from the action space.

    Does not call the LLM client. Useful for testing the full pipeline
    without needing a running LLM server.
    """

    def __init__(self, action_space: list[str], seed: int | None = None):
        super().__init__(llm_client=None, action_space=action_space)
        self._rng = random.Random(seed)

    def act(self, observation: Observation, task_description: str) -> Action:
        """Pick a random action from the action space."""
        self._step_count += 1
        return Action(action_name=self._rng.choice(self.action_space))
