"""Agent memory: shared state between agent and prompt builder."""
from __future__ import annotations

from dataclasses import dataclass, field

from easi.core.episode import Action, Observation


@dataclass
class StepRecord:
    """Record of a single agent step."""

    observation: Observation
    action: Action | None = None
    feedback: str | None = None
    llm_response: str | None = None  # None for buffered actions
    step_number: int = 0


@dataclass
class AgentMemory:
    """Shared state that the agent populates and the prompt builder reads."""

    task_description: str = ""
    action_space: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    current_observation: Observation | None = None

    @property
    def is_first_turn(self) -> bool:
        """True when no completed steps exist yet."""
        return len(self.steps) == 0

    @property
    def action_history(self) -> list[tuple[str, str]]:
        """(action_name, feedback) for completed steps with feedback."""
        return [
            (s.action.action_name, s.feedback)
            for s in self.steps
            if s.action and s.feedback is not None
        ]

    def record_step(
        self,
        observation: Observation,
        action: Action,
        llm_response: str | None,
    ) -> None:
        """Record a completed step (action taken, awaiting feedback)."""
        self.steps.append(
            StepRecord(
                observation=observation,
                action=action,
                llm_response=llm_response,
                step_number=len(self.steps),
            )
        )

    def record_feedback(self, feedback: str) -> None:
        """Attach feedback to the most recent step."""
        if self.steps:
            self.steps[-1].feedback = feedback

    def clear(self) -> None:
        """Reset memory for a new episode."""
        self.steps.clear()
        self.current_observation = None
        self.task_description = ""
