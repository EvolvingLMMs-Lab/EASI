"""Pydantic response schemas for structured LLM output.

Each task can define its own schema extending BaseResponseSchema.
The schema's get_actions() method normalizes task-specific fields
into a standard list of action strings.
"""
from __future__ import annotations

from pydantic import BaseModel


class BaseResponseSchema(BaseModel):
    """Base class for all LLM response schemas.

    Subclasses MUST implement get_actions() to normalize their
    task-specific fields into a list of action strings.
    """

    def get_actions(self) -> list[str]:
        raise NotImplementedError


class ExecutableAction(BaseModel):
    """A single action in an executable plan."""

    action: str


class ActionPlanResponse(BaseResponseSchema):
    """Default structured response for ReAct agents."""

    reasoning: str
    executable_plan: list[ExecutableAction]

    def get_actions(self) -> list[str]:
        return [a.action for a in self.executable_plan]
