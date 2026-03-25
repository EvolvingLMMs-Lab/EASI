"""Core data structures for observations, actions, and step results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """Observation produced by a simulator after reset or step."""

    rgb_path: str
    depth_path: str | None = None
    agent_pose: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Action:
    """An action to be executed in the simulator."""

    action_name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing one step in the simulator."""

    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """Bundles all data for one completed episode.

    Used by aggregate_results() to give the aggregation function
    access to both the raw episode data and the full trajectory.
    """

    episode: dict
    trajectory: list[StepResult]
    episode_results: dict[str, float]
