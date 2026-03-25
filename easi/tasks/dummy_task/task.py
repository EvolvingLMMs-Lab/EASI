"""Dummy task for testing the pipeline end-to-end.

Provides built-in episodes that require no dataset download.
Evaluates success based on whether the agent sent a Stop action.
"""

from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult


DUMMY_ACTIONS = ["MoveAhead", "TurnLeft", "TurnRight", "Stop"]


class DummyTask(BaseTask):
    """A dummy task for testing the evaluation pipeline."""

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "task.yaml"

    def _build_action_space(self) -> list[str]:
        return list(DUMMY_ACTIONS)

    def format_reset_config(self, episode: dict) -> dict:
        """Map a dummy episode to simulator reset config.

        The dummy simulator doesn't need much — just pass through the scene_id.
        """
        return {
            "scene_id": episode.get("scene_id", "dummy_scene"),
            "agent_position": episode.get("start_position", [0.0, 0.0, 0.0]),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Evaluate a completed episode.

        Success criteria: agent sent Stop action within max_steps, OR
        the simulator reported done=True in the last step.
        """
        if not trajectory:
            return {"success": 0.0, "num_steps": 0.0}

        last_step = trajectory[-1]
        success = 1.0 if last_step.done else 0.0
        total_reward = sum(step.reward for step in trajectory)

        return {
            "success": success,
            "num_steps": float(len(trajectory)),
            "total_reward": total_reward,
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Return built-in dummy episodes for testing."""
        return [
            {
                "episode_id": "dummy_ep_001",
                "scene_id": "dummy_scene_A",
                "start_position": [0.0, 0.0, 0.0],
                "goal_position": [5.0, 0.0, 0.0],
                "task_description": "Navigate to the goal position.",
            },
            {
                "episode_id": "dummy_ep_002",
                "scene_id": "dummy_scene_B",
                "start_position": [1.0, 0.0, 1.0],
                "goal_position": [3.0, 0.0, -2.0],
                "task_description": "Find and go to the red cube.",
            },
            {
                "episode_id": "dummy_ep_003",
                "scene_id": "dummy_scene_A",
                "start_position": [2.0, 0.0, -1.0],
                "goal_position": [0.0, 0.0, 0.0],
                "task_description": "Return to the starting area.",
            },
        ]
