"""EB-Navigation task for EASI.

Adapts the EmbodiedBench EB-Navigation benchmark to EASI's task interface.
Supports 5 splits via per-split .yaml configs.

Episode data flows from HF dataset -> task.format_reset_config() -> bridge
reset_config -> EBNavEnv.reset(episode).
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.tasks.ebnavigation.actions import get_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBNavigationTask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "ebnavigation_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map EB-Navigation HF row to bridge reset config."""
        return {
            "episode_id": episode.get("id", "unknown"),
            "scene": episode["scene"],
            "instruction": episode["instruction"],
            "target_object_type": episode.get("target_object_type", ""),
            "target_object_id": episode.get("target_object_id", ""),
            "target_position": episode["target_position"],
            "agent_pose": episode["agent_pose"],
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from trajectory.

        The bridge reports task_success and distance in StepResult.info.
        """
        if not trajectory:
            return {
                "task_success": 0.0,
                "distance_to_target": -1.0,
                "num_steps": 0.0,
            }

        last_step = trajectory[-1]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "distance_to_target": last_step.info.get("distance", -1.0),
            "num_steps": float(len(trajectory)),
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": 0,
                "scene": "FloorPlan11",
                "instruction": "navigate to the Bread in the room and be as close as possible to it",
                "target_object_type": "Bread",
                "target_object_id": "Bread|+01.30|+00.98|-01.53",
                "target_position": {"x": 1.3, "y": 0.98, "z": -1.53},
                "agent_pose": {
                    "position": {"x": -0.75, "y": 0.9009992, "z": -1.75},
                    "rotation": 90.0,
                    "horizon": 0.0,
                },
            },
        ]
