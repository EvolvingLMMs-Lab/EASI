"""EB-Alfred task for EASI.

Adapts the EmbodiedBench EB-Alfred track to EASI's task interface.
Supports multiple splits via per-split .yaml configs.

The vendor/ directory contains EBAlfEnv (Gym env) copied from EmbodiedBench.
The bridge wraps EBAlfEnv via BaseBridge, delegating all skill execution,
scene restoration, and goal evaluation to the vendor code.

Episode data flows from HF dataset → task.format_reset_config() → bridge
reset_config → EBAlfEnv.reset(episode). Each episode dict must have 'task',
'repeat_idx', and 'instruction' keys.
"""
from __future__ import annotations

import json
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.tasks.ebalfred.actions import get_global_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBAlfredTask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_global_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "ebalfred_base.yaml"

    def get_bridge_script_path(self) -> Path:
        """Return path to the EB-Alfred-specific bridge script."""
        return Path(__file__).parent / "bridge.py"

    def on_episode_reset(self, observation, agent) -> None:
        """Update agent action space from per-episode bridge metadata."""
        dynamic_actions_json = observation.metadata.get("dynamic_action_space")
        if dynamic_actions_json and hasattr(agent, 'update_action_space'):
            dynamic_actions = json.loads(dynamic_actions_json)
            agent.update_action_space(dynamic_actions)

    def get_instruction(self, episode: dict) -> str:
        """EB-Alfred uses 'instruction' field from HF row."""
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map EB-Alfred episode to bridge reset config.

        Passes the episode data directly so EBAlfEnv.reset() can load the
        task JSON and restore the scene. data_dir points to the HF dataset's
        tasks/ subdirectory containing the task JSON files.
        """
        data_dir = episode.get("_data_dir", "")
        return {
            "episode_id": episode.get("id", "unknown"),
            "task": episode["task"],
            "repeat_idx": episode["repeat_idx"],
            "instruction": episode["instruction"],
            "data_dir": str(Path(data_dir) / "tasks") if data_dir else "",
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from the trajectory.

        The bridge reports task_success and task_progress in StepResult.info,
        computed by EBAlfEnv's goal_conditions_met() running inside the bridge.
        """
        if not trajectory:
            return {
                "task_success": 0.0,
                "task_progress": 0.0,
                "num_steps": 0.0,
            }

        last_step = trajectory[-1]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "task_progress": last_step.info.get("task_progress", 0.0),
            "num_steps": float(len(trajectory)),
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Return minimal built-in episodes for testing without dataset."""
        return [
            {
                "id": 0,
                "task": "pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001",
                "repeat_idx": 0,
                "instruction": "Put a mug on the shelf.",
                "task_type": "pick_and_place_simple",
                "trial_id": "trial_T20190001",
            },
            {
                "id": 1,
                "task": "pick_clean_then_place_in_recep-Plate-None-CounterTop-2/trial_T20190002",
                "repeat_idx": 0,
                "instruction": "Rinse off a plate and put it on the counter.",
                "task_type": "pick_clean_then_place_in_recep",
                "trial_id": "trial_T20190002",
            },
        ]
