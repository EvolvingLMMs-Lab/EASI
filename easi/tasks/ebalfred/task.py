"""EB-Alfred task for EASI.

Adapts the EmbodiedBench EB-Alfred track to EASI's task interface.
Supports multiple splits via per-split .yaml configs (Task 4 infra).

When loaded via a split yaml (e.g., ebalfred_base), episodes come from
HuggingFace dataset rows. When loaded via get_task_yaml_path(), uses
ebalfred_base.yaml as the default fallback.

Reference: EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.tasks.ebalfred.actions import get_global_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBAlfredTask(BaseTask):

    def __init__(self, data_dir: Path | None = None, split_yaml_path: Path | None = None):
        super().__init__(data_dir=data_dir, split_yaml_path=split_yaml_path)
        # Override static action space with dynamic one
        self._config["action_space"] = get_global_action_space()

    def get_task_yaml_path(self) -> Path:
        # Decision #4: No task.yaml — use ebalfred_base.yaml as default fallback
        return Path(__file__).parent / "ebalfred_base.yaml"

    def get_bridge_script_path(self) -> Path:
        """Return path to the EB-Alfred-specific bridge script."""
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        """Decision #2: EB-Alfred uses 'instruction' field from HF row."""
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map EB-Alfred episode (HF row) to AI2-THOR bridge reset config.

        HF row columns: id, task, repeat_idx, instruction, task_type, trial_id
        The bridge uses task_path to find annotation files inside the extracted tasks.zip:
          <data_dir>/oscarqjh_EB-Alfred_easi/tasks/<task_path>/pp/ann_<repeat_idx>.json
        """
        data_dir = episode.get("_data_dir", "")
        return {
            "task_path": episode["task"],
            "repeat_idx": episode.get("repeat_idx", 0),
            "instruction": episode.get("instruction", ""),
            "episode_id": episode.get("id", ""),
            "task_type": episode.get("task_type", ""),
            "trial_id": episode.get("trial_id", ""),
            "data_dir": data_dir,
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from the trajectory.

        The bridge reports task_success and task_progress in StepResult.info,
        computed by EB-Alfred's goal_conditions_met() running inside the bridge.
        """
        if not trajectory:
            return {
                "task_success": 0.0,
                "task_progress": 0.0,
                "num_steps": 0.0,
                "total_reward": 0.0,
            }

        last_step = trajectory[-1]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "task_progress": last_step.info.get("task_progress", 0.0),
            "num_steps": float(len(trajectory)),
            "total_reward": sum(s.reward for s in trajectory),
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Return minimal built-in episodes for testing without dataset.

        Matches EB-Alfred_easi column structure: id, task, repeat_idx,
        instruction, task_type, trial_id.
        """
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
