"""EB-Manipulation task for EASI.

Adapts the EmbodiedBench EB-Manipulation benchmark to EASI's task interface.
Supports 5 splits via per-split YAML configs.

Key difference from other EB-* tasks: actions are 7D discrete gripper arrays,
not text-based named actions. The action_space is empty -- the prompt builder
handles action formatting.
"""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.tasks.ebmanipulation.actions import get_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBManipulationTask(BaseTask):

    def __init__(self, data_dir=None, split_yaml_path=None):
        super().__init__(data_dir=data_dir, split_yaml_path=split_yaml_path)
        # EB-Manipulation has no fixed action space -- actions are 7D arrays
        self._config["action_space"] = get_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "ebmanipulation_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map EB-Manipulation HF episode row to bridge reset config.

        The bridge constructs full file paths from data_dir + episode metadata.
        data_dir is injected by BaseTask._load_episodes_from_hf() and points
        to the extracted HF dataset cache containing simulator_data.zip contents.
        """
        data_dir = episode.get("_data_dir", "")
        split = self._config.get("dataset", {}).get("split", "base")
        return {
            "episode_id": episode.get("id", "unknown"),
            "data_dir": data_dir,
            "split": split,
            "task_name": episode["task_name"],
            "variation": episode.get("variation", 0),
            "episode_num": episode.get("episode_num", 0),
            "instruction": episode.get("instruction", ""),
            "task_type": episode.get("task_type", ""),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from trajectory.

        The bridge reports task_success and action_success in StepResult.info.
        """
        if not trajectory:
            return {
                "task_success": 0.0,
                "num_steps": 0.0,
                "action_success_rate": 0.0,
            }

        last_step = trajectory[-1]
        action_successes = [
            s.info.get("action_success", 0.0) for s in trajectory
        ]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "num_steps": float(len(trajectory)),
            "action_success_rate": (
                sum(action_successes) / len(action_successes)
                if action_successes
                else 0.0
            ),
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download.

        Uses the HF episode index format (task_name + variation + episode_num).
        The bridge constructs file paths from these fields + data_dir.
        """
        return [
            {
                "id": 0,
                "task_name": "pick_cube_shape",
                "variation": 0,
                "episode_num": 0,
                "instruction": "Pick up the star and place it into the yellow container.",
                "task_type": "pick",
            },
        ]
