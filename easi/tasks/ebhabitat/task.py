"""EB-Habitat task for EASI.

Adapts the EmbodiedBench EB-Habitat benchmark to EASI's task interface.
Supports 6 splits via per-split .yaml configs.

Episode data flows from HF dataset -> task.format_reset_config() -> bridge
reset_config -> EBHabEnv.reset(). The bridge handles episode loading
from pickle files; EASI provides instruction/metadata from JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import StepResult
from easi.tasks.ebhabitat.actions import get_placeholder_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBHabitatTask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_placeholder_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "ebhabitat_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def on_episode_reset(self, observation, agent) -> None:
        """Update agent action space from per-episode bridge metadata."""
        dynamic_actions_json = observation.metadata.get("dynamic_action_space")
        if dynamic_actions_json and hasattr(agent, 'update_action_space'):
            dynamic_actions = json.loads(dynamic_actions_json)
            agent.update_action_space(dynamic_actions)

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map EB-Habitat HF row to bridge reset config."""
        return {
            "episode_id": episode.get("id", "unknown"),
            "eval_set": self._config.get("dataset", {}).get("split", "base"),
            "instruction": episode.get("instruction", ""),
            "data_dir": episode.get("_data_dir", ""),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract metrics from trajectory."""
        if not trajectory:
            return {
                "task_success": 0.0,
                "task_progress": 0.0,
                "subgoal_reward": 0.0,
                "num_steps": 0.0,
            }

        last_step = trajectory[-1]
        return {
            "task_success": last_step.info.get("task_success", 0.0),
            "task_progress": last_step.info.get("task_progress", 0.0),
            "subgoal_reward": last_step.info.get("subgoal_reward", 0.0),
            "num_steps": float(len(trajectory)),
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": 0,
                "episode_id": "140",
                "instruction": "Find a toy airplane and move it to the right counter.",
                "instruct_id": "f0917e29",
                "scene_id": "data/replica_cad/configs/scenes/v3_sc3_staging_02.scene_instance.json",
            },
        ]
