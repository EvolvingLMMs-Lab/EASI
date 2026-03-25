"""HAZARD task for EASI.

Adapts the HAZARD benchmark (fire/flood/wind disaster rescue) to EASI's
task interface. Supports 3 scenarios via per-scenario YAML configs.

Episode data flows from HF dataset -> task.format_reset_config() -> bridge
-> HAZARD env (via LogPlayback scene reconstruction).

Scoring: value_saved / max_possible_value.
"""
from __future__ import annotations

import json
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.hazard.actions import get_action_space, HIGH_VALUE, LOW_VALUE
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Load vendored metadata for scoring
_CONFIG_DIR = Path(__file__).parent / "config"


def _load_json(name: str) -> dict:
    return json.loads((_CONFIG_DIR / name).read_text())


class HAZARDTask(BaseTask):

    def __init__(self, data_dir=None, split_yaml_path=None):
        super().__init__(data_dir=data_dir, split_yaml_path=split_yaml_path)
        self._scenario = self._config.get("scenario", "fire")
        self._value_dict = _load_json("value.json")
        self._fire_dict = _load_json("fire.json")
        self._fluid_dict = _load_json("fluid.json")

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "hazard_fire.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def format_reset_config(self, episode: dict) -> dict:
        """Map HAZARD HF row to bridge reset config.

        Computes absolute path to episode's source directory
        (extracted from simulator_data.zip in the HF dataset).
        """
        data_dir = episode.get("_data_dir", "")
        source_dir = episode["source_dir"]
        # simulator_data.zip extracts to simulator_data/ within data_dir
        absolute_source = str(Path(data_dir) / "simulator_data" / source_dir)

        # Parse JSON string fields
        hazard_params = episode.get("hazard_params", "{}")
        if isinstance(hazard_params, str):
            hazard_params = json.loads(hazard_params)

        objects = episode.get("objects", "[]")
        if isinstance(objects, str):
            objects = json.loads(objects)

        return {
            "episode_id": episode.get("episode_id", str(episode.get("id", "unknown"))),
            "source_dir": absolute_source,
            "scene_name": episode["scene_name"],
            "instruction": episode["instruction"],
            "task": episode["task"],
            "target_categories": episode["target_categories"],
            "target_object_ids": episode["target_object_ids"],
            "agent_position": episode["agent_position"],
            "hazard_params": hazard_params,
            "containers": episode.get("containers", []),
            "objects": objects,
            "max_steps": episode.get("max_steps", self.max_steps),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Compute HAZARD metrics from trajectory.

        Two primary metrics from the HAZARD paper (ICLR 2024):
        1. Value (Average Value Rate): sum(base_value * discount) / sum(all_target_values)
           - discount = 1.0 if object undamaged, 0.5 if damaged at rescue time
        2. Damage (Damage Rate): count(rescued_damaged) / count(total_rescued)
           - Lower is better. What fraction of rescued objects were already degraded.

        The bridge reports per-step: value_score, max_value, rescued_count, damaged_count.
        """
        if not trajectory:
            return {
                "value_score": 0.0,
                "max_value": self._compute_max_value(episode),
                "value_rate": 0.0,
                "damage_rate": 0.0,
                "rescued_count": 0.0,
                "damaged_count": 0.0,
                "targets_rescued": 0.0,
                "targets_total": float(len(episode.get("target_object_ids", []))),
                "num_steps": 0.0,
                "max_rescue_frame": 0.0,
            }

        last_step = trajectory[-1]
        value_score = float(last_step.info.get("value_score", 0.0))
        max_value = float(last_step.info.get("max_value", 0.0)) or self._compute_max_value(episode)
        rescued_count = float(last_step.info.get("rescued_count", 0.0))
        damaged_count = float(last_step.info.get("damaged_count", 0.0))

        # Value rate: weighted fraction of value saved (paper metric "Value")
        value_rate = value_score / max(max_value, 1.0)

        # Damage rate: fraction of rescued objects that were damaged (paper metric "Damage")
        damage_rate = damaged_count / max(rescued_count, 1.0)

        # Max rescue frame: latest frame at which any target was rescued
        # Used for paper "Step" metric (sum_max_rescue_frame / sum_rescued)
        # Bridge computes this as max(target_status.values()) -- the actual
        # latest rescue frame, NOT the total episode frame count.
        max_rescue_frame = float(last_step.info.get("max_rescue_frame", 0.0))

        return {
            "value_score": value_score,
            "max_value": max_value,
            "value_rate": value_rate,            # Paper "Value" metric
            "damage_rate": damage_rate,          # Paper "Damage" metric (lower = better)
            "rescued_count": rescued_count,
            "damaged_count": damaged_count,
            "targets_rescued": float(last_step.info.get("targets_rescued", 0.0)),
            "targets_total": float(last_step.info.get("targets_total", 0.0)),
            "num_steps": float(len(trajectory)),
            "max_rescue_frame": max_rescue_frame,  # For paper "Step" metric
        }

    def aggregate_results(
        self, records: list[EpisodeRecord]
    ) -> dict[str, float]:
        """HAZARD-specific aggregation matching the paper's three metrics.

        From HAZARD paper (ICLR 2024) and calc_value.py:
        - Value (Avg Value Rate): average of per-episode value_rate
        - Damage (Damage Rate): cumulative damaged/rescued across ALL episodes
        - Step (Avg Step): sum(max_rescue_frame) / sum(rescued) across ALL episodes
        """
        if not records:
            return {
                "value_rate": 0.0,
                "damage_rate": 0.0,
                "avg_step": 0.0,
                "avg_llm_steps": 0.0,
                "num_episodes": 0.0,
            }

        n = len(records)

        # Value: average per-episode value_rate (paper "Average value rate")
        value_rates = [r.episode_results.get("value_rate", 0.0) for r in records]
        avg_value_rate = sum(value_rates) / n

        # Damage: cumulative across all episodes (paper "Damage rate")
        total_rescued = sum(r.episode_results.get("rescued_count", 0.0) for r in records)
        total_damaged = sum(r.episode_results.get("damaged_count", 0.0) for r in records)
        damage_rate = total_damaged / max(total_rescued, 1.0)

        # Step (paper metric): sum(max_rescue_frame) / sum(rescued) across ALL episodes
        # This matches calc_value.py: "Average step" = sum_step / sum_picked
        # where step = max rescue frame per episode (latest frame at which a target was rescued)
        total_max_rescue_frame = sum(
            r.episode_results.get("max_rescue_frame", 0.0) for r in records
        )
        avg_step = total_max_rescue_frame / max(total_rescued, 1.0)

        # Also report avg LLM steps (EASI-native metric, more interpretable)
        avg_llm_steps = sum(r.episode_results.get("num_steps", 0.0) for r in records) / n

        return {
            # Paper primary metrics
            "value_rate": avg_value_rate,       # Paper "Value" -- higher is better
            "damage_rate": damage_rate,          # Paper "Damage" -- lower is better
            "avg_step": avg_step,                # Paper "Step" -- lower is better
            # EASI metric
            "avg_llm_steps": avg_llm_steps,      # Average LLM decisions per episode
            # Supporting detail
            "avg_value_score": sum(r.episode_results.get("value_score", 0.0) for r in records) / n,
            "avg_max_value": sum(r.episode_results.get("max_value", 0.0) for r in records) / n,
            "total_rescued": total_rescued,
            "total_damaged": total_damaged,
            "avg_targets_rescued": sum(r.episode_results.get("targets_rescued", 0.0) for r in records) / n,
            "avg_targets_total": sum(r.episode_results.get("targets_total", 0.0) for r in records) / n,
            "num_episodes": float(n),
        }

    def _compute_max_value(self, episode: dict) -> float:
        """Compute maximum possible value for an episode's targets."""
        target_categories = episode.get("target_categories", [])
        # Count number of target object IDs per category
        total = 0.0
        for category in target_categories:
            value = HIGH_VALUE if self._value_dict.get(category) == 1 else LOW_VALUE
            # Count how many target objects of this category
            # (approximation -- each target ID is one object)
            total += value
        # More precise: count by target_object_ids
        n_targets = len(episode.get("target_object_ids", []))
        if n_targets > 0 and target_categories:
            # Rough: average value * count
            avg_value = total / len(target_categories) if target_categories else LOW_VALUE
            total = avg_value * n_targets
        return total

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": 0,
                "episode_id": "test_episode_0",
                "task": "fire",
                "instruction": "Rescue the following objects from the fire: hairbrush.",
                "scene_name": "mm_kitchen_3a",
                "agent_position": [-0.25, 0.0, 0.75],
                "target_categories": ["hairbrush"],
                "target_object_ids": [8265913],
                "hazard_params": '{"fire_positions": [[0.5, 0.0, 2.25]]}',
                "containers": [],
                "objects": '[]',
                "max_steps": 1500,
                "source_dir": "room_setup_fire/test_set/mm_kitchen_3a-1",
            },
        ]
