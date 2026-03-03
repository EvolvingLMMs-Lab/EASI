"""LHPR-VLN task for EASI.

Adapts the LHPR-VLN benchmark to EASI's task interface.
Episodes are loaded from preprocessed JSONL files (data/val.jsonl,
data/test.jsonl) via EASI's default HuggingFace dataset loading.
Each episode contains 2-4 sequential navigation subtasks in HM3D scenes.

Metrics:
- Per-episode: task_success, oracle_success, spl, navigation_error,
  isr, csr, cgt, tar (computed per-episode for result.json)
- Aggregate: All 8 official metrics + contest_score
  (using vendored NavigationMetrics)
"""
from __future__ import annotations

import json
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.lhpr_vln.actions import get_action_space
from easi.tasks.lhpr_vln.vendor.metrics import NavigationMetrics
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class LHPRVLNTask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map LHPR-VLN episode dict to bridge reset config.

        Episode keys come from the preprocessed JSONL (lowercase):
            id, instruction, scene, robot, objects, regions, rooms,
            gt_steps, subtask_list, num_targets, batch
        """
        return {
            "episode_id": episode.get("id", "unknown"),
            "scene_id": episode["scene"],
            "robot": episode.get("robot", "spot"),
            "instruction": episode.get("instruction", ""),
            "targets": episode["objects"],
            "regions": episode["regions"],
            "gt_step": episode.get("gt_steps", []),
            "data_dir": episode.get("_data_dir", ""),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Compute per-episode metrics from subtask completion data.

        Reads serialized subtask arrays from the last step's info dict.
        Returns metrics for result.json AND data for aggregate_results().
        """
        if not trajectory:
            return self._empty_metrics()

        last = trajectory[-1]
        info = last.info

        # Parse subtask arrays from the last step's info
        successes = json.loads(info.get("subtask_successes", "[]"))
        oracle_successes = json.loads(info.get("subtask_oracle_successes", "[]"))
        nav_errors = json.loads(info.get("subtask_nav_errors", "[]"))
        nav_steps = json.loads(info.get("subtask_nav_steps", "[]"))
        gt_steps = json.loads(info.get("gt_steps", "[]"))
        gt_paths = json.loads(info.get("gt_paths", "[]"))

        # Overall success: all subtasks succeeded
        task_success = 1.0 if successes and all(s == 1 for s in successes) else 0.0
        oracle_success = 1.0 if oracle_successes and all(s == 1 for s in oracle_successes) else 0.0

        # SPL: success * (gt_total / max(gt_total, actual_total))
        total_gt = sum(gt_steps) if gt_steps else 0
        total_actual = sum(nav_steps) if nav_steps else 0
        spl = task_success * (total_gt / max(total_gt, total_actual)) if total_actual > 0 else 0.0

        # Navigation error: avg geodesic distance at stop across subtasks
        ne = sum(nav_errors) / len(nav_errors) if nav_errors else 0.0

        return {
            "task_success": task_success,
            "oracle_success": oracle_success,
            "spl": spl,
            "navigation_error": ne,
            "num_steps": float(len(trajectory)),
            "num_subtasks": float(len(successes)),
            "subtasks_completed": float(sum(successes)),
            # Store raw arrays as JSON strings for aggregate_results
            "_subtask_successes": json.dumps(successes),
            "_subtask_oracle_successes": json.dumps(oracle_successes),
            "_subtask_nav_errors": json.dumps(nav_errors),
            "_subtask_nav_steps": json.dumps(nav_steps),
            "_gt_steps": json.dumps(gt_steps),
            "_gt_paths": json.dumps(gt_paths),
        }

    def aggregate_results(
        self, records: list[EpisodeRecord]
    ) -> dict[str, float]:
        """Compute all 8 LHPR-VLN metrics using vendored NavigationMetrics.

        Metric definitions (from CVPR-25 paper):
        - SR: Success Rate (all subtasks completed)
        - OSR: Oracle Success Rate (agent ever passed within 1m of all targets)
        - SPL: Success weighted by Path Length
        - NE: Navigation Error (avg geodesic distance at stop)
        - ISR: Independent Success Rate (fraction of subtasks succeeded)
        - CSR: Conditional Success Rate (sequential dependency weighting)
        - CGT: CSR weighted by GT path length
        - TAR: Target Approach Rate (continuous approach measure)
        - contest_score: 0.4*TAR + 0.2*ISR + 0.2*CSR + 0.2*CGT
        """
        if not records:
            return {}

        metrics = NavigationMetrics()

        for r in records:
            er = r.episode_results
            successes = json.loads(er.get("_subtask_successes", "[]"))
            oracle_successes = json.loads(er.get("_subtask_oracle_successes", "[]"))
            nav_errors = json.loads(er.get("_subtask_nav_errors", "[]"))
            nav_steps = json.loads(er.get("_subtask_nav_steps", "[]"))
            gt_steps = json.loads(er.get("_gt_steps", "[]"))
            gt_paths = json.loads(er.get("_gt_paths", "[]"))

            # Overall success
            success = 1 if successes and all(s == 1 for s in successes) else 0
            oracle_success = 1 if oracle_successes and all(s == 1 for s in oracle_successes) else 0

            # Total steps
            total_gt = sum(gt_steps) if gt_steps else 0
            total_actual = sum(nav_steps) if nav_steps else 0

            # Avg navigation error
            avg_ne = sum(nav_errors) / len(nav_errors) if nav_errors else 0.0

            metrics.add_sample(
                success=success,
                gt_step=total_gt,
                path_step=total_actual,
                oracle_success=oracle_success,
                navigation_error=avg_ne,
                subtask_successes=successes,
                subtask_path_step=gt_steps,
                gt_length=gt_paths,
                error_length=nav_errors,
            )

        result = metrics.compute()

        # Add contest ranking score
        tar = result.get("tar", 0)
        isr = result.get("independent_success_rate", 0)
        csr = result.get("conditional_success_rate", 0)
        cgt = result.get("conditional_path_length", 0)
        contest_score = 0.4 * tar + 0.2 * isr + 0.2 * csr + 0.2 * cgt

        return {
            "SR": round(result["success_rate"], 4),
            "OSR": round(result["oracle_success_rate"], 4),
            "SPL": round(result["spl"], 4),
            "NE": round(result["navigation_error"], 4),
            "ISR": round(result["independent_success_rate"], 4),
            "CSR": round(result["conditional_success_rate"], 4),
            "CGT": round(result["conditional_path_length"], 4),
            "TAR": round(result["tar"], 4),
            "contest_score": round(contest_score, 4),
            "num_episodes": len(records),
            # Convenience alias for EASI dashboard
            "success_rate": round(result["success_rate"], 4),
        }

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "task_success": 0.0,
            "oracle_success": 0.0,
            "spl": 0.0,
            "navigation_error": 0.0,
            "num_steps": 0.0,
            "num_subtasks": 0.0,
            "subtasks_completed": 0.0,
            "_subtask_successes": "[]",
            "_subtask_oracle_successes": "[]",
            "_subtask_nav_errors": "[]",
            "_subtask_nav_steps": "[]",
            "_gt_steps": "[]",
            "_gt_paths": "[]",
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": "test_0",
                "instruction": "Find the chair in the living room, then find the table in the kitchen.",
                "scene": "00384-ceJTwFNjqCt",
                "robot": "spot",
                "objects": ["chair", "table"],
                "regions": ["3", "5"],
                "rooms": ["living room", "kitchen"],
                "gt_steps": [40, 55],
                "subtask_list": ["Move_to('chair_3')", "Move_to('table_5')"],
                "num_targets": 2,
                "batch": "builtin",
            },
        ]
