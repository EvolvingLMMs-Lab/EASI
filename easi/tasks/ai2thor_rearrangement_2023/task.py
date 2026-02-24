"""AI2-THOR Rearrangement task for EASI.

Adapts the AI2-THOR Rearrangement Challenge to EASI's task interface.
Supports 5 splits via per-split YAML configs.
Computes all 6 paper metrics: SR, SRwD, PuSR, PuLen, SuLen, Len.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.ai2thor_rearrangement_2023.actions import get_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class AI2THORRearrangement2023Task(BaseTask):
    """AI2-THOR Rearrangement Challenge task."""

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "ai2thor_rearrangement_2023_val.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get(
            "instruction",
            "Rearrange objects to match the goal configuration.",
        )

    def format_reset_config(self, episode: dict) -> dict:
        """Convert HF dataset row to bridge reset config.

        Complex fields (poses, openable_data) stay as JSON strings;
        the bridge deserializes them.
        """
        return {
            "episode_id": episode.get("id", "unknown"),
            "scene": episode["scene"],
            "instruction": self.get_instruction(episode),
            "agent_position": episode["agent_position"],   # JSON string
            "agent_rotation": episode["agent_rotation"],    # int
            "starting_poses": episode["starting_poses"],    # JSON string
            "target_poses": episode["target_poses"],        # JSON string
            "openable_data": episode["openable_data"],      # JSON string
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract the 6 paper metrics from trajectory.

        The bridge reports rearrangement metrics in StepResult.info.
        Pickup metrics (PuSR, PuLen) are computed from the action log.
        """
        if not trajectory:
            return {
                "success": 0.0,
                "prop_fixed_strict": 0.0,
                "energy_prop": 1.0,
                "num_steps": 0.0,
                "pickup_success_rate": 0.0,
                "num_pickup_actions": 0.0,
                "success_length": float("nan"),
            }

        last_info = trajectory[-1].info or {}
        success = float(last_info.get("success", 0.0))
        num_steps = float(len(trajectory))

        # Compute PuSR and PuLen from action log in trajectory
        pickup_total = 0
        pickup_success = 0
        for step in trajectory:
            info = step.info or {}
            action_name = info.get("action_name", "")
            if action_name.startswith("pickup_"):
                pickup_total += 1
                if info.get("action_success", False):
                    pickup_success += 1

        pickup_success_rate = (
            pickup_success / pickup_total if pickup_total > 0 else 0.0
        )

        # SuLen: episode length IF successful, else NaN
        success_length = num_steps if success > 0.5 else float("nan")

        return {
            "success": success,
            "prop_fixed_strict": float(last_info.get("prop_fixed_strict", 0.0)),
            "energy_prop": float(last_info.get("energy_prop", 1.0)),
            "num_steps": num_steps,
            "pickup_success_rate": pickup_success_rate,
            "num_pickup_actions": float(pickup_total),
            "success_length": success_length,
        }

    # The 6 official paper metrics (+ num_steps for convenience)
    METRIC_KEYS = (
        "success",              # SR  — Episode Success %
        "prop_fixed_strict",    # SRwD — Ep-Success w/o Disturbance %
        "pickup_success_rate",  # PuSR — PickUp Success %
        "num_pickup_actions",   # PuLen — Ep-Len for PickUp
        "success_length",       # SuLen — Ep-Len for Success
        "num_steps",            # Len  — Ep-Len
    )

    def aggregate_results(
        self, records: list[EpisodeRecord]
    ) -> dict[str, float]:
        """Aggregate the 6 official paper metrics plus runtime stats.

        Handles NaN success_length (only averages successful episodes).
        """
        if not records:
            return {}

        # 1. Official paper metrics
        agg = {}
        for key in self.METRIC_KEYS:
            values = []
            for r in records:
                v = r.episode_results.get(key)
                if isinstance(v, (int, float)) and not math.isnan(v):
                    values.append(float(v))
            if values:
                agg[key] = sum(values) / len(values)
            else:
                agg[key] = float("nan")

        # 2. Runtime stats
        elapsed_values = []
        llm_call_counts = []
        for r in records:
            elapsed = r.episode_results.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                elapsed_values.append(float(elapsed))

            usage = r.episode_results.get("llm_usage")
            if isinstance(usage, dict):
                calls = usage.get("total_calls", 0)
                llm_call_counts.append(float(calls))

        if elapsed_values:
            agg["avg_elapsed_seconds"] = sum(elapsed_values) / len(elapsed_values)
            agg["total_elapsed_seconds"] = sum(elapsed_values)
        if llm_call_counts:
            agg["avg_llm_calls"] = sum(llm_call_counts) / len(llm_call_counts)

        return agg

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": "FloorPlan21__smoke_test",
                "scene": "FloorPlan21",
                "agent_position": json.dumps(
                    {"x": -1.0, "y": 0.87, "z": -1.0}
                ),
                "agent_rotation": 0,
                "starting_poses": json.dumps([]),
                "target_poses": json.dumps([]),
                "openable_data": json.dumps([]),
                "instruction": "Smoke test — no objects to rearrange.",
            },
        ]
