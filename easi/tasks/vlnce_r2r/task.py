"""VLN-CE R2R task for EASI.

Adapts VLN-CE Room-to-Room benchmark to EASI's task interface.
Episodes loaded from HuggingFace dataset (oscarqjh/VLN-CE-R2R_easi).

Metrics (aligned with original VLN-CE):
- Per-episode: success, oracle_success, spl, navigation_error, ndtw, sdtw,
  path_length, steps_taken (null for test split except path_length/steps_taken)
- Aggregate: Average of non-null values per metric
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.vlnce_r2r.actions import get_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class VLNCETask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def _load_episodes_from_hf(self, dataset_config: dict) -> list[dict]:
        """Load episodes from HF dataset with custom split names.

        HF auto-detection merges val_seen.jsonl and val_unseen.jsonl into
        a single 'validation' split. We bypass this by loading the specific
        JSONL file directly using data_files parameter.
        """
        from datasets import load_dataset
        from easi.core.base_task import hf_row_to_episode

        data_dir = self.download_dataset()
        split_name = dataset_config.get("split")
        data_file = str(data_dir / "data" / f"{split_name}.jsonl")

        logger.info("Loading episodes from %s (split=%s)", data_file, split_name)

        hf_cache = Path(tempfile.gettempdir()) / "easi_hf_cache"
        ds = load_dataset(
            "json", data_files=data_file, split="train",
            cache_dir=str(hf_cache),
        )
        episodes = [hf_row_to_episode(row) for row in ds]

        for ep in episodes:
            ep["_data_dir"] = str(data_dir)

        logger.info("Loaded %d episodes (split=%s)", len(episodes), split_name)
        return episodes

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map HF episode dict to bridge reset config."""
        gt_locations = episode.get("gt_locations")
        if isinstance(gt_locations, list):
            gt_locations = json.dumps(gt_locations)

        return {
            "episode_id": str(episode.get("episode_id", episode.get("id", "unknown"))),
            "scene_id": episode["scene_id"],
            "instruction": episode.get("instruction", ""),
            "start_position": json.dumps(episode["start_position"]),
            "start_rotation": json.dumps(episode["start_rotation"]),
            "goal_position": json.dumps(episode["goal_position"]) if episode.get("goal_position") else "null",
            "geodesic_distance": str(episode["geodesic_distance"]) if episode.get("geodesic_distance") is not None else "null",
            "gt_locations": gt_locations if gt_locations else "null",
            "data_dir": episode.get("_data_dir", ""),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Extract per-episode metrics from bridge info dict."""
        if not trajectory:
            return self._empty_metrics()

        last = trajectory[-1]
        info = last.info

        return {
            "success": self._parse_nullable(info.get("success")),
            "oracle_success": self._parse_nullable(info.get("oracle_success")),
            "spl": self._parse_nullable(info.get("spl")),
            "navigation_error": self._parse_nullable(info.get("navigation_error")),
            "ndtw": self._parse_nullable(info.get("ndtw")),
            "sdtw": self._parse_nullable(info.get("sdtw")),
            "path_length": float(info.get("path_length", 0.0)),
            "steps_taken": float(len(trajectory)),
        }

    def aggregate_results(self, records: list[EpisodeRecord]) -> dict:
        """Average non-null values per metric across all episodes."""
        if not records:
            return {}

        metric_keys = [
            "success", "oracle_success", "spl", "navigation_error",
            "ndtw", "sdtw", "path_length", "steps_taken",
        ]

        result = {"num_episodes": len(records)}
        null_counts = {}

        for key in metric_keys:
            values = []
            nulls = 0
            for r in records:
                val = r.episode_results.get(key)
                if val is None or val == "null":
                    nulls += 1
                else:
                    values.append(float(val))
            if values:
                result[key] = round(sum(values) / len(values), 4)
            else:
                result[key] = None
            if nulls > 0:
                null_counts[key] = nulls

        if null_counts:
            logger.info(
                "Null metric counts: %s",
                ", ".join(f"{k}={v}" for k, v in null_counts.items()),
            )
            result["_null_counts"] = null_counts

        # Aliases for consistency with VLN-CE conventions
        result["SR"] = result.get("success")
        result["SPL"] = result.get("spl")
        result["NE"] = result.get("navigation_error")
        result["Oracle_SR"] = result.get("oracle_success")
        result["NDTW"] = result.get("ndtw")
        result["SDTW"] = result.get("sdtw")

        return result

    @staticmethod
    def _parse_nullable(value):
        """Parse a value that may be None or 'null' string from IPC."""
        if value is None or value == "null":
            return None
        return float(value)

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "success": None,
            "oracle_success": None,
            "spl": None,
            "navigation_error": None,
            "ndtw": None,
            "sdtw": None,
            "path_length": 0.0,
            "steps_taken": 0.0,
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episode for testing without dataset download."""
        return [
            {
                "id": 0,
                "episode_id": "test_0",
                "scene_id": "17DRP5sb8fy",
                "instruction": "Walk down the hallway and turn right into the bedroom.",
                "start_position": [3.1678, 0.0711, -0.2367],
                "start_rotation": [0, 0.707, 0, 0.707],
                "goal_position": [4.5, 0.0711, 1.2],
                "goal_radius": 3.0,
                "geodesic_distance": 5.2,
                "reference_path": "[]",
                "gt_locations": None,
                "gt_actions": None,
            },
        ]
