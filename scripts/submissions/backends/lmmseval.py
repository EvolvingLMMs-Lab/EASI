"""lmms-eval backend adapter.

Maps EASI benchmark keys to lmms-eval task names, builds accelerate/python
commands, detects completion from ``*_results.json``, and extracts scores
(including the special SiteBench merge logic).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .base import BackendAdapter, BenchmarkScores

# ---------------------------------------------------------------------------
# lmms-eval task name mapping
# ---------------------------------------------------------------------------

_TASK_MAP: dict[str, str] = {
    "vsi_bench": "vsibench_multiimage",
    "mmsi_bench": "mmsi_bench",
    "mindcube_tiny": "mindcube_tiny",
    "viewspatial": "viewspatial",
    "site_image": "site_bench_image",
    "site_video": "site_bench_video_multiimage",
    "blink": "blink",
    "3dsrbench": "3dsrbench_circular",
    "embspatial": "embspatial",
    "mmsi_video_bench": "mmsi_video_u50",
    "omnispatial_(manual_cot)": "omnispatial_test",
    "spar_bench": "sparbench",
    "vsi_debiased": "vsibench_debiased_multiimage",
}

# ---------------------------------------------------------------------------
# Metric extraction mapping
# ---------------------------------------------------------------------------

_METRIC_MAP: dict[str, dict] = {
    "vsi_bench": {
        "task_name": "vsibench_multiimage",
        "overall_metric": "vsibench_overall",
        "scale": 100,
        "sub_scores": {
            "obj_appearance_order_accuracy": "obj_appearance_order_accuracy",
            "object_abs_distance": "object_abs_distance_mra",
            "object_counting": "object_counting_mra",
            "object_rel_direction_accuracy": "object_rel_direction_accuracy",
            "object_rel_distance_accuracy": "object_rel_distance_accuracy",
            "object_size_estimation": "object_size_estimation_mra",
            "room_size_estimation": "room_size_estimation_mra",
            "route_planning_accuracy": "route_planning_accuracy",
        },
    },
    "mmsi_bench": {
        "task_name": "mmsi_bench",
        "overall_metric": "average",
        "scale": 100,
        "sub_scores": {
            "attr_appr_accuracy": "Attribute (Appr.)",
            "attr_meas_accuracy": "Attribute (Meas.)",
            "motion_cam_accuracy": "Motion (Cam.)",
            "motion_obj_accuracy": "Motion (Obj.)",
            "msr_accuracy": "MSR",
            "pos_cam_cam_accuracy": "Positional Relationship (Cam.\u2013Cam.)",
            "pos_cam_obj_accuracy": "Positional Relationship (Cam.\u2013Obj.)",
            "pos_cam_reg_accuracy": "Positional Relationship (Cam.\u2013Reg.)",
            "pos_obj_obj_accuracy": "Positional Relationship (Obj.\u2013Obj.)",
            "pos_obj_reg_accuracy": "Positional Relationship (Obj.\u2013Reg.)",
            "pos_reg_reg_accuracy": "Positional Relationship (Reg.\u2013Reg.)",
        },
    },
    "mindcube_tiny": {
        "task_name": "mindcube_tiny",
        "overall_metric": "overall_accuracy",
        "scale": 100,
        "sub_scores": {
            "among_accuracy": "among_accuracy",
            "around_accuracy": "around_accuracy",
            "rotation_accuracy": "rotation_accuracy",
        },
    },
    "viewspatial": {
        "task_name": "viewspatial",
        "overall_metric": "overall_accuracy",
        "scale": 100,
        "sub_scores": {
            "camera_perspective_object_view_orientation_accuracy": "camera_perspective_object_view_orientation_accuracy",
            "camera_perspective_relative_direction_accuracy": "camera_perspective_relative_direction_accuracy",
            "person_perspective_object_view_orientation_accuracy": "person_perspective_object_view_orientation_accuracy",
            "person_perspective_relative_direction_accuracy": "person_perspective_relative_direction_accuracy",
            "person_perspective_scene_simulation_relative_direction_accuracy": "person_perspective_scene_simulation_relative_direction_accuracy",
        },
    },
    "site": {
        "task_names": ["site_bench_image", "site_bench_video_multiimage"],
        "scale": 100,
        "sub_scores": {
            "3d_information_understanding_caa": "3d information understanding",
            "counting_and_existence_caa": "counting & existence",
            "movement_prediction_and_navigation_caa": "movement prediction & navigation",
            "multiview_and_crossimage_reasoning_caa": "multi-view & cross-image reasoning",
            "object_localization_and_positioning_caa": "object localization & positioning",
            "spatial_relationship_reasoning_caa": "spatial relationship reasoning",
        },
    },
    "blink": {
        "task_name": "blink",
        "overall_metric": "blink_acc",
        "scale": 100,
        "sub_scores": {},
    },
    "3dsrbench": {
        "task_name": "3dsrbench_circular",
        "overall_metric": "vanilla_accuracy",
        "scale": 100,
        "sub_scores": {
            "height_higher": "height_higher_accuracy",
            "location_above": "location_above_accuracy",
            "location_closer_to_camera": "location_closer_to_camera_accuracy",
            "location_next_to": "location_next_to_accuracy",
            "multi_object_closer_to": "multi_object_closer_to_accuracy",
            "multi_object_facing": "multi_object_facing_accuracy",
            "multi_object_parallel": "multi_object_parallel_accuracy",
            "multi_object_same_direction": "multi_object_same_direction_accuracy",
            "multi_object_viewpoint_towards_object": "multi_object_viewpoint_towards_object_accuracy",
            "orientation_in_front_of": "orientation_in_front_of_accuracy",
            "orientation_on_the_left": "orientation_on_the_left_accuracy",
            "orientation_viewpoint": "orientation_viewpoint_accuracy",
            # Circular eval scores
            "circ_eval_overall": "circular_accuracy",
        },
    },
    "embspatial": {
        "task_name": "embspatial",
        "overall_metric": "embspatial_acc",
        "scale": 100,
        "sub_scores": {
            "ai2thor_accuracy": "ai2thor_accuracy",
            "mp3d_accuracy": "mp3d_accuracy",
            "scannet_accuracy": "scannet_accuracy",
        },
    },
    # ---- EXTRA benchmarks ----
    "mmsi_video_bench": {
        "task_name": "mmsi_video_u50",
        "overall_metric": "overall_accuracy",
        "scale": 100,
        "sub_scores": {},
    },
    "omnispatial_(manual_cot)": {
        "task_name": "omnispatial_test",
        "overall_metric": "omnispatial",
        "scale": 100,
        "sub_scores": {},
    },
    "spar_bench": {
        "task_name": "sparbench",
        "overall_metric": "sparbench_score",
        "scale": 1,  # already returns 0-100
        "sub_scores": {},
    },
    "vsi_debiased": {
        "task_name": "vsibench_debiased_multiimage",
        "overall_metric": "vsibench_overall",
        "scale": 100,
        "sub_scores": {
            "obj_appearance_order_accuracy": "obj_appearance_order_accuracy",
            "object_abs_distance": "object_abs_distance_mra",
            "object_counting": "object_counting_mra",
            "object_rel_direction_accuracy": "object_rel_direction_accuracy",
            "object_rel_distance_accuracy": "object_rel_distance_accuracy",
            "object_size_estimation": "object_size_estimation_mra",
            "room_size_estimation": "room_size_estimation_mra",
            "route_planning_accuracy": "route_planning_accuracy",
        },
    },
}

# Subcategories used by SiteBench merge logic
_SITE_SUBCATEGORIES = {
    "3d information understanding",
    "counting & existence",
    "movement prediction & navigation",
    "multi-view & cross-image reasoning",
    "object localization & positioning",
    "spatial relationship reasoning",
}


# ---------------------------------------------------------------------------
# Helper: load and merge all *_results.json
# ---------------------------------------------------------------------------

def _load_all_results(model_dir: Path) -> dict[str, dict[str, object]]:
    """Load and merge all ``*_results.json`` files in *model_dir*.

    Files are sorted by name so that later (timestamped) files overwrite
    earlier ones for the same task.

    Returns:
        ``{task_name: {metric_key: value, ...}}`` with filter suffixes stripped.
    """
    merged: dict[str, dict[str, object]] = {}
    result_files = sorted(model_dir.glob("*_results.json"))

    for path in result_files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        results = data.get("results", {})
        for task_name, metrics in results.items():
            cleaned: dict[str, object] = {}
            for raw_key, value in metrics.items():
                # Strip filter suffixes like ",none" or ",flexible-extract"
                key = raw_key.split(",")[0]
                cleaned[key] = value
            merged[task_name] = cleaned

    return merged


# ---------------------------------------------------------------------------
# SiteBench merge (image + video JSONL)
# ---------------------------------------------------------------------------

def _empty_stats() -> dict[str, float]:
    return {"caa_num": 0.0, "caa_den": 0.0, "acc_num": 0.0, "acc_den": 0.0}


def _compute_stats_from_jsonl(jsonl_path: Path) -> dict:
    """Compute aggregated per-metric statistics from a samples JSONL file."""
    metric_stats: dict[str, dict[str, float]] = defaultdict(_empty_stats)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            acc = item.get("accuracy", {})
            caa = item.get("chance_adjusted_acc", {})

            acc_total = acc.get("total", 0.0)
            caa_total = caa.get("total", 0.0)

            for key, value in acc.items():
                if key == "total":
                    continue
                metric_stats[key]["acc_num"] += value
                metric_stats[key]["acc_den"] += acc_total

            for key, value in caa.items():
                if key == "total":
                    continue
                metric_stats[key]["caa_num"] += value
                metric_stats[key]["caa_den"] += caa_total

    overall = metric_stats.get("overall")
    return {
        "metric_stats": dict(metric_stats),
        "overall": overall,
    }


def _merge_stats(stats1: dict, stats2: dict) -> dict:
    """Merge two ``{key: {num/den}}`` dicts."""
    merged: dict[str, dict[str, float]] = defaultdict(_empty_stats)
    for src in (stats1, stats2):
        for key, val in src.items():
            merged[key]["acc_num"] += val["acc_num"]
            merged[key]["acc_den"] += val["acc_den"]
            merged[key]["caa_num"] += val["caa_num"]
            merged[key]["caa_den"] += val["caa_den"]
    return dict(merged)


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class LmmsEvalAdapter(BackendAdapter):
    """Backend adapter for lmms-eval."""

    TASK_MAP = _TASK_MAP

    def __init__(
        self,
        *,
        model_args: str = "",
        use_accelerate: bool = True,
        rerun: bool = False,
    ):
        self.model_args = model_args
        self.use_accelerate = use_accelerate
        self.rerun = rerun

    @property
    def name(self) -> str:
        return "lmmseval"

    def get_env_overrides(self) -> dict[str, str]:
        """Env vars for the subprocess.

        Disables ``hf_transfer`` which can cause filelock deadlocks with
        multi-process accelerate launches on shared HuggingFace caches.
        """
        return {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }

    # ---- Command building ----

    def build_cmd(
        self,
        model: str,
        benchmarks: dict[str, str],
        output_dir: Path,
        nproc: int,
        *,
        extra_args: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[str]:
        """Build the ``accelerate launch`` or ``python -m lmms_eval`` command."""
        task_names = list(benchmarks.values())

        if self.use_accelerate:
            port = random.randint(12300, 12400)
            cmd = [
                "accelerate", "launch",
                f"--num_processes={nproc}",
                "--num_machines=1",
                "--mixed_precision=no",
                "--dynamo_backend=no",
                f"--main_process_port={port}",
                "-m", "lmms_eval",
            ]
        else:
            cmd = ["python", "-m", "lmms_eval"]

        cmd += [
            "--model", model,
            f"--model_args={self.model_args}",
            "--tasks", ",".join(task_names),
            "--batch_size", "1",
            "--output_path", str(output_dir),
            "--log_samples",
        ]

        if verbose:
            cmd += ["--verbosity", "DEBUG"]

        if extra_args:
            cmd += extra_args

        return cmd

    # ---- Completion detection ----

    def detect_completion(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, bool]:
        """Check which benchmarks have results in ``*_results.json``."""
        merged = _load_all_results(model_dir)
        result: dict[str, bool] = {}
        for key, task_name in benchmarks.items():
            result[key] = task_name in merged
        return result

    def find_completed_tasks(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> set[str]:
        """Return set of already-completed benchmark keys (empty if rerun)."""
        if self.rerun:
            return set()
        return super().find_completed_tasks(model_dir, model_name, benchmarks)

    # ---- Result files ----

    def get_result_files(self, model_dir: Path, model_name: str) -> list[Path]:
        """Return only the latest result files per task for the archive.

        For results JSONs: include only the latest file that contributes
        each task (not stale files from previous runs).
        For sample JSONLs: include only the latest file per task name.
        This ensures the archive matches exactly the scores submitted.
        """
        files: list[Path] = []

        # Results JSONs: walk newest-first, track which tasks are covered.
        # Only include a file if it contributes at least one not-yet-covered task.
        covered_tasks: set[str] = set()
        for path in sorted(model_dir.glob("*_results.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                tasks_in_file = set(data.get("results", {}).keys())
            except (json.JSONDecodeError, OSError):
                continue
            new_tasks = tasks_in_file - covered_tasks
            if new_tasks:
                files.append(path)
                covered_tasks.update(new_tasks)

        # Sample JSONLs: latest per task name
        seen_tasks: set[str] = set()
        for path in sorted(model_dir.glob("*_samples_*.jsonl"), reverse=True):
            name = path.name
            samples_idx = name.find("_samples_")
            if samples_idx < 0:
                continue
            task_name = name[samples_idx + len("_samples_"):].removesuffix(".jsonl")
            if task_name not in seen_tasks:
                seen_tasks.add(task_name)
                files.append(path)

        return sorted(files)

    # ---- Score extraction ----

    def extract_scores(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, BenchmarkScores]:
        """Extract overall + sub-scores for each benchmark from results JSON."""
        merged = _load_all_results(model_dir)
        scores: dict[str, BenchmarkScores] = {}

        # Determine which metric-map keys to process
        keys_to_process: set[str] = set()
        for key in benchmarks:
            if key in ("site_image", "site_video"):
                keys_to_process.add("site")
            elif key in _METRIC_MAP:
                keys_to_process.add(key)

        for bench_key in sorted(keys_to_process):
            config = _METRIC_MAP[bench_key]

            # Special handling for site (combined image + video)
            if "task_names" in config:
                overall, subs = self._extract_site_scores(model_dir, config)
                scores[bench_key] = BenchmarkScores(overall=overall, sub_scores=subs)
                continue

            task_name = config["task_name"]
            task_metrics = merged.get(task_name)
            if task_metrics is None:
                scores[bench_key] = BenchmarkScores()
                continue

            scale = config.get("scale", 1)

            # Overall score
            overall_metric_name = config["overall_metric"]
            raw_overall = task_metrics.get(overall_metric_name)

            # When overall_is_dict, the metric value is a dict containing
            # both "overall" and the sub-scores (e.g. vsibench_score)
            overall_dict: dict | None = None
            if config.get("overall_is_dict") and isinstance(raw_overall, dict):
                overall_dict = raw_overall
                raw_overall = raw_overall.get("overall")

            overall: float | None = None
            if raw_overall is not None:
                try:
                    overall = round(float(raw_overall) * scale, 4)
                except (ValueError, TypeError):
                    pass

            # Sub-scores: look in task_metrics first, then in the overall dict
            sub_scores: dict[str, float | None] = {}
            for payload_key, metric_name in config["sub_scores"].items():
                val = task_metrics.get(metric_name)
                if val is None and overall_dict is not None:
                    val = overall_dict.get(metric_name)
                if val is not None:
                    try:
                        sub_scores[payload_key] = round(float(val) * scale, 4)
                    except (ValueError, TypeError):
                        sub_scores[payload_key] = None
                else:
                    sub_scores[payload_key] = None

            scores[bench_key] = BenchmarkScores(overall=overall, sub_scores=sub_scores)

        return scores

    # ---- SiteBench special merge ----

    def _extract_site_scores(
        self,
        model_dir: Path,
        config: dict,
    ) -> tuple[float | None, dict[str, float | None]]:
        """Combine image + video JSONL sample files to compute CAA scores.

        Uses the same algorithm as ``lmms_eval/tasks/sitebench/merge_results.py``.
        """
        # Find JSONL files
        image_files = sorted(model_dir.glob("*samples_site_bench_image.jsonl"), reverse=True)
        video_files = sorted(model_dir.glob("*samples_site_bench_video*.jsonl"), reverse=True)

        image_stats = None
        video_stats = None

        if image_files:
            image_stats = _compute_stats_from_jsonl(image_files[0])
        if video_files:
            video_stats = _compute_stats_from_jsonl(video_files[0])

        if image_stats is None and video_stats is None:
            return None, {}

        # Merge metric_stats from both
        if image_stats and video_stats:
            combined_metric = _merge_stats(
                image_stats.get("metric_stats", {}),
                video_stats.get("metric_stats", {}),
            )
            img_overall = image_stats.get("overall") or _empty_stats()
            vid_overall = video_stats.get("overall") or _empty_stats()
            combined_overall_stats = {
                "acc_num": img_overall["acc_num"] + vid_overall["acc_num"],
                "acc_den": img_overall["acc_den"] + vid_overall["acc_den"],
                "caa_num": img_overall["caa_num"] + vid_overall["caa_num"],
                "caa_den": img_overall["caa_den"] + vid_overall["caa_den"],
            }
        elif image_stats:
            combined_metric = image_stats.get("metric_stats", {})
            combined_overall_stats = image_stats.get("overall") or _empty_stats()
        else:
            combined_metric = video_stats.get("metric_stats", {})
            combined_overall_stats = video_stats.get("overall") or _empty_stats()

        scale = config.get("scale", 1)

        # Overall CAA
        overall: float | None = None
        if combined_overall_stats["caa_den"] > 0:
            overall = round(
                (combined_overall_stats["caa_num"] / combined_overall_stats["caa_den"]) * scale,
                4,
            )

        # Per-subcategory CAA
        sub_scores: dict[str, float | None] = {}
        for payload_key, category_name in config["sub_scores"].items():
            cat_stats = combined_metric.get(category_name)
            if cat_stats and cat_stats["caa_den"] > 0:
                sub_scores[payload_key] = round(
                    (cat_stats["caa_num"] / cat_stats["caa_den"]) * scale,
                    4,
                )
            else:
                sub_scores[payload_key] = None

        return overall, sub_scores
