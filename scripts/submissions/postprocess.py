"""Post-processing utilities for EASI benchmark results.

Builds the submission payload (``easi_results.json``) and results archive
(``easi_results.zip``) from VLMEvalKit output files.

Usage (as a library):
    from postprocess import build_payload, build_results_archive

    payload = build_payload(model_dir, model_name, benchmarks, configs)
    zip_path = build_results_archive(model_dir, model_name, output_dir)
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Metric name mapping: payload field -> CSV column name
# ---------------------------------------------------------------------------

METRIC_MAP: dict[str, dict] = {
    "vsi_bench": {
        "data_name": "VSI-Bench_32frame",
        "acc_pattern": "*_VSI-Bench_32frame*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "obj_appearance_order_accuracy": "obj_appearance_order_accuracy",
            "object_abs_distance": "object_abs_distance_MRA:.5:.95:.05",
            "object_counting": "object_counting_MRA:.5:.95:.05",
            "object_rel_direction_accuracy": "object_rel_direction_accuracy",
            "object_rel_distance_accuracy": "object_rel_distance_accuracy",
            "object_size_estimation": "object_size_estimation_MRA:.5:.95:.05",
            "room_size_estimation": "room_size_estimation_MRA:.5:.95:.05",
            "route_planning_accuracy": "route_planning_accuracy",
        },
    },
    "mmsi_bench": {
        "data_name": "MMSIBench_wo_circular",
        "acc_pattern": "*_MMSIBench_wo_circular*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "attr_appr_accuracy": "Attr-Appr_accuracy",
            "attr_meas_accuracy": "Attr-Meas_accuracy",
            "motion_cam_accuracy": "Motion-Cam_accuracy",
            "motion_obj_accuracy": "Motion-Obj_accuracy",
            "msr_accuracy": "MSR_accuracy",
            "pos_cam_cam_accuracy": "Pos-Cam-Cam_accuracy",
            "pos_cam_obj_accuracy": "Pos-Cam-Obj_accuracy",
            "pos_cam_reg_accuracy": "Pos-Cam-Reg_accuracy",
            "pos_obj_obj_accuracy": "Pos-Obj-Obj_accuracy",
            "pos_obj_reg_accuracy": "Pos-Obj-Reg_accuracy",
            "pos_reg_reg_accuracy": "Pos-Reg-Reg_accuracy",
        },
    },
    "mindcube_tiny": {
        "data_name": "MindCubeBench_tiny_raw_qa",
        "acc_pattern": "*_MindCubeBench_tiny_raw_qa*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "among_accuracy": "among_accuracy",
            "around_accuracy": "around_accuracy",
            "rotation_accuracy": "rotation_accuracy",
        },
    },
    "viewspatial": {
        "data_name": "ViewSpatialBench",
        "acc_pattern": "*_ViewSpatialBench*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "camera_perspective_object_view_orientation_accuracy": "Camera perspective - Object View Orientation_accuracy",
            "camera_perspective_relative_direction_accuracy": "Camera perspective - Relative Direction_accuracy",
            "person_perspective_object_view_orientation_accuracy": "Person perspective - Object View Orientation_accuracy",
            "person_perspective_relative_direction_accuracy": "Person perspective - Relative Direction_accuracy",
            "person_perspective_scene_simulation_relative_direction_accuracy": "Person perspective - Scene Simulation Relative Direction_accuracy",
        },
    },
    "site": {
        # Special: combined via scoring.score_sitebench()
        # No overall_key — _extract_site_scores handles its own key structure
        "data_names": ["SiteBenchImage", "SiteBenchVideo_32frame"],
        "scale": 100,  # score_sitebench() returns 0-1 fractions
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
        "data_name": "BLINK",
        "acc_pattern": "*_BLINK_acc.csv",
        "overall_key": "Overall",
        "scale": 100,
        "sub_scores": {
            "art_style": "Art_Style",
            "counting": "Counting",
            "forensic_detection": "Forensic_Detection",
            "functional_correspondence": "Functional_Correspondence",
            "iq_test": "IQ_Test",
            "jigsaw": "Jigsaw",
            "multi_view_reasoning": "Multi-view_Reasoning",
            "object_localization": "Object_Localization",
            "relative_depth": "Relative_Depth",
            "relative_reflectance": "Relative_Reflectance",
            "semantic_correspondence": "Semantic_Correspondence",
            "spatial_relation": "Spatial_Relation",
            "visual_correspondence": "Visual_Correspondence",
            "visual_similarity": "Visual_Similarity",
        },
    },
    "3dsrbench": {
        "data_name": "3DSRBench",
        "acc_pattern": "*_3DSRBench_full_acc.csv",
        "overall_key": "Overall",
        "scale": 100,
        "settings": ["vanilla", "circ_eval"],  # extract both settings
        "sub_scores": {
            "height_higher": "height_higher",
            "location_above": "location_above",
            "location_closer_to_camera": "location_closer_to_camera",
            "location_next_to": "location_next_to",
            "multi_object_closer_to": "multi_object_closer_to",
            "multi_object_facing": "multi_object_facing",
            "multi_object_parallel": "multi_object_parallel",
            "multi_object_same_direction": "multi_object_same_direction",
            "multi_object_viewpoint_towards_object": "multi_object_viewpoint_towards_object",
            "orientation_in_front_of": "orientation_in_front_of",
            "orientation_on_the_left": "orientation_on_the_left",
            "orientation_viewpoint": "orientation_viewpoint",
        },
    },
    "embspatial": {
        "data_name": "EmbSpatialBench",
        "acc_pattern": "*_EmbSpatialBench*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "ai2thor_accuracy": "ai2thor_accuracy",
            "mp3d_accuracy": "mp3d_accuracy",
            "scannet_accuracy": "scannet_accuracy",
        },
    },
    # ---- EXTRA benchmarks ----
    "mmsi_video_bench": {
        "data_name": "MMSIVideoBench_50frame",
        "acc_pattern": "*_MMSIVideoBench_50frame*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "cross_video_accuracy": "cross_video_accuracy",
            "motion_understanding_accuracy": "motion_understanding_accuracy",
            "planning_accuracy": "planning_accuracy",
            "prediction_accuracy": "prediction_accuracy",
            "spatial_construction_accuracy": "spatial_construction_accuracy",
        },
    },
    "omnispatial_(manual_cot)": {
        "data_name": "OmniSpatialBench_manual_cot",
        "acc_pattern": "*_OmniSpatialBench_manual_cot*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "allocentric_accuracy": "sub_task_type.Allocentric_accuracy",
            "egocentric_accuracy": "sub_task_type.Egocentric_accuracy",
            "geometric_reasoning_accuracy": "sub_task_type.Geometric_Reasoning_accuracy",
            "geospatial_strategy_accuracy": "sub_task_type.Geospatial_Strategy_accuracy",
            "hypothetical_accuracy": "sub_task_type.Hypothetical_accuracy",
            "localization_accuracy": "sub_task_type.Localization_accuracy",
            "manipulation_accuracy": "sub_task_type.Manipulation_accuracy",
            "motion_analysis_accuracy": "sub_task_type.Motion_Analysis_accuracy",
            "pattern_recognition_accuracy": "sub_task_type.Pattern_Recognition_accuracy",
            "traffic_analysis_accuracy": "sub_task_type.Traffic_Analysis_accuracy",
        },
    },
    "spar_bench": {
        "data_name": "SparBench",
        "acc_pattern": "*_SparBench*_acc.csv",
        "overall_key": "overall",
        "scale": 100,  # CSV values are 0-1
        "sub_scores": {
            "camera_motion_infer_accuracy": "camera_motion_infer_accuracy",
            "depth_prediction_oc": "depth_prediction_oc_MRA:.5:.95:.05",
            "depth_prediction_oc_mv": "depth_prediction_oc_mv_MRA:.5:.95:.05",
            "depth_prediction_oo": "depth_prediction_oo_MRA:.5:.95:.05",
            "depth_prediction_oo_mv": "depth_prediction_oo_mv_MRA:.5:.95:.05",
            "distance_infer_center_oo_accuracy": "distance_infer_center_oo_accuracy",
            "distance_infer_center_oo_mv_accuracy": "distance_infer_center_oo_mv_accuracy",
            "distance_prediction_oc": "distance_prediction_oc_MRA:.5:.95:.05",
            "distance_prediction_oc_mv": "distance_prediction_oc_mv_MRA:.5:.95:.05",
            "distance_prediction_oo": "distance_prediction_oo_MRA:.5:.95:.05",
            "distance_prediction_oo_mv": "distance_prediction_oo_mv_MRA:.5:.95:.05",
            "high": "High",
            "low": "Low",
            "middle": "Middle",
            "obj_spatial_relation_oc_mv_accuracy": "obj_spatial_relation_oc_mv_accuracy",
            "obj_spatial_relation_oo_accuracy": "obj_spatial_relation_oo_accuracy",
            "obj_spatial_relation_oo_mv_accuracy": "obj_spatial_relation_oo_mv_accuracy",
            "position_matching_accuracy": "position_matching_accuracy",
            "spatial_imagination_oc_accuracy": "spatial_imagination_oc_accuracy",
            "spatial_imagination_oc_mv_accuracy": "spatial_imagination_oc_mv_accuracy",
            "spatial_imagination_oo_accuracy": "spatial_imagination_oo_accuracy",
            "spatial_imagination_oo_mv_accuracy": "spatial_imagination_oo_mv_accuracy",
            "view_change_infer_vci_metric": "view_change_infer_vci_metric",
        },
    },
    "vsi_debiased": {
        "data_name": "VSI-Bench-Debiased_32frame",
        "acc_pattern": "*_VSI-Bench-Debiased_32frame*_acc.csv",
        "overall_key": "overall",
        "scale": 1,
        "sub_scores": {
            "obj_appearance_order_accuracy": "obj_appearance_order_accuracy",
            "object_abs_distance": "object_abs_distance_MRA:.5:.95:.05",
            "object_counting": "object_counting_MRA:.5:.95:.05",
            "object_rel_direction_accuracy": "object_rel_direction_accuracy",
            "object_rel_distance_accuracy": "object_rel_distance_accuracy",
            "object_size_estimation": "object_size_estimation_MRA:.5:.95:.05",
            "room_size_estimation": "room_size_estimation_MRA:.5:.95:.05",
            "route_planning_accuracy": "route_planning_accuracy",
        },
    },
}


# ---------------------------------------------------------------------------
# CSV loading (auto-detects columnar vs key-value format)
# ---------------------------------------------------------------------------

def load_acc_csv(path: Path | str) -> dict[str, float]:
    """Load a VLMEvalKit ``_acc.csv`` into a flat ``{metric: value}`` dict.

    Handles two formats:
    - **Columnar** (most benchmarks): headers are metric names, single data row.
    - **Key-value** (SiteBench): ``metric|value`` rows.
    """
    df = pd.read_csv(path, sep=None, engine="python")

    # Key-value format: has 'metric' and 'value' columns
    if "metric" in df.columns and "value" in df.columns:
        result = {}
        for _, row in df.iterrows():
            try:
                result[str(row["metric"])] = float(row["value"])
            except (ValueError, TypeError):
                pass
        return result

    # Columnar format: metric names are column headers
    result = {}
    for col in df.columns:
        try:
            val = df[col].iloc[0]
            result[col] = float(val)
        except (ValueError, TypeError):
            pass  # skip non-numeric columns like 'split', 'setting'
    return result


# ---------------------------------------------------------------------------
# Score extraction per benchmark
# ---------------------------------------------------------------------------

def _find_acc_csv(model_dir: Path, pattern: str) -> Path | None:
    """Find the newest matching acc.csv in model_dir or its subdirectories.

    VLMEvalKit writes result files inside ``T{date}_G{hash}/`` subdirectories
    and creates symlinks at the model_dir root. We search both levels and
    return the most recently modified file to handle multiple runs.
    """
    matches = list(model_dir.glob(pattern)) + list(model_dir.glob(f"*/{pattern}"))
    if not matches:
        return None
    # Return newest by modification time
    return max(matches, key=lambda p: p.resolve().stat().st_mtime)


def _load_acc_csv_by_setting(path: Path | str) -> dict[str, dict[str, float]]:
    """Load a multi-setting ``_full_acc.csv`` into ``{setting: {metric: value}}``.

    Returns a dict keyed by setting name.  Falls back to a single
    ``{"vanilla": ...}`` entry if no ``setting`` column is present.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if "setting" not in df.columns:
        # Single-setting file — treat as vanilla
        return {"vanilla": load_acc_csv(path)}
    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        setting = str(row["setting"])
        metrics: dict[str, float] = {}
        for col in df.columns:
            if col == "setting":
                continue
            try:
                metrics[col] = float(row[col])
            except (ValueError, TypeError):
                pass
        result[setting] = metrics
    return result


def _extract_scores(
    benchmark_key: str,
    model_dir: Path,
    model_name: str,
) -> tuple[float | None, dict[str, float | None]]:
    """Extract overall score and sub-scores for a benchmark.

    Returns (overall_score, {sub_key: sub_value}).
    Returns (None, {}) if results not found.
    """
    config = METRIC_MAP.get(benchmark_key)
    if config is None:
        return None, {}

    # Special handling for site (combined SiteBenchImage + SiteBenchVideo)
    if "data_names" in config:
        return _extract_site_scores(config, model_dir, model_name)

    # Find acc.csv
    csv_path = _find_acc_csv(model_dir, config["acc_pattern"])
    if csv_path is None:
        return None, {}

    scale = config.get("scale", 1)
    settings = config.get("settings")

    # Multi-setting extraction (e.g., 3DSRBench vanilla + circ_eval)
    if settings:
        all_settings = _load_acc_csv_by_setting(csv_path)
        # Primary setting is the first one listed (used for overall score)
        primary = settings[0]
        primary_metrics = all_settings.get(primary, {})

        overall_key = config["overall_key"]
        overall = primary_metrics.get(overall_key)
        if overall is not None:
            overall = round(overall * scale, 4)

        sub_scores: dict[str, float | None] = {}
        for setting in settings:
            setting_metrics = all_settings.get(setting, {})
            prefix = "" if setting == primary else f"{setting}_"
            # Overall for each setting
            val = setting_metrics.get(overall_key)
            if val is not None:
                sub_scores[f"{prefix}overall"] = round(val * scale, 4)
            else:
                sub_scores[f"{prefix}overall"] = None
            # Sub-scores
            for payload_key, csv_key in config["sub_scores"].items():
                val = setting_metrics.get(csv_key)
                if val is not None:
                    sub_scores[f"{prefix}{payload_key}"] = round(val * scale, 4)
                else:
                    sub_scores[f"{prefix}{payload_key}"] = None

        return overall, sub_scores

    # Standard single-setting extraction
    metrics = load_acc_csv(csv_path)

    overall_key = config["overall_key"]
    overall = metrics.get(overall_key)
    if overall is not None:
        overall = round(overall * scale, 4)

    sub_scores = {}
    for payload_key, csv_key in config["sub_scores"].items():
        val = metrics.get(csv_key)
        if val is not None:
            sub_scores[payload_key] = round(val * scale, 4)
        else:
            sub_scores[payload_key] = None

    return overall, sub_scores


def _extract_site_scores(
    config: dict,
    model_dir: Path,
    model_name: str,
) -> tuple[float | None, dict[str, float | None]]:
    """Extract combined SiteBench scores using scoring.score_sitebench()."""
    from scoring import score_sitebench

    result = score_sitebench(model_dir, model_name)
    if result is None:
        return None, {}

    scale = config.get("scale", 1)

    overall = result.get("overall", {}).get("caa")
    if overall is not None:
        overall = round(overall * scale, 4)

    sub_scores: dict[str, float | None] = {}
    for payload_key, category_name in config["sub_scores"].items():
        cat_data = result.get(category_name, {})
        val = cat_data.get("caa")
        if val is not None:
            sub_scores[payload_key] = round(val * scale, 4)
        else:
            sub_scores[payload_key] = None

    return overall, sub_scores


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(
    model_dir: Path,
    model_name: str,
    benchmarks: dict[str, str],
    submission_configs: dict | None = None,
    backend_adapter=None,
    judged_benchmarks: dict[str, str] | None = None,
) -> dict:
    """Build the EASI leaderboard submission payload.

    Args:
        model_dir: Path to ``{output_dir}/{model_name}/``
        model_name: Model name (e.g. ``Qwen2.5-VL-7B-Instruct``)
        benchmarks: ``{benchmark_key: data_name}`` for benchmarks that were run
        submission_configs: User-provided metadata overrides
        backend_adapter: Optional backend adapter; when provided, delegates score
            extraction to ``adapter.extract_scores()``.  When ``None``, falls
            back to the VLMEvalKit-specific logic.
        judged_benchmarks: Optional ``{benchmark_key: judge_model_name}`` for
            benchmarks re-evaluated with an LLM judge. When provided together
            with a backend adapter that supports ``extract_scores_dual``,
            emits dual keys (``{bench}_exact_matching`` and
            ``{bench}_{judge_model}``) for each re-evaluated benchmark.

    Returns:
        JSON-serializable dict matching the API schema (camelCase fields).
    """
    configs = submission_configs or {}

    scores: dict[str, float | None] = {}
    sub_scores: dict[str, dict[str, float | None]] = {}

    if backend_adapter is not None:
        # Adapter path: delegate score extraction to the backend adapter
        if judged_benchmarks and hasattr(backend_adapter, "extract_scores_dual"):
            bench_scores = backend_adapter.extract_scores_dual(
                model_dir, model_name, benchmarks, judged_benchmarks,
            )
        else:
            bench_scores = backend_adapter.extract_scores(
                model_dir, model_name, benchmarks,
            )
        for bench_key, bs in sorted(bench_scores.items()):
            scores[bench_key] = bs.overall
            if bs.sub_scores:
                sub_scores[bench_key] = bs.sub_scores
        backend_name = backend_adapter.name
    else:
        # Legacy VLMEvalKit path
        backend_name = "vlmevalkit"

        # Map benchmark keys to METRIC_MAP keys
        # site_image/site_video -> site (combined)
        benchmark_keys_to_process: set[str] = set()
        for key in benchmarks:
            if key in ("site_image", "site_video"):
                benchmark_keys_to_process.add("site")
            elif key in METRIC_MAP:
                benchmark_keys_to_process.add(key)

        for bench_key in sorted(benchmark_keys_to_process):
            overall, subs = _extract_scores(bench_key, model_dir, model_name)
            scores[bench_key] = overall
            if subs:
                sub_scores[bench_key] = subs

    return {
        "modelName": configs.get("modelName", model_name),
        "modelType": configs.get("modelType", ""),
        "precision": configs.get("precision", ""),
        "revision": configs.get("revision", "main"),
        "weightType": configs.get("weightType", ""),
        "baseModel": configs.get("baseModel", ""),
        "backend": configs.get("backend", backend_name),
        "remarks": configs.get("remarks", ""),
        "scores": scores,
        "subScores": sub_scores,
    }


# ---------------------------------------------------------------------------
# Results archive
# ---------------------------------------------------------------------------

def build_results_archive(
    model_dir: Path,
    model_name: str,
    output_dir: Path,
    backend_adapter=None,
) -> Path:
    """Build a zip archive of result files for submission upload.

    Copies all relevant result files (acc.csv, extract_matching.xlsx, judge pkl)
    from model_dir into a staging directory, then zips it.

    Args:
        model_dir: Path to the model's result directory.
        model_name: Model name used for directory naming inside the zip.
        output_dir: Directory where the zip file will be written.
        backend_adapter: Optional backend adapter; when provided, delegates file
            listing to ``adapter.get_result_files()``.  When ``None``, falls
            back to the VLMEvalKit glob patterns.

    Returns:
        Path to the created zip file.
    """
    staging = output_dir / ".tmp_results" / model_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if backend_adapter is not None:
        # Adapter path: delegate file listing to the backend adapter
        files = backend_adapter.get_result_files(model_dir, model_name)
        for src in files:
            if src.is_file():
                shutil.copy2(src, staging / src.name)
    else:
        # Legacy VLMEvalKit path: copy all result files (following symlinks)
        patterns = [
            "*_acc.csv",
            "*_extract_matching.xlsx",
            "*_extract_matching_acc.csv",
            "*_result.pkl",
            "*_result.xlsx",
            "*_llm_*_judge.pkl",
        ]
        copied = set()
        for pattern in patterns:
            for src in model_dir.glob(pattern):
                # Resolve symlinks to get the actual file
                real_src = src.resolve()
                if real_src.name not in copied and real_src.is_file():
                    shutil.copy2(real_src, staging / real_src.name)
                    copied.add(real_src.name)

    # Create zip
    zip_path = output_dir / "easi_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.iterdir()):
            zf.write(f, arcname=f"{model_name}/{f.name}")

    # Clean up staging
    shutil.rmtree(staging)
    try:
        staging.parent.rmdir()  # remove .tmp_results/ if empty
    except OSError:
        pass

    return zip_path


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

SUBMIT_URL = "https://easi.lmms-lab.com/api/submit-with-file/"
MAX_ZIP_SIZE_MB = 4.5


def validate_payload_for_submit(payload: dict, zip_path: Path | None = None) -> list[str]:
    """Check required fields and zip size before calling the API. Returns list of errors."""
    errors = []
    model_name = payload.get("modelName", "")
    if not model_name or "/" not in model_name:
        errors.append("modelName must be in 'org/model' format (set via --submission-configs)")
    if not payload.get("modelType"):
        errors.append("modelType is required (set via --submission-configs)")
    if not payload.get("precision"):
        errors.append("precision is required (set via --submission-configs)")
    scores = payload.get("scores", {})
    if not any(v is not None for v in scores.values()):
        errors.append("No benchmark scores found — all results are null")
    if zip_path and zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_ZIP_SIZE_MB:
            errors.append(
                f"Zip file is {size_mb:.1f} MB (limit: {MAX_ZIP_SIZE_MB} MB). "
                f"Please upload the generated easi_results.json and easi_results.zip "
                f"at https://easi.lmms-lab.com/submit/ "
                f"or email them to easi-lmms-lab@outlook.com"
            )
    return errors


def submit_results(
    payload: dict,
    zip_path: Path,
    hf_token: str,
    max_retries: int = 5,
    on_retry: callable | None = None,
) -> tuple[bool, str]:
    """Submit results to the EASI leaderboard API.

    Args:
        payload: The JSON payload dict.
        zip_path: Path to the zip archive.
        hf_token: HuggingFace access token.
        max_retries: Max retries on network errors.
        on_retry: Optional callback ``(attempt, max_retries, error_msg)``
                  called before each retry.

    Returns:
        ``(success, message)`` tuple.
    """
    try:
        import requests
    except ImportError:
        return False, "The 'requests' library is required for submission. Install with: pip install requests"

    for attempt in range(1, max_retries + 1):
        try:
            with open(zip_path, "rb") as zf:
                resp = requests.post(
                    SUBMIT_URL,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    data={"payload": json.dumps(payload)},
                    files={"zipFile": ("easi_results.zip", zf, "application/zip")},
                    timeout=120,
                )

            # Parse response
            try:
                data = resp.json()
            except Exception:
                data = {}

            if resp.ok and data.get("success"):
                return True, "Submitted successfully"

            # API error — do not retry (4xx/5xx with a message)
            error = data.get("error", resp.text[:200])
            return False, f"HTTP {resp.status_code}: {error}"

        except (requests.ConnectionError, requests.Timeout) as e:
            error_msg = str(e)[:100]
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, max_retries, error_msg)
                time.sleep(5 * (2 ** (attempt - 1)))  # 5s, 10s, 20s, 40s
            else:
                return False, f"Network error after {max_retries} retries: {error_msg}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    return False, "Max retries exceeded"
