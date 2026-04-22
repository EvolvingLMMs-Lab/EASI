"""VLMEvalKit backend adapter.

Extracts VLMEvalKit-specific constants, dataset preparation, verification,
and progress monitoring from the orchestrator (run_easi_eval.py) into a
standalone adapter conforming to :class:`BackendAdapter`.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .base import BackendAdapter, BenchmarkScores, FileState

# ---------------------------------------------------------------------------
# VLMEvalKit-specific constants
# ---------------------------------------------------------------------------

_HF_REPO = "lmms-lab-si/EASI-Leaderboard-Data"
_TSV_URLS = {
    "VSI-Bench_32frame": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/VSI-Bench.tsv",
    "MMSIBench_wo_circular": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/MMSIBench_wo_circular.tsv",
    "MindCubeBench_tiny_raw_qa": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/MindCubeBench_tiny_raw_qa.tsv",
    "ViewSpatialBench": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/ViewSpatialBench.tsv",
    "SiteBenchImage": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/SiteBenchImage.tsv",
    "SiteBenchVideo_32frame": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/SiteBenchVideo.tsv",
    "BLINK": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/BLINK.tsv",
    "3DSRBench": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/3DSRBench.tsv",
    "EmbSpatialBench": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/EmbSpatialBench.tsv",
    "MMSIVideoBench_50frame": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/MMSIVideoBench.tsv",
    "OmniSpatialBench_manual_cot": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/OmniSpatialBench.tsv",
    "SparBench": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/SparBench.tsv",
    "VSI-Bench-Debiased_32frame": f"https://huggingface.co/datasets/{_HF_REPO}/resolve/main/VSI-Bench-Debiased.tsv",
}

# Per-benchmark default judge. Benchmarks not listed here use exact_matching.
# When --judge is not passed, benchmarks needing LLM judge are run in a
# separate VLMEvalKit invocation after the main exact_matching run.
DEFAULT_JUDGE: dict[str, str] = {
    "BLINK": "gpt-4o-1120",  # 7.5% regex extraction failures with exact_matching
}

_MAX_RETRIES = 5
_RETRY_DELAY = 10  # seconds

# VLMEvalKit data_name for each benchmark key
_TASK_MAP = {
    "vsi_bench": "VSI-Bench_32frame",
    "mmsi_bench": "MMSIBench_wo_circular",
    "mindcube_tiny": "MindCubeBench_tiny_raw_qa",
    "viewspatial": "ViewSpatialBench",
    "site_image": "SiteBenchImage",
    "site_video": "SiteBenchVideo_32frame",
    "blink": "BLINK",
    "3dsrbench": "3DSRBench",
    "embspatial": "EmbSpatialBench",
    "mmsi_video_bench": "MMSIVideoBench_50frame",
    "omnispatial_(manual_cot)": "OmniSpatialBench_manual_cot",
    "spar_bench": "SparBench",
    "vsi_debiased": "VSI-Bench-Debiased_32frame",
}


# ---------------------------------------------------------------------------
# Dataset preparation (runs before GPU work)
# ---------------------------------------------------------------------------

def _retry(fn, retries: int = _MAX_RETRIES) -> bool:
    """Call *fn* with retry logic and exponential back-off. Returns True on success.

    Does not retry on HTTP 4xx client errors (e.g. 404 Not Found).
    """
    for attempt in range(1, retries + 1):
        try:
            fn()
            return True
        except Exception as e:
            # Don't retry on 4xx client errors (file not found, unauthorized, etc.)
            err_str = str(e)
            if any(f"{code} Client Error" in err_str for code in range(400, 500)):
                print(f"    FAILED: {e}")
                return False
            if attempt < retries:
                print(f"    Retry {attempt}/{retries}: {e}")
                time.sleep(_RETRY_DELAY * attempt)
            else:
                print(f"    FAILED after {retries} attempts: {e}")
                return False
    return False


def _download_tsv(url: str, dest: Path, retries: int = _MAX_RETRIES) -> bool:
    """Download a TSV file with retry logic for flaky networks."""
    from huggingface_hub import hf_hub_download
    from urllib.parse import urlparse

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    # HF format: /datasets/{org}/{repo}/resolve/{branch}/{filename}
    if len(parts) >= 6 and parts[0] == "datasets" and parts[3] == "resolve":
        repo_id = f"{parts[1]}/{parts[2]}"
        filename = "/".join(parts[5:])
    else:
        import urllib.request
        return _retry(lambda: urllib.request.urlretrieve(url, str(dest)), retries)

    def _hf_download():
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest.parent),
        ))
        if downloaded != dest and downloaded.exists():
            downloaded.rename(dest)

    return _retry(_hf_download, retries)


def prepare_datasets(
    dataset_dir: Path,
    benchmarks: dict[str, str],
    display: object | None = None,
) -> bool:
    """Download all required datasets before evaluation. Returns True if all OK."""
    def _log(msg: str):
        if not display:
            print(msg)

    def _set_prep(key: str, status: str, detail: str = ""):
        if display:
            display.set_data_prep(key, status, detail)

    # Disable hf_transfer if not installed — causes non-transient failures
    # that waste all retry attempts.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            _log("Note: Disabled HF_HUB_ENABLE_HF_TRANSFER (hf_transfer not installed)")

    if display:
        display.set_phase("Preparing datasets")
    else:
        print(f"\n{'='*60}")
        print("PREPARING DATASETS")
        print(f"{'='*60}")

    all_ok = True

    for key, data_name in benchmarks.items():
        url = _TSV_URLS.get(data_name)
        if url:
            tsv_filename = url.split("/")[-1]
            tsv_path = dataset_dir / tsv_filename
            tsv_path_alt = dataset_dir / f"{data_name}.tsv"
            if tsv_path.exists() or tsv_path_alt.exists():
                existing = tsv_path if tsv_path.exists() else tsv_path_alt
                _set_prep(key, "done", existing.name)
                if not display:
                    print(f"  [OK] {data_name} TSV ({existing.name})")
            else:
                _set_prep(key, "downloading", f"Downloading {tsv_filename}...")
                if not display:
                    print(f"  Downloading {tsv_filename}...")
                if _download_tsv(url, tsv_path):
                    _set_prep(key, "done", tsv_filename)
                    if not display:
                        print(f"  [OK] {tsv_filename}")
                else:
                    _set_prep(key, "failed", f"{tsv_filename} download failed")
                    if not display:
                        print(f"  [FAIL] {tsv_filename}")
                    all_ok = False

    if not display:
        status = "ready" if all_ok else "some downloads failed"
        print(f"\nDataset preparation: {status}")
        print(f"{'='*60}\n")
    return all_ok


# ---------------------------------------------------------------------------
# Result verification
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    key: str
    data_name: str
    success: bool
    completed: int       # samples with predictions
    total: int           # total samples (0 = unknown)
    has_acc_csv: bool
    errors: list[str] = field(default_factory=list)


def count_xlsx_predictions(xlsx_path: Path) -> tuple[int, int]:
    """Count (completed, total) predictions in a VLMEvalKit xlsx."""
    if not xlsx_path.exists():
        return 0, 0
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        total = len(df)
        if "prediction" in df.columns:
            completed = int(df["prediction"].notna().sum())
        else:
            completed = total
        return completed, total
    except Exception:
        return -1, -1


def diagnose_missing_predictions(xlsx_path: Path) -> str | None:
    """Analyze why predictions are missing in a VLMEvalKit xlsx.

    Returns a human-readable diagnosis string, or None if all predictions present.
    """
    if not xlsx_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        return None

    if "prediction" not in df.columns:
        return None

    missing = df[df["prediction"].isna()]
    if len(missing) == 0:
        return None

    n_missing = len(missing)
    n_total = len(df)

    # Check if it's a video/image loading issue:
    # If ALL samples for a given media file are missing, likely a load failure.
    media_col = None
    for col in ("video", "image", "image_path"):
        if col in df.columns:
            media_col = col
            break

    media_failures = 0
    inference_failures = 0
    if media_col:
        missing_media = set(missing[media_col].dropna().unique())
        for media in missing_media:
            all_for_media = df[df[media_col] == media]
            all_missing = all_for_media["prediction"].isna().all()
            n_in_group = len(all_for_media)
            n_missing_in_group = int(all_for_media["prediction"].isna().sum())
            if all_missing:
                media_failures += n_missing_in_group
            else:
                inference_failures += n_missing_in_group
    else:
        inference_failures = n_missing

    # Build diagnosis
    parts = [f"{n_missing}/{n_total} samples missing"]
    if media_failures > 0:
        parts.append(f"{media_failures} media load failure(s)")
    if inference_failures > 0:
        parts.append(
            f"{inference_failures} model inference failure(s) "
            "(empty response, scored as 0 — rerun with --judge gpt-4o-1120, or ignore if acceptable)"
        )

    # Add question type breakdown if available
    if "question_type" in missing.columns:
        type_counts = missing["question_type"].value_counts()
        if len(type_counts) <= 3:
            type_str = ", ".join(f"{t}({c})" for t, c in type_counts.items())
            parts.append(f"affected types: {type_str}")

    return "; ".join(parts)


def parse_errors(stderr: str) -> list[str]:
    """Extract VLMEvalKit ERROR lines from stderr."""
    errors = []
    for line in stderr.splitlines():
        if "ERROR" in line and ("combination failed" in line or "AssertionError" in line):
            errors.append(line.strip())
    return list(dict.fromkeys(errors))


def verify_results(
    output_dir: Path,
    model_name: str,
    benchmarks: dict[str, str],
    stderr: str,
) -> list[BenchmarkResult]:
    """Verify all benchmark results after a single VLMEvalKit run."""
    from scoring import find_acc_csv

    model_dir = output_dir / model_name
    errors = parse_errors(stderr)
    results = []

    for key, data_name in benchmarks.items():
        xlsx_path = model_dir / f"{model_name}_{data_name}.xlsx"
        completed, total = count_xlsx_predictions(xlsx_path)
        has_acc = find_acc_csv(model_dir, model_name, data_name) is not None

        # Match errors to this benchmark
        bench_errors = [e for e in errors if data_name in e]

        if completed <= 0 and total <= 0:
            success = False  # no predictions at all (xlsx missing)
        elif completed <= 0 and total > 0:
            success = False
        elif not has_acc:
            success = False
        else:
            success = True

        # Warn if some samples were skipped (results may be inaccurate)
        if total > 0 and completed < total:
            diagnosis = diagnose_missing_predictions(xlsx_path)
            if diagnosis:
                bench_errors.append(f"WARNING: {diagnosis}")
            else:
                skipped = total - completed
                bench_errors.append(
                    f"WARNING: {skipped}/{total} samples missing"
                )

        results.append(BenchmarkResult(
            key=key, data_name=data_name, success=success,
            completed=completed, total=total, has_acc_csv=has_acc,
            errors=bench_errors,
        ))

    return results


# ---------------------------------------------------------------------------
# Progress monitoring helpers
# ---------------------------------------------------------------------------

def _count_tsv_rows(dataset_dir: Path, data_name: str) -> int:
    """Count data rows in a benchmark's TSV file (excluding header).

    Uses pandas to correctly handle cells with embedded newlines
    (e.g. base64-encoded images) and large fields.
    """
    url = _TSV_URLS.get(data_name, "")
    tsv_filename = url.split("/")[-1] if url else f"{data_name}.tsv"
    for candidate in [dataset_dir / tsv_filename, dataset_dir / f"{data_name}.tsv"]:
        if candidate.exists():
            try:
                import pandas as pd
                # Only read the index column to avoid loading large image data
                df = pd.read_csv(candidate, sep="\t", usecols=["index"])
                return len(df)
            except Exception:
                pass
    return 0


def _count_pkl_predictions(model_dir: Path, data_name: str) -> int:
    """Count predictions in VLMEvalKit's intermediate PKL files for a dataset.

    VLMEvalKit writes ``{rank}{world_size}_{dataset_name}.pkl`` every 10 samples.
    These are dicts mapping sample index -> response.
    """
    total = 0
    # Search all eval_id subdirectories (T{date}_G{hash}/)
    for pkl in model_dir.glob(f"*/*_{data_name}.pkl"):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                total += len(data)
        except Exception:
            pass  # file may be mid-write
    return total


def _has_result_file(model_dir: Path, model_name: str, data_name: str) -> bool:
    """Check if VLMEvalKit has written the final result file (xlsx/tsv/json)."""
    for ext in ("xlsx", "tsv", "json"):
        if list(model_dir.glob(f"*/{model_name}_{data_name}.{ext}")):
            return True
    return False


def _has_acc_csv(model_dir: Path, model_name: str, data_name: str) -> bool:
    """Check if VLMEvalKit has written the _acc.csv (evaluation complete)."""
    return bool(list(model_dir.glob(f"*/{model_name}_{data_name}*_acc.csv")))


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class VLMEvalKitAdapter(BackendAdapter):
    """Backend adapter for VLMEvalKit."""

    TASK_MAP = _TASK_MAP

    def __init__(self, *, repo_root: Path | None = None, rerun: bool = False):
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.rerun = rerun

    @property
    def name(self) -> str:
        return "vlmevalkit"

    def build_cmd(
        self,
        model: str,
        benchmarks: dict[str, str],
        output_dir: Path,
        nproc: int,
        *,
        extra_args: list[str] | None = None,
        judge: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> list[str]:
        """Build VLMEvalKit command: torchrun/python VLMEvalKit/run.py ..."""
        run_py = self.repo_root / "VLMEvalKit" / "run.py"
        data_names = list(benchmarks.values())

        if nproc > 1:
            cmd = ["torchrun", f"--nproc-per-node={nproc}", str(run_py)]
        else:
            cmd = [sys.executable, str(run_py)]

        cmd += [
            "--data", *data_names,
            "--model", model,
            "--work-dir", str(output_dir),
        ]

        if not self.rerun:
            cmd.append("--reuse")

        if judge:
            cmd += ["--judge", judge]
        if verbose:
            cmd.append("--verbose")
        if extra_args:
            cmd += extra_args

        return cmd

    def prepare_datasets(
        self,
        benchmarks: dict[str, str],
        dataset_dir: Path,
        display: object | None = None,
    ) -> bool:
        """Delegate to the module-level prepare_datasets()."""
        return prepare_datasets(dataset_dir, benchmarks, display=display)

    def poll_progress(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, FileState] | None:
        """Return dict[str, FileState] from pkl/xlsx/acc.csv polling."""
        result: dict[str, FileState] = {}
        for key, data_name in benchmarks.items():
            has_eval = _has_acc_csv(model_dir, model_name, data_name)
            has_result = _has_result_file(model_dir, model_name, data_name)
            pkl_count = _count_pkl_predictions(model_dir, data_name) if not has_result else 0
            result[key] = FileState(
                has_result=has_result,
                has_eval=has_eval,
                sample_count=pkl_count,
            )
        return result

    def detect_completion(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, bool]:
        """Check which benchmarks completed via _acc.csv presence."""
        return {
            key: _has_acc_csv(model_dir, model_name, data_name)
            for key, data_name in benchmarks.items()
        }

    def get_result_files(self, model_dir: Path, model_name: str) -> list[Path]:
        """Glob for *_acc.csv, *_extract_matching.xlsx, etc."""
        patterns = [
            f"{model_name}_*_acc.csv",
            f"{model_name}_*.xlsx",
        ]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(model_dir.glob(f"*/{pattern}"))
        return sorted(set(files))

    def extract_scores(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, BenchmarkScores]:
        """Extract overall + sub-scores for each benchmark via postprocess."""
        # Import here to avoid circular dependency at module level
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from postprocess import _extract_scores, METRIC_MAP

        scores: dict[str, BenchmarkScores] = {}
        for key, data_name in benchmarks.items():
            config = METRIC_MAP.get(key)
            if config is None:
                scores[key] = BenchmarkScores()
                continue

            result = _extract_scores(key, model_dir, model_name)
            if result is None:
                scores[key] = BenchmarkScores()
            else:
                overall, sub = result
                scores[key] = BenchmarkScores(overall=overall, sub_scores=sub)
        return scores

    def get_env_overrides(self) -> dict[str, str]:
        """PYTHONUNBUFFERED=1 + LMUData if set."""
        env = {"PYTHONUNBUFFERED": "1"}
        lmu = os.environ.get("LMUData")
        if lmu:
            env["LMUData"] = lmu
        return env

    # ---- Utility methods ----

    @staticmethod
    def get_default_judge() -> dict[str, str]:
        """Return the DEFAULT_JUDGE mapping."""
        return DEFAULT_JUDGE

    @staticmethod
    def count_tsv_rows(dataset_dir: Path, data_name: str) -> int:
        """Wrap module-level _count_tsv_rows."""
        return _count_tsv_rows(dataset_dir, data_name)
