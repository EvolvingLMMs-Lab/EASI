#!/usr/bin/env python3
"""Run EASI benchmark evaluation via VLMEvalKit.

Runs all EASI-8 benchmarks (optionally including extras) on a specified model
using VLMEvalKit's run.py. Handles dataset preparation with retry logic before
launching GPU-heavy inference.

Usage:
    python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct
    python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --output-dir ./results --include-extra

Installation:
    - Update requirements.txt with "transformers>=4.45,<5", "torch==2.7.1", "torchvision==0.22.1", "setuptools<81"
    - Run the following:
    uv venv -p 3.11
    source .venv/bin/activate
    cd VLMEvalKit
    uv pip install .
    uv pip install hf_transfer  # optional, for faster HF downloads (but can be flaky on some networks)
    uv pip install flash-attn --no-build-isolation
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import find_acc_csv

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

EASI_8 = [
    ("vsi_bench", "VSI-Bench_32frame"),
    ("mmsi_bench", "MMSIBench_wo_circular"),
    ("mindcube_tiny", "MindCubeBench_tiny_raw_qa"),
    ("viewspatial", "ViewSpatialBench"),
    ("site_image", "SiteBenchImage"),
    ("site_video", "SiteBenchVideo_32frame"),
    ("blink", "BLINK"),
    ("3dsrbench", "3DSRBench"),
    ("embspatial", "EmbSpatialBench"),
]

EXTRA = [
    ("mmsi_video_bench", "MMSIVideoBench_50frame"),
    ("omnispatial_(manual_cot)", "OmniSpatialBench_manual_cot"),
    ("spar_bench", "SparBench"),
    ("vsi_debiased", "VSI-Bench-Debiased_32frame"),
]

# Per-benchmark default judge. Benchmarks not listed here use exact_matching.
# When --judge is not passed, benchmarks needing LLM judge are run in a
# separate VLMEvalKit invocation after the main exact_matching run.
DEFAULT_JUDGE: dict[str, str] = {
    "BLINK": "gpt-4o-1120",  # 7.5% regex extraction failures with exact_matching
}

# Display grouping: map a group name to its child benchmark keys.
# Groups appear as a single benchmark in the UI with indented sub-entries.
BENCHMARK_GROUPS: dict[str, list[str]] = {
    "sitebench": ["site_image", "site_video"],
}

# Aliases accepted by --benchmarks (in addition to exact keys and group names).
BENCHMARK_ALIASES: dict[str, list[str]] = {
    "site": ["site_image", "site_video"],
}

# Display order for EASI_8. Group names replace their children.
# This is what the user sees (8 items for EASI_8).
EASI_8_DISPLAY_ORDER: list[str | tuple[str, list[str]]] = [
    "vsi_bench",
    "mmsi_bench",
    "mindcube_tiny",
    "viewspatial",
    ("sitebench", ["site_image", "site_video"]),
    "blink",
    "3dsrbench",
    "embspatial",
]

# ---------------------------------------------------------------------------
# Dataset download config
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

_MAX_RETRIES = 5
_RETRY_DELAY = 10  # seconds


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
    display: "ProgressDisplay | None" = None,
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
# Progress monitoring
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


class _DisplayWriter:
    """Silently consume stdout writes while the rich display is active.

    Prevents stray ``print()`` calls from sub-functions (e.g. huggingface_hub)
    from corrupting the terminal. Output is discarded — the rich panel shows
    all relevant info via its own sections.
    """

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


class ProgressDisplay:
    """Rich-based live terminal display for evaluation progress.

    Supports grouped benchmarks: ``display_items`` is a list where each entry
    is either a plain benchmark key (``str``) or a ``(group_name, [child_keys])``
    tuple.  Groups render as a parent row with indented children.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    def __init__(
        self,
        model_name: str,
        benchmarks: dict[str, str],
        totals: dict[str, int],
        display_items: list[str | tuple[str, list[str]]],
    ):
        self.model_name = model_name
        self.benchmarks = benchmarks  # flat key -> data_name
        self.totals = totals
        self.display_items = display_items
        # Dual-phase tracking: inference and evaluation
        self.infer_status: dict[str, str] = {k: self.PENDING for k in benchmarks}
        self.infer_completed: dict[str, int] = {k: 0 for k in benchmarks}
        self.eval_status: dict[str, str] = {k: self.PENDING for k in benchmarks}
        self.phase: str = ""
        # Dataset preparation tracking per benchmark key.
        # status: "pending" | "downloading" | "done" | "failed"
        self.data_prep: dict[str, tuple[str, str]] = {
            k: ("pending", "") for k in benchmarks
        }
        # Per-benchmark warnings (shown below the progress bar row)
        self.warnings: dict[str, str] = {}
        # Post-processing results (set after payload is built)
        # {"scores": {key: float}, "payload_path": str, "zip_path": str}
        self.results_info: dict | None = None
        # Environment checks: list of (name, status, detail) — shown when --submit
        self.env_checks: list[tuple[str, str, str]] = []
        # Submission status: ("submitting"|"retrying"|"success"|"failed", detail)
        self.submission_status: tuple[str, str] | None = None
        self._lock = threading.Lock()
        self._live = None
        self._start_time = time.time()
        self._log_path: str | None = None

    def __rich__(self):
        """Let rich.live.Live call this on each refresh cycle."""
        with self._lock:
            return self._render()

    @staticmethod
    def _aggregate_status(statuses: list[str] | set[str], ordered: list[str]) -> str:
        """Return the first status in *ordered* that matches the aggregate rule.

        - If all statuses are the same, return that status.
        - Otherwise return the first entry in *ordered* that any status matches,
          treating a mixed set containing both done and pending as "running".
        """
        status_set = set(statuses)
        if len(status_set) == 1:
            return next(iter(status_set))
        for candidate in ordered:
            if candidate in status_set:
                return candidate
        return ordered[-1] if ordered else next(iter(status_set))

    def _benchmark_overall_status(self, key: str) -> str:
        """Derive overall status from both phases for a single benchmark."""
        if self.eval_status[key] == self.DONE:
            return self.DONE
        if self.infer_status[key] == self.FAILED or self.eval_status[key] == self.FAILED:
            return self.FAILED
        if self.infer_status[key] in (self.RUNNING, self.DONE):
            return self.RUNNING
        return self.PENDING

    def _group_status(self, children: list[str]) -> str:
        """Derive a group's aggregate status from its children's overall status."""
        statuses = {self._benchmark_overall_status(c) for c in children}
        if statuses == {self.DONE}:
            return self.DONE
        if self.FAILED in statuses:
            return self.FAILED
        if self.RUNNING in statuses or (self.DONE in statuses and self.PENDING in statuses):
            return self.RUNNING
        return self.PENDING

    @staticmethod
    def _progress_bar(completed: int, total: int, bar_len: int = 20) -> str:
        if total <= 0:
            return f"{completed}/?" if completed > 0 else ""
        pct = min(completed * 100 // total, 100)
        filled = min(bar_len * completed // total, bar_len)
        bar = (
            "[green]" + "\u2588" * filled + "[/green]"
            + "[dim]" + "\u2591" * (bar_len - filled) + "[/dim]"
        )
        # Fixed-width count: right-align "completed/total" to 11 chars
        count = f"{completed}/{total}"
        return f"{bar} {count:>11s} ({pct:>3d}%)"

    def _render_data_prep_icon(self, status: str) -> str:
        if status == "done":
            return "[green]\u2713[/green]"
        if status == "failed":
            return "[red]\u2717[/red]"
        if status == "downloading":
            return "[yellow]\u25D4[/yellow]"
        return "[dim]\u2013[/dim]"  # pending: en-dash

    def _render_data_prep_row(self, key: str, indent: str = "") -> str:
        status, detail = self.data_prep.get(key, ("pending", ""))
        icon = self._render_data_prep_icon(status)
        detail_str = f"  [dim]{detail}[/dim]" if detail else ""
        if status == "failed":
            detail_str = f"  [red]{detail}[/red]" if detail else ""
        return f" {indent}{icon} {key}{detail_str}"

    def _render_data_prep_section(self) -> str:
        """Render the dataset prep section — full or collapsed."""
        is_prep_phase = self.phase == "Preparing datasets"

        if is_prep_phase:
            # Full view during prep
            lines = [" [bold]\u2500\u2500 Preparing datasets \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold]"]
            for item in self.display_items:
                if isinstance(item, tuple):
                    group_name, children = item
                    child_statuses = [self.data_prep.get(c, ("pending", ""))[0] for c in children]
                    grp_icon = self._render_data_prep_icon(
                        self._aggregate_status(child_statuses, ordered=["done", "failed", "downloading", "pending"])
                    )
                    lines.append(f" {grp_icon} {group_name}")
                    for child in children:
                        lines.append(self._render_data_prep_row(child, indent="  "))
                else:
                    lines.append(self._render_data_prep_row(item))
            return "\n".join(lines)
        else:
            # Collapsed summary during eval/complete
            done_count = sum(1 for s, _ in self.data_prep.values() if s == "done")
            total_count = len(self.data_prep)
            failed = [k for k, (s, _) in self.data_prep.items() if s == "failed"]
            summary = f" [bold]\u2500\u2500 Datasets \u2500\u2500[/bold] {done_count}/{total_count} ready"
            if failed:
                failed_str = "  ".join(f"[red]\u2717 {k}[/red]" for k in failed)
                summary += f"  {failed_str}"
            return summary

    def _phase_icon(self, status: str) -> str:
        """Render a phase status icon (same style as data prep icons)."""
        if status == self.DONE:
            return "[green]\u2713[/green]"
        if status == self.FAILED:
            return "[red]\u2717[/red]"
        if status == self.RUNNING:
            return "[yellow]\u25D4[/yellow]"
        return "[dim]\u2013[/dim]"  # pending

    def _add_benchmark_row(self, table, key: str, indent: str = "",
                           pending_warnings: list | None = None):
        """Add a benchmark row with dual-phase icons."""
        infer_st = self.infer_status[key]
        eval_st = self.eval_status[key]
        completed = self.infer_completed[key]
        total = self.totals.get(key, 0)
        name = f"{indent}[dim]{key}[/dim]" if indent else key

        # Build phase string: "✓ infer  ◔ eval"
        phases = f"{self._phase_icon(infer_st)} infer"
        if infer_st in (self.DONE, self.FAILED):
            phases += f"  {self._phase_icon(eval_st)} eval"

        # Progress bar: show inference progress when inferring, keep at 100% after
        if infer_st in (self.PENDING, self.FAILED):
            progress = ""
        else:
            progress = self._progress_bar(completed, total)

        table.add_row(name, phases, progress)
        if key in self.warnings and pending_warnings is not None:
            pending_warnings.append((key, self.warnings[key]))

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.console import Group as RichGroup

        parts: list = []

        # Environment section (shown when --submit)
        if self.env_checks:
            env_lines = [" [bold]\u2500\u2500 Environment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold]"]
            for name, status, detail in self.env_checks:
                icon = self._phase_icon(
                    self.DONE if status == "ok" else self.FAILED
                )
                env_lines.append(f" {icon} {name}  [dim]{detail}[/dim]")
            parts.append(Text.from_markup("\n".join(env_lines) + "\n"))

        # Dataset prep section
        parts.append(Text.from_markup(self._render_data_prep_section()))

        # Evaluation section label
        eval_label = self.phase if self.phase and self.phase != "Preparing datasets" else "Evaluation"
        parts.append(Text.from_markup(
            f"\n [bold]\u2500\u2500 {eval_label} \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold]"
        ))

        # Benchmark table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Benchmark", min_width=20, style="bold")
        table.add_column("Phases", min_width=22)
        table.add_column("Progress", min_width=16, justify="right")

        display_count = len(self.display_items)
        done_display = 0
        pending_warnings: list[tuple[str, str]] = []

        for item in self.display_items:
            if isinstance(item, tuple):
                group_name, children = item
                grp_status = self._group_status(children)
                grp_completed = sum(self.infer_completed[c] for c in children)
                grp_total = sum(self.totals.get(c, 0) for c in children)

                # Group phase icons: aggregate from children
                any_infer_terminal = any(self.infer_status[c] in (self.DONE, self.FAILED) for c in children)
                any_infer_failed = any(self.infer_status[c] == self.FAILED for c in children)
                all_infer_done = all(self.infer_status[c] == self.DONE for c in children)
                grp_infer_icon = self._phase_icon(
                    self.FAILED if any_infer_failed else
                    self.DONE if all_infer_done else
                    self.RUNNING if any(self.infer_status[c] == self.RUNNING for c in children) else
                    self.PENDING
                )
                grp_phases = f"{grp_infer_icon} infer"
                if any_infer_terminal:
                    any_eval_failed = any(self.eval_status[c] == self.FAILED for c in children)
                    all_eval_done = all(self.eval_status[c] == self.DONE for c in children)
                    any_eval_running = any(self.eval_status[c] == self.RUNNING for c in children)
                    grp_eval_icon = self._phase_icon(
                        self.FAILED if any_eval_failed else
                        self.DONE if all_eval_done else
                        self.RUNNING if any_eval_running else
                        self.PENDING
                    )
                    grp_phases += f"  {grp_eval_icon} eval"

                table.add_row(
                    group_name,
                    grp_phases,
                    self._progress_bar(grp_completed, grp_total) if grp_status != self.PENDING else "",
                )
                for child in children:
                    self._add_benchmark_row(table, child, indent="  ",
                                            pending_warnings=pending_warnings)
                if grp_status in (self.DONE, self.FAILED):
                    done_display += 1
            else:
                self._add_benchmark_row(table, item,
                                        pending_warnings=pending_warnings)
                overall = self._benchmark_overall_status(item)
                if overall in (self.DONE, self.FAILED):
                    done_display += 1

        parts.append(table)

        # Warnings (rendered outside the table so they don't affect column widths)
        if pending_warnings:
            warning_lines = []
            for key, warning in pending_warnings:
                warning_lines.append(f" [yellow]!! {key}: {warning}[/yellow]")
            parts.append(Text.from_markup("\n".join(warning_lines)))

        # Log path (under evaluation section)
        if self._log_path and self.phase not in ("Preparing datasets", ""):
            parts.append(Text.from_markup(f"\n [dim]Log: {self._log_path}[/dim]"))

        # Results section (post-processing)
        if self.results_info is not None or self.phase == "Building submission":
            parts.append(Text.from_markup(
                f"\n [bold]\u2500\u2500 Results \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold]"
            ))
            if self.results_info is not None:
                score_table = Table(show_header=True, box=None, padding=(0, 2))
                score_table.add_column("Benchmark", min_width=20, style="bold")
                score_table.add_column("Score", min_width=10, justify="right")
                for bench_key, score in self.results_info.get("scores", {}).items():
                    score_str = f"{score:.2f}" if score is not None else "[dim]n/a[/dim]"
                    score_table.add_row(bench_key, score_str)
                parts.append(score_table)
                if "payload_path" in self.results_info:
                    parts.append(Text.from_markup(
                        f"\n [dim]Payload: {self.results_info['payload_path']}[/dim]"
                        f"\n [dim]Archive: {self.results_info['zip_path']}[/dim]"
                    ))
            else:
                parts.append(Text.from_markup(" [yellow]\u25D4[/yellow] Building submission payload..."))

        # Submission section
        if self.submission_status is not None:
            parts.append(Text.from_markup(
                f"\n [bold]\u2500\u2500 Submission \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold]"
            ))
            status, detail = self.submission_status
            if status == "submitting":
                parts.append(Text.from_markup(f" [yellow]\u25D4[/yellow] Submitting to easi.lmms-lab.com..."))
            elif status == "retrying":
                parts.append(Text.from_markup(f" [yellow]\u25D4[/yellow] {detail}"))
            elif status == "success":
                parts.append(Text.from_markup(f" [green]\u2713[/green] {detail}"))
            elif status == "failed":
                parts.append(Text.from_markup(f" [red]\u2717[/red] {detail}"))

        # Footer
        elapsed = int(time.time() - self._start_time)
        elapsed_str = f"{elapsed // 60}m {elapsed % 60:02d}s"

        footer_parts: list[str] = []

        bar_len = 30
        filled = bar_len * done_display // display_count if display_count > 0 else 0
        overall_bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        footer_parts.append(
            f" [bold]Overall[/bold]  [green]{overall_bar}[/green]  "
            f"{done_display}/{display_count} benchmarks  [dim]({elapsed_str})[/dim]"
        )

        # Show hint when nothing has started yet
        any_started = any(s != self.PENDING for s in self.infer_status.values())
        if not any_started and self.phase not in ("Preparing datasets", ""):
            footer_parts.append(" [dim]Initializing (loading model / building datasets)...[/dim]")
            if self._log_path:
                footer_parts.append(f" [dim]tail -f {self._log_path}[/dim]")

        parts.append(Text.from_markup("\n" + "\n".join(footer_parts)))

        return Panel(
            RichGroup(*parts),
            title=f"[bold]EASI Evaluation: {self.model_name}[/bold]",
            border_style="blue",
        )

    def start(self):
        from rich.live import Live
        from rich.console import Console
        self._orig_stdout = sys.stdout
        # Give Live a console pinned to the real stdout so the redirect
        # only captures stray print() calls, not rich's own rendering.
        console = Console(file=self._orig_stdout)
        sys.stdout = _DisplayWriter()
        self._live = Live(self, refresh_per_second=1, console=console)
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
        if hasattr(self, "_orig_stdout"):
            sys.stdout = self._orig_stdout

    def update_infer(self, key: str, completed: int):
        with self._lock:
            self.infer_completed[key] = completed
            if self.infer_status[key] == self.PENDING and completed > 0:
                self.infer_status[key] = self.RUNNING

    def mark_infer_done(self, key: str):
        with self._lock:
            if self.infer_status[key] == self.FAILED:
                return  # don't overwrite a failure
            self.infer_status[key] = self.DONE
            # Set completed = total so bar shows 100%
            total = self.totals.get(key, 0)
            if total > 0:
                self.infer_completed[key] = total

    def mark_eval_running(self, key: str):
        with self._lock:
            self.eval_status[key] = self.RUNNING

    def mark_eval_done(self, key: str):
        with self._lock:
            self.eval_status[key] = self.DONE

    def mark_failed(self, key: str):
        with self._lock:
            # Fail both phases
            if self.infer_status[key] != self.DONE:
                self.infer_status[key] = self.FAILED
            else:
                self.eval_status[key] = self.FAILED

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase

    def set_data_prep(self, key: str, status: str, detail: str = ""):
        with self._lock:
            self.data_prep[key] = (status, detail)

    def set_warning(self, key: str, warning: str):
        with self._lock:
            self.warnings[key] = warning



def _poll_progress(
    display: ProgressDisplay,
    model_dir: Path,
    model_name: str,
    benchmarks: dict[str, str],
    detect_failures: bool = False,
):
    """Single pass: update display from PKL/xlsx/_acc.csv files on disk.

    Args:
        detect_failures: If True, detect failed benchmarks from sequential
            ordering. Only enable this after the subprocess has exited —
            during live monitoring, later benchmarks may have results from
            a previous ``--reuse`` run, not the current one.
    """
    # Phase 1: Collect file state outside the lock (disk I/O)
    file_state: dict[str, tuple[bool, bool, int]] = {}
    for key, data_name in benchmarks.items():
        has_acc = _has_acc_csv(model_dir, model_name, data_name)
        has_result = _has_result_file(model_dir, model_name, data_name)
        pkl_count = _count_pkl_predictions(model_dir, data_name) if not has_result else 0
        file_state[key] = (has_acc, has_result, pkl_count)

    # Phase 2: Update display state under the lock
    with display._lock:
        for key in benchmarks:
            # Skip terminal states
            if (display.eval_status[key] in (ProgressDisplay.DONE, ProgressDisplay.FAILED)
                    or display.infer_status[key] == ProgressDisplay.FAILED):
                continue

            has_acc, has_result, pkl_count = file_state[key]

            if has_acc:
                display.infer_status[key] = ProgressDisplay.DONE
                total = display.totals.get(key, 0)
                if total > 0:
                    display.infer_completed[key] = total
                display.eval_status[key] = ProgressDisplay.DONE
            elif has_result:
                display.infer_status[key] = ProgressDisplay.DONE
                total = display.totals.get(key, 0)
                if total > 0:
                    display.infer_completed[key] = total
                if display.eval_status[key] == ProgressDisplay.PENDING:
                    display.eval_status[key] = ProgressDisplay.RUNNING
            elif pkl_count > 0:
                display.infer_completed[key] = pkl_count
                if display.infer_status[key] == ProgressDisplay.PENDING:
                    display.infer_status[key] = ProgressDisplay.RUNNING

        # Phase 3: Detect failed benchmarks from sequential ordering.
        # Only run after subprocess exits — during live monitoring, later
        # benchmarks may have results from a previous --reuse run.
        if detect_failures:
            keys_list = list(benchmarks.keys())
            for i, key in enumerate(keys_list):
                if display.infer_status[key] != ProgressDisplay.RUNNING:
                    continue
                has_acc, has_result, _ = file_state[key]
                if has_result:
                    continue  # has xlsx, not stuck
                # Check if any later benchmark has NEW activity (pkl that
                # wasn't there before, or newly appeared result/acc files
                # that Phase 2 just transitioned to RUNNING/DONE)
                for later_key in keys_list[i + 1:]:
                    later_acc, later_result, later_pkl = file_state[later_key]
                    if later_acc or later_result or later_pkl > 0:
                        display.infer_status[key] = ProgressDisplay.FAILED
                        break


def _monitor_loop(
    display: ProgressDisplay,
    model_dir: Path,
    model_name: str,
    benchmarks: dict[str, str],
    stop_event: threading.Event,
):
    """Background thread: poll PKL/xlsx files and update the display."""
    while not stop_event.is_set():
        _poll_progress(display, model_dir, model_name, benchmarks)
        stop_event.wait(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run EASI benchmark evaluation via VLMEvalKit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run EASI-8 core benchmarks
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct

  # Run all benchmarks including extras
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --include-extra

  # Run specific benchmarks only
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --benchmarks vsi_bench,blink,site

  # With data parallelism and LLM judge
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --nproc 8 --judge chatgpt-0125
        """,
    )
    parser.add_argument("--model", required=True, help="Model name (HuggingFace ID or VLMEvalKit model name)")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory (default: ./eval_results)")
    parser.add_argument("--include-extra", action="store_true", help="Also run extra benchmarks")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark keys (e.g. vsi_bench,blink,site)")
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs for data parallelism via torchrun")
    parser.add_argument("--judge", type=str, default=None,
                        help="Judge model for answer extraction. If not specified, VLMEvalKit "
                             "uses per-benchmark defaults (e.g. chatgpt-0125 for MCQ benchmarks).")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory (default: ./datasets in repo root)")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to VLMEvalKit (prints per-sample model responses)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich progress display (passthrough raw subprocess output)")
    parser.add_argument("--submission-configs", type=str, default=None,
                        help='JSON string with submission metadata, e.g.: '
                             '\'{"modelName": "org/model", "modelType": "pretrained", "precision": "bfloat16"}\'')
    parser.add_argument("--submit", action="store_true",
                        help="Submit results to EASI leaderboard after evaluation (requires HF token)")

    args, extra_args = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent  # scripts/submissions/ -> repo root
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else repo_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(dataset_dir.resolve())
    # NOTE: We do NOT override HF_HUB_CACHE. Both our prep step and
    # VLMEvalKit's internal snapshot_download use the default HF cache
    # (~/.cache/huggingface/hub/). This avoids re-downloading model weights
    # to a different location.

    # Build benchmark list
    all_benchmarks = dict(EASI_8)
    if args.include_extra:
        all_benchmarks.update(dict(EXTRA))

    if args.benchmarks:
        requested = set(args.benchmarks.split(","))
        # Expand aliases and group names to their child keys
        expanded: set[str] = set()
        for name in requested:
            if name in BENCHMARK_ALIASES:
                expanded.update(BENCHMARK_ALIASES[name])
            elif name in BENCHMARK_GROUPS:
                expanded.update(BENCHMARK_GROUPS[name])
            else:
                expanded.add(name)
        available = dict(EASI_8 + EXTRA)
        unknown = expanded - set(available.keys())
        if unknown:
            group_names = sorted(BENCHMARK_GROUPS.keys())
            alias_names = sorted(BENCHMARK_ALIASES.keys())
            print(f"ERROR: Unknown benchmarks: {unknown}")
            print(f"Available: {sorted(available.keys())}")
            print(f"Groups: {group_names}, Aliases: {alias_names}")
            sys.exit(1)
        all_benchmarks = {k: available[k] for k in expanded if k in available}

    model_name = args.model.split("/")[-1]

    # Build display structure: group children under their parent for UI
    active_keys = set(all_benchmarks.keys())
    display_items: list[str | tuple[str, list[str]]] = []
    covered: set[str] = set()
    for item in EASI_8_DISPLAY_ORDER:
        if isinstance(item, tuple):
            group_name, children = item
            active_children = [c for c in children if c in active_keys]
            if active_children:
                display_items.append((group_name, active_children))
                covered.update(active_children)
        elif item in active_keys:
            display_items.append(item)
            covered.add(item)
    # Append any extra/ungrouped benchmarks not in EASI_8_DISPLAY_ORDER
    for key in all_benchmarks:
        if key not in covered:
            display_items.append(key)

    # Decide whether to use the rich progress display
    use_rich = not args.no_rich and sys.stdout.isatty()

    model_dir = output_dir / model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"eval_{model_name}_{timestamp}.log"

    # In rich mode, create the display early so ALL logs appear inside the panel.
    # The display redirects stdout, so subsequent print() calls are captured.
    display: ProgressDisplay | None = None
    if use_rich:
        display = ProgressDisplay(model_name, all_benchmarks, {}, display_items)
        display._log_path = str(log_path)
        display.start()  # starts stdout redirect

    # Print config info (only in non-rich mode; rich panel shows this via its sections)
    if not use_rich:
        print(f"Model:    {args.model}")
        print(f"Output:   {output_dir}")
        print(f"Datasets: {dataset_dir.resolve()}")
        print(f"Benchmarks ({len(display_items)}):")
        for item in display_items:
            if isinstance(item, tuple):
                group_name, children = item
                print(f"  {group_name}:")
                for child in children:
                    print(f"    {child} -> {all_benchmarks[child]}")
            else:
                print(f"  {item} -> {all_benchmarks[item]}")

    # ---- Phase 0: Check HF token (if --submit) ----
    hf_token = None
    if args.submit:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            env_file = repo_root / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):]
                    if line.startswith("HF_TOKEN="):
                        val = line.split("=", 1)[1].strip()
                        if val.startswith(('"', "'")):
                            hf_token = val.strip('"').strip("'")
                        else:
                            hf_token = val.split("#")[0].strip()

        if not hf_token:
            hf_token = None

        if display:
            masked = f"{hf_token[:4]}****{hf_token[-4:]}" if hf_token and len(hf_token) > 8 else "****"
            with display._lock:
                if hf_token:
                    display.env_checks.append(("HF_TOKEN", "ok", masked))
                    display.env_checks.append(("Submit", "ok", "enabled"))
                else:
                    display.env_checks.append(("HF_TOKEN", "failed", "not found"))

        if not hf_token:
            if display:
                time.sleep(1)
                display.stop()
            print("ERROR: --submit requires HF_TOKEN environment variable or .env file in project root")
            sys.exit(1)

    # ---- Phase 1: Prepare datasets (CPU only, with retries) ----
    try:
        datasets_ok = prepare_datasets(dataset_dir, all_benchmarks, display=display)
    except KeyboardInterrupt:
        if display:
            display.stop()
        print("\n\nInterrupted during dataset preparation.")
        sys.exit(130)

    if not datasets_ok:
        if not display:
            print("WARNING: Some datasets failed to download. Affected benchmarks may fail.")
            print("Fix network issues and rerun, or press Ctrl+C to abort.\n")
        # In rich mode, the data prep section already shows ✗ for failed datasets

    # Now that TSVs are downloaded, compute totals for progress tracking
    if display:
        for key, data_name in all_benchmarks.items():
            display.totals[key] = _count_tsv_rows(dataset_dir, data_name)

    # ---- Phase 2: Run benchmarks via VLMEvalKit ----
    run_py = repo_root / "VLMEvalKit" / "run.py"
    if not run_py.exists():
        if display:
            display.stop()
        print(f"ERROR: VLMEvalKit not found at {run_py}")
        print("Run: git submodule update --init VLMEvalKit")
        sys.exit(1)

    def _build_cmd(data_names: list[str], judge: str | None = None) -> list[str]:
        if args.nproc > 1:
            cmd = ["torchrun", f"--nproc-per-node={args.nproc}", str(run_py)]
        else:
            cmd = [sys.executable, str(run_py)]
        cmd += [
            "--data", *data_names,
            "--model", args.model,
            "--reuse",
            "--work-dir", str(output_dir),
        ]
        if judge:
            cmd += ["--judge", judge]
        if args.verbose:
            cmd.append("--verbose")
        cmd += extra_args
        return cmd

    def _run_vlmevalkit(cmd: list[str], phase_label: str):
        """Run a single VLMEvalKit invocation with monitoring."""
        if use_rich:
            display.set_phase(phase_label)
            stop_event = threading.Event()
            proc = None

            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("DIST_TIMEOUT", "7200")  # 2 hours
                with open(log_path, "ab", buffering=0) as log_f:
                    proc = subprocess.Popen(
                        cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                    )
                    monitor = threading.Thread(
                        target=_monitor_loop,
                        args=(display, model_dir, model_name, all_benchmarks, stop_event),
                        daemon=True,
                    )
                    monitor.start()
                    proc.wait()
                    stop_event.set()
                    monitor.join()
                    _poll_progress(display, model_dir, model_name, all_benchmarks,
                                   detect_failures=True)
            except KeyboardInterrupt:
                stop_event.set()
                display.stop()
                if proc is not None:
                    proc.terminate()
                    proc.wait()
                elapsed = time.time() - start
                print(f"\n\nInterrupted after {elapsed:.0f}s. Partial results may be available.")
                print(f"Log file: {log_path}")
                print("Rerun with --reuse to resume.")
                sys.exit(130)
        else:
            print(f"\n{'='*60}")
            print(phase_label)
            print(f"Command: {' '.join(cmd)}")
            print(f"{'='*60}\n")

            proc = None
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("DIST_TIMEOUT", "36000")
                with open(log_path, "ab") as log_f:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                    )
                    for line in iter(proc.stdout.readline, b""):
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
                        log_f.write(line)
                    proc.wait()
            except KeyboardInterrupt:
                if proc is not None:
                    proc.terminate()
                    proc.wait()
                elapsed = time.time() - start
                print(f"\n\nInterrupted after {elapsed:.0f}s. Partial results may be available.")
                print("Rerun with --reuse to resume.")
                sys.exit(130)

    start = time.time()

    if args.judge:
        # User explicitly specified judge — single invocation for all benchmarks
        data_names = list(all_benchmarks.values())
        cmd = _build_cmd(data_names, judge=args.judge)
        _run_vlmevalkit(cmd, f"Running evaluation (judge: {args.judge})")
    else:
        # Split benchmarks by default judge
        llm_judge_benchmarks = {
            k: v for k, v in all_benchmarks.items() if v in DEFAULT_JUDGE
        }
        exact_benchmarks = {
            k: v for k, v in all_benchmarks.items() if v not in DEFAULT_JUDGE
        }

        # Invocation 1: ALL benchmarks with exact_matching (inference + fast eval)
        all_data_names = list(all_benchmarks.values())
        cmd = _build_cmd(all_data_names, judge="exact_matching")
        _run_vlmevalkit(cmd, "Running evaluation (exact_matching)")

        # Invocation 2: LLM-judge benchmarks only (reuses inference, re-evaluates)
        if llm_judge_benchmarks:
            for key, data_name in llm_judge_benchmarks.items():
                judge_model = DEFAULT_JUDGE[data_name]
                # Reset eval status so the display shows re-evaluation in progress
                if display:
                    with display._lock:
                        display.eval_status[key] = ProgressDisplay.PENDING
                cmd = _build_cmd([data_name], judge=judge_model)
                _run_vlmevalkit(cmd, f"Re-evaluating {data_name} (judge: {judge_model})")

    elapsed = time.time() - start

    stderr_text = log_path.read_text(errors="replace") if log_path.exists() else ""

    # ---- Phase 3: Verify results per benchmark ----
    if display:
        display.set_phase("Verifying results")

    results = verify_results(output_dir, model_name, all_benchmarks, stderr_text)
    results_by_key = {r.key: r for r in results}
    failed_keys = [r.key for r in results if not r.success]

    # Update display status from verification results and collect summary lines
    summary_lines: list[str] = []

    def _update_display_for_result(key: str, r: BenchmarkResult):
        """Update display status and warnings from verification."""
        if not display:
            return
        if r.success:
            display.mark_infer_done(key)
            display.mark_eval_done(key)
        else:
            display.mark_failed(key)
        if r.errors:
            display.set_warning(key, r.errors[0])

    def _format_result(r: BenchmarkResult, indent: str = "  ") -> str:
        total_str = str(r.total) if r.total > 0 else "?"
        completed_str = str(r.completed) if r.completed >= 0 else "?"
        status = "OK" if r.success else "FAIL"
        line = f"{indent}[{status}] {r.key:<25} {completed_str}/{total_str} samples"
        if r.errors:
            line += f"\n{indent}     !! {r.errors[0]}"
        return line

    for item in display_items:
        if isinstance(item, tuple):
            group_name, children = item
            child_results = [results_by_key[c] for c in children if c in results_by_key]
            all_ok = all(r.success for r in child_results)
            grp_status = "OK" if all_ok else "FAIL"
            grp_completed = sum(r.completed for r in child_results if r.completed >= 0)
            grp_total = sum(r.total for r in child_results if r.total > 0)
            total_str = str(grp_total) if grp_total > 0 else "?"
            completed_str = str(grp_completed) if grp_completed >= 0 else "?"
            summary_lines.append(f"  [{grp_status}] {group_name:<25} {completed_str}/{total_str} samples")
            for child in children:
                if child in results_by_key:
                    r = results_by_key[child]
                    summary_lines.append(_format_result(r, indent="    "))
                    _update_display_for_result(child, r)
        else:
            if item in results_by_key:
                r = results_by_key[item]
                summary_lines.append(_format_result(r))
                _update_display_for_result(item, r)

    ok = sum(1 for r in results if r.success)
    summary_lines.append(f"\n{ok}/{len(results)} sub-benchmarks passed ({len(display_items)} benchmarks)")
    summary_lines.append(f"Results saved to: {output_dir}")
    if log_path.exists():
        summary_lines.append(f"Full log: {log_path}")

    rerun_cmd = ""
    if failed_keys:
        rerun_cmd = f"python scripts/submissions/run_easi_eval.py --model {args.model} --benchmarks {','.join(failed_keys)}"
        if args.nproc > 1:
            rerun_cmd += f" --nproc {args.nproc}"
        if args.judge:
            rerun_cmd += f" --judge {args.judge}"
        if args.dataset_dir:
            rerun_cmd += f" --dataset-dir {args.dataset_dir}"
        if str(output_dir) != "./eval_results":
            rerun_cmd += f" --output-dir {output_dir}"
        summary_lines.append(f"\nRerun failed:\n  {rerun_cmd}")

    # ---- Phase 4: Post-processing (build submission payload + archive) ----
    from postprocess import build_payload, build_results_archive

    submission_configs = json.loads(args.submission_configs) if args.submission_configs else {}

    if display:
        display.set_phase("Building submission")

    payload = build_payload(model_dir, model_name, all_benchmarks, submission_configs)
    json_path = output_dir / "easi_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    zip_path = build_results_archive(model_dir, model_name, output_dir)

    if display:
        display.results_info = {
            "scores": payload["scores"],
            "payload_path": str(json_path),
            "zip_path": str(zip_path),
        }

    # ---- Phase 5: Submit to leaderboard (if --submit) ----
    def _set_submission(status: str, detail: str = ""):
        if display:
            with display._lock:
                display.submission_status = (status, detail)

    if args.submit and hf_token:
        from postprocess import validate_payload_for_submit, submit_results

        if failed_keys:
            # Don't submit if any benchmark failed
            msg = f"Skipping submission: {len(failed_keys)} benchmark(s) failed ({', '.join(failed_keys)})"
            _set_submission("failed", msg)
            if not display:
                print(f"\n{msg}")
        else:
            validation_errors = validate_payload_for_submit(payload, zip_path=zip_path)
            if validation_errors:
                error_msg = "Payload validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                _set_submission("failed", error_msg)
                if not display:
                    print(f"\n{error_msg}")
            else:
                _set_submission("submitting")

                def _on_retry(attempt, max_retries, error_msg):
                    msg = f"Retry {attempt}/{max_retries}: {error_msg}"
                    _set_submission("retrying", msg)
                    if not display:
                        print(f"  {msg}")

                success, message = submit_results(
                    payload, zip_path, hf_token, on_retry=_on_retry,
                )
                _set_submission("success" if success else "failed", message)
                if not display:
                    if success:
                        print(f"\nSubmission: {message}")
                    else:
                        print(f"\nSubmission failed: {message}")

    if display:
        display.set_phase(f"Complete ({elapsed:.0f}s)")
        time.sleep(0.5)
        display.stop()
    else:
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY (total {elapsed:.0f}s)")
        print(f"{'='*60}")
        for line in summary_lines:
            print(line)
        print(f"\nSubmission payload: {json_path}")
        print(f"Results archive: {zip_path}")

    if failed_keys:
        sys.exit(1)


if __name__ == "__main__":
    main()
