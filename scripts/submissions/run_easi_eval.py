#!/usr/bin/env python3
"""Run EASI benchmark evaluation via VLMEvalKit or lmms-eval.

Runs all EASI-8 benchmarks (optionally including extras) on a specified model
using the selected backend. Handles dataset preparation with retry logic before
launching GPU-heavy inference.

Usage (VLMEvalKit — default):
    python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct
    python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --output-dir ./results --include-extra

Usage (lmms-eval):
    python scripts/submissions/run_easi_eval.py --backend lmms-eval \\
        --model qwen2_vl --model-args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2' \\
        --nproc 8

Installation (VLMEvalKit):
    - Update requirements.txt with "transformers>=4.45,<5", "torch==2.7.1", "torchvision==0.22.1", "setuptools<81"
    - Run the following:
    uv venv -p 3.11
    source .venv/bin/activate
    cd VLMEvalKit
    uv pip install .
    uv pip install hf_transfer  # optional, for faster HF downloads (but can be flaky on some networks)
    uv pip install flash-attn --no-build-isolation

Installation (lmms-eval):
    pip install lmms-eval accelerate
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import get_backend
from backends.vlmevalkit import (
    VLMEvalKitAdapter,
    verify_results,
    BenchmarkResult,
    prepare_datasets,
)

# ---------------------------------------------------------------------------
# Benchmark registry (backend-agnostic keys)
# ---------------------------------------------------------------------------

EASI_8_KEYS = [
    "vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial",
    "site_image", "site_video", "blink", "3dsrbench", "embspatial",
]

EXTRA_KEYS = [
    "mmsi_video_bench", "omnispatial_(manual_cot)", "spar_bench", "vsi_debiased",
]

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
# Progress display
# ---------------------------------------------------------------------------

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
        display_items: list[str | tuple[str, list[str]]],
    ):
        self.model_name = model_name
        self.benchmarks = benchmarks  # flat key -> data_name
        self.display_items = display_items
        # Single status per benchmark: pending / running / done / failed
        self.status: dict[str, str] = {k: self.PENDING for k in benchmarks}
        self.status_detail: dict[str, str] = {}  # optional "(0.8% failures)" etc.
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

    def _status_icon(self, status: str) -> str:
        """Return styled icon for status, including spinner frame for RUNNING."""
        if status == self.DONE:
            return "[green]\u2713[/green]"
        if status == self.FAILED:
            return "[red]\u2717[/red]"
        if status == self.RUNNING:
            frames = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827"
            # Time-based frame selection: stable at ~8 FPS regardless of render rate
            idx = int(time.time() * 8) % len(frames)
            return f"[yellow]{frames[idx]}[/yellow]"
        return "[dim]\u2500[/dim]"  # pending

    def _group_status(self, children: list[str]) -> str:
        """Aggregate child statuses into a group status."""
        statuses = {self.status[c] for c in children}
        if statuses == {self.DONE}:
            return self.DONE
        if self.FAILED in statuses:
            return self.FAILED
        if self.RUNNING in statuses or (self.DONE in statuses and self.PENDING in statuses):
            return self.RUNNING
        return self.PENDING

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
                    child_statuses = {self.data_prep.get(c, ("pending", ""))[0] for c in children}
                    if len(child_statuses) == 1:
                        grp_status = next(iter(child_statuses))
                    else:
                        grp_status = next(
                            (s for s in ("done", "failed", "downloading", "pending") if s in child_statuses),
                            "pending",
                        )
                    grp_icon = self._render_data_prep_icon(grp_status)
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
                icon = self._status_icon(
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
        table.add_column("Status", min_width=30)

        display_count = len(self.display_items)
        done_display = 0
        pending_warnings: list[tuple[str, str]] = []

        for item in self.display_items:
            if isinstance(item, tuple):
                group_name, children = item
                grp_status = self._group_status(children)
                grp_icon = self._status_icon(grp_status)
                table.add_row(group_name, f"{grp_icon} {grp_status}")
                for child in children:
                    child_status = self.status[child]
                    child_icon = self._status_icon(child_status)
                    detail = self.status_detail.get(child, "")
                    detail_str = f"  [dim]{detail}[/dim]" if detail else ""
                    table.add_row(
                        f"  [dim]{child}[/dim]",
                        f"{child_icon} {child_status}{detail_str}",
                    )
                    if child in self.warnings:
                        pending_warnings.append((child, self.warnings[child]))
                if grp_status in (self.DONE, self.FAILED):
                    done_display += 1
            else:
                status = self.status[item]
                icon = self._status_icon(status)
                detail = self.status_detail.get(item, "")
                detail_str = f"  [dim]{detail}[/dim]" if detail else ""
                table.add_row(item, f"{icon} {status}{detail_str}")
                if item in self.warnings:
                    pending_warnings.append((item, self.warnings[item]))
                if status in (self.DONE, self.FAILED):
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
        any_started = any(s != self.PENDING for s in self.status.values())
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
        # screen=True uses alternate screen buffer — prevents smearing
        # on terminal resize and gives clean redraws.
        self._live = Live(
            self,
            refresh_per_second=8,
            console=console,
            screen=True,
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None
        if hasattr(self, "_orig_stdout"):
            sys.stdout = self._orig_stdout

    def mark_done(self, key: str, detail: str = ""):
        with self._lock:
            self.status[key] = self.DONE
            if detail:
                self.status_detail[key] = detail
            elif key in self.status_detail:
                del self.status_detail[key]

    def mark_running(self, key: str, detail: str = ""):
        with self._lock:
            self.status[key] = self.RUNNING
            if detail:
                self.status_detail[key] = detail
            elif key in self.status_detail:
                del self.status_detail[key]

    def mark_failed(self, key: str, detail: str = ""):
        with self._lock:
            self.status[key] = self.FAILED
            if detail:
                self.status_detail[key] = detail

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase

    def set_data_prep(self, key: str, status: str, detail: str = ""):
        with self._lock:
            self.data_prep[key] = (status, detail)

    def set_warning(self, key: str, warning: str):
        with self._lock:
            self.warnings[key] = warning



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run EASI benchmark evaluation via VLMEvalKit or lmms-eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (VLMEvalKit):
  # Run EASI-8 core benchmarks
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct

  # Run all benchmarks including extras
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --include-extra

  # Run specific benchmarks only
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --benchmarks vsi_bench,blink,site

  # With data parallelism (auto-judge re-runs benchmarks whose extraction
  # failure rate exceeds --extraction-threshold using --judge-model)
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --nproc 8

  # Skip the extraction quality check and trust exact_matching scores
  python scripts/submissions/run_easi_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --no-judge

Examples (lmms-eval):
  # Run EASI-8 via lmms-eval
  python scripts/submissions/run_easi_eval.py --backend lmms-eval \\
      --model qwen2_vl \\
      --model-args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2' \\
      --nproc 8

  # Force re-evaluation (skip resume)
  python scripts/submissions/run_easi_eval.py --backend lmms-eval --rerun \\
      --model qwen2_vl --model-args 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct'
        """,
    )
    parser.add_argument("--model", required=True, help="Model name (HuggingFace ID or VLMEvalKit model name)")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory (default: ./eval_results)")
    parser.add_argument("--include-extra", action="store_true", help="Also run extra benchmarks")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark keys (e.g. vsi_bench,blink,site)")
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs for data parallelism via torchrun")
    parser.add_argument("--no-judge", action="store_true",
                        help="VLMEvalKit only: skip extraction quality check, use exact_matching scores")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-1120",
                        help="VLMEvalKit only: judge model for auto re-evaluation (default: gpt-4o-1120)")
    parser.add_argument("--extraction-threshold", type=float, default=0.025,
                        help="VLMEvalKit only: extraction failure rate threshold for triggering judge (default: 0.025)")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory (default: ./datasets in repo root)")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to VLMEvalKit (prints per-sample model responses)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich progress display (passthrough raw subprocess output)")
    parser.add_argument("--submission-configs", type=str, default=None,
                        help='JSON string with submission metadata, e.g.: '
                             '\'{"modelName": "org/model", "modelType": "pretrained", "precision": "bfloat16"}\'')
    parser.add_argument("--submit", action="store_true",
                        help="Submit results to EASI leaderboard after evaluation (requires HF token)")
    # Backend selection
    parser.add_argument("--backend", choices=["vlmevalkit", "lmms-eval"], default="vlmevalkit",
                        help="Evaluation backend (default: vlmevalkit)")
    parser.add_argument("--model-args", type=str, default=None,
                        help="Model arguments for lmms-eval (e.g. 'pretrained=Qwen/...,attn_implementation=flash_attention_2')")
    parser.add_argument("--accelerate", dest="use_accelerate", action="store_true", default=True,
                        help="Use accelerate launch for lmms-eval (default)")
    parser.add_argument("--no-accelerate", dest="use_accelerate", action="store_false",
                        help="Don't use accelerate for lmms-eval")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-evaluation (skip resume logic)")

    args, extra_args = parser.parse_known_args()

    # Backend-specific validation
    if args.backend == "lmms-eval":
        if not args.model_args:
            print("ERROR: --model-args is required with --backend lmms-eval")
            sys.exit(1)
        if args.no_judge:
            print("WARNING: --no-judge is ignored for lmms-eval backend")
    elif args.backend == "vlmevalkit":
        if args.model_args:
            print("WARNING: --model-args is ignored for vlmevalkit backend")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent.parent  # scripts/submissions/ -> repo root

    # Create adapter early
    if args.backend == "lmms-eval":
        adapter = get_backend("lmms-eval",
            model_args=args.model_args,
            use_accelerate=args.use_accelerate,
            rerun=args.rerun,
        )
    else:
        adapter = get_backend("vlmevalkit",
            repo_root=repo_root,
            rerun=args.rerun,
        )

    # Build benchmark list using adapter's TASK_MAP
    all_keys = list(EASI_8_KEYS)
    if args.include_extra:
        all_keys += EXTRA_KEYS

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
        available = set(adapter.TASK_MAP.keys())
        unknown = expanded - available
        if unknown:
            group_names = sorted(BENCHMARK_GROUPS.keys())
            alias_names = sorted(BENCHMARK_ALIASES.keys())
            print(f"ERROR: Unknown benchmarks: {unknown}")
            print(f"Available: {sorted(available)}")
            print(f"Groups: {group_names}, Aliases: {alias_names}")
            sys.exit(1)
        all_keys = [k for k in expanded if k in adapter.TASK_MAP]

    all_benchmarks = {k: adapter.TASK_MAP[k] for k in all_keys if k in adapter.TASK_MAP}

    # Model name and model directory depend on backend
    if args.backend == "lmms-eval":
        model_name = args.model  # model type for display
        # lmms-eval uses sanitized pretrained path as directory name
        model_args_dict = dict(kv.split("=", 1) for kv in args.model_args.split(",") if "=" in kv)
        pretrained = model_args_dict.get("pretrained", args.model)
        sanitized = pretrained.replace("/", "__")
        model_dir = output_dir / sanitized
    else:
        model_name = args.model.split("/")[-1]
        model_dir = output_dir / model_name

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"eval_{model_name}_{timestamp}.log"

    # In rich mode, create the display early so ALL logs appear inside the panel.
    # The display redirects stdout, so subsequent print() calls are captured.
    display: ProgressDisplay | None = None
    if use_rich:
        display = ProgressDisplay(model_name, all_benchmarks, display_items)
        display._log_path = str(log_path)
        display.start()  # starts stdout redirect

    # Print config info (only in non-rich mode; rich panel shows this via its sections)
    if not use_rich:
        print(f"Backend:  {args.backend}")
        print(f"Model:    {args.model}")
        print(f"Output:   {output_dir}")
        if args.backend == "vlmevalkit":
            dataset_dir_display = (Path(args.dataset_dir) if args.dataset_dir else repo_root / "datasets").resolve()
            print(f"Datasets: {dataset_dir_display}")
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
    if args.backend == "vlmevalkit":
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else repo_root / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        os.environ["LMUData"] = str(dataset_dir.resolve())
        # NOTE: We do NOT override HF_HUB_CACHE. Both our prep step and
        # VLMEvalKit's internal snapshot_download use the default HF cache
        # (~/.cache/huggingface/hub/). This avoids re-downloading model weights
        # to a different location.

        try:
            datasets_ok = adapter.prepare_datasets(all_benchmarks, dataset_dir, display=display)
        except KeyboardInterrupt:
            if display:
                display.stop()
            print("\n\nInterrupted during dataset preparation.")
            sys.exit(130)

        if not datasets_ok:
            if not display:
                print("WARNING: Some datasets failed to download. Affected benchmarks may fail.")
                print("Fix network issues and rerun, or press Ctrl+C to abort.\n")
            # In rich mode, the data prep section already shows X for failed datasets

    else:
        # lmms-eval manages its own datasets
        datasets_ok = True
        if display:
            for key in all_benchmarks:
                display.set_data_prep(key, "done", "managed by lmms-eval")

    # ---- Phase 2: Run benchmarks ----
    start = time.time()

    # Resume logic: find already-completed tasks
    completed = set()
    if not args.rerun:
        completed = adapter.find_completed_tasks(model_dir, model_name, all_benchmarks)
        if display:
            for key in completed:
                display.mark_done(key)

    pending_benchmarks = {k: v for k, v in all_benchmarks.items() if k not in completed}

    # Track benchmarks that got re-evaluated with LLM judge (VLMEvalKit only)
    judged_benchmarks: dict[str, str] = {}  # key -> judge_model

    if args.backend == "lmms-eval":
        # ---- lmms-eval subprocess (one benchmark at a time) ----
        def _run_lmmseval(cmd: list[str], phase_label: str):
            """Run a single lmms-eval subprocess."""
            if use_rich:
                display.set_phase(phase_label)
                proc = None
                try:
                    env = os.environ.copy()
                    env.update(adapter.get_env_overrides())
                    with open(log_path, "ab", buffering=0) as log_f:
                        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
                        proc.wait()
                except KeyboardInterrupt:
                    display.stop()
                    if proc:
                        proc.terminate()
                        proc.wait()
                    print(f"\n\nInterrupted. Partial results may be available.")
                    print(f"Log: {log_path}")
                    sys.exit(130)
            else:
                print(f"\n{'='*60}")
                print(phase_label)
                print(f"Command: {' '.join(cmd)}")
                print(f"{'='*60}\n")
                proc = None
                try:
                    env = os.environ.copy()
                    env.update(adapter.get_env_overrides())
                    with open(log_path, "ab") as log_f:
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                        for line in iter(proc.stdout.readline, b""):
                            sys.stdout.buffer.write(line)
                            sys.stdout.buffer.flush()
                            log_f.write(line)
                        proc.wait()
                except KeyboardInterrupt:
                    if proc:
                        proc.terminate()
                        proc.wait()
                    print(f"\n\nInterrupted. Rerun to resume.")
                    sys.exit(130)

        if pending_benchmarks:
            # Run each benchmark individually to avoid dataset filelock
            # deadlocks when multiple accelerate workers try to download
            # different datasets simultaneously.
            for key, task_name in pending_benchmarks.items():
                if display:
                    display.mark_running(key)

                single = {key: task_name}
                cmd = adapter.build_cmd(args.model, single, output_dir, args.nproc,
                                        extra_args=extra_args, verbose=args.verbose)
                _run_lmmseval(cmd, f"Running {key} ({task_name})")

                # Check completion after each benchmark
                completion = adapter.detect_completion(model_dir, model_name, {key: task_name})
                if display:
                    if completion.get(key):
                        display.mark_done(key)
                    else:
                        display.mark_failed(key)
                elif not completion.get(key):
                    print(f"  WARNING: {key} may have failed")
        elif not use_rich:
            print("All benchmarks already completed. Skipping to postprocessing.")

    else:
        # ---- VLMEvalKit subprocess ----
        run_py = repo_root / "VLMEvalKit" / "run.py"
        if not run_py.exists():
            if display:
                display.stop()
            print(f"ERROR: VLMEvalKit not found at {run_py}")
            print("Run: git submodule update --init VLMEvalKit")
            sys.exit(1)

        def _run_subprocess(cmd: list[str], phase_label: str):
            """Run a VLMEvalKit subprocess with log capture."""
            if use_rich:
                display.set_phase(phase_label)
                proc = None
                try:
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    env.setdefault("DIST_TIMEOUT", "7200")  # 2 hours
                    with open(log_path, "ab", buffering=0) as log_f:
                        proc = subprocess.Popen(
                            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                        )
                        proc.wait()
                except KeyboardInterrupt:
                    display.stop()
                    if proc is not None:
                        proc.terminate()
                        proc.wait()
                    elapsed = time.time() - start
                    print(f"\n\nInterrupted after {elapsed:.0f}s. Partial results may be available.")
                    print(f"Log file: {log_path}")
                    print("Rerun to resume.")
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
                    print("Rerun to resume.")
                    sys.exit(130)

        # --- Phase 2a: Run ALL pending benchmarks with exact_matching ---
        if pending_benchmarks:
            if display:
                for key in pending_benchmarks:
                    display.mark_running(key)
            cmd = adapter.build_cmd(
                args.model, pending_benchmarks, output_dir, args.nproc,
                extra_args=extra_args, judge="exact_matching", verbose=args.verbose,
            )
            _run_subprocess(cmd, "Evaluating (exact_matching)")

            # Update status from completion check
            completion = adapter.detect_completion(model_dir, model_name, pending_benchmarks)
            if display:
                for key, done in completion.items():
                    if done:
                        display.mark_done(key)
                    else:
                        display.mark_failed(key)

        # --- Phase 2b: Extraction quality check + judge re-run ---
        if not args.no_judge:
            if display:
                display.set_phase("Checking extraction quality")

            # Check across ALL benchmarks (not just pending) — if user reran
            # with existing results, still check them
            reports = adapter.check_extraction_quality(model_dir, model_name, all_benchmarks)

            # Write extraction report
            report_data = {
                "threshold": args.extraction_threshold,
                "judge_model": args.judge_model,
                "benchmarks": {
                    key: {
                        "total": r.total,
                        "failed": r.failed,
                        "failure_rate": round(r.failure_rate, 4),
                        "method": r.method,
                    }
                    for key, r in reports.items()
                },
            }
            report_path = output_dir / "extraction_report.json"
            report_path.write_text(json.dumps(report_data, indent=2))

            # Identify benchmarks needing judge re-evaluation
            needs_judge = {
                key: all_benchmarks[key]
                for key, r in reports.items()
                if r.method != "skipped_no_artifact" and r.failure_rate > args.extraction_threshold
            }

            # If either site_image or site_video needs judge, include both —
            # the combined site score requires consistent extraction method.
            if "site_image" in needs_judge or "site_video" in needs_judge:
                for k in ("site_image", "site_video"):
                    if k in all_benchmarks:
                        needs_judge[k] = all_benchmarks[k]

            # Update display with failure rate details
            if display:
                for key, r in reports.items():
                    pct = f"{r.failure_rate*100:.1f}%"
                    if key in needs_judge:
                        display.mark_running(key, f"needs judge ({pct} failures)")
                    else:
                        display.mark_done(key, f"({pct} failures)")

            # --- Phase 2c: Re-evaluate failing benchmarks with judge ---
            if needs_judge:
                if display:
                    display.set_phase(f"Re-evaluating with {args.judge_model}")

                for key, data_name in needs_judge.items():
                    # Archive existing exact_matching artifacts
                    adapter.archive_artifacts(model_dir, model_name, data_name)

                    if display:
                        display.mark_running(key, "(judge)")

                    single = {key: data_name}
                    judge_cmd = adapter.build_judge_cmd(
                        args.model, single, output_dir, args.nproc,
                        args.judge_model, extra_args=extra_args,
                    )
                    _run_subprocess(
                        judge_cmd,
                        f"Re-evaluating {key} ({args.judge_model})",
                    )

                    # Track that this benchmark was re-evaluated
                    judged_benchmarks[key] = args.judge_model

                    # Update completion
                    completion = adapter.detect_completion(model_dir, model_name, single)
                    if display:
                        if completion.get(key):
                            display.mark_done(key, "(judge)")
                        else:
                            display.mark_failed(key, "(judge failed)")
        elif not use_rich:
            print("--no-judge: skipping extraction quality check")

    elapsed = time.time() - start

    stderr_text = log_path.read_text(errors="replace") if log_path.exists() else ""

    # ---- Phase 3: Verify results per benchmark ----
    if display:
        display.set_phase("Verifying results")

    summary_lines: list[str] = []

    if args.backend == "vlmevalkit":
        results = verify_results(output_dir, model_name, all_benchmarks, stderr_text)
        results_by_key = {r.key: r for r in results}
        failed_keys = [r.key for r in results if not r.success]

        # Update display status from verification results and collect summary lines

        def _update_display_for_result(key: str, r: BenchmarkResult):
            """Update display status and warnings from verification."""
            if not display:
                return
            if r.success:
                display.mark_done(key)
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
    else:
        # lmms-eval verification
        completion = adapter.detect_completion(model_dir, model_name, all_benchmarks)
        failed_keys = [k for k, done in completion.items() if not done]
        if display:
            for key, done in completion.items():
                if done:
                    display.mark_done(key)
                else:
                    display.mark_failed(key)
        if not use_rich:
            for key in all_benchmarks:
                status = "OK" if completion.get(key) else "FAIL"
                summary_lines.append(f"  [{status}] {key}")

        ok = sum(1 for done in completion.values() if done)
        summary_lines.append(f"\n{ok}/{len(completion)} benchmarks passed ({len(display_items)} benchmarks)")

    summary_lines.append(f"Results saved to: {output_dir}")
    if log_path.exists():
        summary_lines.append(f"Full log: {log_path}")

    rerun_cmd = ""
    if failed_keys:
        rerun_cmd = f"python scripts/submissions/run_easi_eval.py --model {args.model} --benchmarks {','.join(failed_keys)}"
        if args.backend != "vlmevalkit":
            rerun_cmd += f" --backend {args.backend}"
        if args.model_args:
            rerun_cmd += f" --model-args '{args.model_args}'"
        if args.nproc > 1:
            rerun_cmd += f" --nproc {args.nproc}"
        if args.no_judge:
            rerun_cmd += " --no-judge"
        if args.judge_model and args.judge_model != "gpt-4o-1120":
            rerun_cmd += f" --judge-model {args.judge_model}"
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

    payload = build_payload(
        model_dir, model_name, all_benchmarks, submission_configs,
        backend_adapter=adapter,
        judged_benchmarks=judged_benchmarks or None,
    )
    json_path = output_dir / "easi_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    zip_path = build_results_archive(model_dir, model_name, output_dir,
                                      backend_adapter=adapter)

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
