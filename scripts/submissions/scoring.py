"""Scoring utilities for EASI benchmark results.

Provides functions to load VLMEvalKit extract_matching xlsx files and
compute metrics (accuracy, CAA).  Designed to be imported by the main
eval script or used standalone.

Usage (as a library):
    from scripts.scoring import score_sitebench, score_benchmark

    # Score a single benchmark
    result = score_benchmark(model_dir, model_name, "SiteBenchImage")

    # Score SiteBench (combined image + video)
    result = score_sitebench(model_dir, model_name)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _count_options(options_str: object) -> int | None:
    """Count answer options from an options/candidates column value."""
    if pd.isna(options_str):
        return None
    try:
        opts = ast.literal_eval(str(options_str))
        if isinstance(opts, list):
            return len(opts)
    except (ValueError, SyntaxError):
        pass
    return None


def load_extract_matching(xlsx_path: Path | str) -> pd.DataFrame | None:
    """Load a VLMEvalKit ``*_extract_matching.xlsx`` into a scoring DataFrame.

    Returns a DataFrame with columns ``[hit, category, random_chance]``,
    or ``None`` if required columns are missing.
    """
    df = pd.read_excel(xlsx_path)

    if "hit" not in df.columns or "category" not in df.columns:
        return None

    # Options column may be named "options" or "candidates"
    opts_col = next((c for c in ("options", "candidates") if c in df.columns), None)
    if opts_col is None:
        return None

    df["n_options"] = df[opts_col].apply(_count_options)
    df = df.dropna(subset=["n_options"])
    df["random_chance"] = 1.0 / df["n_options"]

    return df[["hit", "category", "random_chance"]]


def compute_caa(df: pd.DataFrame) -> dict[str, dict]:
    """Compute accuracy and CAA (Calibrated Accuracy Adjusted) from a scoring DataFrame.

    Uses the SiteBench official formula::

        CAA = sum(hit_i - 1/n_i) / sum(1 - 1/n_i)

    Returns::

        {
            "overall": {"acc": float, "caa": float, "n": int},
            "<category>": {"acc": float, "caa": float, "n": int},
            ...
        }
    """

    def _caa(group: pd.DataFrame) -> float:
        numerator = (group["hit"] - group["random_chance"]).sum()
        denominator = (1 - group["random_chance"]).sum()
        return float(numerator / denominator) if denominator > 0 else 0.0

    results: dict[str, dict] = {}
    results["overall"] = {
        "acc": float(df["hit"].mean()),
        "caa": _caa(df),
        "n": len(df),
    }
    for cat, group in df.groupby("category"):
        results[str(cat)] = {
            "acc": float(group["hit"].mean()),
            "caa": _caa(group),
            "n": len(group),
        }
    return results


# ---------------------------------------------------------------------------
# Finding result files
# ---------------------------------------------------------------------------

def _find_result_file(model_dir: Path, model_name: str, data_name: str, suffix: str) -> Path | None:
    """Find the first file matching ``{model}_{data}*{suffix}`` in *model_dir*."""
    matches = sorted(Path(model_dir).glob(f"{model_name}_{data_name}*{suffix}"))
    return matches[0] if matches else None


def find_acc_csv(model_dir: Path, model_name: str, data_name: str) -> Path | None:
    """Find the ``_acc.csv`` result file for a benchmark."""
    return _find_result_file(model_dir, model_name, data_name, "_acc.csv")


def find_extract_matching(model_dir: Path, model_name: str, data_name: str) -> Path | None:
    """Find the ``_extract_matching.xlsx`` for a benchmark."""
    return _find_result_file(model_dir, model_name, data_name, "_extract_matching.xlsx")


# ---------------------------------------------------------------------------
# High-level scoring
# ---------------------------------------------------------------------------

def score_benchmark(model_dir: Path | str, model_name: str, data_name: str) -> dict | None:
    """Score a single benchmark from its extract_matching xlsx.

    Returns the ``compute_caa`` result dict, or ``None`` if not found.
    """
    model_dir = Path(model_dir)
    xlsx = find_extract_matching(model_dir, model_name, data_name)
    if xlsx is None:
        return None
    df = load_extract_matching(xlsx)
    if df is None:
        return None
    return compute_caa(df)


def score_sitebench(model_dir: Path | str, model_name: str) -> dict | None:
    """Score SiteBench by combining image and video extract_matching results.

    Returns combined CAA/accuracy, or ``None`` if neither file is found.
    """
    model_dir = Path(model_dir)
    dfs: list[pd.DataFrame] = []

    for data_name in ("SiteBenchImage", "SiteBenchVideo_32frame"):
        xlsx = find_extract_matching(model_dir, model_name, data_name)
        if xlsx is not None:
            df = load_extract_matching(xlsx)
            if df is not None:
                dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    return compute_caa(combined)


def format_results(name: str, results: dict[str, dict]) -> str:
    """Format scoring results as a human-readable string."""
    lines = [name]
    overall = results.get("overall", {})
    lines.append(f"  Overall: CAA={overall.get('caa', 0):.4f}, Acc={overall.get('acc', 0):.4f}, N={overall.get('n', 0)}")
    for cat in sorted(results):
        if cat == "overall":
            continue
        r = results[cat]
        lines.append(f"  {cat}: CAA={r['caa']:.4f}, Acc={r['acc']:.4f}, N={r['n']}")
    return "\n".join(lines)
