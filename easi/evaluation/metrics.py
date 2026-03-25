"""Metric aggregation utilities."""
from __future__ import annotations

from easi.core.episode import EpisodeRecord


def default_aggregate(records: list[EpisodeRecord]) -> dict:
    """Default aggregation: average all numeric keys from episode_results.

    Args:
        records: List of EpisodeRecord objects (one per episode).

    Returns:
        Summary dict with avg_<key> for each numeric key,
        plus convenience aliases (success_rate, avg_steps).
    """
    if not records:
        return {}

    summary: dict = {}

    # Collect all numeric keys from episode_results
    numeric_keys: dict[str, list[float]] = {}
    for r in records:
        for key, value in r.episode_results.items():
            if isinstance(value, (int, float)):
                numeric_keys.setdefault(key, []).append(float(value))

    # Average each numeric metric over ALL episodes (not just those that emitted the key).
    # Failed episodes may not emit task-specific keys — they should contribute 0, not be excluded.
    total = len(records)
    for key, values in numeric_keys.items():
        summary[f"avg_{key}"] = round(sum(values) / total, 4)

    # Convenience aliases
    if "avg_success" in summary:
        summary["success_rate"] = summary["avg_success"]
    if "avg_task_success" in summary:
        summary["success_rate"] = summary["avg_task_success"]
    if "avg_num_steps" in summary:
        summary["avg_steps"] = summary["avg_num_steps"]

    return summary
