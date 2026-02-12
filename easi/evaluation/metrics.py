"""Metric aggregation utilities."""

from __future__ import annotations


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-episode metrics into a summary dict."""
    if not results:
        return {"num_episodes": 0}

    summary = {"num_episodes": len(results)}

    # Collect all numeric keys
    numeric_keys: dict[str, list[float]] = {}
    for r in results:
        for key, value in r.items():
            if isinstance(value, (int, float)):
                numeric_keys.setdefault(key, []).append(float(value))

    # Average each numeric metric
    for key, values in numeric_keys.items():
        summary[f"avg_{key}"] = round(sum(values) / len(values), 4)

    # Convenience aliases
    if "avg_success" in summary:
        summary["success_rate"] = summary["avg_success"]
    if "avg_num_steps" in summary:
        summary["avg_steps"] = summary["avg_num_steps"]

    return summary
