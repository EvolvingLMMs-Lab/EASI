"""YAML template inheritance utilities.

Provides deep_merge() and resolve_task_yaml() for the 'extends' mechanism.
Split YAMLs reference a base template via 'extends: _base.yaml'.
The base and split configs are deep-merged (split values win on conflicts).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts.

    - Dicts: merged recursively
    - Lists/scalars: replaced entirely (no appending)
    - Missing keys in override: base value kept
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_task_yaml(yaml_path: Path) -> dict:
    """Load a task YAML, resolving 'extends' inheritance if present.

    The 'extends' value is a relative path from the YAML file's directory.
    Supports single-level inheritance only (base cannot extend another base).

    Returns:
        Fully resolved config dict (extends key removed).

    Raises:
        FileNotFoundError: If the base YAML referenced by extends doesn't exist.
        ValueError: If chained extends are detected (base also has extends).
    """
    config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(config, dict):
        return config or {}

    extends = config.pop("extends", None)
    if extends is None:
        return config

    base_path = yaml_path.parent / extends
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base YAML not found: {base_path} "
            f"(referenced by extends in {yaml_path})"
        )

    base_config = yaml.safe_load(base_path.read_text())
    if not isinstance(base_config, dict):
        base_config = {}

    if "extends" in base_config:
        raise ValueError(
            f"Chained extends not supported: {base_path} also has 'extends'"
        )

    return deep_merge(base_config, config)
