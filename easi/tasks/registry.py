"""Task registry with auto-discovery from task.yaml files.

Scans easi/tasks/*/task.yaml to discover available tasks (benchmarks).

Usage:
    list_tasks() → ["dummy_task", "objectnav_hm3d", ...]
    get_task_entry("dummy_task") → TaskEntry(...)
    load_task_class("dummy_task") → DummyTask class
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from easi.tasks.yaml_utils import resolve_task_yaml
from easi.utils.import_utils import import_class as _import_class
from easi.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskEntry:
    """Registry entry for a task."""

    name: str
    display_name: str
    description: str
    simulator_key: str
    task_class: str  # fully qualified class name
    max_steps: int
    config_path: Path


# Module-level registry populated on first access
_registry: dict[str, TaskEntry] | None = None


def _discover_tasks(tasks_dir: Path | None = None) -> dict[str, TaskEntry]:
    """Scan task directories for .yaml config files.

    Discovers all *.yaml files in immediate subdirectories of tasks_dir.
    Each .yaml file registers as its own task (enabling multi-split tasks).
    """
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent
    entries: dict[str, TaskEntry] = {}

    for subdir in sorted(tasks_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue

        # Find all .yaml files in this task folder
        yaml_files = sorted(subdir.glob("*.yaml"))

        for task_yaml_path in yaml_files:
            try:
                config = resolve_task_yaml(task_yaml_path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", task_yaml_path, e)
                continue

            if not isinstance(config, dict) or "name" not in config:
                continue

            name = config["name"]
            entries[name] = TaskEntry(
                name=name,
                display_name=config.get("display_name", name),
                description=config.get("description", ""),
                simulator_key=config["simulator"],
                task_class=config["task_class"],
                max_steps=config.get("max_steps", 500),
                config_path=task_yaml_path,
            )

            logger.trace("Discovered task: %s from %s", name, task_yaml_path.name)

    return entries


def _get_registry() -> dict[str, TaskEntry]:
    """Get the task registry, discovering on first access."""
    global _registry
    if _registry is None:
        _registry = _discover_tasks()
    return _registry


def get_task_entry(name: str) -> TaskEntry:
    """Look up a task entry by name.

    Raises:
        KeyError: If the task is not found.
    """
    registry = _get_registry()
    if name not in registry:
        available = list_tasks()
        raise KeyError(f"Task '{name}' not found. Available: {available}")
    return registry[name]


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_get_registry().keys())


def load_task_class(name: str):
    """Import and return the task class for the given name."""
    entry = get_task_entry(name)
    return _import_class(entry.task_class)


def refresh() -> None:
    """Force re-discovery of tasks."""
    global _registry
    _registry = None
