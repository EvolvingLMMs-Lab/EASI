"""Model registry with manifest-based auto-discovery.

Scans easi/llm/models/*/manifest.yaml to discover available custom model
server configurations. Follows the same pattern as the simulator registry.

Lookup semantics:
- list_models() → all registered model names
- get_model_entry("my_model") → ModelEntry dataclass
- load_model_class("my_model") → imported class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from easi.utils.import_utils import import_class as _import_class
from easi.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    """Registry entry for a custom model server."""

    name: str
    display_name: str
    description: str
    model_class: str  # fully qualified class name
    default_kwargs: dict = field(default_factory=dict)


# Module-level registry populated on first access
_registry: dict[str, ModelEntry] | None = None


def _get_models_dir() -> Path:
    """Return the directory containing model subdirectories."""
    return Path(__file__).parent


def _discover_models() -> dict[str, ModelEntry]:
    """Scan model directories for manifest.yaml files."""
    models_dir = _get_models_dir()
    entries: dict[str, ModelEntry] = {}

    for manifest_path in sorted(models_dir.glob("*/manifest.yaml")):
        try:
            manifest = yaml.safe_load(manifest_path.read_text())
        except Exception as e:
            logger.warning("Failed to load %s: %s", manifest_path, e)
            continue

        try:
            entry = ModelEntry(
                name=manifest["name"],
                display_name=manifest.get("display_name", manifest["name"]),
                description=manifest.get("description", ""),
                model_class=manifest["model_class"],
                default_kwargs=manifest.get("default_kwargs", {}),
            )
            entries[entry.name] = entry
            logger.trace("Discovered model: %s (%s)", entry.name, entry.display_name)
        except KeyError as e:
            logger.warning(
                "Invalid manifest %s: missing required field %s", manifest_path, e
            )
            continue

    return entries


def _get_registry() -> dict[str, ModelEntry]:
    """Get the model registry, discovering on first access."""
    global _registry
    if _registry is None:
        _registry = _discover_models()
    return _registry


def list_models() -> list[str]:
    """List all registered model names."""
    return sorted(_get_registry().keys())


def get_model_entry(name: str) -> ModelEntry:
    """Look up a model entry by name.

    Args:
        name: The model name as defined in its manifest.yaml.

    Raises:
        KeyError: If the model is not found.
    """
    registry = _get_registry()
    if name not in registry:
        available = list_models()
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return registry[name]


def load_model_class(name: str):
    """Import and return the model class for the given name."""
    entry = get_model_entry(name)
    return _import_class(entry.model_class)


def refresh() -> None:
    """Force re-discovery of models (useful after adding new ones at runtime)."""
    global _registry
    _registry = None
