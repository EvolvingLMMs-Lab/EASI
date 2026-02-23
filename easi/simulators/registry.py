"""Simulator registry with manifest-based auto-discovery.

Scans easi/simulators/*/manifest.yaml at import time to discover available
simulators and their versions.

Lookup semantics:
- get_simulator("ai2thor") → resolves to default version
- get_simulator("ai2thor:v2_1_0") → resolves to explicit version
- list_simulators() → all registered keys
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from easi.utils.import_utils import import_class as _import_class
from easi.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatorEntry:
    """Registry entry for a simulator version."""

    name: str
    version: str
    description: str
    simulator_class: str  # fully qualified class name
    env_manager_class: str  # fully qualified class name
    python_version: str
    installation_kwargs: dict = field(default_factory=dict)
    render_platforms: dict[str, str] = field(default_factory=dict)  # platform_name -> FQN class


# Module-level registry populated on first access
_registry: dict[str, SimulatorEntry] | None = None


def _discover_simulators() -> dict[str, SimulatorEntry]:
    """Scan simulator directories for manifest.yaml files."""
    simulators_dir = Path(__file__).parent
    entries: dict[str, SimulatorEntry] = {}

    for manifest_path in sorted(simulators_dir.glob("*/manifest.yaml")):
        try:
            manifest = yaml.safe_load(manifest_path.read_text())
        except Exception as e:
            logger.warning("Failed to load %s: %s", manifest_path, e)
            continue

        sim_name = manifest["name"]
        default_ver = manifest.get("default_version")

        for ver_key, ver_info in manifest.get("versions", {}).items():
            entry = SimulatorEntry(
                name=sim_name,
                version=ver_key,
                description=ver_info.get("description", ""),
                simulator_class=ver_info["simulator_class"],
                env_manager_class=ver_info["env_manager_class"],
                python_version=ver_info.get("python_version", "3.10"),
                installation_kwargs=ver_info.get("installation_kwargs", {}),
                render_platforms=ver_info.get("render_platforms", {}),
            )

            # Register with explicit key: "ai2thor:v2_1_0"
            full_key = f"{sim_name}:{ver_key}"
            entries[full_key] = entry

            # Register default alias: "ai2thor"
            if ver_key == default_ver:
                entries[sim_name] = entry

        logger.trace(
            "Discovered simulator %s with versions: %s (default: %s)",
            sim_name,
            list(manifest.get("versions", {}).keys()),
            default_ver,
        )

    return entries


def _get_registry() -> dict[str, SimulatorEntry]:
    """Get the simulator registry, discovering on first access."""
    global _registry
    if _registry is None:
        _registry = _discover_simulators()
    return _registry


def get_simulator_entry(key: str) -> SimulatorEntry:
    """Look up a simulator entry by key.

    Args:
        key: Either "name" (uses default version) or "name:version" (explicit).

    Raises:
        KeyError: If the simulator is not found.
    """
    registry = _get_registry()
    if key not in registry:
        available = list_simulators()
        raise KeyError(
            f"Simulator '{key}' not found. Available: {available}"
        )
    return registry[key]


def list_simulators() -> list[str]:
    """List all registered simulator keys."""
    return sorted(_get_registry().keys())


def load_simulator_class(key: str):
    """Import and return the simulator class for the given key."""
    entry = get_simulator_entry(key)
    return _import_class(entry.simulator_class)


def load_env_manager_class(key: str):
    """Import and return the env manager class for the given key."""
    entry = get_simulator_entry(key)
    return _import_class(entry.env_manager_class)


def create_env_manager(key: str):
    """Create an env manager instance with installation_kwargs from the manifest."""
    entry = get_simulator_entry(key)
    cls = _import_class(entry.env_manager_class)
    return cls(installation_kwargs=entry.installation_kwargs)


def resolve_render_platform(key: str, platform_name: str, env_manager=None):
    """Resolve a render platform, checking simulator manifest first, then built-ins.

    Args:
        key: Simulator registry key (e.g. "coppeliasim:v4_1_0" or "coppeliasim").
        platform_name: Platform name (e.g. "xvfb", "isaac_headless").
        env_manager: Optional env manager instance, passed to custom platform constructors.

    Returns:
        Instantiated RenderPlatform.
    """
    from easi.core.render_platform import get_render_platform

    entry = get_simulator_entry(key)
    custom_class_path = entry.render_platforms.get(platform_name)

    if custom_class_path:
        cls = _import_class(custom_class_path)
        instance = cls(env_manager=env_manager)
        if instance.name != platform_name:
            raise ValueError(
                f"Custom render platform class '{custom_class_path}' has "
                f"name '{instance.name}', expected '{platform_name}'"
            )
        logger.trace("Using custom render platform '%s' from %s", platform_name, key)
        return instance

    return get_render_platform(platform_name)


def refresh() -> None:
    """Force re-discovery of simulators (useful after adding new ones at runtime)."""
    global _registry
    _registry = None
