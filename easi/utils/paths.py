"""Workspace and path management utilities."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

# Default cache directory for datasets, locks, etc.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "easi"


def get_cache_dir() -> Path:
    """Return the EASI cache directory, creating it if needed."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_locks_dir() -> Path:
    """Return the directory for file-based locks."""
    locks_dir = get_cache_dir() / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir


def get_datasets_dir() -> Path:
    """Return the directory for cached datasets."""
    datasets_dir = get_cache_dir() / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def create_temp_workspace(prefix: str = "easi_") -> Path:
    """Create a unique temporary directory for an IPC workspace."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_dir(path: Path) -> None:
    """Remove a directory tree, ignoring errors."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
