"""Workspace and path management utilities.

IMPORTANT: The default datasets directory is ``./datasets`` (relative to cwd).
All code that needs a datasets directory MUST use ``get_datasets_dir()`` or
``DEFAULT_DATASETS_DIR`` from this module — never hardcode a path.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Default directories — single source of truth
# ---------------------------------------------------------------------------

#: Default directory for downloaded datasets, relative to the working directory.
#: Used by ``easi start``, ``easi task download``, runner, and base_task.
DEFAULT_DATASETS_DIR = Path("./datasets")

#: Default cache directory for locks and other transient state.
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
    """Return the default directory for datasets.

    Returns ``./datasets`` relative to the current working directory.
    This is the single source of truth — used by the runner, base_task,
    and ``easi task download``.
    """
    DEFAULT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATASETS_DIR


def create_temp_workspace(prefix: str = "easi_") -> Path:
    """Create a unique temporary directory for an IPC workspace."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_dir(path: Path) -> None:
    """Remove a directory tree, ignoring errors."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
