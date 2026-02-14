"""File-based locking for serializing concurrent operations.

Used to prevent race conditions when multiple EASI processes try to:
- Install the same conda environment simultaneously
- Download the same dataset simultaneously
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from easi.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def file_lock(lock_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock, blocking until available.

    Args:
        lock_path: Path to the lock file. Parent directories are created automatically.

    Usage::

        with file_lock(Path("~/.cache/easi/locks/env_install.lock")):
            # only one process at a time can execute this block
            install_conda_env()
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    logger.trace("Acquiring lock: %s", lock_path)

    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            logger.trace("Lock acquired: %s", lock_path)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            logger.trace("Lock released: %s", lock_path)
