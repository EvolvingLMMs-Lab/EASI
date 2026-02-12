"""Custom easi logger using stdlib logging.

Library modules should only use get_logger(). The CLI entry point calls
setup_logging() once to configure handlers and level.
"""

from __future__ import annotations

import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a namespaced easi logger.

    Accepts ``__name__`` directly — the ``easi.`` prefix is handled
    automatically so callers never double-prefix.

    Usage::

        from easi.utils.logging import get_logger
        logger = get_logger(__name__)  # recommended — auto-discovers module path
        logger.info("Launching bridge: %s", bridge_path)
    """
    if name.startswith("easi."):
        return logging.getLogger(name)
    return logging.getLogger(f"easi.{name}")


def setup_logging(level: str = "WARNING") -> None:
    """Configure the easi logger hierarchy.

    Called once by the CLI entry point. Library code should never call this.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR".
    """
    logger = logging.getLogger("easi")
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
