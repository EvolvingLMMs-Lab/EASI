"""Custom easi logger using stdlib logging.

Library modules should only use get_logger(). The CLI entry point calls
setup_logging() once to configure handlers and level.

Custom levels:
    TRACE (5) — internal library diagnostics, below DEBUG.
    Use logger.trace() for library internals. Reserve logger.debug() for
    user-facing debug output.
"""

from __future__ import annotations

import logging
import sys

# --- Custom TRACE level (below DEBUG=10) ---
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]


# --- ANSI color codes ---
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GRAY = "\033[90m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_YELLOW = "\033[33m"
_RED = "\033[31m"

# Per-level styling: (level_color, message_color)
# - level_color: applied to the [LEVEL] tag
# - message_color: applied to the message text
_LEVEL_STYLES = {
    TRACE:           (_GRAY,           _GRAY),
    logging.DEBUG:   (_CYAN,           ""),
    logging.INFO:    (_BLUE,           ""),
    logging.WARNING: (_YELLOW,         _YELLOW),
    logging.ERROR:   (_RED + _BOLD,    _RED),
}


class _ColorFormatter(logging.Formatter):
    """Formatter that applies per-level ANSI colors.

    Color scheme (modern minimalistic):
        - Timestamp:    always dim gray — subtle, consistent
        - [LEVEL]:      colored per level — primary visual differentiator
        - Logger name:  always dim gray — metadata, not content
        - Separator:    always dim gray
        - Message:      neutral for TRACE/DEBUG/INFO, colored for WARNING/ERROR
    """

    def __init__(self, use_color: bool = True):
        super().__init__(datefmt="%H:%M:%S")
        self._use_color = use_color

    def format(self, record):
        if not self._use_color:
            return self._format_plain(record)

        level_color, msg_color = _LEVEL_STYLES.get(
            record.levelno, ("", "")
        )

        timestamp = self.formatTime(record, self.datefmt)
        name = record.name
        levelname = record.levelname
        message = record.getMessage()

        # Handle exception info
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = message + "\n" + record.exc_text

        parts = [
            f"{_GRAY}{timestamp}{_RESET}",
            f" {level_color}[{levelname}]{_RESET}",
            f" {_GRAY}{name}{_RESET}",
            f" {_GRAY}-{_RESET}",
            f" {msg_color}{message}{_RESET}" if msg_color else f" {message}",
        ]
        return "".join(parts)

    def _format_plain(self, record):
        """No-color fallback for non-TTY output."""
        timestamp = self.formatTime(record, self.datefmt)
        message = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = message + "\n" + record.exc_text
        return f"{timestamp} [{record.levelname}] {record.name} - {message}"


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
        level: One of "TRACE", "DEBUG", "INFO", "WARNING", "ERROR".
    """
    logger = logging.getLogger("easi")
    resolved = TRACE if level.upper() == "TRACE" else getattr(logging, level.upper())
    logger.setLevel(resolved)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        handler.setFormatter(_ColorFormatter(use_color=use_color))
        logger.addHandler(handler)
