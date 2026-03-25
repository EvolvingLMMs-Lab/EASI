"""Sticky terminal progress bar that coexists with logging output.

Usage::

    from easi.utils.progress import ProgressBar

    bar = ProgressBar(total=50, num_workers=4)
    bar.start()
    # ... in worker threads:
    bar.update(completed=10, failed=1)
    # ... when done:
    bar.stop()

The progress bar renders on stderr, pinned to the bottom of the terminal.
When a log record is emitted, the bar is cleared first so log output is
never corrupted — identical pattern to spinner.py.

In non-TTY mode (piped output), periodic log-line updates are emitted instead.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from easi.utils.logging import get_logger

_logger = get_logger(__name__)


_CLEAR = "\r\033[K"  # carriage return + ANSI erase to end of line
_REFRESH_INTERVAL = 0.5  # seconds between redraws
_NON_TTY_LOG_INTERVAL = 30.0  # seconds between log-line updates in non-TTY mode

# Bar characters
_FILL = "█"
_EMPTY = "░"

# Colors (match logging.py conventions)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_GRAY = "\033[90m"
_RED = "\033[31m"


class ProgressBar:
    """Thread-safe sticky progress bar for evaluation runs.

    Wraps log handler emit() methods (same pattern as spinner.py) to keep
    the bar at the bottom while logs scroll above.

    Parameters
    ----------
    total:
        Total number of episodes.
    num_workers:
        Number of parallel workers (1 for sequential).
    start_index:
        Number of already-completed episodes (for resume).
    """

    def __init__(
        self,
        total: int,
        num_workers: int = 1,
        start_index: int = 0,
    ):
        self.total = total
        self.num_workers = num_workers
        self.start_index = start_index

        self._completed = start_index
        self._failed = 0
        self._active_workers = 0
        self._elapsed_start: float = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._wrapped: list[tuple[logging.Handler, object]] = []
        self._last_non_tty_log: float = 0.0

    def start(self) -> None:
        """Start the progress bar background thread."""
        self._elapsed_start = time.monotonic()

        if self._is_tty:
            # Wrap existing handlers so they clear the bar before emitting
            for handler in logging.getLogger("easi").handlers:
                original_emit = handler.emit

                def wrapped_emit(record, _orig=original_emit):
                    with self._lock:
                        sys.stderr.write(_CLEAR)
                        sys.stderr.flush()
                        _orig(record)
                        # Redraw bar after log line
                        self._render()

                self._wrapped.append((handler, original_emit))
                handler.emit = wrapped_emit  # type: ignore[method-assign]

            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the progress bar and restore handlers."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        if self._is_tty:
            # Final clear
            with self._lock:
                sys.stderr.write(_CLEAR)
                sys.stderr.flush()

            # Restore original emit methods
            for handler, original_emit in self._wrapped:
                handler.emit = original_emit  # type: ignore[method-assign]
            self._wrapped.clear()

    def update(
        self,
        completed: int | None = None,
        failed: int | None = None,
        active_workers: int | None = None,
    ) -> None:
        """Thread-safe progress update.

        Parameters
        ----------
        completed:
            Total completed episodes (including resumed).
        failed:
            Total failed episodes.
        active_workers:
            Number of currently active worker threads.
        """
        do_log = False
        with self._lock:
            if completed is not None:
                self._completed = completed
            if failed is not None:
                self._failed = failed
            if active_workers is not None:
                self._active_workers = active_workers

            # Non-TTY: emit periodic log lines
            if not self._is_tty:
                now = time.monotonic()
                if now - self._last_non_tty_log >= _NON_TTY_LOG_INTERVAL:
                    self._last_non_tty_log = now
                    do_log = True

        if do_log:
            self._log_progress()

    def _refresh_loop(self) -> None:
        """Background thread: periodically redraw the bar."""
        while not self._stop_event.is_set():
            with self._lock:
                self._render()
            self._stop_event.wait(_REFRESH_INTERVAL)

    def _render(self) -> None:
        """Render the progress bar to stderr. Must be called with _lock held."""
        line = self._format_bar()
        sys.stderr.write(f"{_CLEAR}{line}")
        sys.stderr.flush()

    def _format_bar(self) -> str:
        """Build the progress bar string."""
        completed = self._completed
        total = self.total
        failed = self._failed
        active = self._active_workers
        elapsed = time.monotonic() - self._elapsed_start

        # Percentage and bar
        pct = completed / total if total > 0 else 0
        bar_width = min(self._get_terminal_width() - 60, 30)
        bar_width = max(bar_width, 10)
        filled = int(bar_width * pct)
        bar = _FILL * filled + _EMPTY * (bar_width - filled)

        # Color the bar
        if failed > 0:
            bar_color = _YELLOW
        elif completed >= total:
            bar_color = _GREEN
        else:
            bar_color = _CYAN

        # Elapsed time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            time_str = f"{hours}h{mins:02d}m"
        else:
            time_str = f"{mins}m{secs:02d}s"

        # ETA
        if completed > self.start_index and pct < 1.0:
            new_completed = completed - self.start_index
            elapsed_per_ep = elapsed / new_completed
            remaining = (total - completed) * elapsed_per_ep
            r_mins, r_secs = divmod(int(remaining), 60)
            r_hours, r_mins = divmod(r_mins, 60)
            if r_hours > 0:
                eta_str = f"{r_hours}h{r_mins:02d}m"
            else:
                eta_str = f"{r_mins}m{r_secs:02d}s"
        else:
            eta_str = "--"

        # Build line
        parts = [
            f"{bar_color}{bar}{_RESET}",
            f" {_BOLD}{completed}/{total}{_RESET}",
        ]
        if failed > 0:
            parts.append(f" {_RED}({failed} failed){_RESET}")
        if self.num_workers > 1 and active > 0:
            parts.append(f" {_GRAY}[{active}w]{_RESET}")
        parts.append(f" {_GRAY}{time_str} elapsed, ETA {eta_str}{_RESET}")

        return "".join(parts)

    def _log_progress(self) -> None:
        """Emit a plain log line (for non-TTY mode)."""
        elapsed = time.monotonic() - self._elapsed_start
        mins, secs = divmod(int(elapsed), 60)
        _logger.info(
            "[Progress] %d/%d episodes (%d failed) — %dm%02ds elapsed",
            self._completed, self.total, self._failed, mins, secs,
        )

    @staticmethod
    def _get_terminal_width() -> int:
        """Get terminal width, with fallback."""
        try:
            return os.get_terminal_size(sys.stderr.fileno()).columns
        except (ValueError, OSError):
            return 80

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
