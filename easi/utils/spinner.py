"""Terminal spinner that coexists with logging output.

Usage::

    from easi.utils.spinner import spinner

    with spinner("Installing environment"):
        long_running_operation()

The spinner renders on stderr. When a log record is emitted, the existing
handler's emit() is wrapped to clear the spinner line first — so log output
never gets corrupted and there is no duplication.

In non-TTY mode (piped output), no animation is shown.
"""

from __future__ import annotations

import logging
import sys
import threading
import time


_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_INTERVAL = 0.08  # seconds between frames
_CLEAR = "\r\033[K"  # carriage return + ANSI erase to end of line


class _Spinner:
    """Background-thread spinner with log-safe output.

    Instead of adding a second log handler (which duplicates output), this
    wraps the emit() method of every existing handler on the ``easi`` logger
    so that the spinner line is cleared before each log record and the
    spinner thread is paused during the write.
    """

    def __init__(self, message: str):
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._wrapped: list[tuple[logging.Handler, object]] = []

    def start(self):
        if not self._is_tty:
            return
        # Wrap existing handlers so they clear the spinner before emitting
        for handler in logging.getLogger("easi").handlers:
            original_emit = handler.emit

            def wrapped_emit(record, _orig=original_emit):
                with self._lock:
                    sys.stderr.write(_CLEAR)
                    sys.stderr.flush()
                    _orig(record)

            self._wrapped.append((handler, original_emit))
            handler.emit = wrapped_emit  # type: ignore[method-assign]

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._is_tty or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()

        # Final clear
        with self._lock:
            sys.stderr.write(_CLEAR)
            sys.stderr.flush()

        # Restore original emit methods
        for handler, original_emit in self._wrapped:
            handler.emit = original_emit  # type: ignore[method-assign]
        self._wrapped.clear()

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            frame = _FRAMES[idx % len(_FRAMES)]
            with self._lock:
                sys.stderr.write(f"{_CLEAR}{frame} {self._message}")
                sys.stderr.flush()
            idx += 1
            self._stop_event.wait(_INTERVAL)


class spinner:
    """Context manager for a terminal spinner.

    Example::

        with spinner("Creating conda env"):
            run_slow_command()
    """

    def __init__(self, message: str):
        self._spinner = _Spinner(message)

    def __enter__(self):
        self._spinner.start()
        return self._spinner

    def __exit__(self, *exc):
        self._spinner.stop()
        return False
