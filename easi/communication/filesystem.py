"""Filesystem-based IPC for command/response exchange between parent and bridge subprocess.

All writes use the atomic write pattern: write to a .tmp file, then os.rename() to the
final path. This guarantees readers never see partial files (rename is atomic on Linux
when source and dest are on the same filesystem).

The parent deletes response.json before writing a new command.json to avoid stale reads.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from easi.core.exceptions import SimulatorTimeoutError
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Default filenames used in the IPC workspace
COMMAND_FILE = "command.json"
RESPONSE_FILE = "response.json"
STATUS_FILE = "status.json"


def create_workspace(prefix: str = "easi_") -> Path:
    """Create a unique temporary workspace directory for IPC.

    Each EASI process/episode gets its own workspace, so concurrent processes
    never collide on file paths.
    """
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    logger.trace("Created IPC workspace: %s", workspace)
    return workspace


def cleanup_workspace(workspace: Path) -> None:
    """Remove the IPC workspace directory and all contents."""
    import shutil

    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
        logger.trace("Cleaned up IPC workspace: %s", workspace)


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON data atomically using write-to-tmp + rename pattern.

    This ensures readers never see a partially written file.
    """
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.rename(str(tmp_path), str(path))


def read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist or is invalid.

    Uses try/except instead of exists-then-read to avoid TOCTOU races.
    """
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def delete_file(path: Path) -> None:
    """Delete a file if it exists, ignoring FileNotFoundError."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_command(workspace: Path, command: dict) -> None:
    """Write a command for the bridge subprocess to read.

    Deletes any existing response first to prevent stale reads.
    """
    response_path = workspace / RESPONSE_FILE
    command_path = workspace / COMMAND_FILE

    delete_file(response_path)
    atomic_write_json(command_path, command)
    logger.trace("Wrote command: %s", command.get("type", "unknown"))


def poll_for_response(
    workspace: Path,
    poll_interval: float = 0.1,
    timeout: float = 60.0,
    process: object | None = None,
) -> dict:
    """Poll the workspace for a response.json file from the bridge subprocess.

    Args:
        workspace: IPC workspace directory.
        poll_interval: Seconds between poll attempts.
        timeout: Maximum seconds to wait before raising SimulatorTimeoutError.
        process: Optional subprocess.Popen instance. If provided, checks whether
                 the subprocess has exited (crashed) during polling.

    Returns:
        Parsed response dict.

    Raises:
        SimulatorTimeoutError: If timeout is exceeded.
        SimulatorError: If the subprocess has exited unexpectedly.
    """
    from easi.core.exceptions import SimulatorError

    response_path = workspace / RESPONSE_FILE
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Check if subprocess has crashed
        if process is not None and hasattr(process, "poll"):
            if process.poll() is not None:
                raise SimulatorError(
                    f"Bridge subprocess exited with code {process.returncode} "
                    f"while waiting for response"
                )

        data = read_json(response_path)
        if data is not None:
            logger.trace("Received response: status=%s", data.get("status", "unknown"))
            return data

        time.sleep(poll_interval)

    raise SimulatorTimeoutError(
        f"Timed out waiting for response after {timeout}s",
        timeout=timeout,
    )


def poll_for_status(
    workspace: Path,
    poll_interval: float = 0.1,
    timeout: float = 30.0,
    process: object | None = None,
) -> dict:
    """Poll the workspace for a status.json file (bridge startup health check).

    Same semantics as poll_for_response but reads status.json instead.
    """
    from easi.core.exceptions import SimulatorError

    status_path = workspace / STATUS_FILE
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process is not None and hasattr(process, "poll"):
            if process.poll() is not None:
                raise SimulatorError(
                    f"Bridge subprocess exited with code {process.returncode} "
                    f"during startup"
                )

        data = read_json(status_path)
        if data is not None:
            logger.trace("Received status: ready=%s", data.get("ready", False))
            return data

        time.sleep(poll_interval)

    raise SimulatorTimeoutError(
        f"Bridge subprocess did not report ready within {timeout}s",
        timeout=timeout,
    )


def poll_for_command(
    workspace: Path,
    poll_interval: float = 0.1,
    timeout: float = 60.0,
) -> dict:
    """Poll the workspace for a command.json file (bridge-side).

    Used by the bridge subprocess to wait for commands from the parent.

    Returns:
        Parsed command dict.

    Raises:
        SimulatorTimeoutError: If timeout is exceeded.
    """
    command_path = workspace / COMMAND_FILE
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        data = read_json(command_path)
        if data is not None:
            # Delete the command file after reading to signal we've consumed it
            delete_file(command_path)
            logger.trace("Bridge received command: %s", data.get("type", "unknown"))
            return data

        time.sleep(poll_interval)

    raise SimulatorTimeoutError(
        f"No command received within {timeout}s",
        timeout=timeout,
    )


def write_response(workspace: Path, response: dict) -> None:
    """Write a response for the parent process to read (bridge-side)."""
    response_path = workspace / RESPONSE_FILE
    atomic_write_json(response_path, response)
    logger.trace("Bridge wrote response: status=%s", response.get("status", "unknown"))


def write_status(workspace: Path, ready: bool) -> None:
    """Write a status file to signal bridge readiness (bridge-side)."""
    status_path = workspace / STATUS_FILE
    atomic_write_json(status_path, {"ready": ready})
    logger.trace("Bridge wrote status: ready=%s", ready)
