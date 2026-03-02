"""Subprocess lifecycle manager for bridge processes.

Manages the full lifecycle of a simulator bridge subprocess:
- Launch: Spawns the bridge in a new process group (for clean shutdown)
- Health check: Polls status.json with configurable timeout
- Send command: Writes command.json, polls response.json
- Crash recovery: Detects subprocess exit during polling
- Cleanup: SIGTERM -> wait -> SIGKILL the entire process group; removes temp workspace

Supports pluggable render platforms (xvfb, egl, native, headless, auto).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

from easi.communication.filesystem import (
    cleanup_workspace,
    create_workspace,
    poll_for_response,
    poll_for_status,
    write_command,
)
from easi.core.exceptions import SimulatorError, SimulatorTimeoutError
from easi.core.render_platform import EnvVars, RenderPlatform
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class SubprocessRunner:
    """Manages a bridge subprocess for a single simulator instance."""

    def __init__(
        self,
        python_executable: str,
        bridge_script_path: Path,
        render_platform: RenderPlatform,
        screen_config: str = "1024x768x24",
        startup_timeout: float = 30.0,
        command_timeout: float = 300.0,
        poll_interval: float = 0.1,
        extra_args: list[str] | None = None,
        extra_env: EnvVars | None = None,
    ):
        self.python_executable = python_executable
        self.bridge_script_path = bridge_script_path
        self.render_platform = render_platform
        self.screen_config = screen_config
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self.poll_interval = poll_interval
        self.extra_args = extra_args or []
        self.extra_env = extra_env

        self._process: subprocess.Popen | None = None
        self._workspace: Path | None = None
        self._output_lines: deque[str] = deque(maxlen=200)
        self._reader_thread: threading.Thread | None = None

    @property
    def workspace(self) -> Path | None:
        """The IPC workspace directory for this runner."""
        return self._workspace

    def launch(self) -> None:
        """Launch the bridge subprocess and wait for it to report ready."""
        if self._process is not None:
            raise RuntimeError("Subprocess already running")

        self._workspace = create_workspace()
        self._output_lines.clear()
        cmd = self._build_launch_command()

        logger.info(
            "Launching bridge: %s (workspace: %s)",
            self.bridge_script_path.name,
            self._workspace,
        )
        logger.trace("Full command: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,  # new session = new process group
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            text=True,
            bufsize=1,
            env=self._build_subprocess_env(),
        )

        # Stream bridge output through logger in a background thread
        self._reader_thread = threading.Thread(
            target=self._stream_output,
            daemon=True,
        )
        self._reader_thread.start()

        # Wait for the bridge to signal readiness
        try:
            status = poll_for_status(
                self._workspace,
                poll_interval=self.poll_interval,
                timeout=self.startup_timeout,
                process=self._process,
            )
            if not status.get("ready", False):
                output = self._get_recent_output()
                raise SimulatorError(
                    f"Bridge reported not ready. Output:\n{output}"
                )
            logger.info("Bridge subprocess ready (PID: %d)", self._process.pid)
        except (SimulatorError, SimulatorTimeoutError) as exc:
            # Collect output before shutdown destroys the process
            output = self._get_recent_output()
            self.shutdown()
            if output:
                raise SimulatorError(f"{exc}\n\nBridge output:\n{output}") from exc
            raise

    def launch_docker(
        self,
        docker_env_manager,
        episode_output_dir: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        """Launch bridge subprocess inside a Docker container."""
        if self._process is not None:
            raise RuntimeError("Subprocess already running")

        self._workspace = create_workspace()
        self._output_lines.clear()
        cmd = self._build_docker_launch_command(
            docker_env_manager=docker_env_manager,
            workspace_dir=str(self._workspace),
            episode_output_dir=episode_output_dir,
            data_dir=data_dir,
        )

        logger.info("Launching Docker bridge: %s", " ".join(cmd[:6]) + " ...")
        logger.trace("Full Docker command: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        # Start output reader thread
        self._reader_thread = threading.Thread(
            target=self._stream_output, daemon=True
        )
        self._reader_thread.start()

        # Wait for bridge to signal ready
        try:
            status = poll_for_status(
                self._workspace,
                poll_interval=self.poll_interval,
                timeout=self.startup_timeout,
                process=self._process,
            )
            if not status.get("ready", False):
                output = self._get_recent_output()
                raise SimulatorError(
                    f"Docker bridge reported not ready. Output:\n{output}"
                )
            logger.info("Docker bridge subprocess ready (PID: %d)", self._process.pid)
        except (SimulatorError, SimulatorTimeoutError) as exc:
            output = self._get_recent_output()
            self.shutdown()
            if output:
                raise SimulatorError(f"{exc}\n\nDocker bridge output:\n{output}") from exc
            raise

    def send_command(self, command: dict, timeout: float | None = None) -> dict:
        """Send a command to the bridge and wait for the response.

        Args:
            command: The command dict to send.
            timeout: Override the default command timeout.

        Returns:
            The response dict from the bridge.
        """
        if self._process is None or self._workspace is None:
            raise RuntimeError("Subprocess not running")

        write_command(self._workspace, command)

        return poll_for_response(
            self._workspace,
            poll_interval=self.poll_interval,
            timeout=timeout or self.command_timeout,
            process=self._process,
        )

    def is_alive(self) -> bool:
        """Check if the bridge subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def shutdown(self) -> None:
        """Shut down the bridge subprocess and clean up.

        Kills the entire process group (bridge + any child processes like
        Unity binaries) to prevent zombie processes.
        """
        self._terminate_process_tree()
        # Wait for the reader thread to drain any final output from the bridge
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        self._process = None
        self._reader_thread = None

        if self._workspace is not None:
            cleanup_workspace(self._workspace)
            self._workspace = None

    def _build_subprocess_env(self) -> dict[str, str] | None:
        """Build env dict for subprocess, merging platform + extra_env.

        Returns None if no env vars to set (subprocess inherits parent env).
        """
        platform_env = self.render_platform.get_env_vars()
        combined = EnvVars.merge(platform_env, self.extra_env) if self.extra_env else platform_env

        if not combined:
            return None

        return combined.apply_to_env(os.environ.copy())

    def _build_launch_command(self) -> list[str]:
        """Build the command to launch the bridge subprocess."""
        cmd = [
            self.python_executable,
            str(self.bridge_script_path),
            "--workspace",
            str(self._workspace),
        ]
        cmd.extend(self.extra_args)

        return self.render_platform.wrap_command(cmd, self.screen_config)

    def _build_docker_launch_command(
        self,
        docker_env_manager,
        workspace_dir: str,
        episode_output_dir: str | None = None,
        data_dir: str | None = None,
    ) -> list[str]:
        """Build launch command using docker run via DockerEnvironmentManager."""
        bridge_command = [
            docker_env_manager.container_python_path,
            str(Path(docker_env_manager.easi_mount) / self.bridge_script_path.relative_to(
                Path(__file__).resolve().parents[1]
            )),
            "--workspace",
            workspace_dir,
        ]
        bridge_command.extend(self.extra_args)

        return docker_env_manager.build_docker_run_command(
            bridge_command=bridge_command,
            workspace_dir=workspace_dir,
            episode_output_dir=episode_output_dir,
            data_dir=data_dir,
        )

    # Pattern to extract log level from bridge output (e.g. "[WARNING]" or "[ERROR]")
    _BRIDGE_LEVEL_RE = re.compile(r"\[(\w+)\]")

    # Map bridge level names to logging levels
    _BRIDGE_LEVEL_MAP = {
        "TRACE": 5,  # easi TRACE level
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def _stream_output(self) -> None:
        """Read bridge stdout line-by-line and re-log at the matching level.

        Parses the log level from each bridge line (e.g. "[WARNING]") and
        re-emits at that level so the parent process applies correct coloring.
        Lines without a recognized level default to TRACE.

        Runs in a daemon thread for the lifetime of the subprocess.
        """
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip()
                self._output_lines.append(line)
                level = self._parse_bridge_level(line)
                logger.log(level, "[bridge] %s", line)
        except (ValueError, OSError):
            pass  # pipe closed

    def _parse_bridge_level(self, line: str) -> int:
        """Extract log level from a bridge output line."""
        match = self._BRIDGE_LEVEL_RE.search(line)
        if match:
            return self._BRIDGE_LEVEL_MAP.get(match.group(1), 5)
        return 5  # default to TRACE

    def _get_recent_output(self) -> str:
        """Return the last N lines of captured bridge output."""
        return "\n".join(self._output_lines)

    def _terminate_process_tree(self) -> None:
        """Kill the bridge and ALL its child processes."""
        if self._process is None:
            return

        pid = self._process.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            logger.info("Bridge process (PID %d) already exited", pid)
            return

        try:
            # SIGTERM the entire group first (graceful)
            logger.info("Sending SIGTERM to bridge process group (PID %d)", pid)
            os.killpg(pgid, signal.SIGTERM)
            self._process.wait(timeout=10)
            logger.info("Bridge process (PID %d) exited after SIGTERM", pid)
        except subprocess.TimeoutExpired:
            # SIGKILL the entire group (force)
            logger.warning("Bridge did not exit after SIGTERM, sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
                self._process.wait(timeout=5)
                logger.info("Bridge process (PID %d) killed", pid)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        except ProcessLookupError:
            logger.info("Bridge process (PID %d) already exited", pid)
