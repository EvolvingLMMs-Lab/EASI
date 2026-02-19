"""Manages lifecycle of local LLM inference servers (vLLM, etc.).

Starts the server as a subprocess, waits for health check, and stops on exit.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from easi.utils.logging import get_logger

logger = get_logger(__name__)

_HEALTH_POLL_INTERVAL = 5.0
_DEFAULT_STARTUP_TIMEOUT = 300.0


class ServerManager:
    """Manages a local inference server subprocess."""

    def __init__(
        self,
        backend: str,
        model: str,
        port: int = 8080,
        server_kwargs: dict | None = None,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
        log_dir: Path | None = None,  # Deprecated: server output now goes to logger
    ):
        self.backend = backend
        self.model = model
        self.port = port
        self.server_kwargs = server_kwargs or {}
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    def start(self) -> str:
        """Start the server, wait for health, return base_url."""
        self._check_port()

        cmd = self._build_command()
        logger.info("Starting %s server: %s", self.backend, " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._log_thread = threading.Thread(
            target=self._stream_output,
            args=(self._process,),
            daemon=True,
        )
        self._log_thread.start()

        base_url = f"http://localhost:{self.port}/v1"
        self._wait_for_health(base_url)
        logger.info("Server ready at %s", base_url)
        return base_url

    def stop(self) -> None:
        """Terminate the server process."""
        if self._process is not None:
            logger.info("Stopping %s server (pid=%d)", self.backend, self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate, killing...")
                self._process.kill()
                self._process.wait(timeout=10)
            self._process = None
        if self._log_thread is not None:
            self._log_thread.join(timeout=5)
            self._log_thread = None

    def is_running(self) -> bool:
        """Check if server process is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def _check_port(self) -> None:
        """Raise if port is already in use."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", self.port))
        except OSError:
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                f"Use --port <N> to specify a different port, "
                f"or --llm-url to connect to an existing server."
            )
        finally:
            sock.close()

    def _build_command(self) -> list[str]:
        """Build the server launch command."""
        if self.backend == "vllm":
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model,
                "--port", str(self.port),
            ]
            for key, value in self.server_kwargs.items():
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                    # Skip False booleans (don't add the flag)
                else:
                    cmd.extend([flag, str(value)])
            return cmd
        else:
            raise ValueError(f"Unsupported server backend: {self.backend}")

    @staticmethod
    def _stream_output(proc: subprocess.Popen) -> None:
        """Read server stdout/stderr line by line and log at TRACE level."""
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.trace("[server] %s", line)
        proc.stdout.close()

    def _wait_for_health(self, base_url: str) -> None:
        """Poll /health until the server responds or timeout."""
        health_url = base_url.replace("/v1", "") + "/health"
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.backend} server exited with code {self._process.returncode}. "
                    f"Run with --verbosity TRACE to see server output."
                )
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass

            time.sleep(_HEALTH_POLL_INTERVAL)

        self.stop()
        raise RuntimeError(
            f"{self.backend} server failed to start within "
            f"{self.startup_timeout}s. Run with --verbosity TRACE to see server output."
        )

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
