"""Manages lifecycle of local LLM inference servers (vLLM, etc.).

Starts the server as a subprocess, waits for health check, and stops on exit.
"""
from __future__ import annotations

import socket
import subprocess
import sys
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
        log_dir: Path | None = None,
    ):
        self.backend = backend
        self.model = model
        self.port = port
        self.server_kwargs = server_kwargs or {}
        self.startup_timeout = startup_timeout
        self.log_dir = log_dir
        self._process: subprocess.Popen | None = None

    def start(self) -> str:
        """Start the server, wait for health, return base_url."""
        self._check_port()

        cmd = self._build_command()
        logger.info("Starting %s server: %s", self.backend, " ".join(cmd))

        log_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{self.backend}_server.log"
            log_file = open(log_path, "w")
            logger.info("Server logs: %s", log_path)

        self._process = subprocess.Popen(
            cmd,
            stdout=log_file or subprocess.DEVNULL,
            stderr=log_file or subprocess.DEVNULL,
        )

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
                cmd.extend([flag, str(value)])
            return cmd
        else:
            raise ValueError(f"Unsupported server backend: {self.backend}")

    def _wait_for_health(self, base_url: str) -> None:
        """Poll /health until the server responds or timeout."""
        health_url = base_url.replace("/v1", "") + "/health"
        deadline = time.monotonic() + self.startup_timeout

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.backend} server exited with code {self._process.returncode}. "
                    f"Check server logs for details."
                )
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass

            time.sleep(_HEALTH_POLL_INTERVAL)

        self.stop()
        log_hint = ""
        if self.log_dir:
            log_hint = f" Check logs at {self.log_dir / f'{self.backend}_server.log'}"
        raise RuntimeError(
            f"{self.backend} server failed to start within "
            f"{self.startup_timeout}s.{log_hint}"
        )

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
