"""Manages lifecycle of local LLM inference servers (vLLM, etc.).

Starts the server as a subprocess, waits for health check, and stops on exit.
"""
from __future__ import annotations

import os
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
_DEFAULT_STARTUP_TIMEOUT = 600.0
_DEFAULT_VLLM_FLAGS = {
    "enable_prefix_caching": True,
    "disable_log_requests": True,
}


def _port_is_available(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


class ServerManager:
    """Manages a local inference server subprocess."""

    def __init__(
        self,
        backend: str,
        model: str,
        port: int = 8080,
        server_kwargs: dict | None = None,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
        cuda_visible_devices: str | None = None,
        label: str = "server",
    ):
        self.backend = backend
        self.model = model
        self.port = port
        self.server_kwargs = server_kwargs or {}
        self.startup_timeout = startup_timeout
        self.cuda_visible_devices = cuda_visible_devices
        self.label = label
        self._process: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None
        logger.trace(
            "[%s] ServerManager init: backend=%s, model=%s, port=%d, "
            "server_kwargs=%s, cuda_visible_devices=%s",
            label, backend, model, port, self.server_kwargs, cuda_visible_devices,
        )

    def start(self) -> str:
        """Start the server, wait for health, return base_url."""
        self.launch()
        return self.wait_until_ready()

    def launch(self) -> None:
        """Spawn the server process without waiting for health."""
        self._check_port()

        cmd, extra_env = self._build_command()
        logger.info("[%s] Starting %s server: %s", self.label, self.backend, " ".join(cmd))

        spawn_env = os.environ.copy()
        spawn_env.update(extra_env)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=spawn_env,
        )
        self._log_thread = threading.Thread(
            target=self._stream_output,
            args=(self._process, self.label),
            daemon=True,
        )
        self._log_thread.start()

    def wait_until_ready(self) -> str:
        """Poll health endpoint until the server is ready. Returns base_url."""
        base_url = f"http://localhost:{self.port}/v1"
        self._wait_for_health(base_url)
        logger.info("[%s] Server ready at %s", self.label, base_url)
        return base_url

    def stop(self) -> None:
        """Terminate the server process."""
        if self._process is not None:
            logger.info("[%s] Stopping %s server (pid=%d)", self.label, self.backend, self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] Server did not terminate, killing...", self.label)
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
        logger.trace("[%s] Checking if port %d is available...", self.label, self.port)
        if not _port_is_available(self.port):
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                f"Use --port <N> to specify a different port, "
                f"or --llm-url to connect to an existing server."
            )

    def _build_command(self) -> tuple[list[str], dict]:
        """Build the server launch command and environment overrides.

        Returns:
            Tuple of (command list, env dict). The env dict contains
            ``CUDA_VISIBLE_DEVICES`` when *cuda_visible_devices* is set.
        """
        if self.backend == "vllm":
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model,
                "--port", str(self.port),
            ]
            # Merge defaults with user overrides (user wins)
            merged_kwargs = {**_DEFAULT_VLLM_FLAGS, **self.server_kwargs}
            overridden = {
                k: v for k, v in self.server_kwargs.items()
                if k in _DEFAULT_VLLM_FLAGS and v != _DEFAULT_VLLM_FLAGS[k]
            }
            if overridden:
                logger.trace("[%s] User overrides for default vLLM flags: %s", self.label, overridden)
            logger.trace("[%s] Merged vLLM kwargs: %s", self.label, merged_kwargs)
            for key, value in merged_kwargs.items():
                flag = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                    else:
                        logger.trace("[%s] Skipping disabled bool flag: %s", self.label, flag)
                else:
                    cmd.extend([flag, str(value)])
        else:
            raise ValueError(f"Unsupported server backend: {self.backend}")

        env: dict[str, str] = {}
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        logger.trace("[%s] Built command: %s", self.label, cmd)
        logger.trace("[%s] Extra env: %s", self.label, env)
        return cmd, env

    @staticmethod
    def _stream_output(proc: subprocess.Popen, label: str = "server") -> None:
        """Read server stdout/stderr line by line and log at TRACE level."""
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.trace("[%s] %s", label, line)
        proc.stdout.close()

    def _wait_for_health(self, base_url: str) -> None:
        """Poll /health until the server responds or timeout."""
        health_url = base_url.replace("/v1", "") + "/health"
        deadline = time.monotonic() + self.startup_timeout
        logger.trace(
            "[%s] Waiting for health at %s (timeout=%.0fs)",
            self.label, health_url, self.startup_timeout,
        )

        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"[{self.label}] {self.backend} server exited with code "
                    f"{self._process.returncode}. "
                    f"Run with --verbosity TRACE to see server output."
                )
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.trace("[%s] Health check passed (status=%d)", self.label, resp.status_code)
                    return
                logger.trace("[%s] Health check returned status %d, retrying...", self.label, resp.status_code)
            except requests.ConnectionError:
                logger.trace("[%s] Health check connection refused, retrying...", self.label)

            time.sleep(_HEALTH_POLL_INTERVAL)

        self.stop()
        raise RuntimeError(
            f"[{self.label}] {self.backend} server failed to start within "
            f"{self.startup_timeout}s. Run with --verbosity TRACE to see server output."
        )

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class MultiServerManager:
    """Manages multiple vLLM server instances across GPUs."""

    def __init__(
        self,
        model: str,
        num_instances: int,
        gpu_ids: list[int] | None = None,
        base_port: int = 8000,
        server_kwargs: dict | None = None,
        startup_timeout: float = 300.0,
    ):
        if gpu_ids is not None and len(gpu_ids) % num_instances != 0:
            raise ValueError(
                f"Cannot divide {len(gpu_ids)} GPUs evenly across "
                f"{num_instances} instances"
            )
        self.model = model
        self.num_instances = num_instances
        self.gpu_ids = gpu_ids
        self.base_port = base_port
        self.server_kwargs = server_kwargs or {}
        self.startup_timeout = startup_timeout
        self._managers: list[ServerManager] = []

    def start(self) -> list[str]:
        """Start all instances in parallel, return list of base_urls.

        All server processes are spawned first, then health checks run
        concurrently via threads.  Ports are assigned by probing from
        *base_port* upward, skipping any that are already in use.  If
        any instance fails, all are stopped before re-raising.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gpus_per = len(self.gpu_ids) // self.num_instances if self.gpu_ids else None
        next_port = self.base_port

        try:
            # Phase 1: Spawn all processes (fast, no blocking)
            for i in range(self.num_instances):
                if gpus_per is not None:
                    instance_gpus = self.gpu_ids[i * gpus_per : (i + 1) * gpus_per]
                    cuda_devices = ",".join(str(g) for g in instance_gpus)
                else:
                    cuda_devices = None
                port = self._find_available_port(next_port)
                next_port = port + 1
                mgr = ServerManager(
                    backend="vllm",
                    model=self.model,
                    port=port,
                    server_kwargs=self.server_kwargs,
                    startup_timeout=self.startup_timeout,
                    cuda_visible_devices=cuda_devices,
                    label=f"vllm-{i}",
                )
                mgr.launch()
                self._managers.append(mgr)

            # Phase 2: Wait for all health checks concurrently
            logger.info(
                "All %d vLLM processes spawned, waiting for health checks...",
                self.num_instances,
            )
            urls = [None] * len(self._managers)
            with ThreadPoolExecutor(max_workers=len(self._managers)) as pool:
                future_to_idx = {
                    pool.submit(mgr.wait_until_ready): idx
                    for idx, mgr in enumerate(self._managers)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    urls[idx] = future.result()

        except Exception:
            logger.warning(
                "vLLM startup failed, stopping %d spawned instances",
                len(self._managers),
            )
            self.stop()
            raise

        return urls

    @staticmethod
    def _find_available_port(start: int, max_probe: int = 100) -> int:
        """Find the first available port starting from *start*."""
        for port in range(start, start + max_probe):
            if _port_is_available(port):
                return port
        raise RuntimeError(
            f"No available port found in range {start}-{start + max_probe - 1}"
        )

    def stop(self):
        """Stop all managed instances."""
        for mgr in self._managers:
            mgr.stop()
        self._managers.clear()
