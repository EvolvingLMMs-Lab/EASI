"""Manages lifecycle of Xorg servers for GPU-accelerated rendering.

Starts one Xorg per GPU, waits for health, stops on exit.
Follows the same lifecycle pattern as ``easi.llm.server_manager.ServerManager``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from typing import NamedTuple

from easi.utils.logging import get_logger

logger = get_logger(__name__)

_STARTUP_TIMEOUT = 10.0
_HEALTH_POLL_INTERVAL = 0.5


class XorgInstance(NamedTuple):
    """A running Xorg server bound to a GPU."""

    display: int
    gpu_id: int
    pid: int


def _find_available_display(start: int, max_probe: int = 50) -> int:
    """Find the first available X display number starting from *start*."""
    for num in range(start, start + max_probe):
        lock_file = f"/tmp/.X{num}-lock"
        if not os.path.exists(lock_file):
            return num
    raise RuntimeError(
        f"No available X display in range :{start}-:{start + max_probe - 1}"
    )


def _get_pci_bus_id(gpu_index: int) -> str:
    """Query PCI BusID for a GPU via nvidia-smi, return Xorg format (PCI:B:D:F)."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
            "-i",
            str(gpu_index),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"nvidia-smi failed for GPU {gpu_index}: {result.stderr.strip()}"
        )

    raw = result.stdout.strip()
    # Format: 00000000:3F:00.0 → strip domain, parse hex bus:dev.func
    no_domain = raw.split(":", 1)[1]  # "3F:00.0"
    bus_hex, rest = no_domain.split(":", 1)  # "3F", "00.0"
    dev_hex, func = rest.split(".", 1)  # "00", "0"
    bus_dec = int(bus_hex, 16)
    dev_dec = int(dev_hex, 16)
    return f"PCI:{bus_dec}:{dev_dec}:{func}"


def _write_xorg_conf(gpu_index: int, pci_bus_id: str) -> str:
    """Write a minimal xorg.conf and return its path."""
    conf_path = f"/tmp/easi-xorg-gpu{gpu_index}.conf"
    conf = f"""\
Section "Device"
    Identifier "Device{gpu_index}"
    Driver     "nvidia"
    BusID      "{pci_bus_id}"
    Option     "AllowEmptyInitialConfiguration" "True"
EndSection

Section "Screen"
    Identifier "Screen{gpu_index}"
    Device     "Device{gpu_index}"
    DefaultDepth 24
    SubSection "Display"
        Depth   24
        Virtual 1920 1080
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier "Layout{gpu_index}"
    Screen     "Screen{gpu_index}"
EndSection
"""
    with open(conf_path, "w") as f:
        f.write(conf)
    return conf_path


class XorgManager:
    """Manages Xorg server processes for GPU-accelerated rendering.

    Starts one Xorg server per GPU ID, waits for each to become ready,
    and cleans up all on stop.
    """

    def __init__(self, gpu_ids: list[int], base_display: int = 10):
        self.gpu_ids = gpu_ids
        self.base_display = base_display
        self._processes: list[subprocess.Popen] = []
        self._used_sudo: list[bool] = []
        self._instances: list[XorgInstance] = []
        self._conf_files: list[str] = []

    def start(self) -> list[XorgInstance]:
        """Start Xorg on each GPU. Returns list of XorgInstance."""
        xorg_path = shutil.which("Xorg")
        if xorg_path is None:
            raise RuntimeError(
                "Xorg is not installed. Install with: apt install xserver-xorg"
            )

        try:
            next_display = self.base_display
            for gpu_id in self.gpu_ids:
                display_num = _find_available_display(next_display)
                next_display = display_num + 1
                instance = self._start_one(xorg_path, gpu_id, display_num)
                self._instances.append(instance)
        except Exception:
            logger.warning(
                "Xorg startup failed, stopping %d already-started servers",
                len(self._processes),
            )
            self.stop()
            raise

        logger.info(
            "All %d Xorg servers ready: %s",
            len(self._instances),
            [(f":{i.display}", f"GPU {i.gpu_id}") for i in self._instances],
        )
        return list(self._instances)

    def _start_one(
        self,
        xorg_path: str,
        gpu_id: int,
        display_num: int,
    ) -> XorgInstance:
        """Start a single Xorg server on the given GPU and display."""
        pci_bus_id = _get_pci_bus_id(gpu_id)
        conf_path = _write_xorg_conf(gpu_id, pci_bus_id)
        self._conf_files.append(conf_path)

        display_str = f":{display_num}"
        cmd = [
            xorg_path,
            display_str,
            "-config",
            conf_path,
            "-noreset",
            "-nolisten",
            "tcp",
        ]

        logger.info(
            "Starting Xorg on display %s using GPU %d (%s)",
            display_str,
            gpu_id,
            pci_bus_id,
        )

        proc, used_sudo = self._launch_xorg(cmd, xorg_path)
        self._processes.append(proc)
        self._used_sudo.append(used_sudo)

        try:
            self._wait_for_ready(display_num, proc)
        except RuntimeError as exc:
            if not used_sudo and proc.poll() is not None:
                if self._passwordless_sudo_available(xorg_path):
                    logger.info(
                        "Direct Xorg launch exited early on display %s; retrying with sudo",
                        display_str,
                    )
                    self._processes.pop()
                    self._used_sudo.pop()
                    proc, used_sudo = self._launch_xorg_with_sudo(cmd, xorg_path)
                    self._processes.append(proc)
                    self._used_sudo.append(used_sudo)
                    self._wait_for_ready(display_num, proc)
                else:
                    raise RuntimeError(
                        self._sudo_required_message(xorg_path, display_str)
                    ) from exc
            else:
                raise

        logger.info(
            "Xorg ready on display %s (PID %d, GPU %d)",
            display_str,
            proc.pid,
            gpu_id,
        )
        return XorgInstance(display=display_num, gpu_id=gpu_id, pid=proc.pid)

    def _launch_xorg(
        self, cmd: list[str], xorg_path: str
    ) -> tuple[subprocess.Popen, bool]:
        """Try launching Xorg directly, fall back to sudo on PermissionError.

        Returns (process, used_sudo) so stop() knows whether to use sudo kill.
        """
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            return proc, False
        except PermissionError:
            logger.info(
                "Direct Xorg launch failed (permission denied), retrying with sudo"
            )

        return self._launch_xorg_with_sudo(cmd, xorg_path)

    def _launch_xorg_with_sudo(
        self,
        cmd: list[str],
        xorg_path: str,
    ) -> tuple[subprocess.Popen, bool]:
        sudo_cmd = ["sudo", "-n"] + cmd
        try:
            proc = subprocess.Popen(
                sudo_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            return proc, True
        except (PermissionError, FileNotFoundError) as exc:
            raise RuntimeError(self._sudo_required_message(xorg_path)) from exc

    def _wait_for_ready(self, display_num: int, proc: subprocess.Popen) -> None:
        """Poll until the X server responds or timeout."""
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        display_str = f":{display_num}"

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Xorg exited with code {proc.returncode} on display {display_str}. "
                    f"Check /var/log/Xorg.{display_num}.log for details."
                )
            try:
                result = subprocess.run(
                    ["xset", "-display", display_str, "q"],
                    capture_output=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise RuntimeError(
            f"Xorg on display {display_str} did not become ready within "
            f"{_STARTUP_TIMEOUT}s"
        )

    def stop(self) -> None:
        """Stop all Xorg servers and clean up."""
        for proc, sudo in zip(self._processes, self._used_sudo):
            self._kill_proc(proc, signal.SIGTERM, sudo)

        for proc, sudo in zip(self._processes, self._used_sudo):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._kill_proc(proc, signal.SIGKILL, sudo)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

        self._processes.clear()
        self._used_sudo.clear()
        self._instances.clear()

        for conf in self._conf_files:
            try:
                os.unlink(conf)
            except OSError:
                pass
        self._conf_files.clear()

    @staticmethod
    def _kill_proc(proc: subprocess.Popen, sig: int, used_sudo: bool) -> None:
        """Send a signal to a process group, using sudo if the process was sudo-launched."""
        try:
            pgid = os.getpgid(proc.pid)
            if used_sudo:
                subprocess.run(
                    ["sudo", "-n", "kill", f"-{sig}", f"-{pgid}"],
                    capture_output=True,
                    timeout=5,
                )
            else:
                os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _passwordless_sudo_available(xorg_path: str) -> bool:
        """Probe ``sudo -n -l <xorg_path>`` — sudoers may only whitelist specific commands."""
        try:
            result = subprocess.run(
                ["sudo", "-n", "-l", xorg_path],
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    @staticmethod
    def _sudo_required_message(xorg_path: str, display_str: str | None = None) -> str:
        display_hint = ""
        if display_str is not None:
            display_hint = (
                f"Xorg exited before becoming ready on display {display_str}. "
            )
        user = os.environ.get("USER", "$USER")
        return (
            f"{display_hint}This usually requires root or a console session.\n"
            f"To fix this, run EASI as root or authorize passwordless sudo:\n\n"
            f"  sudo bash -c 'echo \"{user} ALL=(ALL) NOPASSWD: {xorg_path}, /usr/bin/kill\" > /etc/sudoers.d/easi-xorg'"
        )
