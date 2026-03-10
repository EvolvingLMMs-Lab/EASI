"""Tests for Xorg render platform and manager."""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

import pytest


class TestXorgPlatform:
    """Test XorgPlatform env vars and command wrapping."""

    def test_name(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        assert p.name == "xorg"

    def test_env_vars(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        ev = p.get_env_vars()
        assert ev.replace["DISPLAY"] == ":10"
        assert ev.replace["CUDA_VISIBLE_DEVICES"] == "4"
        assert ev.replace["EASI_GPU_DISPLAY"] == "1"

    def test_wrap_command_passthrough(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        cmd = ["python", "bridge.py", "--workspace", "/tmp"]
        assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_is_available(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        assert p.is_available() is True


class TestXorgManager:
    """Test XorgManager lifecycle with mocked subprocesses."""

    def test_no_xorg_binary_raises(self):
        from easi.core.xorg_manager import XorgManager

        with patch("easi.core.xorg_manager.shutil.which", return_value=None):
            mgr = XorgManager(gpu_ids=[0])
            with pytest.raises(RuntimeError, match="Xorg is not installed"):
                mgr.start()

    def test_start_single_gpu(self):
        from easi.core.xorg_manager import XorgManager

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with (
            patch("easi.core.xorg_manager.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch("easi.core.xorg_manager.subprocess.Popen", return_value=mock_proc),
            patch("easi.core.xorg_manager.subprocess.run") as mock_run,
            patch("easi.core.xorg_manager.os.path.exists", return_value=False),
            patch("easi.core.xorg_manager._write_xorg_conf", return_value="/tmp/easi-xorg-gpu0.conf"),
        ):
            nvidia_result = MagicMock()
            nvidia_result.returncode = 0
            nvidia_result.stdout = "00000000:3F:00.0\n"
            xset_result = MagicMock()
            xset_result.returncode = 0
            mock_run.side_effect = [nvidia_result, xset_result]

            mgr = XorgManager(gpu_ids=[0], base_display=10)
            instances = mgr.start()

            assert len(instances) == 1
            assert instances[0].display == 10
            assert instances[0].gpu_id == 0
            assert instances[0].pid == 12345

    def test_start_multi_gpu(self):
        from easi.core.xorg_manager import XorgManager

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None

        with (
            patch("easi.core.xorg_manager.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch("easi.core.xorg_manager.subprocess.Popen", return_value=mock_proc),
            patch("easi.core.xorg_manager.subprocess.run") as mock_run,
            patch("easi.core.xorg_manager.os.path.exists", return_value=False),
            patch("easi.core.xorg_manager._write_xorg_conf", return_value="/tmp/easi-xorg-gpu.conf"),
        ):
            nvidia0 = MagicMock(returncode=0, stdout="00000000:3F:00.0\n")
            xset0 = MagicMock(returncode=0)
            nvidia1 = MagicMock(returncode=0, stdout="00000000:9B:00.0\n")
            xset1 = MagicMock(returncode=0)
            mock_run.side_effect = [nvidia0, xset0, nvidia1, xset1]

            mgr = XorgManager(gpu_ids=[4, 5], base_display=10)
            instances = mgr.start()

            assert len(instances) == 2
            assert instances[0].gpu_id == 4
            assert instances[1].gpu_id == 5
            assert instances[0].display != instances[1].display

    def test_stop_sends_sigterm(self):
        from easi.core.xorg_manager import XorgInstance, XorgManager

        mgr = XorgManager(gpu_ids=[0])
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        mgr._processes = [mock_proc]
        mgr._instances = [XorgInstance(display=10, gpu_id=0, pid=12345)]
        mgr._conf_files = []

        with patch("easi.core.xorg_manager.os.getpgid", return_value=12345), \
             patch("easi.core.xorg_manager.os.killpg") as mock_killpg:
            mgr.stop()
            mock_killpg.assert_called_with(12345, signal.SIGTERM)

    def test_sudo_fallback(self):
        """If direct Xorg fails with PermissionError, retry with sudo."""
        from easi.core.xorg_manager import XorgManager

        call_count = 0
        def mock_popen(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "sudo" not in cmd:
                raise PermissionError("Operation not permitted")
            proc = MagicMock()
            proc.pid = 55555
            proc.poll.return_value = None
            return proc

        with (
            patch("easi.core.xorg_manager.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch("easi.core.xorg_manager.subprocess.Popen", side_effect=mock_popen),
            patch("easi.core.xorg_manager.subprocess.run") as mock_run,
            patch("easi.core.xorg_manager.os.path.exists", return_value=False),
            patch("easi.core.xorg_manager._write_xorg_conf", return_value="/tmp/easi-xorg-gpu0.conf"),
        ):
            nvidia_result = MagicMock(returncode=0, stdout="00000000:3F:00.0\n")
            xset_result = MagicMock(returncode=0)
            mock_run.side_effect = [nvidia_result, xset_result]

            mgr = XorgManager(gpu_ids=[0], base_display=10)
            instances = mgr.start()

            assert len(instances) == 1
            assert call_count == 2

    def test_start_failure_stops_all(self):
        """If one GPU's Xorg fails, previously started ones are stopped."""
        from easi.core.xorg_manager import XorgManager

        started_procs = []

        def mock_popen(cmd, **kwargs):
            if len(started_procs) >= 1:
                raise RuntimeError("GPU 1 Xorg failed")
            proc = MagicMock()
            proc.pid = 11111
            proc.poll.return_value = None
            proc.wait.return_value = 0
            started_procs.append(proc)
            return proc

        with (
            patch("easi.core.xorg_manager.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch("easi.core.xorg_manager.subprocess.Popen", side_effect=mock_popen),
            patch("easi.core.xorg_manager.subprocess.run") as mock_run,
            patch("easi.core.xorg_manager.os.path.exists", return_value=False),
            patch("easi.core.xorg_manager.os.getpgid", return_value=11111),
            patch("easi.core.xorg_manager.os.killpg"),
            patch("easi.core.xorg_manager._write_xorg_conf", return_value="/tmp/easi-xorg-gpu.conf"),
        ):
            nvidia0 = MagicMock(returncode=0, stdout="00000000:3F:00.0\n")
            xset0 = MagicMock(returncode=0)
            nvidia1 = MagicMock(returncode=0, stdout="00000000:9B:00.0\n")
            mock_run.side_effect = [nvidia0, xset0, nvidia1]

            mgr = XorgManager(gpu_ids=[0, 1], base_display=10)
            with pytest.raises(RuntimeError, match="GPU 1 Xorg failed"):
                mgr.start()

            assert len(started_procs) == 1
