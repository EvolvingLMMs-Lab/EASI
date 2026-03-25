"""Tests for Xorg render platform and manager."""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

_MGR = "easi.core.render_platforms.xorg_manager"  # canonical module for XorgManager internals
_XORG = "easi.core.render_platforms.xorg"  # canonical module for XorgPlatform


class TestXorgPlatformLifecycle:
    """Test XorgPlatform setup/teardown/for_worker lifecycle."""

    def test_module_no_longer_exports_worker_platform_shim(self):
        import easi.core.render_platforms.xorg as xorg_module

        assert not hasattr(xorg_module, "_XorgWorkerPlatform")

    def test_name(self):
        from easi.core.render_platforms.xorg import XorgPlatform

        p = XorgPlatform()
        assert p.name == "xorg"

    def test_for_worker_without_setup_raises(self):
        from easi.core.render_platforms.xorg import XorgPlatform

        p = XorgPlatform()
        with pytest.raises(RuntimeError, match="setup.*must be called"):
            p.for_worker(0)

    def test_teardown_without_setup_is_safe(self):
        from easi.core.render_platforms.xorg import XorgPlatform

        p = XorgPlatform()
        p.teardown()  # should not raise


class TestXorgManager:
    """Test XorgManager lifecycle with mocked subprocesses."""

    def test_no_xorg_binary_raises(self):
        from easi.core.render_platforms.xorg_manager import XorgManager

        with patch(f"{_MGR}.shutil.which", return_value=None):
            mgr = XorgManager(gpu_ids=[0])
            with pytest.raises(RuntimeError, match="Xorg is not installed"):
                mgr.start()

    def test_start_single_gpu(self):
        from easi.core.render_platforms.xorg_manager import XorgManager

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with (
            patch(f"{_MGR}.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch(f"{_MGR}.subprocess.Popen", return_value=mock_proc),
            patch(f"{_MGR}.subprocess.run") as mock_run,
            patch(f"{_MGR}.os.path.exists", return_value=False),
            patch(f"{_MGR}._write_xorg_conf", return_value="/tmp/easi-xorg-gpu0.conf"),
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
        from easi.core.render_platforms.xorg_manager import XorgManager

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None

        with (
            patch(f"{_MGR}.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch(f"{_MGR}.subprocess.Popen", return_value=mock_proc),
            patch(f"{_MGR}.subprocess.run") as mock_run,
            patch(f"{_MGR}.os.path.exists", return_value=False),
            patch(f"{_MGR}._write_xorg_conf", return_value="/tmp/easi-xorg-gpu.conf"),
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
        from easi.core.render_platforms.xorg_manager import XorgInstance, XorgManager

        mgr = XorgManager(gpu_ids=[0])
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        mgr._processes = [mock_proc]
        mgr._used_sudo = [False]
        mgr._instances = [XorgInstance(display=10, gpu_id=0, pid=12345)]
        mgr._conf_files = []

        with (
            patch(f"{_MGR}.os.getpgid", return_value=12345),
            patch(f"{_MGR}.os.killpg") as mock_killpg,
        ):
            mgr.stop()
            mock_killpg.assert_called_with(12345, signal.SIGTERM)

    def test_stop_uses_sudo_kill_for_sudo_launched(self):
        """Sudo-launched Xorg processes are killed via sudo kill."""
        from easi.core.render_platforms.xorg_manager import XorgInstance, XorgManager

        mgr = XorgManager(gpu_ids=[0])
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        mgr._processes = [mock_proc]
        mgr._used_sudo = [True]
        mgr._instances = [XorgInstance(display=10, gpu_id=0, pid=12345)]
        mgr._conf_files = []

        with (
            patch(f"{_MGR}.os.getpgid", return_value=12345),
            patch(f"{_MGR}.os.killpg") as mock_killpg,
            patch(f"{_MGR}.subprocess.run") as mock_run,
        ):
            mgr.stop()
            mock_killpg.assert_not_called()
            mock_run.assert_called_once_with(
                ["sudo", "-n", "kill", f"-{signal.SIGTERM}", "-12345"],
                capture_output=True,
                timeout=5,
            )

    def test_sudo_fallback(self):
        """If direct Xorg fails with PermissionError, retry with sudo."""
        from easi.core.render_platforms.xorg_manager import XorgManager

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
            patch(f"{_MGR}.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch(f"{_MGR}.subprocess.Popen", side_effect=mock_popen),
            patch(f"{_MGR}.subprocess.run") as mock_run,
            patch(f"{_MGR}.os.path.exists", return_value=False),
            patch(f"{_MGR}._write_xorg_conf", return_value="/tmp/easi-xorg-gpu0.conf"),
        ):
            nvidia_result = MagicMock(returncode=0, stdout="00000000:3F:00.0\n")
            xset_result = MagicMock(returncode=0)
            mock_run.side_effect = [nvidia_result, xset_result]

            mgr = XorgManager(gpu_ids=[0], base_display=10)
            instances = mgr.start()

            assert len(instances) == 1
            assert call_count == 2
            assert mgr._used_sudo == [True]

    def test_start_failure_stops_all(self):
        """If one GPU's Xorg fails, previously started ones are stopped."""
        from easi.core.render_platforms.xorg_manager import XorgManager

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
            patch(f"{_MGR}.shutil.which", return_value="/usr/lib/xorg/Xorg"),
            patch(f"{_MGR}.subprocess.Popen", side_effect=mock_popen),
            patch(f"{_MGR}.subprocess.run") as mock_run,
            patch(f"{_MGR}.os.path.exists", return_value=False),
            patch(f"{_MGR}.os.getpgid", return_value=11111),
            patch(f"{_MGR}.os.killpg"),
            patch(f"{_MGR}._write_xorg_conf", return_value="/tmp/easi-xorg-gpu.conf"),
        ):
            nvidia0 = MagicMock(returncode=0, stdout="00000000:3F:00.0\n")
            xset0 = MagicMock(returncode=0)
            nvidia1 = MagicMock(returncode=0, stdout="00000000:9B:00.0\n")
            mock_run.side_effect = [nvidia0, xset0, nvidia1]

            mgr = XorgManager(gpu_ids=[0, 1], base_display=10)
            with pytest.raises(RuntimeError, match="GPU 1 Xorg failed"):
                mgr.start()

            assert len(started_procs) == 1


class TestPerWorkerGpuPinning:
    """Test per-worker GPU round-robin via XorgPlatform.for_worker()."""

    def test_round_robin_two_gpus(self):
        """Workers are assigned to GPUs in round-robin order."""
        from easi.core.render_platforms.xorg import XorgPlatform
        from easi.core.render_platforms.xorg_manager import XorgInstance

        p = XorgPlatform()
        p._instances = [
            XorgInstance(display=10, gpu_id=4, pid=1),
            XorgInstance(display=11, gpu_id=5, pid=2),
        ]
        gpus = [p.for_worker(i).cuda_visible_devices for i in range(6)]
        assert gpus == ["4", "5", "4", "5", "4", "5"]

    def test_single_gpu_all_workers_same(self):
        """With one GPU, all workers get the same GPU."""
        from easi.core.render_platforms.xorg import XorgPlatform
        from easi.core.render_platforms.xorg_manager import XorgInstance

        p = XorgPlatform()
        p._instances = [XorgInstance(display=10, gpu_id=4, pid=1)]
        for worker_id in range(4):
            assert p.for_worker(worker_id).cuda_visible_devices == "4"

    def test_binding_contains_display_and_env(self):
        from easi.core.render_platforms import WorkerBinding
        from easi.core.render_platforms.xorg import XorgPlatform
        from easi.core.render_platforms.xorg_manager import XorgInstance

        p = XorgPlatform()
        p._instances = [XorgInstance(display=12, gpu_id=7, pid=1)]

        binding = p.for_worker(0)

        assert isinstance(binding, WorkerBinding)
        assert binding.display == ":12"
        assert binding.cuda_visible_devices == "7"
        assert binding.extra_env.replace == {"EASI_GPU_DISPLAY": "1"}


class TestXorgRunnerIntegration:
    """Test Xorg integration logic in runners."""

    def test_xorg_defaults_to_gpu_0_when_no_sim_gpus(self):
        """XorgPlatform.setup() uses GPU 0 when gpu_ids is None."""
        from easi.core.render_platforms.xorg import XorgPlatform
        from easi.core.render_platforms.xorg_manager import XorgInstance

        p = XorgPlatform()
        mock_mgr = MagicMock()
        mock_mgr.start.return_value = [XorgInstance(display=10, gpu_id=0, pid=1)]

        with patch(f"{_XORG}.XorgManager", return_value=mock_mgr) as MockMgr:
            p.setup(gpu_ids=None)
        MockMgr.assert_called_once_with(gpu_ids=[0])

    def test_xorg_uses_sim_gpus_when_specified(self):
        """XorgPlatform.setup() uses provided gpu_ids."""
        from easi.core.render_platforms.xorg import XorgPlatform
        from easi.core.render_platforms.xorg_manager import XorgInstance

        p = XorgPlatform()
        mock_mgr = MagicMock()
        mock_mgr.start.return_value = [
            XorgInstance(display=10, gpu_id=2, pid=1),
            XorgInstance(display=11, gpu_id=3, pid=2),
        ]

        with patch(f"{_XORG}.XorgManager", return_value=mock_mgr) as MockMgr:
            p.setup(gpu_ids=[2, 3])
        MockMgr.assert_called_once_with(gpu_ids=[2, 3])
