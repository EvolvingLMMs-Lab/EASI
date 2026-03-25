"""Tests for ServerManager — vLLM lifecycle management."""
import socket
import pytest


class TestServerManagerInit:
    def test_stores_config(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "Qwen/Qwen2.5-VL-72B", port=8080,
                           server_kwargs={"tensor_parallel_size": 4})
        assert sm.backend == "vllm"
        assert sm.model == "Qwen/Qwen2.5-VL-72B"
        assert sm.port == 8080
        assert sm.server_kwargs == {"tensor_parallel_size": 4}

    def test_default_port(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "some-model")
        assert sm.port == 8080


class TestBuildCommand:
    def test_vllm_basic(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "Qwen/Qwen2.5-VL-72B", port=9090)
        cmd, env = sm._build_command()
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd
        assert "Qwen/Qwen2.5-VL-72B" in cmd
        assert "--port" in cmd
        assert "9090" in cmd

    def test_vllm_with_kwargs(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model", port=8080,
                           server_kwargs={"tensor_parallel_size": 4,
                                          "gpu_memory_utilization": 0.9})
        cmd, env = sm._build_command()
        assert "--tensor-parallel-size" in cmd
        assert "4" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    def test_unsupported_backend_raises(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("unknown_backend", "test-model")
        with pytest.raises(ValueError, match="Unsupported server backend"):
            sm._build_command()


class TestPortCheck:
    def test_port_available(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test", port=19876)
        sm._check_port()  # should not raise

    def test_port_taken(self):
        from easi.llm.server_manager import ServerManager
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 19877))
        sock.listen(1)
        try:
            sm = ServerManager("vllm", "test", port=19877)
            with pytest.raises(RuntimeError, match="in use"):
                sm._check_port(retries=1, delay=0)
        finally:
            sock.close()


class TestContextManager:
    def test_context_manager_calls_stop(self):
        from unittest.mock import MagicMock
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test", port=8080)
        sm.start = MagicMock(return_value="http://localhost:8080/v1")
        sm.stop = MagicMock()
        with sm as url:
            assert url == "http://localhost:8080/v1"
        sm.stop.assert_called_once()


class TestIsRunning:
    def test_not_running_when_no_process(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test")
        assert sm.is_running() is False


class TestDefaultVllmFlags:
    def test_server_manager_default_vllm_flags(self):
        """ServerManager should include default vLLM flags (prefix caching, quiet logs)."""
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(backend="vllm", model="test-model", port=9999)
        cmd, _ = mgr._build_command()
        assert "--enable-prefix-caching" in cmd
        assert "--disable-log-requests" in cmd

    def test_server_manager_user_can_override_defaults(self):
        """User-provided server_kwargs should override defaults."""
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="vllm", model="test-model", port=9999,
            server_kwargs={"enable_prefix_caching": False},
        )
        cmd, _ = mgr._build_command()
        # False bool flags should NOT appear in command
        assert "--enable-prefix-caching" not in cmd


class TestCustomBackendCommand:
    def test_custom_backend_builds_command(self):
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="custom",
            model="my_model",
            port=8001,
            server_kwargs={"model_path": "/path/to/weights"},
        )
        cmd, env = mgr._build_command()
        assert "easi.llm.models.http_server" in cmd
        assert "--model-name" in cmd
        assert "my_model" in cmd
        assert "--port" in cmd

    def test_custom_backend_passes_model_path(self):
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="custom",
            model="my_model",
            port=8001,
            server_kwargs={"model_path": "/weights"},
        )
        cmd, _ = mgr._build_command()
        idx = cmd.index("--model-path")
        assert cmd[idx + 1] == "/weights"

    def test_custom_backend_default_device(self):
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="custom",
            model="my_model",
            port=8001,
        )
        cmd, _ = mgr._build_command()
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "cuda:0"

    def test_custom_backend_extra_kwargs_as_json(self):
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="custom",
            model="my_model",
            port=8001,
            server_kwargs={"model_path": "/w", "torch_dtype": "bfloat16"},
        )
        cmd, _ = mgr._build_command()
        assert "--kwargs" in cmd
        import json
        idx = cmd.index("--kwargs")
        parsed = json.loads(cmd[idx + 1])
        assert parsed["torch_dtype"] == "bfloat16"
        assert "model_path" not in parsed


class TestCudaVisibleDevices:
    def test_server_manager_sets_cuda_visible_devices(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model", cuda_visible_devices="0,1")
        _cmd, env = sm._build_command()
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_server_manager_no_cuda_by_default(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model")
        _cmd, env = sm._build_command()
        assert "CUDA_VISIBLE_DEVICES" not in env


class TestMultiServerManager:
    def test_multi_server_manager_starts_n_instances(self):
        """MultiServerManager should start N ServerManager instances with correct GPU assignment."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM, \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mock_instance = MagicMock()
            mock_instance.wait_until_ready.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1],
                base_port=8000,
            )
            urls = mgr.start()

            assert len(urls) == 2
            assert MockSM.call_count == 2
            # All instances should be launched then health-checked
            assert mock_instance.launch.call_count == 2
            assert mock_instance.wait_until_ready.call_count == 2

            # Check GPU assignment: instance 0 → GPU 0, instance 1 → GPU 1
            calls = MockSM.call_args_list
            assert calls[0].kwargs["cuda_visible_devices"] == "0"
            assert calls[0].kwargs["port"] == 8000
            assert calls[1].kwargs["cuda_visible_devices"] == "1"
            assert calls[1].kwargs["port"] == 8001

    def test_multi_server_manager_tp2_gpu_assignment(self):
        """With TP=2, each instance should get 2 GPUs."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM, \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mock_instance = MagicMock()
            mock_instance.wait_until_ready.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1, 2, 3],
                base_port=8000,
            )
            urls = mgr.start()

            calls = MockSM.call_args_list
            assert calls[0].kwargs["cuda_visible_devices"] == "0,1"
            assert calls[1].kwargs["cuda_visible_devices"] == "2,3"

    def test_multi_server_manager_stop_all(self):
        """stop() should stop all managed instances."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM, \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mock_instance = MagicMock()
            mock_instance.wait_until_ready.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(model="m", num_instances=2, gpu_ids=[0, 1])
            mgr.start()
            mgr.stop()

            assert mock_instance.stop.call_count == 2

    def test_multi_server_manager_validates_gpu_count(self):
        """Should raise if GPUs don't divide evenly across instances."""
        from easi.llm.server_manager import MultiServerManager

        with pytest.raises(ValueError, match="divide.*evenly"):
            MultiServerManager(model="m", num_instances=3, gpu_ids=[0, 1])

    def test_no_gpu_ids_skips_cuda_env(self):
        """With gpu_ids=None, cuda_visible_devices should not be set."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM, \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mock_instance = MagicMock()
            mock_instance.wait_until_ready.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=1,
                gpu_ids=None,
                base_port=8000,
            )
            mgr.start()

            calls = MockSM.call_args_list
            assert calls[0].kwargs["cuda_visible_devices"] is None

    def test_port_skips_taken(self):
        """Should auto-increment past taken ports."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        # Port 8000 available, 8001 taken, 8002 available
        port_map = {8000: True, 8001: False, 8002: True}

        with patch("easi.llm.server_manager.ServerManager") as MockSM, \
             patch("easi.llm.server_manager._port_is_available",
                   side_effect=lambda p: port_map.get(p, True)):
            mock_instance = MagicMock()
            mock_instance.wait_until_ready.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1],
                base_port=8000,
            )
            mgr.start()

            calls = MockSM.call_args_list
            assert calls[0].kwargs["port"] == 8000
            assert calls[1].kwargs["port"] == 8002  # skipped 8001

    def test_partial_failure_cleanup_on_launch(self):
        """If launch() fails for instance N, instances 0..N-1 should be stopped."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        instance_0 = MagicMock()
        instance_1 = MagicMock()
        instance_1.launch.side_effect = RuntimeError("launch failed")

        with patch("easi.llm.server_manager.ServerManager",
                   side_effect=[instance_0, instance_1]), \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1],
                base_port=8000,
            )
            with pytest.raises(RuntimeError, match="launch failed"):
                mgr.start()

            instance_0.stop.assert_called_once()

    def test_partial_failure_cleanup_on_health(self):
        """If wait_until_ready() fails, all spawned instances should be stopped."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        instance_0 = MagicMock()
        instance_0.wait_until_ready.return_value = "http://localhost:8000/v1"
        instance_1 = MagicMock()
        instance_1.wait_until_ready.side_effect = RuntimeError("health check failed")

        with patch("easi.llm.server_manager.ServerManager",
                   side_effect=[instance_0, instance_1]), \
             patch("easi.llm.server_manager._port_is_available", return_value=True):
            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1],
                base_port=8000,
            )
            with pytest.raises(RuntimeError, match="health check failed"):
                mgr.start()

            # Both should be stopped during cleanup
            instance_0.stop.assert_called_once()
            instance_1.stop.assert_called_once()
