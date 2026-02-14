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
        cmd = sm._build_command()
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
        cmd = sm._build_command()
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
            with pytest.raises(RuntimeError, match="already in use"):
                sm._check_port()
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
