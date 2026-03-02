# tests/test_docker_isolation.py
"""Tests for Docker simulator isolation layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSimulatorEntryRuntime:
    """SimulatorEntry supports runtime field."""

    def test_default_runtime_is_conda(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
        )
        assert entry.runtime == "conda"

    def test_runtime_can_be_docker(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
            runtime="docker",
        )
        assert entry.runtime == "docker"

    def test_data_dir_default_empty(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
        )
        assert entry.data_dir == ""

    def test_data_dir_can_be_set(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
            data_dir="/datasets/test",
        )
        assert entry.data_dir == "/datasets/test"


class TestDockerEnvironmentManager:
    """Tests for DockerEnvironmentManager base class."""

    def _make_manager(self, **overrides):
        """Create a concrete DockerEnvironmentManager for testing."""
        from easi.core.docker_env_manager import DockerEnvironmentManager

        defaults = dict(
            _simulator_name="test_docker_sim",
            _version="v1",
            _image_name="easi_test_docker_sim_v1",
            _dockerfile_path=Path("/tmp/Dockerfile"),
            _gpu_required=False,
            _container_python_path="/usr/bin/python3",
            _container_data_mount="/data",
            _easi_mount="/opt/easi",
            _system_deps=["docker"],
        )
        defaults.update(overrides)

        class ConcreteDockerEnvManager(DockerEnvironmentManager):
            @property
            def simulator_name(self):
                return defaults["_simulator_name"]

            @property
            def version(self):
                return defaults["_version"]

            @property
            def image_name(self):
                return defaults["_image_name"]

            @property
            def dockerfile_path(self):
                return defaults["_dockerfile_path"]

            @property
            def gpu_required(self):
                return defaults["_gpu_required"]

            @property
            def container_python_path(self):
                return defaults["_container_python_path"]

            @property
            def container_data_mount(self):
                return defaults["_container_data_mount"]

            @property
            def easi_mount(self):
                return defaults["_easi_mount"]

            def get_system_deps(self):
                return defaults["_system_deps"]

        return ConcreteDockerEnvManager()

    def test_image_name(self):
        mgr = self._make_manager()
        assert mgr.image_name == "easi_test_docker_sim_v1"

    def test_gpu_required_default_false(self):
        mgr = self._make_manager()
        assert mgr.gpu_required is False

    def test_gpu_required_true(self):
        mgr = self._make_manager(_gpu_required=True)
        assert mgr.gpu_required is True

    def test_system_deps_includes_docker(self):
        mgr = self._make_manager()
        assert "docker" in mgr.get_system_deps()

    def test_system_deps_includes_nvidia_docker_when_gpu(self):
        mgr = self._make_manager(_gpu_required=True, _system_deps=["docker", "nvidia-docker"])
        deps = mgr.get_system_deps()
        assert "nvidia-docker" in deps

    def test_env_is_ready_false_when_no_docker(self):
        """env_is_ready returns False when docker image doesn't exist."""
        mgr = self._make_manager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("docker not found")
            assert mgr.env_is_ready() is False

    def test_env_is_ready_true_when_image_exists(self):
        """env_is_ready returns True when docker image inspect succeeds."""
        mgr = self._make_manager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert mgr.env_is_ready() is True

    def test_build_docker_run_command_basic(self):
        """Build docker run command without GPU."""
        mgr = self._make_manager()
        cmd = mgr.build_docker_run_command(
            bridge_command=["/usr/bin/python3", "/opt/easi/bridge.py", "--workspace", "/tmp/easi_xxx"],
            workspace_dir="/tmp/easi_xxx",
            episode_output_dir="/logs/ep_0",
            data_dir="/host/data",
        )
        assert "docker" in cmd[0]
        assert "--rm" in cmd
        assert "--gpus" not in cmd
        # Check volume mounts
        cmd_str = " ".join(cmd)
        assert "/tmp/easi_xxx" in cmd_str
        assert "/logs/ep_0" in cmd_str
        assert "/host/data" in cmd_str

    def test_build_docker_run_command_with_gpu(self):
        """Build docker run command with GPU."""
        mgr = self._make_manager(_gpu_required=True)
        cmd = mgr.build_docker_run_command(
            bridge_command=["/usr/bin/python3", "/opt/easi/bridge.py"],
            workspace_dir="/tmp/easi_xxx",
        )
        assert "--gpus" in cmd
        idx = cmd.index("--gpus")
        assert cmd[idx + 1] == "all"


class TestSubprocessRunnerDockerMode:
    """Tests for Docker launch mode in SubprocessRunner."""

    def test_launch_docker_builds_correct_command(self):
        """launch_docker() builds a docker run command and spawns it."""
        from easi.core.docker_env_manager import DockerEnvironmentManager
        from easi.simulators.subprocess_runner import SubprocessRunner
        from easi.core.render_platform import get_render_platform

        # Use a real bridge script path under the EASI repo root so that
        # relative_to() in _build_docker_launch_command works correctly.
        repo_root = Path(__file__).resolve().parents[1]
        bridge_path = repo_root / "easi" / "simulators" / "dummy" / "v1" / "bridge.py"

        # Create a mock docker env manager
        mock_mgr = MagicMock(spec=DockerEnvironmentManager)
        mock_mgr.image_name = "easi_test_v1"
        mock_mgr.gpu_required = False
        mock_mgr.container_python_path = "/usr/bin/python3"
        mock_mgr.easi_mount = "/opt/easi"
        mock_mgr.build_docker_run_command.return_value = [
            "docker", "run", "--rm",
            "-v", "/tmp/easi_xxx:/tmp/easi_xxx",
            "easi_test_v1",
            "/usr/bin/python3", "/opt/easi/easi/simulators/dummy/v1/bridge.py",
            "--workspace", "/tmp/easi_xxx",
        ]

        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=bridge_path,
            render_platform=get_render_platform("headless"),
        )

        cmd = runner._build_docker_launch_command(
            docker_env_manager=mock_mgr,
            workspace_dir="/tmp/easi_xxx",
        )
        mock_mgr.build_docker_run_command.assert_called_once()
        assert cmd[0] == "docker"

    # Note: Docker containers use --rm and foreground mode.
    # Shutdown uses the same process-tree kill as conda mode.
    # The --rm flag ensures container cleanup after process exit.


class TestDockerSystemDeps:
    """Test docker system dependency checking."""

    def test_docker_dep_registered(self):
        """Docker dependency checker is registered."""
        from easi.utils.system_deps import SystemDependencyChecker

        checker = SystemDependencyChecker()
        result = checker.check("docker")
        assert isinstance(result, bool)

    def test_nvidia_docker_dep_registered(self):
        """nvidia-docker dependency checker is registered."""
        from easi.utils.system_deps import SystemDependencyChecker

        checker = SystemDependencyChecker()
        result = checker.check("nvidia-docker")
        assert isinstance(result, bool)
