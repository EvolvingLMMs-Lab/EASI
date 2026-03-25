"""Tests for installation_kwargs flow: manifest -> registry -> env_manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSimulatorEntryHasInstallationKwargs:
    """SimulatorEntry dataclass includes installation_kwargs."""

    def test_dummy_entry_has_empty_installation_kwargs(self):
        from easi.simulators.registry import get_simulator_entry
        entry = get_simulator_entry("dummy")
        assert hasattr(entry, "installation_kwargs")
        assert entry.installation_kwargs == {}

    def test_ai2thor_entry_has_empty_installation_kwargs(self):
        from easi.simulators.registry import get_simulator_entry
        entry = get_simulator_entry("ai2thor:v2_1_0")
        assert entry.installation_kwargs == {}


class TestBaseEnvManagerAcceptsInstallationKwargs:
    """BaseEnvironmentManager stores installation_kwargs."""

    def test_default_installation_kwargs_is_empty(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager
        mgr = DummyEnvManager()
        assert mgr.installation_kwargs == {}

    def test_accepts_installation_kwargs(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager
        mgr = DummyEnvManager(installation_kwargs={"foo": "bar"})
        assert mgr.installation_kwargs == {"foo": "bar"}


class TestCreateEnvManagerFactory:
    """Registry factory function passes installation_kwargs."""

    def test_create_env_manager_returns_instance(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("dummy")
        assert mgr.simulator_name == "dummy"

    def test_create_env_manager_passes_kwargs(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("dummy")
        assert isinstance(mgr.installation_kwargs, dict)


class TestRunnerUsesFactory:
    """EvaluationRunner._create_simulator uses the factory."""

    def test_runner_passes_installation_kwargs(self):
        from easi.core.render_platforms import EnvVars, get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = None
        runner.sim_gpus = None
        runner._render_platform = None

        mock_env_mgr = MagicMock()
        mock_env_mgr.env_is_ready.return_value = True
        mock_env_mgr.get_python_executable.return_value = "/usr/bin/python3"
        mock_env_mgr.default_render_platform = "headless"
        mock_env_mgr.supported_render_platforms = ["headless"]
        mock_env_mgr.screen_config = "1024x768x24"
        mock_env_mgr.get_env_vars.return_value = EnvVars()

        mock_sim_cls = MagicMock()
        mock_sim = mock_sim_cls.return_value
        mock_sim._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr) as mock_factory, \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.launch.return_value = None

            runner._create_simulator("fake:v1")

            mock_factory.assert_called_once_with("fake:v1")
