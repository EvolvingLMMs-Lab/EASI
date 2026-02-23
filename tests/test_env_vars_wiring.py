"""Tests verifying env vars flow from env_manager through to SubprocessRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestEnvVarsWiring:
    """Verify env vars flow from env_manager to SubprocessRunner."""

    def test_runner_passes_env_vars_to_subprocess(self):
        from easi.core.render_platform import EnvVars, get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = None

        mock_env_mgr = MagicMock()
        mock_env_mgr.env_is_ready.return_value = True
        mock_env_mgr.get_python_executable.return_value = "/usr/bin/python3"
        mock_env_mgr.default_render_platform = "headless"
        mock_env_mgr.supported_render_platforms = ["headless"]
        mock_env_mgr.screen_config = "1024x768x24"
        mock_env_mgr.get_env_vars.return_value = EnvVars(replace={"SIM_ROOT": "/opt/sim"})

        mock_sim_cls = MagicMock()
        mock_sim = mock_sim_cls.return_value
        mock_sim._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        with patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.launch.return_value = None

            runner._create_simulator("fake:v1")

            call_kwargs = MockRunner.call_args
            extra_env = call_kwargs.kwargs.get("extra_env")
            assert isinstance(extra_env, EnvVars)
            assert extra_env.replace == {"SIM_ROOT": "/opt/sim"}

    def test_runner_passes_none_when_no_env_vars(self):
        from easi.core.render_platform import EnvVars, get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = None

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

        with patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            mock_runner_instance = MockRunner.return_value
            mock_runner_instance.launch.return_value = None

            runner._create_simulator("fake:v1")

            call_kwargs = MockRunner.call_args
            assert call_kwargs.kwargs.get("extra_env") is None
