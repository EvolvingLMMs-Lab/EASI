"""Tests verifying env vars flow from env_manager through to SubprocessRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_runner(sim_gpus=None, render_platform_name=None):
    from easi.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner.__new__(EvaluationRunner)
    runner.data_dir = Path("/tmp/fake")
    runner.render_platform_name = render_platform_name
    runner.sim_gpus = sim_gpus
    runner._render_platform = None
    return runner


def _make_env_mgr(env_vars=None, platform="headless"):
    from easi.core.render_platforms import EnvVars

    mgr = MagicMock()
    mgr.env_is_ready.return_value = True
    mgr.get_python_executable.return_value = "/usr/bin/python3"
    mgr.default_render_platform = platform
    mgr.supported_render_platforms = [platform]
    mgr.screen_config = "1024x768x24"
    mgr.get_env_vars.return_value = env_vars if env_vars is not None else EnvVars()
    return mgr


def _make_entry(render_adapter=None):
    entry = MagicMock()
    entry.runtime = "conda"
    entry.render_adapter = render_adapter
    return entry


def _make_sim_cls(bridge_path="/fake/bridge.py"):
    cls = MagicMock()
    cls.return_value._get_bridge_script_path.return_value = Path(bridge_path)
    return cls


class TestEnvVarsWiring:
    """Verify env vars flow from env_manager to SubprocessRunner."""

    def test_runner_passes_env_vars_to_subprocess(self):
        from easi.core.render_platforms import EnvVars, get_render_platform

        runner = _make_runner()
        mock_env_mgr = _make_env_mgr(EnvVars(replace={"SIM_ROOT": "/opt/sim"}))
        mock_entry = _make_entry()

        with (
            patch(
                "easi.simulators.registry.get_simulator_entry", return_value=mock_entry
            ),
            patch(
                "easi.simulators.registry.create_env_manager", return_value=mock_env_mgr
            ),
            patch(
                "easi.simulators.registry.load_simulator_class",
                return_value=_make_sim_cls(),
            ),
            patch(
                "easi.simulators.registry.resolve_render_platform",
                side_effect=lambda key, name, env_manager=None: get_render_platform(
                    name
                ),
            ),
            patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner,
        ):
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            extra_env = MockRunner.call_args.kwargs.get("extra_env")
            assert isinstance(extra_env, EnvVars)
            assert extra_env.replace == {"SIM_ROOT": "/opt/sim"}

    def test_runner_passes_none_when_no_env_vars(self):
        from easi.core.render_platforms import get_render_platform

        runner = _make_runner()
        mock_env_mgr = _make_env_mgr()
        mock_entry = _make_entry()

        with (
            patch(
                "easi.simulators.registry.get_simulator_entry", return_value=mock_entry
            ),
            patch(
                "easi.simulators.registry.create_env_manager", return_value=mock_env_mgr
            ),
            patch(
                "easi.simulators.registry.load_simulator_class",
                return_value=_make_sim_cls(),
            ),
            patch(
                "easi.simulators.registry.resolve_render_platform",
                side_effect=lambda key, name, env_manager=None: get_render_platform(
                    name
                ),
            ),
            patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner,
        ):
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            assert MockRunner.call_args.kwargs.get("extra_env") is None


class TestBindingCompositionWiring:
    """Prove binding+adapter env composition is correct in the runner."""

    def _run_with_binding(self, binding, sim_gpus=None, sim_env=None, adapter=None):
        from easi.core.render_platforms import EnvVars, get_render_platform

        runner = _make_runner(sim_gpus=sim_gpus)

        mock_platform = MagicMock()
        mock_platform.name = "xorg"
        mock_platform.for_worker.return_value = binding
        mock_platform.get_env_vars.return_value = EnvVars()
        mock_platform.wrap_command.side_effect = lambda cmd, cfg: cmd

        mock_env_mgr = _make_env_mgr(env_vars=sim_env, platform="xorg")
        mock_env_mgr.supported_render_platforms = ["xorg"]
        mock_env_mgr.default_render_platform = "xorg"
        mock_entry = _make_entry()

        with (
            patch(
                "easi.simulators.registry.get_simulator_entry", return_value=mock_entry
            ),
            patch(
                "easi.simulators.registry.create_env_manager", return_value=mock_env_mgr
            ),
            patch(
                "easi.simulators.registry.load_simulator_class",
                return_value=_make_sim_cls(),
            ),
            patch(
                "easi.simulators.registry.resolve_render_platform",
                return_value=mock_platform,
            ),
            patch(
                "easi.simulators.registry.resolve_render_adapter",
                return_value=adapter,
            ),
            patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner,
        ):
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            return MockRunner.call_args.kwargs

    def test_binding_display_appears_in_extra_env(self):
        from easi.core.render_platforms import WorkerBinding

        binding = WorkerBinding(display=":10", cuda_visible_devices="0")
        kwargs = self._run_with_binding(binding)
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("DISPLAY") == ":10"

    def test_binding_cuda_appears_in_extra_env(self):
        from easi.core.render_platforms import WorkerBinding

        binding = WorkerBinding(display=":10", cuda_visible_devices="2")
        kwargs = self._run_with_binding(binding)
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("CUDA_VISIBLE_DEVICES") == "2"

    def test_adapter_env_merged_into_extra_env(self):
        from easi.core.render_platforms import (
            EnvVars,
            SimulatorRenderAdapter,
            WorkerBinding,
        )

        class MarkingAdapter(SimulatorRenderAdapter):
            def get_env_vars(self, binding):
                return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0"})

        binding = WorkerBinding(display=":10", cuda_visible_devices="0")
        kwargs = self._run_with_binding(binding, adapter=MarkingAdapter())
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("OMNIGIBSON_HEADLESS") == "0"

    def test_general_sim_env_preserved_alongside_binding_env(self):
        from easi.core.render_platforms import EnvVars, WorkerBinding

        binding = WorkerBinding(display=":11", cuda_visible_devices="1")
        sim_env = EnvVars(replace={"DATA_ROOT": "/mnt/data"})
        kwargs = self._run_with_binding(binding, sim_env=sim_env)
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("DATA_ROOT") == "/mnt/data"
        assert extra_env.replace.get("DISPLAY") == ":11"

    def test_explicit_gpu_binding_prevents_sim_gpus_override(self):
        from easi.core.render_platforms import WorkerBinding

        binding = WorkerBinding(display=":10", cuda_visible_devices="0")
        kwargs = self._run_with_binding(binding, sim_gpus=[3])
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_gpu_pinning_applied_when_binding_has_no_cuda(self):
        from easi.core.render_platforms import WorkerBinding

        binding = WorkerBinding(display=":10", cuda_visible_devices=None)
        kwargs = self._run_with_binding(binding, sim_gpus=[5])
        extra_env = kwargs.get("extra_env")
        assert extra_env is not None
        assert extra_env.replace.get("CUDA_VISIBLE_DEVICES") == "5"

    def test_base_platform_used_for_command_wrapping_not_binding(self):
        from easi.core.render_platforms import WorkerBinding

        binding = WorkerBinding(display=":10", cuda_visible_devices="0")
        kwargs = self._run_with_binding(binding)
        rp = kwargs.get("render_platform")
        assert rp is not None
        assert rp.name == "xorg"

    def test_binding_path_passes_adapter_and_binding_to_subprocess_runner(self):
        from easi.core.render_platforms import (
            EnvVars,
            SimulatorRenderAdapter,
            WorkerBinding,
        )

        class SentinelAdapter(SimulatorRenderAdapter):
            def get_env_vars(self, binding):
                return EnvVars()

        adapter = SentinelAdapter()
        binding = WorkerBinding(display=":10", cuda_visible_devices="0")
        kwargs = self._run_with_binding(binding, adapter=adapter)
        assert kwargs.get("render_adapter") is adapter
        assert kwargs.get("worker_binding") is binding

    def test_no_adapter_registered_passes_none_adapter_with_default_binding(self):
        from easi.core.render_platforms import (
            EnvVars,
            WorkerBinding,
            get_render_platform,
        )

        runner = _make_runner()
        mock_env_mgr = _make_env_mgr(env_vars=EnvVars(replace={"X": "1"}))
        mock_entry = _make_entry()

        with (
            patch(
                "easi.simulators.registry.get_simulator_entry", return_value=mock_entry
            ),
            patch(
                "easi.simulators.registry.create_env_manager", return_value=mock_env_mgr
            ),
            patch(
                "easi.simulators.registry.load_simulator_class",
                return_value=_make_sim_cls(),
            ),
            patch(
                "easi.simulators.registry.resolve_render_platform",
                side_effect=lambda key, name, env_manager=None: get_render_platform(
                    name
                ),
            ),
            patch(
                "easi.simulators.registry.resolve_render_adapter",
                return_value=None,
            ),
            patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner,
        ):
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            kwargs = MockRunner.call_args.kwargs
            assert kwargs.get("render_adapter") is None
            assert isinstance(kwargs.get("worker_binding"), WorkerBinding)
