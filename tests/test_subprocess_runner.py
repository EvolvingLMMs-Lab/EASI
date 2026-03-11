"""Tests for SubprocessRunner env var injection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from easi.core.render_platforms import EnvVars, WorkerBinding, get_render_platform
from easi.simulators.subprocess_runner import SubprocessRunner


class TestSubprocessRunnerEnvVars:
    """Tests for env var injection into bridge subprocess."""

    def test_constructor_accepts_env_vars(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(replace={"MY_VAR": "my_value"}),
        )
        assert runner.extra_env.replace == {"MY_VAR": "my_value"}

    def test_default_env_is_none(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
        )
        assert runner.extra_env is None

    def test_build_env_merges_with_os_environ(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(replace={"COPPELIASIM_ROOT": "/opt/coppeliasim"}),
        )
        env = runner._build_subprocess_env()
        assert env["COPPELIASIM_ROOT"] == "/opt/coppeliasim"
        assert "PATH" in env

    def test_build_env_returns_none_when_no_extra(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
        )
        assert runner._build_subprocess_env() is None

    def test_extra_env_prepends_to_path_vars(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(prepend={"LD_LIBRARY_PATH": "/opt/sim/lib"}),
        )
        env = runner._build_subprocess_env()
        ld_path = env.get("LD_LIBRARY_PATH", "")
        assert ld_path.startswith("/opt/sim/lib")

    def test_non_path_var_replaces(self):
        os.environ["MY_EXISTING"] = "old_value"
        try:
            runner = SubprocessRunner(
                python_executable="/usr/bin/python3",
                bridge_script_path=Path("/dev/null"),
                render_platform=get_render_platform("headless"),
                extra_env=EnvVars(replace={"MY_EXISTING": "new_value"}),
            )
            env = runner._build_subprocess_env()
            assert env["MY_EXISTING"] == "new_value"
        finally:
            del os.environ["MY_EXISTING"]


class TestSubprocessRunnerRenderAdapter:
    """Tests for render_adapter + worker_binding params in SubprocessRunner."""

    def test_constructor_accepts_render_adapter_and_binding(self):
        from easi.core.render_platforms import SimulatorRenderAdapter

        adapter = MagicMock(spec=SimulatorRenderAdapter)
        binding = WorkerBinding(display=":5", cuda_visible_devices="0")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            render_adapter=adapter,
            worker_binding=binding,
        )
        assert runner.render_adapter is adapter
        assert runner.worker_binding is binding

    def test_default_adapter_and_binding_are_none(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
        )
        assert runner.render_adapter is None
        assert runner.worker_binding is None

    def test_build_launch_command_calls_adapter_wrap_when_both_set(self):
        from easi.core.render_platforms import SimulatorRenderAdapter

        adapter = MagicMock(spec=SimulatorRenderAdapter)
        adapter.wrap_command.side_effect = lambda cmd, binding: ["wrapped"] + cmd
        binding = WorkerBinding()
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            render_adapter=adapter,
            worker_binding=binding,
        )
        runner._workspace = Path("/tmp/fake_ws")
        cmd = runner._build_launch_command()
        adapter.wrap_command.assert_called_once()
        assert cmd[0] == "wrapped"

    def test_build_launch_command_skips_adapter_when_adapter_none(self):
        from easi.core.render_platforms import SimulatorRenderAdapter

        adapter = MagicMock(spec=SimulatorRenderAdapter)
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            render_adapter=None,
            worker_binding=WorkerBinding(),
        )
        runner._workspace = Path("/tmp/fake_ws")
        runner._build_launch_command()
        adapter.wrap_command.assert_not_called()

    def test_build_launch_command_skips_adapter_when_binding_none(self):
        from easi.core.render_platforms import SimulatorRenderAdapter

        adapter = MagicMock(spec=SimulatorRenderAdapter)
        adapter.wrap_command.side_effect = lambda cmd, binding: ["wrapped"] + cmd
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            render_adapter=adapter,
            worker_binding=None,
        )
        runner._workspace = Path("/tmp/fake_ws")
        cmd = runner._build_launch_command()
        adapter.wrap_command.assert_not_called()
        assert cmd[0] != "wrapped"

    def test_adapter_receives_correct_binding(self):
        from easi.core.render_platforms import SimulatorRenderAdapter

        received_bindings = []

        class RecordingAdapter(SimulatorRenderAdapter):
            def wrap_command(self, cmd, binding):
                received_bindings.append(binding)
                return cmd

        binding = WorkerBinding(display=":10", cuda_visible_devices="2")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            render_adapter=RecordingAdapter(),
            worker_binding=binding,
        )
        runner._workspace = Path("/tmp/fake_ws")
        runner._build_launch_command()
        assert received_bindings[0] is binding
