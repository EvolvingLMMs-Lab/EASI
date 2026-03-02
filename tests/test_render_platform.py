"""Tests for render platform abstraction."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEnvVarsDataclass:
    """Test the EnvVars dataclass."""

    def test_empty_is_falsy(self):
        from easi.core.render_platform import EnvVars

        assert not EnvVars()

    def test_replace_is_truthy(self):
        from easi.core.render_platform import EnvVars

        assert EnvVars(replace={"FOO": "bar"})

    def test_prepend_is_truthy(self):
        from easi.core.render_platform import EnvVars

        assert EnvVars(prepend={"PATH": "/extra"})

    def test_to_flat_dict(self):
        from easi.core.render_platform import EnvVars

        ev = EnvVars(replace={"A": "1"}, prepend={"B": "2"})
        assert ev.to_flat_dict() == {"A": "1", "B": "2"}

    def test_apply_to_env_replace(self):
        from easi.core.render_platform import EnvVars

        ev = EnvVars(replace={"FOO": "new"})
        result = ev.apply_to_env({"FOO": "old", "BAR": "keep"})
        assert result["FOO"] == "new"
        assert result["BAR"] == "keep"

    def test_apply_to_env_prepend_existing(self):
        from easi.core.render_platform import EnvVars

        ev = EnvVars(prepend={"PATH": "/new"})
        result = ev.apply_to_env({"PATH": "/old"})
        assert result["PATH"] == "/new:/old"

    def test_apply_to_env_prepend_missing(self):
        from easi.core.render_platform import EnvVars

        ev = EnvVars(prepend={"LD_LIBRARY_PATH": "/lib"})
        result = ev.apply_to_env({})
        assert result["LD_LIBRARY_PATH"] == "/lib"

    def test_merge_replace_later_wins(self):
        from easi.core.render_platform import EnvVars

        a = EnvVars(replace={"K": "a"})
        b = EnvVars(replace={"K": "b"})
        merged = EnvVars.merge(a, b)
        assert merged.replace["K"] == "b"

    def test_merge_prepend_concatenates(self):
        from easi.core.render_platform import EnvVars

        a = EnvVars(prepend={"PATH": "/a"})
        b = EnvVars(prepend={"PATH": "/b"})
        merged = EnvVars.merge(a, b)
        assert merged.prepend["PATH"] == "/b:/a"

    def test_merge_mixed(self):
        from easi.core.render_platform import EnvVars

        a = EnvVars(replace={"ROOT": "/opt"}, prepend={"PATH": "/a/bin"})
        b = EnvVars(replace={"HOME": "/home"}, prepend={"PATH": "/b/bin"})
        merged = EnvVars.merge(a, b)
        assert merged.replace == {"ROOT": "/opt", "HOME": "/home"}
        assert merged.prepend["PATH"] == "/b/bin:/a/bin"


class TestRenderPlatformRegistry:
    """Test platform discovery and instantiation."""

    def test_get_platform_returns_auto(self):
        from easi.core.render_platform import get_render_platform

        platform = get_render_platform("auto")
        assert platform.name == "auto"

    def test_get_platform_returns_xvfb(self):
        from easi.core.render_platform import get_render_platform

        platform = get_render_platform("xvfb")
        assert platform.name == "xvfb"

    def test_get_platform_returns_native(self):
        from easi.core.render_platform import get_render_platform

        platform = get_render_platform("native")
        assert platform.name == "native"

    def test_get_platform_returns_egl(self):
        from easi.core.render_platform import get_render_platform

        platform = get_render_platform("egl")
        assert platform.name == "egl"

    def test_get_platform_returns_headless(self):
        from easi.core.render_platform import get_render_platform

        platform = get_render_platform("headless")
        assert platform.name == "headless"

    def test_get_platform_unknown_raises(self):
        from easi.core.render_platform import get_render_platform

        with pytest.raises(ValueError, match="Unknown render platform"):
            get_render_platform("nonexistent")

    def test_available_platforms_returns_names(self):
        from easi.core.render_platform import available_platforms

        names = available_platforms()
        assert set(names) >= {"auto", "native", "xvfb", "egl", "headless"}


class TestHeadlessPlatform:
    """Headless: no wrapping, no env vars."""

    def test_wrap_command_passthrough(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("headless")
        cmd = ["python", "bridge.py"]
        assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_get_env_vars_empty(self):
        from easi.core.render_platform import EnvVars, get_render_platform

        p = get_render_platform("headless")
        ev = p.get_env_vars()
        assert isinstance(ev, EnvVars)
        assert not ev

    def test_get_system_deps_empty(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("headless")
        assert p.get_system_deps() == []


class TestXvfbPlatform:
    """Xvfb: always wraps with xvfb-run."""

    def test_wrap_command_prepends_xvfb_run(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("xvfb")
        cmd = ["python", "bridge.py"]
        wrapped = p.wrap_command(cmd, "1280x720x24")
        assert wrapped[:2] == ["xvfb-run", "-a"]
        assert "-screen 0 1280x720x24" in wrapped[3]
        assert wrapped[-2:] == cmd

    def test_get_env_vars_empty(self):
        from easi.core.render_platform import EnvVars, get_render_platform

        p = get_render_platform("xvfb")
        ev = p.get_env_vars()
        assert isinstance(ev, EnvVars)
        assert not ev

    def test_get_system_deps_includes_xvfb(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("xvfb")
        assert "xvfb" in p.get_system_deps()


class TestNativePlatform:
    """Native: passthrough, requires DISPLAY."""

    def test_wrap_command_passthrough(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("native")
        cmd = ["python", "bridge.py"]
        assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_get_env_vars_empty(self):
        from easi.core.render_platform import EnvVars, get_render_platform

        p = get_render_platform("native")
        ev = p.get_env_vars()
        assert isinstance(ev, EnvVars)
        assert not ev

    def test_is_available_true_when_display_set(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("native")
        with patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert p.is_available() is True

    def test_is_available_false_when_no_display(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("native")
        with patch.dict(os.environ, {}, clear=True):
            assert p.is_available() is False


class TestEGLPlatform:
    """EGL: no wrapping, sets PYOPENGL_PLATFORM."""

    def test_wrap_command_passthrough(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("egl")
        cmd = ["python", "bridge.py"]
        assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_get_env_vars_sets_pyopengl(self):
        from easi.core.render_platform import EnvVars, get_render_platform

        p = get_render_platform("egl")
        ev = p.get_env_vars()
        assert isinstance(ev, EnvVars)
        assert ev.replace["PYOPENGL_PLATFORM"] == "egl"

    def test_get_system_deps_includes_egl(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("egl")
        assert "egl" in p.get_system_deps()


class TestAutoPlatform:
    """Auto: native if DISPLAY exists, xvfb fallback otherwise."""

    def test_wrap_command_uses_native_when_display_set(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("auto")
        cmd = ["python", "bridge.py"]
        with patch.dict(os.environ, {"DISPLAY": ":0"}):
            assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_wrap_command_uses_xvfb_when_no_display(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("auto")
        cmd = ["python", "bridge.py"]
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True):
            wrapped = p.wrap_command(cmd, "1024x768x24")
            assert wrapped[0] == "xvfb-run"

    def test_get_system_deps_empty(self):
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("auto")
        assert p.get_system_deps() == []


class TestBaseEnvManagerRenderPlatform:
    """Verify env_manager exposes render platform config."""

    def _make_stub(self, **overrides):
        """Create a minimal concrete BaseEnvironmentManager subclass."""
        from easi.core.base_env_manager import BaseEnvironmentManager

        attrs = {
            "simulator_name": property(lambda self: "stub"),
            "version": property(lambda self: "v0"),
            "get_conda_env_yaml_path": lambda self: Path("/fake/conda.yaml"),
            "get_requirements_txt_path": lambda self: Path("/fake/req.txt"),
            "get_system_deps": lambda self: [],
            "get_validation_import": lambda self: "import sys",
        }
        attrs.update(overrides)
        Stub = type("Stub", (BaseEnvironmentManager,), attrs)
        return Stub()

    def test_default_render_platform_is_headless(self):
        mgr = self._make_stub()
        assert mgr.default_render_platform == "headless"

    def test_supported_render_platforms_default(self):
        mgr = self._make_stub()
        assert "headless" in mgr.supported_render_platforms

    def test_screen_config_default(self):
        mgr = self._make_stub()
        assert mgr.screen_config == "1024x768x24"

    def test_get_env_vars_returns_envvars(self):
        from easi.core.render_platform import EnvVars

        mgr = self._make_stub()
        ev = mgr.get_env_vars()
        assert isinstance(ev, EnvVars)
        assert not ev


class TestSubprocessRunnerRenderPlatform:
    """Verify SubprocessRunner uses RenderPlatform for command wrapping."""

    def test_accepts_render_platform_param(self):
        from easi.core.render_platform import get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        p = get_render_platform("headless")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=p,
        )
        assert runner.render_platform.name == "headless"

    def test_build_command_uses_platform_wrap(self):
        from easi.core.render_platform import get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        p = get_render_platform("xvfb")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=p,
            screen_config="1280x720x24",
        )
        runner._workspace = Path("/tmp/fake_ws")
        cmd = runner._build_launch_command()
        assert cmd[0] == "xvfb-run"
        assert "1280x720x24" in cmd[3]

    def test_build_command_headless_no_wrap(self):
        from easi.core.render_platform import get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        p = get_render_platform("headless")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=p,
        )
        runner._workspace = Path("/tmp/fake_ws")
        cmd = runner._build_launch_command()
        assert cmd[0] == "/usr/bin/python3"

    def test_platform_env_vars_merged(self):
        from easi.core.render_platform import EnvVars, get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        p = get_render_platform("egl")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=p,
            extra_env=EnvVars(replace={"SIM_ROOT": "/opt/sim"}),
        )
        env = runner._build_subprocess_env()
        assert env["PYOPENGL_PLATFORM"] == "egl"
        assert env["SIM_ROOT"] == "/opt/sim"

    def test_no_env_vars_returns_none(self):
        from easi.core.render_platform import get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        p = get_render_platform("headless")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=p,
        )
        assert runner._build_subprocess_env() is None


class TestSimulatorRenderPlatforms:
    """Each simulator declares render platform preferences."""

    def test_ai2thor_v2(self):
        from easi.simulators.ai2thor.v2_1_0.env_manager import AI2ThorEnvManagerV210

        mgr = AI2ThorEnvManagerV210()
        assert mgr.default_render_platform == "auto"
        assert "xvfb" in mgr.supported_render_platforms
        assert "native" in mgr.supported_render_platforms

    def test_ai2thor_v5(self):
        from easi.simulators.ai2thor.v5_0_0.env_manager import AI2ThorEnvManagerV500

        mgr = AI2ThorEnvManagerV500()
        assert mgr.default_render_platform == "auto"
        assert mgr.screen_config == "1280x720x24"

    def test_habitat(self):
        from easi.simulators.habitat_sim.v0_3_0.env_manager import HabitatEnvManagerV030

        mgr = HabitatEnvManagerV030()
        assert mgr.default_render_platform == "auto"
        assert "egl" in mgr.supported_render_platforms

    def test_coppeliasim(self):
        from easi.simulators.coppeliasim.v4_1_0.env_manager import CoppeliaSimEnvManagerV410

        mgr = CoppeliaSimEnvManagerV410()
        assert mgr.default_render_platform == "auto"
        assert mgr.screen_config == "1280x720x24"

    def test_coppeliasim_env_vars_are_platform_agnostic(self):
        """CoppeliaSim env_manager should NOT set QT_QPA_PLATFORM_PLUGIN_PATH or __EGL_VENDOR_LIBRARY_FILENAMES.

        Those platform-specific vars are now handled by custom render platform classes.
        """
        from easi.simulators.coppeliasim.v4_1_0.env_manager import CoppeliaSimEnvManagerV410

        mgr = CoppeliaSimEnvManagerV410(installation_kwargs={"binary_dir_name": "CoppeliaSim"})
        ev = mgr.get_env_vars()
        all_keys = set(ev.replace) | set(ev.prepend)
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in all_keys
        assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in all_keys
        # Should still have the platform-agnostic vars
        assert "COPPELIASIM_ROOT" in ev.replace
        assert "LD_LIBRARY_PATH" in ev.prepend

    def test_tdw(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager

        mgr = TDWEnvManager()
        assert mgr.default_render_platform == "auto"

    def test_omnigibson(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.default_render_platform == "native"

    def test_dummy(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        assert mgr.default_render_platform == "headless"


class TestCustomRenderPlatforms:
    """Test custom render platform registration and resolution."""

    def test_simulator_entry_has_empty_render_platforms(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("ai2thor:v2_1_0")
        assert entry.render_platforms == {}

    def test_simulator_entry_has_custom_render_platforms(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("dummy")
        assert entry.render_platforms == {
            "dummy_custom": "easi.simulators.dummy.v1.render_platforms.DummyCustomPlatform"
        }

    def test_resolve_falls_back_to_builtin(self):
        from easi.core.render_platform import HeadlessPlatform
        from easi.simulators.registry import resolve_render_platform

        platform = resolve_render_platform("dummy", "headless")
        assert isinstance(platform, HeadlessPlatform)
        assert platform.name == "headless"

    def test_resolve_uses_custom_class(self):
        from easi.simulators.dummy.v1.render_platforms import DummyCustomPlatform
        from easi.simulators.registry import resolve_render_platform

        platform = resolve_render_platform("dummy", "dummy_custom")
        assert isinstance(platform, DummyCustomPlatform)
        assert platform.name == "dummy_custom"
        ev = platform.get_env_vars()
        assert ev.replace == {"DUMMY_CUSTOM_PLATFORM": "1"}

    def test_resolve_name_mismatch_raises(self):
        from easi.simulators.registry import resolve_render_platform

        # Patch the dummy entry to map "wrong_name" to the DummyCustomPlatform class
        # (which has name "dummy_custom", not "wrong_name")
        with patch("easi.simulators.registry._get_registry") as mock_reg:
            from easi.simulators.registry import SimulatorEntry

            fake_entry = SimulatorEntry(
                name="fake", version="v1", description="",
                simulator_class="", env_manager_class="",
                python_version="3.10",
                render_platforms={
                    "wrong_name": "easi.simulators.dummy.v1.render_platforms.DummyCustomPlatform"
                },
            )
            mock_reg.return_value = {"fake:v1": fake_entry}
            with pytest.raises(ValueError, match="expected 'wrong_name'"):
                resolve_render_platform("fake:v1", "wrong_name")

    def test_custom_shadows_builtin(self):
        """A custom 'headless' class should be returned instead of the built-in."""
        from easi.core.render_platform import HeadlessPlatform
        from easi.simulators.registry import resolve_render_platform

        with patch("easi.simulators.registry._get_registry") as mock_reg:
            from easi.simulators.registry import SimulatorEntry

            fake_entry = SimulatorEntry(
                name="fake", version="v1", description="",
                simulator_class="", env_manager_class="",
                python_version="3.10",
                render_platforms={
                    "headless": "easi.simulators.dummy.v1.render_platforms.DummyCustomPlatform"
                },
            )
            mock_reg.return_value = {"fake:v1": fake_entry}
            # DummyCustomPlatform.name is "dummy_custom", not "headless",
            # so this should raise a name mismatch error.
            # To truly shadow, we need a class whose name IS "headless".
            # Let's test the lookup path: if a custom class is found, it's used.
            # We'll patch _import_class to return a class with name "headless".
            from easi.core.render_platform import EnvVars

            class CustomHeadless(HeadlessPlatform):
                def get_env_vars(self):
                    return EnvVars(replace={"CUSTOM": "1"})

            with patch("easi.simulators.registry._import_class", return_value=CustomHeadless):
                platform = resolve_render_platform("fake:v1", "headless")
                assert isinstance(platform, CustomHeadless)
                assert platform.name == "headless"
                assert platform.get_env_vars().replace == {"CUSTOM": "1"}


class TestCoppeliaSimCustomPlatforms:
    """Test CoppeliaSim-specific custom render platform classes."""

    def _make_mock_env_manager(self, binary_dir_name="CoppeliaSim"):
        from unittest.mock import MagicMock

        mgr = MagicMock()
        mgr.installation_kwargs = {"binary_dir_name": binary_dir_name}
        mgr._get_template_variables.return_value = {
            "env_dir": "/fake/envs/easi_coppeliasim_v4_1_0",
            "extras_dir": "/fake/envs/easi_coppeliasim_v4_1_0/extras",
        }
        mgr._resolve_template.side_effect = lambda tmpl, t: tmpl.replace(
            "{extras_dir}", t["extras_dir"]
        ).replace("{env_dir}", t["env_dir"])
        return mgr

    def test_native_has_qt_plugin_path_and_headless_false(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimNativePlatform,
        )

        mgr = self._make_mock_env_manager()
        p = CoppeliaSimNativePlatform(env_manager=mgr)
        assert p.name == "native"
        ev = p.get_env_vars()
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" in ev.prepend
        assert ev.replace["COPPELIASIM_HEADLESS"] == "0"
        assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in ev.replace

    def test_xvfb_has_qt_plugin_path_and_headless_true(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimXvfbPlatform,
        )

        mgr = self._make_mock_env_manager()
        p = CoppeliaSimXvfbPlatform(env_manager=mgr)
        assert p.name == "xvfb"
        ev = p.get_env_vars()
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" in ev.prepend
        assert "CoppeliaSim" in ev.prepend["QT_QPA_PLATFORM_PLUGIN_PATH"]
        assert ev.replace["COPPELIASIM_HEADLESS"] == "1"

    def test_xvfb_wraps_command(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimXvfbPlatform,
        )

        p = CoppeliaSimXvfbPlatform()
        cmd = ["python", "bridge.py"]
        wrapped = p.wrap_command(cmd, "1280x720x24")
        assert wrapped[0] == "xvfb-run"

    def test_auto_native_when_display(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimAutoPlatform,
        )

        mgr = self._make_mock_env_manager()
        p = CoppeliaSimAutoPlatform(env_manager=mgr)
        assert p.name == "auto"
        with patch.dict(os.environ, {"DISPLAY": ":0"}):
            ev = p.get_env_vars()
            assert "QT_QPA_PLATFORM_PLUGIN_PATH" in ev.prepend
            assert ev.replace["COPPELIASIM_HEADLESS"] == "0"

    def test_auto_xvfb_when_no_display(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimAutoPlatform,
        )

        mgr = self._make_mock_env_manager()
        p = CoppeliaSimAutoPlatform(env_manager=mgr)
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True):
            ev = p.get_env_vars()
            assert "QT_QPA_PLATFORM_PLUGIN_PATH" in ev.prepend
            assert ev.replace["COPPELIASIM_HEADLESS"] == "1"

    def test_xvfb_no_env_manager_returns_empty(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimXvfbPlatform,
        )

        p = CoppeliaSimXvfbPlatform()  # no env_manager
        ev = p.get_env_vars()
        assert not ev

    def test_manifest_registers_custom_platforms(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("coppeliasim:v4_1_0")
        assert "auto" in entry.render_platforms
        assert "native" in entry.render_platforms
        assert "xvfb" in entry.render_platforms

    def test_resolve_coppeliasim_custom_platform(self):
        from easi.simulators.coppeliasim.v4_1_0.render_platforms import (
            CoppeliaSimAutoPlatform,
        )
        from easi.simulators.registry import resolve_render_platform

        platform = resolve_render_platform("coppeliasim:v4_1_0", "auto")
        assert isinstance(platform, CoppeliaSimAutoPlatform)
        assert platform.name == "auto"


from unittest.mock import MagicMock


class TestCLIRenderPlatform:
    """Verify --render-platform is accepted by CLI parser."""

    def test_start_parser_accepts_render_platform(self):
        from easi.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--render-platform", "egl"])
        assert args.render_platform == "egl"

    def test_start_parser_default_is_none(self):
        from easi.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.render_platform is None

    def test_sim_test_parser_accepts_render_platform(self):
        from easi.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["sim", "test", "dummy", "--render-platform", "xvfb"])
        assert args.render_platform == "xvfb"


class TestRunnerRenderPlatformWiring:
    """Verify EvaluationRunner resolves and passes render platform."""

    def _make_mock_env_mgr(self):
        from easi.core.render_platform import EnvVars

        mgr = MagicMock()
        mgr.env_is_ready.return_value = True
        mgr.get_python_executable.return_value = "/usr/bin/python3"
        mgr.default_render_platform = "auto"
        mgr.supported_render_platforms = ["auto", "xvfb", "native", "egl"]
        mgr.screen_config = "1024x768x24"
        mgr.get_env_vars.return_value = EnvVars()
        return mgr

    def _make_mock_task(self, render_platform=None):
        task = MagicMock()
        task.additional_deps = []
        task.simulator_kwargs = {}
        task.extra_env_vars = {}
        task.simulator_configs = {}
        if render_platform:
            task.simulator_configs = {"render_platform": render_platform}
        task.get_bridge_script_path.return_value = None
        return task

    def test_default_uses_env_manager_platform(self):
        from easi.core.render_platform import get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = None

        mock_env_mgr = self._make_mock_env_mgr()
        mock_sim_cls = MagicMock()
        mock_sim_cls.return_value._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            rp = MockRunner.call_args.kwargs.get("render_platform")
            assert rp.name == "auto"

    def test_cli_override_wins(self):
        from easi.core.render_platform import get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = "xvfb"

        mock_env_mgr = self._make_mock_env_mgr()
        mock_sim_cls = MagicMock()
        mock_sim_cls.return_value._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1")
            rp = MockRunner.call_args.kwargs.get("render_platform")
            assert rp.name == "xvfb"

    def test_yaml_override_used_when_no_cli(self):
        from easi.core.render_platform import get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = None

        mock_env_mgr = self._make_mock_env_mgr()
        mock_task = self._make_mock_task(render_platform="egl")
        mock_sim_cls = MagicMock()
        mock_sim_cls.return_value._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1", task=mock_task)
            rp = MockRunner.call_args.kwargs.get("render_platform")
            assert rp.name == "egl"

    def test_cli_beats_yaml(self):
        from easi.core.render_platform import get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = "xvfb"

        mock_env_mgr = self._make_mock_env_mgr()
        mock_task = self._make_mock_task(render_platform="egl")
        mock_sim_cls = MagicMock()
        mock_sim_cls.return_value._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             patch("easi.simulators.subprocess_runner.SubprocessRunner") as MockRunner:
            MockRunner.return_value.launch.return_value = None
            runner._create_simulator("fake:v1", task=mock_task)
            rp = MockRunner.call_args.kwargs.get("render_platform")
            assert rp.name == "xvfb"

    def test_unsupported_platform_raises(self):
        from easi.core.render_platform import get_render_platform
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.data_dir = Path("/tmp/fake")
        runner.render_platform_name = "egl"

        mock_env_mgr = self._make_mock_env_mgr()
        mock_env_mgr.supported_render_platforms = ["auto", "xvfb"]  # no egl
        mock_sim_cls = MagicMock()
        mock_sim_cls.return_value._get_bridge_script_path.return_value = Path("/fake/bridge.py")

        mock_entry = MagicMock()
        mock_entry.runtime = "conda"

        with patch("easi.simulators.registry.get_simulator_entry", return_value=mock_entry), \
             patch("easi.simulators.registry.create_env_manager", return_value=mock_env_mgr), \
             patch("easi.simulators.registry.load_simulator_class", return_value=mock_sim_cls), \
             patch("easi.simulators.registry.resolve_render_platform",
                   side_effect=lambda key, name, env_manager=None: get_render_platform(name)), \
             pytest.raises(ValueError, match="not supported"):
            runner._create_simulator("fake:v1")


class TestRenderPlatformEndToEnd:
    """End-to-end: platform flows through runner to subprocess."""

    def test_egl_env_vars_in_subprocess(self):
        from easi.core.render_platform import EnvVars, get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        platform = get_render_platform("egl")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=platform,
            extra_env=EnvVars(replace={"SIM_ROOT": "/opt/sim"}),
        )
        env = runner._build_subprocess_env()
        assert env["PYOPENGL_PLATFORM"] == "egl"
        assert env["SIM_ROOT"] == "/opt/sim"

    def test_auto_wraps_xvfb_when_no_display(self):
        from easi.core.render_platform import get_render_platform
        from easi.simulators.subprocess_runner import SubprocessRunner

        platform = get_render_platform("auto")
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/fake/bridge.py"),
            render_platform=platform,
            screen_config="1280x720x24",
        )
        runner._workspace = Path("/tmp/fake_ws")

        env = os.environ.copy()
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True):
            cmd = runner._build_launch_command()
            assert cmd[0] == "xvfb-run"
            assert "1280x720x24" in cmd[3]

    def test_render_platform_saved_in_config(self):
        """render_platform should be captured in _cli_options for config.json."""
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            task_name="dummy_task",
            render_platform="egl",
        )
        assert runner._cli_options["render_platform"] == "egl"
