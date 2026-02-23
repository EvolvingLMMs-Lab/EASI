"""Tests for render platform abstraction."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


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
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("headless")
        assert p.get_env_vars() == {}

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
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("xvfb")
        assert p.get_env_vars() == {}

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
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("native")
        assert p.get_env_vars() == {}

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
        from easi.core.render_platform import get_render_platform

        p = get_render_platform("egl")
        env = p.get_env_vars()
        assert env["PYOPENGL_PLATFORM"] == "egl"

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

    def test_needs_display_false_for_headless(self):
        mgr = self._make_stub()
        assert mgr.needs_display is False

    def test_needs_display_true_when_platform_not_headless(self):
        mgr = self._make_stub(
            default_render_platform=property(lambda self: "auto"),
        )
        assert mgr.needs_display is True

    def test_xvfb_screen_config_aliases_screen_config(self):
        mgr = self._make_stub(
            screen_config=property(lambda self: "1920x1080x24"),
        )
        assert mgr.xvfb_screen_config == "1920x1080x24"
