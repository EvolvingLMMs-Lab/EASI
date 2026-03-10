"""Tests for Xorg render platform and manager."""

from __future__ import annotations


class TestXorgPlatform:
    """Test XorgPlatform env vars and command wrapping."""

    def test_name(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        assert p.name == "xorg"

    def test_env_vars(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        ev = p.get_env_vars()
        assert ev.replace["DISPLAY"] == ":10"
        assert ev.replace["CUDA_VISIBLE_DEVICES"] == "4"
        assert ev.replace["EASI_GPU_DISPLAY"] == "1"

    def test_wrap_command_passthrough(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        cmd = ["python", "bridge.py", "--workspace", "/tmp"]
        assert p.wrap_command(cmd, "1024x768x24") == cmd

    def test_is_available(self):
        from easi.core.xorg_platform import XorgPlatform

        p = XorgPlatform(display_num=10, gpu_id=4)
        assert p.is_available() is True
