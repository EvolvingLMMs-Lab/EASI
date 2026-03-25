"""Tests for BaseEnvironmentManager extensions (env vars, post_install, helpers)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


class TestEnvManagerGetEnvVars:
    """Tests for the get_env_vars() method."""

    def test_default_returns_empty_envvars(self):
        from easi.core.render_platforms import EnvVars
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        result = mgr.get_env_vars()
        assert isinstance(result, EnvVars)
        assert not result

    def test_returns_envvars_type(self):
        from easi.core.render_platforms import EnvVars
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        result = mgr.get_env_vars()
        assert isinstance(result, EnvVars)


class TestEnvManagerPostInstall:
    """Tests for the post_install() hook and helper utilities."""

    def test_default_post_install_is_noop(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        ctx = mgr._get_template_variables()
        ctx["env_vars"] = mgr.get_env_vars().to_flat_dict()
        mgr.post_install(ctx)  # Should not raise

    def test_post_install_receives_context_keys(self):
        from easi.core.base_env_manager import BaseEnvironmentManager

        received_ctx = {}

        class SpyManager(BaseEnvironmentManager):
            simulator_name = "spy"
            version = "v1"

            def get_conda_env_yaml_path(self):
                return Path("/dev/null")

            def get_requirements_txt_path(self):
                return Path("/dev/null")

            def get_system_deps(self):
                return []

            def get_validation_import(self):
                return "pass"

            def post_install(self, context):
                received_ctx.update(context)

        mgr = SpyManager()
        mgr._run_post_install()
        assert "env_dir" in received_ctx
        assert "extras_dir" in received_ctx
        assert "env_vars" in received_ctx

    def test_extras_dir_inside_env(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        extras_dir = mgr.get_extras_dir()
        env_name = mgr.get_env_name()
        assert env_name in str(extras_dir)
        assert str(extras_dir).endswith("extras")

    def test_resolve_template(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        env_dir = str(mgr.conda_prefix / "envs" / mgr.get_env_name())
        result = mgr._resolve_template("{env_dir}/lib", {"env_dir": env_dir})
        assert result == f"{env_dir}/lib"

    def test_resolve_extras_dir_template(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        t = mgr._get_template_variables()
        result = mgr._resolve_template("{extras_dir}/CoppeliaSim", t)
        assert "extras/CoppeliaSim" in result
        assert "{" not in result

    def test_post_install_can_copy_files(self, tmp_path):
        from easi.core.base_env_manager import BaseEnvironmentManager

        src_file = tmp_path / "source" / "addon.lua"
        src_file.parent.mkdir()
        src_file.write_text("-- lua script")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        class CopyManager(BaseEnvironmentManager):
            simulator_name = "copy"
            version = "v1"

            def get_conda_env_yaml_path(self):
                return Path("/dev/null")

            def get_requirements_txt_path(self):
                return Path("/dev/null")

            def get_system_deps(self):
                return []

            def get_validation_import(self):
                return "pass"

            def post_install(self, context):
                shutil.copy(str(src_file), str(dest_dir / "addon.lua"))

        mgr = CopyManager()
        ctx = mgr._get_template_variables()
        ctx["env_vars"] = {}
        mgr.post_install(ctx)
        assert (dest_dir / "addon.lua").exists()
        assert (dest_dir / "addon.lua").read_text() == "-- lua script"

    def test_env_vars_with_templates_are_resolved(self):
        from easi.core.base_env_manager import BaseEnvironmentManager

        class FakeManager(BaseEnvironmentManager):
            simulator_name = "fake"
            version = "v1"

            def get_conda_env_yaml_path(self):
                return Path("/dev/null")

            def get_requirements_txt_path(self):
                return Path("/dev/null")

            def get_system_deps(self):
                return []

            def get_validation_import(self):
                return "pass"

            def get_env_vars(self):
                t = self._get_template_variables()
                return {
                    "SIM_ROOT": self._resolve_template("{extras_dir}/CoppeliaSim", t),
                    "LD_LIBRARY_PATH": self._resolve_template("{extras_dir}/CoppeliaSim", t),
                }

        mgr = FakeManager()
        env_vars = mgr.get_env_vars()
        assert "extras/CoppeliaSim" in env_vars["SIM_ROOT"]
        assert "extras/CoppeliaSim" in env_vars["LD_LIBRARY_PATH"]
        for v in env_vars.values():
            assert "{" not in v

    def test_run_command_accepts_env(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        mgr._run_command(
            [sys.executable, "-c", "import os; assert os.environ.get('TEST_EASI_VAR') == 'hello'"],
            "test env injection",
            env={**os.environ, "TEST_EASI_VAR": "hello"},
        )

    def test_run_command_without_env_inherits_parent(self):
        from easi.simulators.dummy.v1.env_manager import DummyEnvManager

        mgr = DummyEnvManager()
        mgr._run_command(
            [sys.executable, "-c", "print('ok')"],
            "test no env",
        )
