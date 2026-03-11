"""Tests for OmniGibson simulator integration (offline, no simulator needed)."""

import ast
from pathlib import Path

import yaml


class TestOmniGibsonManifest:
    """Test that manifest.yaml is correctly structured."""

    def test_manifest_exists(self):
        manifest = Path("easi/simulators/omnigibson/manifest.yaml")
        assert manifest.exists()

    def test_manifest_contents(self):
        with open("easi/simulators/omnigibson/manifest.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "omnigibson"
        assert "v3_7_2" in data["versions"]
        assert data["default_version"] == "v3_7_2"

    def test_manifest_classes_importable(self):
        with open("easi/simulators/omnigibson/manifest.yaml") as f:
            data = yaml.safe_load(f)
        v = data["versions"]["v3_7_2"]
        assert "simulator.OmniGibsonSimulator" in v["simulator_class"]
        assert "env_manager.OmniGibsonEnvManager" in v["env_manager_class"]

    def test_manifest_installation_kwargs(self):
        with open("easi/simulators/omnigibson/manifest.yaml") as f:
            data = yaml.safe_load(f)
        kwargs = data["versions"]["v3_7_2"]["installation_kwargs"]
        assert kwargs["cuda_version"] == "12.4"
        assert "BEHAVIOR-1K" in kwargs["behavior_1k_repo"]
        assert kwargs["behavior_1k_tag"] == "v3.7.2"

    def test_manifest_uses_render_adapter_without_custom_platforms(self):
        with open("easi/simulators/omnigibson/manifest.yaml") as f:
            data = yaml.safe_load(f)
        version = data["versions"]["v3_7_2"]
        assert (
            version["render_adapter"]
            == "easi.simulators.omnigibson.v3_7_2.render_platforms.OmniGibsonRenderAdapter"
        )
        assert version.get("render_platforms", {}) == {}


class TestOmniGibsonEnvManager:
    """Test OmniGibsonEnvManager class."""

    def test_import(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.simulator_name == "omnigibson"
        assert mgr.version == "v3_7_2"

    def test_env_name(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.get_env_name() == "easi_omnigibson_v3_7_2"

    def test_default_render_platform_native(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.default_render_platform == "native"

    def test_env_vars(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        ev = mgr.get_env_vars()
        assert "OMNIGIBSON_HEADLESS" not in ev.replace
        assert ev.replace["OMNI_KIT_ACCEPT_EULA"] == "YES"
        assert "PYTHONHOME" in ev.replace

    def test_get_python_executable_is_local(self, monkeypatch):
        """Python executable should be a /tmp copy (NFS workaround)."""
        import sys
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        # Mock _get_conda_python to return a real existing binary so the copy
        # can succeed without the conda env being installed.
        monkeypatch.setattr(mgr, "_get_conda_python", lambda: sys.executable)
        python_path = mgr.get_python_executable()
        assert python_path.startswith("/tmp/easi_python_")
        assert Path(python_path).exists()

    def test_get_conda_python_is_conda_env(self):
        """_get_conda_python() returns the real conda env path."""
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        conda_python = mgr._get_conda_python()
        assert "easi_omnigibson_v3_7_2" in conda_python
        assert conda_python.endswith("bin/python")

    def test_system_deps(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        deps = mgr.get_system_deps()
        assert "conda" in deps

    def test_validation_import(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        val = mgr.get_validation_import()
        assert "omnigibson" in val

    def test_conda_env_yaml_exists(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.get_conda_env_yaml_path().exists()

    def test_requirements_txt_exists(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager

        mgr = OmniGibsonEnvManager()
        assert mgr.get_requirements_txt_path().exists()

    def test_isaac_sim_packages_count(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import ISAAC_SIM_PACKAGES

        assert len(ISAAC_SIM_PACKAGES) == 26


class TestOmniGibsonSimulator:
    """Test OmniGibsonSimulator class."""

    def test_import(self):
        from easi.simulators.omnigibson.v3_7_2.simulator import OmniGibsonSimulator

        sim = OmniGibsonSimulator()
        assert sim.name == "omnigibson"
        assert sim.version == "v3_7_2"

    def test_bridge_script_path(self):
        from easi.simulators.omnigibson.v3_7_2.simulator import OmniGibsonSimulator

        sim = OmniGibsonSimulator()
        bridge_path = sim._get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"

    def test_bridge_script_has_bridge_class(self):
        from easi.simulators.omnigibson.v3_7_2.simulator import OmniGibsonSimulator

        content = OmniGibsonSimulator()._get_bridge_script_path().read_text()
        assert "OmniGibsonBridge" in content


class TestOmniGibsonBridgeSyntax:
    """Test that bridge.py is valid Python (without importing OmniGibson)."""

    def test_bridge_syntax(self):
        bridge_path = Path("easi/simulators/omnigibson/v3_7_2/bridge.py")
        ast.parse(bridge_path.read_text())


class TestOmniGibsonRegistry:
    """Test that registry discovers OmniGibson simulator."""

    def test_env_list_includes_omnigibson(self):
        from easi.simulators.registry import list_simulators

        sims = list_simulators()
        assert "omnigibson:v3_7_2" in sims

    def test_load_simulator_class(self):
        from easi.simulators.registry import load_simulator_class

        SimClass = load_simulator_class("omnigibson:v3_7_2")
        sim = SimClass()
        assert sim.name == "omnigibson"

    def test_create_env_manager(self):
        from easi.simulators.registry import create_env_manager

        mgr = create_env_manager("omnigibson:v3_7_2")
        assert mgr.simulator_name == "omnigibson"
        assert mgr.default_render_platform == "native"

    def test_create_env_manager_has_installation_kwargs(self):
        from easi.simulators.registry import create_env_manager

        mgr = create_env_manager("omnigibson")
        assert mgr.installation_kwargs["cuda_version"] == "12.4"
        assert "BEHAVIOR-1K" in mgr.installation_kwargs["behavior_1k_repo"]


class TestOmniGibsonRenderAdapterIntegration:
    def test_resolve_builtin_native_platform(self):
        from easi.core.render_platforms import NativePlatform
        from easi.simulators.registry import resolve_render_platform

        platform = resolve_render_platform("omnigibson:v3_7_2", "native")
        assert isinstance(platform, NativePlatform)
        assert platform.name == "native"

    def test_resolve_builtin_auto_platform(self):
        from easi.core.render_platforms import AutoPlatform
        from easi.simulators.registry import resolve_render_platform

        platform = resolve_render_platform("omnigibson:v3_7_2", "auto")
        assert isinstance(platform, AutoPlatform)
        assert platform.name == "auto"

    def test_adapter_auto_with_display(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":1")
        from easi.core.render_platforms import WorkerBinding
        from easi.simulators.omnigibson.v3_7_2.render_platforms import (
            OmniGibsonRenderAdapter,
        )

        ev = OmniGibsonRenderAdapter().get_env_vars(WorkerBinding())
        assert ev.replace["OMNIGIBSON_HEADLESS"] == "0"

    def test_adapter_auto_without_display(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        from easi.core.render_platforms import WorkerBinding
        from easi.simulators.omnigibson.v3_7_2.render_platforms import (
            OmniGibsonRenderAdapter,
        )

        ev = OmniGibsonRenderAdapter().get_env_vars(WorkerBinding())
        assert ev.replace["OMNIGIBSON_HEADLESS"] == "1"
