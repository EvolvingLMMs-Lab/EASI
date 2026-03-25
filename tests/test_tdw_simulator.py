"""Tests for TDW simulator integration (offline, no TDW needed)."""
import pytest
from pathlib import Path


class TestTDWManifest:
    """Test that manifest.yaml is correctly structured."""

    def test_manifest_exists(self):
        manifest = Path("easi/simulators/tdw/manifest.yaml")
        assert manifest.exists()

    def test_manifest_contents(self):
        import yaml
        with open("easi/simulators/tdw/manifest.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "tdw"
        assert "v1_11_23" in data["versions"]
        assert data["default_version"] == "v1_11_23"

    def test_manifest_classes_importable(self):
        import yaml
        with open("easi/simulators/tdw/manifest.yaml") as f:
            data = yaml.safe_load(f)
        v = data["versions"]["v1_11_23"]
        assert "simulator.TDWSimulator" in v["simulator_class"]
        assert "env_manager.TDWEnvManager" in v["env_manager_class"]


class TestTDWSimulator:
    """Test TDWSimulator class."""

    def test_import(self):
        from easi.simulators.tdw.v1_11_23.simulator import TDWSimulator
        sim = TDWSimulator()
        assert sim.name == "tdw"
        assert sim.version == "v1_11_23"

    def test_bridge_script_path(self):
        from easi.simulators.tdw.v1_11_23.simulator import TDWSimulator
        sim = TDWSimulator()
        bridge_path = sim._get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"


class TestTDWEnvManager:
    """Test TDWEnvManager class."""

    def test_import(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        assert mgr.simulator_name == "tdw"
        assert mgr.version == "v1_11_23"

    def test_env_name(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        assert mgr.get_env_name() == "easi_tdw_v1_11_23"

    def test_default_render_platform(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        assert mgr.default_render_platform == "auto"

    def test_system_deps(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        deps = mgr.get_system_deps()
        assert "conda" in deps
        assert "xvfb" in deps

    def test_conda_env_yaml_exists(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        assert mgr.get_conda_env_yaml_path().exists()

    def test_requirements_txt_exists(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        assert mgr.get_requirements_txt_path().exists()

    def test_validation_import(self):
        from easi.simulators.tdw.v1_11_23.env_manager import TDWEnvManager
        mgr = TDWEnvManager()
        val = mgr.get_validation_import()
        assert "tdw" in val


class TestTDWBridgeSyntax:
    """Test that bridge.py is valid Python (without importing TDW)."""

    def test_bridge_syntax(self):
        import ast
        bridge_path = Path("easi/simulators/tdw/v1_11_23/bridge.py")
        ast.parse(bridge_path.read_text())


class TestTDWRegistry:
    """Test that registry discovers TDW simulator."""

    def test_env_list_includes_tdw(self):
        from easi.simulators.registry import list_simulators
        sims = list_simulators()
        assert "tdw:v1_11_23" in sims

    def test_load_simulator_class(self):
        from easi.simulators.registry import load_simulator_class
        SimClass = load_simulator_class("tdw:v1_11_23")
        sim = SimClass()
        assert sim.name == "tdw"

    def test_create_env_manager(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("tdw:v1_11_23")
        assert mgr.simulator_name == "tdw"
        assert mgr.default_render_platform == "auto"
