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

    def test_needs_display(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager
        mgr = OmniGibsonEnvManager()
        assert mgr.needs_display is False

    def test_env_vars(self):
        from easi.simulators.omnigibson.v3_7_2.env_manager import OmniGibsonEnvManager
        mgr = OmniGibsonEnvManager()
        env_vars = mgr.get_env_vars()
        assert env_vars == {"OMNIGIBSON_HEADLESS": "1"}

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
        assert mgr.needs_display is False

    def test_create_env_manager_has_installation_kwargs(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("omnigibson")
        assert mgr.installation_kwargs["cuda_version"] == "12.4"
        assert "BEHAVIOR-1K" in mgr.installation_kwargs["behavior_1k_repo"]
