"""Tests for CoppeliaSim V4.1.0 simulator integration (offline)."""

from pathlib import Path


class TestRegistryDiscovery:
    """Verify CoppeliaSim is discovered by the simulator registry."""

    def test_coppeliasim_in_simulator_list(self):
        from easi.simulators.registry import list_simulators
        assert "coppeliasim" in list_simulators()

    def test_coppeliasim_versioned_key(self):
        from easi.simulators.registry import list_simulators
        assert "coppeliasim:v4_1_0" in list_simulators()

    def test_get_simulator_entry(self):
        from easi.simulators.registry import get_simulator_entry
        entry = get_simulator_entry("coppeliasim")
        assert entry.name == "coppeliasim"
        assert entry.version == "v4_1_0"

    def test_installation_kwargs_populated(self):
        from easi.simulators.registry import get_simulator_entry
        entry = get_simulator_entry("coppeliasim")
        assert "binary_url" in entry.installation_kwargs
        assert "binary_dir_name" in entry.installation_kwargs
        assert "pyrep_git_url" in entry.installation_kwargs

    def test_load_simulator_class(self):
        from easi.simulators.registry import load_simulator_class
        cls = load_simulator_class("coppeliasim:v4_1_0")
        sim = cls()
        assert sim.name == "coppeliasim"

    def test_create_env_manager_with_kwargs(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.installation_kwargs["binary_dir_name"] == "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"


class TestEnvManagerConfig:
    """Verify env manager configuration without installing anything."""

    def test_env_name(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.get_env_name() == "easi_coppeliasim_v4_1_0"

    def test_needs_display(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.needs_display is True

    def test_xvfb_screen_config(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.xvfb_screen_config == "1280x720x24"

    def test_system_deps(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        deps = mgr.get_system_deps()
        assert "conda" in deps
        assert "xvfb" in deps

    def test_validation_import_references_pyrep(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert "pyrep" in mgr.get_validation_import().lower()

    def test_conda_yaml_exists(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.get_conda_env_yaml_path().exists()

    def test_requirements_txt_exists(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert mgr.get_requirements_txt_path().exists()


class TestEnvVars:
    """Verify env_vars are correctly resolved from installation_kwargs."""

    def test_env_vars_has_coppeliasim_root(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert "COPPELIASIM_ROOT" in mgr.get_env_vars()

    def test_env_vars_has_ld_library_path(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert "LD_LIBRARY_PATH" in mgr.get_env_vars()

    def test_env_vars_has_qt_platform(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" in mgr.get_env_vars()

    def test_env_vars_no_unresolved_templates(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        for key, val in mgr.get_env_vars().items():
            assert "{" not in val, f"Unresolved template in {key}: {val}"

    def test_env_vars_contain_coppeliasim_dir(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        root = mgr.get_env_vars()["COPPELIASIM_ROOT"]
        assert "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04" in root
        assert "extras" in root

    def test_ld_library_path_includes_coppeliasim_root(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        env_vars = mgr.get_env_vars()
        assert env_vars["COPPELIASIM_ROOT"] in env_vars["LD_LIBRARY_PATH"]
        assert env_vars["COPPELIASIM_ROOT"] == env_vars["QT_QPA_PLATFORM_PLUGIN_PATH"]

    def test_ld_library_path_includes_conda_lib(self):
        from easi.simulators.registry import create_env_manager
        mgr = create_env_manager("coppeliasim")
        env_vars = mgr.get_env_vars()
        assert "/lib" in env_vars["LD_LIBRARY_PATH"]

    def test_empty_installation_kwargs_returns_empty_env_vars(self):
        from easi.simulators.coppeliasim.v4_1_0.env_manager import CoppeliaSimEnvManagerV410
        mgr = CoppeliaSimEnvManagerV410()  # No installation_kwargs
        assert mgr.get_env_vars() == {}


class TestSimulatorClass:
    """Verify simulator class basic properties."""

    def test_name(self):
        from easi.simulators.coppeliasim.v4_1_0.simulator import CoppeliaSimSimulatorV410
        assert CoppeliaSimSimulatorV410().name == "coppeliasim"

    def test_version(self):
        from easi.simulators.coppeliasim.v4_1_0.simulator import CoppeliaSimSimulatorV410
        assert CoppeliaSimSimulatorV410().version == "v4_1_0"

    def test_bridge_script_path_exists(self):
        from easi.simulators.coppeliasim.v4_1_0.simulator import CoppeliaSimSimulatorV410
        bridge_path = CoppeliaSimSimulatorV410()._get_bridge_script_path()
        assert bridge_path.exists()
        assert bridge_path.name == "bridge.py"

    def test_bridge_script_has_bridge_class(self):
        from easi.simulators.coppeliasim.v4_1_0.simulator import CoppeliaSimSimulatorV410
        content = CoppeliaSimSimulatorV410()._get_bridge_script_path().read_text()
        assert "CoppeliaSimBridge" in content


class TestBridgeFileStructure:
    """Verify bridge file and supporting assets exist."""

    def test_ttt_scene_file_exists(self):
        ttt = Path(__file__).parent.parent / "easi" / "simulators" / "coppeliasim" / "v4_1_0" / "task_design.ttt"
        assert ttt.exists(), f"task_design.ttt not found at {ttt}"

    def test_lua_addon_exists(self):
        lua = Path(__file__).parent.parent / "easi" / "simulators" / "coppeliasim" / "v4_1_0" / "simAddOnScript_PyRep.lua"
        assert lua.exists(), f"simAddOnScript_PyRep.lua not found at {lua}"
