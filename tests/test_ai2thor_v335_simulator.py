"""Tests for AI2-THOR v3.3.5 simulator registration and configuration."""
from __future__ import annotations

import pytest


class TestRegistryDiscovery:
    """Verify v3_3_5 is auto-discovered by the simulator registry."""

    def test_explicit_key_registered(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("ai2thor:v3_3_5")
        assert entry.name == "ai2thor"
        assert entry.version == "v3_3_5"

    def test_listed_in_registry(self):
        from easi.simulators.registry import list_simulators

        sims = list_simulators()
        assert "ai2thor:v3_3_5" in sims

    def test_default_is_not_v335(self):
        """v5_0_0 should remain the default, not v3_3_5."""
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("ai2thor")
        assert entry.version == "v5_0_0"

    def test_python_version(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("ai2thor:v3_3_5")
        assert entry.python_version == "3.8"

    def test_description(self):
        from easi.simulators.registry import get_simulator_entry

        entry = get_simulator_entry("ai2thor:v3_3_5")
        assert "arm mode" in entry.description
        assert "manipulathor" in entry.description


class TestEnvManager:
    """Test env_manager properties without actually installing."""

    def test_simulator_name(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.simulator_name == "ai2thor"

    def test_version(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.version == "v3_3_5"

    def test_env_name(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.get_env_name() == "easi_ai2thor_v3_3_5"

    def test_conda_env_yaml_exists(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.get_conda_env_yaml_path().exists()

    def test_requirements_txt_exists(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.get_requirements_txt_path().exists()

    def test_system_deps_include_xvfb(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        deps = mgr.get_system_deps()
        assert "conda" in deps
        assert "xvfb" in deps

    def test_supported_render_platforms(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        platforms = mgr.supported_render_platforms
        assert "auto" in platforms
        assert "xvfb" in platforms
        assert "native" in platforms

    def test_default_render_platform(self):
        from easi.simulators.ai2thor.v3_3_5.env_manager import (
            AI2ThorEnvManagerV335,
        )

        mgr = AI2ThorEnvManagerV335()
        assert mgr.default_render_platform == "auto"


class TestSimulator:
    """Test simulator stub properties."""

    def test_name(self):
        from easi.simulators.ai2thor.v3_3_5.simulator import (
            AI2ThorSimulatorV335,
        )

        sim = AI2ThorSimulatorV335()
        assert sim.name == "ai2thor"

    def test_version(self):
        from easi.simulators.ai2thor.v3_3_5.simulator import (
            AI2ThorSimulatorV335,
        )

        sim = AI2ThorSimulatorV335()
        assert sim.version == "v3_3_5"

    def test_bridge_path_exists(self):
        from easi.simulators.ai2thor.v3_3_5.simulator import (
            AI2ThorSimulatorV335,
        )

        sim = AI2ThorSimulatorV335()
        assert sim._get_bridge_script_path().exists()
