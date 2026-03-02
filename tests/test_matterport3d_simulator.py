"""Tests for Matterport3D simulator integration (Docker-based)."""

from __future__ import annotations

from pathlib import Path


class TestMatterport3DManifest:
    """Manifest discovery and fields."""

    def test_manifest_exists(self):
        manifest = Path("easi/simulators/matterport3d/manifest.yaml")
        assert manifest.exists()

    def test_manifest_contents(self):
        import yaml
        manifest = Path("easi/simulators/matterport3d/manifest.yaml")
        data = yaml.safe_load(manifest.read_text())
        assert data["name"] == "matterport3d"
        assert "v0_1" in data["versions"]
        assert data["versions"]["v0_1"]["runtime"] == "docker"

    def test_manifest_classes_importable(self):
        from easi.utils.import_utils import import_class
        import yaml
        manifest = Path("easi/simulators/matterport3d/manifest.yaml")
        data = yaml.safe_load(manifest.read_text())
        ver = data["versions"]["v0_1"]
        sim_cls = import_class(ver["simulator_class"])
        mgr_cls = import_class(ver["env_manager_class"])
        assert sim_cls is not None
        assert mgr_cls is not None


class TestMatterport3DEnvManager:
    """Environment manager properties."""

    def test_import(self):
        from easi.simulators.matterport3d.v0_1.env_manager import Matterport3DEnvManager
        mgr = Matterport3DEnvManager()
        assert mgr.simulator_name == "matterport3d"
        assert mgr.version == "v0_1"

    def test_is_docker_env_manager(self):
        from easi.core.docker_env_manager import DockerEnvironmentManager
        from easi.simulators.matterport3d.v0_1.env_manager import Matterport3DEnvManager
        mgr = Matterport3DEnvManager()
        assert isinstance(mgr, DockerEnvironmentManager)

    def test_image_name(self):
        from easi.simulators.matterport3d.v0_1.env_manager import Matterport3DEnvManager
        mgr = Matterport3DEnvManager()
        assert mgr.image_name == "easi_matterport3d_v0_1"

    def test_gpu_required(self):
        from easi.simulators.matterport3d.v0_1.env_manager import Matterport3DEnvManager
        mgr = Matterport3DEnvManager()
        assert mgr.gpu_required is True

    def test_dockerfile_exists(self):
        from easi.simulators.matterport3d.v0_1.env_manager import Matterport3DEnvManager
        mgr = Matterport3DEnvManager()
        assert mgr.dockerfile_path.exists()

    def test_registry_discovers(self):
        from easi.simulators.registry import list_simulators
        sims = list_simulators()
        assert "matterport3d" in sims or "matterport3d:v0_1" in sims


class TestMatterport3DBridgeSyntax:
    """Bridge script can be parsed."""

    def test_bridge_syntax(self):
        import py_compile
        py_compile.compile(
            "easi/simulators/matterport3d/v0_1/bridge.py",
            doraise=True,
        )
