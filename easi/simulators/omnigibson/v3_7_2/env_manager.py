"""Environment manager for OmniGibson v3.7.2 + Isaac Sim 4.5.0.

Used by BEHAVIOR-1K benchmark. Requires Python 3.10, CUDA 12.4, and conda.

Replicates the BEHAVIOR-1K setup.sh install process:
1. Conda env creation (Python 3.10) + numpy/setuptools via base _do_install()
2. Git clone BEHAVIOR-1K repo (post_install)
3. Install PyTorch with CUDA (post_install)
4. Install bddl from cloned repo (post_install)
5. Install OmniGibson from cloned repo (post_install)
6. Download + install 26 Isaac Sim 4.5.0 wheels from pypi.nvidia.com (post_install)
7. Fix websockets conflict in Isaac Sim extscache (post_install)
8. Fix cffi compatibility (post_install)

Rendering mode is controlled via OMNIGIBSON_HEADLESS, set by the simulator's
render adapter on top of the active core render backend. No Xvfb needed.

NFS workaround: On NFS/FUSE filesystems, /proc/self/exe can resolve to
"python3.10 (deleted)" which crashes Isaac Sim's Carbonite library. The
get_python_executable() override copies the Python binary to /tmp (local
filesystem) where /proc/self/exe resolves correctly.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager
from easi.core.render_platforms import EnvVars
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# 26 Isaac Sim 4.5.0 packages (from BEHAVIOR-1K/setup.sh)
ISAAC_SIM_PACKAGES = [
    "omniverse_kit-106.5.0.162521",
    "isaacsim_kernel-4.5.0.0",
    "isaacsim_app-4.5.0.0",
    "isaacsim_core-4.5.0.0",
    "isaacsim_gui-4.5.0.0",
    "isaacsim_utils-4.5.0.0",
    "isaacsim_storage-4.5.0.0",
    "isaacsim_asset-4.5.0.0",
    "isaacsim_sensor-4.5.0.0",
    "isaacsim_robot_motion-4.5.0.0",
    "isaacsim_robot-4.5.0.0",
    "isaacsim_benchmark-4.5.0.0",
    "isaacsim_code_editor-4.5.0.0",
    "isaacsim_ros1-4.5.0.0",
    "isaacsim_cortex-4.5.0.0",
    "isaacsim_example-4.5.0.0",
    "isaacsim_replicator-4.5.0.0",
    "isaacsim_rl-4.5.0.0",
    "isaacsim_robot_setup-4.5.0.0",
    "isaacsim_ros2-4.5.0.0",
    "isaacsim_template-4.5.0.0",
    "isaacsim_test-4.5.0.0",
    "isaacsim-4.5.0.0",
    "isaacsim_extscache_physics-4.5.0.0",
    "isaacsim_extscache_kit-4.5.0.0",
    "isaacsim_extscache_kit_sdk-4.5.0.0",
]


class OmniGibsonEnvManager(BaseEnvironmentManager):
    """Environment manager for OmniGibson 3.7.2 + Isaac Sim 4.5.0."""

    _local_python_dir: str | None = None  # Cached /tmp copy directory

    @property
    def simulator_name(self) -> str:
        return "omnigibson"

    @property
    def version(self) -> str:
        return "v3_7_2"

    @property
    def default_render_platform(self) -> str:
        return "native"

    @property
    def supported_render_platforms(self) -> list[str]:
        return ["native", "auto", "xorg"]

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda"]

    def get_validation_import(self) -> str:
        return "import omnigibson; assert omnigibson.__version__.startswith('3.7')"

    def _get_conda_python(self) -> str:
        """Return the real conda env Python path (for install-time use)."""
        return super().get_python_executable()

    def get_python_executable(self) -> str:
        """Return a local /tmp copy of the Python binary.

        On NFS/FUSE filesystems, /proc/self/exe resolves to
        "python3.10 (deleted)" which causes Isaac Sim's Carbonite library
        to abort. Copying the binary to /tmp (a local filesystem) ensures
        /proc/self/exe resolves correctly.

        The copy is cached for the lifetime of this env manager instance
        and cleaned up via atexit.
        """
        if self._local_python_dir is not None:
            local_python = Path(self._local_python_dir) / "python3"
            if local_python.exists():
                return str(local_python)

        conda_python = self._get_conda_python()
        # Resolve symlinks to get the real binary
        real_binary = str(Path(conda_python).resolve())

        # Guard: if the conda env doesn't exist yet (e.g. during env_is_ready()
        # before installation), return the raw path so the caller can detect
        # non-existence via Path(path).exists() rather than crashing on copy.
        if not Path(real_binary).exists():
            return conda_python

        tmp_dir = tempfile.mkdtemp(prefix="easi_python_")
        local_python = Path(tmp_dir) / "python3"
        shutil.copy2(real_binary, str(local_python))
        local_python.chmod(0o755)

        self._local_python_dir = tmp_dir
        atexit.register(shutil.rmtree, tmp_dir, True)

        logger.trace(
            "Copied Python to local filesystem: %s -> %s",
            real_binary,
            local_python,
        )
        return str(local_python)

    def get_env_vars(self, render_platform_name: str | None = None) -> EnvVars:
        """Export env vars for EULA acceptance and PYTHONHOME.

        OMNIGIBSON_HEADLESS is set by the simulator render adapter, not here.

        PYTHONHOME is set to the conda env directory so the /tmp Python
        copy can find the conda env's stdlib and site-packages.
        """
        conda_python = self._get_conda_python()
        # conda env dir is two levels up from bin/python
        conda_env_dir = str(Path(conda_python).resolve().parent.parent)
        return EnvVars(
            replace={
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONHOME": conda_env_dir,
            }
        )

    def post_install(self, context: dict) -> None:
        """Replicate BEHAVIOR-1K setup.sh install process.

        Args:
            context: Dict with env_dir, extras_dir, env_vars keys.
        """
        extras_dir = Path(context["extras_dir"])
        extras_dir.mkdir(parents=True, exist_ok=True)
        python = self._get_conda_python()

        # Git clone BEHAVIOR-1K
        self._clone_behavior_1k(extras_dir)

        behavior_1k_dir = extras_dir / "BEHAVIOR-1K"

        # Install PyTorch with CUDA
        self._install_pytorch(python)

        # Install bddl from cloned repo
        self._install_bddl(python, behavior_1k_dir)

        # Install OmniGibson from cloned repo
        self._install_omnigibson(python, behavior_1k_dir)

        # Download + install Isaac Sim 4.5.0 wheels
        self._install_isaac_sim(python)

        # Fix websockets conflict in Isaac Sim extscache
        self._fix_websockets_conflict(python)

        # Fix cffi compatibility
        self._fix_cffi(python)

    def _clone_behavior_1k(self, extras_dir: Path) -> None:
        """Git clone BEHAVIOR-1K repo (idempotent)."""
        repo_url = self.installation_kwargs.get(
            "behavior_1k_repo",
            "https://github.com/StanfordVL/BEHAVIOR-1K.git",
        )
        tag = self.installation_kwargs.get("behavior_1k_tag", "v3.7.2")
        behavior_1k_dir = extras_dir / "BEHAVIOR-1K"

        if behavior_1k_dir.exists():
            logger.info("BEHAVIOR-1K already cloned at %s, skipping", behavior_1k_dir)
            return

        logger.info("Cloning BEHAVIOR-1K (tag %s) from %s", tag, repo_url)
        self._run_command(
            ["git", "clone", "-b", tag, "--depth", "1", repo_url, str(behavior_1k_dir)],
            "clone BEHAVIOR-1K",
        )

    def _install_pytorch(self, python: str) -> None:
        """Install PyTorch with CUDA support."""
        cuda_version = self.installation_kwargs.get("cuda_version", "12.4")
        cuda_ver_short = cuda_version.replace(".", "")

        logger.info("Installing PyTorch with CUDA %s support", cuda_version)
        self._run_command(
            [
                python,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                f"https://download.pytorch.org/whl/cu{cuda_ver_short}",
            ],
            "install PyTorch with CUDA",
        )

    def _install_bddl(self, python: str, behavior_1k_dir: Path) -> None:
        """Install bddl from cloned BEHAVIOR-1K repo."""
        bddl_dir = behavior_1k_dir / "bddl3"
        logger.info("Installing bddl from %s", bddl_dir)
        self._run_command(
            [python, "-m", "pip", "install", "-e", str(bddl_dir)],
            "install bddl",
        )

    def _install_omnigibson(self, python: str, behavior_1k_dir: Path) -> None:
        """Install OmniGibson from cloned BEHAVIOR-1K repo."""
        og_dir = behavior_1k_dir / "OmniGibson"
        logger.info("Installing OmniGibson from %s", og_dir)
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        self._run_command(
            [python, "-m", "pip", "install", "-e", str(og_dir)],
            "install OmniGibson",
            env=env,
        )

    def _install_isaac_sim(self, python: str) -> None:
        """Download and install 26 Isaac Sim 4.5.0 wheels from pypi.nvidia.com.

        Handles GLIBC < 2.34 by renaming wheel filenames from
        manylinux_2_34 to manylinux_2_31.
        """
        logger.info("Installing Isaac Sim 4.5.0 (%d packages)", len(ISAAC_SIM_PACKAGES))
        glibc_old = self._check_glibc_old()
        if glibc_old:
            logger.info("Detected GLIBC < 2.34, will rename wheel tags")

        temp_dir = Path(tempfile.mkdtemp(prefix="easi_isaac_sim_"))
        wheel_files = []

        try:
            for pkg in ISAAC_SIM_PACKAGES:
                pkg_name = pkg.rsplit("-", 1)[0]  # e.g. "isaacsim_core"
                filename = f"{pkg}-cp310-none-manylinux_2_34_x86_64.whl"
                # pypi.nvidia.com uses dashes in URL path
                url = f"https://pypi.nvidia.com/{pkg_name.replace('_', '-')}/{filename}"
                filepath = temp_dir / filename

                logger.info("Downloading %s", pkg)
                self._download_wheel(url, filepath)

                if glibc_old:
                    new_filename = filename.replace("manylinux_2_34", "manylinux_2_31")
                    new_filepath = temp_dir / new_filename
                    filepath.rename(new_filepath)
                    filepath = new_filepath

                wheel_files.append(str(filepath))

            logger.info("Installing %d Isaac Sim wheels", len(wheel_files))
            env = os.environ.copy()
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            self._run_command(
                [python, "-m", "pip", "install"] + wheel_files,
                "install Isaac Sim wheels",
                env=env,
            )
        finally:
            # Clean up temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _fix_websockets_conflict(self, python: str) -> None:
        """Remove pip_prebundle/websockets dirs under Isaac Sim extscache.

        Isaac Sim bundles an old websockets version that conflicts with
        the system one. Removing it forces the system package to be used.
        """
        try:
            result = subprocess.run(
                [
                    python,
                    "-c",
                    "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            isaac_path = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            isaac_path = ""

        if not isaac_path or not Path(isaac_path).exists():
            logger.info("ISAAC_PATH not found, skipping websockets fix")
            return

        extscache = Path(isaac_path) / "extscache"
        if not extscache.exists():
            return

        logger.info("Fixing websockets conflict under %s", extscache)
        import shutil

        for ws_dir in extscache.rglob("pip_prebundle/websockets"):
            if ws_dir.is_dir():
                logger.trace("Removing %s", ws_dir)
                shutil.rmtree(ws_dir, ignore_errors=True)

    def _fix_cffi(self, python: str) -> None:
        """Force reinstall cffi==1.17.1 to resolve Isaac Sim compatibility."""
        logger.info("Fixing cffi compatibility")
        self._run_command(
            [python, "-m", "pip", "install", "--force-reinstall", "cffi==1.17.1"],
            "fix cffi",
        )

    @staticmethod
    def _check_glibc_old() -> bool:
        """Check if system GLIBC version is < 2.34."""
        try:
            result = subprocess.run(
                ["ldd", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            # Look for version like "2.31", "2.32", "2.33"
            import re

            match = re.search(r"(\d+\.\d+)", output)
            if match:
                version = float(match.group(1))
                return version < 2.34
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False

    @staticmethod
    def _download_wheel(url: str, dest: Path) -> None:
        """Download a single wheel file."""
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "easi/1.0"})
        with urllib.request.urlopen(req) as response, open(str(dest), "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
