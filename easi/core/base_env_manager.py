"""Abstract base class for per-simulator-version environment management.

Each simulator version provides a concrete subclass that declares:
- conda_env.yaml path (conda-channel packages only)
- requirements.txt path (pip-installable Python deps, installed via uv)
- system dependencies (xvfb, EGL, etc.)
- validation import to confirm the env works

The shared install() logic handles the full sequence:
    check_system_deps -> conda create -> pip install uv -> uv pip install -> validate
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from easi.core.exceptions import EnvironmentSetupError
from easi.utils.locking import file_lock
from easi.utils.logging import get_logger
from easi.utils.paths import get_locks_dir
from easi.utils.system_deps import SystemDependencyChecker

logger = get_logger(__name__)


class BaseEnvironmentManager(ABC):
    """Abstract base for per-simulator-version environment management."""

    def __init__(self, conda_prefix: Path | None = None):
        self.conda_prefix = conda_prefix or self._default_conda_prefix()
        self._dep_checker = SystemDependencyChecker()

    @property
    @abstractmethod
    def simulator_name(self) -> str:
        """Name of the simulator (e.g., 'ai2thor')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier (e.g., 'v2_1_0')."""
        ...

    @abstractmethod
    def get_conda_env_yaml_path(self) -> Path:
        """Path to the conda environment YAML (conda-only deps)."""
        ...

    @abstractmethod
    def get_requirements_txt_path(self) -> Path:
        """Path to requirements.txt (uv-installed Python deps)."""
        ...

    @abstractmethod
    def get_system_deps(self) -> list[str]:
        """List of required system packages (e.g., ['xvfb', 'conda'])."""
        ...

    @abstractmethod
    def get_validation_import(self) -> str:
        """Python import statement to validate env works.

        Example: "import ai2thor; assert ai2thor.__version__.startswith('2.1')"
        """
        ...

    @property
    def needs_display(self) -> bool:
        """Whether this simulator requires a display (X11/Xvfb).

        Override to return True for simulators that need it (e.g., AI2-THOR).
        """
        return False

    @property
    def xvfb_screen_config(self) -> str:
        """Xvfb screen config. Override for custom resolution/depth."""
        return "1024x768x24"

    def get_env_name(self) -> str:
        """Conda environment name for this simulator version."""
        return f"easi_{self.simulator_name}_{self.version}"

    def get_python_executable(self) -> str:
        """Return the full path to the Python executable in this conda env."""
        env_path = self.conda_prefix / "envs" / self.get_env_name()
        return str(env_path / "bin" / "python")

    def check_system_deps(self) -> list[str]:
        """Check system dependencies, returning list of missing ones."""
        return self._dep_checker.check_all(self.get_system_deps())

    def env_is_ready(self) -> bool:
        """Check if the conda environment exists and passes validation."""
        python_exec = self.get_python_executable()
        if not Path(python_exec).exists():
            return False

        try:
            result = subprocess.run(
                [python_exec, "-c", self.get_validation_import()],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install(self) -> None:
        """Install the conda+uv environment with file-based locking.

        Serializes concurrent installs of the same env across processes.
        """
        lock_path = get_locks_dir() / f"{self.get_env_name()}.lock"
        with file_lock(lock_path):
            if self.env_is_ready():
                logger.info("Environment %s already ready, skipping install", self.get_env_name())
                return
            self._do_install()

    def _do_install(self) -> None:
        """Execute the full install sequence (called under lock)."""
        env_name = self.get_env_name()
        logger.info("[Env Installation] %s", env_name)

        # Step 1: Check system deps
        self._dep_checker.assert_all(self.get_system_deps())

        # Step 2: Create/update conda env
        conda_yaml = self.get_conda_env_yaml_path()
        if conda_yaml.exists():
            self._run_conda_create(env_name, conda_yaml)
        else:
            logger.warning("[Env Installation] No conda_env.yaml found at %s, skipping conda setup", conda_yaml)

        # Step 3: Install uv in the conda env
        python_exec = self.get_python_executable()
        self._run_command([python_exec, "-m", "pip", "install", "uv"], "pip install uv")

        # Step 4: Install Python deps via uv
        requirements = self.get_requirements_txt_path()
        if requirements.exists():
            self._run_command(
                [python_exec, "-m", "uv", "pip", "install", "-r", str(requirements)],
                "uv pip install",
            )
        else:
            logger.warning("[Env Installation] No requirements.txt found at %s, skipping uv install", requirements)

        # Step 5: Validate (stream output like other commands)
        self._run_command(
            [python_exec, "-c", self.get_validation_import()],
            "environment validation",
        )

        logger.info("[Env Installation] Environment %s installed and validated successfully", env_name)

    def _run_conda_create(self, env_name: str, yaml_path: Path) -> None:
        """Create or update a conda environment from a YAML file."""
        env_path = self.conda_prefix / "envs" / env_name

        if env_path.exists():
            cmd = ["conda", "env", "update", "-f", str(yaml_path), "-n", env_name]
            desc = "conda env update"
        else:
            cmd = ["conda", "env", "create", "-f", str(yaml_path), "-n", env_name]
            desc = "conda env create"

        self._run_command(cmd, desc)

    def _run_command(self, cmd: list[str], description: str) -> None:
        """Run a subprocess command, streaming output through the logger."""
        logger.info("[Env Installation] %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            logger.debug("  %s", line)
        process.wait()
        if process.returncode != 0:
            raise EnvironmentSetupError(
                f"{description} failed (exit {process.returncode}):\n"
                + "\n".join(output_lines[-20:])
            )

    @staticmethod
    def _default_conda_prefix() -> Path:
        """Determine the conda prefix from the conda executable."""
        try:
            result = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        # Fallback to common default
        return Path.home() / "miniconda3"
