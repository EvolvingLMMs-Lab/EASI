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
import tarfile
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from easi.core.exceptions import EnvironmentSetupError
from easi.utils.locking import file_lock
from easi.utils.logging import get_logger
from easi.utils.paths import get_locks_dir
from easi.utils.spinner import spinner
from easi.utils.system_deps import SystemDependencyChecker

logger = get_logger(__name__)


class BaseEnvironmentManager(ABC):
    """Abstract base for per-simulator-version environment management."""

    def __init__(self, conda_prefix: Path | None = None, installation_kwargs: dict | None = None):
        self.conda_prefix = conda_prefix or self._default_conda_prefix()
        self.installation_kwargs = installation_kwargs or {}
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
    def default_render_platform(self) -> str:
        """Default rendering platform for this simulator.

        Override in subclasses. Common values:
            "auto"     -- native display if available, xvfb fallback
            "headless" -- no display (simulator handles internally)
            "egl"      -- GPU-accelerated headless via EGL

        See ``easi.core.render_platform`` for all options.
        """
        return "headless"

    @property
    def supported_render_platforms(self) -> list[str]:
        """Render platforms this simulator can use.

        Override in subclasses to advertise which platforms are compatible.
        Validated when user passes ``--render-platform``.
        """
        return [self.default_render_platform]

    @property
    def screen_config(self) -> str:
        """Screen resolution config (e.g. ``"1024x768x24"``).

        Used by platforms that create a virtual display (xvfb).
        Override for custom resolution/depth.
        """
        return "1024x768x24"

    def get_env_vars(self, render_platform_name: str | None = None) -> "EnvVars":
        """Return environment variables to inject into the bridge subprocess.

        Override in subclasses to provide simulator-specific env vars.

        Args:
            render_platform_name: Active render platform name (e.g. "egl").
                Subclasses can use this to conditionally set env vars.

        Returns:
            EnvVars instance. Empty by default.
        """
        from easi.core.render_platforms import EnvVars

        return EnvVars()

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

        # Include simulator env vars (e.g. LD_LIBRARY_PATH for CoppeliaSim)
        env_vars = self.get_env_vars()
        run_env = None
        if env_vars:
            import os
            run_env = env_vars.apply_to_env(os.environ.copy())

        try:
            result = subprocess.run(
                [python_exec, "-c", self.get_validation_import()],
                capture_output=True,
                text=True,
                timeout=30,
                env=run_env,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def remove(self) -> None:
        """Remove the conda environment entirely (for --reinstall)."""
        env_name = self.get_env_name()
        env_path = self.conda_prefix / "envs" / env_name
        if not env_path.exists():
            logger.info("Environment %s does not exist, nothing to remove", env_name)
            return
        with spinner(f"Removing environment {env_name}"):
            self._run_command(
                ["conda", "env", "remove", "-n", env_name, "-y"],
                f"conda env remove {env_name}",
            )
        logger.info("Environment %s removed", env_name)

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

    def install_additional_deps(self, packages: list[str]) -> None:
        """Install extra pip packages into this conda env via uv (idempotent).

        Called by EvaluationRunner when a task declares additional_deps
        in its YAML config.
        """
        if not packages:
            return
        python_exec = self.get_python_executable()
        with spinner(f"Installing task dependencies: {', '.join(packages)}"):
            self._run_command(
                [python_exec, "-m", "uv", "pip", "install"] + packages,
                "uv pip install (task deps)",
            )

    def _do_install(self) -> None:
        """Execute the full install sequence (called under lock)."""
        env_name = self.get_env_name()
        logger.info("Installing environment %s for %s %s", env_name, self.simulator_name, self.version)

        # Check system deps
        self._dep_checker.assert_all(self.get_system_deps())

        # Create/update conda env
        conda_yaml = self.get_conda_env_yaml_path()
        if conda_yaml.exists():
            with spinner(f"Creating conda environment {env_name}"):
                self._run_conda_create(env_name, conda_yaml)
        else:
            logger.warning("No conda_env.yaml found at %s, skipping conda setup", conda_yaml)

        # Install uv in the conda env
        python_exec = self.get_python_executable()
        with spinner("Installing uv"):
            self._run_command([python_exec, "-m", "pip", "install", "uv"], "pip install uv")

        # Install Python deps via uv
        requirements = self.get_requirements_txt_path()
        if requirements.exists():
            with spinner("Installing Python dependencies"):
                self._run_command(
                    [python_exec, "-m", "uv", "pip", "install", "-r", str(requirements)],
                    "uv pip install",
                )
        else:
            logger.warning("No requirements.txt found at %s, skipping uv install", requirements)

        # Run post-install hook (binary downloads, file copies, etc.)
        self._run_post_install()

        # Validate (with env vars so e.g. LD_LIBRARY_PATH is set)
        env_vars = self.get_env_vars()
        validation_env = None
        if env_vars:
            import os
            validation_env = env_vars.apply_to_env(os.environ.copy())
        with spinner("Validating environment"):
            self._run_command(
                [python_exec, "-c", self.get_validation_import()],
                "environment validation",
                env=validation_env,
            )

        logger.info("Environment %s installed and validated successfully", env_name)

    # ── Post-install hook and helpers ──────────────────────────────────

    def get_extras_dir(self) -> Path:
        """Directory for downloaded binaries and other extras (inside conda env dir)."""
        env_path = self.conda_prefix / "envs" / self.get_env_name()
        return env_path / "extras"

    @staticmethod
    def _resolve_template(template: str, variables: dict[str, str]) -> str:
        """Resolve {var} placeholders in a string."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def _get_template_variables(self) -> dict[str, str]:
        """Return template variables for env_vars and post_install use.

        Available variables:
          {env_dir}    — conda env directory
          {extras_dir} — extras directory for binaries/downloads
        """
        env_dir = str(self.conda_prefix / "envs" / self.get_env_name())
        return {
            "env_dir": env_dir,
            "extras_dir": str(self.get_extras_dir()),
        }

    def post_install(self, context: dict) -> None:
        """Override for custom post-install steps (binary downloads, file copies, etc.).

        Called after conda + pip installs, before validation. Use helper methods
        like _download_and_extract() for common operations. Use _run_command()
        with an env dict to run pip install with custom env vars.

        Args:
            context: Dict with keys:
                env_dir    — conda env directory path
                extras_dir — directory for downloaded extras
                env_vars   — resolved env vars from get_env_vars()

        Does nothing by default.
        """

    def _run_post_install(self) -> None:
        """Build context and call post_install() hook."""
        ctx = self._get_template_variables()
        ctx["env_vars"] = self.get_env_vars().to_flat_dict()
        self.post_install(ctx)

    def _download_and_extract(
        self,
        url: str,
        filename: str,
        dest_dir: Path,
        extract: bool = True,
        strip_components: int = 0,
    ) -> None:
        """Download a file and optionally extract it. Idempotent (skips if done).

        Helper for use inside post_install() overrides.

        Args:
            url: Download URL.
            filename: Local filename to save as.
            dest_dir: Directory to download/extract into.
            extract: Whether to extract archives (tar.xz, tar.gz, zip).
            strip_components: Remove N leading path components when extracting.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        # Idempotency: check marker for extracted archives, or file existence
        marker = dest_dir / f".{filename}.done"
        if marker.exists():
            logger.info("Already installed: %s, skipping", filename)
            return
        if not extract and dest.exists():
            logger.info("Already downloaded: %s, skipping", filename)
            return

        logger.trace("Download target: %s -> %s", url, dest)
        with spinner(f"Downloading {filename}"):
            logger.info("Downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "easi/1.0"})
            with urllib.request.urlopen(req) as response, open(str(dest), "wb") as out:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                logger.trace("Download complete: %s (%.1f MB)", filename, total / 1024 / 1024)

        if extract:
            logger.trace("Extracting %s to %s (strip_components=%d)", filename, dest_dir, strip_components)
            with spinner(f"Extracting {filename}"):
                self._extract_archive(dest, dest_dir, strip_components)
            logger.trace("Extraction complete, removing archive %s", dest)
            dest.unlink(missing_ok=True)  # Remove archive to save space
            # Log extracted contents (top-level only)
            top_items = sorted(p.name for p in dest_dir.iterdir() if not p.name.startswith("."))
            logger.trace("Contents of %s after extraction: %s", dest_dir, top_items)

        marker.touch()
        logger.trace("Wrote done marker: %s", marker)

    def _extract_archive(self, archive: Path, dest_dir: Path, strip_components: int = 0) -> None:
        """Extract a tar.xz, tar.gz, tar.bz2, or zip archive."""
        name = archive.name
        if name.endswith((".tar.xz", ".tar.gz", ".tgz", ".tar.bz2")):
            with tarfile.open(str(archive)) as tf:
                if strip_components > 0:
                    for member in tf.getmembers():
                        parts = Path(member.name).parts
                        if len(parts) > strip_components:
                            member.name = str(Path(*parts[strip_components:]))
                            tf.extract(member, dest_dir)
                else:
                    tf.extractall(dest_dir)
        elif name.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(str(archive)) as zf:
                zf.extractall(dest_dir)
        else:
            logger.warning("Unknown archive format: %s, skipping extraction", name)

    # ── Conda / command helpers ─────────────────────────────────────

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

    def _run_command(self, cmd: list[str], description: str, env: dict[str, str] | None = None) -> None:
        """Run a subprocess command, streaming output through the logger.

        Args:
            cmd: Command and arguments.
            description: Human-readable description for error messages.
            env: Optional environment dict. If None, inherits parent env.
        """
        logger.trace("%s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        output_lines = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                logger.trace("  %s", line)
        finally:
            process.stdout.close()
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
