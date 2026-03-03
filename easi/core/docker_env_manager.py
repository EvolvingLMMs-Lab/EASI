"""Docker-based environment manager for simulators that require containerization.

Parallel to BaseEnvironmentManager (conda-based). Docker simulators subclass this
instead. The bridge code inside the container is identical — same BaseBridge,
same filesystem IPC.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from easi.utils.logging import get_logger
from easi.utils.system_deps import SystemDependencyChecker

logger = get_logger(__name__)


class DockerEnvironmentManager(ABC):
    """Abstract base for Docker-isolated simulator environments.

    Subclasses must define: simulator_name, version, image_name,
    dockerfile_path, gpu_required, container_python_path,
    container_data_mount, easi_mount, get_system_deps().
    """

    def __init__(self, installation_kwargs: dict | None = None):
        self.installation_kwargs = installation_kwargs or {}
        self._dep_checker = SystemDependencyChecker()

    # --- Abstract properties (subclass must implement) ---

    @property
    @abstractmethod
    def simulator_name(self) -> str:
        """Name of the simulator (e.g., 'matterport3d')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version identifier (e.g., 'v0_1')."""
        ...

    @property
    @abstractmethod
    def image_name(self) -> str:
        """Docker image name (e.g., 'easi_matterport3d_v0_1')."""
        ...

    @property
    @abstractmethod
    def dockerfile_path(self) -> Path:
        """Path to Dockerfile for building the image."""
        ...

    @property
    @abstractmethod
    def gpu_required(self) -> bool:
        """Whether the container needs GPU access (--gpus all)."""
        ...

    @property
    @abstractmethod
    def container_python_path(self) -> str:
        """Python executable path inside the container."""
        ...

    @property
    @abstractmethod
    def container_data_mount(self) -> str:
        """Mount point for simulator scene data inside the container."""
        ...

    @property
    @abstractmethod
    def easi_mount(self) -> str:
        """Mount point for EASI repo inside the container (read-only)."""
        ...

    # --- Concrete methods ---

    @abstractmethod
    def get_system_deps(self) -> list[str]:
        """System dependencies (e.g., ['docker'] or ['docker', 'nvidia-docker'])."""
        ...

    def check_system_deps(self) -> list[str]:
        """Check system dependencies, returning list of missing ones."""
        return self._dep_checker.check_all(self.get_system_deps())

    def get_env_vars(self) -> dict[str, str]:
        """Environment variables to set inside the container. Override if needed."""
        return {}

    def get_env_name(self) -> str:
        """Return a name for this environment (used for display/logging)."""
        return f"docker:{self.image_name}"

    def env_is_ready(self) -> bool:
        """Check if the Docker image exists."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def install(self) -> None:
        """Build the Docker image and run post_install (e.g., dataset download)."""
        if not self.env_is_ready():
            dockerfile = self.dockerfile_path
            if not dockerfile.exists():
                raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

            build_context = dockerfile.parent
            logger.info(
                "Building Docker image %s from %s ...",
                self.image_name,
                dockerfile,
            )
            subprocess.run(
                [
                    "docker", "build",
                    "-t", self.image_name,
                    "-f", str(dockerfile),
                    str(build_context),
                ],
                check=True,
            )
            logger.info("Docker image %s built successfully.", self.image_name)
        else:
            logger.info("Docker image %s already exists.", self.image_name)

        # Run post-install hook (e.g., dataset download)
        self.post_install()

    def post_install(self) -> None:
        """Hook for subclasses to run after image build (e.g., download datasets).

        Called by install() after the Docker image is ready.
        Default is a no-op.
        """
        pass

    def remove(self) -> None:
        """Remove the Docker image."""
        logger.info("Removing Docker image %s ...", self.image_name)
        subprocess.run(
            ["docker", "rmi", self.image_name],
            capture_output=True,
        )

    def build_docker_run_command(
        self,
        bridge_command: list[str],
        workspace_dir: str | None = None,
        episode_output_dir: str | None = None,
        data_dir: str | None = None,
    ) -> list[str]:
        """Build a `docker run` command for launching the bridge.

        Mounts IPC workspace, episode output dir, EASI repo, and scene data
        at the same host paths (so rgb_path in response.json works on both sides).
        """
        easi_repo_root = str(Path(__file__).resolve().parents[1])

        cmd = ["docker", "run", "--rm"]

        # GPU
        if self.gpu_required:
            cmd.extend(["--gpus", "all"])

        # Volume mounts (same path on host and container for IPC compatibility)
        if workspace_dir:
            cmd.extend(["-v", f"{workspace_dir}:{workspace_dir}"])
        if episode_output_dir:
            cmd.extend(["-v", f"{episode_output_dir}:{episode_output_dir}"])

        # Data mount (host path -> container mount point)
        if data_dir:
            cmd.extend(["-v", f"{data_dir}:{self.container_data_mount}:ro"])

        # EASI repo (read-only)
        cmd.extend(["-v", f"{easi_repo_root}:{self.easi_mount}:ro"])

        # Environment variables
        cmd.extend(["-e", "PYTHONUNBUFFERED=1"])  # real-time log output
        env_vars = self.get_env_vars()
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Image name
        cmd.append(self.image_name)

        # Bridge command
        cmd.extend(bridge_command)

        return cmd
