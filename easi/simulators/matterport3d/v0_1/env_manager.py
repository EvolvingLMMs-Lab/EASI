"""Docker environment manager for Matterport3DSimulator."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from easi.core.docker_env_manager import DockerEnvironmentManager
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# HuggingFace dataset repo containing Matterport3D scene data
_HF_REPO = "Gen3DF/Matterport3d"


class Matterport3DEnvManager(DockerEnvironmentManager):
    """Manages Docker image for Matterport3DSimulator with EGL rendering.

    post_install() auto-downloads the Matterport3D scene data from
    HuggingFace (Gen3DF/Matterport3d) via snapshot_download, then
    reassembles and extracts the archive. The HF repo is gated — users
    must accept the TOS at https://huggingface.co/datasets/Gen3DF/Matterport3d
    before downloading.
    """

    @property
    def simulator_name(self) -> str:
        return "matterport3d"

    @property
    def version(self) -> str:
        return "v0_1"

    @property
    def image_name(self) -> str:
        return "easi_matterport3d_v0_1"

    @property
    def dockerfile_path(self) -> Path:
        return Path(__file__).parent / "Dockerfile"

    @property
    def gpu_required(self) -> bool:
        return True

    @property
    def container_python_path(self) -> str:
        return "/usr/bin/python3"

    @property
    def container_data_mount(self) -> str:
        return "/data/v1/scans"

    @property
    def easi_mount(self) -> str:
        return "/opt/easi"

    def get_system_deps(self) -> list[str]:
        return ["docker"]

    def get_env_vars(self) -> dict[str, str]:
        return {"PYTHONPATH": "/opt/MatterSim/build:/opt/easi"}

    @property
    def data_dir(self) -> Path:
        """Default data directory. User can override via manifest data_dir or task YAML."""
        override = self.installation_kwargs.get("data_dir")
        if override:
            return Path(override)
        return Path("/datasets/matterport3d")

    def post_install(self) -> None:
        """Download Matterport3D scene data from HuggingFace.

        Uses huggingface_hub.snapshot_download to download all files
        (no git-lfs needed). Then runs merge.sh + unzip.sh to reassemble
        and extract scene scans.

        HF repo layout:
          Gen3DF/Matterport3d/
          ├── README.md
          └── matterport3d/
              ├── download.py
              ├── merge.sh
              ├── unzip.sh
              ├── matterport3d_part_000  (1.07 GB)
              ├── ...
              └── matterport3d_part_007  (242 MB)

        snapshot_download downloads into dest/, so merge.sh is at
        dest/matterport3d/merge.sh.
        """
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
        )

        dest = self.data_dir
        hf_repo = self.installation_kwargs.get("hf_dataset_repo", _HF_REPO)

        # Skip if data already exists (idempotent)
        # After extraction, scans live at dest/matterport3d/v1/scans/
        mp3d_subdir = dest / "matterport3d"
        if (mp3d_subdir / "v1" / "scans").exists():
            logger.info("Matterport3D data already exists at %s, skipping download.", dest)
            return

        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Matterport3D dataset from HF repo %s to %s ...", hf_repo, dest)
        logger.info(
            "NOTE: This dataset is gated. If download fails, visit\n"
            "  https://huggingface.co/datasets/%s\n"
            "and accept the Terms of Service, then retry.",
            hf_repo,
        )

        # Step 1: Download all files via snapshot_download
        # Distinguish TOS/auth errors (fail immediately) from
        # transient network errors (retry up to 3 times).
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                snapshot_download(
                    repo_id=hf_repo,
                    repo_type="dataset",
                    local_dir=str(dest),
                )
                break
            except (GatedRepoError, RepositoryNotFoundError) as e:
                raise RuntimeError(
                    f"Access denied downloading {hf_repo}. "
                    f"Please visit https://huggingface.co/datasets/{hf_repo} "
                    f"and accept the Terms of Service, then retry.\n"
                    f"Error: {e}"
                ) from e
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "Download failed (attempt %d/%d): %s. Retrying ...",
                        attempt, max_retries, e,
                    )
                    time.sleep(5 * attempt)  # brief backoff
                else:
                    raise RuntimeError(
                        f"Download failed after {max_retries} attempts: {e}"
                    ) from e

        # Step 2: Reassemble chunks (merge.sh is at dest/matterport3d/merge.sh)
        if (mp3d_subdir / "merge.sh").exists():
            logger.info("Reassembling dataset chunks ...")
            subprocess.run(
                ["bash", "merge.sh"],
                cwd=str(mp3d_subdir),
                check=True,
            )

        # Step 3: Extract
        if (mp3d_subdir / "unzip.sh").exists():
            logger.info("Extracting dataset ...")
            subprocess.run(
                ["bash", "unzip.sh"],
                cwd=str(mp3d_subdir),
                check=True,
            )

        logger.info("Matterport3D dataset ready at %s", dest)
