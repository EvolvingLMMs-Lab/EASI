"""Dataset downloader supporting local paths and HuggingFace Hub.

Used by BaseTask.download_dataset() but also available standalone.
"""

from __future__ import annotations

from pathlib import Path

from easi.core.exceptions import DatasetError
from easi.utils.locking import file_lock
from easi.utils.logging import get_logger
from easi.utils.paths import get_datasets_dir, get_locks_dir

logger = get_logger(__name__)


class DatasetDownloader:
    """Handles dataset acquisition from local paths and HuggingFace."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or get_datasets_dir()

    def download(self, config: dict) -> Path:
        """Download or resolve a dataset given a config block from task.yaml.

        Args:
            config: Dataset config dict, e.g.:
                {"source": "local", "path": "/data/episodes"}
                {"source": "huggingface", "repo_id": "org/dataset", "split": "val"}

        Returns:
            Path to the local dataset directory.
        """
        source = config.get("source", "local")

        if source == "local":
            return self._resolve_local(config)
        elif source == "huggingface":
            return self._download_huggingface(config)
        else:
            raise DatasetError(f"Unknown dataset source: {source}")

    def _resolve_local(self, config: dict) -> Path:
        """Validate local path exists and return it."""
        path = config.get("path")
        if path is None:
            return Path()  # no path = built-in episodes

        local_path = Path(path)
        if not local_path.exists():
            raise DatasetError(f"Local dataset path does not exist: {local_path}")

        logger.info("Using local dataset: %s", local_path)
        return local_path

    def _download_huggingface(self, config: dict) -> Path:
        """Download from HuggingFace Hub with file-based locking.

        Uses fcntl.flock to prevent concurrent downloads of the same dataset.
        """
        repo_id = config["repo_id"]
        lock_name = f"dataset_{repo_id.replace('/', '_')}.lock"
        lock_path = get_locks_dir() / lock_name

        with file_lock(lock_path):
            target = self.cache_dir / repo_id.replace("/", "_")
            if target.exists():
                logger.info("Dataset %s already cached at %s", repo_id, target)
                return target

            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise DatasetError(
                    f"huggingface_hub is required to download '{repo_id}'. "
                    f"Install with: pip install huggingface_hub"
                )

            logger.info("Downloading dataset %s from HuggingFace...", repo_id)
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    repo_type="dataset",
                )
            except Exception as e:
                # Clean up partial download
                if target.exists():
                    import shutil
                    shutil.rmtree(target, ignore_errors=True)
                raise DatasetError(f"Failed to download {repo_id}: {e}")

            logger.info("Downloaded dataset %s to %s", repo_id, target)
            return target
