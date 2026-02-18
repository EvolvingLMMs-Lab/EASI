"""Abstract base class for tasks (benchmarks).

A task owns:
- Which simulator+version to use (pinned via task.yaml)
- The action space available to the agent
- The dataset (episodes to evaluate on)
- Episode-to-simulator mapping (format_reset_config)
- Success criteria and metrics (evaluate_episode)

Concrete tasks subclass this and implement:
- get_task_yaml_path() — where the task.yaml lives
- format_reset_config() — adapter from dataset episodes to simulator configs
- evaluate_episode() — compute metrics for a completed episode
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from easi.core.episode import StepResult
from easi.core.exceptions import DatasetError
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def hf_row_to_episode(row: dict) -> dict:
    """Convert a HuggingFace dataset row to an episode dict.

    HF dataset rows contain all information for a single episode.
    For EB-Alfred_easi: {id, task, repeat_idx, instruction, task_type, trial_id}
    This is a passthrough — the row IS the episode.
    """
    return dict(row)


class BaseTask(ABC):
    """Abstract base for all tasks (benchmarks)."""

    def __init__(
        self,
        data_dir: Path | None = None,
        split_yaml_path: Path | None = None,
    ):
        self._split_yaml_path = split_yaml_path
        self._config = self._load_config()
        self._episodes: list[dict] | None = None
        self._action_space_cache: list[str] | None = None
        self._data_dir = data_dir

    @abstractmethod
    def get_task_yaml_path(self) -> Path:
        """Return path to this task's task.yaml."""
        ...

    @abstractmethod
    def format_reset_config(self, episode: dict) -> dict:
        """Translate a dataset episode into simulator reset kwargs.

        This is the core adapter method between dataset format and simulator API.
        """
        ...

    @abstractmethod
    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Evaluate a completed episode, returning a metric dict.

        Example return: {"success": 1.0, "spl": 0.73, "distance_to_goal": 0.15}
        """
        ...

    # --- Hooks ---

    def on_episode_reset(self, observation, agent) -> None:
        """Called after simulator reset, before the agent-simulator loop.

        Override in subclasses to perform task-specific setup, e.g. updating
        the agent's action space from bridge metadata.

        Args:
            observation: The initial observation from sim.reset().
            agent: The agent instance (may have update_action_space, etc.).
        """

    # --- Shared implementation ---

    def get_bridge_script_path(self) -> Path | None:
        """Return task-specific bridge script path, or None for simulator default.

        Override in subclasses to provide a task-specific bridge that extends
        the generic simulator bridge (e.g., EBAlfredBridge extends AI2ThorBridge).
        """
        return None

    @property
    def simulator_configs(self) -> dict:
        """Full simulator configuration from task YAML (includes additional_deps)."""
        return self._config.get("simulator_configs", {})

    @property
    def additional_deps(self) -> list[str]:
        """Extra pip packages to install in the simulator conda env."""
        return self.simulator_configs.get("additional_deps", [])

    @property
    def simulator_kwargs(self) -> dict:
        """Bridge-facing kwargs (simulator_configs minus additional_deps/env_vars + max_steps)."""
        cfg = dict(self.simulator_configs)
        cfg.pop("additional_deps", None)
        cfg.pop("env_vars", None)
        cfg["max_steps"] = self.max_steps
        return cfg

    @property
    def extra_env_vars(self) -> dict[str, str]:
        """Task-level environment variables from simulator_configs.env_vars."""
        return self.simulator_configs.get("env_vars", {})

    def get_instruction(self, episode: dict) -> str:
        """Return human-readable task instruction for this episode.

        Default tries common field names. Override in subclasses
        for benchmarks that use different keys.
        """
        return episode.get("instruction", episode.get("task_description", self.name))

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def simulator_key(self) -> str:
        """Returns e.g. 'dummy:v1' — used to look up from simulator registry."""
        return self._config["simulator"]

    @property
    def action_space(self) -> list[str]:
        if self._action_space_cache is None:
            self._action_space_cache = self._build_action_space()
        return self._action_space_cache

    def _build_action_space(self) -> list[str]:
        """Return the action space for this task.

        Override in subclasses to define the action space programmatically.
        """
        return []

    @property
    def max_steps(self) -> int:
        return self._config.get("max_steps", 500)

    def download_dataset(self, force: bool = False) -> Path:
        """Download dataset if needed. Returns path to local data directory.

        Args:
            force: If True, delete cached dataset and re-download.

        - source=local: validate path exists, return it
        - source=huggingface: download via huggingface_hub, cache locally
        """
        dataset_config = self._config.get("dataset", {})
        source = dataset_config.get("source", "local")

        if source == "local":
            path = dataset_config.get("path")
            if path is None:
                # Use built-in episodes (no download needed)
                return Path()
            local_path = Path(path)
            if not local_path.exists():
                raise DatasetError(f"Local dataset path does not exist: {local_path}")
            return local_path

        elif source == "huggingface":
            return self._download_huggingface(dataset_config, force=force)

        else:
            raise DatasetError(f"Unknown dataset source: {source}")

    def load_episodes(self) -> list[dict]:
        """Load and return all episodes from the dataset."""
        if self._episodes is not None:
            return self._episodes

        self._episodes = self._load_episodes_from_config()
        logger.info("Loaded %d episodes for task %s", len(self._episodes), self.name)
        return self._episodes

    def get_episode(self, index: int) -> dict:
        """Get a single episode by index."""
        episodes = self.load_episodes()
        if index < 0 or index >= len(episodes):
            raise IndexError(f"Episode index {index} out of range [0, {len(episodes)})")
        return episodes[index]

    def __len__(self) -> int:
        """Number of episodes."""
        return len(self.load_episodes())

    def _load_config(self) -> dict:
        """Load task config from split yaml (if provided) or default task.yaml."""
        yaml_path = self._split_yaml_path or self.get_task_yaml_path()
        if not yaml_path.exists():
            raise DatasetError(f"Task config not found: {yaml_path}")
        return yaml.safe_load(yaml_path.read_text())

    def _load_episodes_from_config(self) -> list[dict]:
        """Load episodes from the dataset.

        For HuggingFace datasets: downloads the repo, then loads the split
        using the datasets library. Each row = one episode.
        For local datasets: looks for episodes.json.
        """
        dataset_config = self._config.get("dataset", {})
        source = dataset_config.get("source", "local")

        if source == "huggingface":
            return self._load_episodes_from_hf(dataset_config)

        # Local source — existing behavior
        data_dir = self.download_dataset()
        if not data_dir or data_dir == Path():
            return self._get_builtin_episodes()

        episodes_file = data_dir / "episodes.json"
        if episodes_file.exists():
            return json.loads(episodes_file.read_text())

        raise DatasetError(
            f"No episodes.json found in {data_dir}. "
            f"Override _load_episodes_from_config() for custom loading."
        )

    def _load_episodes_from_hf(self, dataset_config: dict) -> list[dict]:
        """Load episodes from a HuggingFace dataset (subset + split).

        Each row in the dataset = one episode dict.
        Downloads all files via snapshot_download, then loads locally.
        """
        data_dir = self.download_dataset()

        subset = dataset_config.get("subset")
        split_name = dataset_config.get("split")

        try:
            from datasets import (
                get_dataset_config_names,
                get_dataset_split_names,
                load_dataset,
            )
        except ImportError:
            raise DatasetError(
                "The 'datasets' library is required for HF episode loading. "
                "Install with: pip install datasets"
            )

        local_path = str(data_dir)

        # Auto-detect subset if not specified
        if subset is None:
            configs = get_dataset_config_names(local_path)
            if len(configs) == 1:
                subset = configs[0]
                logger.info("Auto-detected single subset: %s", subset)
            elif "default" in configs:
                subset = "default"
            else:
                raise DatasetError(
                    f"Dataset at {local_path} has multiple subsets {configs} — "
                    f"please specify 'subset' in task yaml."
                )

        # Auto-detect split if not specified
        if split_name is None:
            splits = get_dataset_split_names(local_path, subset)
            if len(splits) == 1:
                split_name = splits[0]
                logger.info("Auto-detected single split: %s", split_name)
            else:
                raise DatasetError(
                    f"Dataset at {local_path} subset={subset} has "
                    f"multiple splits {splits} — "
                    f"please specify 'split' in task yaml."
                )

        logger.info(
            "Loading episodes from local HF dataset %s subset=%s split=%s",
            local_path, subset, split_name,
        )

        import tempfile
        hf_cache = Path(tempfile.gettempdir()) / "easi_hf_cache"
        ds = load_dataset(local_path, subset, split=split_name,
                          cache_dir=str(hf_cache))
        episodes = [hf_row_to_episode(row) for row in ds]

        for ep in episodes:
            ep["_data_dir"] = str(data_dir)

        logger.info("Loaded %d episodes from %s/%s/%s",
                     len(episodes), local_path, subset, split_name)
        return episodes

    def _get_builtin_episodes(self) -> list[dict]:
        """Return built-in episodes when no dataset download is needed.

        Override in subclasses that provide built-in test episodes (e.g., DummyTask).
        """
        return []

    def _download_huggingface(self, config: dict, force: bool = False) -> Path:
        """Download a dataset from HuggingFace Hub with file-based locking.

        Uses snapshot_download to get the full repo (including .zip files),
        then extracts any listed zip_files.
        """
        from easi.utils.locking import file_lock
        from easi.utils.paths import get_locks_dir

        repo_id = config["repo_id"]
        lock_path = get_locks_dir() / f"dataset_{repo_id.replace('/', '_')}.lock"

        # Use data_dir if set, otherwise default datasets dir
        if self._data_dir:
            base_dir = self._data_dir
        else:
            from easi.utils.paths import get_datasets_dir
            base_dir = get_datasets_dir()

        with file_lock(lock_path):
            target = base_dir / repo_id.replace("/", "_")

            if force and target.exists():
                import shutil
                logger.info("Force re-download: removing cached %s", target)
                shutil.rmtree(target, ignore_errors=True)

            if not target.exists():
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    raise DatasetError(
                        "huggingface_hub is required for HuggingFace downloads. "
                        "Install with: pip install huggingface_hub"
                    )

                import time as _time

                max_attempts = 5
                for attempt in range(1, max_attempts + 1):
                    logger.info(
                        "Downloading dataset %s from HuggingFace (attempt %d/%d)...",
                        repo_id, attempt, max_attempts,
                    )
                    try:
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=str(target),
                            repo_type="dataset",
                        )
                        break  # success
                    except Exception as e:
                        if attempt < max_attempts:
                            wait = 2 ** attempt  # 2, 4, 8, 16s
                            logger.warning(
                                "Download attempt %d failed: %s. Retrying in %ds...",
                                attempt, e, wait,
                            )
                            _time.sleep(wait)
                        else:
                            if target.exists():
                                import shutil
                                shutil.rmtree(target, ignore_errors=True)
                            raise DatasetError(
                                f"Failed to download {repo_id} after {max_attempts} attempts: {e}"
                            )

                logger.info("Downloaded dataset %s to %s", repo_id, target)
            else:
                logger.info("Dataset %s already cached at %s", repo_id, target)

            # Extract any .zip files listed in config
            zip_files = config.get("zip_files", [])
            if zip_files:
                self._extract_zip_files(target, zip_files)

            return target

    @staticmethod
    def _extract_zip_files(dataset_dir: Path, zip_filenames: list[str]) -> None:
        """Extract listed .zip files within a downloaded dataset directory."""
        import zipfile as zf

        for zip_name in zip_filenames:
            zip_path = dataset_dir / zip_name
            if not zip_path.exists():
                logger.warning("Zip file not found: %s", zip_path)
                continue

            marker = dataset_dir / f".{zip_name}.extracted"
            if marker.exists():
                logger.trace("Already extracted: %s", zip_name)
                continue

            logger.info("Extracting %s...", zip_path)
            with zf.ZipFile(zip_path, "r") as z:
                z.extractall(dataset_dir)

            marker.write_text("extracted")
            logger.info("Extracted %s to %s", zip_name, dataset_dir)
