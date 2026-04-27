"""Base class for evaluation backend adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkScores:
    """Scores for a single benchmark."""
    overall: float | None = None
    sub_scores: dict[str, float | None] = field(default_factory=dict)


@dataclass
class ExtractionReport:
    """Extraction quality stats for a single benchmark."""
    total: int
    failed: int
    failure_rate: float
    method: str  # "extract_matching" or "multiple_choice"


class BackendAdapter(ABC):
    """Interface for evaluation backend adapters.

    Each adapter maps user-facing benchmark keys to backend-specific
    task names and knows how to launch, monitor, and parse results
    from its backend.
    """

    TASK_MAP: dict[str, str]

    @abstractmethod
    def build_cmd(
        self,
        model: str,
        benchmarks: dict[str, str],
        output_dir: Path,
        nproc: int,
        *,
        extra_args: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        """Build the subprocess command to run evaluation."""
        ...

    def prepare_datasets(
        self,
        benchmarks: dict[str, str],
        dataset_dir: Path,
        display: object | None = None,
    ) -> bool:
        """Download/prepare datasets before evaluation. Returns True if all OK.

        Default: no-op (returns True). Override for backends that need prep.
        """
        return True

    def check_extraction_quality(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, ExtractionReport]:
        """Check extraction quality per benchmark. Empty dict if not supported."""
        return {}

    def archive_artifacts(
        self,
        model_dir: Path,
        model_name: str,
        data_name: str,
    ) -> None:
        """Archive existing artifacts before judge re-evaluation."""
        pass

    def build_judge_cmd(
        self,
        model: str,
        benchmarks: dict[str, str],
        output_dir: Path,
        nproc: int,
        judge_model: str,
        *,
        extra_args: list[str] | None = None,
        **kwargs,
    ) -> list[str] | None:
        """Build command for judge re-evaluation. None if not supported."""
        return None

    @abstractmethod
    def detect_completion(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, bool]:
        """Check which benchmarks completed. Returns {key: True/False}."""
        ...

    @abstractmethod
    def get_result_files(self, model_dir: Path, model_name: str) -> list[Path]:
        """Return list of result file paths for the archive."""
        ...

    @abstractmethod
    def extract_scores(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> dict[str, BenchmarkScores]:
        """Extract overall + sub-scores for each benchmark."""
        ...

    def get_env_overrides(self) -> dict[str, str]:
        """Return env vars to set for the subprocess. Default: PYTHONUNBUFFERED=1."""
        return {"PYTHONUNBUFFERED": "1"}

    def find_completed_tasks(
        self,
        model_dir: Path,
        model_name: str,
        benchmarks: dict[str, str],
    ) -> set[str]:
        """Return set of benchmark keys that already have results (for resume).

        Default: delegates to detect_completion.
        """
        completion = self.detect_completion(model_dir, model_name, benchmarks)
        return {k for k, done in completion.items() if done}

    @property
    def name(self) -> str:
        """Backend name for the submission payload."""
        return self.__class__.__name__.lower()
