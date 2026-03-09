from __future__ import annotations

from abc import ABC, abstractmethod

from easi.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModelServer(ABC):
    """Abstract base class for custom model servers.

    Subclasses must implement ``load`` and ``generate``.  The ``unload``
    method is optional and defaults to a no-op.
    """

    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> None:
        """Load model weights, tokenizer, processors."""

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate response from OpenAI-format messages."""

    def unload(self) -> None:
        """Release GPU memory. Optional override."""
        pass
