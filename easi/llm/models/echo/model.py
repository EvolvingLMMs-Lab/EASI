"""Echo model for testing the custom model server pipeline."""
from __future__ import annotations

from easi.llm.models.base_model_server import BaseModelServer
from easi.llm.models.helpers import extract_text_only


class EchoModel(BaseModelServer):
    """Returns the user's message back. Useful for testing."""

    def load(self, model_path: str, device: str, **kwargs) -> None:
        pass

    def generate(self, messages: list[dict], **kwargs) -> str:
        text = extract_text_only(messages)
        return f"Echo: {text}"
