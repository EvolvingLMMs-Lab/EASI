"""HTTP client for OpenAI-compatible LLM inference servers.

Works with vLLM, SGLang, Ollama, and the built-in dummy server —
any server that implements the /v1/chat/completions endpoint.
"""

from __future__ import annotations

from typing import Any

import requests

from easi.utils.logging import get_logger

logger = get_logger(__name__)


class LLMApiClient:
    """Stateless HTTP client for LLM inference endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str = "default",
        timeout: float = 120.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        messages: list[dict],
    ) -> str:
        """Send a chat completion request and return the assistant's response text.

        Args:
            messages: Chat history in OpenAI format. Images are embedded as
                content parts with type "image_url" (Decision #10).

        Returns:
            The assistant's response text.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        url = f"{self.base_url}/v1/chat/completions"
        logger.trace("POST %s (messages: %d)", url, len(messages))

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                f"Start one with: easi llm-server"
            )
        except requests.Timeout:
            raise TimeoutError(
                f"LLM server request timed out after {self.timeout}s"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"LLM server returned error: {e}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM server returned no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        logger.trace("Response: %s", content[:100])
        return content
