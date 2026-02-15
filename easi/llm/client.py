"""Unified LLM client wrapping LiteLLM.

Provides text generation with optional response_format pass-through
for API-level JSON schema enforcement.

Usage tracking is cumulative — call get_usage() to snapshot, reset_usage() between episodes.
"""
from __future__ import annotations

from typing import Any

from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid requiring litellm when not needed.
litellm = None


def _ensure_imports() -> None:
    """Import litellm on first use."""
    global litellm
    if litellm is None:
        try:
            import litellm as _litellm
        except ImportError as e:
            raise ImportError(
                "LLMClient requires litellm. "
                "Install with: pip install easi[llm]"
            ) from e
        litellm = _litellm
        # Suppress litellm's verbose logging
        litellm.suppress_debug_info = True


class LLMClient:
    """Unified LLM client for all backends."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = base_url
        self.default_kwargs = kwargs
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "num_calls": 0,
            "cost_usd": 0.0,
        }

    def generate(self, messages: list[dict], response_format: dict | None = None) -> str:
        """Generate text completion. Drop-in for LLMApiClient.generate()."""
        _ensure_imports()

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.default_kwargs,
        }
        if self.base_url:
            call_kwargs["api_base"] = self.base_url
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        logger.trace("LLM call: model=%s, messages=%d", self.model, len(messages))
        response = litellm.completion(**call_kwargs)
        self._track_usage(response)

        content = response.choices[0].message.content
        logger.trace("LLM response: %s", content[:200] if content else "")
        return content

    def get_usage(self) -> dict:
        """Return cumulative usage stats (copy)."""
        return dict(self._usage)

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "num_calls": 0,
            "cost_usd": 0.0,
        }

    def _track_usage(self, response: Any) -> None:
        """Accumulate token usage and cost from a LiteLLM response."""
        usage = getattr(response, "usage", None)
        if usage:
            self._usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            self._usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
        self._usage["num_calls"] += 1
        try:
            cost = litellm.completion_cost(completion_response=response)
            self._usage["cost_usd"] += float(cost)
        except Exception:
            pass  # Cost unavailable for local/unknown models
