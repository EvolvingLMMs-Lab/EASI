"""Unified LLM client wrapping LiteLLM + Instructor.

Provides two generation modes:
- generate(): returns raw text (backward-compatible with LLMApiClient)
- generate_structured(): returns a validated Pydantic model (via Instructor)

Usage tracking is cumulative — call get_usage() to snapshot, reset_usage() between episodes.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid requiring litellm/instructor when not needed.
litellm = None
instructor = None


def _ensure_imports() -> None:
    """Import litellm and instructor on first use."""
    global litellm, instructor
    if litellm is None:
        try:
            import litellm as _litellm
            import instructor as _instructor
        except ImportError as e:
            raise ImportError(
                "LLMClient requires litellm and instructor. "
                "Install with: pip install easi[llm]"
            ) from e
        litellm = _litellm
        instructor = _instructor
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

    def generate(self, messages: list[dict]) -> str:
        """Generate text completion. Drop-in for LLMApiClient.generate()."""
        _ensure_imports()

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.default_kwargs,
        }
        if self.base_url:
            call_kwargs["api_base"] = self.base_url

        logger.trace("LLM call: model=%s, messages=%d", self.model, len(messages))
        response = litellm.completion(**call_kwargs)
        self._track_usage(response)

        content = response.choices[0].message.content
        logger.trace("LLM response: %s", content[:200] if content else "")
        return content

    def generate_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Generate structured output validated against a Pydantic model."""
        _ensure_imports()

        client = instructor.from_litellm(litellm.completion)

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_model": response_model,
            "max_retries": 2,
            **self.default_kwargs,
        }
        if self.base_url:
            call_kwargs["api_base"] = self.base_url

        logger.trace(
            "LLM structured call: model=%s, schema=%s",
            self.model, response_model.__name__,
        )
        result = client.chat.completions.create(**call_kwargs)
        return result

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
