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
# Parameters accepted by litellm.completion() / OpenAI chat completions API.
# Anything not in this set is silently dropped to avoid provider rejections.
_LITELLM_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "n", "stop", "seed",
    "frequency_penalty", "presence_penalty", "logit_bias",
    "logprobs", "top_logprobs",
    "response_format", "tools", "tool_choice",
    "stream", "stream_options",
    "user", "metadata",
})

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
        num_retries: int = 3,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = base_url
        self.num_retries = num_retries
        self.timeout = timeout
        # Only keep params that litellm/OpenAI API recognises.
        dropped = {k: v for k, v in kwargs.items() if k not in _LITELLM_PARAMS}
        if dropped:
            logger.debug("Dropping unsupported generation kwargs: %s", dropped)
        self.default_kwargs = {k: v for k, v in kwargs.items() if k in _LITELLM_PARAMS}
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
            "num_retries": self.num_retries,
            "timeout": self.timeout,
            "drop_params": True,
            **self.default_kwargs,
        }
        if self.base_url:
            call_kwargs["api_base"] = self.base_url
            # Local servers (vLLM, custom) don't need a real API key,
            # but LiteLLM requires one for the openai/ prefix.
            call_kwargs.setdefault("api_key", "dummy")
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        logger.trace("LLM call: model=%s, messages=%d", self.model, len(messages))
        try:
            response = litellm.completion(**call_kwargs)
        except Exception as e:
            logger.trace("LLM API error: %s: %s", type(e).__name__, e)
            raise
        self._track_usage(response)

        content = response.choices[0].message.content or ""
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
