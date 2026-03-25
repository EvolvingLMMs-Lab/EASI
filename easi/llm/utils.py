"""Utility functions for LLM backend configuration."""
from __future__ import annotations

import json
import os

# Backend prefix mapping for LiteLLM model strings.
# vLLM and dummy both use OpenAI-compatible APIs.
_BACKEND_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "vllm": "openai",
    "custom": "openai",
    "dummy": "openai",
}

# Env vars required per backend.
_REQUIRED_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# kwargs that belong to the inference server, not the API call.
# Includes vLLM CLI flags and custom server params (model_path).
_SERVER_KWARGS = {
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "max_model_len",
    "dtype",
    "quantization",
    "enforce_eager",
    "enable_prefix_caching",
    "trust_remote_code",
    "tokenizer_mode",
    "seed",
    "max_num_seqs",
    "enable_log_requests",
    "limit_mm_per_prompt",
    "chat_template_content_format",
    "chat_template",
    "model_path",
}


def parse_llm_kwargs(raw: str | None) -> dict:
    """Parse --llm-kwargs JSON string into a dict."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --llm-kwargs: {e}") from e
    if not isinstance(result, dict):
        raise ValueError("--llm-kwargs must be a JSON object (dict)")
    return result


def split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split kwargs into (server_kwargs, client_kwargs).

    Known server params (tensor_parallel_size, etc.) go to ServerManager.
    Everything else (temperature, max_tokens, etc.) goes to LLMClient.
    """
    server = {k: v for k, v in kwargs.items() if k in _SERVER_KWARGS}
    client = {k: v for k, v in kwargs.items() if k not in _SERVER_KWARGS}
    return server, client


def build_litellm_model(backend: str, model: str) -> str:
    """Map EASI backend+model to a LiteLLM model string."""
    prefix = _BACKEND_PREFIX.get(backend, "openai")
    return f"{prefix}/{model}"


def validate_backend(backend: str) -> None:
    """Raise EnvironmentError if required env var is missing for backend."""
    env_var = _REQUIRED_ENV_VARS.get(backend)
    if env_var and not os.environ.get(env_var):
        raise EnvironmentError(
            f"Backend '{backend}' requires the {env_var} environment variable. "
            f"Set it with: export {env_var}=<your-key>"
        )
