"""Backend adapter factory."""
from __future__ import annotations

from .base import BackendAdapter, BenchmarkScores, ExtractionReport


def get_backend(name: str, **kwargs) -> BackendAdapter:
    """Create a backend adapter by name.

    Args:
        name: "vlmevalkit" or "lmms-eval"
        **kwargs: Backend-specific options (e.g., model_args, use_accelerate)
    """
    if name == "vlmevalkit":
        from .vlmevalkit import VLMEvalKitAdapter
        return VLMEvalKitAdapter(**kwargs)
    elif name == "lmms-eval":
        from .lmmseval import LmmsEvalAdapter
        return LmmsEvalAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'vlmevalkit' or 'lmms-eval'.")


__all__ = ["get_backend", "BackendAdapter", "BenchmarkScores", "ExtractionReport"]
