"""Tests for LLM utility functions."""
import pytest


class TestParseLlmKwargs:
    def test_none_returns_empty_dict(self):
        from easi.llm.utils import parse_llm_kwargs
        assert parse_llm_kwargs(None) == {}

    def test_empty_string_returns_empty_dict(self):
        from easi.llm.utils import parse_llm_kwargs
        assert parse_llm_kwargs("") == {}

    def test_valid_json(self):
        from easi.llm.utils import parse_llm_kwargs
        result = parse_llm_kwargs('{"tensor_parallel_size": 4, "temperature": 0.7}')
        assert result == {"tensor_parallel_size": 4, "temperature": 0.7}

    def test_nested_json(self):
        from easi.llm.utils import parse_llm_kwargs
        result = parse_llm_kwargs('{"extra": {"key": "val"}}')
        assert result == {"extra": {"key": "val"}}

    def test_invalid_json_raises(self):
        from easi.llm.utils import parse_llm_kwargs
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_llm_kwargs("not json")


class TestSplitKwargs:
    def test_splits_server_and_client(self):
        from easi.llm.utils import split_kwargs
        kwargs = {
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        server, client = split_kwargs(kwargs)
        assert server == {"tensor_parallel_size": 4, "gpu_memory_utilization": 0.9}
        assert client == {"temperature": 0.7, "max_tokens": 1024}

    def test_empty_dict(self):
        from easi.llm.utils import split_kwargs
        server, client = split_kwargs({})
        assert server == {}
        assert client == {}

    def test_all_server_kwargs(self):
        from easi.llm.utils import split_kwargs
        kwargs = {"tensor_parallel_size": 4, "max_model_len": 8192}
        server, client = split_kwargs(kwargs)
        assert server == kwargs
        assert client == {}

    def test_all_client_kwargs(self):
        from easi.llm.utils import split_kwargs
        kwargs = {"temperature": 0.5, "top_p": 0.9}
        server, client = split_kwargs(kwargs)
        assert server == {}
        assert client == kwargs


class TestBuildLitellmModel:
    def test_openai(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("openai", "gpt-4o") == "openai/gpt-4o"

    def test_anthropic(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("anthropic", "claude-sonnet-4-5-20250929") == "anthropic/claude-sonnet-4-5-20250929"

    def test_gemini(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("gemini", "gemini-2.0-flash") == "gemini/gemini-2.0-flash"

    def test_vllm_uses_openai_prefix(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("vllm", "Qwen/Qwen2.5-VL-72B") == "openai/Qwen/Qwen2.5-VL-72B"

    def test_dummy_uses_openai_prefix(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("dummy", "default") == "openai/default"

    def test_custom_uses_openai_prefix(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("custom", "my_model") == "openai/my_model"

    def test_unknown_backend_falls_back(self):
        from easi.llm.utils import build_litellm_model
        assert build_litellm_model("lmdeploy", "some-model") == "openai/some-model"


class TestValidateBackend:
    def test_openai_missing_key(self, monkeypatch):
        from easi.llm.utils import validate_backend
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            validate_backend("openai")

    def test_openai_with_key(self, monkeypatch):
        from easi.llm.utils import validate_backend
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_backend("openai")  # should not raise

    def test_anthropic_missing_key(self, monkeypatch):
        from easi.llm.utils import validate_backend
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            validate_backend("anthropic")

    def test_gemini_missing_key(self, monkeypatch):
        from easi.llm.utils import validate_backend
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            validate_backend("gemini")

    def test_vllm_no_key_needed(self):
        from easi.llm.utils import validate_backend
        validate_backend("vllm")  # should not raise

    def test_dummy_no_key_needed(self):
        from easi.llm.utils import validate_backend
        validate_backend("dummy")  # should not raise
