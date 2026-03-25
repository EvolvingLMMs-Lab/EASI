"""Tests for LLMClient."""
import json
import pytest
from unittest.mock import MagicMock, patch


class TestLLMClientInit:
    def test_stores_config(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/gpt-4o", temperature=0.5, max_tokens=1024)
        assert client.model == "openai/gpt-4o"
        assert client.base_url is None
        assert client.default_kwargs == {"temperature": 0.5, "max_tokens": 1024}

    def test_stores_base_url(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/test", base_url="http://localhost:8080/v1")
        assert client.base_url == "http://localhost:8080/v1"


class TestLLMClientUsageTracking:
    def test_initial_usage_is_zero(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/test")
        usage = client.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["num_calls"] == 0
        assert usage["cost_usd"] == 0.0

    def test_reset_usage(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/test")
        client._usage["prompt_tokens"] = 100
        client._usage["num_calls"] = 5
        client.reset_usage()
        assert client.get_usage()["prompt_tokens"] == 0
        assert client.get_usage()["num_calls"] == 0

    def test_get_usage_returns_copy(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/test")
        usage = client.get_usage()
        usage["prompt_tokens"] = 999
        assert client.get_usage()["prompt_tokens"] == 0


class TestLLMClientGenerate:
    @patch("easi.llm.client.litellm")
    def test_generate_returns_content(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = '{"executable_plan": [{"action": "Stop"}]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 20
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.001

        client = LLMClient(model="openai/gpt-4o")
        result = client.generate([{"role": "user", "content": "test"}])

        assert result == '{"executable_plan": [{"action": "Stop"}]}'
        mock_litellm.completion.assert_called_once()

    @patch("easi.llm.client.litellm")
    def test_generate_tracks_usage(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "test"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.002

        client = LLMClient(model="openai/gpt-4o")
        client.generate([{"role": "user", "content": "hello"}])
        client.generate([{"role": "user", "content": "world"}])

        usage = client.get_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 20
        assert usage["num_calls"] == 2
        assert usage["cost_usd"] == pytest.approx(0.004)

    @patch("easi.llm.client.litellm")
    def test_generate_passes_base_url(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/test", base_url="http://localhost:8080/v1")
        client.generate([{"role": "user", "content": "hi"}])

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_base") == "http://localhost:8080/v1"

    @patch("easi.llm.client.litellm")
    def test_generate_passes_response_format(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = '{"executable_plan": [{"action": "Stop"}]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        rf = {"type": "json_schema", "json_schema": {"name": "test", "schema": {"type": "object"}}}
        client = LLMClient(model="openai/gpt-4o")
        client.generate([{"role": "user", "content": "test"}], response_format=rf)

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("response_format") == rf

    @patch("easi.llm.client.litellm")
    def test_generate_omits_response_format_when_none(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/gpt-4o")
        client.generate([{"role": "user", "content": "test"}])

        call_kwargs = mock_litellm.completion.call_args
        assert "response_format" not in call_kwargs.kwargs


class TestLLMClientRetries:
    def test_default_num_retries(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/gpt-4o")
        assert client.num_retries == 3

    def test_custom_num_retries(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/gpt-4o", num_retries=5)
        assert client.num_retries == 5

    @patch("easi.llm.client.litellm")
    def test_num_retries_passed_to_litellm(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/gpt-4o", num_retries=7)
        client.generate([{"role": "user", "content": "hi"}])

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("num_retries") == 7

    @patch("easi.llm.client.litellm")
    def test_zero_retries_passed_through(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/gpt-4o", num_retries=0)
        client.generate([{"role": "user", "content": "hi"}])

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("num_retries") == 0


class TestLLMClientProtocolCompat:
    def test_satisfies_llm_client_protocol(self):
        from easi.core.protocols import LLMClientProtocol
        from easi.llm.client import LLMClient

        client = LLMClient(model="openai/test")
        assert isinstance(client, LLMClientProtocol)
