"""Full integration test for LLM inference pipeline."""
import json
import pytest
from unittest.mock import MagicMock, patch


class TestFullFlow:
    """Test: CLI args -> parse kwargs -> build model -> create client -> generate -> track usage."""

    def test_parse_kwargs_to_split_to_client(self):
        from easi.llm.utils import parse_llm_kwargs, split_kwargs, build_litellm_model

        raw = '{"tensor_parallel_size": 4, "temperature": 0.7, "max_tokens": 1024}'
        kwargs = parse_llm_kwargs(raw)
        server_kw, client_kw = split_kwargs(kwargs)
        model = build_litellm_model("vllm", "Qwen/Qwen2.5-VL-72B")

        assert server_kw == {"tensor_parallel_size": 4}
        assert client_kw == {"temperature": 0.7, "max_tokens": 1024}
        assert model == "openai/Qwen/Qwen2.5-VL-72B"

    @patch("easi.llm.client.litellm")
    def test_client_generate_and_track(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = '{"executable_plan": [{"action": "Stop"}]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 30
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.005

        client = LLMClient(model="openai/gpt-4o", temperature=0.7)

        # Simulate 3 LLM calls (one episode)
        for _ in range(3):
            client.generate([{"role": "user", "content": "test"}])

        usage = client.get_usage()
        assert usage["num_calls"] == 3
        assert usage["prompt_tokens"] == 600
        assert usage["completion_tokens"] == 90
        assert usage["cost_usd"] == pytest.approx(0.015)

        # Reset for next episode
        client.reset_usage()
        assert client.get_usage()["num_calls"] == 0

