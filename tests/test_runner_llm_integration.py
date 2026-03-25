# tests/test_runner_llm_integration.py
"""Tests for EvaluationRunner LLM integration."""
import pytest


class TestResolveBackend:
    def test_backend_set_returns_backend(self):
        from easi.evaluation.runner import EvaluationRunner
        runner = EvaluationRunner("dummy_task", agent_type="react",
                                  backend="openai", model="gpt-4o")
        backend, url = runner._resolve_llm_backend()
        assert backend == "openai"

    def test_llm_url_without_backend_returns_legacy(self):
        from easi.evaluation.runner import EvaluationRunner
        runner = EvaluationRunner("dummy_task", agent_type="react",
                                  llm_base_url="http://localhost:8000")
        backend, url = runner._resolve_llm_backend()
        assert backend == "legacy"
        assert url == "http://localhost:8000"

    def test_neither_set_for_react_raises(self):
        from easi.evaluation.runner import EvaluationRunner
        runner = EvaluationRunner("dummy_task", agent_type="react")
        with pytest.raises(ValueError, match="requires --backend or --llm-url"):
            runner._resolve_llm_backend()

    def test_dummy_agent_returns_none(self):
        from easi.evaluation.runner import EvaluationRunner
        runner = EvaluationRunner("dummy_task", agent_type="dummy")
        backend, url = runner._resolve_llm_backend()
        assert backend is None

    def test_backend_vllm_with_llm_url_skips_server(self):
        from easi.evaluation.runner import EvaluationRunner
        runner = EvaluationRunner("dummy_task", agent_type="react",
                                  backend="vllm", model="test",
                                  llm_base_url="http://localhost:9090/v1")
        backend, url = runner._resolve_llm_backend()
        assert backend == "vllm"
        assert url == "http://localhost:9090/v1"


class TestAggregateLlmUsage:
    def test_aggregate_empty(self):
        from easi.evaluation.runner import EvaluationRunner
        result = EvaluationRunner._aggregate_llm_usage([])
        assert result["total_calls"] == 0
        assert result["total_tokens"] == 0

    def test_aggregate_single_episode(self):
        from easi.evaluation.runner import EvaluationRunner
        results = [
            {"llm_usage": {"num_calls": 4, "prompt_tokens": 800, "completion_tokens": 200, "cost_usd": 0.01}}
        ]
        agg = EvaluationRunner._aggregate_llm_usage(results)
        assert agg["total_calls"] == 4
        assert agg["total_prompt_tokens"] == 800
        assert agg["total_completion_tokens"] == 200
        assert agg["total_tokens"] == 1000
        assert agg["total_cost_usd"] == 0.01

    def test_aggregate_multiple_episodes(self):
        from easi.evaluation.runner import EvaluationRunner
        results = [
            {"llm_usage": {"num_calls": 3, "prompt_tokens": 500, "completion_tokens": 100, "cost_usd": 0.005}},
            {"llm_usage": {"num_calls": 5, "prompt_tokens": 700, "completion_tokens": 150, "cost_usd": 0.008}},
        ]
        agg = EvaluationRunner._aggregate_llm_usage(results)
        assert agg["total_calls"] == 8
        assert agg["total_prompt_tokens"] == 1200
        assert agg["total_completion_tokens"] == 250
        assert agg["total_tokens"] == 1450
        assert agg["avg_prompt_tokens_per_episode"] == 600
