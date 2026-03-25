# tests/test_cli_llm_args.py
"""Tests for CLI LLM-related arguments."""
import pytest


class TestBuildParser:
    def test_run_has_backend_arg(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--backend", "vllm"])
        assert args.backend == "vllm"

    def test_run_backend_default_is_none(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.backend is None

    def test_run_has_model_arg(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_run_model_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.model is None

    def test_run_has_port_arg(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--port", "9090"])
        assert args.port == 9090

    def test_run_port_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.port is None

    def test_run_has_llm_kwargs(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "start", "dummy_task",
            "--llm-kwargs", '{"tensor_parallel_size": 4}',
        ])
        assert args.llm_kwargs_raw == '{"tensor_parallel_size": 4}'

    def test_run_llm_kwargs_default_none(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.llm_kwargs_raw is None

    def test_backward_compat_llm_url_still_works(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--llm-url", "http://localhost:8000"])
        assert args.llm_base_url == "http://localhost:8000"
