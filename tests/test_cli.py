# tests/test_cli.py
"""Tests for CLI GPU and vLLM arguments."""


def test_cli_parses_vllm_gpu_args():
    """CLI should parse --vllm-instances, --vllm-gpus, --sim-gpus."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "start", "dummy_task",
        "--agent", "dummy",
        "--backend", "vllm",
        "--model", "test",
        "--num-parallel", "12",
        "--vllm-instances", "2",
        "--vllm-gpus", "0,1",
        "--sim-gpus", "2,3",
    ])
    assert args.vllm_instances == 2
    assert args.vllm_gpus == "0,1"
    assert args.sim_gpus == "2,3"


def test_cli_parses_comma_separated_llm_url():
    """--llm-url should accept comma-separated URLs."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "start", "dummy_task",
        "--agent", "dummy",
        "--backend", "vllm",
        "--model", "test",
        "--llm-url", "http://localhost:8000/v1,http://localhost:8001/v1",
    ])
    assert args.llm_base_url == "http://localhost:8000/v1,http://localhost:8001/v1"
