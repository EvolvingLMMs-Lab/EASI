# tests/test_cli.py
"""Tests for CLI GPU and LLM arguments."""


def test_cli_parses_llm_gpu_args():
    """CLI should parse --llm-instances, --llm-gpus, --sim-gpus."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "start", "dummy_task",
        "--agent", "dummy",
        "--backend", "vllm",
        "--model", "test",
        "--num-parallel", "12",
        "--llm-instances", "2",
        "--llm-gpus", "0,1",
        "--sim-gpus", "2,3",
    ])
    assert args.llm_instances == 2
    assert args.llm_gpus == "0,1"
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


def test_cli_model_list_subcommand():
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["model", "list"])
    assert args.command == "model"
    assert args.model_action == "list"


def test_cli_model_info_subcommand():
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["model", "info", "cambrian"])
    assert args.command == "model"
    assert args.model_action == "info"
    assert args.model_name == "cambrian"
