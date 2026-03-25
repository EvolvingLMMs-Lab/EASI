# tests/test_cli.py
"""Tests for CLI GPU and LLM arguments."""

from unittest.mock import MagicMock, patch

import pytest


def test_cli_parses_llm_gpu_args():
    """CLI should parse --llm-instances, --llm-gpus, --sim-gpus."""
    from easi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "start",
            "dummy_task",
            "--agent",
            "dummy",
            "--backend",
            "vllm",
            "--model",
            "test",
            "--num-parallel",
            "12",
            "--llm-instances",
            "2",
            "--llm-gpus",
            "0,1",
            "--sim-gpus",
            "2,3",
        ]
    )
    assert args.llm_instances == 2
    assert args.llm_gpus == "0,1"
    assert args.sim_gpus == "2,3"


def test_cli_parses_comma_separated_llm_url():
    """--llm-url should accept comma-separated URLs."""
    from easi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "start",
            "dummy_task",
            "--agent",
            "dummy",
            "--backend",
            "vllm",
            "--model",
            "test",
            "--llm-url",
            "http://localhost:8000/v1,http://localhost:8001/v1",
        ]
    )
    assert args.llm_base_url == "http://localhost:8000/v1,http://localhost:8001/v1"


def test_cli_ps_subcommand():
    from easi.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["ps"])
    assert args.command == "ps"
    assert args.kill is False

    args_kill = parser.parse_args(["ps", "--kill"])
    assert args_kill.kill is True


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


def test_cmd_sim_test_xorg_setup_warning_exits_cleanly():
    from easi.cli import cmd_sim_test

    entry = MagicMock(runtime="conda")
    env_manager = MagicMock()
    env_manager.default_render_platform = "xorg"
    env_manager.supported_render_platforms = ["xorg"]

    sim = MagicMock()
    render_platform = MagicMock()
    render_platform.setup.side_effect = RuntimeError("xorg guidance")

    with (
        patch("easi.simulators.registry.get_simulator_entry", return_value=entry),
        patch("easi.simulators.registry.create_env_manager", return_value=env_manager),
        patch(
            "easi.simulators.registry.load_simulator_class", return_value=lambda: sim
        ),
        patch(
            "easi.simulators.registry.resolve_render_platform",
            return_value=render_platform,
        ),
        patch("easi.cli.logger") as mock_logger,
    ):
        with pytest.raises(SystemExit) as exc_info:
            cmd_sim_test(
                "coppeliasim", steps=1, timeout=5.0, render_platform_name="xorg"
            )

    assert exc_info.value.code == 0
    mock_logger.warning.assert_called_once_with("%s", "xorg guidance")
    render_platform.teardown.assert_called_once()
    sim.close.assert_not_called()
