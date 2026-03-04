"""Tests for the sequential evaluation runner."""

import json

import pytest

from easi.evaluation.runner import EvaluationRunner


def _find_run_dir(output_dir, task_name="dummy_task"):
    """Find the single run directory under output_dir/<task_name>/."""
    task_dir = output_dir / task_name
    run_dirs = list(task_dir.iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    return run_dirs[0]


class TestEvaluationRunner:
    def test_run_single_episode(self, tmp_path):
        """Run one episode with dummy task + dummy simulator + dummy agent."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
            max_episodes=1,
        )
        results = runner.run()

        assert len(results) == 1
        assert "success" in results[0]
        assert "num_steps" in results[0]

    def test_run_multiple_episodes(self, tmp_path):
        """Run all 3 dummy episodes."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
        )
        results = runner.run()

        assert len(results) == 3  # dummy_task has 3 episodes

    def test_results_saved_to_disk(self, tmp_path):
        """Verify structured output directory is created."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_episodes=1,
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "summary.json").exists()

        # Episode directory
        episodes_dir = run_dir / "episodes"
        assert episodes_dir.exists()
        episode_dirs = sorted(episodes_dir.iterdir())
        assert len(episode_dirs) == 1

        ep_dir = episode_dirs[0]
        assert (ep_dir / "result.json").exists()
        assert (ep_dir / "trajectory.jsonl").exists()
        assert (ep_dir / "rgb_0000.png").exists()  # Reset observation

    def test_summary_aggregates_metrics(self, tmp_path):
        """Verify summary.json contains averaged metrics nested under 'metrics'."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert "num_episodes" in summary
        assert "metrics" in summary
        assert "success_rate" in summary["metrics"]
        assert "avg_steps" in summary["metrics"]


class TestCliOptionsCapture:
    """Tests for auto-captured _cli_options via inspect."""

    def test_cli_options_contains_all_init_args(self):
        runner = EvaluationRunner(task_name="dummy_task")
        opts = runner._cli_options
        # All non-excluded init params should be present
        expected_keys = {
            "task_name", "agent_type", "output_dir", "data_dir",
            "max_episodes", "llm_base_url", "agent_seed", "backend",
            "model", "port", "llm_kwargs_raw", "max_retries",
            "render_platform",
            "vllm_instances", "vllm_gpus", "sim_gpus",
        }
        assert set(opts.keys()) == expected_keys

    def test_cli_options_excludes_session_params(self):
        runner = EvaluationRunner(
            task_name="dummy_task",
            resume_dir="/some/path",
            refresh_data=True,
        )
        assert "resume_dir" not in runner._cli_options
        assert "refresh_data" not in runner._cli_options

    def test_cli_options_captures_raw_values(self):
        """Values are captured before Path conversion."""
        runner = EvaluationRunner(
            task_name="test_task",
            agent_type="react",
            output_dir="/custom/output",
            data_dir="/custom/data",
            backend="openai",
            model="gpt-4o",
            port=9090,
            max_episodes=5,
            max_retries=2,
        )
        opts = runner._cli_options
        assert opts["task_name"] == "test_task"
        assert opts["agent_type"] == "react"
        assert opts["output_dir"] == "/custom/output"
        assert opts["data_dir"] == "/custom/data"
        assert opts["backend"] == "openai"
        assert opts["model"] == "gpt-4o"
        assert opts["port"] == 9090
        assert opts["max_episodes"] == 5
        assert opts["max_retries"] == 2

    def test_cli_options_defaults(self):
        """Default values are captured correctly."""
        runner = EvaluationRunner(task_name="dummy_task")
        opts = runner._cli_options
        assert opts["agent_type"] == "react"
        assert opts["output_dir"] == "./logs"
        assert opts["data_dir"] == "./datasets"
        assert opts["model"] == "default"
        assert opts["port"] == 8080
        assert opts["max_retries"] == 3
        assert opts["max_episodes"] is None
        assert opts["llm_base_url"] is None
        assert opts["agent_seed"] is None
        assert opts["backend"] is None
        assert opts["llm_kwargs_raw"] is None

    def test_serialize_cli_options_converts_paths(self):
        """Path objects in _cli_options are serialized to strings."""
        from pathlib import Path
        runner = EvaluationRunner(
            task_name="dummy_task",
            output_dir=Path("/some/path"),
        )
        serialized = runner._serialize_cli_options()
        assert isinstance(serialized["output_dir"], str)
        assert serialized["output_dir"] == "/some/path"

    def test_config_json_uses_cli_options(self, tmp_path):
        """config.json cli_options matches _serialize_cli_options()."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_episodes=1,
            model="test-model",
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)
        config = json.loads((run_dir / "config.json").read_text())
        opts = config["cli_options"]
        assert opts["task_name"] == "dummy_task"
        assert opts["agent_type"] == "dummy"
        assert opts["model"] == "test-model"
        assert opts["max_episodes"] == 1
        # Session-specific params should NOT be in config
        assert "resume_dir" not in opts
        assert "redownload" not in opts
