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
        )
        results = runner.run(max_episodes=1)

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
        )
        runner.run(max_episodes=1)

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
        """Verify summary.json contains averaged metrics."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert "success_rate" in summary
        assert "avg_steps" in summary
        assert "num_episodes" in summary
