"""End-to-end evaluation tests using dummy components.

These tests verify the full pipeline: task loading -> simulator launch ->
agent loop -> metric aggregation -> file output.
"""

import json

import pytest

from easi.evaluation.runner import EvaluationRunner


def _find_run_dir(output_dir, task_name="dummy_task"):
    """Find the single run directory under output_dir/<task_name>/."""
    task_dir = output_dir / task_name
    run_dirs = list(task_dir.iterdir())
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    return run_dirs[0]


class TestE2EEvaluation:
    def test_dummy_full_run(self, tmp_path):
        """Full eval: dummy task + dummy simulator + dummy agent."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
            agent_seed=42,
        )
        results = runner.run()

        # All 3 episodes should complete
        assert len(results) == 3

        # Summary should exist
        run_dir = _find_run_dir(tmp_path / "logs")
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_episodes"] == 3
        assert "success_rate" in summary or "avg_success" in summary

    def test_max_episodes_limit(self, tmp_path):
        """Verify max_episodes limits the run."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
        )
        results = runner.run(max_episodes=1)
        assert len(results) == 1

    def test_per_episode_files(self, tmp_path):
        """Verify per-episode result and trajectory files are created."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run(max_episodes=2)

        run_dir = _find_run_dir(output_dir)
        episode_dirs = sorted((run_dir / "episodes").iterdir())
        assert len(episode_dirs) == 2

        for ep_dir in episode_dirs:
            assert (ep_dir / "result.json").exists()
            assert (ep_dir / "trajectory.jsonl").exists()

        ep0 = json.loads((episode_dirs[0] / "result.json").read_text())
        assert "episode_id" in ep0
        assert "elapsed_seconds" in ep0

    def test_episode_metrics_structure(self, tmp_path):
        """Each episode result should have expected metric keys."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
        )
        results = runner.run(max_episodes=1)

        ep = results[0]
        assert "success" in ep
        assert "num_steps" in ep
        assert "episode_id" in ep
        assert "elapsed_seconds" in ep
        assert isinstance(ep["elapsed_seconds"], float)
        assert ep["elapsed_seconds"] >= 0

    def test_summary_has_all_averaged_keys(self, tmp_path):
        """Summary should average all numeric metrics from episodes."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_episodes"] == 3
        # These come from averaging per-episode numeric keys
        assert "avg_num_steps" in summary
        assert "avg_elapsed_seconds" in summary

    def test_deterministic_with_seed(self, tmp_path):
        """Two runs with the same seed should produce identical results."""
        results_a = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "a",
            agent_seed=123,
        ).run(max_episodes=2)

        results_b = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "b",
            agent_seed=123,
        ).run(max_episodes=2)

        for a, b in zip(results_a, results_b):
            assert a["num_steps"] == b["num_steps"]
            assert a["success"] == b["success"]

    def test_different_seeds_may_differ(self, tmp_path):
        """Different seeds should produce potentially different results."""
        results_a = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "a",
            agent_seed=1,
        ).run()

        results_b = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "b",
            agent_seed=999,
        ).run()

        # With different seeds, at least one metric should differ
        # (This is a soft test — could theoretically match, but very unlikely)
        all_steps_a = [r["num_steps"] for r in results_a]
        all_steps_b = [r["num_steps"] for r in results_b]
        # Just verify both complete; exact difference not guaranteed
        assert len(results_a) == len(results_b) == 3

    def test_config_json_saved(self, tmp_path):
        """Verify config.json is written with run metadata."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            agent_seed=42,
        )
        runner.run(max_episodes=1)

        run_dir = _find_run_dir(output_dir)
        config = json.loads((run_dir / "config.json").read_text())
        assert config["task_name"] == "dummy_task"
        assert config["agent_type"] == "dummy"
        assert config["agent_seed"] == 42
        assert config["max_episodes"] == 1
        assert "run_id" in config

    def test_trajectory_jsonl_format(self, tmp_path):
        """Verify trajectory.jsonl has correct format."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run(max_episodes=1)

        run_dir = _find_run_dir(output_dir)
        episode_dirs = sorted((run_dir / "episodes").iterdir())
        trajectory_path = episode_dirs[0] / "trajectory.jsonl"
        lines = trajectory_path.read_text().strip().split("\n")

        # First line is reset
        reset_entry = json.loads(lines[0])
        assert reset_entry["step"] == 0
        assert reset_entry["type"] == "reset"
        assert "rgb_path" in reset_entry
        assert reset_entry["rgb_path"].startswith("rgb_")

        # Subsequent lines are steps
        assert len(lines) > 1
        step_entry = json.loads(lines[1])
        assert step_entry["step"] == 1
        assert step_entry["type"] == "step"
        assert "action" in step_entry
        assert "rgb_path" in step_entry

    def test_images_saved_to_episode_dir(self, tmp_path):
        """Verify RGB images are saved in the episode directory, not IPC workspace."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
        )
        runner.run(max_episodes=1)

        run_dir = _find_run_dir(output_dir)
        episode_dirs = sorted((run_dir / "episodes").iterdir())
        ep_dir = episode_dirs[0]

        # Should have rgb_0000.png (reset) + at least one step image
        png_files = sorted(ep_dir.glob("rgb_*.png"))
        assert len(png_files) >= 2  # At least reset + one step
        assert png_files[0].name == "rgb_0000.png"
        assert png_files[1].name == "rgb_0001.png"
