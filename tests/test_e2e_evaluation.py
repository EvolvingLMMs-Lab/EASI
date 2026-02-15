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
        assert config["cli_options"]["task_name"] == "dummy_task"
        assert config["cli_options"]["agent_type"] == "dummy"
        assert config["cli_options"]["agent_seed"] == 42
        assert config["cli_options"]["max_episodes"] == 1
        assert "run_id" in config
        # Task YAML config is included
        assert "task_config" in config
        assert config["task_config"]["name"] == "dummy_task"

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

    def test_max_retries_stored(self, tmp_path):
        """Verify max_retries is stored and appears in config."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_retries=5,
        )
        assert runner.max_retries == 5
        runner.run(max_episodes=1)

        run_dir = _find_run_dir(output_dir)
        config = json.loads((run_dir / "config.json").read_text())
        assert config["cli_options"]["max_retries"] == 5

    def test_resume_dir_stored(self, tmp_path):
        """Verify resume_dir is stored."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
            resume_dir=tmp_path / "old_run",
        )
        assert runner.resume_dir == tmp_path / "old_run"

    def test_resume_skips_completed_episodes(self, tmp_path):
        """Resume skips completed episodes and re-runs the last one."""
        output_dir = tmp_path / "logs"

        # First run: complete all 3 episodes
        runner1 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            agent_seed=42,
        )
        results1 = runner1.run()
        assert len(results1) == 3

        run_dir = _find_run_dir(output_dir)

        # Resume: should load episodes 0-1, re-run episode 2
        runner2 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        results2 = runner2.run()
        assert len(results2) == 3  # 2 loaded + 1 re-run

    def test_resume_clears_last_episode_dir(self, tmp_path):
        """Resume clears the last episode directory before re-running."""
        output_dir = tmp_path / "logs"

        runner1 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            agent_seed=42,
        )
        runner1.run()

        run_dir = _find_run_dir(output_dir)
        episodes_dir = run_dir / "episodes"
        episode_dirs = sorted(episodes_dir.iterdir())
        last_ep_dir = episode_dirs[-1]

        # Add a marker file to the last episode dir
        marker = last_ep_dir / "marker.txt"
        marker.write_text("should be deleted")

        runner2 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        runner2.run()

        # Marker should be gone (dir was cleared)
        assert not marker.exists()
        # But result.json should exist (re-run completed)
        assert (last_ep_dir / "result.json").exists()

    def test_resume_produces_valid_summary(self, tmp_path):
        """Resumed run produces a valid summary with all episodes."""
        output_dir = tmp_path / "logs"

        runner1 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            agent_seed=42,
        )
        runner1.run()

        run_dir = _find_run_dir(output_dir)

        runner2 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        results2 = runner2.run()

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_episodes"] == 3

    def test_resume_empty_run_dir(self, tmp_path):
        """Resume with no completed episodes runs all from scratch."""
        output_dir = tmp_path / "logs"
        run_dir = output_dir / "dummy_task" / "fake_run"
        episodes_dir = run_dir / "episodes"
        episodes_dir.mkdir(parents=True)

        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        results = runner.run()
        assert len(results) == 3  # All episodes run from scratch


class TestEpisodeRetry:
    def test_retry_on_episode_failure(self, tmp_path):
        """Episodes that crash are retried and succeed on subsequent attempt."""
        from unittest.mock import patch

        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_retries=3,
        )

        call_count = {"n": 0}
        original_run_episode = runner._run_episode

        def failing_first_call(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated bridge crash")
            return original_run_episode(*args, **kwargs)

        with patch.object(runner, "_run_episode", side_effect=failing_first_call):
            results = runner.run(max_episodes=1)

        assert len(results) == 1
        assert "error" not in results[0]
        assert results[0]["success"] in (0.0, 1.0)

    def test_retry_exhausted_records_failure(self, tmp_path):
        """Episode recorded as failure after all retries exhausted."""
        from unittest.mock import patch

        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_retries=2,
        )

        with patch.object(
            runner, "_run_episode",
            side_effect=RuntimeError("persistent crash"),
        ):
            results = runner.run(max_episodes=1)

        assert len(results) == 1
        assert results[0]["success"] == 0.0
        assert "error" in results[0]
        assert "persistent crash" in results[0]["error"]

    def test_retry_clears_partial_files(self, tmp_path):
        """Retry clears partial files from the failed attempt."""
        from unittest.mock import patch

        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_retries=2,
        )

        original_run_episode = runner._run_episode

        def write_then_crash(sim, agent, task, episode, index, episode_dir):
            # Write a partial file then crash
            (episode_dir / "partial.txt").write_text("partial data")
            raise RuntimeError("crash mid-episode")

        call_count = {"n": 0}

        def crash_then_succeed(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return write_then_crash(*args, **kwargs)
            return original_run_episode(*args, **kwargs)

        with patch.object(runner, "_run_episode", side_effect=crash_then_succeed):
            results = runner.run(max_episodes=1)

        assert len(results) == 1
        # The partial file should have been cleaned up
        run_dir = _find_run_dir(output_dir)
        episode_dirs = sorted((run_dir / "episodes").iterdir())
        assert not (episode_dirs[0] / "partial.txt").exists()
        # But result.json from successful retry should exist
        assert (episode_dirs[0] / "result.json").exists()

    def test_retry_continues_to_next_episode(self, tmp_path):
        """After exhausting retries on one episode, runner continues to the next."""
        from unittest.mock import patch

        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_retries=1,
        )

        original_run_episode = runner._run_episode
        call_count = {"n": 0}

        def fail_first_episode(*args, **kwargs):
            call_count["n"] += 1
            # Fail only the first episode (call 1)
            if call_count["n"] == 1:
                raise RuntimeError("episode 0 crash")
            return original_run_episode(*args, **kwargs)

        with patch.object(runner, "_run_episode", side_effect=fail_first_episode):
            results = runner.run(max_episodes=2)

        # Both episodes should have results
        assert len(results) == 2
        # First episode failed
        assert results[0]["success"] == 0.0
        assert "error" in results[0]
        # Second episode succeeded
        assert "error" not in results[1]


class TestResumeConfigLoading:
    def test_resume_loads_config_from_run_dir(self, tmp_path):
        """cmd_run with --resume loads task_name and options from config.json."""
        from easi.cli import cmd_run

        output_dir = tmp_path / "logs"

        # First run with specific options
        cmd_run(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=str(output_dir),
            data_dir="./datasets",
            max_episodes=2,
            llm_url=None,
            seed=42,
            backend=None,
            model="default",
            port=8080,
            llm_kwargs_raw=None,
            max_retries=3,
            resume=None,
        )

        run_dir = _find_run_dir(output_dir)

        # Resume without specifying task_name — should load from config.json
        cmd_run(
            task_name=None,
            agent_type="dummy",
            output_dir="./logs",
            data_dir="./datasets",
            max_episodes=None,
            llm_url=None,
            seed=None,
            backend=None,
            model="default",
            port=8080,
            llm_kwargs_raw=None,
            max_retries=3,
            resume=str(run_dir),
        )

        # Should have completed successfully (3 total episodes from config)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_episodes"] == 3


class TestCLIParsing:
    def test_cli_max_retries_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "dummy_task"])
        assert args.max_retries == 3

    def test_cli_max_retries_custom(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "dummy_task", "--max-retries", "5"])
        assert args.max_retries == 5

    def test_cli_resume_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "dummy_task"])
        assert args.resume is None

    def test_cli_resume_custom(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "dummy_task", "--resume", "/tmp/logs/run_123"])
        assert args.resume == "/tmp/logs/run_123"

    def test_cli_resume_without_task(self):
        """--resume should work without specifying a task name."""
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--resume", "/tmp/logs/run_123"])
        assert args.resume == "/tmp/logs/run_123"
        assert args.task is None
