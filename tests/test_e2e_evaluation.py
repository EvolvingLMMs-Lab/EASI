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
        assert "metrics" in summary
        assert "success_rate" in summary["metrics"] or "avg_success" in summary["metrics"]

    def test_max_episodes_limit(self, tmp_path):
        """Verify max_episodes limits the run."""
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "logs",
            max_episodes=1,
        )
        results = runner.run()
        assert len(results) == 1

    def test_per_episode_files(self, tmp_path):
        """Verify per-episode result and trajectory files are created."""
        output_dir = tmp_path / "logs"
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_episodes=2,
        )
        runner.run()

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
            max_episodes=1,
        )
        results = runner.run()

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
        # These come from averaging per-episode numeric keys (nested under "metrics")
        assert "metrics" in summary
        assert "avg_num_steps" in summary["metrics"]
        assert "avg_elapsed_seconds" in summary["metrics"]

    def test_deterministic_with_seed(self, tmp_path):
        """Two runs with the same seed should produce identical results."""
        results_a = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "a",
            agent_seed=123,
            max_episodes=2,
        ).run()

        results_b = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=tmp_path / "b",
            agent_seed=123,
            max_episodes=2,
        ).run()

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
            max_episodes=1,
        )
        runner.run()

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
            max_episodes=1,
        )
        runner.run()

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
            max_episodes=1,
        )
        runner.run()

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
            max_episodes=1,
        )
        assert runner.max_retries == 5
        runner.run()

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
        """Resume with all episodes complete re-aggregates but doesn't re-run."""
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

        # Resume: all episodes already complete — should load all, re-run none
        runner2 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        results2 = runner2.run()
        assert len(results2) == 3  # All loaded, none re-run

    def test_resume_clears_from_first_incomplete(self, tmp_path):
        """Resume clears directories from the first incomplete episode onward."""
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

        # Simulate incomplete episode 1 by removing its result.json
        (episode_dirs[1] / "result.json").unlink()

        runner2 = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            resume_dir=run_dir,
            agent_seed=42,
        )
        results2 = runner2.run()
        assert len(results2) == 3  # 1 loaded + 2 re-run

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
            max_episodes=1,
        )

        call_count = {"n": 0}
        original_run_episode = runner._run_episode

        def failing_first_call(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated bridge crash")
            return original_run_episode(*args, **kwargs)

        with patch.object(runner, "_run_episode", side_effect=failing_first_call):
            results = runner.run()

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
            max_episodes=1,
        )

        with patch.object(
            runner, "_run_episode",
            side_effect=RuntimeError("persistent crash"),
        ):
            results = runner.run()

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
            max_episodes=1,
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
            results = runner.run()

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
            max_episodes=2,
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
            results = runner.run()

        # Both episodes should have results
        assert len(results) == 2
        # First episode failed
        assert results[0]["success"] == 0.0
        assert "error" in results[0]
        # Second episode succeeded
        assert "error" not in results[1]


class TestResumeConfigLoading:
    def test_resume_loads_config_from_run_dir(self, tmp_path):
        """cmd_start with --resume loads task_name and options from config.json."""
        from argparse import Namespace
        from easi.cli import cmd_start

        output_dir = tmp_path / "logs"

        # First run — only 2 of 3 episodes
        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=output_dir,
            max_episodes=2,
            agent_seed=42,
        )
        runner.run()

        run_dir = _find_run_dir(output_dir)

        # Resume with higher max_episodes to complete remaining
        # Saved config has max_episodes=2; override to 3 to run all.
        args = Namespace(
            command="start",
            verbosity="INFO",
            task_names_positional=[],
            tasks_csv=None,
            agent_type="dummy",
            output_dir=None,
            data_dir=None,
            max_episodes=3,
            llm_base_url=None,
            agent_seed=None,
            backend=None,
            model=None,
            port=None,
            llm_kwargs_raw=None,
            max_retries=None,
            resume_dir=str(run_dir),
            refresh_data=False,
        )
        cmd_start(args)

        # Should have completed all 3 episodes (2 from first run + 1 from resume)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["num_episodes"] == 3


class TestCLIParsing:
    def test_cli_single_task_positional(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.task_names_positional == ["dummy_task"]
        assert args.tasks_csv is None

    def test_cli_multiple_tasks_positional(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "ebalfred_base"])
        assert args.task_names_positional == ["dummy_task", "ebalfred_base"]

    def test_cli_tasks_flag_csv(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "--tasks", "dummy_task,ebalfred_base"])
        assert args.tasks_csv == "dummy_task,ebalfred_base"

    def test_cli_tasks_flag_wins_over_positional(self):
        from easi.cli import _resolve_task_list
        from argparse import Namespace
        args = Namespace(
            task_names_positional=["dummy_task"],
            tasks_csv="ebalfred_base,ebnavigation_base",
        )
        assert _resolve_task_list(args) == ["ebalfred_base", "ebnavigation_base"]

    def test_cli_resume_without_task(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "--resume", "/tmp/logs/run_123"])
        assert args.resume_dir == "/tmp/logs/run_123"
        assert args.task_names_positional == []

    def test_cli_all_defaults_are_none(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.agent_type is None
        assert args.output_dir is None
        assert args.data_dir is None
        assert args.model is None
        assert args.port is None
        assert args.max_retries is None
        assert args.llm_base_url is None
        assert args.agent_seed is None
        assert args.llm_kwargs_raw is None

    def test_cli_max_retries_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.max_retries is None

    def test_cli_max_retries_custom(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--max-retries", "5"])
        assert args.max_retries == 5

    def test_cli_resume_default(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.resume_dir is None

    def test_cli_resume_custom(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--resume", "/tmp/logs/run_123"])
        assert args.resume_dir == "/tmp/logs/run_123"


class TestMultiTaskRun:
    def test_sequential_single_task_via_flag(self, tmp_path):
        """Run a single task via --tasks flag."""
        from argparse import Namespace
        from easi.cli import cmd_start

        output_dir = tmp_path / "logs"
        args = Namespace(
            command="start",
            verbosity="INFO",
            task_names_positional=[],
            tasks_csv="dummy_task",
            agent_type="dummy",
            output_dir=str(output_dir),
            data_dir=None,
            max_episodes=1,
            llm_base_url=None,
            agent_seed=None,
            backend=None,
            model=None,
            port=None,
            llm_kwargs_raw=None,
            max_retries=None,
            resume_dir=None,
            refresh_data=False,
        )
        cmd_start(args)

        task_dir = output_dir / "dummy_task"
        assert task_dir.exists()
        run_dirs = list(task_dir.iterdir())
        assert len(run_dirs) == 1

    def test_resume_blocked_with_multi_task(self):
        """--resume with multiple tasks should fail."""
        from argparse import Namespace
        from easi.cli import cmd_start

        args = Namespace(
            command="start",
            verbosity="INFO",
            task_names_positional=[],
            tasks_csv="dummy_task,ebalfred_base",
            agent_type="dummy",
            output_dir="./logs",
            data_dir=None,
            max_episodes=1,
            llm_base_url=None,
            agent_seed=None,
            backend=None,
            model=None,
            port=None,
            llm_kwargs_raw=None,
            max_retries=None,
            resume_dir="/tmp/some/path",
            refresh_data=False,
        )
        with pytest.raises(SystemExit):
            cmd_start(args)
