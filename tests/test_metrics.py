"""Tests for the pluggable metric system."""
import pytest
from pathlib import Path

from easi.core.episode import EpisodeRecord, Observation, StepResult


def _make_step(info=None, reward=0.0, done=False):
    """Helper to create a StepResult with given info."""
    obs = Observation(rgb_path="/tmp/fake.png")
    return StepResult(observation=obs, reward=reward, done=done, info=info or {})


class TestEpisodeRecord:
    """Test the EpisodeRecord dataclass."""

    def test_create(self):
        record = EpisodeRecord(
            episode={"episode_id": "ep1"},
            trajectory=[_make_step()],
            episode_results={"success": 1.0},
        )
        assert record.episode["episode_id"] == "ep1"
        assert len(record.trajectory) == 1
        assert record.episode_results["success"] == 1.0

    def test_defaults(self):
        record = EpisodeRecord(
            episode={},
            trajectory=[],
            episode_results={},
        )
        assert record.episode == {}
        assert record.trajectory == []
        assert record.episode_results == {}


class TestDefaultAggregate:
    """Test default_aggregate (extracted from aggregate_metrics)."""

    def test_empty_results(self):
        from easi.evaluation.metrics import default_aggregate
        assert default_aggregate([]) == {}

    def test_averages_numeric_keys(self):
        from easi.evaluation.metrics import default_aggregate
        records = [
            EpisodeRecord({}, [], {"task_success": 1.0, "num_steps": 10.0}),
            EpisodeRecord({}, [], {"task_success": 0.0, "num_steps": 20.0}),
        ]
        agg = default_aggregate(records)
        assert agg["avg_task_success"] == 0.5
        assert agg["avg_num_steps"] == 15.0

    def test_ignores_non_numeric_keys_in_episode_results(self):
        from easi.evaluation.metrics import default_aggregate
        records = [
            EpisodeRecord({}, [], {"task_success": 1.0, "note": "good"}),
        ]
        agg = default_aggregate(records)
        assert "avg_task_success" in agg
        assert "avg_note" not in agg

    def test_missing_keys_count_as_zero(self):
        """Failed episodes missing task-specific keys should not inflate averages."""
        from easi.evaluation.metrics import default_aggregate
        records = [
            EpisodeRecord({}, [], {"task_success": 1.0, "spl": 0.8, "num_steps": 10.0}),
            EpisodeRecord({}, [], {"task_success": 1.0, "spl": 0.6, "num_steps": 5.0}),
            # Failed episode: only has success and num_steps, no spl
            EpisodeRecord({}, [], {"task_success": 0.0, "num_steps": 0.0}),
        ]
        agg = default_aggregate(records)
        # spl should be averaged over 3 episodes (0.8 + 0.6 + 0) / 3 = 0.4667
        assert agg["avg_spl"] == round((0.8 + 0.6) / 3, 4)
        # task_success averaged over 3: (1 + 1 + 0) / 3 = 0.6667
        assert agg["avg_task_success"] == round(2.0 / 3, 4)

    def test_convenience_aliases(self):
        from easi.evaluation.metrics import default_aggregate
        records = [
            EpisodeRecord({}, [], {"success": 1.0, "num_steps": 5.0}),
            EpisodeRecord({}, [], {"success": 0.0, "num_steps": 15.0}),
        ]
        agg = default_aggregate(records)
        assert agg.get("success_rate") == 0.5
        assert agg.get("avg_steps") == 10.0


class TestBaseTaskAggregateResults:
    """Test BaseTask.aggregate_results() default implementation."""

    def _make_task(self):
        """Create a minimal concrete BaseTask."""
        from easi.core.base_task import BaseTask

        class MinimalTask(BaseTask):
            def get_task_yaml_path(self):
                return Path("/dev/null")
            def format_reset_config(self, episode):
                return {}
            def evaluate_episode(self, episode, trajectory):
                return {"success": 1.0}
            def _load_config(self):
                return {"name": "test", "simulator": "dummy:v1"}
        return MinimalTask()

    def test_default_aggregate_averages(self):
        task = self._make_task()
        records = [
            EpisodeRecord({}, [], {"success": 1.0, "num_steps": 10.0}),
            EpisodeRecord({}, [], {"success": 0.0, "num_steps": 20.0}),
        ]
        agg = task.aggregate_results(records)
        assert agg["avg_success"] == 0.5
        assert agg["avg_num_steps"] == 15.0

    def test_default_aggregate_empty(self):
        task = self._make_task()
        assert task.aggregate_results([]) == {}

    def test_custom_override(self):
        """Subclass can override aggregate_results for custom logic."""
        from easi.core.base_task import BaseTask

        class CustomTask(BaseTask):
            def get_task_yaml_path(self):
                return Path("/dev/null")
            def format_reset_config(self, episode):
                return {}
            def evaluate_episode(self, episode, trajectory):
                return {}
            def _load_config(self):
                return {"name": "test", "simulator": "dummy:v1"}
            def aggregate_results(self, records):
                n_success = sum(
                    1 for r in records
                    if r.episode_results.get("success", 0) > 0
                )
                total_steps = sum(
                    len(r.trajectory) for r in records
                )
                return {
                    "success_rate": n_success / len(records) if records else 0,
                    "total_steps": float(total_steps),
                }

        task = CustomTask()
        records = [
            EpisodeRecord({}, [_make_step(), _make_step()], {"success": 1.0}),
            EpisodeRecord({}, [_make_step()], {"success": 0.0}),
        ]
        agg = task.aggregate_results(records)
        assert agg["success_rate"] == 0.5
        assert agg["total_steps"] == 3.0


class TestSummaryJsonStructure:
    """Test that summary.json nests metrics under 'metrics' key."""

    def test_metrics_nested_in_summary(self, tmp_path):
        """Run dummy_task and verify metrics are under summary['metrics']."""
        from easi.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            task_name="dummy_task",
            agent_type="dummy",
            output_dir=str(tmp_path),
            episodes=":2",
        )
        runner.run()

        import json
        summary_path = list(tmp_path.rglob("summary.json"))[0]
        summary = json.loads(summary_path.read_text())

        # Metrics should be nested
        assert "metrics" in summary
        assert isinstance(summary["metrics"], dict)
        assert len(summary["metrics"]) > 0

        # Metadata should be at top level
        assert summary["num_episodes"] == 2
        assert "effective_episodes" in summary
        assert summary["effective_episodes"] <= summary["num_episodes"]

        # Metric keys should NOT be at the top level
        assert "avg_success" not in summary
        assert "success_rate" not in summary
