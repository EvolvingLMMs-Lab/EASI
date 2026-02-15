"""Tests for the dummy task and task registry."""

import pytest

from easi.core.episode import Observation, StepResult
from easi.tasks.dummy_task.task import DummyTask
from easi.tasks.registry import get_task_entry, list_tasks, load_task_class


class TestDummyTask:
    @pytest.fixture
    def task(self):
        return DummyTask()

    def test_name(self, task):
        assert task.name == "dummy_task"

    def test_simulator_key(self, task):
        assert task.simulator_key == "dummy:v1"

    def test_action_space(self, task):
        assert "MoveAhead" in task.action_space
        assert "Stop" in task.action_space

    def test_max_steps(self, task):
        assert task.max_steps == 100

    def test_load_episodes(self, task):
        episodes = task.load_episodes()
        assert len(episodes) == 3
        assert episodes[0]["episode_id"] == "dummy_ep_001"

    def test_get_episode(self, task):
        ep = task.get_episode(1)
        assert ep["episode_id"] == "dummy_ep_002"

    def test_len(self, task):
        assert len(task) == 3

    def test_format_reset_config(self, task):
        ep = task.get_episode(0)
        config = task.format_reset_config(ep)
        assert "scene_id" in config
        assert config["scene_id"] == "dummy_scene_A"

    def test_evaluate_episode_success(self, task):
        ep = task.get_episode(0)
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False),
            StepResult(observation=obs, reward=1.0, done=True),
        ]
        metrics = task.evaluate_episode(ep, trajectory)
        assert metrics["success"] == 1.0
        assert metrics["num_steps"] == 2.0
        assert metrics["total_reward"] == 1.0

    def test_evaluate_episode_failure(self, task):
        ep = task.get_episode(0)
        obs = Observation(rgb_path="/tmp/rgb.png")
        trajectory = [
            StepResult(observation=obs, reward=0.0, done=False),
        ]
        metrics = task.evaluate_episode(ep, trajectory)
        assert metrics["success"] == 0.0

    def test_evaluate_empty_trajectory(self, task):
        ep = task.get_episode(0)
        metrics = task.evaluate_episode(ep, [])
        assert metrics["success"] == 0.0
        assert metrics["num_steps"] == 0.0


class TestTaskRegistry:
    def test_list_tasks(self):
        tasks = list_tasks()
        assert "dummy_task" in tasks

    def test_get_task_entry(self):
        entry = get_task_entry("dummy_task")
        assert entry.name == "dummy_task"
        assert entry.simulator_key == "dummy:v1"

    def test_load_task_class(self):
        TaskClass = load_task_class("dummy_task")
        assert TaskClass is DummyTask

    def test_unknown_task(self):
        with pytest.raises(KeyError):
            get_task_entry("nonexistent_task")
