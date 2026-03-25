"""Tests for AgentMemory and StepRecord."""
import pytest

from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory, StepRecord


class TestStepRecord:
    def test_defaults(self):
        obs = Observation(rgb_path="/tmp/rgb.png")
        rec = StepRecord(observation=obs)
        assert rec.action is None
        assert rec.feedback is None
        assert rec.llm_response is None
        assert rec.step_number == 0


class TestAgentMemory:
    @pytest.fixture
    def memory(self):
        return AgentMemory(
            task_description="Pick up the mug.",
            action_space=["MoveAhead", "TurnLeft", "Stop"],
        )

    def test_is_first_turn_initially(self, memory):
        assert memory.is_first_turn is True

    def test_is_first_turn_after_step(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="resp")
        assert memory.is_first_turn is False

    def test_action_history_empty(self, memory):
        assert memory.action_history == []

    def test_action_history_with_feedback(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="resp")
        memory.record_feedback("success")
        assert memory.action_history == [("MoveAhead", "success")]

    def test_action_history_excludes_no_feedback(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="resp")
        # No feedback recorded
        assert memory.action_history == []

    def test_record_step_increments_number(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="r1")
        memory.record_step(obs, Action(action_name="TurnLeft"), llm_response="r2")
        assert memory.steps[0].step_number == 0
        assert memory.steps[1].step_number == 1

    def test_record_feedback_attaches_to_last(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="r1")
        memory.record_step(obs, Action(action_name="TurnLeft"), llm_response="r2")
        memory.record_feedback("failed")
        assert memory.steps[0].feedback is None
        assert memory.steps[1].feedback == "failed"

    def test_record_feedback_no_steps(self, memory):
        # Should not raise
        memory.record_feedback("orphan")

    def test_clear(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.current_observation = obs
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="r1")
        memory.clear()
        assert memory.steps == []
        assert memory.current_observation is None
        assert memory.task_description == ""
        assert memory.is_first_turn is True

    def test_buffered_step_has_no_llm_response(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response=None)
        assert memory.steps[0].llm_response is None

    def test_action_history_multiple(self, memory):
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="r1")
        memory.record_feedback("success")
        memory.record_step(obs, Action(action_name="TurnLeft"), llm_response="r2")
        memory.record_feedback("failed")
        assert memory.action_history == [
            ("MoveAhead", "success"),
            ("TurnLeft", "failed"),
        ]
