"""Tests for the dummy agent."""
import pytest

from easi.agents.dummy_agent import DummyAgent
from easi.core.episode import Observation


class TestDummyAgent:
    @pytest.fixture
    def agent(self):
        return DummyAgent(
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
            seed=42,
        )

    def test_act_returns_valid_action(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name in agent.action_space

    def test_deterministic_with_seed(self):
        agent1 = DummyAgent(action_space=["A", "B", "C"], seed=123)
        agent2 = DummyAgent(action_space=["A", "B", "C"], seed=123)

        obs = Observation(rgb_path="/tmp/rgb.png")
        for _ in range(10):
            a1 = agent1.act(obs, "test")
            a2 = agent2.act(obs, "test")
            assert a1.action_name == a2.action_name

    def test_reset(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "test")
        assert agent._step_count == 1
        agent.reset()
        assert agent._step_count == 0
