"""Tests for the ReAct agent and PromptBuilder protocol."""
import base64
import json
import pytest

from easi.agents.prompt_builder import PromptBuilderProtocol, DefaultPromptBuilder
from easi.agents.react_agent import ReActAgent
from easi.core.episode import Action, Observation


class MockLLMClient:
    """Mock LLM that returns a fixed JSON plan."""
    def __init__(self, actions=None):
        self.actions = actions or [{"action": "MoveAhead"}]
        self.call_count = 0

    def generate(self, messages):
        self.call_count += 1
        return json.dumps({
            "observation": "I see a room.",
            "reasoning": "I should move forward.",
            "plan": "1. Move ahead",
            "executable_plan": self.actions,
        })


class TestDefaultPromptBuilder:
    def test_build_system_prompt_returns_messages(self):
        builder = DefaultPromptBuilder()
        messages = builder.build_system_prompt(
            action_space=["MoveAhead", "Stop"],
            task_description="Go to the kitchen.",
        )
        # Decision #10: returns OpenAI message format
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        content = messages[0]["content"]
        assert "MoveAhead" in content
        assert "JSON" in content

    def test_build_step_prompt_returns_messages(self):
        obs = Observation(rgb_path="/tmp/rgb.png")
        builder = DefaultPromptBuilder()
        messages = builder.build_step_prompt(
            observation=obs,
            task_description="Go to the kitchen.",
            action_history=[],
        )
        # Decision #10: returns list of OpenAI message dicts
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        # Content should be a list with text + image parts
        content = messages[0]["content"]
        assert isinstance(content, list)
        text_parts = [p for p in content if p["type"] == "text"]
        assert len(text_parts) >= 1

    def test_build_step_prompt_includes_image(self, tmp_path):
        # Create a real image file
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        obs = Observation(rgb_path=str(img_path))
        builder = DefaultPromptBuilder()
        messages = builder.build_step_prompt(
            observation=obs,
            task_description="Go to the kitchen.",
            action_history=[],
        )
        content = messages[0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_build_step_prompt_with_history(self):
        obs = Observation(rgb_path="/tmp/rgb.png")
        builder = DefaultPromptBuilder()
        messages = builder.build_step_prompt(
            observation=obs,
            task_description="Go to the kitchen.",
            action_history=[("MoveAhead", "success"), ("TurnLeft", "failed")],
        )
        text_content = ""
        for part in messages[0]["content"]:
            if part["type"] == "text":
                text_content += part["text"]
        assert "MoveAhead" in text_content
        assert "TurnLeft" in text_content


class CustomPromptBuilder:
    """A custom prompt builder for testing the delegation pattern."""
    def build_system_prompt(self, action_space, task_description):
        return [{"role": "system", "content": f"CUSTOM SYSTEM: {task_description}"}]

    def build_step_prompt(self, observation, task_description, action_history):
        return [{"role": "user", "content": f"CUSTOM STEP: history_len={len(action_history)}"}]


class TestReActAgent:
    @pytest.fixture
    def agent(self):
        llm = MockLLMClient([{"action": "MoveAhead"}])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )

    @pytest.fixture
    def multi_action_agent(self):
        """Agent whose LLM returns a multi-action plan."""
        llm = MockLLMClient([
            {"action": "MoveAhead"},
            {"action": "TurnLeft"},
            {"action": "MoveAhead"},
        ])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )

    @pytest.fixture
    def custom_agent(self):
        llm = MockLLMClient([{"action": "MoveAhead"}])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
            prompt_builder=CustomPromptBuilder(),
        )

    def test_act_returns_action(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "MoveAhead"

    def test_calls_llm_once(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "Go to the goal.")
        assert agent.llm_client.call_count == 1

    def test_chat_history_has_system_prompt(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "Go to the goal.")
        history = agent.chat_history
        assert history[0]["role"] == "system"

    def test_default_prompt_builder_used(self, agent):
        assert isinstance(agent.prompt_builder, DefaultPromptBuilder)

    def test_custom_prompt_builder(self, custom_agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        custom_agent.act(obs, "Go to the goal.")
        history = custom_agent.chat_history
        assert history[0]["role"] == "system"
        assert "CUSTOM SYSTEM:" in history[0]["content"]

    # --- Decision #9: Action buffering tests ---

    def test_multi_action_buffer(self, multi_action_agent):
        """LLM returns 3 actions; first act() returns first, subsequent act()s
        return buffered actions WITHOUT calling LLM again."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        a1 = multi_action_agent.act(obs, "Go to the goal.")
        assert a1.action_name == "MoveAhead"
        assert multi_action_agent.llm_client.call_count == 1  # LLM called

        multi_action_agent.add_feedback("MoveAhead", "success")
        a2 = multi_action_agent.act(obs, "Go to the goal.")
        assert a2.action_name == "TurnLeft"
        assert multi_action_agent.llm_client.call_count == 1  # NOT called again

        multi_action_agent.add_feedback("TurnLeft", "success")
        a3 = multi_action_agent.act(obs, "Go to the goal.")
        assert a3.action_name == "MoveAhead"
        assert multi_action_agent.llm_client.call_count == 1  # Still NOT called

    def test_buffer_cleared_on_failure(self, multi_action_agent):
        """When add_feedback reports failure, buffer is cleared.
        Next act() re-queries LLM."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        multi_action_agent.act(obs, "Go to the goal.")  # returns MoveAhead, buffers [TurnLeft, MoveAhead]
        assert multi_action_agent.llm_client.call_count == 1

        multi_action_agent.add_feedback("MoveAhead", "failed: obstacle ahead")
        # Buffer should be cleared

        multi_action_agent.act(obs, "Go to the goal.")  # should re-query LLM
        assert multi_action_agent.llm_client.call_count == 2

    def test_buffer_empty_after_all_consumed(self, multi_action_agent):
        """After all buffered actions consumed, next act() queries LLM."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("MoveAhead", "success")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("TurnLeft", "success")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("MoveAhead", "success")
        assert multi_action_agent.llm_client.call_count == 1

        # Buffer exhausted — next call should query LLM
        multi_action_agent.act(obs, "Go to the goal.")
        assert multi_action_agent.llm_client.call_count == 2

    def test_parse_error_returns_stop(self):
        """When LLM returns invalid JSON, agent returns Stop (no buffering)."""
        llm = type('MockLLM', (), {'generate': lambda self, m: 'not json at all'})()
        agent = ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "Stop"
        assert not agent._action_buffer  # buffer empty

    def test_invalid_action_name_fallback(self):
        """When LLM returns action not in action_space, fallback to Stop."""
        llm = MockLLMClient([{"action": "FlyToMoon"}])
        agent = ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "Stop"

    def test_reset_clears_buffer_and_history(self, multi_action_agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.reset()
        assert len(multi_action_agent.chat_history) == 0
        assert len(multi_action_agent._action_buffer) == 0
        assert len(multi_action_agent._action_feedback) == 0
