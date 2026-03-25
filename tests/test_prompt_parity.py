"""Tests verifying EBAlfredPromptBuilder produces prompts matching VLMPlanner."""
import json

import pytest

from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.tasks.ebalfred.actions import get_global_action_space
from easi.tasks.ebalfred.prompts import (
    ALFRED_SYSTEM_PROMPT,
    OUTPUT_TEMPLATE,
    EBAlfredPromptBuilder,
)


@pytest.fixture
def action_space():
    return get_global_action_space()


@pytest.fixture
def builder(action_space):
    b = EBAlfredPromptBuilder(n_shot=10, split="base")
    b.set_action_space(action_space)
    return b


@pytest.fixture
def obs(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
    return Observation(rgb_path=str(img))


def _make_memory(action_space, obs, task="Put a mug on the shelf.",
                 steps=None):
    """Helper to build AgentMemory with optional step history."""
    mem = AgentMemory(
        task_description=task,
        action_space=action_space,
        current_observation=obs,
    )
    if steps:
        for action_name, feedback, llm_resp in steps:
            mem.record_step(obs, Action(action_name=action_name), llm_response=llm_resp)
            if feedback is not None:
                mem.record_feedback(feedback)
    return mem


class TestPromptParity:
    def test_first_turn_contains_system_prompt_text(self, builder, obs, action_space):
        """Full system prompt embedded in step prompt."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "You are a robot operating in a home" in text
        assert "Action Descriptions and Validity Rules" in text

    def test_action_list_format(self, builder, obs, action_space):
        """Action IDs in 'action id N: name' format."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "action id 0: find a Cart" in text
        actions = get_global_action_space()
        last_idx = len(actions) - 1
        assert f"action id {last_idx}: {actions[-1]}" in text

    def test_examples_included(self, builder, obs, action_space):
        """10 few-shot examples present."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "Task Execution Example 0" in text
        assert "Task Execution Example 9" in text

    def test_output_template_appended(self, builder, obs, action_space):
        """VLMPlanner output template with field descriptions."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "visual_state_description" in text
        assert "avoid using any contractions" in text
        assert "action_id" in text
        assert "action_name" in text

    def test_image_before_text(self, builder, obs, action_space):
        """Image content part comes before text part."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        content = msgs[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[-1]["type"] == "text"

    def test_history_format_matches_vlmplanner(self, builder, obs, action_space):
        """History uses 'Step N, action id ID, name, env feedback: ...'."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp"),
        ])
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        alarm_id = builder.action_name_to_id("find a AlarmClock")
        expected = f"Step 0, action id {alarm_id}, find a AlarmClock, env feedback: Last action executed successfully."
        assert expected in text

    def test_instruction_rstrip_dot(self, builder, obs, action_space):
        """Trailing period stripped from instruction."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "human instruction is: Put a mug on the shelf." in text
        assert "Put a mug on the shelf.." not in text

    def test_long_horizon_uses_different_examples(self):
        """long_horizon split loads from alfred_long_horizon_examples.json."""
        builder_lh = EBAlfredPromptBuilder(n_shot=10, split="long_horizon")
        builder_base = EBAlfredPromptBuilder(n_shot=10, split="base")
        assert builder_lh._examples != builder_base._examples

    def test_subsequent_turn_includes_full_prompt(self, builder, obs, action_space):
        """Even with history, full system prompt + examples re-included (stateless)."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a Mug", "success", "resp"),
        ])
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "You are a robot operating in a home" in text
        assert "Action Descriptions and Validity Rules" in text
        assert "Task Execution Example" in text
        assert "action history:" in text

    def test_first_turn_instruction_format(self, builder, obs, action_space):
        """First turn has specific instruction format from VLMPlanner."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "Now the human instruction is: Put a mug on the shelf." in text
        assert "describe current visual state from the image" in text

    def test_subsequent_turn_reflection_prompt(self, builder, obs, action_space):
        """Subsequent turns include reflection prompt from VLMPlanner."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a Mug", "success", "resp"),
        ])
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "Considering the above interaction history" in text
        assert "reason why the last action or plan failed" in text

    def test_guidelines_section(self, builder, obs, action_space):
        """Guidelines section matches VLMPlanner system prompt."""
        mem = _make_memory(action_space, obs)
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "Avoid generating empty plan" in text
        assert "Prevent Repeating Action Sequences" in text
        assert "Multiple Instances" in text
        assert "Reflection on History and Feedback" in text

    def test_no_feedback_mode(self, obs):
        """use_feedback=False omits env feedback from history."""
        builder = EBAlfredPromptBuilder(n_shot=10, split="base", use_feedback=False)
        actions = get_global_action_space()
        mem = _make_memory(actions, obs, task="test", steps=[
            ("find a Mug", "success", "resp"),
        ])
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "env feedback:" not in text

    def test_action_id_map(self, builder):
        """Action ID map correctly maps names to indices."""
        assert builder.action_name_to_id("find a Cart") == 0
        assert builder.action_name_to_id("find a Potato") == 1
        assert builder.action_id_to_name(0) == "find a Cart"

    def test_n_shot_0_no_examples(self, obs):
        """n_shot=0 produces no examples in prompt."""
        builder = EBAlfredPromptBuilder(n_shot=0, split="base")
        actions = get_global_action_space()
        mem = _make_memory(actions, obs, task="test")
        msgs = builder.build_messages(mem)
        text = msgs[0]["content"][-1]["text"]
        assert "Task Execution Example" not in text


class TestStatelessAgent:
    @pytest.fixture
    def mock_llm(self):
        class MockLLM:
            def __init__(self):
                self.call_count = 0
                self.last_messages = None

            def generate(self, messages, **kwargs):
                self.call_count += 1
                self.last_messages = messages
                return json.dumps({
                    "visual_state_description": "I see a room.",
                    "reasoning_and_reflection": "I need to find the mug.",
                    "language_plan": "1. find a Mug",
                    "executable_plan": [
                        {"action_id": 14, "action_name": "find a Mug"}
                    ],
                })
        return MockLLM()

    def test_no_history_accumulation(self, mock_llm):
        """Stateless builder: memory grows but prompt is always rebuilt from scratch."""
        from easi.agents.react_agent import ReActAgent

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=mock_llm,
            action_space=actions,
            prompt_builder=builder,
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "Put a mug on the shelf.")
        agent.add_feedback("find a Mug", "success")
        # Memory should have steps but builder rebuilds each turn
        assert len(agent.memory.steps) == 1
        agent.act(obs, "Put a mug on the shelf.")
        assert len(agent.memory.steps) == 2

    def test_action_id_parsing(self, mock_llm):
        """Parses {"action_id": int, "action_name": str} format."""
        from easi.agents.react_agent import ReActAgent

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=mock_llm,
            action_space=actions,
            prompt_builder=builder,
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Put a mug on the shelf.")
        assert action.action_name == "find a Mug"

    def test_update_action_space(self, mock_llm):
        """Dynamic action space update works."""
        from easi.agents.react_agent import ReActAgent

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=mock_llm,
            action_space=actions,
            prompt_builder=builder,
        )

        new_actions = actions + ["find a Cabinet_2", "find a Cabinet_3"]
        agent.update_action_space(new_actions)
        assert "find a Cabinet_2" in agent.action_space
        assert builder.action_name_to_id("find a Cabinet_2") is not None

    def test_buffer_cleared_on_invalid(self, mock_llm):
        """'invalid' in feedback clears buffer."""
        from easi.agents.react_agent import ReActAgent

        # LLM returns multi-action plan
        mock_llm.generate = lambda msgs, **kwargs: json.dumps({
            "executable_plan": [
                {"action_id": 14, "action_name": "find a Mug"},
                {"action_id": 0, "action_name": "find a Cart"},
            ],
        })

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=mock_llm,
            action_space=actions,
            prompt_builder=builder,
        )

        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "test")  # returns find a Mug, buffers find a Cart
        assert len(agent._action_buffer) == 1

        agent.add_feedback("find a Mug", "invalid action")
        assert len(agent._action_buffer) == 0

    def test_json_repair_applied(self):
        """JSON repair is applied via parse_response."""
        from easi.agents.react_agent import ReActAgent

        class MockLLMWithBadJson:
            def generate(self, messages, **kwargs):
                return "{'visual_state_description': 'room', 'reasoning_and_reflection': 'think', 'language_plan': 'plan', 'executable_plan': [{'action_id': 14, 'action_name': 'find a Mug'}]}"

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=MockLLMWithBadJson(),
            action_space=actions,
            prompt_builder=builder,
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "test")
        assert action.action_name == "find a Mug"

    def test_reset_clears_memory(self, mock_llm):
        """Reset clears memory."""
        from easi.agents.react_agent import ReActAgent

        actions = get_global_action_space()
        builder = EBAlfredPromptBuilder(n_shot=10, split="base")
        agent = ReActAgent(
            llm_client=mock_llm,
            action_space=actions,
            prompt_builder=builder,
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "test")
        assert len(agent.memory.steps) == 1
        agent.reset()
        assert len(agent.memory.steps) == 0


class TestChatHistoryMode:
    """Tests for chat_history=True mode matching VLMPlanner."""

    @pytest.fixture
    def ch_builder(self, action_space):
        b = EBAlfredPromptBuilder(n_shot=10, split="base", chat_history=True)
        b.set_action_space(action_space)
        return b

    def test_first_turn_same_as_stateless(self, ch_builder, obs, action_space):
        """First turn prompt is identical for both modes."""
        stateless_builder = EBAlfredPromptBuilder(n_shot=10, split="base", chat_history=False)
        stateless_builder.set_action_space(action_space)

        mem = _make_memory(action_space, obs)
        ch_msgs = ch_builder.build_messages(mem)
        st_msgs = stateless_builder.build_messages(mem)

        ch_text = ch_msgs[0]["content"][-1]["text"]
        st_text = st_msgs[0]["content"][-1]["text"]
        assert ch_text == st_text

    def test_subsequent_turn_no_system_prompt(self, ch_builder, obs, action_space):
        """Subsequent turns do NOT contain 'You are a robot' system prompt."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        # Last message is the current turn
        last_text = msgs[-1]["content"][-1]["text"]
        assert "You are a robot operating in a home" not in last_text

    def test_subsequent_turn_no_examples(self, ch_builder, obs, action_space):
        """Subsequent turns do NOT contain examples."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        last_text = msgs[-1]["content"][-1]["text"]
        assert "Task Execution Example" not in last_text

    def test_subsequent_turn_instruction_prefix(self, ch_builder, obs, action_space):
        """Subsequent turns use 'The human instruction is:' not '## Now the'."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        last_text = msgs[-1]["content"][-1]["text"]
        assert "The human instruction is: Put a mug on the shelf." in last_text
        assert "## Now the human instruction is" not in last_text

    def test_subsequent_turn_has_history(self, ch_builder, obs, action_space):
        """Subsequent turns include action history."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        last_text = msgs[-1]["content"][-1]["text"]
        alarm_id = ch_builder.action_name_to_id("find a AlarmClock")
        assert f"Step 0, action id {alarm_id}, find a AlarmClock" in last_text

    def test_subsequent_turn_has_output_template(self, ch_builder, obs, action_space):
        """Subsequent turns include OUTPUT_TEMPLATE."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        last_text = msgs[-1]["content"][-1]["text"]
        assert "visual_state_description" in last_text
        assert "avoid using any contractions" in last_text

    def test_subsequent_turn_has_reflection(self, ch_builder, obs, action_space):
        """Subsequent turns include reflection prompt."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "Last action executed successfully.", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        last_text = msgs[-1]["content"][-1]["text"]
        assert "Considering the above interaction history" in last_text

    def test_message_accumulation(self, ch_builder, obs, action_space):
        """Messages grow with user/assistant pairs."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "success", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        # 1 past user + 1 past assistant + 1 current user = 3
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

        # Add another step
        mem2 = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "success", "resp1"),
            ("find a Mug", "success", "resp2"),
        ])
        msgs2 = ch_builder.build_messages(mem2)
        # 2 past user + 2 past assistant + 1 current user = 5
        assert len(msgs2) == 5

    def test_assistant_response_list_format(self, ch_builder, obs, action_space):
        """Assistant response uses [{"type": "text", "text": ...}] format."""
        mem = _make_memory(action_space, obs, steps=[
            ("find a AlarmClock", "success", "resp1"),
        ])
        msgs = ch_builder.build_messages(mem)
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][0]["text"] == "resp1"

    def test_buffered_steps_skipped_in_messages(self, ch_builder, obs, action_space):
        """Steps with llm_response=None (buffered) don't produce user/assistant pairs."""
        mem = AgentMemory(
            task_description="Put a mug on the shelf.",
            action_space=action_space,
            current_observation=obs,
        )
        # Step 0: LLM call
        mem.record_step(obs, Action(action_name="find a AlarmClock"), llm_response="resp1")
        mem.record_feedback("success")
        # Step 1: buffered (no LLM call)
        mem.record_step(obs, Action(action_name="find a Mug"), llm_response=None)
        mem.record_feedback("success")

        msgs = ch_builder.build_messages(mem)
        # Only 1 past user/assistant pair (step 0), skip step 1, + 1 current user = 3
        assert len(msgs) == 3


class TestParseResponse:
    """Tests for EBAlfredPromptBuilder.parse_response."""

    @pytest.fixture
    def builder(self, action_space):
        b = EBAlfredPromptBuilder(n_shot=10, split="base")
        b.set_action_space(action_space)
        return b

    def test_parse_valid_action_id_format(self, builder, action_space):
        """Parses {"action_id": int, "action_name": str}."""
        mem = AgentMemory(action_space=action_space)
        response = json.dumps({
            "executable_plan": [{"action_id": 14, "action_name": "find a Mug"}],
        })
        actions = builder.parse_response(response, mem)
        assert len(actions) == 1
        assert actions[0].action_name == "find a Mug"

    def test_parse_valid_action_format(self, builder, action_space):
        """Parses {"action": str}."""
        mem = AgentMemory(action_space=action_space)
        response = json.dumps({
            "executable_plan": [{"action": "find a Mug"}],
        })
        actions = builder.parse_response(response, mem)
        assert len(actions) == 1
        assert actions[0].action_name == "find a Mug"

    def test_parse_json_repair(self, builder, action_space):
        """Single quotes are fixed by JSON repair."""
        mem = AgentMemory(action_space=action_space)
        response = "{'executable_plan': [{'action_id': 14, 'action_name': 'find a Mug'}]}"
        actions = builder.parse_response(response, mem)
        assert len(actions) == 1
        assert actions[0].action_name == "find a Mug"

    def test_parse_invalid_action_stops(self, builder, action_space):
        """Stops processing on first invalid action."""
        mem = AgentMemory(action_space=action_space)
        response = json.dumps({
            "executable_plan": [
                {"action_id": 14, "action_name": "find a Mug"},
                {"action_id": 9999, "action_name": "fly to moon"},
            ],
        })
        actions = builder.parse_response(response, mem)
        assert len(actions) == 1
        assert actions[0].action_name == "find a Mug"

    def test_parse_invalid_json_returns_empty(self, builder, action_space):
        """Completely invalid JSON returns empty list."""
        mem = AgentMemory(action_space=action_space)
        actions = builder.parse_response("not json at all", mem)
        assert actions == []


class TestGetResponseFormat:
    """Test that get_response_format returns correct schema."""

    def test_returns_json_schema_dict(self):
        builder = EBAlfredPromptBuilder(n_shot=1, split="base")
        builder.set_action_space(["find a Mug", "pick up the Mug", "Stop"])
        memory = AgentMemory(
            action_space=["find a Mug", "pick up the Mug", "Stop"],
        )
        rf = builder.get_response_format(memory)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "embodied_planning"
        schema = rf["json_schema"]["schema"]
        assert "executable_plan" in schema["properties"]
        assert schema["properties"]["executable_plan"]["type"] == "array"
        items = schema["properties"]["executable_plan"]["items"]
        assert "action_id" in items["properties"]
        assert "action_name" in items["properties"]

    def test_schema_matches_vlm_generation_guide(self):
        """Verify our schema has the same required fields as EmbodiedBench."""
        from easi.tasks.ebalfred.prompts import EBALFRED_RESPONSE_SCHEMA
        schema = EBALFRED_RESPONSE_SCHEMA["json_schema"]["schema"]
        assert set(schema["required"]) == {
            "visual_state_description",
            "reasoning_and_reflection",
            "language_plan",
            "executable_plan",
        }
        items_required = set(schema["properties"]["executable_plan"]["items"]["required"])
        assert items_required == {"action_id", "action_name"}
