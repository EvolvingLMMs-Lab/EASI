"""Tests for JSON repair utility (ported from EmbodiedBench planner_utils)."""
import json

import pytest

from easi.utils.json_repair import fix_json


class TestFixJson:
    def test_single_quotes_to_double(self):
        s = "{'key': 'value'}"
        result = fix_json(s)
        assert '"key"' in result
        assert '"value"' in result

    def test_contractions_preserved(self):
        s = "{'text': \"it's a test and we're here\"}"
        result = fix_json(s)
        assert "'s" in result
        assert "'re" in result

    def test_markdown_fences_stripped(self):
        s = '```json\n{"key": "value"}\n```'
        result = fix_json(s)
        assert "```" not in result
        data = json.loads(result)
        assert data["key"] == "value"

    def test_unescaped_quotes_in_reasoning(self):
        s = '{"reasoning_and_reflection": "He said "hello" to the robot", "language_plan": "go"}'
        result = fix_json(s)
        data = json.loads(result)
        assert "language_plan" in data

    def test_valid_json_passthrough(self):
        original = '{"key": "value", "num": 42}'
        result = fix_json(original)
        assert json.loads(result) == json.loads(original)

    def test_contraction_ll(self):
        s = "{'text': \"I'll do it\"}"
        result = fix_json(s)
        assert "'ll" in result

    def test_contraction_t(self):
        s = "{'text': \"don't stop\"}"
        result = fix_json(s)
        assert "'t" in result

    def test_contraction_d(self):
        s = "{'text': \"I'd go\"}"
        result = fix_json(s)
        assert "'d" in result

    def test_contraction_m(self):
        s = "{'text': \"I'm here\"}"
        result = fix_json(s)
        assert "'m" in result

    def test_contraction_ve(self):
        s = "{'text': \"I've been\"}"
        result = fix_json(s)
        assert "'ve" in result

    def test_full_llm_output_with_single_quotes(self):
        """Simulate a typical LLM output that uses single quotes."""
        s = """{'visual_state_description': 'I see a kitchen', 'reasoning_and_reflection': 'The robot hasn't found the mug yet', 'language_plan': 'Find the mug', 'executable_plan': [{'action_id': 14, 'action_name': 'find a Mug'}]}"""
        result = fix_json(s)
        data = json.loads(result)
        assert data["executable_plan"][0]["action_id"] == 14
        assert data["executable_plan"][0]["action_name"] == "find a Mug"
