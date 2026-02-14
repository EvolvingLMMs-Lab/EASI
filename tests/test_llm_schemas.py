"""Tests for LLM response schemas."""
import json
import pytest
from pydantic import BaseModel


class TestBaseResponseSchema:
    def test_base_schema_get_actions_raises(self):
        from easi.llm.schemas import BaseResponseSchema

        class Bare(BaseResponseSchema):
            pass

        with pytest.raises(NotImplementedError):
            Bare().get_actions()

    def test_base_schema_is_pydantic_model(self):
        from easi.llm.schemas import BaseResponseSchema
        assert issubclass(BaseResponseSchema, BaseModel)


class TestActionPlanResponse:
    def test_get_actions_returns_list_of_strings(self):
        from easi.llm.schemas import ActionPlanResponse

        resp = ActionPlanResponse(
            reasoning="I see a mug",
            executable_plan=[
                {"action": "find a Mug"},
                {"action": "pick up the Mug"},
            ],
        )
        assert resp.get_actions() == ["find a Mug", "pick up the Mug"]

    def test_empty_plan(self):
        from easi.llm.schemas import ActionPlanResponse

        resp = ActionPlanResponse(reasoning="nothing to do", executable_plan=[])
        assert resp.get_actions() == []

    def test_roundtrip_json(self):
        from easi.llm.schemas import ActionPlanResponse

        resp = ActionPlanResponse(
            reasoning="test",
            executable_plan=[{"action": "Stop"}],
        )
        data = json.loads(resp.model_dump_json())
        assert data["executable_plan"][0]["action"] == "Stop"


class TestExecutableAction:
    def test_action_field(self):
        from easi.llm.schemas import ExecutableAction

        a = ExecutableAction(action="MoveAhead")
        assert a.action == "MoveAhead"
