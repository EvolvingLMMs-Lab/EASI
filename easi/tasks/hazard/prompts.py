"""HAZARD prompt builder matching original HAZARD LLM agent.

Reference: HAZARD/src/HAZARD/policy/llm.py
           HAZARD/src/HAZARD/policy/llm_configs/prompt_v2.csv
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from easi.agents.prompt_builder import _encode_image_base64
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"


def _load_prompt_templates() -> dict[str, str]:
    """Load scenario-specific prompt templates from vendored CSV."""
    templates = {}
    csv_path = _CONFIG_DIR / "prompts.csv"
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            templates[row["type"]] = row["prompt"]
    return templates


class HAZARDPromptBuilder:
    """Prompt builder for HAZARD benchmark.

    Constructs HAZARD-style prompts with:
    - Scenario-specific preamble (fire/flood/wind)
    - Target object descriptions with values and attributes
    - Current state (visible objects with distances and hazard info)
    - Action history with results
    - Dynamic multiple-choice action options
    """

    def __init__(
        self,
        scenario: str = "fire",
        cot: bool = False,
        show_object_history: bool = False,
    ):
        self.scenario = scenario
        self.cot = cot
        self.show_object_history = show_object_history
        self._templates = _load_prompt_templates()
        self._value_dict = json.loads((_CONFIG_DIR / "value.json").read_text())
        self._fire_dict = json.loads((_CONFIG_DIR / "fire.json").read_text())
        self._fluid_dict = json.loads((_CONFIG_DIR / "fluid.json").read_text())

    def set_action_space(self, actions: list[str]) -> None:
        """No-op -- HAZARD uses dynamic action space from bridge metadata."""
        pass

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build HAZARD-style prompt from memory state."""
        # Get current observation info
        obs = memory.current_observation
        info = obs.metadata if obs else {}

        # Parse bridge metadata
        available_plans = json.loads(info.get("available_plans", "[]"))

        # Build target objects description
        target_desc = self._build_target_description(memory)

        # Build state description
        state_desc = self._build_state_description(info)

        # Build action history (from memory, matching original format)
        history_desc = self._build_action_history(memory)

        # Build object state history (tracking temperature/water level changes)
        object_history = self._build_object_state_history(info)

        # Build available actions
        actions_desc = self._build_available_actions(available_plans)

        # Construct full prompt from template
        template = self._templates.get(self.scenario, self._templates.get("fire", ""))
        prompt = template
        prompt = prompt.replace("$TARGET_OBJECTS$", target_desc + "\n")
        prompt = prompt.replace("$STATE$", state_desc)
        prompt = prompt.replace("$ACTION_HISTORY$", history_desc + "\n")
        prompt = prompt.replace("$OBJECT_HISTORY$", object_history + "\n")
        prompt = prompt.replace("$AVAILABLE_ACTIONS$", actions_desc)

        if self.cot:
            prompt += " Let's think step by step."

        # Build messages
        content_parts = []

        # Add image if available
        if obs and obs.rgb_path:
            image_url = _encode_image_base64(obs.rgb_path)
            if image_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

        content_parts.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content_parts}]

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM response to extract selected action.

        The LLM should respond with an option letter (A, B, C, ...) or
        the full action text. We parse this against the available plans.
        """
        obs = memory.current_observation
        info = obs.metadata if obs else {}
        available_plans = json.loads(info.get("available_plans", "[]"))

        if not available_plans:
            return []

        # Find selected plan
        selected = self._match_response(llm_response.strip(), available_plans)

        if selected is not None:
            return [Action(action_name=selected)]

        # Fallback: couldn't parse -> explore
        logger.warning("Could not parse HAZARD response: %s", llm_response[:100])
        return [Action(action_name="look around")]

    def _match_response(self, text: str, available_plans: list[str]) -> str | None:
        """Match LLM response text to one of the available plans.

        Reference: HAZARD/src/HAZARD/policy/llm.py:parse_answer()
        """
        # Exact match
        for plan in available_plans:
            if plan in text:
                return plan

        # Option letter match
        for i, plan in enumerate(available_plans):
            option = chr(ord('A') + i)
            if (f"option {option}" in text.lower()
                    or f"{option}." in text.split()
                    or f"({option})" in text
                    or f"Option {option}" in text
                    or (len(text) <= 2 and option in text)):
                return plan

        # Fuzzy match by keywords
        for plan in available_plans:
            if "pick up" in plan and "pick" in text.lower():
                return plan
            if "put" in plan and "put" in text.lower():
                return plan
            if "look around" in plan and ("look" in text.lower() or "explore" in text.lower()):
                return plan

        # Single letter match
        if len(text) == 1 and text.isalpha():
            idx = ord(text.upper()) - ord('A')
            if 0 <= idx < len(available_plans):
                return available_plans[idx]

        return None

    # --- Description builders ---

    def _build_target_description(self, memory: AgentMemory) -> str:
        """Build target objects description with values and attributes.

        Matches original llm.py:objects_list2text() exactly:
        - Value: raw numeric (5 or 1)
        - Attribute: "waterproof"/"non-waterproof" for flood, "None" otherwise
        """
        obs = memory.current_observation
        info = obs.metadata if obs else {}

        target_cats = json.loads(info.get("target_categories", "[]"))
        if not target_cats:
            return "No target information available."

        lines = []
        for category in target_cats:
            # Raw numeric value matching challenge.py:get_target_info()
            value = 5 if self._value_dict.get(category) == 1 else 1
            # Only flood shows waterproof attribute; fire/wind show "None"
            if self.scenario == "flood":
                wp = self._fluid_dict.get(category, 0)
                attr = "waterproof" if wp == 1 else "non-waterproof"
            else:
                attr = "None"
            lines.append(f"name: {category}, value: {str(value)}, attribute: {attr}")

        return "\n".join(lines)

    def _build_state_description(self, info: dict) -> str:
        """Build current state description matching llm.py:progress2text().

        Format:
        [Wind only] Shopping carts already found:
        Target objects currently seen:
          name: {cat}, id: {id}, value: {val}, distance: {d} m, temperature/water_level/status
        Target objects previously seen:
          [same format for objects not currently visible]
        """
        object_list = json.loads(info.get("object_list", "[]"))
        seen_ids = json.loads(info.get("current_seen_objects_id", "[]"))
        distances = json.loads(info.get("object_distances", "{}"))
        env_record = json.loads(info.get("env_change_record", "{}"))
        target_cats = json.loads(info.get("target_categories", "[]"))

        ps = ""

        # Wind: shopping carts first
        if self.scenario == "wind":
            ps += "Shopping carts already found:\n"
            for obj in object_list:
                if obj["category"] != "shopping cart":
                    continue
                dist = distances.get(obj["id"], "?")
                ps += f"name: {obj['category']}, id: {obj['id']}, distance: {dist} m\n"

        # Target objects currently seen
        ps += "Target objects currently seen:\n"
        for obj in object_list:
            if obj["category"] not in target_cats or obj["id"] not in seen_ids:
                continue
            value = 5 if self._value_dict.get(obj["category"]) == 1 else 1
            dist = distances.get(obj["id"], "?")
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {value}, distance: {dist} m, "
            ps += self._format_hazard_info(obj["id"], env_record)

        # Target objects previously seen
        ps += "Target objects previously seen:\n"
        for obj in object_list:
            if obj["category"] not in target_cats or obj["id"] in seen_ids:
                continue
            value = 5 if self._value_dict.get(obj["category"]) == 1 else 1
            dist = distances.get(obj["id"], "?")
            ps += f"name: {obj['category']}, id: {obj['id']}, value: {value}, distance: {dist} m, "
            ps += self._format_hazard_info(obj["id"], env_record)

        return ps

    def _format_hazard_info(self, obj_id: str, env_record: dict) -> str:
        """Format per-object hazard info (temperature/water_level/status).

        Matches original progress2text() per-object suffix.
        Fire: temperature in Celsius (exp of log_temp), or "unknown"
        Flood: water level in meters, or "unknown"
        Wind: status "Unknown"
        """
        if self.scenario == "fire":
            if obj_id in env_record:
                temp = round(math.exp(env_record[obj_id]), 2)
                return f"temperature: {temp} Celsius\n"
            return "temperature: unknown\n"
        elif self.scenario == "flood":
            if obj_id in env_record:
                level = round(env_record[obj_id], 2)
                return f"water level: {level} m\n"
            return "water level: unknown\n"
        else:
            return "status: Unknown\n"

    def _build_action_history(self, memory: AgentMemory) -> str:
        """Build action history string matching llm.py format.

        Format: "plan_text (result_desc), plan_text (result_desc)"
        Where result_desc = "success" / "paused after taking 100 steps" / "fail, because {info}"
        Uses last 10 entries. Reads from memory.action_history.
        """
        action_history = memory.action_history  # list of (action_name, feedback)
        if not action_history:
            return ""

        entries = []
        for action_name, feedback in action_history[-10:]:
            result_desc = self._action_result_to_description(feedback)
            entries.append(f"{action_name} ({result_desc})")
        return ", ".join(entries)

    @staticmethod
    def _action_result_to_description(feedback: str) -> str:
        """Convert feedback to result description matching llm.py.

        Reference: llm.py:action_result_to_description()
        """
        if not feedback:
            return "success"
        lower = feedback.lower()
        if "success" in lower:
            return "success"
        if "max steps reached" in lower:
            return "paused after taking 100 steps"
        return f"fail, because {feedback}"

    def _build_object_state_history(self, info: dict) -> str:
        """Build object state history section.

        Shows how object temperature/water_level has changed over time.
        Format: "Object <id>: <val1> -> <val2> -> <val3> Celsius"

        Controlled by show_object_history config (default: False for benchmark parity).
        """
        if not self.show_object_history:
            return ""

        history = json.loads(info.get("env_change_record_history", "{}"))
        if not history:
            return ""

        lines = []
        for obj_id, values in history.items():
            if len(values) <= 1:
                continue  # No change to report
            if self.scenario == "fire":
                # Convert log_temp to Celsius and show progression
                temps = [round(math.exp(v), 1) for v in values[-5:]]  # Last 5 readings
                progression = " -> ".join(str(t) for t in temps)
                lines.append(f"Object {obj_id}: {progression} Celsius")
            elif self.scenario == "flood":
                # Show water level progression
                levels = [round(v, 2) for v in values[-5:]]
                progression = " -> ".join(str(l) for l in levels)
                lines.append(f"Object {obj_id}: water level {progression} m")

        return "\n".join(lines)

    def _build_available_actions(self, plans: list[str]) -> str:
        """Format available plans as lettered options."""
        lines = []
        for i, plan in enumerate(plans):
            lines.append(f"{chr(ord('A') + i)}. {plan}")
        return "\n".join(lines) + "\n"
