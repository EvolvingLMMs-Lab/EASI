"""Prompt builder for AI2-THOR Rearrangement task.

Constructs multi-modal prompts with observation images, GPS data,
action history, and structured JSON output schema.
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.json_repair import fix_json
from easi.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an embodied AI agent performing an object rearrangement task in a 3D indoor environment (AI2-THOR).

## Goal
{instruction}

## Available Actions
{action_list}

## Action Types
- **Navigation**: move_ahead (0.25m step), move_left/right/back, rotate_left/right (90°), look_up/look_down (30°), stand/crouch
- **Pickup**: pickup_<object_type> — picks up the nearest visible object of that type (must have empty hands)
- **Drop**: drop_held_object_with_snap — drops held object, snapping to goal position if close enough
- **Open/Close**: open_by_type_<type> — toggles openness of nearest visible cabinet/drawer/fridge/etc.
- **Done**: Signal that all objects are in their correct positions

## Strategy
1. Look around to survey the scene and identify misplaced objects
2. Navigate to a misplaced object
3. Pick it up with the matching pickup_<type> action
4. Navigate toward where it belongs (near its goal position)
5. Use drop_held_object_with_snap — it snaps the object to its goal if you're close
6. Repeat for remaining misplaced objects
7. Say "done" when finished

## Rules
- You can hold only ONE object at a time
- drop_held_object_with_snap works best when the goal location is visible
- If pickup fails, try moving closer or adjusting your view angle
- Grid movement: 0.25m steps, 90° rotations
"""

OUTPUT_SCHEMA = """\
Respond in this exact JSON format:
```json
{{
  "observation": "describe what you see",
  "reasoning": "what to do next and why",
  "plan": [
    {{"action_name": "<name>"}},
    ...
  ]
}}
```
Plan 1-5 actions. Use exact action names from the list above."""


class AI2THORRearrangement2023PromptBuilder:
    """Prompt builder for the rearrangement task."""

    def __init__(
        self,
        use_feedback: bool = True,
        chat_history: bool = True,
        message_window_len: int = 10,
        **kwargs,
    ):
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.message_window_len = message_window_len
        self._action_list_str = ""
        self._action_name_set: set[str] = set()

    def set_action_space(self, actions: list[str]):
        self._action_name_set = set(actions)
        parts = [f"{i}: {a}" for i, a in enumerate(actions)]
        self._action_list_str = ", ".join(parts)

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        instruction = memory.task_description or "Rearrange objects to match the goal."

        system_text = SYSTEM_PROMPT.format(
            instruction=instruction,
            action_list=self._action_list_str,
        )

        messages = [{"role": "system", "content": system_text}]

        # Chat history (if enabled)
        if self.chat_history and memory.steps:
            window = memory.steps[-self.message_window_len:]
            for step in window:
                user_content = self._make_history_content(step)
                messages.append({"role": "user", "content": user_content})
                if step.llm_response:
                    messages.append({
                        "role": "assistant",
                        "content": step.llm_response,
                    })

        # Current turn
        user_content = self._make_current_content(memory)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _make_history_content(self, step) -> list[dict]:
        content = []
        if step.observation and step.observation.rgb_path:
            img_url = _encode_image_base64(step.observation.rgb_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        text = "Observation."
        if self.use_feedback and step.feedback:
            text += f"\nFeedback: {step.feedback}"

        # GPS overlay from step info
        gps_text = self._format_gps(step.info)
        if gps_text:
            text += f"\n{gps_text}"

        content.append({"type": "text", "text": text})
        return content

    def _make_current_content(self, memory: AgentMemory) -> list[dict]:
        content = []

        if memory.current_observation and memory.current_observation.rgb_path:
            img_url = _encode_image_base64(memory.current_observation.rgb_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        if memory.is_first_turn:
            text = "First observation. Begin the rearrangement task."
        else:
            text = "Current observation."
            if self.use_feedback and memory.steps:
                last = memory.steps[-1]
                if last.feedback:
                    text += f"\nFeedback: {last.feedback}"

        # GPS from last step info
        if memory.steps:
            gps_text = self._format_gps(memory.steps[-1].info)
            if gps_text:
                text += f"\n{gps_text}"
            held = (memory.steps[-1].info or {}).get("held_object", "none")
            text += f"\nHolding: {held}"

        text += f"\n\n{OUTPUT_SCHEMA}"
        content.append({"type": "text", "text": text})
        return content

    def _format_gps(self, info: dict | None) -> str:
        if not info:
            return ""
        parts = []
        if "agent_x" in info:
            parts.append(
                f"Position: ({info['agent_x']:.2f}, {info['agent_y']:.2f}, {info['agent_z']:.2f})"
            )
        if "agent_rotation" in info:
            parts.append(f"Rotation: {info['agent_rotation']:.0f}")
        if "agent_horizon" in info:
            parts.append(f"Horizon: {info['agent_horizon']:.0f}")
        return "GPS: " + ", ".join(parts) if parts else ""

    def parse_response(
        self, llm_response: str, memory: AgentMemory
    ) -> list[Action]:
        try:
            fixed = fix_json(llm_response)
            data = json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM response as JSON")
            return [Action(action_name="done")]

        plan = data.get("plan", [])
        if not plan:
            return [Action(action_name="done")]

        actions = []
        for entry in plan:
            name = entry.get("action_name", "")
            if name in self._action_name_set:
                actions.append(Action(action_name=name))
            else:
                logger.warning("Unknown action '%s', stopping plan", name)
                break

        return actions if actions else [Action(action_name="done")]
