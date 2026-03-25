"""VLN-CE R2R prompt builder.

Follows the EASI Standard Prompt Format Reference (docs/easi-prompt-format-reference.md).
Single front camera, 4 actions (move_forward, turn_left, turn_right, stop).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.json_repair import fix_json
from easi.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
## Role and Environment
You are a robot navigating in a 3D indoor environment. You observe the \
environment through a front-facing camera and must follow natural language \
instructions to navigate to a goal location.

## Observation Description
- **Distance to goal**: Geodesic (shortest walkable path) distance in meters \
to the goal location. Decreases as you get closer.

## Available Actions
- move_forward: Move forward by 0.25 meters
- turn_left: Turn left by 15 degrees
- turn_right: Turn right by 15 degrees
- stop: Stop and end navigation (use ONLY when you believe you have reached \
the destination described in the instruction)

## Strategy
1. Carefully read the navigation instruction
2. Observe your surroundings in the image
3. Follow the instruction step by step, matching landmarks and directions mentioned
4. Use move_forward to advance and turn_left/turn_right to change direction
5. Use stop ONLY when you are confident you have reached the described destination

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.

## Response Format
Output a JSON object with exactly these 4 fields:
{
    "visual_state_description": "Describe what you see in the current image",
    "reasoning_and_reflection": "Reason about your situation, reflect on \
history and feedback",
    "language_plan": "Describe your next navigation plan in natural language",
    "executable_plan": [{"action": "<action_name>"}]
}

You may include multiple actions in executable_plan. Actions execute \
sequentially."""


def _encode_image_base64(image_path: str) -> str | None:
    """Encode image file to base64 data URL."""
    p = Path(image_path)
    if not p.exists():
        return None
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _validate_action(name: str, action_space: list[str]) -> str | None:
    """Validate and normalize action name."""
    if name in action_space:
        return name
    lower = name.lower().strip()
    for a in action_space:
        if a.lower() == lower:
            return a
    return None


class VLNCEPromptBuilder:
    """Prompt builder for VLN-CE R2R benchmark."""

    def __init__(
        self,
        use_feedback: bool = True,
        use_geo_distance: bool = True,
        action_history_len: int = 20,
        chat_history: bool = False,
        message_window_len: int = 5,
    ):
        self.use_feedback = use_feedback
        self.use_geo_distance = use_geo_distance
        self.action_history_len = action_history_len
        self.chat_history = chat_history
        self.message_window_len = message_window_len
        self._actions = []

    def set_action_space(self, actions: list[str]) -> None:
        self._actions = list(actions)

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        if self._actions != memory.action_space:
            self.set_action_space(memory.action_space)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        content = []

        # Image first
        obs = memory.current_observation
        if obs and obs.rgb_path:
            img_url = _encode_image_base64(obs.rgb_path)
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

        text_parts = []

        # Task
        text_parts.append(f"## Task\n{memory.task_description}")

        # Environment Feedback
        if self.use_feedback and obs and obs.metadata:
            feedback_lines = []
            if self.use_geo_distance:
                geo = obs.metadata.get("geo_distance")
                if geo is not None and geo != "null":
                    feedback_lines.append(f"Distance to goal: {float(geo):.1f}m")
            else:
                # Only use generic feedback when geo_distance is not shown
                # (bridge puts distance into both fields, avoid duplication)
                feedback = obs.metadata.get("feedback", "")
                if feedback:
                    feedback_lines.append(feedback)
            if feedback_lines:
                text_parts.append("## Environment Feedback\n" + "\n".join(feedback_lines))

        # Action History
        if memory.action_history and self.action_history_len > 0:
            history = memory.action_history[-self.action_history_len:]
            history_lines = []
            for i, (action_name, feedback) in enumerate(history):
                if self.use_feedback and feedback:
                    history_lines.append(f"Step {i}: {action_name} -> {feedback}")
                else:
                    history_lines.append(f"Step {i}: {action_name}")
            if history_lines:
                text_parts.append(
                    f"## Action History (last {len(history_lines)} steps)\n"
                    + "\n".join(history_lines)
                )

        # Chat History
        if self.chat_history and memory.steps:
            responses = [
                s.llm_response for s in memory.steps
                if s.llm_response is not None
            ][-self.message_window_len:]
            if responses:
                chat_lines = []
                offset = len(memory.steps) - len(responses)
                for j, resp in enumerate(responses):
                    chat_lines.append(f"[Step {offset + j} Response]\n{resp}")
                text_parts.append(
                    f"## Chat History (last {len(responses)} responses)\n"
                    + "\n\n".join(chat_lines)
                )

        # Response format reminder
        text_parts.append("Respond with the JSON format specified above.")

        content.append({"type": "text", "text": "\n\n".join(text_parts)})
        messages.append({"role": "user", "content": content})

        return messages

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM JSON response into validated Actions."""
        llm_response = fix_json(llm_response)

        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            return []

        plan = data.get("executable_plan", [])
        if not isinstance(plan, list) or not plan:
            logger.warning("No executable_plan in LLM response")
            return []

        actions = []
        for entry in plan:
            if not isinstance(entry, dict):
                continue
            action_name = entry.get("action", entry.get("action_name", ""))
            validated = _validate_action(action_name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Skipping invalid action: '%s'", action_name)
                break

        return actions
