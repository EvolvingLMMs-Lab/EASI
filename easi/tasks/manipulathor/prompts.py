"""ManipulaTHOR prompt builder for EASI LLM agents.

Follows the EASI Standard Prompt Format Reference (docs/easi-prompt-format-reference.md).
Presents RGB image + GPS state sensors + action history.
Outputs JSON with executable plan of named actions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.tasks.manipulathor.actions import ACTION_SPACE
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# ── Action descriptions for the system prompt ──
ACTION_DESCRIPTIONS = {
    "MoveArmHeightP": "Move arm up by 0.05m",
    "MoveArmHeightM": "Move arm down by 0.05m",
    "MoveArmXP": "Move arm right by 0.05m",
    "MoveArmXM": "Move arm left by 0.05m",
    "MoveArmYP": "Move arm up by 0.05m (Y axis)",
    "MoveArmYM": "Move arm down by 0.05m (Y axis)",
    "MoveArmZP": "Move arm forward by 0.05m",
    "MoveArmZM": "Move arm backward by 0.05m",
    "MoveAheadContinuous": "Move agent forward by 0.2m",
    "RotateRightContinuous": "Rotate agent right by 45 degrees",
    "RotateLeftContinuous": "Rotate agent left by 45 degrees",
    "PickUpMidLevel": "Pick up the target object (arm must be close)",
    "DoneMidLevel": "Signal task completion (object must be at goal)",
}


def _build_action_list() -> str:
    """Build formatted action list for system prompt."""
    lines = []
    for name in ACTION_SPACE:
        desc = ACTION_DESCRIPTIONS.get(name, "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
## Role and Environment
You are a robotic arm agent in a kitchen environment (AI2-THOR simulator). \
Your task is to pick up a target object and move it to a goal location. You \
control a robotic arm mounted on a mobile base. Each arm movement moves \
0.05m. Navigation moves 0.2m forward or rotates 45 degrees. Maximum \
{{max_steps}} steps per episode.

## Observation Description
- **Object Position & Rotation**: Object 6D state (x,y,z position and \
rx,ry,rz rotation) relative to the agent.
- **Object-to-Goal Distance**: Absolute x,y,z distance from the object to \
the goal in agent frame.
- **Arm-to-Object Distance**: Absolute x,y,z distance from the arm tip to \
the object in agent frame.
- **Object Held**: Whether you are currently holding the object ("Yes"/"No").

## Available Actions
{action_list}

## Strategy
**Phase 1 — Approach the object:**
- Check Arm-to-Object Distance to see how far your arm tip is from the target.
- Positive X = object is to your right; positive Z = object is ahead.
- Use arm movements (MoveArmXP/XM, MoveArmYP/YM, MoveArmZP/ZM) to reduce \
distance on each axis.
- Use navigation (MoveAheadContinuous/RotateLeft/RotateRight) if the object \
is far (>0.5m).

**Phase 2 — Pick up the object:**
- When arm-to-object distance is small on all axes (<0.1m), use PickUpMidLevel.
- After pickup, check Object Held — it should read "Yes".

**Phase 3 — Navigate to the goal:**
- Check Object-to-Goal Distance. Navigate to reduce it. The object moves \
with you when held.
- Use arm movements for fine positioning when close.

**Phase 4 — Place and finish:**
- When Object-to-Goal Distance is small on all axes (<0.1m), use DoneMidLevel.

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the Available Actions list.
3. If previous actions failed, reason about why and try a different approach.
4. Do not repeatedly execute the same action sequence.
5. Keep your plan efficient and concise.
6. Output 1-5 actions per step.

## Response Format
Output a JSON object with exactly these 4 fields:
{{
    "visual_state_description": "Describe what you see in the image and \
the GPS state",
    "reasoning_and_reflection": "Your reasoning about the current state \
and what to do next",
    "language_plan": "Your plan in natural language",
    "executable_plan": [{{"action": "<action_name>"}}]
}}

You may include multiple actions in executable_plan. Actions execute \
sequentially."""


# ── Prompt builder ──────────────────────────────────────────────────────────

class ManipulaTHORPromptBuilder:
    """Formats ManipulaTHOR observations for VLM, parses responses into named actions."""

    def __init__(
        self,
        n_shot: int = 0,
        split: str = "test_seen",
        use_feedback: bool = True,
        chat_history: bool = False,
        message_window_len: int = 5,
        max_steps: int = 200,
        use_rgb: bool = True,
        use_gps: bool = True,
        use_depth: bool = False,
        action_history_len: int = 20,
    ):
        self.n_shot = n_shot
        self.split = split
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.message_window_len = message_window_len
        self.max_steps = max_steps
        self.use_rgb = use_rgb
        self.use_gps = use_gps
        self.use_depth = use_depth
        self.action_history_len = action_history_len

        # Load few-shot examples
        examples_file = Path(__file__).parent / "config" / "manipulathor_examples.json"
        if examples_file.exists():
            with open(examples_file) as f:
                self._examples = json.load(f)
        else:
            self._examples = []

        self._action_id_map = {name: i for i, name in enumerate(ACTION_SPACE)}
        self._id_action_map = {i: name for i, name in enumerate(ACTION_SPACE)}

    def set_action_space(self, actions: list) -> None:
        """Update action space (called by agent)."""
        pass  # ManipulaTHOR has a fixed action space

    def build_messages(self, memory: AgentMemory) -> list:
        """Build 2-message format: system + user."""
        import base64

        # Build system prompt
        system_text = SYSTEM_PROMPT.format(
            max_steps=self.max_steps,
            action_list=_build_action_list(),
        )
        messages = [{"role": "system", "content": system_text}]

        # Build user message
        content = []
        obs = memory.current_observation

        # Images first
        if self.use_rgb and obs and obs.rgb_path:
            try:
                with open(obs.rgb_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except FileNotFoundError:
                logger.warning("RGB image not found: %s", obs.rgb_path)

        if self.use_depth and obs and obs.depth_path:
            try:
                with open(obs.depth_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except FileNotFoundError:
                logger.warning("Depth image not found: %s", obs.depth_path)

        text_parts = []

        # Task
        text_parts.append(f"## Task\n{memory.task_description}")

        # Environment Feedback (GPS state)
        if self.use_gps and obs and obs.metadata:
            feedback_lines = []
            gps_fields = [
                ("relative_current_obj_state", "Object Position & Rotation (agent-relative)"),
                ("relative_obj_to_goal", "Object-to-Goal Distance"),
                ("relative_agent_arm_to_obj", "Arm-to-Object Distance"),
                ("pickedup_object", "Object Held"),
            ]
            for key, label in gps_fields:
                val = obs.metadata.get(key)
                if val is not None:
                    if key == "pickedup_object":
                        held = float(val) > 0.5
                        feedback_lines.append(f"{label}: {'Yes' if held else 'No'}")
                    else:
                        try:
                            arr = json.loads(val) if isinstance(val, str) else val
                            formatted = [f"{v:.3f}" for v in arr]
                            feedback_lines.append(f"{label}: [{', '.join(formatted)}]")
                        except (json.JSONDecodeError, TypeError):
                            feedback_lines.append(f"{label}: {val}")
            if feedback_lines:
                text_parts.append("## Environment Feedback\n" + "\n".join(feedback_lines))

        # Action History
        action_history = memory.action_history
        if action_history and self.action_history_len > 0:
            history = action_history[-self.action_history_len:]
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

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list:
        """Parse LLM JSON response into Action objects."""
        from easi.utils.json_repair import fix_json

        llm_response = fix_json(llm_response)

        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return []

        plan = data.get("executable_plan", [])
        if not isinstance(plan, list) or not plan:
            logger.warning("No executable_plan in LLM response")
            return []

        actions = []
        for entry in plan:
            if not isinstance(entry, dict):
                continue

            # Prefer "action" key (EASI standard), fall back to action_id/action_name
            action_name = entry.get("action", None)
            if action_name is None and "action_id" in entry:
                aid = entry["action_id"]
                if isinstance(aid, int) and 0 <= aid < len(ACTION_SPACE):
                    action_name = self._id_action_map[aid]
            if action_name is None:
                action_name = entry.get("action_name", "")

            if action_name in self._action_id_map:
                actions.append(Action(action_name=action_name))
            else:
                logger.warning("Invalid action: '%s'", action_name)
                break

        return actions

    def get_response_format(self, memory: AgentMemory) -> Optional[dict]:
        """Return JSON schema for structured output (optional)."""
        return None  # Use free-form JSON for now
