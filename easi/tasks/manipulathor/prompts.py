"""ManipulaTHOR prompt builder for EASI LLM agents.

Presents RGB image + GPS state sensors + action history.
Outputs JSON with executable plan of named action IDs.
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

# ── System prompt template ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robotic arm agent in a kitchen environment. Your task is to pick up a target object and move it to a goal location.

## Environment
- You control a robotic arm mounted on a mobile base in a kitchen (AI2-THOR simulator).
- You observe: an RGB image from the agent's camera{sensor_note}.
- Each arm movement moves 0.05m. Navigation moves 0.2m forward or rotates 45°.
- Maximum {max_steps} steps per episode.

## Available Actions (choose by ID 0-{max_id}):
{action_list}
{gps_description}
## Strategy for Using GPS Sensors
The GPS sensors give you precise spatial information in YOUR reference frame (agent-relative coordinates).

**Phase 1 — Approach the object:**
- Check "Arm-to-Object Distance": this tells you how far your arm tip is from the target object on each axis.
- Positive X means the object is to your right; positive Z means it is ahead of you.
- Use MoveArmXP/XM, MoveArmYP/YM, MoveArmZP/ZM to reduce arm-to-object distance on each axis.
- Use MoveAheadContinuous / RotateLeft / RotateRight if the object is far (>0.5m) — navigation is faster than arm movement.

**Phase 2 — Pick up the object:**
- When arm-to-object distance is small on all axes (<0.1m), use PickUpMidLevel (action 11).
- After pickup, check "Object Held" — it should read "Yes".

**Phase 3 — Navigate to the goal:**
- Check "Object-to-Goal Distance": this tells you how far the object is from the goal.
- Navigate (MoveAhead, Rotate) to reduce the distance. The object moves with you when held.
- Use arm movements for fine positioning when close to the goal.

**Phase 4 — Place and finish:**
- When Object-to-Goal Distance is small on all axes (<0.1m), use DoneMidLevel (action 12).

{examples}"""

OUTPUT_TEMPLATE = """

## Output Format
Respond in JSON with this exact structure:
```json
{{
  "visual_state_description": "Describe what you see in the image and the GPS state",
  "reasoning_and_reflection": "Your reasoning about the current state and what to do next",
  "language_plan": "Your plan in natural language",
  "executable_plan": [
    {{"action_id": <int>, "action_name": "<string>"}},
    ...
  ]
}}
```
Output 1-5 actions in the executable_plan. Choose action IDs from the available actions list."""


# ── Prompt builder ──────────────────────────────────────────────────────────

class ManipulaTHORPromptBuilder:
    """Formats ManipulaTHOR observations for VLM, parses responses into named actions."""

    def __init__(
        self,
        n_shot: int = 0,
        split: str = "test_seen",
        use_feedback: bool = True,
        chat_history: bool = False,
        max_steps: int = 200,
        use_rgb: bool = True,
        use_gps: bool = True,
        use_depth: bool = False,
    ):
        self.n_shot = n_shot
        self.split = split
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.max_steps = max_steps
        self.use_rgb = use_rgb
        self.use_gps = use_gps
        self.use_depth = use_depth

        # Load few-shot examples
        examples_file = Path(__file__).parent / "config" / "manipulathor_examples.json"
        if examples_file.exists():
            with open(examples_file) as f:
                self._examples = json.load(f)
        else:
            self._examples = []

        # Build action list string
        self._action_list = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(ACTION_SPACE)
        )
        self._action_id_map = {name: i for i, name in enumerate(ACTION_SPACE)}
        self._id_action_map = {i: name for i, name in enumerate(ACTION_SPACE)}

    def set_action_space(self, actions: list) -> None:
        """Update action space (called by agent)."""
        pass  # ManipulaTHOR has a fixed action space

    def build_messages(self, memory: AgentMemory) -> list:
        """Build OpenAI message format from agent memory."""
        prompt = self._build_prompt_text(memory)
        return self._wrap_as_user_message(prompt, memory.current_observation)

    def _build_prompt_text(self, memory: AgentMemory) -> str:
        """Build the full prompt text."""
        max_id = len(ACTION_SPACE) - 1

        # Build examples section
        examples_str = ""
        if self.n_shot >= 1 and self._examples:
            examples_str = "\n\n## Examples\n" + "\n\n".join(
                f"### Example {i+1}:\n{ex}"
                for i, ex in enumerate(self._examples[:self.n_shot])
            )

        # Build sensor note for system prompt
        sensor_parts = []
        if self.use_gps:
            sensor_parts.append("GPS-like state sensors showing spatial relationships")
        if self.use_depth:
            sensor_parts.append("a depth image (colormap)")
        sensor_note = (", and " + ", ".join(sensor_parts)) if sensor_parts else ""

        # GPS description section (only if GPS enabled)
        gps_description = ""
        if self.use_gps:
            gps_description = """
## GPS State Sensors
At each step you receive:
- **Object State** (6D): object position (x,y,z) and rotation (rx,ry,rz) relative to agent
- **Object-to-Goal Distance** (3D): absolute x,y,z distance from object to goal in agent frame
- **Arm-to-Object Distance** (3D): absolute x,y,z distance from arm tip to object in agent frame
- **Object Held**: whether you are currently holding the object

"""

        # System prompt
        prompt = SYSTEM_PROMPT.format(
            max_steps=self.max_steps,
            max_id=max_id,
            action_list=self._action_list,
            examples=examples_str,
            sensor_note=sensor_note,
            gps_description=gps_description,
        )

        # Task instruction
        task_desc = memory.task_description
        prompt += f"\n\n## Current Task: {task_desc}"

        # GPS state from current observation metadata (only if GPS enabled)
        obs = memory.current_observation
        if self.use_gps and obs and obs.metadata:
            prompt += "\n\n## Current GPS State:"
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
                        prompt += f"\n- {label}: {'Yes' if held else 'No'}"
                    else:
                        try:
                            arr = json.loads(val) if isinstance(val, str) else val
                            formatted = [f"{v:.3f}" for v in arr]
                            prompt += f"\n- {label}: [{', '.join(formatted)}]"
                        except (json.JSONDecodeError, TypeError):
                            prompt += f"\n- {label}: {val}"

        # Action history + feedback
        action_history = getattr(memory, 'action_history', [])
        if action_history:
            prompt += "\n\n## Action History:"
            for i, (action_name, feedback) in enumerate(action_history):
                action_id = self._action_id_map.get(action_name, -1)
                if self.use_feedback:
                    prompt += f"\nStep {i+1}: action {action_id} ({action_name}) — {feedback}"
                else:
                    prompt += f"\nStep {i+1}: action {action_id} ({action_name})"

            prompt += "\n\nBased on the action history and current state, plan your next actions."

        prompt += OUTPUT_TEMPLATE
        return prompt

    def _wrap_as_user_message(self, prompt: str, observation) -> list:
        """Wrap prompt + images as OpenAI user message.

        Includes RGB and/or depth images based on use_rgb and use_depth toggles.
        """
        import base64

        content = []

        def _add_image(path, label):
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except FileNotFoundError:
                logger.warning("%s image not found: %s", label, path)

        # Add RGB image (before text, matching EASI convention)
        if self.use_rgb and observation and observation.rgb_path:
            _add_image(observation.rgb_path, "RGB")

        # Add depth colormap image if enabled
        if self.use_depth and observation and observation.depth_path:
            _add_image(observation.depth_path, "Depth")

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

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

            action_name = None
            if "action_id" in entry:
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
