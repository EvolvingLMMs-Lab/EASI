"""Prompt builder for AI2-THOR Rearrangement task.

Constructs multi-modal prompts with observation images, GPS data,
action history, and structured JSON output schema.

Sensor inputs (RGB, Depth, Goal image, GPS) are individually toggleable
via prompt_builder_kwargs. The system prompt and per-step image labels
adapt automatically to reflect which sensors are active.
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.json_repair import fix_json
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# --- System prompt template (observations section is injected dynamically) ---

SYSTEM_PROMPT = """\
You are an embodied AI agent performing an object rearrangement task in a 3D indoor environment (AI2-THOR).

## Goal
{instruction}

**Important:** You are evaluated on the final state of ALL objects in the scene, not just the ones listed above. If you accidentally bump into or displace objects that were already in their correct positions, it counts against you. Navigate carefully.

## Observations
At each step you receive the following sensor inputs:
{observation_description}

## Available Actions
{action_list}

## Action Types
- **Navigation**: move_ahead (0.25m step), move_left/right/back, rotate_left/right (90°), look_up/look_down (30°), stand/crouch
- **Pickup**: pickup_<object_type> — picks up the nearest visible object of that type (must have empty hands)
- **Drop**: drop_held_object_with_snap — drops held object, snapping to goal position if close enough
- **Open/Close**: open_by_type_<type> — toggles openness of nearest visible cabinet/drawer/fridge/etc.
- **Done**: Signal that all objects are in their correct positions

## Strategy
{strategy}

## Rules
- You can hold only ONE object at a time
- drop_held_object_with_snap works best when the goal location is visible
- If pickup fails, try moving closer or adjusting your view angle
- Grid movement: 0.25m steps, 90° rotations
- **Do NOT disturb objects already in their correct positions** — avoid bumping into furniture or pushing items while navigating. Take careful, minimal paths.
"""

OUTPUT_SCHEMA = """\
Respond in this exact JSON format:
```json
{{
  "observation": "describe what you see in the observation images",
  "reasoning": "what to do next and why",
  "plan": [
    {{"action_name": "<name>"}},
    ...
  ]
}}
```
Plan 1-5 actions. Use exact action names from the list above."""


class AI2THORRearrangement2023PromptBuilder:
    """Prompt builder for the rearrangement task.

    Sensor toggles (use_rgb, use_depth, use_gps, use_goal_image) control
    both what appears in the system prompt description and what sensor
    data is included in per-step messages.
    """

    def __init__(
        self,
        use_feedback: bool = True,
        chat_history: bool = True,
        message_window_len: int = 10,
        use_rgb: bool = True,
        use_depth: bool = False,
        use_gps: bool = True,
        use_goal_image: bool = True,
        **kwargs,
    ):
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.message_window_len = message_window_len
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_gps = use_gps
        self.use_goal_image = use_goal_image
        self._action_list_str = ""
        self._action_name_set: set[str] = set()

    def set_action_space(self, actions: list[str]):
        self._action_name_set = set(actions)
        parts = [f"{i}: {a}" for i, a in enumerate(actions)]
        self._action_list_str = ", ".join(parts)

    # --- System prompt helpers ---

    def _build_observation_description(self) -> str:
        """Build the observation section of the system prompt based on active sensors."""
        sections = []
        idx = 1

        if self.use_rgb:
            sections.append(
                f"- **RGB Image — Current Scene** (Image {idx}): A first-person "
                f"camera view showing the environment as it looks RIGHT NOW "
                f"(shuffled state). Objects may be misplaced relative to the goal."
            )
            idx += 1

        if self.use_depth:
            sections.append(
                f"- **Depth Image** (Image {idx}): A grayscale depth map from "
                f"the agent's viewpoint. Brighter pixels are closer, darker "
                f"pixels are farther away. Use this to judge distances to "
                f"objects and obstacles."
            )
            idx += 1

        if self.use_goal_image:
            sections.append(
                f"- **RGB Image — Goal Scene** (Image {idx}): A first-person "
                f"camera view from the SAME position showing what the "
                f"environment SHOULD look like (target state). Compare with "
                f"the current scene to identify misplaced objects."
            )
            idx += 1

        if self.use_gps:
            sections.append(
                "- **GPS Data** (text): The agent's 3D position (x, y, z) in "
                "meters where y is the vertical axis (height). Yaw rotation in "
                "degrees (0°–360°). Horizon angle: 0° = looking straight "
                "ahead, positive = looking down, negative = looking up. Also "
                "reports which object the agent is currently holding, if any."
            )

        return "\n".join(sections) if sections else "No sensor inputs configured."

    def _build_strategy(self) -> str:
        """Build the strategy section based on whether goal images are available."""
        if self.use_goal_image:
            return (
                "1. Compare the current and goal images to identify misplaced objects\n"
                "2. Navigate to a misplaced object\n"
                "3. Pick it up with the matching pickup_<type> action\n"
                "4. Navigate toward where it belongs (compare with goal image)\n"
                "5. Use drop_held_object_with_snap — it snaps the object to its goal if you're close\n"
                "6. Repeat for remaining misplaced objects\n"
                '7. Say "done" when finished'
            )
        return (
            "1. Explore the environment to identify objects that appear misplaced\n"
            "2. Navigate to a misplaced object\n"
            "3. Pick it up with the matching pickup_<type> action\n"
            "4. Navigate to where it should go based on context\n"
            "5. Use drop_held_object_with_snap — it snaps the object to its goal if you're close\n"
            "6. Repeat for remaining misplaced objects\n"
            '7. Say "done" when finished'
        )

    # --- Message building ---

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        # Lazy-init action space from memory (set by agent constructor)
        if not self._action_list_str and memory.action_space:
            self.set_action_space(memory.action_space)

        instruction = memory.task_description or "Rearrange objects to match the goal."

        system_text = SYSTEM_PROMPT.format(
            instruction=instruction,
            observation_description=self._build_observation_description(),
            action_list=self._action_list_str,
            strategy=self._build_strategy(),
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

    def _add_images(
        self, content: list, rgb_path: str | None, metadata: dict,
    ) -> str:
        """Append sensor images to *content* list.

        Returns a parenthetical label string like
        ``" (Image 1: RGB current, Image 2: Depth, Image 3: RGB goal)"``
        that describes the images actually added.
        """
        labels: list[str] = []
        idx = 1

        # 1. RGB current scene
        if self.use_rgb and rgb_path:
            img_url = _encode_image_base64(rgb_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })
                labels.append(f"Image {idx}: RGB current")
                idx += 1

        # 2. Depth map
        if self.use_depth:
            depth_path = metadata.get("depth_path")
            if depth_path:
                img_url = _encode_image_base64(depth_path)
                if img_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    })
                    labels.append(f"Image {idx}: Depth")
                    idx += 1

        # 3. Goal scene (walkthrough state from same viewpoint)
        if self.use_goal_image:
            goal_path = metadata.get("goal_rgb_path")
            if goal_path:
                img_url = _encode_image_base64(goal_path)
                if img_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    })
                    labels.append(f"Image {idx}: RGB goal")
                    idx += 1

        return f" ({', '.join(labels)})" if labels else ""

    def _make_history_content(self, step) -> list[dict]:
        content: list[dict] = []
        metadata = step.observation.metadata if step.observation else {}
        rgb_path = step.observation.rgb_path if step.observation else None

        img_label = self._add_images(content, rgb_path, metadata)

        text = f"Observation{img_label}."
        if self.use_feedback and step.feedback:
            text += f"\nFeedback: {step.feedback}"

        # GPS overlay from observation metadata
        if self.use_gps:
            gps_text = self._format_gps(metadata)
            if gps_text:
                text += f"\n{gps_text}"

        content.append({"type": "text", "text": text})
        return content

    def _make_current_content(self, memory: AgentMemory) -> list[dict]:
        content: list[dict] = []
        metadata = (
            memory.current_observation.metadata
            if memory.current_observation else {}
        )
        rgb_path = (
            memory.current_observation.rgb_path
            if memory.current_observation else None
        )

        img_label = self._add_images(content, rgb_path, metadata)

        if memory.is_first_turn:
            text = f"First observation{img_label}. Begin the rearrangement task."
        else:
            text = f"Current observation{img_label}."
            if self.use_feedback and memory.steps:
                last = memory.steps[-1]
                if last.feedback:
                    text += f"\nFeedback: {last.feedback}"

        # GPS and held-object from last step's observation metadata
        if memory.steps:
            last_metadata = (
                memory.steps[-1].observation.metadata
                if memory.steps[-1].observation else {}
            )
            if self.use_gps:
                gps_text = self._format_gps(last_metadata)
                if gps_text:
                    text += f"\n{gps_text}"
            held = last_metadata.get("held_object", "none")
            text += f"\nHolding: {held}"

        text += f"\n\n{OUTPUT_SCHEMA}"
        content.append({"type": "text", "text": text})
        return content

    def _format_gps(self, metadata: dict | None) -> str:
        if not metadata:
            return ""
        parts = []
        if "agent_x" in metadata:
            x = float(metadata["agent_x"])
            y = float(metadata["agent_y"])
            z = float(metadata["agent_z"])
            parts.append(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
        if "agent_rotation" in metadata:
            parts.append(f"Rotation: {float(metadata['agent_rotation']):.0f}\u00b0")
        if "agent_horizon" in metadata:
            parts.append(f"Horizon: {float(metadata['agent_horizon']):.0f}\u00b0")
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
