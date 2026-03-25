"""Prompt builder for AI2-THOR Rearrangement task.

Follows the EASI Standard Prompt Format Reference (docs/easi-prompt-format-reference.md).

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


class AI2THORRearrangement2023PromptBuilder:
    """Prompt builder for the rearrangement task.

    Sensor toggles (use_rgb, use_depth, use_gps, use_goal_image) control
    both what appears in the system prompt description and what sensor
    data is included in per-step messages.
    """

    def __init__(
        self,
        use_feedback: bool = True,
        action_history_len: int = 20,
        chat_history: bool = True,
        message_window_len: int = 10,
        use_rgb: bool = True,
        use_depth: bool = False,
        use_gps: bool = True,
        use_goal_image: bool = True,
        **kwargs,
    ):
        self.use_feedback = use_feedback
        self.action_history_len = action_history_len
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
        parts = [f"- {a}" for a in actions]
        self._action_list_str = "\n".join(parts)

    # --- System prompt ---

    def _build_system_prompt(self) -> str:
        """Build the full system prompt following EASI standard sections."""
        sections = []

        # Role and Environment
        sections.append(
            "## Role and Environment\n"
            "You are an embodied AI agent performing an object rearrangement "
            "task in a 3D indoor environment (AI2-THOR). You must rearrange "
            "objects to match a goal state. You are evaluated on ALL objects "
            "in the scene — avoid accidentally displacing objects already in "
            "their correct positions."
        )

        # Observation Description
        obs_desc = self._build_observation_description()
        if obs_desc:
            sections.append(f"## Observation Description\n{obs_desc}")

        # Available Actions
        sections.append(
            f"## Available Actions\n{self._action_list_str}\n\n"
            "**Action Types:**\n"
            "- **Navigation**: move_ahead (0.25m step), move_left/right/back, "
            "rotate_left/right (90 deg), look_up/look_down (30 deg), stand/crouch\n"
            "- **Pickup**: pickup_<object_type> — picks up the nearest visible "
            "object of that type (must have empty hands)\n"
            "- **Drop**: drop_held_object_with_snap — drops held object, "
            "snapping to goal position if close enough\n"
            "- **Open/Close**: open_by_type_<type> — toggles openness of "
            "nearest visible cabinet/drawer/fridge/etc.\n"
            "- **Done**: Signal that all objects are in their correct positions"
        )

        # Strategy
        sections.append(f"## Strategy\n{self._build_strategy()}")

        # Guidelines
        sections.append(
            "## Guidelines\n"
            "1. Always output at least one action in executable_plan.\n"
            "2. Only use actions from the Available Actions list.\n"
            "3. If previous actions failed, reason about why and try a different approach.\n"
            "4. Do not repeatedly execute the same action sequence.\n"
            "5. Keep your plan efficient and concise.\n"
            "6. You can hold only ONE object at a time.\n"
            "7. Do NOT disturb objects already in their correct positions."
        )

        # Response Format
        sections.append(
            "## Response Format\n"
            "Output a JSON object with exactly these 4 fields:\n"
            "{\n"
            '    "visual_state_description": "Describe what you see in the observation images",\n'
            '    "reasoning_and_reflection": "What to do next and why",\n'
            '    "language_plan": "Your plan in natural language",\n'
            '    "executable_plan": [{"action": "<action_name>"}]\n'
            "}\n\n"
            "You may include 1-5 actions in executable_plan. Actions execute "
            "sequentially."
        )

        return "\n\n".join(sections)

    def _build_observation_description(self) -> str:
        """Build the observation description based on active sensors."""
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
                f"pixels are farther away."
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
                "degrees (0-360). Horizon angle: 0 = looking straight "
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
                "5. Use drop_held_object_with_snap — it snaps the object to its goal if you are close\n"
                "6. Repeat for remaining misplaced objects\n"
                '7. Say "done" when finished'
            )
        return (
            "1. Explore the environment to identify objects that appear misplaced\n"
            "2. Navigate to a misplaced object\n"
            "3. Pick it up with the matching pickup_<type> action\n"
            "4. Navigate to where it should go based on context\n"
            "5. Use drop_held_object_with_snap — it snaps the object to its goal if you are close\n"
            "6. Repeat for remaining misplaced objects\n"
            '7. Say "done" when finished'
        )

    # --- Message building ---

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        # Lazy-init action space from memory
        if not self._action_list_str and memory.action_space:
            self.set_action_space(memory.action_space)

        system_text = self._build_system_prompt()
        messages = [{"role": "system", "content": system_text}]

        # Build user message
        user_content = self._make_user_content(memory)
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

    def _make_user_content(self, memory: AgentMemory) -> list[dict]:
        """Build user message content: images + text sections."""
        content: list[dict] = []
        obs = memory.current_observation
        metadata = obs.metadata if obs else {}
        rgb_path = obs.rgb_path if obs else None

        # Images first
        img_label = self._add_images(content, rgb_path, metadata)

        text_parts = []

        # Task
        instruction = memory.task_description or "Rearrange objects to match the goal."
        text_parts.append(f"## Task\n{instruction}")

        # Environment Feedback (GPS + held object)
        feedback_lines = []
        if self.use_gps and memory.steps:
            last_metadata = (
                memory.steps[-1].observation.metadata
                if memory.steps[-1].observation else {}
            )
            gps_text = self._format_gps(last_metadata)
            if gps_text:
                feedback_lines.append(gps_text)
            held = last_metadata.get("held_object", "none")
            feedback_lines.append(f"Holding: {held}")
        elif self.use_gps and obs and metadata:
            gps_text = self._format_gps(metadata)
            if gps_text:
                feedback_lines.append(gps_text)

        if self.use_feedback and memory.steps:
            last = memory.steps[-1]
            if last.feedback:
                feedback_lines.append(f"Last action result: {last.feedback}")

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

        # Chat History (text section, not multi-turn)
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
        if memory.is_first_turn:
            text_parts.append(
                f"First observation{img_label}. Begin the rearrangement task.\n\n"
                "Respond with the JSON format specified above."
            )
        else:
            text_parts.append("Respond with the JSON format specified above.")

        content.append({"type": "text", "text": "\n\n".join(text_parts)})
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

        # Accept both "executable_plan" (standard) and "plan" (legacy)
        plan = data.get("executable_plan", data.get("plan", []))
        if not plan:
            return [Action(action_name="done")]

        actions = []
        for entry in plan:
            # Accept "action" (standard), "action_name" (legacy)
            name = entry.get("action", entry.get("action_name", ""))
            if name in self._action_name_set:
                actions.append(Action(action_name=name))
            else:
                logger.warning("Unknown action '%s', stopping plan", name)
                break

        return actions if actions else [Action(action_name="done")]
