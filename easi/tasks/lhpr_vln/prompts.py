"""LHPR-VLN prompt builder for LLM-based navigation agents.

Aligned with the EmbodiedBench 4-field response format used by EB-Navigation,
EB-Habitat, and EB-Alfred, adapted for LHPR-VLN multi-subtask navigation.

Builds prompts with:
- 3 RGB camera views (left -60deg, front 0deg, right +60deg)
- Task instruction with subtask targets
- Toggleable environmental feedback (geodesic distance, subtask progress,
  target name, agent position, target coordinate)
- Action history with execution feedback
- 4-action space: move_forward, turn_left, turn_right, stop
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# System prompt adapted for LHPR-VLN multi-subtask navigation.
# Three format placeholders: max_action_id, action_list, examples.
LHPRVLN_SYSTEM_PROMPT = '''## You are a robot navigating in a 3D indoor environment. You observe the environment through three camera views (left at -60 degrees, front at 0 degrees, right at +60 degrees) and must navigate to sequential target objects.

## The available action id (0 ~ {}) and action names are: {}.

*** Strategy ***

1. Identify Target Objects: Clearly describe what you see in the three camera views. Look for the current target object or visual cues (room types, furniture, doorways) that indicate its likely location.

2. Navigate using Move forward and Turn left/right as your main strategy, since any point can be reached through a combination of those. When planning movement, reason based on the target object location and obstacles around you.

3. Multi-Subtask Navigation: This task requires you to navigate to multiple target objects in sequence. The "stop" action does NOT end the episode — it marks the CURRENT subtask as complete and advances you to the NEXT target. You must use "stop" once for each target. The episode only ends after you have used "stop" for every target.

4. Use Stop Carefully: Only use "stop" when you are confident you are within 1 meter of the current target object. Using "stop" too far from the target will mark that subtask as failed, but the episode will still continue to the next target.

5. Efficient Exploration: If the target is not visible, explore by turning to survey the environment. Use the three camera views to gather spatial information before deciding which direction to move. When close to the target, you can choose to take lesser actions each step to finetune your trajectories, for example taking 1 action at a time when you are 2m away from target and only use stop when you are less than 1m away from the target.

{}

----------

'''

# Output template matching EmbodiedBench format.
OUTPUT_TEMPLATE = "\n" \
"The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}\n" \
"The fields in above JSON follows the purpose below:\n" \
"1. visual_state_description is for description of current state from the three visual images (left, front, right views), \n" \
"2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \n" \
"3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \n" \
"4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.\n" \
"5. keep your plan efficient and concise.\n" \
"!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.\n"

# JSON schema for API-level enforcement.
LHPRVLN_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "lhprvln_planning",
        "schema": {
            "type": "object",
            "properties": {
                "visual_state_description": {
                    "type": "string",
                    "description": "Description of current state from the three visual images (left, front, right camera views)",
                },
                "reasoning_and_reflection": {
                    "type": "string",
                    "description": "Summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
                },
                "language_plan": {
                    "type": "string",
                    "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name",
                },
                "executable_plan": {
                    "type": "array",
                    "description": "A list of actions needed to achieve the user instruction, with each action having an action ID and a name. Do not output empty list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_id": {
                                "type": "integer",
                                "description": "The action ID to select from the available actions given by the prompt",
                            },
                            "action_name": {
                                "type": "string",
                                "description": "The name of the action",
                            },
                        },
                        "required": ["action_id", "action_name"],
                    },
                },
            },
            "required": [
                "visual_state_description",
                "reasoning_and_reflection",
                "language_plan",
                "executable_plan",
            ],
        },
    },
}


class LHPRVLNPromptBuilder:
    """Prompt builder for LHPR-VLN navigation tasks.

    Matches the EmbodiedBench 4-field response format with LHPR-VLN-specific
    environmental feedback toggles. Supports stateless and chat history modes.

    Environmental feedback toggles (all configurable in _base.yaml):
    - use_feedback: Include action execution feedback in history
    - use_geo_distance: Show geodesic distance to current target
    - use_subtask_progress: Show subtask stage/total + current target name
    - use_agent_position: Show agent 3D position
    - use_target_coordinate: Show target 3D coordinate
    """

    def __init__(
        self,
        use_feedback: bool = True,
        use_geo_distance: bool = True,
        use_subtask_progress: bool = True,
        use_agent_position: bool = False,
        use_target_coordinate: bool = False,
        action_history_len: int = -1,
        chat_history: bool = False,
        message_window_len: int = 5,
        **kwargs,
    ):
        self.use_feedback = use_feedback
        self.use_geo_distance = use_geo_distance
        self.use_subtask_progress = use_subtask_progress
        self.use_agent_position = use_agent_position
        self.use_target_coordinate = use_target_coordinate
        self.action_history_len = action_history_len
        self.chat_history = chat_history
        self.message_window_len = message_window_len

        # Action space state
        self._actions: list[str] = []
        self._action_str: str = ""
        self._action_id_map: dict[str, int] = {}
        self._id_action_map: dict[int, str] = {}

    def set_action_space(self, actions: list[str]) -> None:
        """Update action space."""
        self._actions = list(actions)
        self._action_str = self._build_action_list_str(actions)
        self._action_id_map = {name: i for i, name in enumerate(actions)}
        self._id_action_map = {i: name for i, name in enumerate(actions)}

    def action_name_to_id(self, name: str) -> int | None:
        return self._action_id_map.get(name)

    def action_id_to_name(self, action_id: int) -> str | None:
        return self._id_action_map.get(action_id)

    @staticmethod
    def _build_action_list_str(actions: list[str]) -> str:
        parts = ''
        for i in range(len(actions)):
            parts += '\naction id ' + str(i) + ': ' + str(actions[i])
            if i < len(actions) - 1:
                parts += ', '
        return parts

    # ---- PromptBuilderProtocol methods ----

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        if not self._action_id_map or self._actions != memory.action_space:
            self.set_action_space(memory.action_space)

        if self.chat_history:
            return self._build_chat_history_messages(memory)
        else:
            return self._build_stateless_messages(memory)

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM response into validated actions."""
        from easi.utils.json_repair import fix_json
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

            if "action_id" in entry:
                action_id = entry["action_id"]
                action_name = self.action_id_to_name(action_id)
                if action_name is None:
                    action_name = entry.get("action_name", "")
            else:
                action_name = entry.get("action", entry.get("action_name", ""))

            validated = validate_action_name(action_name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Skipping invalid action: '%s'", action_name)
                break

        return actions

    def get_response_format(self, memory: AgentMemory) -> dict:
        return LHPRVLN_RESPONSE_SCHEMA

    # ---- Stateless mode ----

    def _build_stateless_messages(self, memory: AgentMemory) -> list[dict]:
        prompt = self._build_prompt_text(
            memory.task_description, memory.action_history,
            memory.current_observation,
        )
        return self._wrap_as_user_message(prompt, memory.current_observation)

    def _build_prompt_text(
        self,
        task_description: str,
        action_history: list[tuple[str, str]],
        observation=None,
    ) -> str:
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        prompt = LHPRVLN_SYSTEM_PROMPT.format(
            max_id, self._action_str, '',
        )

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

        # Add environmental feedback section
        env_feedback = self._format_env_feedback(observation)
        if env_feedback:
            prompt += env_feedback

        if len(action_history) == 0:
            prompt += self._make_first_prompt_suffix(max_id)
        else:
            history_text = self._format_action_history(action_history)
            if history_text:
                prompt += history_text
            prompt += f"\n\n{self._make_following_prompt_suffix(max_id)}"

        return prompt

    # ---- Chat history mode ----

    def _build_chat_history_messages(self, memory: AgentMemory) -> list[dict]:
        if memory.is_first_turn:
            return self._build_stateless_messages(memory)

        messages: list[dict] = []

        for step_idx, step in enumerate(memory.steps):
            if step.llm_response is None:
                continue

            if step_idx == 0:
                history_up_to = []
            else:
                history_up_to = [
                    (s.action.action_name, s.feedback)
                    for s in memory.steps[:step_idx]
                    if s.action and s.feedback is not None
                ]

            if step_idx == 0:
                prompt = self._build_prompt_text(
                    memory.task_description, [], step.observation,
                )
            else:
                prompt = self._build_chat_subsequent_prompt(
                    memory.task_description, history_up_to, step.observation,
                )

            messages.extend(
                self._wrap_as_user_message(prompt, step.observation)
            )
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": step.llm_response}],
            })

        # Current turn
        all_history = memory.action_history
        current_prompt = self._build_chat_subsequent_prompt(
            memory.task_description, all_history, memory.current_observation,
        )
        messages.extend(
            self._wrap_as_user_message(current_prompt, memory.current_observation)
        )

        return messages[-self.message_window_len:]

    def _build_chat_subsequent_prompt(
        self,
        task_description: str,
        action_history: list[tuple[str, str]],
        observation=None,
    ) -> str:
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        prompt = LHPRVLN_SYSTEM_PROMPT.format(
            max_id, self._action_str, '',
        )

        prompt += f'\n\n## The human instruction is: {user_instruction}.'

        env_feedback = self._format_env_feedback(observation)
        if env_feedback:
            prompt += env_feedback

        history_text = self._format_action_history(action_history)
        if history_text:
            prompt += history_text
        prompt += f"\n\n{self._make_following_prompt_suffix(max_id)}"

        return prompt

    # ---- Helpers ----

    def _format_env_feedback(self, observation) -> str:
        """Format toggleable environmental feedback from observation metadata."""
        if observation is None or not observation.metadata:
            return ""

        meta = observation.metadata
        parts = []

        if self.use_subtask_progress:
            stage = meta.get("subtask_stage", "")
            total = meta.get("subtask_total", "")
            target = meta.get("current_target", "")
            if stage and total:
                stage_int = int(float(stage)) + 1
                total_int = int(float(total))
                line = f"Current subtask: {stage_int}/{total_int}"
                if target:
                    line += f", target object: {target}"
                parts.append(line)

        if self.use_geo_distance:
            distance = meta.get("current_geo_distance", "")
            if distance:
                parts.append(f"Geodesic distance to current target: {float(distance):.2f}m")

        if self.use_agent_position:
            pos_str = meta.get("agent_position", "")
            if pos_str:
                pos = json.loads(pos_str)
                parts.append(f"Agent position (x, y, z): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        if self.use_target_coordinate:
            coord_str = meta.get("target_coordinate", "")
            if coord_str:
                coord = json.loads(coord_str)
                parts.append(f"Target coordinate (x, y, z): ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")

        if not parts:
            return ""

        return "\n\n## Environmental Feedback:\n" + "\n".join(parts)

    def _make_first_prompt_suffix(self, max_id: int) -> str:
        return (
            f'''\nTo achieve the task, 1. Reason about the current visual state from the three camera views and your final goal, and 2. Reflect on the effect of previous actions.'''
            f'''\nAim for about 1-3 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.'''
            f'''\nAt last, output the action id(s) (0 ~ {max_id}) from the available actions to execute. '''
            f'''\n\nThe input given to you is three first person view observations (left, front, right). Plan accordingly based on the visual observations.'''
            f'''\n\nYou are supposed to output in JSON.{OUTPUT_TEMPLATE}'''
        )

    def _make_following_prompt_suffix(self, max_id: int) -> str:
        return (
            f'''\nTo achieve the task, 1. Reason about the current visual state from the three camera views and your final goal, and 2. Reflect on the effect of previous actions.'''
            f'''\nAim for about 3-5 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.'''
            f'''\nAt last, output the action id(s) (0 ~ {max_id}) from the available actions to execute. '''
            f'''\n\nThe input given to you is three first person view observations (left, front, right). Plan accordingly based on the visual observations.'''
            f'''\n\nYou are supposed to output in JSON.{OUTPUT_TEMPLATE}'''
        )

    def _format_action_history(self, action_history: list[tuple[str, str]]) -> str:
        if self.action_history_len == 0:
            return ""
        if self.action_history_len > 0:
            action_history = action_history[-self.action_history_len:]
        text = '\n\n The action history:'
        for i, (action_name, feedback) in enumerate(action_history):
            action_id = self._action_id_map.get(action_name, -1)
            if self.use_feedback:
                text += '\n Step {}, action id {}, {}, env feedback: {}'.format(
                    i, action_id, action_name, feedback,
                )
            else:
                text += '\n Step {}, action id {}, {}'.format(
                    i, action_id, action_name,
                )
        return text

    def _wrap_as_user_message(self, prompt: str, observation) -> list[dict]:
        """Wrap prompt text + observation images as a user message.

        For LHPR-VLN: 3 images (left, front, right) BEFORE text.
        """
        content: list[dict] = []

        # Add 3 RGB views as separate images
        if observation and observation.metadata:
            for view_label, meta_key in [("Left view (-60 deg)", "left_rgb_path"),
                                          ("Front view (0 deg)", "front_rgb_path"),
                                          ("Right view (+60 deg)", "right_rgb_path")]:
                path = observation.metadata.get(meta_key)
                if path:
                    image_url = _encode_image_base64(path)
                    if image_url:
                        content.append({"type": "text", "text": f"[{view_label}]"})
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif observation and observation.rgb_path:
            # Fallback: single image (e.g., testing without full bridge)
            image_url = _encode_image_base64(observation.rgb_path)
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
