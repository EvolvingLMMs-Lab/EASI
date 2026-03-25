"""EB-Navigation prompt builder matching EmbodiedBench nav_planner.

Reference: EmbodiedBench/embodiedbench/planner/nav_planner.py
           EmbodiedBench/embodiedbench/evaluator/config/system_prompts.py:75-99
           EmbodiedBench/embodiedbench/evaluator/config/eb_navigation_example.py
           EmbodiedBench/embodiedbench/planner/planner_utils.py:20-29
"""
from __future__ import annotations

import json
from pathlib import Path

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# System prompt from EmbodiedBench (eb_navigation_system_prompt).
# Three format placeholders: max_action_id, action_list, examples.
NAVIGATION_SYSTEM_PROMPT = '''## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

## The available action id (0 ~ {}) and action names are: {}.

*** Strategy ***

1. Locate the Target Object Type: Clearly describe the spatial location of the target object \
from the observation image (i.e. in the front left side, a few steps from current standing point).

2. Navigate by *** Using Move forward and Move right/left as main strategy ***, since any point can be reached through a combination of those. \
When planning for movement, reason based on target object's location and obstacles around you. \

3. Focus on primary goal: Only address invalid action when it blocks you from moving closer in the direction to target object. In other words, \
do not overly focus on correcting invalid actions when direct movement towards target object can still bring you closer. \

4. *** Use Rotation Sparingly ***, only when you lose track of the target object and it's not in your view. If so, plan nothing but ONE ROTATION at a step until that object appears in your view. \
After the target object appears, start navigation and avoid using rotation until you lose sight of the target again.

5. *** Do not complete task too early until you can not move any closer to the object, i.e. try to be as close as possible.

{}

----------

'''

# Output template from EmbodiedBench planner_utils.py (template, not template_lang).
OUTPUT_TEMPLATE = "\n" \
"The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}\n" \
"The fields in above JSON follows the purpose below:\n" \
"1. visual_state_description is for description of current state from the visual image, \n" \
"2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \n" \
"3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \n" \
"4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.\n" \
"5. keep your plan efficient and concise.\n" \
"!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.\n"

# JSON schema matching EmbodiedBench navigation output format.
EBNAVIGATION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "embodied_planning",
        "schema": {
            "type": "object",
            "properties": {
                "visual_state_description": {
                    "type": "string",
                    "description": "Description of current state from the visual image",
                },
                "reasoning_and_reflection": {
                    "type": "string",
                    "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
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

_CONFIG_DIR = Path(__file__).parent / "config"


class EBNavigationPromptBuilder:
    """Prompt builder for EB-Navigation matching nav_planner output.

    Supports two modes:
    - chat_history=False (default): Stateless. Every turn sends a single user
      message with the full system prompt, examples, instruction, and history.
    - chat_history=True: Messages accumulate with sliding window.
      Each user message still includes the full system prompt and examples.
    """

    def __init__(
        self,
        n_shot: int = 3,
        split: str = "base",
        use_feedback: bool = True,
        chat_history: bool = True,
        message_window_len: int = 5,
    ):
        self.n_shot = n_shot
        self.split = split
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.message_window_len = message_window_len

        # Load examples
        examples_file = _CONFIG_DIR / "navigation_examples.json"
        with open(examples_file) as f:
            self._examples: list[str] = json.load(f)

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
        """Look up action ID from name. Returns None if not found."""
        return self._action_id_map.get(name)

    def action_id_to_name(self, action_id: int) -> str | None:
        """Look up action name from ID. Returns None if not found."""
        return self._id_action_map.get(action_id)

    @staticmethod
    def _build_action_list_str(actions: list[str]) -> str:
        """Build action list in nav_planner's get_availabel_action_prompt format."""
        parts = ''
        for i in range(len(actions)):
            parts += '\naction id ' + str(i) + ': ' + str(actions[i])
            if i < len(actions) - 1:
                parts += ', '
        return parts

    # ---- PromptBuilderProtocol methods ----

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build COMPLETE message list to send to LLM."""
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

            # Support both {"action_id": int, "action_name": str} and {"action": str}
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
        """Return JSON schema for API-level enforcement."""
        return EBNAVIGATION_RESPONSE_SCHEMA

    # ---- Stateless mode (chat_history=False) ----

    def _build_stateless_messages(self, memory: AgentMemory) -> list[dict]:
        """Build full prompt each turn. No history accumulation."""
        prompt = self._build_prompt_text(
            memory.task_description, memory.action_history,
        )
        return self._wrap_as_user_message(prompt, memory.current_observation)

    def _build_prompt_text(
        self,
        task_description: str,
        action_history: list[tuple[str, str]],
    ) -> str:
        """Build the full prompt text matching nav_planner.process_prompt().

        Covers both first-turn and subsequent-turn paths with
        chat_history=False (nav_planner.py lines 92-136).
        """
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        if len(action_history) == 0:
            # First turn path (nav_planner.py lines 96-104)
            if self.n_shot >= 1:
                examples_str = '\n\n'.join([
                    f'## Task Execution Example {i}: \n {x}'
                    for i, x in enumerate(self._examples[:self.n_shot])
                ])
                prompt = NAVIGATION_SYSTEM_PROMPT.format(
                    max_id, self._action_str, examples_str,
                )
            else:
                prompt = NAVIGATION_SYSTEM_PROMPT.format(
                    max_id, self._action_str, '',
                )

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += self._make_first_prompt_suffix(max_id)

        else:
            # Subsequent turn path, chat_history=False (nav_planner.py lines 122-134)
            # Note: Double space after "Example" on subsequent turns (typo preserved)
            if self.n_shot >= 1:
                examples_str = '\n\n'.join([
                    f'## Task Execution Example  {i}: \n {x}'
                    for i, x in enumerate(self._examples[:self.n_shot])
                ])
                prompt = NAVIGATION_SYSTEM_PROMPT.format(
                    max_id, self._action_str, examples_str,
                )
            else:
                prompt = NAVIGATION_SYSTEM_PROMPT.format(
                    max_id, self._action_str, '',
                )

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += self._format_action_history(action_history)
            prompt += f"\n\n{self._make_following_prompt_suffix(max_id)}"

        return prompt

    # ---- Chat history mode (chat_history=True) ----

    def _build_chat_history_messages(self, memory: AgentMemory) -> list[dict]:
        """Build accumulated message history for chat_history=True mode.

        First turn: identical to stateless first turn.
        Subsequent turns: reconstruct message history from memory.steps,
        each user message includes full system prompt (nav_planner pattern).
        Apply sliding window (MESSAGE_WINDOW_LEN=5).
        """
        if memory.is_first_turn:
            return self._build_stateless_messages(memory)

        # Reconstruct message history from completed steps
        messages: list[dict] = []

        for step_idx, step in enumerate(memory.steps):
            if step.llm_response is None:
                # Buffered step -- no user/assistant pair
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
                prompt = self._build_first_turn_full_prompt(memory.task_description)
            else:
                prompt = self._build_chat_subsequent_prompt(
                    memory.task_description, history_up_to,
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
            memory.task_description, all_history,
        )
        messages.extend(
            self._wrap_as_user_message(current_prompt, memory.current_observation)
        )

        # Apply sliding window
        return messages[-self.message_window_len:]

    def _build_first_turn_full_prompt(self, task_description: str) -> str:
        """Build first-turn prompt (identical for both modes)."""
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        if self.n_shot >= 1:
            examples_str = '\n\n'.join([
                f'## Task Execution Example {i}: \n {x}'
                for i, x in enumerate(self._examples[:self.n_shot])
            ])
            prompt = NAVIGATION_SYSTEM_PROMPT.format(
                max_id, self._action_str, examples_str,
            )
        else:
            prompt = NAVIGATION_SYSTEM_PROMPT.format(
                max_id, self._action_str, '',
            )

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        prompt += self._make_first_prompt_suffix(max_id)

        return prompt

    def _build_chat_subsequent_prompt(
        self,
        task_description: str,
        action_history: list[tuple[str, str]],
    ) -> str:
        """Build subsequent-turn prompt for chat_history=True.

        In nav_planner, chat_history subsequent turns still include the full
        system prompt but use "## The human instruction" (no "Now the").
        """
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        # Double space after "Example" on subsequent turns (typo preserved)
        if self.n_shot >= 1:
            examples_str = '\n\n'.join([
                f'## Task Execution Example  {i}: \n {x}'
                for i, x in enumerate(self._examples[:self.n_shot])
            ])
            prompt = NAVIGATION_SYSTEM_PROMPT.format(
                max_id, self._action_str, examples_str,
            )
        else:
            prompt = NAVIGATION_SYSTEM_PROMPT.format(
                max_id, self._action_str, '',
            )

        prompt += f'\n\n## The human instruction is: {user_instruction}.'
        prompt += self._format_action_history(action_history)
        prompt += f"\n\n{self._make_following_prompt_suffix(max_id)}"

        return prompt

    # ---- Helpers ----

    def _make_first_prompt_suffix(self, max_id: int) -> str:
        """First-turn prompt suffix (nav_planner.py lines 56-62).

        Uses 'Aim for about 1-2 actions'.
        """
        return (
            f'''\nTo achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided '''
            f'''\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.'''
            f'''\nAt last, output the action id(s) (0 ~ {max_id}) from the available actions to execute. '''
            f'''\n\nThe input given to you is an first person view observation. Plan accordingly based on the visual observation.'''
            f'''\n\nYou are supposed to output in JSON.{OUTPUT_TEMPLATE}'''
        )

    def _make_following_prompt_suffix(self, max_id: int) -> str:
        """Following-turn prompt suffix (nav_planner.py lines 64-70).

        Uses 'Aim for about 5-6 actions'.
        """
        return (
            f'''\nTo achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided '''
            f'''\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.'''
            f'''\nAt last, output the action id(s) (0 ~ {max_id}) from the available actions to execute. '''
            f'''\n\nThe input given to you is an first person view observation. Plan accordingly based on the visual observation.'''
            f'''\n\nYou are supposed to output in JSON.{OUTPUT_TEMPLATE}'''
        )

    def _format_action_history(self, action_history: list[tuple[str, str]]) -> str:
        """Format action history matching nav_planner line 117-118.

        Format: 'Step {i}, action id {id}, {name}, env feedback: {fb}'
        """
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

    @staticmethod
    def _wrap_as_user_message(prompt: str, observation) -> list[dict]:
        """Wrap prompt text + observation image as a user message.

        Image BEFORE text (nav_planner line 200-211).
        """
        content: list[dict] = []
        if observation and observation.rgb_path:
            image_url = _encode_image_base64(observation.rgb_path)
            if image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]
