"""EB-Alfred prompt builder matching EmbodiedBench VLMPlanner exactly.

Supports both chat_history=False (stateless) and chat_history=True modes.
Referenced in ebalfred*.yaml via:
    agent:
      prompt_builder: "easi.tasks.ebalfred.prompts.EBAlfredPromptBuilder"
      prompt_builder_kwargs:
        n_shot: 10
        split: "base"

Reference: EmbodiedBench/embodiedbench/planner/vlm_planner.py
           EmbodiedBench/embodiedbench/evaluator/config/system_prompts.py
           EmbodiedBench/embodiedbench/planner/planner_utils.py
"""
from __future__ import annotations

import json
from pathlib import Path

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Exact system prompt from EmbodiedBench system_prompts.py (alfred_system_prompt).
# Three format placeholders: max_action_id, action_list, examples.
ALFRED_SYSTEM_PROMPT = '''## You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

## Action Descriptions and Validity Rules
\u2022 Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
\u2022 Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
\u2022 Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
\u2022 Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle.
\u2022 Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
\u2022 Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
\u2022 Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
\u2022 Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
\u2022 Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.


## The available action id (0 ~ {}) and action names are: {}.

{}

## Guidelines
1. **Output Plan**: Avoid generating empty plan. Each plan should include no more than 20 actions.
2. **Visibility**: Always locate a visible object by the 'find' action before interacting with it.
3. **Action Guidelines**: Make sure match the action name and its corresponding action id in the output.\n Avoid performing actions that do not meet the defined validity criteria. For instance, if you want to put object in a receptacle, use 'put down' rather than 'drop' actions.
4. **Prevent Repeating Action Sequences**: Do not repeatedly execute the same action or sequence of actions.\n Try to modify the action sequence because previous actions do not lead to success.
5. **Multiple Instances**: There may be multiple instances of the same object, distinguished by an index following their names, e.g., Cabinet_2, Cabinet_3. You can explore these instances if you do not find the desired object in the current receptacle.
6. **Reflection on History and Feedback**: Use interaction history and feedback from the environment to refine and improve your current plan.\n If the last action is invalid, reflect on the reason, such as not adhering to action rules or missing preliminary actions, and adjust your plan accordingly.
'''

# Output template from EmbodiedBench planner_utils.py (template, not template_lang).
OUTPUT_TEMPLATE = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image,
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task,
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name,
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

# JSON schema matching EmbodiedBench's vlm_generation_guide from
# embodiedbench/planner/planner_config/generation_guide.py
EBALFRED_RESPONSE_SCHEMA = {
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


class EBAlfredPromptBuilder:
    """Prompt builder for EB-Alfred that exactly matches VLMPlanner output.

    Supports two modes:
    - chat_history=False (default): Stateless. Every turn sends the full system
      prompt + examples + instruction. No message accumulation.
    - chat_history=True: First turn is identical. Subsequent turns send a minimal
      prompt (instruction + history + reflection). Messages accumulate.
    """

    def __init__(
        self,
        n_shot: int = 10,
        split: str = "base",
        use_feedback: bool = True,
        chat_history: bool = False,
    ):
        self.n_shot = n_shot
        self.split = split
        self.use_feedback = use_feedback
        self.chat_history = chat_history

        # Load examples
        if split == "long_horizon":
            examples_file = _CONFIG_DIR / "alfred_long_horizon_examples.json"
        else:
            examples_file = _CONFIG_DIR / "alfred_examples.json"

        with open(examples_file) as f:
            self._examples: list[str] = json.load(f)

        # Action space state (set via set_action_space or lazily in build_messages)
        self._actions: list[str] = []
        self._action_str: str = ""
        self._action_id_map: dict[str, int] = {}
        self._id_action_map: dict[int, str] = {}

    def set_action_space(self, actions: list[str]) -> None:
        """Update action space (e.g., after dynamic expansion per episode)."""
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
        """Build action list in VLMPlanner's get_availabel_action_prompt format."""
        parts = ''
        for i in range(len(actions)):
            parts += '\naction id ' + str(i) + ': ' + str(actions[i])
            if i < len(actions) - 1:
                parts += ', '
        return parts

    # ---- New PromptBuilderProtocol methods ----

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build COMPLETE message list to send to LLM."""
        # Lazy-init action maps from memory.action_space
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
                action_name = entry.get("action", "")

            validated = validate_action_name(action_name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Skipping invalid action: '%s'", action_name)
                break

        return actions

    def get_response_format(self, memory: AgentMemory) -> dict:
        """Return JSON schema for API-level enforcement.

        Matches EmbodiedBench's vlm_generation_guide. The OUTPUT_TEMPLATE
        is always in the prompt as a fallback for backends that don't
        support response_format.
        """
        return EBALFRED_RESPONSE_SCHEMA

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
        """Build the full prompt text matching VLMPlanner.process_prompt().

        Covers both first-turn and subsequent-turn paths with
        chat_history=False (lines 53-63 and 78-94 of vlm_planner.py).
        """
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        if len(action_history) == 0:
            # First turn path (vlm_planner.py lines 53-63)
            if self.n_shot >= 1:
                examples_str = '\n\n'.join([
                    f'## Task Execution Example {i}: \n {x}'
                    for i, x in enumerate(self._examples[:self.n_shot])
                ])
                prompt = ALFRED_SYSTEM_PROMPT.format(
                    max_id, self._action_str, examples_str,
                )
            else:
                prompt = ALFRED_SYSTEM_PROMPT.format(
                    max_id, self._action_str, '',
                )

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {max_id}) from the available actions to excute."

        else:
            # Subsequent turn path, chat_history=False (vlm_planner.py lines 78-94)
            # Note: VLMPlanner has TWO spaces after "Example" on line 80 (typo preserved)
            if self.n_shot >= 1:
                examples_str = '\n\n'.join([
                    f'## Task Execution Example  {i}: \n {x}'
                    for i, x in enumerate(self._examples[:self.n_shot])
                ])
                prompt = ALFRED_SYSTEM_PROMPT.format(
                    max_id, self._action_str, examples_str,
                )
            else:
                prompt = ALFRED_SYSTEM_PROMPT.format(
                    max_id, self._action_str, '',
                )

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, (action_name, feedback) in enumerate(action_history):
                action_id = self._action_id_map.get(action_name, -1)
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                        i, action_id, action_name, feedback,
                    )
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(
                        i, action_id, action_name,
                    )

            prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {max_id}) from the available actions.'''

        # Append output template (VLMPlanner act() line 203: prompt + template)
        prompt += OUTPUT_TEMPLATE

        return prompt

    # ---- Chat history mode (chat_history=True) ----

    def _build_chat_history_messages(self, memory: AgentMemory) -> list[dict]:
        """Build accumulated message history for chat_history=True mode.

        First turn: identical to stateless first turn.
        Subsequent turns: reconstruct full message history from memory.steps,
        then append current turn with minimal prompt.
        """
        if memory.is_first_turn:
            # First turn: same as stateless
            return self._build_stateless_messages(memory)

        # Reconstruct message history from completed steps
        messages: list[dict] = []

        for step_idx, step in enumerate(memory.steps):
            if step.llm_response is None:
                # Buffered step — no user/assistant pair
                continue

            if step_idx == 0:
                # First step: full prompt (same as stateless first turn)
                history_up_to = []
            else:
                # Subsequent steps: collect action history up to this step
                history_up_to = [
                    (s.action.action_name, s.feedback)
                    for s in memory.steps[:step_idx]
                    if s.action and s.feedback is not None
                ]

            if step_idx == 0:
                prompt = self._build_first_turn_prompt(memory.task_description)
            else:
                prompt = self._build_chat_subsequent_prompt(
                    memory.task_description, history_up_to,
                )

            # User message
            messages.extend(
                self._wrap_as_user_message(prompt, step.observation)
            )
            # Assistant message
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": step.llm_response}],
            })

        # Current turn: append new user message
        # Collect full action history from all steps with feedback
        all_history = memory.action_history
        current_prompt = self._build_chat_subsequent_prompt(
            memory.task_description, all_history,
        )
        messages.extend(
            self._wrap_as_user_message(current_prompt, memory.current_observation)
        )

        return messages

    def _build_first_turn_prompt(self, task_description: str) -> str:
        """Build first-turn prompt text (identical for both modes)."""
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        if self.n_shot >= 1:
            examples_str = '\n\n'.join([
                f'## Task Execution Example {i}: \n {x}'
                for i, x in enumerate(self._examples[:self.n_shot])
            ])
            prompt = ALFRED_SYSTEM_PROMPT.format(
                max_id, self._action_str, examples_str,
            )
        else:
            prompt = ALFRED_SYSTEM_PROMPT.format(
                max_id, self._action_str, '',
            )

        prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
        prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {max_id}) from the available actions to excute."
        prompt += OUTPUT_TEMPLATE

        return prompt

    def _build_chat_subsequent_prompt(
        self,
        task_description: str,
        action_history: list[tuple[str, str]],
    ) -> str:
        """Build minimal subsequent-turn prompt for chat_history=True.

        VLMPlanner lines 66-77: no system prompt, no examples.
        Uses "The human instruction is:" (no "## Now the").
        """
        user_instruction = task_description.rstrip('.')
        max_id = len(self._actions) - 1

        prompt = f'The human instruction is: {user_instruction}.'
        prompt += '\n\n The action history:'
        for i, (action_name, feedback) in enumerate(action_history):
            action_id = self._action_id_map.get(action_name, -1)
            if self.use_feedback:
                prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(
                    i, action_id, action_name, feedback,
                )
            else:
                prompt += '\nStep {}, action id {}, {}'.format(
                    i, action_id, action_name,
                )

        prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {max_id}) from the available actions.'''

        prompt += OUTPUT_TEMPLATE

        return prompt

    # ---- Helpers ----

    @staticmethod
    def _wrap_as_user_message(prompt: str, observation) -> list[dict]:
        """Wrap prompt text + observation image as a user message.

        Image BEFORE text (VLMPlanner line 127).
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
