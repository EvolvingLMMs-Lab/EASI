"""EB-Manipulation prompt builder matching EmbodiedBench ManipPlanner.

Reference: EmbodiedBench/embodiedbench/planner/manip_planner.py
           EmbodiedBench/embodiedbench/evaluator/config/system_prompts.py:50-73
           EmbodiedBench/embodiedbench/evaluator/config/eb_manipulation_example.py
           EmbodiedBench/embodiedbench/planner/planner_utils.py:41-50

Key differences from EB-Alfred/Navigation prompt builders:
- Actions are 7D discrete arrays, not {action_id, action_name}
- executable_plan is a string (not a list of objects)
- Examples are per-task-type (pick/stack/place/wipe)
- System prompt has 4 numeric format placeholders for discretization params
- Response schema executable_plan is type:string (not type:array)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from easi.agents.prompt_builder import _encode_image_base64
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.tasks.ebmanipulation.actions import (
    DEFAULT_ROTATION_RESOLUTION,
    DEFAULT_VOXEL_SIZE,
    extract_pose_list,
    serialize_action,
)
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# System prompt from EmbodiedBench (eb_manipulation_system_prompt).
# Five format placeholders: VOXEL_SIZE, VOXEL_SIZE, 360/ROTATION_RESOLUTION,
# ROTATION_RESOLUTION, examples_string
MANIPULATION_SYSTEM_PROMPT = (
    "## You are a Franka Panda robot with a parallel gripper. "
    "You can perform various tasks and output a sequence of gripper actions "
    "to accomplish a given task with images of your status. The input space, "
    "output action space and color space are defined as follows:\n"
    "\n"
    "** Input Space **\n"
    "- Each input object is represented as a 3D discrete position in the "
    "following format: [X, Y, Z]. \n"
    "- There is a red XYZ coordinate frame located in the top-left corner "
    "of the table. The X-Y plane is the table surface. \n"
    "- The allowed range of X, Y, Z is [0, {}]. \n"
    "- Objects are ordered by Y in ascending order.\n"
    "\n"
    "** Output Action Space **\n"
    "- Each output action is represented as a 7D discrete gripper action "
    "in the following format: [X, Y, Z, Roll, Pitch, Yaw, Gripper state].\n"
    "- X, Y, Z are the 3D discrete position of the gripper in the "
    "environment. It follows the same coordinate system as the input "
    "object coordinates.\n"
    "- The allowed range of X, Y, Z is [0, {}].\n"
    "- Roll, Pitch, Yaw are the 3D discrete orientation of the gripper "
    "in the environment, represented as discrete Euler Angles. \n"
    "- The allowed range of Roll, Pitch, Yaw is [0, {}] and each unit "
    "represents {} degrees.\n"
    "- Gripper state is 0 for close and 1 for open.\n"
    "\n"
    "** Color space **\n"
    "- Each object can be described using one of the colors below:\n"
    '  ["red", "maroon", "lime", "green", "blue", "navy", "yellow", '
    '"cyan", "magenta", "silver", "gray", "olive", "purple", "teal", '
    '"azure", "violet", "rose", "black", "white"],\n'
    "\n"
    "Below are some examples to guide you in completing the task. \n"
    "\n"
    "{}\n"
)

# Output template from planner_utils.py template_manip (lines 41-50).
# Note: executable_plan is type STRING (list of 7D arrays serialized as text).
OUTPUT_TEMPLATE_MANIP = (
    "The output json format should be "
    "{'visual_state_description':str, 'reasoning_and_reflection':str, "
    "'language_plan':str, 'executable_plan':str}\n"
    "The fields in above JSON follows the purpose below:\n"
    "1. visual_state_description: Describe the color and shape of each "
    "object in the detection box in the numerical order in the image. "
    "Then provide the 3D coordinates of the objects chosen from input.\n"
    "2. reasoning_and_reflection: Reason about the overall plan that "
    "needs to be taken on the target objects, and reflect on the previous "
    "actions taken if available.\n"
    "3. language_plan: Natural language actions to achieve the user "
    "instruction. Each language action is started by the step number "
    "and the language action name.\n"
    "4. executable_plan: A list of discrete actions needed to achieve "
    "the user instruction, with each discrete action being a "
    "7-dimensional discrete action.\n"
    "5. keep your plan efficient and concise.\n"
    "!!! When generating content for JSON strings, avoid using any "
    "contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, "
    "n't) that use apostrophes. Instead, write out full forms (is, are, "
    "have, will, would, not) to prevent parsing errors in JSON. Please "
    "do not output any other thing more than the above-mentioned JSON, "
    "do not include ```json and ```!!!.\n"
)

# JSON schema for structured output -- executable_plan is STRING, not array.
EBMANIPULATION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "manipulation_planning",
        "schema": {
            "type": "object",
            "properties": {
                "visual_state_description": {
                    "type": "string",
                    "description": (
                        "Describe the color and shape of each object in the "
                        "detection box in the numerical order in the image. "
                        "Then provide the 3D coordinates of the objects "
                        "chosen from input."
                    ),
                },
                "reasoning_and_reflection": {
                    "type": "string",
                    "description": (
                        "Reason about the overall plan that needs to be taken "
                        "on the target objects, and reflect on the previous "
                        "actions taken if available."
                    ),
                },
                "language_plan": {
                    "type": "string",
                    "description": (
                        "A list of natural language actions to achieve the "
                        "user instruction. Each language action is started by "
                        "the step number and the language action name."
                    ),
                },
                "executable_plan": {
                    "type": "string",
                    "description": (
                        "A list of discrete actions needed to achieve the "
                        "user instruction, with each discrete action being a "
                        "7-dimensional discrete action."
                    ),
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


class EBManipulationPromptBuilder:
    """Prompt builder for EB-Manipulation matching ManipPlanner output.

    Key differences from EB-Alfred/Navigation prompt builders:
    - Actions are 7D arrays, not named actions with IDs
    - Examples are per-task-type (selected by task_variation)
    - executable_plan in response is a STRING of 7D arrays
    - parse_response() extracts 7D arrays and serializes each as Action.action_name
    """

    def __init__(
        self,
        n_shot: int = 10,
        split: str = "base",
        use_feedback: bool = True,
        chat_history: bool = False,
        voxel_size: int = DEFAULT_VOXEL_SIZE,
        rotation_resolution: int = DEFAULT_ROTATION_RESOLUTION,
    ):
        self.n_shot = n_shot
        self.split = split
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.voxel_size = voxel_size
        self.rotation_resolution = rotation_resolution

        # Load per-task-type examples
        examples_file = _CONFIG_DIR / "manipulation_examples.json"
        with open(examples_file) as f:
            self._examples: dict[str, list[str]] = json.load(f)

    def set_action_space(self, actions: list[str]) -> None:
        """No-op -- EB-Manipulation has no fixed named action space."""
        pass

    # ---- PromptBuilderProtocol methods ----

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build COMPLETE message list to send to LLM."""
        # Determine task variation from observation metadata
        task_variation = ""
        if memory.current_observation and memory.current_observation.metadata:
            task_variation = memory.current_observation.metadata.get(
                "task_variation", ""
            )

        if self.chat_history:
            return self._build_chat_history_messages(memory, task_variation)
        return self._build_stateless_messages(memory, task_variation)

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM response into 7D actions.

        ManipPlanner.json_to_action() extracts executable_plan (string of 7D arrays).
        Each 7D array becomes an Action with action_name = serialized array string.
        """
        from easi.utils.json_repair import fix_json

        llm_response = fix_json(llm_response)

        try:
            json_object = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            return []

        executable_plan = json_object.get("executable_plan")
        if executable_plan is None:
            # Fallback: try "properties.language_plan" (ManipPlanner line 306)
            try:
                executable_plan = json_object["properties"]["language_plan"]
            except (KeyError, TypeError):
                logger.warning("No executable_plan in LLM response")
                return []

        # executable_plan can be a string (expected) or already a list
        if isinstance(executable_plan, str):
            try:
                executable_plan = ast.literal_eval(executable_plan)
            except Exception:
                # Try extract_pose_list as fallback
                poses = extract_pose_list(executable_plan)
                if poses:
                    return [
                        Action(action_name=serialize_action(p)) for p in poses
                    ]
                logger.warning("Could not parse executable_plan string")
                return []

        if not isinstance(executable_plan, list) or not executable_plan:
            logger.warning("Empty executable_plan")
            return []

        actions = []
        for x in executable_plan:
            if isinstance(x, tuple):
                x = list(x)
            # Extract action list from various formats (ManipPlanner lines 329-341)
            if isinstance(x, dict):
                list_action = x.get("action", x)
            elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], (int, float)):
                list_action = x
            else:
                list_action = x

            if isinstance(list_action, str):
                poses = extract_pose_list(list_action)
                for pose in poses:
                    actions.append(Action(action_name=serialize_action(pose)))
                continue

            if isinstance(list_action, list) and len(list_action) == 7:
                actions.append(Action(action_name=serialize_action(list_action)))
            elif isinstance(list_action, list):
                for action_single in list_action:
                    if isinstance(action_single, (list, tuple)) and len(action_single) == 7:
                        actions.append(
                            Action(action_name=serialize_action(list(action_single)))
                        )

        return actions

    def get_response_format(self, memory: AgentMemory) -> dict:
        """Return JSON schema for API-level enforcement."""
        return EBMANIPULATION_RESPONSE_SCHEMA

    # ---- Stateless mode (chat_history=False) ----

    def _build_stateless_messages(
        self, memory: AgentMemory, task_variation: str
    ) -> list[dict]:
        """Build full prompt each turn. No history accumulation.

        Reference: ManipPlanner.process_prompt() + ManipPlanner.act()
        """
        prompt, task_prompt = self._process_prompt(
            memory.task_description,
            self._get_avg_obj_coord(memory),
            task_variation,
            prev_act_feedback=self._build_feedback_list(memory),
        )

        # Append template_manip (ManipPlanner.act() line 433-435)
        task_prompt += "\n\n" + OUTPUT_TEMPLATE_MANIP

        return self._build_message(prompt, task_prompt, memory)

    # ---- Chat history mode (chat_history=True) ----

    def _build_chat_history_messages(
        self, memory: AgentMemory, task_variation: str
    ) -> list[dict]:
        """Build messages with chat history accumulation.

        Reference: ManipPlanner.process_prompt() chat_history branch
        """
        if memory.is_first_turn:
            return self._build_stateless_messages(memory, task_variation)

        # Subsequent turns: rebuild message history
        messages: list[dict] = []

        for step_idx, step in enumerate(memory.steps):
            if step.llm_response is None:
                continue

            # Build prompt for this step
            history_up_to = [
                (
                    step.action.action_name if step.action else "",
                    step.feedback or "",
                )
                for s_idx, s in enumerate(memory.steps[:step_idx])
                if s.action and s.feedback is not None
            ]

            if step_idx == 0:
                prompt, task_prompt = self._process_prompt(
                    memory.task_description,
                    self._get_avg_obj_coord(memory),
                    task_variation,
                    prev_act_feedback=[],
                )
            else:
                prompt, task_prompt = self._process_prompt(
                    memory.task_description,
                    self._get_avg_obj_coord(memory),
                    task_variation,
                    prev_act_feedback=[(a, f) for a, f in history_up_to],
                )

            task_prompt += "\n\n" + OUTPUT_TEMPLATE_MANIP

            # User message with image
            user_msg = self._wrap_as_user_message(
                prompt + task_prompt, step.observation
            )
            messages.extend(user_msg)

            # Assistant message
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": step.llm_response}],
                }
            )

        # Current turn
        all_feedback = self._build_feedback_list(memory)
        prompt, task_prompt = self._process_prompt(
            memory.task_description,
            self._get_avg_obj_coord(memory),
            task_variation,
            prev_act_feedback=all_feedback,
        )
        task_prompt += "\n\n" + OUTPUT_TEMPLATE_MANIP

        messages.extend(
            self._wrap_as_user_message(
                prompt + task_prompt, memory.current_observation
            )
        )

        return messages

    # ---- Prompt construction (matching ManipPlanner.process_prompt) ----

    def _process_prompt(
        self,
        user_instruction: str,
        avg_obj_coord: str,
        task_variation: str,
        prev_act_feedback: list[tuple[str, str]] | None = None,
    ) -> tuple[str, str]:
        """Build system prompt + task prompt.

        Reference: ManipPlanner.process_prompt() lines 48-70
        """
        user_instruction = user_instruction.rstrip(".")
        prev_act_feedback = prev_act_feedback or []

        rotation_bins = int(360 / self.rotation_resolution)

        # Get task type for example selection (e.g., "pick" from "pick_cube_shape")
        task_type = task_variation.split("_")[0] if task_variation else ""

        if len(prev_act_feedback) == 0:
            # First turn (ManipPlanner lines 50-55)
            examples_for_task = self._examples.get(task_type, [])
            if self.n_shot >= 1 and examples_for_task:
                examples_str = "\n".join(
                    [
                        f"Example {i}: \n{x}"
                        for i, x in enumerate(examples_for_task[: self.n_shot])
                    ]
                )
                general_prompt = MANIPULATION_SYSTEM_PROMPT.format(
                    self.voxel_size,
                    self.voxel_size,
                    rotation_bins,
                    self.rotation_resolution,
                    examples_str,
                )
            else:
                general_prompt = MANIPULATION_SYSTEM_PROMPT.format(
                    self.voxel_size,
                    self.voxel_size,
                    rotation_bins,
                    self.rotation_resolution,
                    "",
                )
            task_prompt = (
                f"\n## Now you are supposed to follow the above examples to "
                f"generate a sequence of discrete gripper actions that completes "
                f"the below human instruction. \n"
                f"Human Instruction: {user_instruction}.\n"
                f"Input: {avg_obj_coord}\n"
                f"Output gripper actions: "
            )

        elif self.chat_history:
            # Chat history subsequent turn (ManipPlanner lines 56-61)
            general_prompt = f"The human instruction is: {user_instruction}."
            general_prompt += "\n\n The gripper action history:"
            for i, (action, feedback) in enumerate(prev_act_feedback):
                general_prompt += (
                    "\n Step {}, the output action **{}**, env feedback: {}".format(
                        i, action, feedback
                    )
                )
            task_prompt = (
                f"\n\n Considering the above interaction history and the current "
                f"image state, to achieve the human instruction: '{user_instruction}', "
                f"you are supposed to output in json. You need to describe current "
                f"visual state from the image, summarize interaction history and "
                f"environment feedback and reason why the last action or plan failed "
                f"and did not finish the task, output your new plan to achieve the "
                f"goal from current state. At the end, output the executable plan "
                f"with the 7-dimsension action."
            )

        else:
            # Stateless subsequent turn (ManipPlanner lines 62-69)
            examples_for_task = self._examples.get(task_type, [])
            if self.n_shot >= 1 and examples_for_task:
                examples_str = "\n".join(
                    [
                        f"Example {i}: \n{x}"
                        for i, x in enumerate(examples_for_task[: self.n_shot])
                    ]
                )
                general_prompt = MANIPULATION_SYSTEM_PROMPT.format(
                    self.voxel_size,
                    self.voxel_size,
                    rotation_bins,
                    self.rotation_resolution,
                    examples_str,
                )
            else:
                general_prompt = MANIPULATION_SYSTEM_PROMPT.format(
                    self.voxel_size,
                    self.voxel_size,
                    rotation_bins,
                    self.rotation_resolution,
                    "",
                )
            task_prompt = (
                f"\n## Now you are supposed to follow the above examples to "
                f"generate a sequence of discrete gripper actions that completes "
                f"the below human instruction. \n"
                f"Human Instruction: {user_instruction}.\n"
                f"Input: {avg_obj_coord}\n"
                f"Output gripper actions: "
            )
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "

        return general_prompt, task_prompt

    # ---- Helpers ----

    def _get_avg_obj_coord(self, memory: AgentMemory) -> str:
        """Get object coordinates from observation metadata.

        Object coordinates are computed by the bridge and stored in
        observation metadata.
        """
        if memory.current_observation and memory.current_observation.metadata:
            return memory.current_observation.metadata.get("avg_obj_coord", "{}")
        return "{}"

    def _build_feedback_list(
        self, memory: AgentMemory
    ) -> list[tuple[str, str]]:
        """Build action + feedback list from memory."""
        feedback = []
        for action_name, fb in memory.action_history:
            feedback.append((action_name, fb))
        return feedback

    def _build_message(
        self, prompt: str, task_prompt: str, memory: AgentMemory
    ) -> list[dict]:
        """Build user message with image and prompt text.

        Reference: ManipPlanner.get_message() -- text first, then image.
        """
        full_prompt = prompt + task_prompt
        return self._wrap_as_user_message(full_prompt, memory.current_observation)

    @staticmethod
    def _wrap_as_user_message(prompt: str, observation) -> list[dict]:
        """Wrap prompt text + observation image as a user message.

        ManipPlanner: text BEFORE image (opposite of VLMPlanner).
        Reference: ManipPlanner.get_message() lines 191-216
        """
        content: list[dict] = [{"type": "text", "text": prompt}]

        if observation and observation.rgb_path:
            image_url = _encode_image_base64(observation.rgb_path)
            if image_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )

        return [{"role": "user", "content": content}]
