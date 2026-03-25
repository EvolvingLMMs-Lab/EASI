"""PromptBuilder protocol and default implementation.

Both protocol methods receive AgentMemory as their state source.
Contributors adding a new task only need to implement build_messages
and parse_response.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _encode_image_base64(image_path: str) -> str | None:
    """Read an image file and return base64-encoded data URL.

    Returns None if file doesn't exist or can't be read.
    """
    p = Path(image_path)
    if not p.exists():
        logger.warning("Image file not found: %s", image_path)
        return None
    suffix = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def validate_action_name(action_name: str, action_space: list[str]) -> str | None:
    """Validate action name against action_space. Returns canonical name or None."""
    if action_name in action_space:
        return action_name
    # Case-insensitive fallback
    for valid in action_space:
        if valid.lower() == action_name.lower():
            return valid
    return None


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    """Interface for task-specific prompt construction.

    Implementations are referenced in task.yaml via:
        agent:
          prompt_builder: "easi.tasks.my_task.prompts.MyPromptBuilder"

    Required methods:
        build_messages(memory) -> list[dict]
        parse_response(llm_response, memory) -> list[Action]

    Optional methods:
        get_response_format(memory) -> dict | None
            Return a response_format dict for API-level JSON enforcement.
            E.g. {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
            When provided, the agent passes it to LLMClient.generate().
            Builders that don't implement this get no schema enforcement.
    """

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build COMPLETE message list to send to LLM."""
        ...

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM response into validated actions."""
        ...


class DefaultPromptBuilder:
    """Generic prompt builder that works with any task.

    Produces OpenAI-format messages with interleaved text+image.
    """

    SYSTEM_TEMPLATE = """You are an embodied agent operating in a simulated environment. Given a task, you must accomplish it by choosing actions from the available action space.

## Task
{task_description}

## Available Actions
{action_list}

## Output Format
You MUST respond with valid JSON in this exact format:
{{
    "observation": "Describe what you see in the current image",
    "reasoning": "Explain your step-by-step reasoning",
    "plan": "Your high-level plan",
    "executable_plan": [
        {{"action": "<action_name>"}},
        {{"action": "<action_name>"}}
    ]
}}

## Guidelines
1. Always output at least one action in executable_plan.
2. Only use actions from the available action list.
3. If previous actions failed, reason about why and try a different approach.
4. Output at most 10 actions per plan.
"""

    STEP_TEMPLATE = """Task: {task_description}

{history_section}

Based on the current observation image, decide your next action(s). Respond with valid JSON."""

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        """Build complete message list from memory state."""
        messages: list[dict] = []

        # System message
        action_list = "\n".join(
            f"  {i}. {name}" for i, name in enumerate(memory.action_space)
        )
        system_text = self.SYSTEM_TEMPLATE.format(
            action_list=action_list,
            task_description=memory.task_description,
        )
        messages.append({"role": "system", "content": system_text})

        # User message with observation
        action_history = memory.action_history
        if action_history:
            history_lines = []
            for i, (action_name, feedback) in enumerate(action_history):
                history_lines.append(f"  Step {i+1}: {action_name} -> {feedback}")
            history_section = "## Action History\n" + "\n".join(history_lines)
        else:
            history_section = "This is the first step."

        text = self.STEP_TEMPLATE.format(
            task_description=memory.task_description,
            history_section=history_section,
        )

        content_parts: list[dict] = [{"type": "text", "text": text}]
        if memory.current_observation and memory.current_observation.rgb_path:
            image_url = _encode_image_base64(memory.current_observation.rgb_path)
            if image_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

        messages.append({"role": "user", "content": content_parts})
        return messages

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse JSON response into validated actions."""
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
            action_name = entry.get("action", "")
            validated = validate_action_name(action_name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Skipping invalid action: '%s'", action_name)
                break

        return actions
