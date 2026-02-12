"""PromptBuilder protocol and default implementation.

Decision #10: Both methods return OpenAI message format (list[dict]).
This supports interleaved text+image content for vision models,
and also works for text-only models (content is just a string).

Contributors adding a new task only need to implement these 2 methods.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol, runtime_checkable

from easi.core.episode import Observation
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


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    """Interface for task-specific prompt construction.

    Both methods return OpenAI message format: list[dict].
    Each dict has "role" and "content" keys.
    Content can be a string (text-only) or a list of content parts
    (interleaved text + image_url for vision models).

    Implementations are referenced in task.yaml via:
        agent:
          prompt_builder: "easi.tasks.my_task.prompts.MyPromptBuilder"
    """

    def build_system_prompt(
        self,
        action_space: list[str],
        task_description: str,
    ) -> list[dict]:
        """Build system message(s).

        Returns:
            List of OpenAI message dicts, e.g.:
            [{"role": "system", "content": "You are an agent..."}]
        """
        ...

    def build_step_prompt(
        self,
        observation: Observation,
        task_description: str,
        action_history: list[tuple[str, str]],
    ) -> list[dict]:
        """Build user message(s) for a single step, including observation image.

        Returns:
            List of OpenAI message dicts with interleaved text+image, e.g.:
            [{"role": "user", "content": [
                {"type": "text", "text": "Task: ..."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}]
        """
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

    def build_system_prompt(
        self,
        action_space: list[str],
        task_description: str,
    ) -> list[dict]:
        action_list = "\n".join(
            f"  {i}. {name}" for i, name in enumerate(action_space)
        )
        text = self.SYSTEM_TEMPLATE.format(
            action_list=action_list,
            task_description=task_description,
        )
        return [{"role": "system", "content": text}]

    def build_step_prompt(
        self,
        observation: Observation,
        task_description: str,
        action_history: list[tuple[str, str]],
    ) -> list[dict]:
        if action_history:
            history_lines = []
            for i, (action_name, feedback) in enumerate(action_history):
                history_lines.append(f"  Step {i+1}: {action_name} → {feedback}")
            history_section = "## Action History\n" + "\n".join(history_lines)
        else:
            history_section = "This is the first step."

        text = self.STEP_TEMPLATE.format(
            task_description=task_description,
            history_section=history_section,
        )

        # Build interleaved content parts
        content_parts: list[dict] = [{"type": "text", "text": text}]

        # Add observation image if available
        if observation.rgb_path:
            image_url = _encode_image_base64(observation.rgb_path)
            if image_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

        return [{"role": "user", "content": content_parts}]
