"""EB-Alfred-specific prompt builder (Decision #10: OpenAI message format).

Produces prompts in OpenAI message format with interleaved text+image,
tailored for the EB-Alfred household task domain.
Referenced in ebalfred*.yaml via:
    agent:
      prompt_builder: "easi.tasks.ebalfred.prompts.EBAlfredPromptBuilder"

Reference: EmbodiedBench/embodiedbench/evaluator/config/system_prompts.py
"""
from __future__ import annotations

from easi.agents.prompt_builder import _encode_image_base64
from easi.core.episode import Observation


class EBAlfredPromptBuilder:
    """Prompt builder for EB-Alfred household tasks in AI2-THOR.

    Returns OpenAI message format (list[dict]) with interleaved text+image.
    """

    SYSTEM_TEMPLATE = """You are an embodied agent in an AI2-THOR household environment. You must complete household tasks by choosing high-level skill actions.

## Task
{task_description}

## Available Actions
Each action is a high-level skill. The simulator will handle low-level navigation and interaction.
{action_list}

## Action Categories
- **find a <object>**: Navigate to the specified object
- **pick up the <object>**: Pick up an object (must be near it first)
- **put down the object in hand**: Place held object in the nearest receptacle
- **open/close the <object>**: Open or close a container (Fridge, Cabinet, etc.)
- **turn on/off the <object>**: Toggle an appliance (Faucet, Microwave, etc.)
- **slice the <object>**: Slice a sliceable object (must be holding a knife)

## Output Format
Respond with valid JSON:
{{
    "observation": "Describe what you see in the current image",
    "reasoning": "Step-by-step reasoning about what to do next",
    "plan": "Your high-level plan to complete the task",
    "executable_plan": [
        {{"action": "<action_name>"}}
    ]
}}

## Tips
- Find objects before trying to interact with them.
- To clean: put object in sink, turn on faucet, turn off faucet, pick up object.
- To heat: put object in microwave, turn on microwave, turn off microwave, pick up object.
- To cool: put object in fridge, close fridge, open fridge, pick up object.
- Output one action at a time for reliability.
"""

    STEP_TEMPLATE = """Task: {task_description}

{history_section}

Look at the current observation image and decide your next action. Respond with valid JSON."""

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
                status = "OK" if "success" in feedback.lower() else "FAILED"
                history_lines.append(f"  Step {i+1}: {action_name} → {status} ({feedback})")
            history_section = "## Previous Actions\n" + "\n".join(history_lines)
        else:
            history_section = "This is your first action. Start by observing the environment."

        text = self.STEP_TEMPLATE.format(
            task_description=task_description,
            history_section=history_section,
        )

        # Build interleaved content parts (text + image)
        content_parts: list[dict] = [{"type": "text", "text": text}]
        if observation.rgb_path:
            image_url = _encode_image_base64(observation.rgb_path)
            if image_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

        return [{"role": "user", "content": content_parts}]
