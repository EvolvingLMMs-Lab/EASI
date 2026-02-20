"""ReAct agent with multi-action buffering and PromptBuilder delegation.

The agent is a thin orchestrator: it populates AgentMemory, delegates
prompt construction and response parsing to the PromptBuilder, and
manages action buffering.
"""
from __future__ import annotations

from easi.agents.prompt_builder import DefaultPromptBuilder, PromptBuilderProtocol
from easi.core.base_agent import BaseAgent
from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _format_messages_for_log(messages: list[dict]) -> str:
    """Extract readable text from OpenAI-format messages for logging."""
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            n_images = sum(1 for p in content if p.get("type") == "image_url")
            text = "".join(text_parts)
            if n_images:
                text = f"[{n_images} image(s)]\n{text}"
        else:
            text = str(content)
        parts.append(f"--- {role} ---\n{text}")
    return "\n".join(parts)


class ReActAgent(BaseAgent):
    """ReAct agent with action buffering and pluggable prompt building.

    Flow per LLM call:
    1. PromptBuilder constructs messages from AgentMemory
    2. LLM returns response text
    3. PromptBuilder parses response into validated Actions
    4. Agent buffers actions, returns first
    5. Subsequent act() calls pop from buffer without LLM call
    6. On failure feedback -> clear buffer -> next act() re-queries LLM
    """

    def __init__(
        self,
        llm_client,
        action_space: list[str] | None = None,
        prompt_builder: PromptBuilderProtocol | None = None,
    ):
        super().__init__(llm_client=llm_client, action_space=action_space or [])
        self.prompt_builder: PromptBuilderProtocol = prompt_builder or DefaultPromptBuilder()
        self.memory = AgentMemory(action_space=self.action_space)
        self._action_buffer: list[Action] = []
        self._supports_response_format: bool | None = None  # None = unknown

    def reset(self) -> None:
        super().reset()
        self.memory.clear()
        self._action_buffer.clear()

    def update_action_space(self, action_space: list[str]) -> None:
        """Update the action space (e.g., after dynamic expansion per episode)."""
        self.action_space = action_space
        self.memory.action_space = action_space
        if hasattr(self.prompt_builder, 'set_action_space'):
            self.prompt_builder.set_action_space(action_space)

    def act(self, observation: Observation, task_description: str) -> Action:
        """Return the next action.

        If buffer has pending actions, pop and return (no LLM call).
        Otherwise, call LLM, parse response via builder, buffer actions.
        """
        # Buffered action path
        if self._action_buffer:
            action = self._action_buffer.pop(0)
            self.memory.record_step(observation, action, llm_response=None)
            return action

        # LLM call path
        self.memory.current_observation = observation
        self.memory.task_description = task_description

        messages = self.prompt_builder.build_messages(self.memory)

        logger.trace("Step %d prompt (%d messages):\n%s",
                     self._step_count + 1, len(messages),
                     _format_messages_for_log(messages))

        # Query builder for response_format (optional method)
        get_rf = getattr(self.prompt_builder, 'get_response_format', None)
        response_format = get_rf(self.memory) if get_rf else None

        response = self._generate_with_fallback(messages, response_format)

        logger.trace("Step %d LLM response:\n%s",
                     self._step_count + 1, response)

        actions = self.prompt_builder.parse_response(response, self.memory)
        if not actions:
            action = self._fallback_action()
            self.memory.record_step(observation, action, llm_response=response)
            self._step_count += 1
            return action

        self.memory.record_step(observation, actions[0], llm_response=response)
        self._step_count += 1

        if len(actions) > 1:
            self._action_buffer = actions[1:]

        return actions[0]

    def add_feedback(self, action_name: str, feedback: str) -> None:
        """Record action feedback. Clear buffer on failure."""
        self.memory.record_feedback(feedback)
        if any(kw in feedback.lower() for kw in ("fail", "error", "invalid")):
            if self._action_buffer:
                logger.info(
                    "Action '%s' failed, clearing %d buffered actions",
                    action_name, len(self._action_buffer),
                )
                self._action_buffer.clear()

    def _fallback_action(self) -> Action:
        """Return a safe fallback action when parsing fails.

        If the action space has "Stop", use it.
        Otherwise, signal that the episode should end via "<<STOP>>"
        """
        if "Stop" in self.action_space:
            return Action(action_name="Stop")
        return Action(action_name="<<STOP>>")

    def _generate_with_fallback(
        self, messages: list[dict], response_format: dict | None,
    ) -> str:
        """Call LLM with optional response_format, falling back on failure.

        If response_format is provided and the backend doesn't support it,
        the failure is caught, cached, and retried without response_format.
        The prompt template is already in messages, so fallback always works.
        """
        if response_format is None or self._supports_response_format is False:
            return self.llm_client.generate(messages)

        try:
            return self.llm_client.generate(messages, response_format=response_format)
        except Exception as e:
            logger.warning(
                "response_format not supported by backend, "
                "falling back to prompt-only: %s", e,
            )
            self._supports_response_format = False
            return self.llm_client.generate(messages)
