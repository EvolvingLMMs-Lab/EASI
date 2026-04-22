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

# Exception types that indicate response_format is unsupported by the backend.
# Lazy-resolved on first use to avoid importing litellm at module level.
_FORMAT_UNSUPPORTED_ERRORS: tuple[type[Exception], ...] | None = None


def _get_format_unsupported_errors() -> tuple[type[Exception], ...]:
    """Return exception types for unsupported response_format."""
    global _FORMAT_UNSUPPORTED_ERRORS
    if _FORMAT_UNSUPPORTED_ERRORS is None:
        try:
            from litellm.exceptions import BadRequestError
            _FORMAT_UNSUPPORTED_ERRORS = (BadRequestError,)
        except ImportError:
            _FORMAT_UNSUPPORTED_ERRORS = ()
    return _FORMAT_UNSUPPORTED_ERRORS


def _format_messages_for_log(messages: list[dict]) -> str:
    """Extract readable text from OpenAI-format messages for logging.

    Shows image positions inline as [img_N] markers so interleaved
    placement is visible in the log output.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            img_idx = 0
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    img_idx += 1
                    text_parts.append(f"[img_{img_idx}]")
            text = "".join(text_parts)
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

    # Registry of fallback strategies. To add a new strategy:
    # 1. Add a _fallback_<name> method that takes (messages, response_format, failed_response)
    #    and returns list[Action] (empty = give up, use default action)
    # 2. Register it here
    _FALLBACK_STRATEGIES = {"default_action", "reprompt"}

    def __init__(
        self,
        llm_client,
        action_space: list[str] | None = None,
        prompt_builder: PromptBuilderProtocol | None = None,
        fallback_action: str | None = None,
        fallback_strategy: str = "default_action",
        max_fallback_retries: int = 1,
        max_consecutive_fallbacks: int = 0,
    ):
        super().__init__(llm_client=llm_client, action_space=action_space or [])
        self.prompt_builder: PromptBuilderProtocol = prompt_builder or DefaultPromptBuilder()
        self.memory = AgentMemory(action_space=self.action_space)
        self._action_buffer: list[Action] = []
        self._supports_response_format: bool | None = None  # None = unknown
        self._fallback_action_name = fallback_action
        self._fallback_strategy = fallback_strategy
        self._max_fallback_retries = max_fallback_retries
        self._max_consecutive_fallbacks = max_consecutive_fallbacks  # 0 = disabled
        self._consecutive_fallbacks: int = 0
        self.triggered_fallback: bool = False
        self.forced_early_stop: bool = False
        if fallback_strategy not in self._FALLBACK_STRATEGIES:
            raise ValueError(
                f"Unknown fallback_strategy '{fallback_strategy}'. "
                f"Available: {sorted(self._FALLBACK_STRATEGIES)}"
            )

    def reset(self) -> None:
        super().reset()
        self.memory.clear()
        self._action_buffer.clear()
        self._consecutive_fallbacks = 0
        self.forced_early_stop = False

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
            self.triggered_fallback = False
            return action

        # LLM call path
        self.memory.current_observation = observation
        self.memory.task_description = task_description

        messages = self.prompt_builder.build_messages(self.memory)
        prompt_text = _format_messages_for_log(messages)

        logger.trace("Step %d prompt (%d messages):\n%s",
                     self._step_count + 1, len(messages), prompt_text)

        # Query builder for response_format (optional method)
        get_rf = getattr(self.prompt_builder, 'get_response_format', None)
        response_format = get_rf(self.memory) if get_rf else None

        response = self._generate_with_fallback(messages, response_format)

        logger.trace("Step %d LLM response:\n%s",
                     self._step_count + 1, response)

        actions = self.prompt_builder.parse_response(response, self.memory)

        # If parsing failed, run the configured fallback strategy
        if not actions:
            actions = self._run_fallback(messages, response_format, response)

        # If still no actions after fallback, use the default action
        if not actions:
            action = self._default_fallback_action()
            self.memory.record_step(
                observation, action, llm_response=response, prompt_text=prompt_text,
            )
            self._step_count += 1
            self.triggered_fallback = True
            self._consecutive_fallbacks += 1

            # Force stop if too many consecutive fallbacks
            if (self._max_consecutive_fallbacks > 0
                    and self._consecutive_fallbacks >= self._max_consecutive_fallbacks):
                logger.warning(
                    "Forcing stop: %d consecutive fallbacks reached limit (%d)",
                    self._consecutive_fallbacks, self._max_consecutive_fallbacks,
                )
                self.forced_early_stop = True
                return Action(action_name="stop")

            return action

        self.triggered_fallback = False
        self._consecutive_fallbacks = 0
        self.memory.record_step(
            observation, actions[0], llm_response=response, prompt_text=prompt_text,
        )
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

    # ---- Fallback system ----

    def _run_fallback(
        self,
        messages: list[dict],
        response_format: dict | None,
        failed_response: str,
    ) -> list[Action]:
        """Dispatch to the configured fallback strategy.

        Returns a list of actions if the strategy recovered, or [] to
        fall through to _default_fallback_action().
        """
        handler = getattr(self, f"_fallback_{self._fallback_strategy}", None)
        if handler is None:
            return []
        return handler(messages, response_format, failed_response)

    def _default_fallback_action(self) -> Action:
        """Last-resort action when all fallback strategies fail.

        Priority:
        1. Configured fallback_action (from YAML agent config)
        2. "stop"/"Stop" if in action space (case-insensitive)
        3. "<<STOP>>" sentinel to end the episode
        """
        if self._fallback_action_name:
            logger.warning("Fallback: using configured action '%s'", self._fallback_action_name)
            return Action(action_name=self._fallback_action_name)
        stop_names = {a for a in self.action_space if a.lower() == "stop"}
        if stop_names:
            name = next(iter(stop_names))
            logger.warning("Fallback: using '%s' from action space", name)
            return Action(action_name=name)
        logger.warning("Fallback: no suitable action, signalling <<STOP>>")
        return Action(action_name="<<STOP>>")

    def _fallback_default_action(
        self, messages, response_format, failed_response,
    ) -> list[Action]:
        """Strategy 'default_action': skip reprompt, go straight to default."""
        return []

    def _fallback_reprompt(
        self, messages, response_format, failed_response,
    ) -> list[Action]:
        """Strategy 'reprompt': re-query the LLM with a warning about the failure.

        Appends the failed response + a correction prompt, then retries.
        Falls through to default action after max_fallback_retries attempts.
        """
        retry_messages = list(messages)

        # Get correction prompt from builder, or use default
        get_correction = getattr(self.prompt_builder, 'get_reprompt_message', None)
        if get_correction:
            correction_text = get_correction()
        else:
            correction_text = (
                "Your previous response could not be executed. "
                "Make sure you reply in proper JSON format and "
                "do NOT leave the executable_plan field as an empty list. "
                "You MUST include at least one valid action."
            )

        for attempt in range(1, self._max_fallback_retries + 1):
            # Append the failed response as assistant + correction as user
            retry_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": failed_response}],
            })
            retry_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": correction_text}],
            })

            logger.info(
                "Fallback reprompt attempt %d/%d",
                attempt, self._max_fallback_retries,
            )

            response = self._generate_with_fallback(retry_messages, response_format)
            logger.trace("Reprompt attempt %d response:\n%s", attempt, response)

            actions = self.prompt_builder.parse_response(response, self.memory)
            if actions:
                logger.info("Reprompt attempt %d succeeded with %d actions", attempt, len(actions))
                return actions

            # Update failed_response for next iteration
            failed_response = response

        logger.warning(
            "Reprompt failed after %d attempts, falling back to default action",
            self._max_fallback_retries,
        )
        return []

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
        except _get_format_unsupported_errors() as e:
            logger.warning(
                "response_format not supported by backend, "
                "falling back to prompt-only: %s", e,
            )
            self._supports_response_format = False
            return self.llm_client.generate(messages)
