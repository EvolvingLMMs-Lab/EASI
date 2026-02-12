"""ReAct agent with multi-action buffering and PromptBuilder delegation.

Decision #9: LLM returns executable_plan as a list of actions. The agent
buffers all actions, pops one per act() call. While buffer is non-empty,
act() returns next buffered action without calling LLM. On action failure
(reported via add_feedback), buffer is cleared and next act() re-queries LLM.

Decision #10: PromptBuilder returns OpenAI message format (list[dict]).
The agent appends these directly to chat_history and passes to LLM.
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import DefaultPromptBuilder, PromptBuilderProtocol
from easi.core.base_agent import BaseAgent
from easi.core.episode import Action, Observation
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class ReActAgent(BaseAgent):
    """ReAct agent with action buffering and pluggable prompt building.

    Flow per LLM call:
    1. PromptBuilder constructs OpenAI messages (with image)
    2. LLM returns JSON with executable_plan: [{action: ...}, ...]
    3. Agent validates ALL actions, buffers valid ones
    4. Returns first action; subsequent act() calls pop from buffer
    5. On failure feedback -> clear buffer -> next act() re-queries LLM
    """

    def __init__(
        self,
        llm_client,
        action_space: list[str],
        prompt_builder: PromptBuilderProtocol | None = None,
    ):
        super().__init__(llm_client=llm_client, action_space=action_space)
        self.prompt_builder: PromptBuilderProtocol = prompt_builder or DefaultPromptBuilder()
        self._action_buffer: list[Action] = []
        self._action_feedback: list[tuple[str, str]] = []
        self._task_description: str = ""

    def reset(self) -> None:
        super().reset()
        self._action_buffer = []
        self._action_feedback = []
        self._task_description = ""

    def act(self, observation: Observation, task_description: str) -> Action:
        """Return the next action.

        If buffer has pending actions, pop and return (no LLM call).
        Otherwise, call LLM, parse executable_plan, buffer all actions,
        return the first one.
        """
        # Decision #9: return buffered action if available
        if self._action_buffer:
            return self._action_buffer.pop(0)

        # Store task description for step prompts
        self._task_description = task_description

        # Build system prompt on first call
        if not self._chat_history:
            system_messages = self.prompt_builder.build_system_prompt(
                action_space=self.action_space,
                task_description=task_description,
            )
            self._chat_history.extend(system_messages)

        # Build step prompt (OpenAI message format with image)
        step_messages = self.prompt_builder.build_step_prompt(
            observation=observation,
            task_description=task_description,
            action_history=self._action_feedback,
        )
        self._chat_history.extend(step_messages)

        # Call LLM
        response = self.llm_client.generate(self._chat_history)
        self._chat_history.append({"role": "assistant", "content": response})

        self._step_count += 1

        # Parse all actions from response
        actions = self._parse_actions(response)
        if not actions:
            # Fallback: Stop or first action
            return self._fallback_action()

        # Buffer remaining actions, return first
        if len(actions) > 1:
            self._action_buffer = actions[1:]
        return actions[0]

    def _parse_actions(self, llm_response: str) -> list[Action]:
        """Parse JSON response into a list of validated Actions.

        Returns empty list on parse failure (caller handles fallback).
        """
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
            validated = self._validate_action_name(action_name)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Skipping invalid action: '%s'", action_name)
                # Stop processing further actions after an invalid one
                break

        return actions

    def _validate_action_name(self, action_name: str) -> str | None:
        """Validate action name against action_space. Returns canonical name or None."""
        if action_name in self.action_space:
            return action_name
        # Case-insensitive fallback
        for valid in self.action_space:
            if valid.lower() == action_name.lower():
                return valid
        return None

    def _fallback_action(self) -> Action:
        """Return a safe fallback action when parsing fails."""
        if "Stop" in self.action_space:
            return Action(action_name="Stop")
        return Action(action_name=self.action_space[0])

    def add_feedback(self, action_name: str, feedback: str) -> None:
        """Record action feedback. Clear buffer on failure (Decision #9)."""
        self._action_feedback.append((action_name, feedback))
        # Clear buffer on failure so next act() re-queries LLM
        if "fail" in feedback.lower() or "error" in feedback.lower():
            if self._action_buffer:
                logger.info("Action '%s' failed, clearing %d buffered actions",
                           action_name, len(self._action_buffer))
                self._action_buffer.clear()

    # --- BaseAgent abstract methods (not used directly, but required) ---

    def _build_system_prompt(self, task_description: str) -> str:
        """Not used — ReActAgent delegates to PromptBuilder."""
        return ""

    def _build_step_prompt(self, observation: Observation) -> str:
        """Not used — ReActAgent delegates to PromptBuilder."""
        return ""

    def _parse_action(self, llm_response: str) -> Action:
        """Not used — ReActAgent uses _parse_actions instead."""
        return self._fallback_action()
