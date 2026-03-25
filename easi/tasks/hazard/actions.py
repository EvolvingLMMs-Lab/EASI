"""HAZARD action definitions.

HAZARD uses dynamic action selection — the bridge computes available plans
each step and the LLM picks from them. These definitions describe the
underlying action types.

Reference: HAZARD/src/HAZARD/policy/env_actions.py
           HAZARD/src/HAZARD/policy/llm.py:get_available_plans()
"""
from __future__ import annotations

# High-level action types (used internally by bridge)
ACTION_TYPES = [
    "walk_to",
    "pick_up",
    "drop",
    "explore",
    "stop",
]

# Max steps per scenario
SCENARIO_MAX_STEPS = {
    "fire": 1500,
    "flood": 1500,
    "wind": 3000,
}

# Object values
HIGH_VALUE = 5
LOW_VALUE = 1


def get_action_space() -> list[str]:
    """Return the HAZARD action type list.

    Note: The actual available actions are dynamic per-step and computed
    by the bridge. This returns the canonical action types.
    """
    return list(ACTION_TYPES)
