"""EB-Habitat action space definitions.

The action space is DYNAMIC — it depends on the PDDL domain grounding
which varies per scene/episode. After EBHabEnv.__init__(), the actions
are available as env.language_skill_set (list of ~70 natural language strings).

This module provides the transform_action_to_natural_language function
and a placeholder global action space for offline testing.

Reference: EBHabEnv.py lines 76-107, 158-159
"""

from __future__ import annotations

# Placeholder global action space for offline testing.
# Real action space is extracted from the env at runtime.
PLACEHOLDER_ACTIONS = [
    "navigate to the table 1",
    "navigate to the table 2",
    "navigate to the TV stand",
    "navigate to the left counter in the kitchen",
    "navigate to the right counter in the kitchen",
    "navigate to the sink in the kitchen",
    "navigate to the sofa",
    "navigate to the refrigerator push point",
    "pick up the toy airplane",
    "place at the table 1",
    "open the refrigerator",
    "close the refrigerator",
    "open the cabinet 4",
    "close the cabinet 4",
]


def get_placeholder_action_space() -> list[str]:
    """Return a placeholder action space for offline testing."""
    return list(PLACEHOLDER_ACTIONS)
