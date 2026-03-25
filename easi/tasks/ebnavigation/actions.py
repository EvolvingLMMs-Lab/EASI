"""EB-Navigation action space definitions.

Reference: EmbodiedBench/embodiedbench/envs/eb_navigation/EBNavEnv.py
"""
from __future__ import annotations

DISCRETE_ACTIONS = [
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.",
    "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.",
    "Tilt the camera downward by 30 degrees.",
]

# Mapping from action text to integer index (for bridge)
ACTION_NAME_TO_ID = {name: i for i, name in enumerate(DISCRETE_ACTIONS)}


def get_action_space() -> list[str]:
    """Return the EB-Navigation discrete action space (8 actions)."""
    return list(DISCRETE_ACTIONS)
