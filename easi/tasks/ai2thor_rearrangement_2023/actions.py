"""AI2-THOR Rearrangement action space (84 discrete actions).

Matches the original baseline_configs/rearrange_base.py exactly.
Uses regex-based PascalCase->snake_case to avoid stringcase dependency.
"""
from __future__ import annotations

import re

from easi.tasks.ai2thor_rearrangement_2023.vendor.rearrange.constants import (
    OBJECT_TYPES_WITH_PROPERTIES,
)


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case (matches stringcase.snakecase behavior)."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# 62 pickup actions (pickupable objects from OBJECT_TYPES_WITH_PROPERTIES)
PICKUP_ACTIONS = tuple(sorted(
    f"pickup_{_pascal_to_snake(obj_type)}"
    for obj_type, props in OBJECT_TYPES_WITH_PROPERTIES.items()
    if props["pickupable"]
))

# 10 open actions (openable non-pickupable objects)
OPEN_ACTIONS = tuple(sorted(
    f"open_by_type_{_pascal_to_snake(obj_type)}"
    for obj_type, props in OBJECT_TYPES_WITH_PROPERTIES.items()
    if props["openable"] and not props["pickupable"]
))

# 12 navigation actions (matching rearrange_base.py order)
NAVIGATION_ACTIONS = (
    "done",
    "move_ahead",
    "move_left",
    "move_right",
    "move_back",
    "rotate_right",
    "rotate_left",
    "stand",
    "crouch",
    "look_up",
    "look_down",
    "drop_held_object_with_snap",
)

# Full 84-action space in original order: nav + open + pickup
ALL_ACTIONS = NAVIGATION_ACTIONS + OPEN_ACTIONS + PICKUP_ACTIONS

# Mappings for bridge
ACTION_NAME_TO_ID = {name: i for i, name in enumerate(ALL_ACTIONS)}
ACTION_ID_TO_NAME = {i: name for i, name in enumerate(ALL_ACTIONS)}


def get_action_space() -> list[str]:
    """Return the rearrangement discrete action space (84 actions)."""
    return list(ALL_ACTIONS)
