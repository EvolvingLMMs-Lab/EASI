"""VLN-CE R2R action space.

4 discrete actions matching VLN-CE config (0.25m forward, 15° turns).
"""

DISCRETE_ACTIONS = [
    "move_forward",
    "turn_left",
    "turn_right",
    "stop",
]

ACTION_NAME_TO_ID = {name: i for i, name in enumerate(DISCRETE_ACTIONS)}


def get_action_space() -> list[str]:
    """Return the VLN-CE R2R discrete action space."""
    return list(DISCRETE_ACTIONS)
