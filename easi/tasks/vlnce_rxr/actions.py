# easi/tasks/vlnce_rxr/actions.py
"""VLN-CE RxR action space.

6 discrete actions: 0.25m forward, 30° turns, 30° tilts.
"""

DISCRETE_ACTIONS = [
    "move_forward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "stop",
]

ACTION_NAME_TO_ID = {name: i for i, name in enumerate(DISCRETE_ACTIONS)}


def get_action_space() -> list[str]:
    """Return the VLN-CE RxR discrete action space."""
    return list(DISCRETE_ACTIONS)
