"""LHPR-VLN action space constants."""

# Action names (text used by agent and bridge)
MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
STOP = "stop"

ACTION_SPACE = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP]

# Map text -> Habitat action key (stop is handled by bridge, not sim)
ACTION_NAME_TO_HABITAT = {
    MOVE_FORWARD: "move_forward",
    TURN_LEFT: "turn_left",
    TURN_RIGHT: "turn_right",
}


def get_action_space() -> list[str]:
    return list(ACTION_SPACE)
