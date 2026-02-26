"""ManipulaTHOR Arm Point Navigation action space and constants."""

# --- Action names (from ithor_arm_constants.py) ---
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_X_P = "MoveArmXP"
MOVE_ARM_X_M = "MoveArmXM"
MOVE_ARM_Y_P = "MoveArmYP"
MOVE_ARM_Y_M = "MoveArmYM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
MOVE_AHEAD = "MoveAheadContinuous"
ROTATE_RIGHT = "RotateRightContinuous"
ROTATE_LEFT = "RotateLeftContinuous"
PICKUP = "PickUpMidLevel"
DONE = "DoneMidLevel"

# Ordered action list (matches original ArmPointNavTask._actions)
ACTION_SPACE = [
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_X_P,
    MOVE_ARM_X_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    MOVE_AHEAD,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    PICKUP,
    DONE,
]

# --- Constants (from ithor_arm_constants.py) ---
MOVE_ARM_CONSTANT = 0.05
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT
MOVE_THR = 0.01  # Threshold for disturbance detection
ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
AGENT_BASE_LOCATION_Y = 0.9009995460510254

ADITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

# MAX_STEPS from original paper
MAX_STEPS = 200

# AI2-THOR commit used by ManipulaTHOR
MANIPULATHOR_COMMIT_ID = "a84dd29471ec2201f583de00257d84fac1a03de2"


def get_action_space() -> list:
    """Return the ManipulaTHOR action space as a list of action name strings."""
    return list(ACTION_SPACE)
