"""EB-Manipulation action space definitions.

Unlike EB-Alfred/Navigation which have named discrete actions,
EB-Manipulation uses 7D discrete gripper actions:
  [X, Y, Z, Roll, Pitch, Yaw, Gripper_state]

Reference: EmbodiedBench/embodiedbench/envs/eb_manipulation/eb_man_utils.py
"""
from __future__ import annotations

import ast
import re

# Default discretization parameters (configurable via simulator_kwargs)
DEFAULT_VOXEL_SIZE = 100
DEFAULT_ROTATION_RESOLUTION = 3
DEFAULT_SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
DEFAULT_MAX_STEPS = 15

# Eval set definitions matching EBManEnv.py EVAL_SETS
EVAL_SETS = {
    "base": [
        "pick_cube_shape",
        "stack_cubes_color",
        "place_into_shape_sorter_color",
        "wipe_table_direction",
    ],
    "common_sense": [
        "pick_cube_shape",
        "stack_cubes_color",
        "place_into_shape_sorter_color",
        "wipe_table_direction",
    ],
    "complex": [
        "pick_cube_shape",
        "stack_cubes_color",
        "place_into_shape_sorter_color",
        "wipe_table_direction",
    ],
    "spatial": [
        "pick_cube_relative",
        "stack_cubes_relative",
        "place_into_shape_sorter_relative",
        "wipe_table_relative",
    ],
    "visual": [
        "pick_cube_shape",
        "stack_cubes_color",
        "place_into_shape_sorter_color",
    ],
}

VALID_EVAL_SETS = list(EVAL_SETS.keys())


def get_action_space() -> list[str]:
    """Return empty action space -- EB-Manipulation uses 7D coordinate arrays."""
    return []


def serialize_action(action: list) -> str:
    """Serialize a 7D action list to a string for EASI Action.action_name."""
    return str(action)


def deserialize_action(action_str: str) -> list:
    """Deserialize action string back to a 7D list."""
    try:
        result = ast.literal_eval(action_str)
        if isinstance(result, (list, tuple)) and len(result) == 7:
            return list(result)
    except (ValueError, SyntaxError):
        pass
    return []


def extract_pose_list(text: str) -> list[list]:
    """Extract multiple [x, y, z, a, b, c, n] arrays from a string.

    Reference: ManipPlanner.extract_pose_list()
    """
    matches = re.findall(r"\[([^\[\]]+)\]", text)
    poses = []
    for m in matches:
        arr_str = "[" + m + "]"
        try:
            parsed = ast.literal_eval(arr_str)
            if isinstance(parsed, list) and len(parsed) == 7:
                poses.append(parsed)
        except Exception:
            continue
    return poses
