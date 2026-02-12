"""Helper functions for the AI2-THOR v2.1.0 bridge (EB-Alfred track).

Ported from EmbodiedBench:
- envs/eb_alfred/env/tasks.py (goal evaluation)
- envs/eb_alfred/utils.py (object name mapping, task loading)
- envs/eb_alfred/gen/constants.py (THOR constants)

This file runs inside the ai2thor conda env (Python 3.8).
"""
from __future__ import annotations

import json
import logging
import os
import string
from pathlib import Path

import numpy as np

logger = logging.getLogger("easi.bridge.thor_utils")

# --- Constants (from EmbodiedBench gen/constants.py) ---

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 1.5
AGENT_STEP_SIZE = 0.25
RECORD_SMOOTHING_FACTOR = 1
AGENT_HORIZON_ADJ = 15
AGENT_ROTATE_ADJ = 90
RENDER_IMAGE = True
RENDER_DEPTH_IMAGE = False
RENDER_CLASS_IMAGE = False
RENDER_OBJECT_IMAGE = False

GOALS = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_and_place_with_movable_recep",
]


# --- Object Name Mapping ---

def natural_word_to_ithor_name(w: str) -> str:
    """Map natural language object name to iTHOR name.

    e.g., 'floor lamp' -> 'FloorLamp', 'alarm clock' -> 'AlarmClock'
    If the word contains digits (e.g., 'Cabinet_2'), return as-is.
    """
    if any(i.isdigit() for i in w):
        return w
    if w == "CD":
        return w
    return "".join([string.capwords(x) for x in w.split()])


# --- Task JSON Loading ---

def load_task_json(task_path: str, data_dir: str) -> dict:
    """Load preprocessed traj_data.json for an EB-Alfred episode.

    Args:
        task_path: e.g. 'pick_and_place_simple-Mug-None-Shelf-1/trial_T20190001'
        data_dir: path to the downloaded dataset directory
    """
    # Try standard path: <data_dir>/tasks/<task_path>/pp/ann_0.json
    # (repeat_idx is handled by the caller)
    json_path = os.path.join(data_dir, "tasks", task_path, "pp", "ann_0.json")
    if not os.path.exists(json_path):
        # Fallback: try traj_data.json
        json_path = os.path.join(data_dir, "tasks", task_path, "traj_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Task annotation not found at {json_path}. "
            f"Ensure tasks.zip was extracted in {data_dir}"
        )
    with open(json_path) as f:
        data = json.load(f)
    return data


def load_task_json_with_repeat(task_path: str, repeat_idx: int, data_dir: str) -> dict:
    """Load the specific repeat annotation for an EB-Alfred episode."""
    json_path = os.path.join(data_dir, "tasks", task_path, "pp", f"ann_{repeat_idx}.json")
    if not os.path.exists(json_path):
        # Fallback to ann_0.json
        json_path = os.path.join(data_dir, "tasks", task_path, "pp", "ann_0.json")
    if not os.path.exists(json_path):
        # Fallback to traj_data.json
        json_path = os.path.join(data_dir, "tasks", task_path, "traj_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Task annotation not found for {task_path} repeat {repeat_idx} in {data_dir}"
        )
    with open(json_path) as f:
        data = json.load(f)
    return data


# --- Object Lookup Helpers ---

def get_objects_of_type(obj_type: str, metadata: dict) -> list:
    """Get all objects of a given type from THOR metadata."""
    return [obj for obj in metadata["objects"] if obj_type in obj["objectId"]]


def get_objects_with_name_and_prop(name: str, prop: str, metadata: dict) -> list:
    """Get objects matching name that have a truthy property."""
    return [obj for obj in metadata["objects"]
            if name in obj["objectId"] and obj.get(prop)]


def get_obj_of_type_closest_to_obj(obj_type: str, ref_obj_id: str, metadata: dict):
    """Get the object of obj_type closest to ref_obj_id."""
    ref_obj = None
    for obj in metadata["objects"]:
        if obj["objectId"] == ref_obj_id:
            ref_obj = obj
            break
    if ref_obj is None:
        return None

    candidates = get_objects_of_type(obj_type, metadata)
    if not candidates:
        return None

    min_dist = float("inf")
    closest = None
    for c in candidates:
        dx = c["position"]["x"] - ref_obj["position"]["x"]
        dz = c["position"]["z"] - ref_obj["position"]["z"]
        dist = (dx ** 2 + dz ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = c
    return closest


# --- Goal Evaluation ---

def get_targets_from_traj(traj_data: dict) -> dict:
    """Extract goal targets from trajectory data."""
    pddl = traj_data.get("pddl_params", {})
    targets = {}
    # Map from traj keys to goal target keys
    if "object_target" in pddl:
        targets["object"] = pddl["object_target"]
    if "parent_target" in pddl:
        targets["parent"] = pddl["parent_target"]
    if "toggle_target" in pddl:
        targets["toggle"] = pddl["toggle_target"]
    if "mrecep_target" in pddl:
        targets["mrecep"] = pddl["mrecep_target"]
    # Sliced variant
    if "object_sliced" in pddl and pddl["object_sliced"]:
        targets["object"] = targets.get("object", "") + "Sliced"
    return targets


def evaluate_goal_conditions(
    traj_data: dict,
    event,
    cleaned_objects: set,
    cooled_objects: set,
    heated_objects: set,
) -> tuple:
    """Evaluate EB-Alfred goal conditions.

    Returns:
        (goal_satisfied: bool, progress: float)

    progress = satisfied_conditions / total_conditions
    """
    task_type = traj_data.get("task_type", "")
    targets = get_targets_from_traj(traj_data)
    metadata = event.metadata

    evaluator = GOAL_EVALUATORS.get(task_type)
    if evaluator is None:
        logger.warning("Unknown task type: %s", task_type)
        return False, 0.0

    s, ts = evaluator(targets, metadata, cleaned_objects, cooled_objects, heated_objects)
    progress = s / ts if ts > 0 else 0.0
    satisfied = s >= ts
    return satisfied, progress


# --- Per-task-type goal evaluators ---

def _eval_pick_and_place_simple(targets, metadata, cleaned, cooled, heated):
    ts = 1
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    if np.any([np.any([p["objectId"] in (r.get("receptacleObjectIds") or [])
                       for r in receptacles])
               for p in pickupables]):
        s += 1
    return s, ts


def _eval_pick_two_obj_and_place(targets, metadata, cleaned, cooled, heated):
    ts = 2
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 2
        s += min(len([p for p in pickupables if "Sliced" in p["objectId"]]), 2)

    s += min(np.max([sum([1 if (r.get("receptacleObjectIds") is not None
                               and p["objectId"] in r["receptacleObjectIds"]) else 0
                          for p in pickupables])
                     for r in receptacles]) if receptacles else 0, 2)
    return s, ts


def _eval_look_at_obj_in_light(targets, metadata, cleaned, cooled, heated):
    ts = 2
    s = 0
    toggleables = get_objects_with_name_and_prop(targets.get("toggle", ""), "toggleable", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)
    inventory = metadata.get("inventoryObjects", [])

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    if len(inventory) > 0 and inventory[0]["objectId"] in [p["objectId"] for p in pickupables]:
        s += 1
    if np.any([t["isToggled"] and t["visible"] for t in toggleables]):
        s += 1
    return s, ts


def _eval_pick_heat_then_place(targets, metadata, cleaned, cooled, heated):
    ts = 3
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    objs_in_place = [p["objectId"] for p in pickupables for r in receptacles
                     if r.get("receptacleObjectIds") is not None and p["objectId"] in r["receptacleObjectIds"]]
    objs_heated = [p["objectId"] for p in pickupables if p["objectId"] in heated]

    if len(objs_in_place) > 0:
        s += 1
    if len(objs_heated) > 0:
        s += 1
    if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
        s += 1
    return s, ts


def _eval_pick_cool_then_place(targets, metadata, cleaned, cooled, heated):
    ts = 3
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    objs_in_place = [p["objectId"] for p in pickupables for r in receptacles
                     if r.get("receptacleObjectIds") is not None and p["objectId"] in r["receptacleObjectIds"]]
    objs_cooled = [p["objectId"] for p in pickupables if p["objectId"] in cooled]

    if len(objs_in_place) > 0:
        s += 1
    if len(objs_cooled) > 0:
        s += 1
    if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
        s += 1
    return s, ts


def _eval_pick_clean_then_place(targets, metadata, cleaned, cooled, heated):
    ts = 3
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    objs_in_place = [p["objectId"] for p in pickupables for r in receptacles
                     if r.get("receptacleObjectIds") is not None and p["objectId"] in r["receptacleObjectIds"]]
    objs_cleaned = [p["objectId"] for p in pickupables if p["objectId"] in cleaned]

    if len(objs_in_place) > 0:
        s += 1
    if len(objs_cleaned) > 0:
        s += 1
    if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
        s += 1
    return s, ts


def _eval_pick_and_place_with_movable_recep(targets, metadata, cleaned, cooled, heated):
    ts = 3
    s = 0
    receptacles = get_objects_with_name_and_prop(targets.get("parent", ""), "receptacle", metadata)
    pickupables = get_objects_with_name_and_prop(targets.get("object", ""), "pickupable", metadata)
    movables = get_objects_with_name_and_prop(targets.get("mrecep", ""), "pickupable", metadata)

    if "Sliced" in targets.get("object", ""):
        ts += 1
        if len([p for p in pickupables if "Sliced" in p["objectId"]]) >= 1:
            s += 1

    pickup_in_movable = [p for p in pickupables for m in movables
                         if m.get("receptacleObjectIds") is not None
                         and p["objectId"] in m["receptacleObjectIds"]]
    movable_in_recep = [m for m in movables for r in receptacles
                        if r.get("receptacleObjectIds") is not None
                        and m["objectId"] in r["receptacleObjectIds"]]

    if len(pickup_in_movable) > 0:
        s += 1
    if len(movable_in_recep) > 0:
        s += 1
    if np.any([np.any([p["objectId"] in (m.get("receptacleObjectIds") or []) for p in pickupables]) and
               np.any([r["objectId"] in (m.get("parentReceptacles") or []) for r in receptacles])
               for m in movables
               if m.get("parentReceptacles") is not None and m.get("receptacleObjectIds") is not None]):
        s += 1
    return s, ts


GOAL_EVALUATORS = {
    "pick_and_place_simple": _eval_pick_and_place_simple,
    "pick_two_obj_and_place": _eval_pick_two_obj_and_place,
    "look_at_obj_in_light": _eval_look_at_obj_in_light,
    "pick_heat_then_place_in_recep": _eval_pick_heat_then_place,
    "pick_cool_then_place_in_recep": _eval_pick_cool_then_place,
    "pick_clean_then_place_in_recep": _eval_pick_clean_then_place,
    "pick_and_place_with_movable_recep": _eval_pick_and_place_with_movable_recep,
}
