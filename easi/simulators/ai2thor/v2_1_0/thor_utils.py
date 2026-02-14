"""Generic helper functions for AI2-THOR v2.1.0 bridges.

Constants and object query utilities shared across any benchmark
that uses AI2-THOR 2.1.0. Task-specific logic (goal evaluation,
dataset loading) lives in the task layer (e.g., easi/tasks/ebalfred/).

This file runs inside the ai2thor conda env (Python 3.8).
"""
from __future__ import annotations

import string

from easi.utils.logging import get_logger

logger = get_logger(__name__)

# --- Constants ---

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
