"""Self-contained ManipulaTHOR environment wrapping ai2thor.controller.

Extracts essential logic from the original ManipulaTHOR codebase without
AllenAct dependencies. Handles episode setup, action execution, GPS sensor
computation, and metric state tracking.

Reference files from original repo:
- ithor_arm/ithor_arm_environment.py (ManipulaTHOREnvironment)
- ithor_arm/ithor_arm_tasks.py (ArmPointNavTask)
- ithor_arm/ithor_arm_constants.py (constants)
- ithor_arm/arm_calculation_utils.py (coordinate transforms)
- ithor_arm/ithor_arm_task_samplers.py (episode setup)
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NOTE: ai2thor and scipy are available in the v3.3.5 conda env
import ai2thor.controller
import ai2thor.fifo_server
from scipy.spatial.transform import Rotation as R

from easi.tasks.manipulathor.actions import (
    ACTION_SPACE,
    ADITIONAL_ARM_ARGS,
    AGENT_BASE_LOCATION_Y,
    ARM_MAX_HEIGHT,
    ARM_MIN_HEIGHT,
    DONE,
    MANIPULATHOR_COMMIT_ID,
    MOVE_AHEAD,
    MOVE_ARM_CONSTANT,
    MOVE_ARM_HEIGHT_CONSTANT,
    MOVE_THR,
    PICKUP,
    ROTATE_LEFT,
    ROTATE_RIGHT,
)
from easi.utils.logging import get_logger

logger = get_logger(__name__)


# ── Coordinate transform utilities (from arm_calculation_utils.py) ──────────

def _make_rotation_matrix(position: dict, rotation: dict) -> np.ndarray:
    mat = np.zeros((4, 4))
    r = R.from_euler("xyz", [rotation["x"], rotation["y"], rotation["z"]], degrees=True)
    mat[:3, :3] = r.as_matrix()
    mat[3, 3] = 1
    mat[:3, 3] = [position["x"], position["y"], position["z"]]
    return mat


def _position_rotation_from_mat(matrix: np.ndarray) -> dict:
    rotation = R.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    pos = matrix[:3, 3]
    return {
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "rotation": {"x": float(rotation[0]), "y": float(rotation[1]), "z": float(rotation[2])},
    }


# Pre-computed inverse rotation matrices for agent-relative transforms
_INVERSE_ROTATIONS = {}
for _deg in range(0, 361, 45):
    _r = R.from_euler("xyz", [0, _deg % 360, 0], degrees=True)
    _INVERSE_ROTATIONS[_deg % 360] = np.linalg.inv(_r.as_matrix())


def _find_closest_inverse(deg: float) -> np.ndarray:
    for k, v in _INVERSE_ROTATIONS.items():
        if abs(k - deg) < 5:
            return v
    r = R.from_euler("xyz", [0, deg, 0], degrees=True)
    return np.linalg.inv(r.as_matrix())


def convert_world_to_agent_coordinate(world_obj: dict, agent_state: dict) -> dict:
    """Convert world-frame object state to agent-relative coordinates.

    Safety: original code asserts agent only rotates around Y-axis. We warn
    instead of asserting so the bridge doesn't crash mid-episode.
    """
    pos = agent_state["position"]
    rot = agent_state["rotation"]

    # Safety check: agent should only rotate around Y-axis (from original code)
    if abs(rot.get("x", 0)) > 0.01 or abs(rot.get("z", 0)) > 0.01:
        logger.warning(
            "Agent rotation has non-zero x/z: %s — coordinate transform may be inaccurate",
            rot,
        )

    agent_translation = [pos["x"], pos["y"], pos["z"]]
    inv_rot = _find_closest_inverse(rot["y"])
    obj_mat = _make_rotation_matrix(world_obj["position"], world_obj.get("rotation", {"x": 0, "y": 0, "z": 0}))
    obj_translation = np.matmul(inv_rot, obj_mat[:3, 3] - agent_translation)
    obj_mat[:3, 3] = obj_translation
    return _position_rotation_from_mat(obj_mat)


def _diff_position(p1: dict, p2: dict) -> dict:
    """Absolute difference between two position dicts."""
    return {k: abs(p1[k] - p2[k]) for k in p1.keys()}


def _position_distance(s1: dict, s2: dict) -> float:
    """Euclidean distance between two objects with 'position' keys."""
    p1, p2 = s1["position"], s2["position"]
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2
        + (p1["y"] - p2["y"]) ** 2
        + (p1["z"] - p2["z"]) ** 2
    )


def _correct_nan_inf(d: dict) -> dict:
    """Replace NaN/Inf values with 0."""
    return {k: (0 if (v != v or math.isinf(v)) else v) for k, v in d.items()}


# ── Main Environment ────────────────────────────────────────────────────────

# Default controller kwargs (from ENV_ARGS in ithor_arm_constants.py)
DEFAULT_CONTROLLER_KWARGS = {
    "gridSize": 0.25,
    "width": 224,
    "height": 224,
    "visibilityDistance": 1.0,
    "agentMode": "arm",
    "fieldOfView": 100,
    "agentControllerType": "mid-level",
    "useMassThreshold": True,
    "massThreshold": 10,
    "autoSimulation": False,
    "autoSyncTransforms": True,
}


class ManipulaTHOREnv:
    """Self-contained ManipulaTHOR environment for EASI bridge.

    Provides:
    - Episode setup (reset scene, init arm, transport object, teleport agent)
    - Action execution (13 discrete actions)
    - GPS sensor computation (4 state vectors)
    - Metric state tracking (success, pickup, disturbance)
    """

    def __init__(self, controller_kwargs: Optional[dict] = None,
                 verbose_feedback: bool = True):
        self.controller: Optional[ai2thor.controller.Controller] = None
        self._controller_kwargs = controller_kwargs or {}
        self.verbose_feedback = verbose_feedback

        # Episode state
        self.object_id: Optional[str] = None
        self.object_type: Optional[str] = None
        self.target_position: Optional[dict] = None  # {"x":..., "y":..., "z":...}
        self.source_position: Optional[dict] = None
        self.initial_object_locations: Optional[dict] = None

        # Metric state
        self.object_picked_up: bool = False
        self.eplen_pickup: int = 0
        self._took_done_action: bool = False
        self._success: bool = False
        self._step_count: int = 0

    def start(self):
        """Initialize the AI2-THOR controller."""
        kwargs = dict(DEFAULT_CONTROLLER_KWARGS)
        for k, v in self._controller_kwargs.items():
            kwargs[k] = v
        kwargs["server_class"] = ai2thor.fifo_server.FifoServer
        kwargs["commit_id"] = MANIPULATHOR_COMMIT_ID
        self.controller = ai2thor.controller.Controller(**kwargs)
        logger.info("ManipulaTHOR controller started (arm mode, commit=%s)", MANIPULATHOR_COMMIT_ID)

    def stop(self):
        """Shut down the controller."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def reset(self, reset_config: dict) -> dict:
        """Reset to an episode. Returns initial observation dict.

        Args:
            reset_config: Dict with keys:
                scene, object_id, object_type,
                source_position (dict), target_position (dict),
                initial_agent_pose (dict), arm_starting_pose (dict)

        Returns:
            obs dict with keys: frame, depth_frame, metadata
        """
        if self.controller is None:
            self.start()

        scene = reset_config["scene"]
        self.object_id = reset_config["object_id"]
        self.object_type = reset_config.get("object_type", "")
        self.source_position = reset_config["source_position"]
        self.target_position = reset_config["target_position"]

        # Reset metric state
        self.object_picked_up = False
        self.eplen_pickup = 0
        self._took_done_action = False
        self._success = False
        self._step_count = 0

        # 1. Reset scene + standard setup
        self._reset_scene(scene)

        # 2. Initialize arm
        arm_pose = reset_config.get("arm_starting_pose")
        self._initialize_arm(arm_pose)

        # 3. Transport object to source position
        self._transport_object(self.object_id, self.source_position)

        # 4. Teleport agent to initial pose
        agent_pose = reset_config["initial_agent_pose"]
        self._teleport_agent(agent_pose)

        # 5. Record initial object locations for disturbance metric
        self.initial_object_locations = self._get_current_object_locations()

        return self._build_obs()

    def step(self, action_name: str) -> Tuple[dict, float, bool, dict]:
        """Execute one action. Returns (obs, reward, done, info).

        Returns:
            obs: dict with keys: frame, depth_frame, metadata
            reward: float (always 0.0 — no reward shaping for LLM eval)
            done: bool
            info: dict with metric fields
        """
        self._step_count += 1

        # Execute the action
        action_success = self._execute_action(action_name)

        # Check pickup status
        if not self.object_picked_up and self._is_object_in_hand():
            self.object_picked_up = True
            self.eplen_pickup = self._step_count

        # Handle DONE action
        done = False
        if action_name == DONE:
            self._took_done_action = True
            done = True
            obj_state = self._get_object_by_id(self.object_id)
            goal_state = {"position": self.target_position}
            self._success = (
                self.object_picked_up
                and self._obj_state_proximity(obj_state, goal_state)
            )
            action_success = self._success

        # Build info dict with all metric-relevant fields
        info = {
            "last_action_success": float(action_success),
            "action_name": action_name,
            "object_picked_up": float(self.object_picked_up),
            "episode_success": float(self._success),
            "pickup_success": float(self.object_picked_up),
        }

        # Compute additional metrics on success
        if self._success:
            moved = self._get_objects_moved()
            info["success_wo_disturb"] = float(len(moved) == 1)
            info["eplen_success"] = float(self._step_count)
            info["num_objects_disturbed"] = float(max(0, len(moved) - 1))

        if self.object_picked_up and self.eplen_pickup > 0:
            info["eplen_pickup"] = float(self.eplen_pickup)

        # Distance metrics (useful for feedback)
        info["obj_to_goal_distance"] = self._obj_distance_from_goal()
        info["arm_to_obj_distance"] = self._arm_distance_from_obj()

        # Feedback string for prompt builder
        if action_success:
            info["feedback"] = f"Action '{action_name}' succeeded."
        else:
            if self.verbose_feedback:
                error_msg = self.controller.last_event.metadata.get("errorMessage", "")
                if error_msg:
                    info["feedback"] = f"Action '{action_name}' failed. Reason: {error_msg}"
                else:
                    info["feedback"] = f"Action '{action_name}' failed."
            else:
                info["feedback"] = f"Action '{action_name}' failed."

        if self._success:
            info["feedback"] += " Task completed successfully!"
        elif self._took_done_action and not self._success:
            info["feedback"] += " Task failed — object not at goal position."

        obs = self._build_obs()
        return obs, 0.0, done, info

    # ── Observation building ────────────────────────────────────────────

    def _build_obs(self) -> dict:
        """Build observation dict from current controller state."""
        event = self.controller.last_event
        return {
            "frame": event.frame.copy() if event.frame is not None else np.zeros((224, 224, 3), dtype=np.uint8),
            "depth_frame": event.depth_frame.copy() if hasattr(event, 'depth_frame') and event.depth_frame is not None else None,
            "metadata": event.metadata,
        }

    def get_gps_state(self) -> dict:
        """Compute GPS-like state sensors matching original ManipulaTHOR.

        Returns dict with:
            relative_current_obj_state: list[float] (6,) — object pos+rot in agent frame
            relative_obj_to_goal: list[float] (3,) — object-to-goal distance in agent frame
            relative_agent_arm_to_obj: list[float] (3,) — arm-to-object distance in agent frame
            pickedup_object: float — 1.0 if held, 0.0 otherwise
        """
        agent_state = self.controller.last_event.metadata["agent"]
        obj_info = self._get_object_by_id(self.object_id)

        # 1. relative_current_obj_state (6D): object pos+rot in agent frame
        rel_obj = convert_world_to_agent_coordinate(obj_info, agent_state)
        rel_obj_state = [
            rel_obj["position"]["x"], rel_obj["position"]["y"], rel_obj["position"]["z"],
            rel_obj["rotation"]["x"], rel_obj["rotation"]["y"], rel_obj["rotation"]["z"],
        ]

        # 2. relative_obj_to_goal (3D): distance from object to goal in agent frame
        target_state = {
            "position": self.target_position,
            "rotation": {"x": 0, "y": 0, "z": 0},
        }
        rel_curr = convert_world_to_agent_coordinate(obj_info, agent_state)
        rel_goal = convert_world_to_agent_coordinate(target_state, agent_state)
        obj_to_goal = _diff_position(rel_curr["position"], rel_goal["position"])
        rel_obj_to_goal = [obj_to_goal["x"], obj_to_goal["y"], obj_to_goal["z"]]

        # 3. relative_agent_arm_to_obj (3D): arm-to-object distance in agent frame
        hand_state = self._get_absolute_hand_state()
        rel_obj_agent = convert_world_to_agent_coordinate(obj_info, agent_state)
        rel_hand = convert_world_to_agent_coordinate(hand_state, agent_state)
        arm_to_obj = _diff_position(rel_obj_agent["position"], rel_hand["position"])
        rel_arm_to_obj = [arm_to_obj["x"], arm_to_obj["y"], arm_to_obj["z"]]

        # 4. pickedup_object
        picked_up = 1.0 if self.object_picked_up else 0.0

        return {
            "relative_current_obj_state": rel_obj_state,
            "relative_obj_to_goal": rel_obj_to_goal,
            "relative_agent_arm_to_obj": rel_arm_to_obj,
            "pickedup_object": picked_up,
        }

    # ── Scene & arm setup ───────────────────────────────────────────────

    def _reset_scene(self, scene: str):
        """Reset scene with standard ManipulaTHOR setup commands."""
        self.controller.reset(scene)
        self.controller.step(action="MakeAllObjectsMoveable")
        self.controller.step(action="MakeObjectsStaticKinematicMassThreshold")
        # Make all breakable objects unbreakable
        breakable = set(
            o["objectType"]
            for o in self.controller.last_event.metadata["objects"]
            if o.get("breakable")
        )
        for obj_type in breakable:
            self.controller.step(action="MakeObjectsOfTypeUnbreakable", objectType=obj_type)

    def _initialize_arm(self, arm_pose: Optional[dict] = None):
        """Initialize arm to starting position (from arm_calculation_utils.initialize_arm).

        Args:
            arm_pose: Dict with keys x, y, z, rotation, horizon.
                      If None, uses a sensible default.
        """
        if arm_pose:
            self.controller.step(dict(
                action="TeleportFull",
                standing=True,
                x=arm_pose["x"],
                y=arm_pose["y"],
                z=arm_pose["z"],
                rotation=dict(x=0, y=arm_pose["rotation"], z=0),
                horizon=arm_pose["horizon"],
            ))
        self.controller.step(dict(
            action="MoveArm",
            position=dict(x=0.0, y=0, z=0.35),
            **ADITIONAL_ARM_ARGS,
        ))
        self.controller.step(dict(
            action="MoveArmBase",
            y=0.8,
            **ADITIONAL_ARM_ARGS,
        ))

    def _transport_object(self, object_id: str, position: dict):
        """Place object at a specific position (from transport_wrapper)."""
        self.controller.step(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position=position,
            forceKinematic=True,
        )
        self.controller.step(action="AdvancePhysicsStep", simSeconds=1.0)

    def _teleport_agent(self, agent_pose: dict):
        """Teleport agent to a specific pose.

        Handles FLAT format from HF dataset: {x, y, z, rotation, horizon}
        where 'rotation' is a float (Y-axis degrees) and 'horizon' is camera tilt.
        """
        self.controller.step(dict(
            action="TeleportFull",
            standing=True,
            x=agent_pose["x"],
            y=agent_pose["y"],
            z=agent_pose["z"],
            rotation=dict(x=0, y=agent_pose["rotation"], z=0),
            horizon=agent_pose["horizon"],
        ))

    # ── Action execution ────────────────────────────────────────────────

    def _execute_action(self, action_name: str) -> bool:
        """Execute a named action and return success bool.

        Implements the action dispatch logic from ManipulaTHOREnvironment.step().
        """
        if action_name == DONE:
            self.controller.step(action="Pass")
            return True  # success determined separately

        if action_name == PICKUP:
            return self._execute_pickup()

        if "MoveArm" in action_name:
            return self._execute_arm_action(action_name)

        if "Continuous" in action_name:
            return self._execute_nav_action(action_name)

        logger.warning("Unknown action: %s", action_name)
        return False

    def _execute_pickup(self) -> bool:
        """Execute PickUpMidLevel action (from ManipulaTHOREnvironment.step)."""
        if self._is_object_in_hand():
            self.controller.step(action="Pass")
            return True

        pickupable = self.controller.last_event.metadata["arm"].get("pickupableObjects", [])
        if self.object_id in pickupable:
            self.controller.step(action="PickupObject")
            held = self.controller.last_event.metadata["arm"].get("heldObjects", [])
            if held and self.object_id not in held:
                self.controller.step(action="ReleaseObject")

        self.controller.step(action="Pass")
        return self._is_object_in_hand()

    def _execute_arm_action(self, action_name: str) -> bool:
        """Execute arm movement action."""
        action_dict = dict(action=action_name)
        action_dict.update(copy.deepcopy(ADITIONAL_ARM_ARGS))
        base_pos = self._get_current_arm_state()

        if "MoveArmHeight" in action_name:
            action_dict["action"] = "MoveArmBase"
            if action_name == "MoveArmHeightP":
                base_pos["h"] += MOVE_ARM_HEIGHT_CONSTANT
            elif action_name == "MoveArmHeightM":
                base_pos["h"] -= MOVE_ARM_HEIGHT_CONSTANT
            action_dict["y"] = base_pos["h"]
        else:
            action_dict["action"] = "MoveArm"
            deltas = {
                "MoveArmXP": ("x", MOVE_ARM_CONSTANT),
                "MoveArmXM": ("x", -MOVE_ARM_CONSTANT),
                "MoveArmYP": ("y", MOVE_ARM_CONSTANT),
                "MoveArmYM": ("y", -MOVE_ARM_CONSTANT),
                "MoveArmZP": ("z", MOVE_ARM_CONSTANT),
                "MoveArmZM": ("z", -MOVE_ARM_CONSTANT),
            }
            axis, delta = deltas[action_name]
            base_pos[axis] += delta
            action_dict["position"] = {
                k: v for k, v in base_pos.items() if k in ("x", "y", "z")
            }

        self.controller.step(action_dict)
        return self.controller.last_event.metadata["lastActionSuccess"]

    def _execute_nav_action(self, action_name: str) -> bool:
        """Execute navigation action."""
        action_dict = dict(action=action_name)
        action_dict.update(copy.deepcopy(ADITIONAL_ARM_ARGS))

        if action_name == MOVE_AHEAD:
            action_dict["action"] = "MoveAgent"
            action_dict["ahead"] = 0.2
        elif action_name == ROTATE_RIGHT:
            action_dict["action"] = "RotateAgent"
            action_dict["degrees"] = 45
        elif action_name == ROTATE_LEFT:
            action_dict["action"] = "RotateAgent"
            action_dict["degrees"] = -45

        self.controller.step(action_dict)
        return self.controller.last_event.metadata["lastActionSuccess"]

    # ── State queries ───────────────────────────────────────────────────

    def _is_object_in_hand(self) -> bool:
        held = self.controller.last_event.metadata["arm"].get("heldObjects", [])
        return self.object_id in held

    def _get_object_by_id(self, object_id: str) -> Optional[dict]:
        for o in self.controller.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                o["position"] = _correct_nan_inf(o["position"])
                return o
        return None

    def _get_absolute_hand_state(self) -> dict:
        joints = self.controller.last_event.metadata["arm"]["joints"]
        arm = copy.deepcopy(joints[-1])
        xyz = _correct_nan_inf(arm["position"])
        return {"position": xyz, "rotation": {"x": 0, "y": 0, "z": 0}}

    def _get_current_arm_state(self) -> dict:
        """Get arm state in root-relative coordinates + normalized height."""
        event = self.controller.last_event
        offset = event.metadata["agent"]["position"]["y"] - AGENT_BASE_LOCATION_Y
        h_min = ARM_MIN_HEIGHT + offset
        h_max = ARM_MAX_HEIGHT + offset
        joints = event.metadata["arm"]["joints"]
        arm = joints[-1]
        xyz = copy.deepcopy(arm["rootRelativePosition"])
        height_arm = joints[0]["position"]["y"]
        xyz["h"] = (height_arm - h_min) / (h_max - h_min)
        return {k: (0 if (v != v or math.isinf(v)) else v) for k, v in xyz.items()}

    def _get_current_object_locations(self) -> dict:
        result = {}
        for o in self.controller.last_event.metadata["objects"]:
            result[o["objectId"]] = dict(
                position=copy.deepcopy(o["position"]),
                rotation=copy.deepcopy(o["rotation"]),
            )
        return result

    def _obj_state_proximity(self, obj_state: dict, goal_state: dict) -> bool:
        """Check if object is within 0.1m of goal (per axis)."""
        eps = MOVE_ARM_CONSTANT * 2  # = 0.1
        p1, p2 = obj_state["position"], goal_state["position"]
        return (
            abs(p1["x"] - p2["x"]) < eps
            and abs(p1["y"] - p2["y"]) < eps
            and abs(p1["z"] - p2["z"]) < eps
        )

    def _obj_distance_from_goal(self) -> float:
        obj = self._get_object_by_id(self.object_id)
        if obj is None:
            return float("inf")
        return _position_distance(obj, {"position": self.target_position})

    def _arm_distance_from_obj(self) -> float:
        obj = self._get_object_by_id(self.object_id)
        if obj is None:
            return float("inf")
        hand = self._get_absolute_hand_state()
        return _position_distance(obj, hand)

    def _get_objects_moved(self) -> list:
        """Return list of object IDs that moved since episode start."""
        if self.initial_object_locations is None:
            return []
        current = self._get_current_object_locations()
        moved = []
        for oid in current:
            if oid not in self.initial_object_locations:
                continue
            curr_pose = current[oid]
            init_pose = self.initial_object_locations[oid]
            if not self._close_enough(curr_pose, init_pose, MOVE_THR):
                moved.append(oid)
        return moved

    @staticmethod
    def _close_enough(current: dict, initial: dict, threshold: float) -> bool:
        """Check if position and rotation are within threshold (per axis)."""
        for k in ("x", "y", "z"):
            if abs(current["position"][k] - initial["position"][k]) > threshold:
                return False
            if abs(current["rotation"][k] - initial["rotation"][k]) > threshold:
                return False
        return True
