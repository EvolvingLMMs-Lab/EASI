"""Bridge subprocess for AI2-THOR v2.1.0 (EB-Alfred track).

This script runs inside the easi_ai2thor_v2_1_0 conda env (Python 3.8).
It communicates with the parent process via filesystem IPC.

Ported from EmbodiedBench:
- envs/eb_alfred/thor_connector.py (high-level skill API)
- envs/eb_alfred/env/thor_env.py (controller init, scene restoration, state tracking)

Usage:
    python bridge.py --workspace /tmp/easi_xxx --data-dir /path/to/datasets
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

# Add repo root to path for easi imports
_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.communication.filesystem import (
    poll_for_command,
    write_response,
    write_status,
)
from easi.communication.schemas import (
    make_error_response,
    make_observation_response,
)
from easi.simulators.ai2thor.v2_1_0.thor_utils import (
    AGENT_HORIZON_ADJ,
    AGENT_ROTATE_ADJ,
    AGENT_STEP_SIZE,
    CAMERA_HEIGHT_OFFSET,
    RECORD_SMOOTHING_FACTOR,
    RENDER_CLASS_IMAGE,
    RENDER_DEPTH_IMAGE,
    RENDER_IMAGE,
    RENDER_OBJECT_IMAGE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    VISIBILITY_DISTANCE,
    evaluate_goal_conditions,
    get_objects_of_type,
    get_obj_of_type_closest_to_obj,
    load_task_json_with_repeat,
    natural_word_to_ithor_name,
)

logger = logging.getLogger("easi.bridge.ai2thor_v2_1_0")


class EBAlfredBridge:
    """EB-Alfred bridge managing AI2-THOR controller and high-level skills.

    Ported from EmbodiedBench's ThorEnv + ThorConnector.
    """

    def __init__(self, workspace, data_dir):
        self.workspace = Path(workspace)
        self.data_dir = Path(data_dir)
        self.controller = None
        self.last_event = None
        self.traj_data = None
        self.step_count = 0

        # State tracking (from EmbodiedBench ThorEnv)
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # Navigation cache
        self.reachable_positions = None
        self.reachable_position_kdtree = None

        # Agent height (set after reset from init_action)
        self.agent_height = 0.9009992

        # Skill state
        self.cur_receptacle = None
        self.put_count_dict = {}
        self.sliced = False

        # Output directory for saving images (set per-episode from reset command)
        self.episode_output_dir = None

    def start(self):
        """Initialize AI2-THOR controller."""
        from ai2thor.controller import Controller

        logger.info("Starting AI2-THOR controller...")
        self.controller = Controller(quality="MediumCloseFitShadows")
        self.controller.start(
            x_display=os.environ.get("DISPLAY", "0"),
            player_screen_height=SCREEN_HEIGHT,
            player_screen_width=SCREEN_WIDTH,
        )
        logger.info("AI2-THOR controller started.")

    def stop(self):
        """Stop the AI2-THOR controller."""
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
            self.controller = None

    def _step(self, action_dict):
        """Execute a raw THOR action and update last_event."""
        self.last_event = self.controller.step(action_dict)
        return self.last_event

    # --- Reset ---

    def reset(self, reset_config):
        """Reset to an EB-Alfred episode (or a default scene for smoke tests).

        If task_path is provided, loads traj_data.json, resets scene,
        restores objects, and teleports agent.
        If task_path is missing (smoke test), resets to a default scene.
        """
        task_path = reset_config.get("task_path")

        if not task_path:
            # Smoke test mode: reset to a default scene without episode data
            return self._reset_default_scene()

        repeat_idx = reset_config.get("repeat_idx", 0)

        self.traj_data = load_task_json_with_repeat(
            task_path, repeat_idx, str(self.data_dir)
        )
        self.cleaned_objects.clear()
        self.cooled_objects.clear()
        self.heated_objects.clear()
        self.step_count = 0
        self.cur_receptacle = None
        self.put_count_dict = {}
        self.sliced = False

        # Reset scene
        scene_num = self.traj_data["scene"]["scene_num"]
        scene_name = "FloorPlan%d" % scene_num
        logger.info("Resetting to scene: %s", scene_name)
        self.controller.reset(scene_name)

        # Restore object poses, toggles, dirty states
        self._restore_scene()

        # Teleport agent to initial position
        init_action = dict(self.traj_data["scene"]["init_action"])
        if init_action.get("action") == "TeleportFull":
            init_action.pop("rotateOnTeleport", None)
            init_action["standing"] = True
        self.last_event = self.controller.step(init_action)

        # Cache agent height
        self.agent_height = self.last_event.metadata["agent"]["position"]["y"]

        # Cache reachable positions for navigation
        self._cache_reachable_positions()

        return self._make_observation_response()

    def _reset_default_scene(self):
        """Reset to FloorPlan10 for smoke testing (no episode data needed)."""
        logger.info("Smoke test mode: resetting to default scene FloorPlan10")
        self.traj_data = None
        self.cleaned_objects.clear()
        self.cooled_objects.clear()
        self.heated_objects.clear()
        self.step_count = 0
        self.cur_receptacle = None
        self.put_count_dict = {}
        self.sliced = False

        self.controller.reset("FloorPlan10")
        self.last_event = self.controller.step(dict(
            action="Initialize",
            gridSize=AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR,
            cameraY=CAMERA_HEIGHT_OFFSET,
            renderImage=RENDER_IMAGE,
            renderDepthImage=RENDER_DEPTH_IMAGE,
            renderClassImage=RENDER_CLASS_IMAGE,
            renderObjectImage=RENDER_OBJECT_IMAGE,
            visibility_distance=VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        self.agent_height = self.last_event.metadata["agent"]["position"]["y"]
        self._cache_reachable_positions()
        return self._make_observation_response()

    def _restore_scene(self):
        """Restore EB-Alfred scene state from traj_data.

        Ported from EmbodiedBench ThorEnv.restore_scene().
        """
        grid_size = AGENT_STEP_SIZE / RECORD_SMOOTHING_FACTOR

        self._step(dict(
            action="Initialize",
            gridSize=grid_size,
            cameraY=CAMERA_HEIGHT_OFFSET,
            renderImage=RENDER_IMAGE,
            renderDepthImage=RENDER_DEPTH_IMAGE,
            renderClassImage=RENDER_CLASS_IMAGE,
            renderObjectImage=RENDER_OBJECT_IMAGE,
            visibility_distance=VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))

        scene = self.traj_data["scene"]
        object_toggles = scene.get("object_toggles", [])
        object_poses = scene.get("object_poses", [])
        dirty_and_empty = scene.get("dirty_and_empty", False)

        if len(object_toggles) > 0:
            self._step(dict(action="SetObjectToggles", objectToggles=object_toggles))

        if dirty_and_empty:
            self._step(dict(
                action="SetStateOfAllObjects",
                StateChange="CanBeDirty",
                forceAction=True,
            ))
            self._step(dict(
                action="SetStateOfAllObjects",
                StateChange="CanBeFilled",
                forceAction=False,
            ))

        if len(object_poses) > 0:
            self._step(dict(action="SetObjectPoses", objectPoses=object_poses))

    def _cache_reachable_positions(self):
        """Cache reachable positions + KD-tree for navigation."""
        event = self._step(dict(action="GetReachablePositions"))
        free_positions = event.metadata["actionReturn"]
        self.reachable_positions = np.array(
            [[p["x"], p["y"], p["z"]] for p in free_positions]
        )
        self.reachable_position_kdtree = spatial.KDTree(self.reachable_positions)

    # --- Step ---

    def step(self, action_text):
        """Execute a high-level skill and return observation + metrics."""
        self.step_count += 1

        # Smoke test mode: execute raw THOR action instead of high-level skill
        if self.traj_data is None:
            self._step(dict(action=action_text))
            return self._make_observation_response(
                reward=0.0,
                done=False,
                info={
                    "last_action_success": 1.0 if self.last_event.metadata["lastActionSuccess"] else 0.0,
                    "env_step": float(self.step_count),
                    "feedback": "success" if self.last_event.metadata["lastActionSuccess"] else "action failed",
                },
            )

        result = self._execute_skill(action_text)
        success = result.get("success", False)

        # Update cleaned/cooled/heated tracking
        self._update_states()

        # Evaluate goal conditions
        goal_satisfied, progress = evaluate_goal_conditions(
            self.traj_data,
            self.last_event,
            self.cleaned_objects,
            self.cooled_objects,
            self.heated_objects,
        )

        feedback = result.get("message", "")
        if success:
            feedback = "success"

        return self._make_observation_response(
            reward=float(progress),
            done=goal_satisfied,
            info={
                "task_success": 1.0 if goal_satisfied else 0.0,
                "task_progress": progress,
                "last_action_success": 1.0 if success else 0.0,
                "env_step": float(self.step_count),
                "feedback": feedback,
            },
        )

    def _execute_skill(self, instruction):
        """Map high-level skill text to THOR API calls.

        Ported from EmbodiedBench ThorConnector.llm_skill_interact().
        """
        # Clear cur_receptacle unless put/open
        if not (instruction.startswith("put down ") or instruction.startswith("open ")):
            self.cur_receptacle = None

        ret_msg = ""

        if instruction.startswith("find "):
            obj_name = instruction.replace("find a ", "").replace("find an ", "")
            self.cur_receptacle = obj_name
            ret_msg = self._nav_obj(
                natural_word_to_ithor_name(obj_name), self.sliced
            )

        elif instruction.startswith("pick up "):
            obj_name = instruction.replace("pick up the ", "")
            ret_msg = self._pick(natural_word_to_ithor_name(obj_name))

        elif instruction.startswith("put down "):
            if self.cur_receptacle is None:
                ret_msg = self._drop()
            else:
                receptacle = self.cur_receptacle
                if receptacle in self.put_count_dict:
                    self.put_count_dict[receptacle] += 1
                else:
                    self.put_count_dict[receptacle] = 1

                ret_msg = self._put(natural_word_to_ithor_name(receptacle))

                if len(ret_msg) > 16 and self.put_count_dict[receptacle] >= 3:
                    self._drop()
                    ret_msg += ". The robot dropped the object instead."
                    self.last_event.metadata["lastActionSuccess"] = False

        elif instruction.startswith("open "):
            obj_name = instruction.replace("open the ", "")
            ret_msg = self._open(natural_word_to_ithor_name(obj_name))

        elif instruction.startswith("close "):
            obj_name = instruction.replace("close the ", "")
            ret_msg = self._close(natural_word_to_ithor_name(obj_name))

        elif instruction.startswith("turn on "):
            obj_name = instruction.replace("turn on the ", "")
            ret_msg = self._toggleon(natural_word_to_ithor_name(obj_name))

        elif instruction.startswith("turn off "):
            obj_name = instruction.replace("turn off the ", "")
            ret_msg = self._toggleoff(natural_word_to_ithor_name(obj_name))

        elif instruction.startswith("slice "):
            obj_name = instruction.replace("slice the ", "")
            ret_msg = self._slice(natural_word_to_ithor_name(obj_name))
            self.sliced = True

        elif instruction.startswith("drop"):
            ret_msg = self._drop()

        else:
            ret_msg = "instruction not supported"

        success = len(ret_msg) == 0
        if not success:
            logger.warning("Skill failed: %s -> %s", instruction, ret_msg)

        return {"action": instruction, "success": success, "message": ret_msg}

    # --- State Tracking ---

    def _update_states(self):
        """Track cleaned/cooled/heated objects.

        Ported from EmbodiedBench ThorEnv.update_states().
        """
        if self.last_event is None:
            return

        metadata = self.last_event.metadata
        if not metadata.get("lastActionSuccess", False):
            return

        # Check if last action was a relevant toggle/close
        last_action = metadata.get("lastAction", "")
        last_obj_id = metadata.get("lastActionObjectId", "")

        # Clean: ToggleObjectOn on Faucet -> items in SinkBasin get cleaned
        if last_action == "ToggleObjectOn" and "Faucet" in last_obj_id:
            sink_basin = get_obj_of_type_closest_to_obj(
                "SinkBasin", last_obj_id, metadata
            )
            if sink_basin and sink_basin.get("receptacleObjectIds"):
                self.cleaned_objects.update(sink_basin["receptacleObjectIds"])

        # Heat: ToggleObjectOn on Microwave -> items inside get heated
        if last_action == "ToggleObjectOn" and "Microwave" in last_obj_id:
            microwaves = get_objects_of_type("Microwave", metadata)
            if microwaves:
                heated_ids = microwaves[0].get("receptacleObjectIds")
                if heated_ids:
                    self.heated_objects.update(heated_ids)

        # Cool: CloseObject on Fridge -> items inside get cooled
        if last_action == "CloseObject" and "Fridge" in last_obj_id:
            fridges = get_objects_of_type("Fridge", metadata)
            if fridges:
                cooled_ids = fridges[0].get("receptacleObjectIds")
                if cooled_ids:
                    self.cooled_objects.update(cooled_ids)

    # --- Individual Skill Implementations ---

    def _nav_obj(self, target_obj, prefer_sliced=False):
        """Navigate to an object by teleporting to closest reachable position.

        Ported from EmbodiedBench ThorConnector.nav_obj().
        """
        objects = self.last_event.metadata["objects"]
        logger.info("nav_obj: %s", target_obj)

        # Resolve object ID
        if "|" in target_obj:
            obj_id = target_obj
            base_name = target_obj.split("|")[0]
            tmp_id, tmp_data = self._get_obj_id_from_name(
                base_name, priority_in_visibility=True, priority_sliced=prefer_sliced
            )
            if tmp_id and "Sliced" in tmp_id and obj_id in tmp_id:
                obj_id = tmp_id
        else:
            obj_id, _ = self._get_obj_id_from_name(
                target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced
            )

        # Find object index
        obj_idx = -1
        for i, o in enumerate(objects):
            if o["objectId"] == obj_id:
                obj_idx = i
                break

        if obj_idx == -1:
            return (
                "Cannot find %s. This object may not exist in this scene. "
                "Try to explore other instances instead." % target_obj
            )

        # Teleport to closest reachable position facing the object
        loc = objects[obj_idx]["position"]
        obj_rot = objects[obj_idx]["rotation"]["y"]
        max_attempts = 20
        teleport_success = False
        reachable_pos_idx = 0

        for i in range(max_attempts):
            reachable_pos_idx += 1
            if i == 10 and target_obj in ("Fridge", "Microwave"):
                reachable_pos_idx -= 10

            closest_loc = self._find_close_reachable_position(
                [loc["x"], loc["y"], loc["z"]], reachable_pos_idx
            )
            if closest_loc is None:
                continue

            # Calculate desired rotation angle
            rot_angle = math.atan2(
                -(loc["x"] - closest_loc[0]), loc["z"] - closest_loc[2]
            )
            if rot_angle > 0:
                rot_angle -= 2 * math.pi
            rot_angle = -(180 / math.pi) * rot_angle

            # Special angle filtering for Fridge/Microwave
            if i < 10 and target_obj in ("Fridge", "Microwave"):
                angle_d = abs(self._angle_diff(rot_angle, obj_rot))
                if target_obj == "Fridge" and not (
                    (70 < angle_d < 110) or (250 < angle_d < 290)
                ):
                    continue
                if target_obj == "Microwave" and not (
                    (160 < angle_d < 200) or (0 <= angle_d < 20)
                ):
                    continue

            # Calculate horizon angle
            camera_height = self.agent_height + CAMERA_HEIGHT_OFFSET
            xz_dist = math.hypot(
                loc["x"] - closest_loc[0], loc["z"] - closest_loc[2]
            )
            hor_angle = math.atan2(loc["y"] - camera_height, xz_dist)
            hor_angle = (180 / math.pi) * hor_angle * 0.9

            self._step(dict(
                action="TeleportFull",
                x=closest_loc[0],
                y=self.agent_height,
                z=closest_loc[2],
                rotation=rot_angle,
                horizon=-hor_angle,
            ))

            if self.last_event.metadata["lastActionSuccess"]:
                teleport_success = True
                break
            else:
                logger.debug(
                    "TeleportFull failed: %s",
                    self.last_event.metadata.get("errorMessage", ""),
                )

        if not teleport_success:
            return "Cannot move to %s" % target_obj

        return ""

    def _pick(self, obj_name):
        """Pick up an object.

        Ported from EmbodiedBench ThorConnector.pick().
        """
        obj_id, obj_data = self._get_obj_id_from_name(
            obj_name, only_pickupable=True, priority_in_visibility=True,
            priority_sliced=self.sliced,
        )
        logger.info("pick: %s -> %s", obj_name, obj_id)

        if obj_id is None:
            return "Cannot find %s to pick up. Find the object before picking up it" % obj_name

        if (
            obj_data.get("visible") is False
            and obj_data.get("parentReceptacles")
            and len(obj_data["parentReceptacles"]) > 0
        ):
            recep_name = obj_data["parentReceptacles"][0]
            ret_msg = (
                "%s is not visible because it is in %s. "
                "Note: multiple instances of %s may exist"
                % (obj_name, recep_name, recep_name)
            )
            # Try anyway
            self._step(dict(
                action="PickupObject", objectId=obj_id, forceAction=False
            ))
        else:
            self._step(dict(
                action="PickupObject", objectId=obj_id, forceAction=False
            ))
            ret_msg = ""

            if not self.last_event.metadata["lastActionSuccess"]:
                inventory = self.last_event.metadata.get("inventoryObjects", [])
                if len(inventory) == 0:
                    ret_msg = "Robot is not holding any object"
                else:
                    holding_type = inventory[0].get("objectType", "unknown")
                    ret_msg = "Robot is currently holding %s" % holding_type

        if self.last_event.metadata["lastActionSuccess"]:
            ret_msg = ""

        return ret_msg

    def _put(self, receptacle_name):
        """Put held object onto a receptacle.

        Ported from EmbodiedBench ThorConnector.put().
        Uses 2x7x2 retry loop with movement adjustments.
        """
        inventory = self.last_event.metadata.get("inventoryObjects", [])
        if len(inventory) == 0:
            return "Robot is not holding any object"

        holding_obj_id = inventory[0]["objectId"]
        orig_receptacle_name = receptacle_name
        ret_msg = ""
        halt = False
        last_recep_id = None
        exclude_obj_id = None

        for k in range(2):  # try closest and next closest receptacle
            for j in range(7):  # movement/look adjustments
                for i in range(2):  # try inherited receptacles (SinkBasin etc.)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id

                    if k == 0 and "|" in orig_receptacle_name:
                        if i == 1:
                            continue
                        recep_id = orig_receptacle_name
                        receptacle_name = orig_receptacle_name.split("|")[0]
                    else:
                        if "Sink" in receptacle_name or "Bathtub" in receptacle_name:
                            if i == 0:
                                recep_id, _ = self._get_obj_id_from_name(
                                    receptacle_name, get_inherited=True,
                                    exclude_obj_id=exclude_obj_id,
                                )
                            else:
                                recep_id, _ = self._get_obj_id_from_name(
                                    receptacle_name, exclude_obj_id=exclude_obj_id,
                                )
                        else:
                            if i == 0:
                                recep_id, _ = self._get_obj_id_from_name(
                                    receptacle_name, exclude_obj_id=exclude_obj_id,
                                )
                            else:
                                recep_id, _ = self._get_obj_id_from_name(
                                    receptacle_name, get_inherited=True,
                                    exclude_obj_id=exclude_obj_id,
                                )

                    if not recep_id:
                        ret_msg = (
                            "Putting the object on %s failed. First check whether "
                            "the receptacle is open or not. Also try other instances "
                            "of the receptacle" % receptacle_name
                        )
                        continue

                    # Movement adjustments to get receptacle in view
                    if j == 1:
                        self._step(dict(action="LookUp"))
                        self._step(dict(action="LookUp"))
                    elif j == 2:
                        self._step(dict(action="LookDown"))
                        self._step(dict(action="LookDown"))
                        self._step(dict(action="LookDown"))
                        self._step(dict(action="LookDown"))
                    elif j == 3:
                        self._step(dict(action="LookUp"))
                        self._step(dict(action="LookUp"))
                        self._step(dict(action="MoveBack"))
                    elif j == 4:
                        self._step(dict(action="MoveAhead"))
                        for _ in range(4):
                            self._step(dict(action="MoveRight"))
                    elif j == 5:
                        for _ in range(8):
                            self._step(dict(action="MoveLeft"))
                    elif j == 6:
                        for _ in range(4):
                            self._step(dict(action="MoveRight"))
                        self._step(dict(action="RotateHand", x=40))

                    self._step(dict(
                        action="PutObject",
                        objectId=holding_obj_id,
                        receptacleObjectId=recep_id,
                        forceAction=True,
                    ))
                    last_recep_id = recep_id

                    if self.last_event.metadata["lastActionSuccess"]:
                        ret_msg = ""
                        halt = True
                        break
                    else:
                        logger.debug(
                            "PutObject failed: %s",
                            self.last_event.metadata.get("errorMessage", ""),
                        )
                        ret_msg = (
                            "Putting the object on %s failed. First check the "
                            "receptacle is open or not. Also try other instances "
                            "of the receptacle" % receptacle_name
                        )
                if halt:
                    break
            if halt:
                break

        return ret_msg

    def _open(self, obj_name):
        """Open a receptacle.

        Ported from EmbodiedBench ThorConnector.open().
        """
        logger.info("open: %s", obj_name)

        if "|" in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split("|")[0]
        else:
            obj_id, _ = self._get_obj_id_from_name(obj_name)

        if obj_id is None:
            return "Cannot find %s to open. Find the object before opening it" % obj_name

        # Check if already open
        open_flag = False
        for ob in self.last_event.metadata["objects"]:
            if ob["objectId"] == obj_id and ob.get("openable") and ob.get("isOpen"):
                open_flag = True
                break

        ret_msg = ""
        for i in range(4):
            self._step(dict(action="OpenObject", objectId=obj_id))

            if self.last_event.metadata["lastActionSuccess"]:
                ret_msg = ""
                break
            else:
                if open_flag:
                    ret_msg = "Open action failed. The %s is already open" % obj_name
                else:
                    ret_msg = "Open action failed."

                # Move around to avoid self-collision
                if i == 0:
                    self._step(dict(action="MoveBack"))
                elif i == 1:
                    self._step(dict(action="MoveBack"))
                    self._step(dict(action="MoveRight"))
                elif i == 2:
                    self._step(dict(action="MoveLeft"))
                    self._step(dict(action="MoveLeft"))

        return ret_msg

    def _close(self, obj_name):
        """Close a receptacle.

        Ported from EmbodiedBench ThorConnector.close().
        """
        logger.info("close: %s", obj_name)

        if "|" in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split("|")[0]
        else:
            obj_id, _ = self._get_obj_id_from_name(obj_name)

        if obj_id is None:
            return "Cannot find %s to close" % obj_name

        self._step(dict(action="CloseObject", objectId=obj_id))

        if not self.last_event.metadata["lastActionSuccess"]:
            ret_msg = "Close action failed"
            for ob in self.last_event.metadata["objects"]:
                if (
                    ob["objectId"] == obj_id
                    and ob.get("openable")
                    and not ob.get("isOpen")
                ):
                    ret_msg += ". The %s is already closed" % obj_name
                    break
            return ret_msg

        return ""

    def _toggleon(self, obj_name):
        """Toggle an object on.

        Ported from EmbodiedBench ThorConnector.toggleon().
        """
        logger.info("toggleon: %s", obj_name)
        obj_id, _ = self._get_obj_id_from_name(obj_name, only_toggleable=True)

        if obj_id is None:
            return "Cannot find %s to turn on" % obj_name

        try:
            self._step(dict(action="ToggleObjectOn", objectId=obj_id))
            if not self.last_event.metadata["lastActionSuccess"]:
                return "Turn on action failed"
        except Exception:
            self.last_event.metadata["lastActionSuccess"] = False
            return "Turn on action failed"

        return ""

    def _toggleoff(self, obj_name):
        """Toggle an object off.

        Ported from EmbodiedBench ThorConnector.toggleoff().
        """
        logger.info("toggleoff: %s", obj_name)
        obj_id, _ = self._get_obj_id_from_name(obj_name, only_toggleable=True)

        if obj_id is None:
            return "Cannot find %s to turn off" % obj_name

        self._step(dict(action="ToggleObjectOff", objectId=obj_id))

        if not self.last_event.metadata["lastActionSuccess"]:
            return "Turn off action failed"

        return ""

    def _slice(self, obj_name):
        """Slice an object.

        Ported from EmbodiedBench ThorConnector.slice().
        """
        logger.info("slice: %s", obj_name)
        obj_id, _ = self._get_obj_id_from_name(obj_name)

        if obj_id is None:
            return "Cannot find %s to slice" % obj_name

        self._step(dict(action="SliceObject", objectId=obj_id))

        if not self.last_event.metadata["lastActionSuccess"]:
            return "Slice action failed"

        return ""

    def _drop(self):
        """Drop the held object.

        Ported from EmbodiedBench ThorConnector.drop().
        """
        logger.info("drop")
        self._step(dict(action="DropHandObject", forceAction=True))

        if not self.last_event.metadata["lastActionSuccess"]:
            inventory = self.last_event.metadata.get("inventoryObjects", [])
            if len(inventory) == 0:
                return "Robot is not holding any object"
            return "Drop action failed"

        return ""

    # --- Object Lookup Helpers ---

    def _get_obj_id_from_name(
        self,
        obj_name,
        only_pickupable=False,
        only_toggleable=False,
        priority_sliced=False,
        get_inherited=False,
        parent_receptacle_penalty=True,
        priority_in_visibility=False,
        exclude_obj_id=None,
    ):
        """Find the closest object matching name, with priority scoring.

        Ported from EmbodiedBench ThorConnector.get_obj_id_from_name().
        """
        obj_id = None
        obj_data = None
        min_distance = 1e8

        # If name contains digits (e.g. 'Cabinet_2'), match by name directly
        if any(c.isdigit() for c in obj_name):
            for obj in self.last_event.metadata["objects"]:
                if obj_name in obj["name"]:
                    return obj["objectId"], obj
            return None, None

        for obj in self.last_event.metadata["objects"]:
            if obj["objectId"] == exclude_obj_id:
                continue

            if only_pickupable and not obj.get("pickupable", False):
                continue
            if only_toggleable and not obj.get("toggleable", False):
                continue

            # Match object type (case-insensitive)
            obj_type = obj["objectId"].split("|")[0]
            if obj_type.casefold() != obj_name.casefold():
                continue

            # For inherited receptacles (e.g. SinkBasin), check ID format
            if get_inherited and len(obj["objectId"].split("|")) != 5:
                continue
            if not get_inherited and len(obj["objectId"].split("|")) == 5:
                # Skip inherited if not requested
                pass

            distance = obj.get("distance", 1e8)
            penalty = 0

            # Penalize objects inside closed receptacles
            if parent_receptacle_penalty and obj.get("parentReceptacles"):
                for p in obj["parentReceptacles"]:
                    is_open = self._get_object_prop(p, "isOpen")
                    openable = self._get_object_prop(p, "openable")
                    if openable is True and is_open is False:
                        penalty += 100000
                        break

            # Prefer empty stove burners
            if obj_name.casefold() == "stoveburner":
                recep_ids = obj.get("receptacleObjectIds") or []
                if len(recep_ids) > 0:
                    penalty += 10000

            # Penalize non-visible objects
            if priority_in_visibility and not obj.get("visible", False):
                penalty += 1000

            # Prefer sliced objects
            if priority_sliced and "_Slice" in obj.get("name", ""):
                penalty -= 100

            if distance + penalty < min_distance:
                min_distance = distance + penalty
                obj_data = obj
                obj_id = obj["objectId"]

        return obj_id, obj_data

    def _get_object_prop(self, name, prop):
        """Get a property of an object by name/ID substring."""
        for obj in self.last_event.metadata["objects"]:
            if name in obj["objectId"]:
                return obj.get(prop)
        return None

    def _find_close_reachable_position(self, loc, nth=1):
        """Find the nth closest reachable position to a location."""
        if self.reachable_position_kdtree is None:
            return None
        n_positions = len(self.reachable_positions)
        k = min(nth + 1, n_positions)
        if k == 0:
            return None
        d, idx = self.reachable_position_kdtree.query(loc, k=k)
        selected = min(nth - 1, k - 1)
        return self.reachable_positions[idx[selected]] if k > 1 else self.reachable_positions[idx]

    @staticmethod
    def _angle_diff(x, y):
        """Calculate angle difference in degrees."""
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))

    # --- Observation ---

    def _make_observation_response(self, reward=0.0, done=False, info=None):
        """Save RGB frame and return IPC response."""
        from PIL import Image

        event = self.last_event

        # Save to episode_output_dir if set, else IPC workspace
        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        rgb_path = save_dir / ("rgb_%04d.png" % self.step_count)

        # Save frame as PNG
        Image.fromarray(event.frame).save(str(rgb_path))

        agent = event.metadata["agent"]
        pose = [
            agent["position"]["x"],
            agent["position"]["y"],
            agent["position"]["z"],
            agent["rotation"]["y"],
            agent.get("cameraHorizon", 0),
            0,
        ]

        return make_observation_response(
            rgb_path=str(rgb_path),
            agent_pose=pose,
            metadata={"step": str(self.step_count)},
            reward=reward,
            done=done,
            info=info or {},
        )


# --- Main bridge loop ---

def run_bridge(workspace, data_dir):
    """Main bridge loop for AI2-THOR v2.1.0."""
    bridge = EBAlfredBridge(workspace=workspace, data_dir=data_dir)

    logger.info("AI2-THOR v2.1.0 bridge starting (workspace: %s)", workspace)
    bridge.start()

    write_status(workspace, ready=True)

    while True:
        try:
            command = poll_for_command(workspace, timeout=300.0)
        except Exception as e:
            logger.error("Failed to read command: %s", e)
            break

        cmd_type = command.get("type")

        if cmd_type == "reset":
            episode_id = command.get("episode_id", "unknown")
            reset_config = command.get("reset_config", {})
            logger.info("Reset: episode_id=%s, task=%s",
                        episode_id, reset_config.get("task_path", "?"))

            # Set episode output directory for image saving
            raw_output_dir = command.get("episode_output_dir")
            if raw_output_dir:
                bridge.episode_output_dir = raw_output_dir
                Path(raw_output_dir).mkdir(parents=True, exist_ok=True)
            else:
                bridge.episode_output_dir = None

            try:
                response = bridge.reset(reset_config)
                write_response(workspace, response)
            except Exception as e:
                logger.exception("Reset failed")
                write_response(workspace, make_error_response(str(e)))

        elif cmd_type == "step":
            action_data = command.get("action", {})
            action_text = action_data.get("action_name", "")
            logger.debug("Step %d: action=%s", bridge.step_count + 1, action_text)

            try:
                response = bridge.step(action_text)
                write_response(workspace, response)
            except Exception as e:
                logger.exception("Step failed")
                write_response(workspace, make_error_response(str(e)))

        elif cmd_type == "close":
            logger.info("Close command received")
            bridge.stop()
            write_response(workspace, {"status": "ok"})
            break

        else:
            write_response(
                workspace, make_error_response("Unknown command: %s" % cmd_type)
            )


def main():
    parser = argparse.ArgumentParser(description="AI2-THOR v2.1.0 bridge (EB-Alfred)")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_bridge(workspace=args.workspace, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
