import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)

from typing import List, Union, Optional
import numpy as np
from tdw.replicant.ik_plans.ik_plan_type import IkPlanType
from tdw.replicant.action_status import ActionStatus
from HAZARD.policy.astar import get_astar_path, get_astar_weight
from tdw.tdw_utils import TDWUtils
from tdw.replicant.arm import Arm

"""
All actions here are single-agent
"""

def visualize_obs(env, obs, suffix="0", save_dir=os.path.join(PATH, "logs"), astar_path=None):
    if obs is None:
        obs = env.controller._obs()
    sem_map = obs["sem_map"]
    rgb = obs["raw"]["rgb"]
    """
    output an image of the semantic map
    red: agent
    white: unexplored
    grey: explored and unoccupied
    black: occupied
    """
    explored = sem_map["explored"]
    height = sem_map["height"]
    ID = sem_map["id"]
    w, h = explored.shape
    
    img = np.zeros((w, h, 3), dtype=np.uint8)
    obj = np.zeros((w, h, 3), dtype=np.uint8)
    obj[ID > 0.5] = [0, 0, 0]
    obj[ID < 0.5] = [255, 255, 255]
    height_img = np.copy(height)
    height_img.astype(np.uint8)
    height_img = np.expand_dims(height_img, axis=2)
    height_img = np.repeat(height_img, 3, axis=2)
    height_img *= int(255 / height.max())
    height_img[height <= 0] = [0, 0, 0]
    img[explored == 0] = [255, 255, 255] # this color is white
    img[explored == 1] = [200, 200, 200] # this color is grey
    img = img - np.reshape(200 * np.minimum(height / 2, 1.0) * (height > 0.3), (w, h, 1))
    
    pos = env.controller.agents[0].dynamic.transform.position
    offset = env.controller.sem_map.map_offset
    grid_size = env.controller.sem_map.grid_size
    pos = [int(pos[0] // grid_size + offset[0]), int(pos[2] // grid_size + offset[1])]
    # set nearby pixels to red
    for i in range(-1, 2):
        for j in range(-1, 2):
            img[pos[0] + i, pos[1] + j] = np.array([255, 0, 0])
    # mark path
    if astar_path is not None:
        for p in astar_path:
            img[p[0], p[1]] = np.array([0, 255, 0])
    
    print(height.max())
    # save the image
    print("rgb", rgb.shape)
    import cv2
    # rgb to bgr
    rgb = rgb[[2, 1, 0], :, :]
    img = img[:, :, [2, 1, 0]]
    obj = obj[:, :, [2, 1, 0]]
    height_img = height_img[:, :, [2, 1, 0]]
    explored = np.copy(explored)
    explored *= 255
    explored.astype(np.uint8)
    explored = np.expand_dims(explored, axis=2)
    explored = np.repeat(explored, 3, axis=2)
    explored = explored[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(save_dir, f"sem_map_{suffix}.png"), img)
    cv2.imwrite(os.path.join(save_dir, f"rgb_{suffix}.png"), rgb.transpose(1, 2, 0) * 255)
    cv2.imwrite(os.path.join(save_dir, f"obj_map_{suffix}.png"), obj)
    cv2.imwrite(os.path.join(save_dir, f"height_map_{suffix}.png"), height_img)
    cv2.imwrite(os.path.join(save_dir, f"explore_{suffix}.png"), explored)

"""this action is environment independent"""
# def agent_walk_to(env: WindEnv, target: Union[int, np.ndarray, List], max_steps=100, reset_arms: bool = False, arrived_at=1.0):
def agent_walk_to(env, target: Union[int, np.ndarray, List], max_steps=100, reset_arms: bool = False, arrived_at=1.0,
                task=None, effect_on_agents=False, record_mode=False):
    # visualize_obs(env, None, suffix="0")
    start_frame = env.controller.frame_count
    if record_mode and task != "wind":
        env.controller.agents[0].collision_detection.avoid = False
        env.controller.agents[0].collision_detection.objects = False

    while True:
        agent_pos = env.controller.agents[0].dynamic.transform.position
        target_pos = env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target)
        if np.linalg.norm(agent_pos[[0, 2]] - target_pos[[0, 2]]) < arrived_at:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return True, "success"
        
        if env.controller.frame_count - start_frame > max_steps:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "max steps reached"
        
        agent_pos = env.controller.real_to_grid(agent_pos)
        target_pos = env.controller.real_to_grid(target_pos)
        obs = env.controller._obs()
        sem_map = obs["sem_map"]
        if isinstance(target, int) and not np.any(sem_map["id"] == env.controller.manager.id_renumbering[target]):
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "target not in vision or memory"
        weight = get_astar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
        path = get_astar_path(weight=weight, origin=agent_pos, destination=target_pos)
        # visualize_obs(env, obs, suffix=str(env.controller.frame_count), astar_path=path)
        # walk to first point
        if path is None or len(path) <= 1:
            env.controller.agents[0].collision_detection.avoid = True
            env.controller.agents[0].collision_detection.objects = True
            return False, "don't know, maybe out of bounds"
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            env.controller.next_key_frame(force_direction=None)
        elif task == "wind":
            wind_v = env.controller.manager.wind_v
            env.controller.next_key_frame(force_direction=wind_v)
        else:
            assert env.controller.manager.flood_manager.source_from == 'x_max'
            env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))


def low_level_action(env, action, effect_on_agents=False, task=None, **kwargs):
    assert action in ['move_by', 'turn_by', 'turn_to', 'reach_for']
    forbidden_params = ['max_distance', 'duration', 'scale_duration', 'arrived_at']
    for param in forbidden_params:
        assert param not in kwargs
    if 'target' in kwargs and type(kwargs['target']) == int:
        kwargs['target'] = env.controller.manager.id_renumbering[kwargs['target']]
    getattr(env.controller.agents[0], action)(**kwargs)
    if action != "move_by":
        effect_on_agents = False
    if not effect_on_agents or task == "fire":
        env.controller.next_key_frame()
    elif task == "wind":
        wind_v = env.controller.manager.wind_v
        env.controller.next_key_frame(force_direction=wind_v)
    else:
        assert env.controller.manager.flood_manager.source_from == 'x_max'
        env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
    return True, "success"


def agent_walk_to_single_step(env, target: Union[int, np.ndarray, List], reset_arms: bool = False, arrived_at=1.0,
                              effect_on_agents=False, task=None, record_mode=False):
    agent_pos = env.controller.agents[0].dynamic.transform.position
    target_pos = env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target)
    # print(target, env.controller.manager.objects[target].position, type(target))
    if np.linalg.norm(agent_pos - target_pos) < arrived_at:
        return True, "success"
    agent_pos = env.controller.real_to_grid(agent_pos)
    target_pos = env.controller.real_to_grid(target_pos)
    obs = env.controller._obs()
    sem_map = obs["sem_map"]
    if isinstance(target, int) and not np.any(sem_map["id"] == env.controller.manager.get_renumbered_id(target)):
        return False, "target not in vision or memory"
    
    weight = get_astar_weight(sem_map=sem_map, origin=agent_pos, destination=target_pos)
    path = get_astar_path(weight=weight, origin=agent_pos, destination=target_pos)
    # walk to first point
    if path is None or len(path) <= 1:
        return False, "don't know, maybe out of bounds"
    env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                    "reset_arms": reset_arms,
                                                                    "arrived_at": 0.05 if record_mode else 0.5})

    if not effect_on_agents or task == "fire":
        env.controller.next_key_frame()
    elif task == "wind":
        wind_v = env.controller.manager.wind_v
        env.controller.next_key_frame(force_direction=wind_v)
    else:
        assert env.controller.manager.flood_manager.source_from == 'x_max'
        env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
    return True, "ongoing"

"""this action is environment independent"""
def agent_pickup(env, target: int, env_type: str = "what?"):
    if target not in env.controller.target_ids:
        return False, f"can only pick up target objects"

    arm = [Arm.left, Arm.right] if env_type == "wind" else Arm.right
    
    above_pos = env.controller.manager.objects[target].top()
    above_pos[1] += 100.0
    target_pos = env.controller.manager.objects[target].center()
    
    env.controller.do_action(agent_idx=0, action="turn_to", params={"target": target})
    status = env.controller.next_key_frame()[0][0]
    if status != ActionStatus.success:
        print("failed to turn to target")
        input()

    env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(target_pos),
                                                                        "arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right,
                                                                        "arrived_at": 0.02,
                                                                        "max_distance": 2.5,
                                                                        "plan": IkPlanType.vertical_horizontal})
    status = env.controller.next_key_frame()[0][0]
    reach_success = (status == ActionStatus.success)

    # if not reach_success:
    #     env.controller.do_action(agent_idx=0, action="reset_arm",
    #                              params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    #     env.controller.next_key_frame()
    #     env.controller.do_action(agent_idx=0, action="turn_by", params={"angle": random.randint(30, 330)})
    #     env.controller.next_key_frame()
    #     return False, f"cannot reach for this object, maybe too far"

    env.controller.communicate([{"$type": "set_kinematic_state", "id": target, "is_kinematic": False, "use_gravity": False}])
    if target in env.controller.other_containers:
        env.controller.communicate(
            {"$type": "teleport_object", "id": target, "position": TDWUtils.array_to_vector3(above_pos)})

    env.controller.do_action(agent_idx=0, action="grasp", params={"target": target, 
                                                                    "arm": Arm.right,
                                                                    "angle": None,
                                                                    "axis": None,})
    status = env.controller.next_key_frame()[0][0]
    
    env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    env.controller.next_key_frame()
    
    if status == ActionStatus.success:
        return True, f"success, reach for {'success' if reach_success else 'failed, may cause problems'}"
    elif status == ActionStatus.cannot_grasp:
        return False, "cannot grasp, maybe too far"
    else:
        return False, "failed to grasp"

"""turn around to find new targets"""
def agent_explore(env):
    TURN_TIMES = 12
    for i in range(TURN_TIMES):
        env.controller.do_action(agent_idx=0, action="turn_by", params={"angle": 360.0 / TURN_TIMES})
        status = env.controller.next_key_frame()[0][0]
        env.controller._obs()
        if status != ActionStatus.success:
            return False, 'can not turn around at this time'
    return True, 'success'

def agent_drop(env, container: Optional[int]=None, env_type: str = "what?"):
    arm = Arm.right
    try:
        grasp_id = env.controller.agents[0].dynamic.held_objects[arm]
    except:
        return False, "not holding an object"

    if hasattr(env.controller, "container_id"):
        assert Arm.left in env.controller.agents[0].dynamic.held_objects and env.controller.agents[0].dynamic.held_objects[Arm.left] == env.controller.container_id
        container = env.controller.container_id

    if (env_type == "flood" or env_type == "fire"):
        top = env.controller.manager.objects[container].top()
        top = top + np.array([0, 0.2, 0])

        env.controller.do_action(agent_idx=0, action="turn_to", params={"target": TDWUtils.array_to_vector3(top)})
        status = env.controller.next_key_frame()[0][0]
        if status != ActionStatus.success:
            print("failed to turn to target")
            input()

        top = env.controller.manager.objects[container].top()
        above = top + np.array([0, 0.2, 0])

        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(above),
                                                                        "absolute": True,
                                                                        "offhand_follows": False,
                                                                        "arm": Arm.left,
                                                                        "from_held": True,
                                                                        "held_point": "top",
                                                                        "max_distance": 1.0,
                                                                        "arrived_at": 0.05,
                                                                        "plan": IkPlanType.vertical_horizontal})
        status = env.controller.next_key_frame()[0][0]
        env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(top),
                                                                        "absolute": True,
                                                                        "offhand_follows": False,
                                                                        "arm": Arm.right,
                                                                        "from_held": True,
                                                                        "held_point": "top",
                                                                        "max_distance": 1.0,
                                                                        "arrived_at": 0.05,
                                                                        "plan": IkPlanType.vertical_horizontal})
        status = env.controller.next_key_frame()[0][0]
        
        env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
        status = env.controller.next_key_frame()[0][0]
        # env.controller.communicate([{"$type": "destroy_object", "id": grasp_id}])
        # instead of destroying it, teleport it to [100, 0, 0] and make it kinematic
        destroy_commands = []
        destroy_commands.append({"$type": "teleport_object", "id": grasp_id, "position": {"x": 100, "y": 20, "z": 0}})
        destroy_commands.append({"$type": "set_kinematic_state", "id": grasp_id, "is_kinematic": True, "use_gravity": False})
        if hasattr(env.controller, "finished") and grasp_id not in env.controller.finished:
            env.controller.finished.append(grasp_id)
        if env_type == "fire":
            env.controller.manager.objects[grasp_id].temperature_threshold = 4000
        env.controller.communicate(destroy_commands)
        # env.controller.do_action(agent_idx=0, action="reset_arm",
        #                          params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right]})
        env.controller.next_key_frame()
        return True, "successfully drop and put away"

    if container is None:
        env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
        status = env.controller.next_key_frame()[0][0]
        env.controller.do_action(agent_idx=0, action="reset_arm",
                                 params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.next_key_frame()
        if status == ActionStatus.success:
            return True, "drop success"
        elif status == ActionStatus.still_dropping:
            return True, "drop success but still dropping"
        else:
            return False, "I don't know what happened but the drop failed"

    env.controller.do_action(agent_idx=0, action="turn_to", params={"target": container})
    env.controller.next_key_frame() # this action cannot fail (seems like)
    
    top = env.controller.manager.objects[container].top()
    top = top + np.array([0, 0.2, 0])
    env.controller.do_action(agent_idx=0, action="reach_for", params={"target": TDWUtils.array_to_vector3(top),
                                                                    "absolute": True,
                                                                    "offhand_follows": False,
                                                                    "arm": Arm.right,
                                                                    "from_held": True,
                                                                    "held_point": "top",
                                                                    "max_distance": 2.5,
                                                                    "arrived_at": 0.05,
                                                                    "plan": IkPlanType.vertical_horizontal})
    status = env.controller.next_key_frame()[0][0]
    reach_success = True
    if status != ActionStatus.success:
        env.controller.do_action(agent_idx=0, action="reset_arm",
                                 params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
        env.controller.next_key_frame()
        reach_success = False
        if env_type == "wind": return False, "failed to reach"
    # if status != ActionStatus.success:
    #     return False, "failed to reach"
    
    if env_type == "wind":
        env.controller.manager.settled.add(grasp_id)
    
    env.controller.do_action(agent_idx=0, action="drop", params={"arm": arm, "max_num_frames": 20})
    status = env.controller.next_key_frame()[0][0]
    env.controller.do_action(agent_idx=0, action="reset_arm", params={"arm": [Arm.left, Arm.right] if env_type == "wind" else Arm.right})
    env.controller.next_key_frame()

    if status == ActionStatus.success:
            return True, f"drop success, reach for {'success' if reach_success else 'failed, may cause problems'}"
    elif status == ActionStatus.still_dropping:
        return True, f"drop success but still dropping, reach for {'success' if reach_success else 'failed, may cause problems'}"
    else:
        return False, "I don't not what happened but the drop failed"