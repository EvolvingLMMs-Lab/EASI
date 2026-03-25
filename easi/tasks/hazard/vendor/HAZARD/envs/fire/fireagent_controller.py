from typing import Any, Tuple, Dict
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.librarian import HumanoidLibrarian
from tdw.add_ons.logger import Logger
from tdw.replicant.arm import Arm
from tdw.replicant.image_frequency import ImageFrequency
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.obi_data.fluids.cube_emitter import CubeEmitter
from tdw.add_ons.obi import Obi

import numpy as np
import cv2
import copy
import os

from .object import ObjectStatus
from .agent import *
from .fire import FireController
from HAZARD.utils.model import Semantic_Mapping
from HAZARD.utils.vision import Detector
from HAZARD.utils.local_asset import get_local_url
from HAZARD.utils.scene_setup import SceneSetup

PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)

import torch

"""
Caution!!! In a single trial, the number of agents should never change!
Currently single agent only.
"""

class FireAgentController(FireController):
    """
    Never ever use self.commands in here! For safety uses, self.commands can only be used in parent class
    """
    def __init__(self, use_local_resources: bool = False, image_capture_path: str = None, log_path: str = None,
                 reverse_observation: bool = False, record_only: bool = False, use_dino: bool = False,
                 **kwargs) -> None:
        self.frame_count = 0
        self.use_local_resources = use_local_resources
        self.image_capture_path = image_capture_path
        self.log_path = log_path
        self.reverse_observation = reverse_observation
        super().__init__(**kwargs)
        self.screen_size = kwargs.get("screen_size", 512)
        self.agents: List[FireAgent] = []
        self.extinguishers = []
        self.comm_counter = 0
        self.use_gt = kwargs.get("use_gt", True)
        self.use_dino = use_dino
        self.record_only = record_only
        if not self.use_gt:
            if self.use_dino:
                from HAZARD.utils.vision_dino import DetectorSAM
                self.detector = DetectorSAM(**kwargs)
            else:
                self.detector = Detector(**kwargs)
        # self.init_seg(vocab_path=f"{os.getcwd()}/seg/vocab.txt")
        if use_local_resources:
            self.update_replicant_url()
        # self.detector = Detector(**kwargs)

        self.map_size_h = kwargs.get("map_size_h", 64)
        self.map_size_v = kwargs.get("map_size_v", 64)
        self.grid_size = kwargs.get("grid_size", 0.25)
        self.sem_map = Semantic_Mapping(device=None, screen_size=self.screen_size,
                                        map_size_h=self.map_size_h, map_size_v=self.map_size_v, grid_size=self.grid_size)

        self.maps = []
        self.id2name = {}
        self.other_containers = {}
        self.containers = {}
        # self.init_seg()

    def init_obi(self):
        self.obi = Obi()
        self.communicate([{"$type": "create_obi_solver"}])
        self.obi.set_solver(solver_id=0, scale_factor=1.0, substeps=1)
        self.add_ons.append(self.obi)
        self.communicate([])
        self.communicate([])

    def update_replicant_url(self):
        if not os.path.isfile(f"{os.getcwd()}/data/assets/replicant_0"): # There is no local model of replicant
            print("There is no local model of replicant. ")
            return
        LOCAL_PATH_PREFIX = f"file://{os.getcwd()}/data/assets"
        Controller.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME] = HumanoidLibrarian(Replicant.LIBRARY_NAME)
        import platform
        for repli in ["replicant_0", "fireman"]:
            record = Controller.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME].get_record(repli)
            new_url = record.urls[platform.system()].split("/")[-1]
            record.urls[platform.system()] = f"{LOCAL_PATH_PREFIX}/{new_url}"
            Controller.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME].add_or_update_record(record, overwrite=True)

    def grid_to_real(self, grid_pos):
        return self.sem_map.grid_to_real(grid_pos)

    def real_to_grid(self, real_pos):
        return self.sem_map.real_to_grid(real_pos)

    def reset_scene(self):
        # whatever the agent is holding, drop it
        for agent in self.agents:
            arms = agent.dynamic.held_objects.keys()
            for arm in arms:
                agent.drop(arm=arm)
                while agent.action.status == ActionStatus.ongoing:
                    self.communicate([])
        commands = []
        for idx in self.fire_info:
            commands.append({"$type": "destroy_visual_effect", "id": self.fire_info[idx].fire_id})
        self.communicate(commands)
        
        self.fire_info = dict()
        self.fire_candidate = dict()
        self.initialized = False
        self.maps = []
        self.add_ons = []
        self.manager.reset()
        self.extinguishers = []
        self.id2name = {}
        self.frame_count = 0
        self.communicate([])
        
        self.last_reward = None
        self.communicate([{"$type": "destroy_all_objects"}])

    def init_scene(self, setup: SceneSetup):
        print("[init_scene] Starting reset_scene", flush=True)
        self.reset_scene()
        print("[init_scene] reset_scene done", flush=True)

        if self.log_path is not None:
            logger = Logger(self.log_path)
            self.add_ons.append(logger)
            self.communicate([])

        # for obj in setup.objects:
        #     self.manager.add_object(obj)
        
        import json
        temperature_record = dict()
        with open(os.path.join(PATH, "data", "meta_data", "temperature.json"), "r") as f:
            L = json.load(f)
            for data in L:
                temperature_record[data["name"]] = data["temp"]
        self.extinguishers = []
        self.containers = setup.containers
        self.other_containers = {}
        print(f"[init_scene] Processing {len(setup.commands_list)} command batches", flush=True)
        for batch_idx, commands in enumerate(setup.commands_list):
            filtered_commands = []
            tp = None
            for command in commands:
                tp = command["$type"]
                if tp == "terminate":
                    break
                if tp[:4] == "send" or tp.find("avatar") != -1 or "avatar_id" in command:
                    continue
                if "url" in command:
                    command["url"] = get_local_url(command["url"])
                if tp == "add_object":
                    name = command["name"]
                    idx = command["id"]
                    self.id2name[idx] = name
                    pos = TDWUtils.vector3_to_array(command["position"])
                    if name in temperature_record:
                        self.manager.add_object(ObjectStatus(idx=idx, temperature_threshold=temperature_record[name], position=pos))
                    else:
                        self.manager.add_object(ObjectStatus(idx=idx, position=pos))
                    if name == "b05_fire_extinguisher":
                        self.extinguishers.append(idx)
                if tp.startswith("add_") and tp.endswith("_container"):
                    self.other_containers[command['id']] = command['container_id']
                filtered_commands.append(command)
            print(f"[init_scene] Sending batch {batch_idx} ({len(filtered_commands)} cmds)", flush=True)
            self.communicate(filtered_commands)
            if tp == "terminate":
                break
        print(f"[init_scene] All command batches sent, adding {len(setup.other.get('fire', []))} fire positions", flush=True)
        for fire_pos in setup.other["fire"]:
            self.add_fire_floor(fire_pos)

        print("[init_scene] Fire positions added, creating agents", flush=True)
        if not self.record_only:
            if len(self.agents) == 0:
                self.agents: List[FireAgent] = []
                for agent_pos in setup.agent_positions:
                    idx = self.get_unique_id()
                    self.agents.append(
                        FireAgent(replicant_id=idx, position=agent_pos, image_frequency=ImageFrequency.always,
                                  name="fireman"))
                    self.add_agent(idx, agent_pos)
                    self.add_ons.append(self.agents[-1])
            else:
                assert (len(self.agents) == len(setup.agent_positions))
                for i in range(len(self.agents)):
                    idx = self.agents[i].replicant_id
                    self.agents[i].reset(position=setup.agent_positions[i])
                    self.add_ons.append(self.agents[i])
                    self.add_agent(idx, setup.agent_positions[i])

        print(f"[init_scene] Agents ready ({len(self.agents)} agents)", flush=True)
        self.maps = [None] * len(self.agents)
        self.add_ons.append(self.manager)
        self.target = setup.targets
        if self.use_dino and not self.use_gt:
            self.detector.set_targets(self.target)
        self.target_ids = setup.target_ids
        self.target_names = setup.target_names
        self.target_id2category = setup.target_id2category
        self.target_id2name = setup.target_id2name
        self.finished = []
        self.communicate([])
        if self.image_capture_path != None:
            if not self.record_only:
                camera = ThirdPersonCamera(avatar_id="record",
                                           position={"x": 0.0, "y": 10.0, "z": 0.0},
                                           look_at={"x": 0.0, "y": 0.0, "z": 0.0}) # or self.agents[0].replicant_id
                commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                            {"$type": "set_target_framerate", "framerate": 30}]
            else:
                camera = ThirdPersonCamera(avatar_id="record", position={"x": 0.0, "y": 6.0, "z": 0.0},
                                           look_at=self.agents[0].replicant_id)
                commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                            {"$type": "set_target_framerate", "framerate": 30}]
            self.add_ons.extend([camera])
        else:
            commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                        {"$type": "set_target_framerate", "framerate": 30}]

        if self.image_capture_path is not None:
            commands.extend([{"$type": "set_pass_masks", "pass_masks": ["_img"], "avatar_id": "record"},
                             {"$type": "send_images", "frequency": "always", "ids": ["record"]}])
            commands.extend([{"$type": "set_floorplan_roof", "show": False}])

        # Capture when running after init_scene. (Because screen size may be modified)
        print("[init_scene] Requesting scene_regions", flush=True)
        resp = self.communicate([{"$type": "send_scene_regions"}])
        self.set_scene_bounds(resp)

        print("[init_scene] Setting up segmentation + kinematic state", flush=True)
        self.manager.prepare_segmentation_data()
        self.initialized = True
        for obj in self.target_ids:
            commands.append({"$type": "set_kinematic_state", "id": obj, "is_kinematic": False, "use_gravity": True})
        self.communicate(commands)

        if len(self.extinguishers) > 0:
            self.init_obi()

        print("[init_scene] Setting up agent equipment (backpack/FoV)", flush=True)
        if not self.record_only and len(self.extinguishers) == 0:
            commands = [
                {"$type": "set_field_of_view", "field_of_view": 120.0, "avatar_id": str(self.agents[0].replicant_id)}]
            """add a backpack or similar to the agent's left hand"""
            self.container_name = "backpack" if "container" not in setup.other else setup.other["container"]
            self.container_id = self.get_unique_id()
            self.manager.add_object(ObjectStatus(idx=self.container_id, inflammable=False, temperature_threshold=3000))

            agent_pos = self.agents[0].dynamic.transform.position
            commands.append(self.get_add_object(model_name=self.container_name, object_id=self.container_id,
                                                position=TDWUtils.array_to_vector3(agent_pos)))
            self.communicate(commands)
            print("[init_scene] Agent grasping backpack", flush=True)
            self.agents[0].grasp(target=self.container_id, arm=Arm.left, axis=None, angle=None)
            print("[init_scene] grasp() issued, waiting for key frame...", flush=True)
            self.next_key_frame()
            print("[init_scene] Grasp done. Agent reaching position", flush=True)
            self.agents[0].reach_for(target=TDWUtils.array_to_vector3([-0.3, 1.0, 0.3]), absolute=False, arrived_at=0.1,
                                     arm=Arm.left)
            print("[init_scene] reach_for() issued, waiting for key frame...", flush=True)
            self.next_key_frame()
            print("[init_scene] Reach done. Agent resetting arm", flush=True)
            self.agents[0].reset_arm(arm=Arm.left)
            print("[init_scene] reset_arm() issued, waiting for key frame...", flush=True)
            self.next_key_frame()
            print("[init_scene] init_scene complete", flush=True)

    def set_scene_bounds(self, resp=None):
        self.scene_bounds = SceneBounds(resp=resp)
        self.x_max = max([region.x_max for region in self.scene_bounds.regions])
        self.x_min = min([region.x_min for region in self.scene_bounds.regions])
        self.z_max = max([region.z_max for region in self.scene_bounds.regions])
        self.z_min = min([region.z_min for region in self.scene_bounds.regions])
    
    def next_key_frame(self, force_direction=None) -> Tuple[List[ActionStatus], List[int]]:
        # print("next_key_frame")
        initial_status = []
        have_ongoing_action = False
        for agent in self.agents:
            if agent.action is not None:
                initial_status.append(copy.deepcopy(agent.action.status))
                if initial_status[-1] == ActionStatus.ongoing:
                    have_ongoing_action = True
            else:
                initial_status.append(None)
        if not have_ongoing_action:
            self.communicate([])
            print("no ongoing action")
            return initial_status, []
        num_step = 0
        while True:
            num_step += 1
            if num_step % 100 == 0:
                print(f"[next_key_frame] step {num_step}, status={[a.action.status if a.action else None for a in self.agents]}", flush=True)
            if num_step > 10000:
                raise RuntimeError("num_step > 10000")
            self.communicate([])

            changed = False
            for i, agent in enumerate(self.agents):
                if initial_status[i] is not None and agent.action.status != initial_status[i]:
                    changed = True
                    break
            if changed:
                break
        
        final_status = []
        changed_agents = []
        for i, agent in enumerate(self.agents):
            final_status.append(copy.deepcopy(agent.action.status) if agent.action is not None else None)
            if initial_status[i] != final_status[i]:
                changed_agents.append(i)

        self.communicate([])
        return final_status, changed_agents
    
    def do_action(self, agent_idx: int, action: str, params: Dict[str, Any] = dict()) -> None:
        """
        See the FireAgent class for allowed actions and their parameters.
        """
        if action == "stop":
            return
        return getattr(self.agents[agent_idx], action)(**params)

    def add_extinguish_water(self, fire_pos):
        object_id = self.get_unique_id()
        rotate = self.agents[0].dynamic.transform.forward
        rotate = rotate / np.linalg.norm(rotate)
        print(rotate)
        extinguisher_distance_to_agent = 1.0
        position = {"x": float(self.agents[0].dynamic.transform.position[0]+rotate[0]*extinguisher_distance_to_agent),
                    "y": float(self.agents[0].dynamic.transform.position[1]) + 0.7,
                    "z": float(self.agents[0].dynamic.transform.position[2]+rotate[2]*extinguisher_distance_to_agent)}
        rotation = {"x": 0,
                    "y": 90,
                    "z": 0}
        self.obi.create_fluid(fluid="water",
                              shape=CubeEmitter(),
                              object_id=object_id,
                              position=position,
                              rotation=rotation,
                              speed=0.9,
                              lifespan=0.5)
    
    def do_extinguish(self, target: np.ndarray):
        for idx, fire in self.fire_info.items():
            if fire.extinguishing:
                continue
            fire_id = fire.fire_id
            if np.linalg.norm(self.manager.objects[fire_id].position - target) < self.constants.EXTINGUISH_RADIUS:
                if idx == fire_id:
                    self.extinguish_fire_floor(idx)
                else:
                    self.extinguish_fire_object(idx)
    
    def get_agent_status(self, idx) -> FireAgent:
        return self.agents[idx]
    
    def get_temperature_observation(self, idx, width=512, height=512):
        depth = TDWUtils.get_depth_values(self.agents[idx].dynamic.images["depth"], width=width, height=height)
        depth = np.flip(depth, axis=0)
        camera_matrix = self.agents[idx].dynamic.camera_matrix

        point_cloud = TDWUtils.get_point_cloud(depth=depth, camera_matrix=camera_matrix, vfov=120.0)
        # fout = open("point_cloud.txt", "w")
        # bs, h, w = 1, height, width
        # for i in range(bs):
        #     for j in range(h):
        #         for k in range(w):
        #             for t in range(3):
        #                 print(point_cloud[t, j, k].item(), end=' ', file=fout)
        #             print('', file=fout)
        # shape: (3, 512, 512)
        # down sample to 16x16
        point_cloud = point_cloud[:, ::(width//16), ::(width//16)]
        temp = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                temperature = self.manager.query_point_temperature(point_cloud[:, i, j])
                temp[i, j] = temperature
        temp = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        return temp
    
    def find_nearest_object(self, agent_idx: int = 0, objects: List[int] = None):
        current_position = self.agents[agent_idx].dynamic.transform.position
        return self.manager.find_nearest_object(pos=current_position, objects=objects)

    def replace_with_local_path(self, commands):
        LOCAL_PATH_PREFIX = f"file://{os.getcwd()}/data/assets"
        download_cmds = []
        for command in commands:
            if 'url' in command and "amazonaws.com" in command['url']:
                new_url = command['url'].split("/")[-1]
                if not os.path.isfile(f"{os.getcwd()}/data/assets/{new_url}"):
                    download_cmds.append(f"wget {command['url']}\n")
                new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
                command['url'] = new_url
        for command in self.commands:
            if 'url' in command and "amazonaws.com" in command['url']:
                new_url = command['url'].split("/")[-1]
                if not os.path.isfile(f"{os.getcwd()}/data/assets/{new_url}"):
                    download_cmds.append(f"wget {command['url']}\n")
                new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
                command['url'] = new_url
        for m in self.add_ons:
            if not m.initialized:
                add_on_commands = m.get_initialization_commands()
            else:
                add_on_commands = m.commands
            for command in add_on_commands:
                if 'url' in command and "amazonaws.com" in command['url']:
                    new_url = command['url'].split("/")[-1]
                    if not os.path.isfile(f"{os.getcwd()}/data/assets/{new_url}"):
                        download_cmds.append(f"wget {command['url']}\n")
                    new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
                    command['url'] = new_url
        if len(download_cmds) > 0:
            fout = open(f"{os.getcwd()}/data/assets/download_assets.sh", "w")
            for cmd in download_cmds:
                fout.write(cmd)
            super().communicate({"$type": "terminate"})
            self.socket.close()
            print("Please run data/assets/download_assets.sh first!")
            exit(0)
        return commands

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        if self.initialized:
            for agent in self.agents:
                idx = agent.grasp_id()
                if idx is None:
                    continue
                dist = np.linalg.norm((agent.dynamic.transform.position - self.manager.objects[idx].position) * np.array([1, 0, 1]))
                if dist > 3.0:
                    agent.fail_grasp()
        if self.use_local_resources:
            commands = self.replace_with_local_path(commands)
            # self.commands = self.replace_with_local_path(self.commands)
        resp = super().communicate(commands)
        if self.initialized:
            # for i, agent in enumerate(self.agents):
            #     # if too hot, agent dies
            #     if self.manager.objects[agent.replicant_id].temperature > \
            #         self.manager.objects[agent.replicant_id].temperature_threshold:
            #         raise RuntimeError(f"agent {i} is on fire")
            #     # if agent is holding an extinguisher: put out fire
            #     held_objects = agent.extinguish_indicator()
            #     if held_objects is not None:
            #         print("holding: ", held_objects)
            #     if held_objects is not None and len(held_objects) == 1 and held_objects[0] in self.extinguishers:
            #         self.do_extinguish(agent.extinguish_target())
            #         print("extinguish")
            #         pass
            for target in self.target_ids:
                if (np.abs(self.manager.objects[target].position).sum() > 50) and (target not in self.finished):
                    # this object is dropped and moved to a distant place
                    self.finished.append(target)
        if self.image_capture_path != None:
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                # Get Images output data.
                if r_id == "imag":
                    images = Images(resp[i])
                    # Determine which avatar captured the image.
                    if images.get_avatar_id() == "record":
                        # Save the image.
                        # if self.comm_counter % 10 == 0:
                        TDWUtils.save_images(images=images, filename=str(self.comm_counter),
                                             output_directory=self.image_capture_path)
                        self.comm_counter += 1
        # The ImageCapture addon already saves the image. No need to save again. 
        return resp

    def get_seg_mask(self, rgb):
        # return self.detector.get_seg_mask(rgb)
        pass
    
    
    @torch.no_grad()
    def _obs(self, agent_idx: int = 0):
        # print("get obs")

        self.manager.prepare_segmentation_data()
        self.communicate([])

        obs = dict()
        """
        raw observation: RGBD, temperature
        """
        rgb = self.agents[agent_idx].dynamic.get_pil_image()
        id_image = np.array(self.agents[agent_idx].dynamic.get_pil_image("id"))
        if self.reverse_observation:
            rgb = np.flip(rgb, axis=0)
            id_image = np.flip(id_image, axis=0)

        if self.use_gt:
            seg_mask = self.manager.segm.get_seg_mask(id_image)
        else:
            rcnn_mask = self.detector.inference(np.array(rgb))
            seg_mask = self.manager.segm.get_seg_mask(np.array(id_image),
                                                        rcnn=rcnn_mask,
                                                        id_list=self.manager.id_list)
        depth = TDWUtils.get_depth_values(self.agents[agent_idx].dynamic.images["depth"], width=self.screen_size, height=self.screen_size)
        depth = np.flip(depth, axis=0)
        
        rgb = np.array(rgb).astype(np.float32).transpose((2, 0, 1)).astype(np.float32) * 1.0 / 255
        depth = depth.reshape((1, self.screen_size, self.screen_size)).astype(np.float32)
        
        log_temp = self.get_temperature_observation(agent_idx, width=self.screen_size, height=self.screen_size)
        log_temp = np.log(log_temp.reshape((1, self.screen_size, self.screen_size)).astype(np.float32))
        obs["raw"] = dict(
            rgb=rgb,
            depth=depth,
            log_temp=log_temp,
            seg_mask=seg_mask
        )
        
        """
        mapped observation
        """
        camera_matrix = self.agents[agent_idx].dynamic.camera_matrix.reshape((4, 4))
        obs_concat = np.concatenate([rgb, depth, log_temp], axis=0)
        
        sem = self.sem_map.forward(obs=obs_concat, id_map=seg_mask, camera_matrix=camera_matrix, maps_last=self.maps[agent_idx],
                                   position=self.agents[agent_idx].dynamic.transform.position, record_mode=self.record_only,
                                   targets=[self.manager.id_renumbering[target] for target in self.target_ids])
        obs["sem_map"] = dict(height=sem["height"].cpu().numpy(),
                              explored=sem["explored"].cpu().numpy(),
                              id=sem["id"].cpu().numpy(),
                              other=sem["other"].cpu().numpy() if sem["other"] is not None else None)
        self.maps[agent_idx] = dict(height=sem["height"].cpu().numpy(),
                              explored=sem["explored"].cpu().numpy(),
                              id=sem["id"].cpu().numpy(),
                              other=sem["other"].cpu().numpy() if sem["other"] is not None else None)

        """
        map of goal and agent
        """
        agent_pos = self.sem_map.real_to_grid(self.agents[agent_idx].dynamic.transform.position)
        target_poss = [self.sem_map.real_to_grid(self.manager.objects[idx].position) for idx in self.target_ids]
        
        goal_map = np.zeros((self.map_size_h, self.map_size_v))
        for (i, target_pos) in enumerate(target_poss):
            if self.target_ids[i] in self.finished:
                continue
            if target_pos[0] < 0 or target_pos[0] >= self.map_size_h or target_pos[1] < 0 or target_pos[1] >= self.map_size_v:
                continue
            if not (sem["id"] == self.manager.get_renumbered_id(self.target_ids[i])).any():
                continue
            goal_map[target_pos[0], target_pos[1]] = 1
        if agent_pos[0] > 0 and agent_pos[0] < self.map_size_h - 1 and agent_pos[1] > 0 and agent_pos[1] < self.map_size_v - 1:
            goal_map[agent_pos[0], agent_pos[1]] = -2
            rad = self.agents[agent_idx].get_facing()
            rad = int(rad / (np.pi / 4))
            if rad < 0:
                rad += 8
            dx = list([1, 1, 0, -1, -1, -1, 0, 1])[rad]
            dz = list([0, 1, 1, 1, 0, -1, -1, -1])[rad]
            goal_map[agent_pos[0] + dx][agent_pos[1] + dz] = -1
        obs["goal_map"] = goal_map

        """
        agent info
        """
        obs["camera_matrix"] = camera_matrix
        # return RL_obs
        RL_obs = np.zeros((5, self.map_size_h, self.map_size_v))
        RL_obs[0] = obs["sem_map"]["height"]
        RL_obs[1] = obs["sem_map"]["explored"]
        RL_obs[2] = obs["sem_map"]["id"]
        RL_obs[3] = goal_map
        RL_obs[4] = obs["sem_map"]["other"][0]
        obs["RL"] = RL_obs
        return obs
    
    def _info(self):
        info = dict()
        info['vector'] = np.zeros(8) # for consistency
        info['agent_positions'] = [agent.dynamic.transform.position for agent in self.agents]
        info['targets'] = self.manager.get_renumbered_list(self.target_ids)
        info['finished targets'] = self.manager.get_renumbered_list(self.finished)
        info['camera_matrices'] = [agent.dynamic.camera_matrix.reshape((4, 4)) for agent in self.agents]
        info['sr'] = f"{len(self.finished)}/{len(self.target_ids)}"
        return info
    
    # this is a toy reward function
    def _reward(self, agent_idx: int = 0):
        if self.last_reward is None:
            self.last_reward = 0
        
        reward = 0
        
        agent_pos = self.agents[agent_idx].dynamic.transform.position
        nearest_target = self.manager.find_nearest_object(agent_pos, self.target_ids)
        
        reward += len(self.finished) * 20
        if Arm.left in self.agents[0].dynamic.held_objects:
            reward -= 10
        else:
            dist = np.linalg.norm(agent_pos - self.manager.objects[nearest_target].position)
            reward -= dist * 10
        ret = reward - self.last_reward - 0.1
        self.last_reward = reward
        return ret
    
    def _done(self, agent_idx: int = 0):
        if len(self.finished) == len(self.target_ids):
            return True
        return False