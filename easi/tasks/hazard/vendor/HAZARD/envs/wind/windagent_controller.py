from typing import Any, Tuple, Dict
from .object import ObjectStatus
from .agent import *
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.logger import Logger
from tdw.replicant.arm import Arm
from tdw.replicant.image_frequency import ImageFrequency
from HAZARD.utils.model import Semantic_Mapping
from .wind import WindController
from tdw.tdw_utils import TDWUtils
from tdw.librarian import HumanoidLibrarian
import numpy as np
import copy
import os
from HAZARD.utils.vision import Detector
from HAZARD.utils.local_asset import get_local_url
from HAZARD.utils.scene_setup import SceneSetup
from tdw.output_data import OutputData, Images

PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)

import torch

"""
Caution!!! In a single trial, the number of agents should never change!
Currently single agent only.
"""
class WindAgentController(WindController):
    """
    Never ever use self.commands in here! For safety uses, self.commands can only be used in parent class
    """
    def __init__(self, use_local_resources: bool = False, image_capture_path: str = None, log_path: str = None,
                 reverse_observation: bool = False, record_only: bool = False, use_dino: bool = False,
                 **kwargs) -> None:
        self.use_local_resources = use_local_resources
        self.image_capture_path = image_capture_path
        self.log_path = log_path
        self.frame_count = 0
        self.reverse_observation = reverse_observation
        super().__init__(**kwargs)
        self.screen_size = kwargs.get("screen_size", 512)
        self.agents: List[WindAgent] = []
        self.comm_counter = 0
        self.use_gt = kwargs.get("use_gt", True)
        self.use_dino = use_dino
        self.record_only = record_only
        self.id2name = {}
        if not self.use_gt:
            if self.use_dino:
                from HAZARD.utils.vision_dino import DetectorSAM
                self.detector = DetectorSAM(**kwargs)
            else:
                self.detector = Detector(**kwargs)
        if use_local_resources:
            self.update_replicant_url()
        
        self.map_size_h = kwargs.get("map_size_h", 256)
        self.map_size_v = kwargs.get("map_size_v", 256)
        self.grid_size = kwargs.get("grid_size", 0.5)
        self.sem_map = Semantic_Mapping(device=None, screen_size=self.screen_size, 
                                        map_size_h=self.map_size_h, map_size_v=self.map_size_v, grid_size=self.grid_size)
        
        self.maps = []
        self.action_slowdown = 0
        self.other_containers = {}
        # self.init_seg()

    def update_replicant_url(self):
        assert os.path.isfile(f"{os.getcwd()}/data/assets/replicant_0")
        LOCAL_PATH_PREFIX = f"file://{os.getcwd()}/data/assets"
        Controller.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME] = HumanoidLibrarian(Replicant.LIBRARY_NAME)
        record = Controller.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME].get_record("replicant_0")
        import platform
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
        
        self.initialized = False
        self.add_ons = []
        self.manager.reset()
        self.frame_count = 0
        self.communicate([])
        self.id2name = {}
        
        self.last_reward = None
        self.communicate([{"$type": "destroy_all_objects"}])

    def init_scene(self, setup: SceneSetup):
        self.reset_scene()

        if self.log_path is not None:
            logger = Logger(self.log_path)
            self.add_ons.append(logger)
            self.communicate([])
        # for obj in setup.objects:
        #     self.manager.add_object(obj)
        if self.log_path is not None:
            logger = Logger(self.log_path)
            self.add_ons.append(logger)
            self.communicate([])
        for commands in setup.commands_list:
            filtered_commands = []
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
                    resistence = setup.other["wind_resistence"][str(idx)] if str(idx) in setup.other["wind_resistence"] else 0
                    self.manager.add_object(ObjectStatus(idx=idx, position=pos, resistence=resistence))
                filtered_commands.append(command)
            self.communicate(filtered_commands)

        if not self.record_only:
            if len(self.agents) == 0:
                self.agents: List[WindAgent] = []
                for agent_pos in setup.agent_positions:
                    idx = self.get_unique_id()
                    self.agents.append(
                        WindAgent(replicant_id=idx, position=agent_pos, image_frequency=ImageFrequency.always))
                    self.add_agent(idx, agent_pos)
                    self.add_ons.append(self.agents[-1])
            else:
                assert (len(self.agents) == len(setup.agent_positions))
                for i in range(len(self.agents)):
                    idx = self.agents[i].replicant_id
                    self.agents[i].reset(position=setup.agent_positions[i])
                    self.add_ons.append(self.agents[i])
                    self.add_agent(idx, setup.agent_positions[i])

        self.maps = [None] * len(self.agents)

        self.add_ons.append(self.manager)
        self.containers = setup.containers
        self.targets = setup.target_ids
        self.target = setup.targets
        if self.use_dino and not self.use_gt:
            self.detector.set_targets(self.target)
        self.target_ids = setup.target_ids
        self.target_names = setup.target_names
        self.target_id2category = setup.target_id2category
        self.target_id2name = setup.target_id2name
        self.finished = []
        
        self.set_wind(np.array(setup.other["wind"]))
        for idx in self.containers:
            self.manager.settled.add(idx)
        self.communicate([])

        if self.image_capture_path != None:
            pos = copy.deepcopy(setup.agent_positions[0])
            pos[1] = 8.0
            # theta = self.RNG.random() * 2 * np.pi
            # pos[0] += np.cos(theta) * 2
            # pos[2] += np.sin(theta) * 2
            pos[0] += 1.0
            pos[2] -= 2.0

            if self.record_only:
                look_at = copy.deepcopy(setup.agent_positions[0])
                # look_at[1] += 3.0
                look_at[0] -= 3.0
                look_at[2] -= 6.0
                look_at = TDWUtils.array_to_vector3(look_at)
                commands = [{"$type": "set_screen_size", "width": self.screen_size * 4, "height": self.screen_size * 4},
                            {"$type": "set_target_framerate", "framerate": 30}]
            else:
                look_at = self.agents[0].replicant_id
                commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                            {"$type": "set_target_framerate", "framerate": 30}]
            camera = ThirdPersonCamera(avatar_id="record", position=TDWUtils.array_to_vector3(pos),
                                       look_at=look_at)
            self.add_ons.extend([camera])
            self.communicate([])
        else:
            commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                        {"$type": "set_target_framerate", "framerate": 30}]

        if self.image_capture_path is not None:
            commands.extend([{"$type": "set_pass_masks", "pass_masks": ["_img"], "avatar_id": "record"},
                             {"$type": "send_images", "frequency": "always", "ids": ["record"]}])
        # if self.video_path is not None:
        #     self.capture = ImageCapture(path=self.video_path, avatar_ids=["a"])
        #     self.add_ons.append(self.capture)
        self.manager.prepare_segmentation_data()
        self.origin_pos = [setup.agent_positions[0][0], setup.agent_positions[0][2]]
        self.sem_map = Semantic_Mapping(device=None, screen_size=self.screen_size, map_size_h=self.map_size_h, map_size_v=self.map_size_v, grid_size=self.grid_size, origin_pos=self.origin_pos)
        self.initialized = True
        self.communicate(commands)
        if not self.record_only:
            self.communicate({"$type": "set_field_of_view", "field_of_view": 120.0, "avatar_id": str(self.agents[0].replicant_id)})
    
    def next_key_frame(self, force_direction=None) -> Tuple[List[ActionStatus], List[int]]:
        # print("next_key_frame")
        agent_idx = 0
        action_slowdown = 0
        if force_direction is not None:
            facing = self.agents[agent_idx].dynamic.transform.forward
            force_direction = force_direction / np.linalg.norm(force_direction)
            force_scale = (force_direction * (facing / np.linalg.norm(facing))).sum()
            action_slowdown += (1 - force_scale) * self.constants.F_ON_AGENT

        self.action_slowdown += action_slowdown
        if self.action_slowdown > 1:
            self.action_slowdown -= 1
            tmp = self.agents[agent_idx].action
            self.agents[agent_idx].action = None
            self.communicate([])
            self.agents[agent_idx].action = tmp

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

        # self.communicate([])
        return final_status, changed_agents
    
    def do_action(self, agent_idx: int, action: str, params: Dict[str, Any] = dict()) -> None:
        """
        See the WindAgent class for allowed actions and their parameters.
        """
        return getattr(self.agents[agent_idx], action)(**params)
    
    def get_agent_status(self, idx) -> WindAgent:
        return self.agents[idx]

    def find_nearest_object(self, agent_idx: int = 0, objects: List[int] = None):
        current_position = self.agents[agent_idx].dynamic.transform.position
        return self.manager.find_nearest_object(pos=current_position, objects=objects)
    
    def find_nearest_container(self, agent_idx: int = 0):
        current_position = self.agents[agent_idx].dynamic.transform.position
        return self.manager.find_nearest_object(pos=current_position, objects=self.containers)

    def replace_with_local_path(self, commands):
        LOCAL_PATH_PREFIX = f"file://{os.getcwd()}/data/assets"
        download_cmds = []
        for command in commands:
            if 'url' in command and "amazonaws.com" in command['url']:
                new_url = command['url'].split("/")[-1]
                if not os.path.isfile(f"{os.getcwd()}/data/assets/{new_url}"):
                    download_cmds.append(f"wget -nc {command['url']}\n")
                new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
                command['url'] = new_url
        for command in self.commands:
            if 'url' in command and "amazonaws.com" in command['url']:
                new_url = command['url'].split("/")[-1]
                if not os.path.isfile(f"{os.getcwd()}/data/assets/{new_url}"):
                    download_cmds.append(f"wget -nc {command['url']}\n")
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
                        download_cmds.append(f"wget -nc {command['url']}\n")
                    new_url = f"{LOCAL_PATH_PREFIX}/{new_url}"
                    command['url'] = new_url
        if len(download_cmds) > 0:
            fout = open(f"{os.getcwd()}/data/assets/download_assets.sh", "w")
            for cmd in download_cmds:
                fout.write(cmd)
            print("Please run data/assets/download_assets.sh first!")
            exit(0)
        return commands

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        """
        for each agent, if it is trying to grasp an object too far away, fail it
        """
        # print(commands)
        if self.use_local_resources:
            commands = self.replace_with_local_path(commands)
        if self.initialized:
            # check for finished objects
            self.frame_count += 1
            for idx in self.manager.settled:
                if idx in self.targets and idx not in self.finished:
                    self.finished.append(idx)
            # check agents
            for agent in self.agents:
                idx = agent.grasp_id()
                if idx is None:
                    continue
                dist = np.linalg.norm((agent.dynamic.transform.position - self.manager.objects[idx].position) * np.array([1, 0, 1]))
                if dist > 3:
                    agent.fail_grasp()
        resp = super().communicate(commands)
        if self.image_capture_path != None:
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                # Get Images output data.
                if r_id == "imag":
                    images = Images(resp[i])
                    # Determine which avatar captured the image.
                    if images.get_avatar_id() == "record":
                        # Save the image.
                        TDWUtils.save_images(images=images, filename=str(self.comm_counter),
                                             output_directory=self.image_capture_path)
                        self.comm_counter += 1
        return resp

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
        # print(seg_mask.shape, seg_mask.max())
        depth = TDWUtils.get_depth_values(self.agents[agent_idx].dynamic.images["depth"], width=self.screen_size, height=self.screen_size)
        depth = np.flip(depth, axis=0)
        
        rgb = np.array(rgb).astype(np.float32).transpose((2, 0, 1)).astype(np.float32) * 1.0 / 255
        depth = depth.reshape((1, self.screen_size, self.screen_size)).astype(np.float32)
        
        obs["raw"] = dict(
            rgb=rgb,
            depth=depth,
            seg_mask=seg_mask
        )
        
        """
        mapped observation
        """
        camera_matrix = self.agents[agent_idx].dynamic.camera_matrix.reshape((4, 4))
        obs_concat = np.concatenate([rgb, depth], axis=0)
        
        sem = self.sem_map.forward(obs=obs_concat, id_map=seg_mask, camera_matrix=camera_matrix, maps_last=self.maps[agent_idx],
                                   position=self.agents[agent_idx].dynamic.transform.position, record_mode=self.record_only,
                                   targets=self.manager.get_renumbered_list(self.targets))
        obs["sem_map"] = dict(height=sem["height"].cpu().numpy(),
                              explored=sem["explored"].cpu().numpy(),
                              id=sem["id"].cpu().numpy(),
                              other=sem["other"].cpu().numpy() if sem["other"] is not None else None)
        self.maps[agent_idx] = dict(height=sem["height"].cpu().numpy(),
                              explored=sem["explored"].cpu().numpy(),
                              id=sem["id"].cpu().numpy(),
                              other=sem["other"].cpu().numpy() if sem["other"] is not None else None)
        # end = time.time()
        # print("sem_map time=", end - start)
        
        """
        map of goal and agent
        """
        agent_pos = self.sem_map.real_to_grid(self.agents[agent_idx].dynamic.transform.position)
        target_poss = [self.sem_map.real_to_grid(self.manager.objects[idx].position) for idx in self.targets]
        
        goal_map = np.zeros((self.map_size_h, self.map_size_v))
        for (i, target_pos) in enumerate(target_poss):
            if self.targets[i] in self.finished:
                continue
            if target_pos[0] < 0 or target_pos[0] >= self.map_size_h or target_pos[1] < 0 or target_pos[1] >= self.map_size_v:
                continue
            if not (sem["id"] == self.manager.get_renumbered_id(self.targets[i])).any():
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
        
        RL_obs = np.zeros((4, self.map_size_h, self.map_size_v))
        RL_obs[0] = obs["sem_map"]["height"]
        RL_obs[1] = obs["sem_map"]["explored"]
        RL_obs[2] = obs["sem_map"]["id"]
        RL_obs[3] = goal_map
        obs["RL"] = RL_obs
        return obs
    
    def _info(self):
        info = dict()
        info['vector'] = np.zeros(8) # for consistency
        info['agent_positions'] = [agent.dynamic.transform.position for agent in self.agents]
        info['targets'] = self.manager.get_renumbered_list(self.targets)
        info['containers'] = self.manager.get_renumbered_list(self.containers)
        info['finished targets'] = self.manager.get_renumbered_list(self.finished)
        info['camera_matrices'] = [agent.dynamic.camera_matrix.reshape((4, 4)) for agent in self.agents]
        info['wind'] = self.manager.wind_v
        info['sr'] = f"{len(self.finished)}/{len(self.targets)}"
        return info
    
    # this is a toy reward function
    def _reward(self, agent_idx = 0):
        if self.last_reward is None:
            self.last_reward = 0
        
        reward = 0
        
        agent_pos = self.agents[agent_idx].dynamic.transform.position
        nearest_target = self.manager.find_nearest_object(agent_pos, self.targets)
        nearest_container = self.manager.find_nearest_object(agent_pos, self.containers)
        
        reward += len(self.finished) * 20
        if Arm.left in self.agents[0].dynamic.held_objects:
            dist = np.linalg.norm(agent_pos - self.manager.objects[nearest_container].position)
            reward -= dist * 10
        else:
            dist = np.linalg.norm(agent_pos - self.manager.objects[nearest_target].position)
            reward -= dist * 10
        ret = reward - self.last_reward - 0.1
        self.last_reward = reward
        return ret
    
    def _done(self, agent_idx: int = 0):
        if len(self.finished) == len(self.targets):
            return True
        return False