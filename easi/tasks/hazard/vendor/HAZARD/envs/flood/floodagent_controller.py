from typing import Any, Tuple, Dict
from .object import ObjectStatus
from .agent import *
from .flood import FloorFlood
from .flood import FloodController
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.logger import Logger
from tdw.replicant.arm import Arm
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.floorplan_flood import FloorplanFlood
from tdw.tdw_utils import TDWUtils
from tdw.librarian import HumanoidLibrarian
from tdw.output_data import OutputData, Images
from HAZARD.utils.vision import Detector
from HAZARD.utils.model import Semantic_Mapping
from HAZARD.utils.scene_setup import SceneSetup
import numpy as np
import cv2
import copy
import os
import torch

class FloodAgentController(FloodController):
    """
    Never ever use self.commands in here! For safety uses, self.commands can only be used in parent class
    """

    def __init__(self, use_local_resources: bool = False, single_room: bool = True, reverse_observation: bool = False,
                 image_capture_path: str = None, log_path: str = None, record_only: bool = False,
                 use_dino: bool = False, **kwargs) -> None:
        self.frame_count = 0
        self.use_local_resources = use_local_resources
        self.image_capture_path = image_capture_path
        self.reverse_observation = reverse_observation
        self.log_path = log_path
        self.action_slowdown = 0
        super().__init__(**kwargs)
        self.screen_size = kwargs.get("screen_size", 512)
        self.agents: List[FloodAgent] = []
        self.comm_counter = 0
        self.use_gt = kwargs.get("use_gt", True)
        self.use_dino = use_dino
        self.single_room = single_room
        self.record_only = record_only
        self.id2name = {}
        self.other_containers = {}
        if not self.use_gt:
            if self.use_dino:
                from HAZARD.utils.vision_dino import DetectorSAM
                self.detector = DetectorSAM(**kwargs)
            else:
                self.detector = Detector(**kwargs)
        # self.init_seg(vocab_path=f"{os.getcwd()}/seg/vocab.txt")
        if use_local_resources:
            self.update_replicant_url()

        self.map_size_h = kwargs.get("map_size_h", 64)
        self.map_size_v = kwargs.get("map_size_v", 64)
        self.grid_size = kwargs.get("grid_size", 0.25)
        self.sem_map = Semantic_Mapping(device=None, screen_size=self.screen_size,
                                        map_size_h=self.map_size_h, map_size_v=self.map_size_v, grid_size=self.grid_size)

        self.maps = []
        # self.init_seg()

    def update_replicant_url(self):
        if not os.path.isfile(f"{os.getcwd()}/data/assets/replicant_0"): # There is no local model of replicant
            print("There is no local model of replicant. ")
            return
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

        self.floor_flood_info = dict()
        self.physical_flood_info = dict()
        self.flood_candidate = dict()
        self.initialized = False
        self.add_ons = []
        self.id2name = {}
        self.manager.reset()
        self.frame_count = 0
        self.communicate([])

        self.last_reward = None
        self.communicate([{"$type": "destroy_all_objects"}])

    def init_scene(self, setup: SceneSetup):
        self.reset_scene()
            
        if self.log_path is not None:
            logger = Logger(self.log_path)
            self.add_ons.append(logger)
            self.communicate([])

        self.other_containers = {}
        for obj in setup.objects:
            self.manager.add_object(obj)
        for commands in setup.commands_list:
            if len(commands) == 1 and commands[0]["$type"] == "terminate":
                continue
            filtered_commands = []
            for command in commands:
                tp = command["$type"]
                if tp == "terminate":
                    break
                if tp[:4] == "send" or tp.find("avatar") != -1 or "avatar_id" in command:
                    continue
                filtered_commands.append(command)
                if tp == "add_object":
                    name = command["name"]
                    idx = command["id"]
                    self.id2name[idx] = name
                if tp.startswith("add_") and tp.endswith("_container"):
                    self.other_containers[command['id']] = command['container_id']
            # json.dump(filtered_commands, open("/data/private/zqh/tmp.json", "w"))
            self.communicate(filtered_commands)


        if not self.record_only:
            if len(self.agents) == 0:
                self.agents: List[FloodAgent] = []
                for agent_pos in setup.agent_positions:
                    idx = self.get_unique_id()
                    self.agents.append(
                        FloodAgent(replicant_id=idx, position=agent_pos, image_frequency=ImageFrequency.always))
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
        self.target = setup.targets
        if self.use_dino and not self.use_gt:
            self.detector.set_targets(self.target)
        self.target_ids = setup.target_ids
        self.target_names = setup.target_names
        self.target_id2category = setup.target_id2category
        self.target_id2name = setup.target_id2name
        self.finished = []
        self.communicate([])
        if self.image_capture_path is not None:
            # camera = ThirdPersonCamera(avatar_id="record", position={"x": -7.78, "y": 7.67, "z": 0.16},
            #                            look_at=self.agents[0].replicant_id)
            if not self.record_only:
                camera = ThirdPersonCamera(avatar_id="record",
                                           position={"x": 0.0, "y": 10.0, "z": 0.0},
                                           look_at={"x": 0.0, "y": 0.0, "z": 0.0})
                # if len(self.agents) == 0 else self.agents[0].replicant_id)
                commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                            {"$type": "set_target_framerate", "framerate": 30}]
            else:
                camera = ThirdPersonCamera(avatar_id="record", position={"x": 1.5, "y": 2.0, "z": 0.0},
                                           look_at={"x": 0.0, "y": 0.0, "z": 0.0})
                commands = [{"$type": "set_screen_size", "width": self.screen_size * 4, "height": self.screen_size * 4},
                            {"$type": "set_target_framerate", "framerate": 30}]
            self.add_ons.extend([camera])
        else:
            commands = [{"$type": "set_screen_size", "width": self.screen_size, "height": self.screen_size},
                        {"$type": "set_target_framerate", "framerate": 30}]
            commands.extend([{"$type": "set_floorplan_roof", "show": False}])

        if self.image_capture_path is not None:
            commands.extend([{"$type": "set_pass_masks", "pass_masks": ["_img"], "avatar_id": "record"},
                             {"$type": "send_images", "frequency": "always", "ids": ["record"]}])

        # Get the floors for this floorplan. Create flood objects for each floor.
        # These will be at 0 in Y, and not visible until their height is adjusted.
        # The index is the number of the scene, i.e. "1".
        if self.single_room:
            # use fake visual effect
            from tdw.librarian import SceneLibrarian
            lib = SceneLibrarian()
            record = lib.get_record(setup.scene_name)
            rooms = record.rooms[0].alcoves
            # rooms = []
            rooms.append(record.rooms[0].main_region)
            effect_name = 'fplan1_floor1'
            original_effect_size = (4.1796417236328125, 2.817157745361328, 5.654577255249023)
            for i, room in enumerate(rooms):
                floor_id = self.get_unique_id()
                rotation = {"x": 90, "y": 0, "z": 0}
                position = {'x': room.center[0], 'y': room.center[1], 'z': room.center[2]}
                self.commands.append(Controller.get_add_visual_effect(name=effect_name,
                                                                      effect_id=floor_id,
                                                                      position=position,
                                                                      rotation=rotation,
                                                                      library="flood_effects.json"))
                self.commands.append({"$type": "scale_visual_effect",
                                      "scale_factor": {"x": room.bounds[0]/original_effect_size[0],
                                                       "y": room.bounds[2]/original_effect_size[1],
                                                       "z": 0.01
                                                       },
                                      "id": floor_id})
                # Add to dictionary of currently-flooded floors. Position will include current height (flood level).
                self.floor_flood_info[i] = FloorFlood(floor_id=floor_id)
                self.manager.add_flood(id=floor_id, position=position, direction=rotation,
                                       scale=np.array([1.0, 1.0, 1.0]))
        else:
            self._floors = FloorplanFlood._FLOOD_DATA[setup.scene_name.split("_")[-1][0]]
            for i in range(len(self._floors)):
                floor = self._floors[str(i + 1)]
                floor_name = floor["name"]
                floor_id = self.get_unique_id()
                position = floor["position"]
                rotation = {"x": 90, "y": 0, "z": 0}
                self.commands.append(Controller.get_add_visual_effect(name=floor_name,
                                                                      effect_id=floor_id,
                                                                      position=position,
                                                                      rotation=rotation,
                                                                      library="flood_effects.json"))
                # Add to dictionary of currently-flooded floors. Position will include current height (flood level).
                self.floor_flood_info[i] = FloorFlood(floor_id=floor_id)
                self.manager.add_flood(id=floor_id, position=position, direction=rotation,
                                       scale=np.array([1.0, 1.0, 1.0]))

        resp = self.communicate([{"$type": "send_scene_regions"}])
        self.manager.flood_manager.source_from = setup.flood_source_from
        self.manager.flood_manager.set_scene_bounds(resp=resp)
        self.add_ons.append(self.manager)

        self.manager.prepare_segmentation_data()
        self.initialized = True
        for obj in self.target_ids:
            commands.append({"$type": "set_kinematic_state", "id": obj, "is_kinematic": False, "use_gravity": True})
        for obj in self.manager.objects:
            if self.manager.objects[obj].has_buoyancy:
                commands.append({"$type": "set_kinematic_state", "id": obj, "is_kinematic": False, "use_gravity": True})
        self.communicate(commands)

        if not self.record_only:
            commands = [
                {"$type": "set_field_of_view", "field_of_view": 120.0, "avatar_id": str(self.agents[0].replicant_id)}]
            """add a backpack or similar to the agent's left hand"""
            self.container_name = "backpack" if "container" not in setup.other else setup.other["container"]
            self.container_id = self.get_unique_id()
            self.manager.add_object(ObjectStatus(idx=self.container_id, waterproof=True, has_buoyancy=False))

            agent_pos = self.agents[0].dynamic.transform.position
            commands.append(self.get_add_object(model_name=self.container_name, object_id=self.container_id,
                                                position=TDWUtils.array_to_vector3(agent_pos)))
            self.communicate(commands)
            self.agents[0].grasp(target=self.container_id, arm=Arm.left, axis=None, angle=None)
            self.next_key_frame()
            self.agents[0].reach_for(target=TDWUtils.array_to_vector3([-0.3, 1.0, 0.3]), absolute=False, arrived_at=0.1,
                                     arm=Arm.left)
            self.next_key_frame()
            self.agents[0].reset_arm(arm=Arm.left)
            self.next_key_frame()

        # TODO add source
        # self.init_obi()
        # for idx in range(len(setup.flood_positions)):
        #     self.add_physical_flood(position=setup.flood_positions[idx],
        #                             direction=setup.flood_directions[idx],
        #                             speed=setup.flood_speed[idx])

    def next_key_frame(self, force_direction=None) -> Tuple[List[ActionStatus], List[int]]:
        # print("next_key_frame")
        agent_idx = 0
        action_slowdown = 0
        if force_direction is not None:
            position = self.agents[agent_idx].dynamic.transform.position
            if self.manager.flood_manager.query_point_underwater(position):
                facing = self.agents[agent_idx].dynamic.transform.forward
                force_scale = (force_direction * (facing / np.linalg.norm(facing))).sum()
                action_slowdown += (1 - force_scale) * self.constants.AGENT_DRAG_COEFFICIENT

        self.action_slowdown += action_slowdown
        if self.action_slowdown > 1:
            self.action_slowdown -= 1
            self.frame_count += 1
            self.manager.flood_manager.evolve()

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

        self.communicate([])
        return final_status, changed_agents

    def do_action(self, agent_idx: int, action: str, params: Dict[str, Any] = dict()) -> None:
        """
        See the FireAgent class for allowed actions and their parameters.
        """
        if action == "stop":
            return
        return getattr(self.agents[agent_idx], action)(**params)

    def get_agent_status(self, idx) -> FloodAgent:
        return self.agents[idx]

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
        if self.initialized:
            self.frame_count += 1
            for agent in self.agents:
                idx = agent.grasp_id()
                if idx is None:
                    continue
                dist = np.linalg.norm((agent.dynamic.transform.position - self.manager.objects[idx].position) * np.array([1, 0, 1]))
                if dist > 3.0:
                    agent.fail_grasp()
        if self.use_local_resources:
            commands = self.replace_with_local_path(commands)
        resp = super().communicate(commands)
        if self.initialized:
            for target in self.target_ids:
                if (np.abs(self.manager.objects[target].position).sum() > 50) and (target not in self.finished):
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
                        TDWUtils.save_images(images=images, filename=str(self.comm_counter),
                                             output_directory=self.image_capture_path)
                        self.comm_counter += 1
        return resp

    def get_seg_mask(self, rgb, obj2idx):
        # TODO check the performance
        semantic_input, _ = self.predictor.run_on_image(rgb, obj2idx)
        return semantic_input

    def get_flood_height_observation(self, idx, width=512, height=512):
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
                flood_height = self.manager.query_point_flood_height(point_cloud[:, i, j])
                temp[i, j] = flood_height
        temp = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        return temp

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
        seg_mask = None
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

        flood_height = self.get_flood_height_observation(agent_idx, width=self.screen_size, height=self.screen_size)
        flood_height = flood_height.reshape((1, self.screen_size, self.screen_size)).astype(np.float32)
        obs["raw"] = dict(
            rgb=rgb,
            depth=depth,
            log_temp=flood_height,
            seg_mask=seg_mask
        )

        """
        mapped observation
        """
        camera_matrix = self.agents[agent_idx].dynamic.camera_matrix.reshape((4, 4))
        obs_concat = np.concatenate([rgb, depth, flood_height], axis=0)
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