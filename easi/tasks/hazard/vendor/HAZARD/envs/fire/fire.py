from typing import List, Union, Dict
from tdw.controller import Controller
from tdw.output_data import OutputData, Overlap

import numpy as np
from .manager import FireObjectManager
from .fire_utils import *
from .object import FireStatus, AgentStatus


def CHOUKA(T, constants=default_const):
    '''
    A draw("Chou Ka") process with guarantee. 
    
    In the first {constants.CHOUKA_THRESHOLD_PROB_INCREASE} draws, 
    the probability is {constants.CHOUKA_INITITIAL_PROB}. 
    
    If failed to hit in the first {constants.CHOUKA_THRESHOLD_PROB_INCREASE} draws, 
    the probability will increase linearly, 
    until the {constants.CHOUKA_THRESHOLD_UPPER}'th times when the probability will be 1. 
    '''
    start_prob = constants.CHOUKA_INITITIAL_PROB
    if T <= constants.CHOUKA_THRESHOLD_PROB_INCREASE:
        return start_prob
    else:
        return min(1.0, 
                   start_prob + 
                   (1 - start_prob) / (constants.CHOUKA_THRESHOLD_UPPER-constants.CHOUKA_THRESHOLD_PROB_INCREASE) 
                   * (T - constants.CHOUKA_THRESHOLD_PROB_INCREASE))

def box_overlap(pos1, size1, pos2, size2):
    pos1 = np.array(pos1)
    size1 = np.array(size1)
    pos2 = np.array(pos2)
    size2 = np.array(size2)
    return np.all(pos1 + size1 / 2 > pos2 - size2 / 2) and np.all(pos1 - size1 / 2 < pos2 + size2 / 2)

class Fire:
    def __init__(self, fire_id, last_spread, scale, spread_dirs):
        self.fire_id = fire_id
        self.last_spread = last_spread
        self.scale = scale
        self.spread_dirs = spread_dirs

        self.extinguishing = False

"""
This controller controls the spread of fire.
"""
class FireController(Controller):
    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, seed = 0, constants = default_const, **kwargs):
        self.initialized = False
        self.commands: List[dict] = list()
        super().__init__(port, check_version, launch_build)
        
        self.manager = FireObjectManager()
        self.add_ons.append(self.manager)

        self.update_fire_per_frame = 10
        self.frame_count = 0
        self.fire_info: Dict[int, Fire] = dict()
        self.fire_candidate = dict()
        self.RNG = np.random.Generator(np.random.PCG64(seed))
        self.constants = constants

    def seed(self, seed):
        self.RNG = np.random.Generator(np.random.PCG64(seed))
    
    def get_unique_id(self):
        while True:
            idx = super().get_unique_id()
            if idx not in self.manager.objects:
                return idx
    
    def add_fire(self, position, object_id=None, scale=None):
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if scale == None: scale = self.constants.FIRE_INIT_SCALE
        self.commands.append(self.get_add_visual_effect("fire", effect_id=object_id, position=position))
        self.commands.append({"$type": "scale_visual_effect", "id": object_id, "scale_factor": {"x": scale, "y": scale, "z": scale}})

    def add_fire_floor(self, position):
        if isinstance(position, np.ndarray):
            position = position.tolist()
        idx = self.get_unique_id()
        self.add_fire(position={"x": position[0], "y": position[1], "z": position[2]}, object_id=idx)
        self.manager.add_object(FireStatus(idx, constants=self.constants, position=position, size=self.constants.FIRE_SPREAD_SIZE))
        # self.fire_idx[idx] = [idx, self.frame_count, [(-1.1, 0), (1.1, 0), (0, -1.1), (0, 1.1)], self.constants.FIRE_INIT_SCALE]
        self.fire_info[idx] = Fire(fire_id=idx,
                                   last_spread=self.frame_count,
                                   scale=self.constants.FIRE_INIT_SCALE,
                                   spread_dirs=[(-1.1, 0), (1.1, 0), (0, -1.1), (0, 1.1)])

    def add_fire_object(self, idx):
        new_idx = self.get_unique_id()
        position = self.manager.objects[idx].top() + np.array([0, 0.01, 0])
        self.add_fire(position={"x": position[0], "y": position[1], "z": position[2]}, object_id=new_idx)
        self.manager.add_object(FireStatus(new_idx, constants=self.constants, position=position, size=self.constants.FIRE_SPREAD_SIZE))
        self.commands.append({"$type": "parent_visual_effect_to_object", "object_id": idx, "id": new_idx})
        # self.fire_idx[idx] = [new_idx, self.frame_count, [(-1.1, 0), (1.1, 0), (0, -1.1), (0, 1.1)], self.constants.FIRE_FINAL_SCALE]
        self.fire_info[idx] = Fire(fire_id=new_idx,
                                   last_spread=self.frame_count,
                                   scale=self.constants.FIRE_FINAL_SCALE,
                                   spread_dirs=[(-1.1, 0), (1.1, 0), (0, -1.1), (0, 1.1)])

        size = self.manager.objects[idx].size
        self.commands.append({"$type": "scale_visual_effect", "id": new_idx, "scale_factor": {"x": float(size[0]*0.5), "y": float(size[1]*0.5), "z": float(size[2]*0.5)}})
    
    def extinguish_fire_floor(self, fire_idx):
        """
        Extinguish a fire on the floor. Send fire id.
        """
        self.fire_info[fire_idx].extinguishing = True
        self.fire_info[fire_idx].spread_dirs = []
        self.manager.objects[fire_idx].temperature = self.constants.ROOM_TEMPERATURE
        pass

    def extinguish_fire_object(self, obj_idx):
        """
        Extinguish a fire on an object. Send object id, not the fire id.
        """
        self.fire_info[obj_idx].extinguishing = True
        self.fire_info[obj_idx].spread_dirs = []
        self.manager.objects[obj_idx].temperature = self.constants.ROOM_TEMPERATURE
        self.manager.objects[obj_idx].is_heat_source = True
        self.manager.objects[self.fire_info[obj_idx].fire_id].temperature = self.constants.ROOM_TEMPERATURE
        pass
    
    def add_agent(self, idx, pos):
        self.manager.add_object(AgentStatus(idx, constants=self.constants, position=pos, size=None, endurance=10000))
    
    def candidate_fire_overlap(self, pos, size):
        for pos2, size2 in self.fire_candidate.values():
            if box_overlap(pos, size, pos2, size2):
                return True
        for idx in self.fire_info:
            if box_overlap(pos, size, self.manager.objects[idx].position, self.constants.FIRE_SPREAD_SIZE):
                return True
        return False

    def try_spread(self, idx):
        if idx in self.manager.objects and np.abs(self.manager.objects[idx].position).sum() > 50:
            return
        if len(self.fire_info[idx].spread_dirs) == 0:
            return
        burning_time = self.frame_count - self.fire_info[idx].last_spread
        spread_prob = CHOUKA(burning_time // 5)
        if self.RNG.uniform() < spread_prob:
            spread_dir = self.fire_info[idx].spread_dirs[self.RNG.choice(len(self.fire_info[idx].spread_dirs))]
            self.fire_info[idx].spread_dirs.remove(spread_dir)
            spread_dir = np.array([spread_dir[0], 0, spread_dir[1]])

            pos = self.manager.objects[idx].position * np.array([1, 0, 1]) + spread_dir * (self.constants.FIRE_SPREAD_SIZE + self.manager.objects[self.fire_info[idx].fire_id].size) * 0.5
            if not self.candidate_fire_overlap(pos, self.constants.FIRE_SPREAD_SIZE):
                self.fire_candidate[self.get_unique_id()] = (pos, self.constants.FIRE_SPREAD_SIZE)
            self.fire_info[idx].last_spread = self.frame_count

    def fire_step(self, resp):
        # for idx in self.manager.objects_start_burning:
        #     self.commands.append({"$type": "set_color",
        #                             "id": idx,
        #                             "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.5}})
        #     self.add_fire_object(idx)
        for idx in self.manager.objects_stop_burning:
            # self.commands.append({"$type": "set_color",
            #                         "id": idx,
            #                         "color": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.5}})
            self.commands.append({"$type": "destroy_visual_effect",
                                    "id": self.fire_info[idx].fire_id})
            self.manager.remove_object(self.fire_info[idx].fire_id)
            self.fire_info.pop(idx)
        # extend fire
        for i in range(len(resp)-1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "over":
                o = Overlap(resp[i])
                idx = o.get_id()
                # if idx in self.fire_candidate:
                #     print("check", idx, o.get_object_ids(), o.get_env())
                if idx in self.fire_candidate and o.get_object_ids().size < 3 and o.get_env() == False:
                    self.add_fire_floor(self.fire_candidate[idx][0])
        self.fire_candidate.clear()
        if self.frame_count % self.update_fire_per_frame == 0:
            removed = []
            for idx in self.fire_info:
                if self.fire_info[idx].scale < self.constants.FIRE_FINAL_SCALE and not self.fire_info[idx].extinguishing:
                    # expand large enough before spreading
                    self.fire_info[idx].scale += self.constants.FIRE_SCALE_STEP
                    ratio = self.fire_info[idx].scale / (self.fire_info[idx].scale - self.constants.FIRE_SCALE_STEP)
                    self.commands.append({"$type": "scale_visual_effect",
                                            "id": self.fire_info[idx].fire_id,
                                            "scale_factor": {"x": ratio, "y": ratio, "z": ratio}})
                elif not self.fire_info[idx].extinguishing:
                    # large enough, try spreading
                    self.try_spread(idx)
                elif self.fire_info[idx].scale >= self.constants.FIRE_INIT_SCALE + self.constants.FIRE_SCALE_STEP:
                    # extinguishing, scale down
                    self.fire_info[idx].scale -= self.constants.FIRE_SCALE_STEP
                    ratio = self.fire_info[idx].scale / (self.fire_info[idx].scale + self.constants.FIRE_SCALE_STEP)
                    self.commands.append({"$type": "scale_visual_effect",
                                            "id": self.fire_info[idx].fire_id,
                                            "scale_factor": {"x": ratio, "y": ratio, "z": ratio}})
                else:
                    # scaled down enough, remove
                    self.commands.append({"$type": "destroy_visual_effect",
                                            "id": self.fire_info[idx].fire_id})
                    self.manager.remove_object(self.fire_info[idx].fire_id)
                    removed.append(idx)
            for idx in removed:
                self.fire_info.pop(idx)

        for i in self.fire_candidate:
            pos, size = self.fire_candidate[i]
            self.commands.append({"$type": "send_overlap_box",
                                  "half_extents": {"x": size[0]*0.5, "y": size[1]*0.5, "z": size[2]*0.5},
                                  "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                                  "position": {"x": pos[0], "y": pos[1]+size[1]*0.5+0.01, "z": pos[2]},
                                  "id": i})
        self.frame_count += 1

    # def run(self):
    #     # self.add_fire_floor([0, 0, 0])
    #     for i in range(1000):
    #         self.communicate([])
    #     self.communicate({"$type": "terminate"})

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        if isinstance(commands, dict):
            commands = [commands]
        commands.extend(self.commands)
        self.commands.clear()
        # for com in commands:
        #     print(com)
        resp = super().communicate(commands)
        if self.initialized:
            self.fire_step(resp)
        return resp
