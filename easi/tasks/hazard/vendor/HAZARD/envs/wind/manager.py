from tdw.add_ons.add_on import AddOn
from .object import *
from tdw.output_data import OutputData, Transforms, Bounds, Rigidbodies
from tdw.output_data import Replicants, SegmentationColors, ReplicantSegmentationColors
from typing import Dict, List
import numpy as np
from HAZARD.utils.seg_id import SegmentationID

"""
Add-on to manage the objects.
"""

class WindObjectManager(AddOn):
    def __init__(self, constants=default_const):
        super().__init__()
        self.constants = constants
        self.objects: Dict[int, ObjectStatus] = dict()
        self.wind_force_manager = WindForceManager()
        self.effects = dict()
        self.settled = set()
        self.wind_v: np.ndarray = np.array([0, 0, 0])
        self.num_frame = 0
        self.segm = SegmentationID()
        self.id_renumbering = dict()
        self.id_list = [0]
    
    def reset(self):
        self.objects = dict()
        self.effects = dict()
        self.settled = set()
        self.num_frame = 0
        self.wind_v = np.array([0, 0, 0])
        self.commands = []
        self.initialized = False
        self.id_renumbering = dict()
        self.id_list = [0]
    
    def get_initialization_commands(self) -> List[dict]:
        return [{"$type": "send_bounds"},
                {"$type": "send_transforms"},
                {"$type": "send_rigidbodies"}]
    
    def on_send(self, resp: List[bytes]) -> None:
        for i in range(len(resp)-1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "tran":
                tran = Transforms(resp[i])
                for j in range(tran.get_num()):
                    idx = tran.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].position = tran.get_position(j)
                        self.objects[idx].rotation = tran.get_rotation(j)
                    else:
                        print("Warning: object with id {} not found in WindObjectManager".format(idx))
                        self.add_object(ObjectStatus(idx, position=tran.get_position(j)))
            elif r_id == "boun":
                boun = Bounds(resp[i])
                for j in range(boun.get_num()):
                    idx = boun.get_id(j)
                    if idx in self.objects:
                        concat = np.zeros([6, 3])
                        concat[0] = boun.get_front(j)
                        concat[1] = boun.get_back(j)
                        concat[2] = boun.get_left(j)
                        concat[3] = boun.get_right(j)
                        concat[4] = boun.get_top(j)
                        concat[5] = boun.get_bottom(j)
                        self.objects[idx].size = np.max(concat, axis=0) - np.min(concat, axis=0)
                        if self.objects[idx].size[0] < 0:
                            print(boun.get_front(j), boun.get_back(j), boun.get_left(j), boun.get_right(j), boun.get_top(j), boun.get_bottom(j))
                    else:
                        print("Warning: object with id {} not found in WindObjectManager".format(idx))
                        self.add_object(ObjectStatus(idx, size=boun.get_front(j) + boun.get_right(j) + boun.get_top(j) - boun.get_back(j) - boun.get_left(j) - boun.get_bottom(j)))
                        concat = np.zeros([6, 3])
                        concat[0] = boun.get_front(j)
                        concat[1] = boun.get_back(j)
                        concat[2] = boun.get_left(j)
                        concat[3] = boun.get_right(j)
                        concat[4] = boun.get_top(j)
                        concat[5] = boun.get_bottom(j)
                        self.objects[idx].size = np.max(concat, axis=0) - np.min(concat, axis=0)
                        if self.objects[idx].size[0] < 0:
                            print(boun.get_front(j), boun.get_back(j), boun.get_left(j), boun.get_right(j), boun.get_top(j), boun.get_bottom(j))
            elif r_id == "repl":
                repl = Replicants(resp[i])
                for j in range(repl.get_num()):
                    idx = repl.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].position = repl.get_position(j)
                    else:
                        print("Warning: object with id {} not found in WindObjectManager".format(idx))
                        self.add_object(AgentStatus(idx, position=repl.get_position(j)))
            elif r_id == "rigi":
                rigi = Rigidbodies(resp[i])
                for j in range(rigi.get_num()):
                    idx = rigi.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].velocity = rigi.get_velocity(j)
                    else:
                        print("Warning: object with id {} not found in WindObjectManager".format(idx))
                        self.add_object(ObjectStatus(idx, velocity=rigi.get_velocity(j)))
        for i in range(len(resp)-1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                self.segm.process(segm, id_renumbering=self.id_renumbering)
            elif r_id == "rseg":
                segm = ReplicantSegmentationColors(resp[i])
                self.segm.process(segm, id_renumbering=self.id_renumbering)
        self.effects = self.wind_force_manager.evolve(self.objects, self.wind_v, self.settled)
        self.commands = [{"$type": "send_bounds"}, {"$type": "send_transforms"}, {"$type": "send_rigidbodies"}]
        self.num_frame += 1
        if np.linalg.norm(self.wind_v * [1, 0, 1]) > 0.1:
            for idx in self.objects:
                if idx in self.settled or self.objects[idx].position is None or self.objects[idx].size is None:
                    continue
                for idx2 in self.settled:
                    if idx2 == idx or self.objects[idx2].position is None or self.objects[idx2].size is None:
                        continue
                    # extent of idx is fully covered by idx2
                    l, r, f, b = self.objects[idx2].left()[0], self.objects[idx2].right()[0], self.objects[idx2].front()[2], self.objects[idx2].back()[2]
                    x, y, z = self.objects[idx].position.tolist()
                    bottom = self.objects[idx2].bottom()[1]
                    top = self.objects[idx2].top()[1]
                    if l <= x and x <= r and b <= z and z <= f and bottom <= y and y <= top:
                        self.settled.add(idx)
                        break
    def add_object(self, obj: ObjectStatus):
        self.objects[obj.idx] = obj
        if obj.idx not in self.id_renumbering:
            self.id_list.append(obj.idx)
            self.id_renumbering[obj.idx] = len(self.id_list) - 1
    
    def remove_object(self, obj: ObjectStatus):
        if obj.idx in self.objects:
            del self.objects[obj.idx]
    
    def find_nearest_object(self, pos: np.ndarray, objects: Optional[List[int]] = None):
        min_dist = 1e10
        min_idx = None
        it = iter(self.objects) if objects is None else iter(objects)
        for idx in it:
            if self.objects[idx].name != "Object":
                continue
            if idx in self.settled and objects is None:
                continue
            dist = np.linalg.norm(self.objects[idx].position - pos)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        return min_idx
    
    def prepare_segmentation_data(self):
        self.commands.extend([{"$type": "send_segmentation_colors"},
                {"$type": "send_categories"}, {"$type": "send_replicant_segmentation_colors"}])
    
    def get_renumbered_id(self, idx: int):
        if idx in self.id_renumbering:
            return self.id_renumbering[idx]
        return 0
    
    def get_renumbered_list(self, L: List[int]):
        return [self.id_renumbering[idx] for idx in L if idx in self.id_renumbering]
    
    def get_real_id(self, idx: int):
        if idx >= len(self.id_list) or idx == 0:
            return None
        return self.id_list[idx]
