from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Transforms, Bounds
from tdw.output_data import Replicants, SegmentationColors, ReplicantSegmentationColors
from typing import Dict, List, Set, Optional
import numpy as np
from .object import *
from .fire_utils import *
from HAZARD.utils.seg_id import SegmentationID

"""
Add-on to control the evolvement of objects, including temperature and state.
Future development may include changing object appearances.
"""
class FireObjectManager(AddOn):
    def __init__(self, constants=default_const):
        super().__init__()
        self.constants = constants
        self.objects: Dict[int, ObjectStatus] = dict()
        self.temperature_manager = TemperatureManager()
        self.objects_start_burning: Set[int] = set()
        self.objects_stop_burning: Set[int] = set()
        self.segm = SegmentationID()
        self.id_renumbering = dict()
        self.id_list = [0]
        self.timer = 0

    def reset(self):
        self.objects = dict()
        self.objects_start_burning = set()
        self.objects_stop_burning = set()
        self.commands = []
        self.initialized = False
        self.id_renumbering = dict()
        self.id_list = [0]
        self.timer = 0

    def get_initialization_commands(self) -> List[dict]:
        return [{"$type": "send_bounds"},
                {"$type": "send_transforms"}]
    
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
                        print("object {} not recorded, may be caused by composite objects which you can ignore".format(idx))
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
                    else:
                        print("object {} not recorded, may be caused by composite objects which you can ignore".format(idx))
                        self.add_object(ObjectStatus(idx, size=boun.get_front(j) + boun.get_right(j) + boun.get_top(j) - boun.get_back(j) - boun.get_left(j) - boun.get_bottom(j)))
                        concat = np.zeros([6, 3])
                        concat[0] = boun.get_front(j)
                        concat[1] = boun.get_back(j)
                        concat[2] = boun.get_left(j)
                        concat[3] = boun.get_right(j)
                        concat[4] = boun.get_top(j)
                        concat[5] = boun.get_bottom(j)
                        self.objects[idx].size = np.max(concat, axis=0) - np.min(concat, axis=0)
            elif r_id == "repl":
                repl = Replicants(resp[i])
                for j in range(repl.get_num()):
                    idx = repl.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].position = repl.get_position(j)
                    else:
                        print("agent {} not recorded, this shouldn't happen".format(idx))
                        self.add_object(AgentStatus(idx, position=repl.get_position(j), size=None))
        for i in range(len(resp)-1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                self.segm.process(segm, self.id_renumbering)
            elif r_id == "rseg":
                segm = ReplicantSegmentationColors(resp[i])
                self.segm.process(segm, self.id_renumbering)
        
        self.timer += 1
        if self.timer % 5 == 0:
            temp_dict = self.temperature_manager.evolve(self.objects)
            for idx in temp_dict:
                self.objects[idx].temperature = temp_dict[idx]
            
            self.objects_start_burning = set()
            self.objects_stop_burning = set()
            for idx in self.objects:
                if np.abs(self.objects[idx].position).sum() > 50:
                    continue
                status = self.objects[idx].step()
                if status == ObjectState.START_BURNING:
                    self.objects_start_burning.add(idx)
                elif status == ObjectState.STOP_BURNING:
                    self.objects_stop_burning.add(idx)
        self.commands = [{"$type": "send_bounds"}, {"$type": "send_transforms"}]
    
    def add_object(self, obj: ObjectStatus):
        self.objects[obj.idx] = obj
        if obj.idx not in self.id_renumbering:
            self.id_list.append(obj.idx)
            self.id_renumbering[obj.idx] = len(self.id_list) - 1
    
    def remove_object(self, idx: int):
        if idx in self.objects:
            del self.objects[idx]
    
    def query_point_temperature(self, point: np.ndarray) -> float:
        return self.temperature_manager.query_point_temperature(point, self.objects)
    
    def find_nearest_object(self, pos: np.ndarray, objects: Optional[List[int]] = None):
        min_dist = 1e10
        min_idx = None
        it = iter(self.objects) if objects is None else iter(objects)
        for idx in it:
            if self.objects[idx].name != "Object":
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
