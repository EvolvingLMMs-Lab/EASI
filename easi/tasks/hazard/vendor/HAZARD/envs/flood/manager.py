from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Transforms, Bounds, Rigidbodies
from tdw.output_data import Replicants, SegmentationColors, ReplicantSegmentationColors
from typing import Dict, List, Set, Optional
import numpy as np
from .object import FloodManager, ObjectStatus
from .utils import *
from HAZARD.utils.seg_id import SegmentationID

"""
Add-on to control the evolvement of objects, including temperature and state.
Future development may include changing object appearances.
"""


class FloodObjectManager(AddOn):
    def __init__(self, constants=default_const, source_position=None, source_from=None, floor_ids=[],
                 floor_positions=[], floor_sizes=[], floor_directions=[], flood_density=1.0):
        super().__init__()
        self.constants = constants
        self.objects: Dict[int, ObjectStatus] = dict()
        self.flood_manager = FloodManager(ascending_speed=constants.ASCENDING_SPEED,
                                          ascending_interval=constants.ASCENDING_INTERVAL,
                                          max_height=constants.MAX_HEIGHT,
                                          floor_ids=[],
                                          source_position=source_position,
                                          source_from=source_from,
                                          floor_positions=floor_positions,
                                          floor_sizes=floor_sizes,
                                          floor_directions=floor_directions,
                                          flood_density=flood_density,
                                          roll_theta=constants.SLOP_ANGLE,
                                          flood_force_scale=constants.FLOOD_FORCE_SCALE,
                                          drag_coefficient=constants.DRAG_COEFFICIENT)
        self.objects_floating: Set[int] = set()
        self.objects_flooded: Set[int] = set()
        self.recover_command_list = []
        self.segm = SegmentationID()
        self.id_renumbering = dict()
        self.id_list = [0]

    def update_visual_effects(self):
        new_effect_dict = self.flood_manager.evolve()
        self.recover_command_list.reverse()
        self.commands = [{"$type": "send_bounds"}, {"$type": "send_transforms"}, {"$type": "send_rigidbodies"}]
        self.commands.extend(self.recover_command_list)
        self.recover_command_list = []
        floor_flood_commands = []
        for floor_effect_idx in new_effect_dict:
            for axis in ["yaw", "pitch", "roll"]:
                if new_effect_dict[floor_effect_idx]["angles"][axis] > 0:
                    floor_flood_commands.append(
                        {"$type": "rotate_visual_effect_by",
                         "angle": new_effect_dict[floor_effect_idx]["angles"][axis],
                         "axis": axis,
                         "id": floor_effect_idx,
                         "is_world": True}
                    )
                    self.recover_command_list.append(
                        {"$type": "rotate_visual_effect_by",
                         "angle": -new_effect_dict[floor_effect_idx]["angles"][axis],
                         "axis": axis,
                         "id": floor_effect_idx,
                         "is_world": True}
                    )
            floor_flood_commands.append(
                {"$type": "scale_visual_effect",
                 "scale_factor": {"x": new_effect_dict[floor_effect_idx]["scales"]['x'],
                                  "y": new_effect_dict[floor_effect_idx]["scales"]['y'],
                                  "z": new_effect_dict[floor_effect_idx]["scales"]['z']
                                  },
                 "id": floor_effect_idx}
            )
            floor_flood_commands.append(
                {"$type": "teleport_visual_effect",
                 "position": {"x": new_effect_dict[floor_effect_idx]["positions"]['x'],
                              "y": new_effect_dict[floor_effect_idx]["positions"]['y'],
                              "z": new_effect_dict[floor_effect_idx]["positions"]['z']},
                 "id": floor_effect_idx}
            )
        self.objects_flooded = set()
        self.objects_floating = set()
        counter = 0
        for idx in self.objects:
            if self.objects[idx].name == "Agent":
                continue
            obj = self.objects[idx]
            status, buoyancy_scale = self.flood_manager.update_object_status_new(self.objects[idx])
            if status.state == ObjectState.FLOODED or status.state == ObjectState.FLOODED_FLOATING:
                self.objects_flooded.add(idx)
            if status.state == ObjectState.FLOATING or status.state == ObjectState.FLOODED_FLOATING:
                self.objects_floating.add(idx)
                # height_diff = self.query_height_diff(position=None)
                # if not start_floating:
                #     floor_flood_commands.append({
                #         "$type": "teleport_object_by", "id": status.idx,
                #         "position": {"x":0.0, "y":height_diff, "z":0.0}, "absolute": True
                #     })
            # add buoyancy
            if isinstance(obj.size, np.ndarray) and isinstance(obj.velocity, np.ndarray):
                counter += 1
                horizontal_force = self.flood_manager.cal_horizontal_force(object=self.objects[idx],
                                                                           source=self.flood_manager.source_position)
                floor_flood_commands.append({"$type": "apply_force_to_object",
                                             "id": status.idx,
                                             "force": {"x": horizontal_force["x"],
                                                       "y": buoyancy_scale + horizontal_force["y"],
                                                       "z": horizontal_force["z"]}})
            else:
                floor_flood_commands.append({"$type": "apply_force_to_object",
                                             "id": status.idx,
                                             "force": {"x": 0,
                                                       "y": buoyancy_scale,
                                                       "z": 0}})
        # print(counter)
        self.commands.extend([{"$type": "send_bounds"}, {"$type": "send_transforms"}])
        self.commands.extend(floor_flood_commands)

    def reset(self):
        self.objects = dict()
        self.objects_floating = set()
        self.objects_flooded = set()
        self.flood_manager.reset()
        self.recover_command_list = []
        self.commands = []
        self.update_visual_effects()
        self.initialized = False
        self.id_renumbering = dict()
        self.id_list = [0]

    def get_initialization_commands(self) -> List[dict]:
        return [{"$type": "send_bounds"},
                {"$type": "send_transforms"},
                {"$type": "send_rigidbodies"}]

    def on_send(self, resp: List[bytes]) -> None:
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "tran":
                tran = Transforms(resp[i])
                for j in range(tran.get_num()):
                    idx = tran.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].position = tran.get_position(j)
                        self.objects[idx].rotation = tran.get_rotation(j)
                    else:
                        print("Warning: object with id {} not found in FloodObjectManager".format(idx))
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
                        print("Warning: object with id {} not found in FloodObjectManager".format(idx))
                        self.add_object(ObjectStatus(idx, size=boun.get_front(j) + boun.get_right(j) + boun.get_top(
                            j) - boun.get_back(j) - boun.get_left(j) - boun.get_bottom(j)))
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
                        print("Warning: object with id {} not found in FloodObjectManager".format(idx))
            elif r_id == "rigi":
                rigi = Rigidbodies(resp[i])
                for j in range(rigi.get_num()):
                    idx = rigi.get_id(j)
                    if idx in self.objects:
                        self.objects[idx].velocity = rigi.get_velocity(j)
                    else:
                        print("Warning: object with id {} not found in FloodObjectManager".format(idx))
                        self.add_object(ObjectStatus(idx, velocity=rigi.get_velocity(j)))
        for i in range(len(resp)-1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "segm":
                segm = SegmentationColors(resp[i])
                self.segm.process(segm, id_renumbering=self.id_renumbering)
            elif r_id == "rseg":
                segm = ReplicantSegmentationColors(resp[i])
                self.segm.process(segm, id_renumbering=self.id_renumbering)
        self.update_visual_effects()

    def query_height_diff(self, position):
        # TODO different height diff
        return self.flood_manager.height_diff

    def add_object(self, obj: ObjectStatus):
        self.objects[obj.idx] = obj
        if obj.idx not in self.id_renumbering:
            self.id_list.append(obj.idx)
            self.id_renumbering[obj.idx] = len(self.id_list) - 1

    def add_flood(self, id, position, scale, direction):
        self.flood_manager.add_floor_flood(id, position, scale, direction)

    def remove_object(self, idx: int):
        if idx in self.objects:
            del self.objects[idx]

    def query_point_underwater(self, point: np.ndarray) -> float:
        return self.flood_manager.query_point_underwater(point)

    def query_point_flood_height(self, point: np.ndarray) -> float:
        return self.flood_manager.query_point_flood_height(point)

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
