import pdb
from typing import Dict, Set, List, Optional
from .utils import *
import numpy as np
import math
from tdw.scene_data.scene_bounds import SceneBounds
import random


class ObjectStatus:
    def __init__(self, idx, constants: Constants = default_const,
                 waterproof: bool = False, has_buoyancy: bool = False, state: int = None,
                 position: np.ndarray = None, rotation: np.ndarray = None,
                 size: np.ndarray = None,
                 velocity: np.ndarray = np.array([0, 0, 0])):
        self.idx: int = idx
        self.waterproof: bool = waterproof
        self.has_buoyancy: bool = has_buoyancy
        self.state: int = (ObjectState.NORMAL if state is None else state)
        self.position: np.ndarray = position
        self.rotation: np.ndarray = rotation
        self.size: np.ndarray = size
        self.velocity: Optional[np.ndarray] = velocity
        self.constants: Constants = constants
        self.name = "Object"
        self.prev_height_under_water = 0.0

    def in_water(self):
        if self.state == ObjectState.NORMAL:
            if self.waterproof and self.has_buoyancy:
                self.state = ObjectState.FLOATING
            elif self.waterproof and not self.has_buoyancy:
                self.state = ObjectState.NORMAL
            elif not self.waterproof and self.has_buoyancy:
                self.state = ObjectState.FLOODED_FLOATING
            else:
                self.state = ObjectState.FLOODED

    def out_of_water(self):
        if self.state == ObjectState.FLOODED_FLOATING:
            self.state = ObjectState.FLOODED
        elif self.state == ObjectState.FLOATING:
            self.state = ObjectState.NORMAL

    def center(self):
        return self.position + self.size * np.array([0, 0.5, 0])

    def bottom(self):
        return self.position

    def top(self):
        return self.position + self.size * np.array([1, 1, 1])

    def left(self):
        return self.position + self.size * np.array([-0.5, 0.5, 0])

    def right(self):
        return self.position + self.size * np.array([0.5, 0.5, 0])

    def front(self):
        return self.position + self.size * np.array([0, 0.5, 0.5])

    def back(self):
        return self.position + self.size * np.array([0, 0.5, -0.5])

    def area(self):
        return self.size[0] * self.size[2]

    def horizontal_area(self):
        return self.size[1] * self.size[2]

class AgentStatus(ObjectStatus):
    def __init__(self, idx, position, size, constants=default_const):
        super().__init__(idx, constants=constants, waterproof = True, has_buoyancy = False,
                         state=ObjectState.NORMAL, position = position, size=size)
        self.name = "Agent"

class FloodManager:
    def __init__(self, ascending_speed: float = 0.01,
                 ascending_interval: int = 1,
                 source_position: np.ndarray = None,
                 source_from: str = "x_max",
                 max_height: float = 1.5,
                 floor_ids: List[int] = [],
                 floor_positions: List[np.ndarray] = [],
                 floor_sizes: List[np.ndarray] = [],
                 floor_directions: List[np.ndarray] = [],
                 roll_theta: float = 10,
                 flood_density: float = 1.0,
                 drag_coefficient: float = 0.47,
                 flood_force_scale: float = 1.0,
                 ):
        self.ascending_speed = ascending_speed
        self.ascending_interval = ascending_interval
        self.max_height = max_height
        self.source_position = source_position
        self.source_from = source_from
        self.floor_ids = floor_ids
        self.floor_positions = floor_positions
        self.floor_sizes = floor_sizes
        self.floor_directions = floor_directions
        self.floor_flood_angles = [
            {
                "w": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            } for i in range(len(self.floor_positions))
        ]
        self.old_pitch_theta = 0
        self.old_roll_theta = 0
        self.original_roll_theta = roll_theta
        self.roll_theta = roll_theta
        self.fluid_velocity = self.ascending_speed / self.ascending_interval / math.tan(abs(roll_theta / 180) * math.pi)
        self.floor_flood_heights = [0.0 for i in range(len(self.floor_positions))]
        self.source_height = 0
        self.ascending_counter = 0
        self.height_diff = 0
        self.flood_density = flood_density
        self.drag_coefficient = drag_coefficient
        self.flood_force_scale = flood_force_scale

    def reset(self):
        self.roll_theta = self.original_roll_theta
        self.source_height = 0

    def add_floor_flood(self, id, position, scale, direction):
        self.floor_ids.append(id)
        self.floor_positions.append(position)
        self.floor_sizes.append(scale)
        self.floor_directions.append(direction)

    def get_updated_flood_angles_and_heights(self):
        flood_effect_dict = {}
        for i, floor_id in enumerate(self.floor_ids):
            flood_effect_dict[floor_id] = {}

        # cal heights
        for i in range(len(self.floor_ids)):
            floor_position_for_calculation = self.floor_positions[i]["x"] if self.source_from in ['x_max', 'x_min'] \
                                                    else self.floor_positions[i]["z"]
            distance_to_source = abs(self.source_location_for_calculation - floor_position_for_calculation)
            height_diff_to_source = math.tan(abs(self.roll_theta/180)*math.pi) * distance_to_source
            self.floor_positions[i]["y"] = self.source_height - height_diff_to_source
            flood_effect_dict[self.floor_ids[i]]["positions"] = {
                "x": self.floor_positions[i]["x"],
                "y": self.floor_positions[i]["y"],
                "z": self.floor_positions[i]["z"]
            }

        for i in range(len(self.floor_ids)):
            if self.source_from == 'x_max':
                flood_effect_dict[self.floor_ids[i]]["angles"] = {
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": self.roll_theta
                }
                flood_effect_dict[self.floor_ids[i]]["scales"] = {
                    "x": math.cos(abs(self.old_roll_theta / 180) * math.pi) / math.cos(
                        abs(self.roll_theta / 180) * math.pi), # * self.floor_sizes[i][0],
                    "y": 1.0,
                    "z": 1.0, # * self.floor_sizes[i][2]
                }
            elif self.source_from == 'x_min':
                flood_effect_dict[self.floor_ids[i]]["angles"] = {
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": -self.roll_theta
                }
                flood_effect_dict[self.floor_ids[i]]["scales"] = {
                    "x": math.cos(abs(self.old_roll_theta / 180) * math.pi) / math.cos(
                        abs(self.roll_theta / 180) * math.pi) * self.floor_sizes[i][0],
                    "y": 1.0,
                    "z": 1.0 * self.floor_sizes[i][2]
                }
            elif self.source_from == 'z_max':
                flood_effect_dict[self.floor_ids[i]]["angles"] = {
                    "pitch": self.roll_theta,
                    "yaw": 0.0,
                    "roll": 0.0
                }
                flood_effect_dict[self.floor_ids[i]]["scales"] = {
                    "x": 1.0 * self.floor_sizes[i][0],
                    "y": 1.0,
                    "z": math.cos(abs(self.old_roll_theta / 180) * math.pi) / math.cos(
                        abs(self.roll_theta / 180) * math.pi) * self.floor_sizes[i][2]
                }
            elif self.source_from == 'z_min':
                flood_effect_dict[self.floor_ids[i]]["angles"] = {
                    "pitch": -self.roll_theta,
                    "yaw": 0.0,
                    "roll": 0.0
                }
                flood_effect_dict[self.floor_ids[i]]["scales"] = {
                    "x": 1.0 * self.floor_sizes[i][0],
                    "y": 1.0,
                    "z": math.cos(abs(self.old_roll_theta / 180) * math.pi) / math.cos(
                        abs(self.roll_theta / 180) * math.pi) * self.floor_sizes[i][2]
                }
            else:
                assert False
        self.old_roll_theta = self.roll_theta
        return flood_effect_dict

    def evolve(self):
        self.ascending_counter += 1
        self.height_diff = 0
        if self.ascending_counter % self.ascending_interval == 0:
            self.ascending_counter = 0
            self.height_diff += (self.ascending_speed * random.random() * 2.0)
            self.source_height += self.ascending_speed
            if self.source_height >= self.max_height:
                self.height_diff -= (self.source_height - self.max_height)
                self.source_height = self.max_height
                if self.roll_theta > 0:
                    old_length = self.source_height / math.tan(abs(self.roll_theta / 180) * math.pi)
                    new_length = old_length + self.fluid_velocity * self.ascending_interval
                    self.roll_theta = math.atan(self.source_height / new_length) * 180 / math.pi
                elif self.roll_theta < 0:
                    old_length = self.source_height / math.tan(abs(self.roll_theta / 180) * math.pi)
                    new_length = old_length + self.fluid_velocity * self.ascending_interval
                    self.roll_theta = -math.atan(self.source_height / new_length) * 180 / math.pi
        return self.get_updated_flood_angles_and_heights()

    def query_point_underwater(self, point: np.ndarray):
        floor_position_for_calculation = point[0] if self.source_from in ['x_max', 'x_min'] else point[2]
        distance_to_source = abs(self.source_location_for_calculation - floor_position_for_calculation)
        height_diff_to_source = math.tan(abs(self.roll_theta / 180) * math.pi) * distance_to_source
        flood_height = self.source_height - height_diff_to_source
        return point[1] <= flood_height

    def query_height_beneath_water(self, object: ObjectStatus):
        floor_position_for_calculation = object.bottom()[0] if self.source_from in ['x_max', 'x_min'] else object.bottom()[2]
        distance_to_source = abs(self.source_location_for_calculation - floor_position_for_calculation)
        height_diff_to_source = math.tan(abs(self.roll_theta / 180) * math.pi) * distance_to_source
        flood_height = self.source_height - height_diff_to_source
        return max(min(flood_height - min(object.bottom()[1], object.top()[1]), object.size[1]), 0)

    def update_object_status_old(self, object: ObjectStatus):
        old_state = object.state
        # Use top
        # if object.top()[1] <= self.source_height:
        # Use center
        if self.query_point_underwater(object.center()):
        # Use bottom
        # if object.bottom()[1] <= self.source_height:
        # Debug
        # if object.bottom()[1] < 100:
            object.in_water()
        else:
            object.out_of_water()
        start_floating = False
        stop_floating = False
        if object.state == ObjectState.FLOODED_FLOATING or object.state == ObjectState.FLOATING:
            if old_state != ObjectState.FLOODED_FLOATING and old_state != ObjectState.FLOATING:
                start_floating = True
        else:
            if old_state == ObjectState.FLOODED_FLOATING or old_state == ObjectState.FLOATING:
                stop_floating = True
        return object, start_floating, stop_floating

    def update_object_status_new(self, object: ObjectStatus, clip_buoyancy: bool = True):
        # Use top
        # if object.top()[1] <= self.source_height:
        # Use center
        height_under_water = self.query_height_beneath_water(object)
        if height_under_water > 0: # start flooding
        # if height_under_water > object.size[1]: # Completely flooded
            object.in_water()
        else:
            object.out_of_water()
        # if clip_buoyancy:
        #     height_diff = height_under_water - object.prev_height_under_water
        #     height_diff = min(0.2*object.size[1], height_diff)
        #     height_under_water = height_diff + object.prev_height_under_water
        # buoyancy_scale = self.flood_density / 1000 * height_under_water * abs(object.area()) * 9.81
        buoyancy_scale = 4.0 * self.flood_density / 1000 * height_under_water * abs(object.area()) * 9.81
        object.prev_height_under_water = height_under_water
        return object, buoyancy_scale

    def cal_horizontal_force(self, source, object: ObjectStatus):
        if self.query_height_beneath_water(object) <= 0:
            return {"x": 0, "y": 0, "z": 0}
        fluid_velocity = np.array([-self.fluid_velocity * self.drag_coefficient, 0, 0])
        velocity_diff = fluid_velocity - object.velocity
        # Drag force 1/2 * fluid density * drag area * drag coefficient * velocity^2
        # drag_force_scale = 0.5 * self.flood_density / 1000 * abs(object.horizontal_area()) * np.linalg.norm(velocity_diff)
        drag_force_scale = 2.0 * self.flood_density / 1000 * abs(object.horizontal_area()) * np.linalg.norm(velocity_diff)
        drag_force_direction = velocity_diff / np.linalg.norm(velocity_diff)
        drag_force = drag_force_scale * drag_force_direction

        # flood force
        # distance = object.center() - source
        # direction = distance / np.linalg.norm(distance)
        # flood_force = direction * self.flood_force_scale * object.horizontal_area()
        return {"x": drag_force[0], "y": drag_force[1], "z": drag_force[2]}

    def set_scene_bounds(self, resp=None):
        self.scene_bounds = SceneBounds(resp=resp)
        new_regions = []
        for fp in self.floor_positions:
            for region in self.scene_bounds.regions:
                position_diff = abs(region.center[0] - fp['x']) + abs(region.center[2] - fp['z'])
                if position_diff < 0.3:
                    new_regions.append(region)
        assert len(new_regions) == len(self.floor_positions)
        self.scene_bounds.regions = new_regions
        if len(new_regions) > 0:
            self.x_max = max([region.x_max for region in self.scene_bounds.regions])
            self.x_min = min([region.x_min for region in self.scene_bounds.regions])
            self.z_max = max([region.z_max for region in self.scene_bounds.regions])
            self.z_min = min([region.z_min for region in self.scene_bounds.regions])
        else:
            self.x_max = 0
            self.x_min = 0
            self.z_max = 0
            self.z_min = 0
        if self.source_from == "x_max":
            self.source_location_for_calculation = self.x_max
        elif self.source_from == "x_min":
            self.source_location_for_calculation = self.x_min
        elif self.source_from == "z_max":
            self.source_location_for_calculation = self.z_max
        elif self.source_from == "z_min":
            self.source_location_for_calculation = self.z_min
        else:
            assert False

    def query_point_flood_height(self, target: np.ndarray):
        floor_position_for_calculation = target[0] if self.source_from in ['x_max', 'x_min'] else target[2]
        distance_to_source = abs(self.source_location_for_calculation - floor_position_for_calculation)
        height_diff_to_source = math.tan(abs(self.roll_theta / 180) * math.pi) * distance_to_source
        flood_height = max(self.source_height - height_diff_to_source, 0)
        return flood_height
