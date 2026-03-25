from typing import Dict, Set
from .fire_utils import *
import numpy as np

class ObjectStatus:
    def __init__(self, idx, constants: Constants=default_const,
                 inflammable: bool = True, state: int = None,
                 burning_time: int = 300, burning_time_left: int = 0,
                 temperature: float = 20.0, temperature_threshold: float = 200.0,
                 is_heat_source: bool = False, position: np.ndarray = None, rotation: np.ndarray = None, size: np.ndarray = None):
        self.idx: int = idx
        self.inflammable: bool = inflammable
        self.state: int = (ObjectState.NORMAL if state is None else state)
        self.burning_time: int = burning_time # if set to -1, burn indefinitely
        self.burning_time_left: int = burning_time_left # if set to -1, burn indefinitely
        self.temperature: float = temperature
        self.temperature_threshold: float = temperature_threshold
        self.is_heat_source: bool = is_heat_source
        self.position: np.ndarray = position
        self.rotation: np.ndarray = rotation
        self.size: np.ndarray = size
        self.constants: Constants = constants
        self.name = "Object"
    
    def step(self):
        if self.state == ObjectState.BURNING:
            if self.burning_time_left == -1:
                return ObjectState.BURNING
            self.burning_time_left -= 1
            if self.burning_time_left <= 0:
                self.state = ObjectState.BURNT
                self.inflammable = False
                return ObjectState.STOP_BURNING
            return ObjectState.BURNING
        
        if self.state == ObjectState.BURNT:
            return ObjectState.BURNT
        
        if self.temperature > self.temperature_threshold and self.inflammable:
            self.state = ObjectState.BURNING
            self.burning_time_left = self.burning_time
            self.is_heat_source = True
            self.temperature = self.constants.FIRE_TEMPERATURE
            return ObjectState.START_BURNING
    
    def center(self): return self.position + self.size * np.array([0, 0.5, 0])
    def bottom(self): return self.position
    def top(self): return self.position + self.size * np.array([1, 1, 1])
    def left(self): return self.position + self.size * np.array([-0.5, 0.5, 0])
    def right(self): return self.position + self.size * np.array([0.5, 0.5, 0])
    def front(self): return self.position + self.size * np.array([0, 0.5, 0.5])
    def back(self): return self.position + self.size * np.array([0, 0.5, -0.5])
class FireStatus(ObjectStatus):
    def __init__(self, idx, position, size, constants=default_const):
        super().__init__(idx, constants, inflammable=False, state=ObjectState.BURNING,
                         burning_time=-1, burning_time_left=-1,
                         temperature=constants.FIRE_TEMPERATURE, temperature_threshold=1000.0,
                         is_heat_source=True, position=position, size=size)
        self.name = "Fire"

class AgentStatus(ObjectStatus):
    def __init__(self, idx, position, size, endurance: float = 60.0, constants=default_const):
        super().__init__(idx, constants, inflammable=False, state=ObjectState.NORMAL,
                         burning_time=-1, burning_time_left=-1,
                         temperature=constants.ROOM_TEMPERATURE, temperature_threshold=endurance,
                         is_heat_source=False, position=position, size=size)
        self.name = "Agent"

"""
Assumed model:

For heat sources, the temperature is constant.
For other objects:

T = T0 * (1 - decay_rate) + T1 * decay_rate
T0 is old temperature
T1 is average temperature of nearby objects, weighted by exp( - max(1, distance / effective_distance) )
room temperature has a weight of exp(-1)

T1 is average temperature of nearby objects, weighted by (effective_distance/distance)^2
room temperature has a weight of 1
This algorithm is scientific, and a lot more computationally efficient.
"""
class TemperatureManager:
    def __init__(self, room_temperature: float = 20.0,
                    effective_distance: float = 0.5,
                    decay_rate: float = 0.03):
        self.room_temperature = room_temperature
        self.effective_distance = effective_distance
        self.decay_rate = decay_rate

    def evolve(self, objects: Dict[int, ObjectStatus]):
        new_temperatures = dict()
        for idx in objects:
            obj = objects[idx]
            if obj.is_heat_source:
                new_temperatures[idx] = obj.temperature
                continue
            # sum_weight = np.exp(-1)
            sum_weight = 1 # Weight of room temperature
            sum_weighted_temperature = self.room_temperature * sum_weight
            for idx2 in objects:
                if idx2 == idx:
                    continue
                distance = np.linalg.norm(objects[idx2].position - obj.position)
                distance = max(distance, 0.1)
                # weight = np.exp(-max(10000, distance / self.effective_distance))
                weight = (self.effective_distance/distance)**2
                sum_weighted_temperature += objects[idx2].temperature * weight
                sum_weight += weight
            # print(sum_weighted_temperature / sum_weight)
            new_temperatures[idx] = obj.temperature * (1 - self.decay_rate) + sum_weighted_temperature / sum_weight * self.decay_rate
        return new_temperatures
    
    def query_point_temperature(self, target: np.ndarray, objects: Dict[int, ObjectStatus]):
        sum_weight = np.exp(-1)
        sum_weighted_temperature = self.room_temperature * sum_weight
        for idx in objects:
            distance = np.linalg.norm(objects[idx].position - target)
            distance = max(distance, 0.1)
            weight = (self.effective_distance/distance)**2
            sum_weighted_temperature += objects[idx].temperature * weight
            sum_weight += weight
        return sum_weighted_temperature / sum_weight