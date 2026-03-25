from typing import Dict, Set, Optional
from .wind_utils import *
import numpy as np

class ObjectStatus:
    def __init__(self, idx, constants: Constants=default_const,
                 mass=1.0, position: np.ndarray = None, rotation: np.ndarray = None,
                 size: np.ndarray = None, velocity: np.ndarray = None, resistence: int = 0):
        self.idx: int = idx
        self.position: Optional[np.ndarray] = position
        self.rotation: Optional[np.ndarray] = rotation
        self.size: Optional[np.ndarray] = size
        self.velocity: Optional[np.ndarray] = velocity
        self.constants: Constants = constants
        self.resistence: int = resistence
        self.name = "Object"
    
    def center(self): return self.position + self.size * np.array([0, 0.5, 0])
    def bottom(self): return self.position
    def top(self): return self.position + self.size * np.array([0, 1, 0])
    def left(self): return self.position + self.size * np.array([-0.5, 0.5, 0])
    def right(self): return self.position + self.size * np.array([0.5, 0.5, 0])
    def front(self): return self.position + self.size * np.array([0, 0.5, 0.5])
    def back(self): return self.position + self.size * np.array([0, 0.5, -0.5])
    def area(self): return (self.top() - self.bottom()).sum() * (self.right() - self.left()).sum()

class AgentStatus(ObjectStatus):
    def __init__(self, idx, position, size=None, constants=default_const):
        super().__init__(idx, constants, position=position, size=size)
        self.name = "Agent"

class WindForceManager:
    def __init__(self, constants=default_const):
        self.constants: Constants = constants
    
    def calc_wind_effect(self, wind_velocity: np.ndarray, obj: ObjectStatus):
        """
        Calculate the wind force and torque on an object
        v_effect: velocity difference on the wind direction between the wind and the object
        """
        if np.dot(wind_velocity, wind_velocity) < 1e-6 or obj.resistence > 10:
            return np.zeros(3), np.zeros(3)
        felt_wind = wind_velocity / (obj.resistence + 1)
        # print(obj.idx, obj.resistence)
        v_effect = felt_wind - felt_wind * np.dot(felt_wind, obj.velocity) / np.dot(felt_wind, felt_wind)
        # print(f"wind= {felt_wind}, v= {obj.velocity}, v_effect={v_effect}")
        area = obj.area()
        
        f_tan = np.linalg.norm(v_effect) * v_effect * area * self.constants.AIR_DENSITY
        
        rand_v = np.minimum(2, np.random.normal(0, self.constants.F_CROSS_SCALE, 3))
        f_cross = np.cross(f_tan, rand_v)
         
        r = np.random.normal(0, 0.1, 3) * (obj.top() - obj.bottom()).sum() / 2
        torque = np.cross(f_tan, r)
        # torque = np.zeros(3)
        
        f = f_tan + f_cross
        return f, torque
    
    def evolve(self, objects: Dict[int, ObjectStatus], wind: np.ndarray, settled: Set[int]):
        effects = dict()
        for idx in objects:
            obj = objects[idx]
            if obj.name != 'Object' or (not isinstance(obj.velocity, np.ndarray)) or (not isinstance(obj.size, np.ndarray)) or idx in settled:
                continue
            effects[idx] = self.calc_wind_effect(wind, obj)
        return effects