from typing import List, Union
from tdw.controller import Controller
from .manager import WindObjectManager
from .object import AgentStatus
from .wind_utils import default_const

import numpy as np

"""
This controller controls the spread of fire.
"""
class WindController(Controller):
    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, seed = 0, constants = default_const, **kwargs):
        self.initialized = False
        self.commands: List[dict] = list()
        super().__init__(port, check_version, launch_build)
        
        self.manager = WindObjectManager()
        self.add_ons.append(self.manager)
        self.frame_count = 0
        self.RNG = np.random.Generator(np.random.PCG64(seed))
        self.constants = constants

    def seed(self, seed):
        self.RNG = np.random.Generator(np.random.PCG64(seed))
    
    def get_unique_id(self):
        while True:
            idx = super().get_unique_id()
            if idx not in self.manager.objects:
                return idx

    def add_agent(self, idx, pos):
        self.manager.add_object(AgentStatus(idx, constants=self.constants, position=pos, size=None))
    
    def set_wind(self, wind_v):
        self.manager.wind_v = wind_v
    
    def reset_wind(self):
        self.manager.wind_v = np.array([0, 0, 0])
    
    def wind_step(self, resp):
        for idx in self.manager.effects:
            force, torque = self.manager.effects[idx]
            # print(idx, force, torque, self.manager.objects[idx].position, self.manager.objects[idx].velocity)
            self.commands.append({"$type": "apply_force_to_object", "id": idx, "force": {"x": force[0], "y": force[1], "z": force[2]}})
            self.commands.append({"$type": "apply_torque_to_object", "id": idx, "torque": {"x": torque[0], "y": torque[1], "z": torque[2]}})
        self.frame_count += 1

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        if isinstance(commands, dict):
            commands = [commands]
        commands.extend(self.commands)
        self.commands.clear()
        # for com in commands:
        #     print(com)
        # try:
        resp = super().communicate(commands)
        # except:
        #     print("Error")
        #     print(commands)
        #     super().communicate([{"$type": "terminate"}])
        #     exit(0)
        if self.initialized:
            self.wind_step(resp)
        return resp