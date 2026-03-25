from typing import List, Union, Dict
from .manager import FloodObjectManager
from .utils import *
from .object import AgentStatus
from tdw.controller import Controller
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.obi_data.fluids.fluid import Fluid
from tdw.add_ons.obi import Obi

import numpy as np

class PhysicalFlood:
    def __init__(self, flood_id, activate, position, direction, speed):
        self.fire_id = flood_id
        self.activate = activate
        self.position = position
        self.direction = direction
        self.speed = speed

class FloorFlood:
    def __init__(self, floor_id):
        self.floor_id = floor_id
        self.height = 0
        self.angles = {
            "w": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }

"""
This controller controls the spread of flood.
"""


class FloodController(Controller):
    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, seed=0,
                 constants=default_const, **kwargs):
        self.initialized = False
        self.commands: List[dict] = list()
        super().__init__(port, check_version, launch_build)

        self.manager = FloodObjectManager(source_position=None,
                                          source_from="x_max",
                                          floor_ids=[],
                                          floor_positions=[],
                                          floor_sizes=[],
                                          floor_directions=[],
                                          flood_density=default_const.FLOOD_DENSITY)
        self.add_ons.append(self.manager)

        self.physical_flood_info: Dict[int, PhysicalFlood] = dict()
        self.floor_flood_info: Dict[int, FloorFlood] = dict()
        self.constants = constants

    def init_obi(self):
        self.obi = Obi()
        self.communicate([{"$type": "create_obi_solver"}])
        self.obi.set_solver(solver_id=1, scale_factor=1.0, substeps=1)
        self.add_ons.append(self.obi)
        self.communicate([])
        self.communicate([])

    def seed(self, seed):
        self.RNG = np.random.Generator(np.random.PCG64(seed))

    def get_unique_id(self):
        while True:
            idx = super().get_unique_id()
            if idx not in self.manager.objects:
                return idx

    def add_physical_flood(self, position, direction, speed=1, object_id=None):
        if isinstance(position, np.ndarray):
            position = position.tolist()
        if isinstance(direction, np.ndarray):
            direction = direction.tolist()
        if object_id == None:
            object_id = self.get_unique_id()
        new_fluid_info = PhysicalFlood(object_id, activate=True, position=position, direction=direction, speed=speed)
        self.physical_flood_info[len(self.physical_flood_info)] = new_fluid_info
        self.manager.flood_manager.source_position = np.array(position)
        fluid = Fluid(capacity=100000,
                      resolution=1.0,
                      color={"r": 0.6, "g": 0.6, "b": 0.33, "a": 0.5},
                      rest_density=self.constants.FLOOD_DENSITY,
                      reflection=0.25,
                      refraction=-0.034,
                      smoothing=3.0,
                      render_smoothness=0.8,
                      metalness=0,
                      viscosity=0.001,
                      absorption=5,
                      vorticity=1.0,
                      surface_tension=0.1,
                      transparency=0.2,
                      thickness_cutoff=1.2,
                      radius_scale=1.6,
                      random_velocity=0.0
                      )
        self.obi.create_fluid(fluid=fluid,
                         shape=DiskEmitter(),
                         object_id=object_id,
                         position={"x": position[0], "y": position[1], "z": position[2]},
                         rotation={"x": direction[0], "y": direction[1], "z": direction[2]},
                         speed=speed,
                         lifespan=2
                              )

    def add_agent(self, idx, pos):
        self.manager.add_object(AgentStatus(idx, constants=self.constants, position=pos, size=np.array([1.0, 2.0, 1.0])))

    def flood_step(self, resp):
        pass

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        if isinstance(commands, dict):
            commands = [commands]
        commands.extend(self.commands)
        self.commands.clear()
        # for com in commands:
        #     print(com)
        resp = super().communicate(commands)
        if self.initialized:
            self.flood_step(resp)
        return resp
