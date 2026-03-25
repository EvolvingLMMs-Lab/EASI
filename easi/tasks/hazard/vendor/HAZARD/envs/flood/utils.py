"""
constants and hyperparameters here
"""
import numpy as np

from enum import Enum
class ObjectState(Enum):
    NORMAL = 0
    FLOODED = 1
    FLOODED_FLOATING = 2
    FLOATING = 3
    FLOODED_PICKED = 4

class Constants:
    def __init__(self,
                    HAS_BUOYANCY: bool = False,
                    WATER_PROOF: bool = False,
                    # ASCENDING_SPEED: float = 0.05,
                    ASCENDING_SPEED: float = 0.01,
                    ASCENDING_INTERVAL: int = 10,
                    MAX_HEIGHT: float = 1.5,
                    FLOOD_DENSITY: float = 1000.0,
                    # FLOOD_FORCE_SCALE: float = 3.0,
                    FLOOD_FORCE_SCALE: float = 5.0,
                    SLOP_ANGLE: float = 3.0,
                    # DRAG_COEFFICIENT: float = 0.47
                    # DRAG_COEFFICIENT: float = 4 # Coefficient of the drag force by the water
                    DRAG_COEFFICIENT: float = 50,
                    AGENT_DRAG_COEFFICIENT: float = 1.0
                 ) -> None:

        self.HAS_BUOYANCY = HAS_BUOYANCY
        self.WATER_PROOF = WATER_PROOF
        self.ASCENDING_SPEED = ASCENDING_SPEED
        self.ASCENDING_INTERVAL = ASCENDING_INTERVAL
        self.MAX_HEIGHT = MAX_HEIGHT
        self.FLOOD_DENSITY = FLOOD_DENSITY
        self.FLOOD_FORCE_SCALE = FLOOD_FORCE_SCALE
        self.DRAG_COEFFICIENT = DRAG_COEFFICIENT
        self.SLOP_ANGLE = SLOP_ANGLE
        self.AGENT_DRAG_COEFFICIENT = AGENT_DRAG_COEFFICIENT

default_const = Constants()
