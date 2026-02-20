"""
constants and hyperparameters here
"""
import numpy as np

from enum import Enum
class ObjectState(Enum):
    NORMAL = 0
    BURNING = 1
    BURNT = 2
    START_BURNING = 3
    STOP_BURNING = 4

class Constants:
    def __init__(self,
                    ROOM_TEMPERATURE: float = 20.0,
                    FIRE_TEMPERATURE: float = 600.0,
                    CHOUKA_INITIAL_PROB: int = 0.02, 
                    CHOUKA_THRESHOLD_PROB_INCREASE: int = 30,
                    CHOUKA_THRESHOLD_UPPER: int = 50,
                    FIRE_SPREAD_SIZE: np.ndarray = np.array([0.2, 0.01, 0.2]),
                    FIRE_VISUAL_SIZE: np.ndarray = np.array([0.6, 0.6, 0.6]),
                    FIRE_INIT_SCALE: float = 0.1,
                    FIRE_SCALE_STEP: float = 0.05,
                    FIRE_FINAL_SCALE: float = 0.8,
                    EXTINGUISH_RADIUS: float = 0.5,
                    EXTINGUISH_TIME: int = 1000
                 ) -> None:

        self.ROOM_TEMPERATURE = ROOM_TEMPERATURE
        self.FIRE_TEMPERATURE = FIRE_TEMPERATURE

        self.CHOUKA_INITITIAL_PROB = CHOUKA_INITIAL_PROB
        self.CHOUKA_THRESHOLD_PROB_INCREASE = CHOUKA_THRESHOLD_PROB_INCREASE
        self.CHOUKA_THRESHOLD_UPPER = CHOUKA_THRESHOLD_UPPER

        self.FIRE_SPREAD_SIZE = FIRE_SPREAD_SIZE # The spread distance of fire. 
        self.FIRE_VISUAL_SIZE = FIRE_VISUAL_SIZE

        self.FIRE_INIT_SCALE = FIRE_INIT_SCALE
        self.FIRE_SCALE_STEP = FIRE_SCALE_STEP
        self.FIRE_FINAL_SCALE = FIRE_FINAL_SCALE

        self.EXTINGUISH_RADIUS = EXTINGUISH_RADIUS
        self.EXTINGUISH_TIME = EXTINGUISH_TIME

default_const = Constants()