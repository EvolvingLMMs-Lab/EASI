from tdw.object_data.object_static import ObjectStatic
from tdw.object_data.bound import Bound
from tdw.object_data.rigidbody import Rigidbody
import numpy as np

class Constants:
    def __init__(self,
                 AIR_DENSITY: float = 1.225,
                 F_CROSS_SCALE: float = 0.05,
                 F_ON_AGENT: float = 1.0
    ) -> None:
        self.AIR_DENSITY = AIR_DENSITY
        self.F_CROSS_SCALE = F_CROSS_SCALE
        self.F_ON_AGENT = F_ON_AGENT
default_const = Constants()
