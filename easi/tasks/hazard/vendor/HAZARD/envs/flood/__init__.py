from .floodagent_controller import FloodAgentController
from .flood_gym import FloodEnv

from gym.envs.registration import register

register(
    id="flood-v0",
    entry_point="envs.flood.flood_gym:FloodEnv"
)
