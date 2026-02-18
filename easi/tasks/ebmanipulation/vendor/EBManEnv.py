"""Vendored EB-Manipulation environment for EASI.

Adapted from EmbodiedBench/embodiedbench/envs/eb_manipulation/EBManEnv.py.
Changes: removed gymnasium dependency, internal dataset loading, image saving.
EASI provides episodes via reset(episode) and the bridge handles image saving.
Hardcoded constants extracted as constructor parameters.
"""
from __future__ import annotations

import time

import numpy as np
from amsolver.action_modes import ActionMode, ArmActionMode
from amsolver.backend.utils import task_file_to_task_class
from amsolver.environment import Environment
from amsolver.observation_config import ObservationConfig
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
    get_continuous_action_from_discrete,
)
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBManEnv:
    """CoppeliaSim manipulation environment for EB-Manipulation benchmark.

    Unlike the original EBManEnv (gym.Env), this version:
    - Does NOT load datasets internally -- EASI provides episodes via reset()
    - Does NOT inherit from gymnasium -- no action_space/observation_space attrs
    - Accepts configurable scene_bounds, voxel_size, rotation_resolution
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (500, 500),
        max_steps: int = 15,
        scene_bounds: list[float] | None = None,
        voxel_size: int = 100,
        rotation_resolution: int = 3,
    ):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.set_image_size(img_size)

        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
        self.env = Environment(action_mode, obs_config=obs_config, headless=True)
        self.env.launch()

        self._max_episode_steps = max_steps
        self._current_step = 0
        self._episode_start_time = 0.0
        self.task = None
        self.task_class = None
        self.current_task_variation = None
        self.episode_language_instruction = ""
        self.last_frame_obs = None

        # Configurable discretization parameters
        self._scene_bounds = np.array(
            scene_bounds
            if scene_bounds is not None
            else [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        )
        self._voxel_size = voxel_size
        self._rotation_resolution = rotation_resolution

    def reset(self, episode: dict) -> tuple[str, dict]:
        """Reset environment with episode data from EASI.

        Args:
            episode: Dict with keys:
                - task_file: str -- task variation name (e.g., "pick_cube_shape")
                - task_base: str -- path to task_base.ttm
                - waypoint_sets: str -- path to waypoint_sets.ttm
                - configs: str -- path to configs.pkl
        """
        self._current_step = 0
        self._episode_start_time = time.time()

        task_file = episode["task_file"]
        task_class = task_file_to_task_class(task_file, parent_folder="vlm")
        self.current_task_variation = task_file
        self.task_class = task_file.split("_")[0]

        self.task = self.env.get_task(task_class)
        descriptions, obs = self.task.load_config(
            episode["task_base"],
            episode["waypoint_sets"],
            episode["configs"],
        )
        self.episode_language_instruction = descriptions[0]
        self.last_frame_obs = vars(obs)

        return descriptions[0], self.last_frame_obs

    def step(self, discrete_action: list) -> tuple[dict, float, bool, dict]:
        """Execute one 7D discrete action.

        Args:
            discrete_action: [X, Y, Z, Roll, Pitch, Yaw, Gripper] (7 values)

        Returns:
            (obs_dict, reward, done, info) tuple.
        """
        self._current_step += 1
        info = {}
        action_success = False

        try:
            action = get_continuous_action_from_discrete(
                discrete_action,
                scene_bounds=self._scene_bounds,
                voxel_size=self._voxel_size,
                rotation_resolution=self._rotation_resolution,
            )
            obs, reward, terminate = self.task.step(action)

            # Special handling for stack task (EBManEnv.py:175-189)
            if self.current_task_variation.startswith("stack"):
                if terminate:
                    if action[-1] == 0.0:
                        reward = 0.0
                        terminate = False
                        logger.debug(
                            "Wrong success condition for stack, "
                            "setting reward to 0 and terminate to False"
                        )
                    elif action[-1] == 1.0:
                        action[2] += 0.03
                        logger.debug("Checking if the object is stacked properly")
                        obs, reward, terminate = self.task.step(action)
                        if terminate and reward == 1.0:
                            logger.debug("Stacking is successful")
                        else:
                            logger.debug("Stacking is unsuccessful")
                            reward = 0.0
                            terminate = False

            self.last_frame_obs = vars(obs)
            action_success = True
        except Exception as e:
            logger.warning("Action execution error: %s", e)
            obs = self.last_frame_obs
            reward = -1
            terminate = False
            action_success = e

        # Build env feedback
        env_feedback = self._get_env_feedback(action_success, reward)

        info["env_feedback"] = env_feedback
        info["instruction"] = self.episode_language_instruction
        info["env_step"] = self._current_step
        info["episode_elapsed_seconds"] = time.time() - self._episode_start_time
        info["action"] = discrete_action
        info["action_success"] = 1.0 if action_success is True else 0.0
        info["task_success"] = 1.0 if (terminate and reward == 1.0) else 0.0

        if self._current_step >= self._max_episode_steps:
            terminate = True

        return self.last_frame_obs, reward, terminate, info

    def _get_env_feedback(self, action_success, reward) -> str:
        """Generate feedback message for the current step.

        Reference: EBManEnv.get_env_feedback()
        """
        msg = (
            f"You are currently performing the task intended to "
            f"{self.episode_language_instruction.lower()} "
            f"At this moment, you have completed executing "
            f"{self._current_step} steps. "
        )
        if action_success is True:
            msg += "Last action is valid. "
        else:
            msg += f"Last action is invalid. {action_success}."
        msg += f"The current reward obtained is {reward}."
        return msg

    def close(self) -> None:
        """Shut down AMSolver/CoppeliaSim."""
        self.env.shutdown()
