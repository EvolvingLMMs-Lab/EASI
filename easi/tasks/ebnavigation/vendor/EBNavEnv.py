"""Vendored EB-Navigation environment for EASI.

Adapted from EmbodiedBench/embodiedbench/envs/eb_navigation/EBNavEnv.py.
Changes: removed internal dataset loading, gym dependency, logging/saving.
EASI provides episodes via reset(episode) and the bridge handles image saving.
"""
from __future__ import annotations

import math

import ai2thor.controller
import numpy as np
from ai2thor.platform import Linux64

SUCCESS_THRESHOLD = 1  # meters (horizontal distance)


class EBNavEnv:
    """AI2-THOR navigation environment for EB-Navigation benchmark."""

    def __init__(self, resolution: int = 500, fov: int = 100, max_steps: int = 20):
        self.resolution = resolution
        self.config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "width": resolution,
            "height": resolution,
            "fieldOfView": fov,
            "platform": Linux64,
        }
        self.env = ai2thor.controller.Controller(**self.config)
        self._max_episode_steps = max_steps
        self._current_step = 0
        self.episode_data = None
        self._last_event = None

    def reset(self, episode: dict) -> dict:
        """Reset environment with episode data from EASI.

        Args:
            episode: Dict with keys: scene, agent_pose, target_object_id,
                     target_position, instruction.
        """
        self.episode_data = episode
        scene_name = episode["scene"]

        self._last_event = self.env.reset(scene=scene_name)

        # Teleport agent to starting pose
        pose = episode["agent_pose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"],
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0,
            },
            horizon=pose["horizon"],
            standing=True,
        )

        self._current_step = 0
        return {"head_rgb": self.env.last_event.frame}

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        """Execute one discrete action.

        Args:
            action_id: Integer 0-7 mapping to DISCRETE_ACTIONS.

        Returns:
            (obs, reward, done, info) tuple.
        """
        self._current_step += 1

        if not isinstance(action_id, int) or action_id < 0 or action_id > 7:
            action_id = np.random.randint(8)

        self._execute_action(action_id)
        reward, distance = self.measure_success()

        done = self._current_step >= self._max_episode_steps or reward > 0

        obs = {"head_rgb": self.env.last_event.frame}
        info = {
            "task_success": reward,
            "distance": distance,
            "env_feedback": self._get_env_feedback(),
            "last_action_success": float(
                self.env.last_event.metadata["lastActionSuccess"]
            ),
            "env_step": self._current_step,
            "action_id": action_id,
        }
        return obs, reward, done, info

    def _execute_action(self, action_id: int) -> None:
        """Map discrete action ID to AI2-THOR controller action.

        Reference: EBNavEnv.discrete_action_mapper()
        """
        if action_id == 0:
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif action_id == 1:
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif action_id == 2:
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif action_id == 3:
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif action_id == 4:
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_id == 5:
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_id == 6:
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_id == 7:
            self._last_event = self.env.step(action="LookDown", degrees=30)

    def measure_success(self) -> tuple[float, float]:
        """Compute success (1.0/0.0) and horizontal distance to target.

        Reference: EBNavEnv.measure_success() -- uses XZ-plane distance.
        """
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_position = self.episode_data["target_position"]

        dist = math.sqrt(
            (agent_position["x"] - target_position["x"]) ** 2
            + (agent_position["z"] - target_position["z"]) ** 2
        )
        success = float(dist <= SUCCESS_THRESHOLD)
        return success, dist

    def _get_env_feedback(self) -> str:
        """Build feedback string from last event.

        Reference: EBNavEnv.get_env_feedback() with verbosity=0.
        """
        event = self._last_event
        last_action = event.metadata.get("lastAction", "")
        success = event.metadata.get("lastActionSuccess", False)
        error = event.metadata.get("errorMessage", "")

        if success:
            return f"Last action {last_action} executed successfully."
        else:
            return f"Last action {last_action} is invalid. {error}"

    def close(self) -> None:
        """Shut down the AI2-THOR controller."""
        self.env.stop()
