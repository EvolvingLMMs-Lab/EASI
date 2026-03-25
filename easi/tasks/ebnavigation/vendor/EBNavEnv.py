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
from PIL import Image, ImageDraw

# Objects eligible for bounding box drawing
# Reference: EmbodiedBench/embodiedbench/envs/eb_navigation/utils.py
VALID_OBJS = {
    'Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD',
    'Pan', 'Watch', 'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone',
    'Kettle', 'Mug', 'StoveBurner', 'Bowl', 'Spoon', 'TissueBox', 'Apple',
    'TennisRacket', 'SoapBar', 'Cloth', 'Plunger', 'FloorLamp',
    'ToiletPaperHanger', 'Spatula', 'Plate', 'Glassbottle', 'Knife', 'Tomato',
    'ButterKnife', 'Dresser', 'Microwave', 'GarbageCan', 'WateringCan',
    'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Newspaper',
    'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock', 'ToiletPaper',
    'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle',
    'Pencil', 'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup',
    'SaltShaker', 'PepperShaker', 'Pillow', 'Bathtub', 'SoapBottle', 'Statue',
    'Fridge', 'Toaster', 'LaundryHamper',
}


def _draw_boxes(frame: np.ndarray, instance_detections: dict) -> np.ndarray:
    """Draw bounding boxes on frame for all valid objects.

    Reference: EmbodiedBench/embodiedbench/envs/eb_navigation/utils.py:draw_boxes
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for class_name, box in instance_detections.items():
        if class_name.split('|')[0] in VALID_OBJS:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    return np.array(img)

class EBNavEnv:
    """AI2-THOR navigation environment for EB-Navigation benchmark."""

    def __init__(
        self,
        resolution: int = 500,
        fov: int = 100,
        max_steps: int = 20,
        success_threshold: float = 1.0,
        grid_size: float = 0.1,
        visibility_distance: float = 10.0,
        boundingbox: bool = False,
    ):
        self.resolution = resolution
        self._success_threshold = success_threshold
        self.boundingbox = boundingbox
        self.config = {
            "agentMode": "default",
            "gridSize": grid_size,
            "visibilityDistance": visibility_distance,
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
        return {"head_rgb": self._get_frame()}

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

        obs = {"head_rgb": self._get_frame()}
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
        success = float(dist <= self._success_threshold)
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

    def _get_frame(self) -> np.ndarray:
        """Return the current frame, with bounding boxes if enabled."""
        frame = self.env.last_event.frame
        if self.boundingbox:
            detections = self.env.last_event.instance_detections2D
            if detections:
                frame = _draw_boxes(frame, detections)
        return frame

    def close(self) -> None:
        """Shut down the AI2-THOR controller."""
        self.env.stop()
