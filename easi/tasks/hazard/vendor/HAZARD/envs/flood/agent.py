from typing import Optional, Union, List
from .utils import *
from tdw.add_ons.replicant import *
from tdw.quaternion_utils import QuaternionUtils
from tdw.replicant.actions.action import Action, ActionStatus
from tdw.replicant.actions.look_at import LookAt
from tdw.replicant.actions.grasp import Grasp
from tdw.replicant.image_frequency import ImageFrequency


import numpy as np

class SequentialAction(Action):
    def __init__(self, actions: List[Action]):
        self._actions = actions
        self._current = 0
        self.status: ActionStatus = None
        self.image_frequency = ImageFrequency.once
        super().__init__()

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        self.image_frequency = image_frequency
        self._current = 0

        self.status = ActionStatus.ongoing
        self.initialized = True

        return super().get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                   image_frequency=image_frequency)

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        commands = super().get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        # for convenience, if self._current < 0, it is in the middle of two actions, wait one frame
        if not self._actions[self._current].initialized:
            # The action's status defaults to `ongoing`, but actions sometimes fail prior to initialization.
            if self._actions[self._current].status == ActionStatus.ongoing:
                # Initialize the action and get initialization commands.
                self._actions[self._current].initialized = True
                initialization_commands = self._actions[self._current].get_initialization_commands(resp=resp,
                                                                                                   static=static,
                                                                                                   dynamic=dynamic,
                                                                                                   image_frequency=self.image_frequency)

                # Most actions are `ongoing` after initialization, but they might've succeeded or failed already.
                if self._actions[self._current].status == ActionStatus.ongoing:
                    commands.extend(initialization_commands)
                else:
                    commands.extend(self._actions[self._current].get_end_commands(resp=resp,
                                                                                  static=static,
                                                                                  dynamic=dynamic,
                                                                                  image_frequency=self.image_frequency))
        # Continue an ongoing action.
        else:
            # Get the ongoing action commands.
            action_commands = self._actions[self._current].get_ongoing_commands(resp=resp,
                                                                                static=static,
                                                                                dynamic=dynamic)
            # This is an ongoing action. Append ongoing commands.
            if self._actions[self._current].status == ActionStatus.ongoing:
                commands.extend(action_commands)
            # This action is done. Append end commands.
            else:
                commands.extend(self._actions[self._current].get_end_commands(resp=resp,
                                                                              static=static,
                                                                              dynamic=dynamic,
                                                                              image_frequency=self.image_frequency))
        # This action ended. If not successful, all commands are done. Otherwise, continue to the next action.
        if self._actions[self._current].status != ActionStatus.ongoing:
            if self._actions[self._current].status == ActionStatus.success:
                self._current += 1
                # If there are more actions, continue.
                if self._current < len(self._actions):
                    self.status = ActionStatus.ongoing
                # Otherwise, this action is done.
                else:
                    self.status = ActionStatus.success
            else:
                self.status = self._actions[self._current].status
        return commands

    def get_end_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                         image_frequency: ImageFrequency) -> List[dict]:
        return super().get_end_commands(resp=resp,
                                        static=static,
                                        dynamic=dynamic,
                                        image_frequency=image_frequency)


"""
The poses are relative to the agent's position and rotation.
Assume the agent is facing positive z.
"""


class FloodAgent(Replicant):
    def __init__(self, constants=default_const, *args, **kwargs):
        self.constants = constants
        self.heatmap = 0
        super().__init__(*args, **kwargs)

    # define custom actions here
    def pick_up(self, target: int, arm: Union[Arm, List[Arm]],
                final_pose_loc: np.ndarray = np.array([0.1, 1.1, 0.6]), absolute: bool = False,
                angle: Optional[float] = 90, axis: Optional[str] = "pitch") -> None:
        if isinstance(arm, Arm):
            self.action = SequentialAction([
                TurnTo(target=target),
                LookAt(target=target,
                       duration=0.1,
                       scale_duration=True),
                ReachFor(target=target,
                         arms=Replicant._arms_to_list(arm),
                         absolute=absolute,
                         dynamic=self.dynamic,
                         collision_detection=self.collision_detection,
                         offhand_follows=False,
                         arrived_at=0.1,
                         previous=None,
                         duration=0.25,
                         scale_duration=True,
                         max_distance=1.0,
                         from_held=False,
                         held_point="bottom"),
                Grasp(target=target, arm=arm, dynamic=self.dynamic, angle=angle, axis=axis, relative_to_hand=True, offset=0.0),
                ReachFor(target=final_pose_loc,
                         arms=Replicant._arms_to_list(arm),
                         absolute=absolute,
                         dynamic=self.dynamic,
                         collision_detection=self.collision_detection,
                         offhand_follows=False,
                         arrived_at=0.1,
                         previous=None,
                         duration=0.25,
                         scale_duration=True,
                         max_distance=1.0,
                         from_held=False,
                         held_point="bottom"),
            ])
        else:
            self.action = SequentialAction([
                TurnTo(target=target),
                LookAt(target=target,
                       duration=0.1,
                       scale_duration=True),
                ReachFor(target=target,
                         arms=Replicant._arms_to_list(arm),
                         absolute=absolute,
                         dynamic=self.dynamic,
                         collision_detection=self.collision_detection,
                         offhand_follows=False,
                         arrived_at=0.1,
                         previous=None,
                         duration=0.25,
                         scale_duration=True,
                         max_distance=1.0,
                         from_held=False,
                         held_point="bottom"),
                Grasp(target=target, arm=Arm.left, dynamic=self.dynamic, angle=angle, axis=axis),
                ReachFor(target=final_pose_loc,
                         arms=[Arm.left],
                         absolute=absolute,
                         dynamic=self.dynamic,
                         collision_detection=self.collision_detection,
                         offhand_follows=True,
                         arrived_at=0.1,
                         previous=None,
                         duration=0.25,
                         scale_duration=True,
                         max_distance=1.0,
                         from_held=False,
                         held_point="bottom"),
            ])

    def get_facing(self):
        """return a radian"""
        v = QuaternionUtils.multiply_by_vector(QuaternionUtils.get_inverse(self.dynamic.transform.rotation),
                                               np.array([0, 0, 1.0]))
        v = v / np.linalg.norm(v)
        v = np.arctan2(v[2], v[0])
        return v
      
    def grasp_id(self):
        """
        if the agent is grasping an object, return the id of the object
        otherwise, return None
        """
        if isinstance(self.action, Grasp):
            return self.action.target
        elif isinstance(self.action, SequentialAction) and isinstance(self.action._actions[self.action._current], Grasp):
            return self.action._actions[self.action._current].target
        return None
    
    def fail_grasp(self):
        """
        if the agent is grasping an object and it is too far away, fail the grasp (this is controlled by the outside)
        """
        if isinstance(self.action, Grasp):
            self.action.status = ActionStatus.cannot_grasp
        elif isinstance(self.action, SequentialAction) and isinstance(self.action._actions[self.action._current], Grasp):
            self.action._actions[self.action._current].status = ActionStatus.cannot_grasp
        else:
            raise Exception("Not grasping now")