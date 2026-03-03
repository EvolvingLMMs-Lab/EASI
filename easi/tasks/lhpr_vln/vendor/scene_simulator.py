"""Habitat-Sim SceneSimulator for LHPR-VLN.

Vendored from LH-VLN/habitat_base/simulation.py with parameterization.
Original creates a new habitat_sim.Simulator per episode, computes geodesic
distances, and tracks multi-subtask stage/success/oracle state.
"""
import math
import operator
from typing import Optional, Union

import habitat_sim
import numpy as np
import quaternion

from .scene_config import make_cfg, make_setting


class SceneSimulator:
    def __init__(
        self,
        *,
        scene_id: str,
        robot: str,
        targets: list[str],
        regions: list[str],
        instruction: str,
        gt_step: list[int] | None = None,
        scene_base_path: str,
        scene_dataset_path: str,
        gpu_device_id: int = 0,
        success_distance: float = 1.0,
        max_steps: int = 500,
        width: int = 512,
        height: int = 512,
    ):
        self.scene_id = scene_id
        self.robot = robot
        self.targets = targets
        self.regions = regions
        self.instruction = instruction
        self.success_distance = success_distance
        self.max_steps = max_steps

        # Resolve scene path: IDs < 800 -> train/, >= 800 -> val/
        split = "train/" if int(scene_id[:5]) < 800 else "val/"
        scene_path = scene_base_path + split + scene_id

        # Init simulator
        sim_settings = make_setting(scene_path, scene_dataset_path, robot, width, height)
        cfg = make_cfg(sim_settings, gpu_device_id)
        self.sim = habitat_sim.Simulator(cfg)
        self.sim_settings = sim_settings

        # Pathfinder
        self.pathfinder = self.sim.pathfinder
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])

        # Random navigable start position
        agent_state = habitat_sim.AgentState()
        sample_navigable_point = self.pathfinder.get_random_navigable_point()
        agent_state.position = sample_navigable_point - np.array([0, 0, -0.25])
        self.agent.set_state(agent_state)

        # Greedy geodesic follower (for GT path computation)
        self.follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder=self.pathfinder,
            agent=self.agent,
            goal_radius=success_distance,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        # Initial observation
        self.observations = self.sim.step("move_forward")

        # Subtask tracking
        self.step = -1
        self.stage = 0
        self.target_num = len(targets)
        self.nav_steps = []
        self.successes = [False] * self.target_num
        self.oracle_successes = [False] * self.target_num
        self.nav_errors = []

        self.info = self._get_info()
        self.gt_step = gt_step
        self.gt_path = [self.info["geo_dis"]]

        self.done = False
        self.episode_over = False

    def actor(self, action: str):
        """Perform one action. Returns (observations, done, info)."""
        if action != "stop":
            self.observations = self.sim.step(action)

        if self.step == -1:
            self.step += 1
            return self.observations, self.done, self.info

        self.step += 1
        self.info = self._get_info()

        if self.info["geo_dis"] < self.success_distance:
            self.oracle_successes[self.stage] = True

        if action == "stop":
            if self.info["geo_dis"] < self.success_distance:
                self.successes[self.stage] = True
            self.nav_errors.append(self.info["geo_dis"])

            if not self.nav_steps:
                self.nav_steps.append(self.step)
            else:
                self.nav_steps.append(self.step - sum(self.nav_steps))

            self.stage += 1
            if self.stage >= self.target_num:
                self.done = True
                self.episode_over = True
                return self.observations, self.done, self.info

            self.info = self._get_info()
            self.gt_path.append(self.info["geo_dis"])

        if self.step >= self.max_steps:
            self.episode_over = True
            if not self.nav_steps:
                self.nav_steps.append(self.step)
            else:
                self.nav_steps.append(self.step - sum(self.nav_steps))
            self.nav_errors.append(self.info["geo_dis"])

        return self.observations, self.done, self.info

    def get_front_rgb(self) -> np.ndarray:
        """Return front RGB as H x W x 3 uint8 numpy array."""
        rgba = self.observations["color_sensor_f"]
        return rgba[:, :, :3]  # drop alpha channel

    def _get_info(self) -> dict:
        """Return info about current state relative to current target."""
        obj_target = self.targets[self.stage]
        coord_list = self._get_coord(obj_target)
        if not coord_list:
            return {"target": obj_target, "geo_dis": math.inf}
        snap_coord_list = [self.pathfinder.snap_point(c) for c in coord_list]
        geo_dis, snap_coord = self._geodesic_distance(snap_coord_list)
        position, rotation = self._return_state()
        return {
            "target": obj_target,
            "target_coord": snap_coord,
            "agent_position": position,
            "agent_rotation": rotation,
            "geo_dis": geo_dis,
        }

    def _get_coord(self, obj_target: str) -> list:
        """Return coordinate list for target object in its specified region."""
        scene = self.sim.semantic_scene
        coord_list = []
        index = self.targets.index(obj_target)
        region_id = self.regions[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    coord_list.append(obj.aabb.center)
        return coord_list

    def _geodesic_distance(self, position_b_list) -> tuple[float, list]:
        """Compute min geodesic distance from agent to any position in list."""
        position_a, _ = self._return_state()
        geo_dis = math.inf
        coord = position_b_list[0]
        for position_b in position_b_list:
            path = habitat_sim.nav.ShortestPath()
            path.requested_end = np.array(position_b, dtype=np.float32)
            path.requested_start = np.array(position_a, dtype=np.float32)
            if self.pathfinder.find_path(path):
                if path.geodesic_distance < geo_dis:
                    geo_dis = path.geodesic_distance
                    coord = position_b
        return geo_dis, coord

    def _return_state(self):
        agent_state = self.agent.get_state()
        return agent_state.position, agent_state.rotation

    def return_results(self) -> dict:
        """Return complete episode results for metrics computation."""
        return {
            "successes": self.successes,
            "oracle_successes": self.oracle_successes,
            "navigation_steps": self.nav_steps,
            "navigation_errors": self.nav_errors,
            "gt_step": self.gt_step,
            "gt_path": self.gt_path,
        }

    def close(self):
        self.sim.close()
