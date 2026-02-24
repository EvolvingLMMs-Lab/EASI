"""EASI bridge for AI2-THOR Rearrangement Challenge.

Runs inside the easi_ai2thor_v5_0_0 conda env (Python 3.10).
Wraps RearrangeTHOREnvironment in SNAP mode with 84 discrete actions.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge


def _snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase (inverse of _pascal_to_snake in actions.py)."""
    return "".join(word.capitalize() for word in name.split("_"))


class AI2THORRearrangement2023Bridge(BaseBridge):
    """Bridge for 1-phase rearrangement (unshuffle only, SNAP mode)."""

    def _create_env(self, reset_config, simulator_kwargs):
        from easi.tasks.ai2thor_rearrangement_2023.vendor.rearrange.environment import (
            RearrangeTHOREnvironment,
            RearrangeMode,
        )

        screen_h = simulator_kwargs.get("screen_height", 224)
        screen_w = simulator_kwargs.get("screen_width", 224)
        fov = simulator_kwargs.get("fov", 90)
        grid_size = simulator_kwargs.get("grid_size", 0.25)
        rotate_step = simulator_kwargs.get("rotate_step_degrees", 90)
        vis_dist = simulator_kwargs.get("visibility_distance", 1.5)
        snap = simulator_kwargs.get("snap_to_grid", True)
        quality = simulator_kwargs.get("quality", "Very Low")

        # Sensor toggles
        self._use_rgb = simulator_kwargs.get("use_rgb", True)
        self._use_depth = simulator_kwargs.get("use_depth", False)
        self._use_gps = simulator_kwargs.get("use_gps", True)

        controller_kwargs = {
            "width": screen_w,
            "height": screen_h,
            "fieldOfView": fov,
            "gridSize": grid_size,
            "rotateStepDegrees": rotate_step,
            "visibilityDistance": vis_dist,
            "snapToGrid": snap,
            "quality": quality,
            "renderDepthImage": self._use_depth,
            "fastActionEmit": True,
        }

        env = RearrangeTHOREnvironment(
            mode=RearrangeMode.SNAP,
            controller_kwargs=controller_kwargs,
            force_cache_reset=True,
        )

        # Action tracking for PuSR/PuLen metrics
        self._actions_taken = []
        self._actions_taken_success = []

        return env

    def _on_reset(self, env, reset_config):
        from easi.tasks.ai2thor_rearrangement_2023.vendor.rearrange.environment import (
            RearrangeTaskSpec,
        )

        # Deserialize JSON strings from HF dataset
        agent_position = json.loads(reset_config["agent_position"]) if isinstance(
            reset_config["agent_position"], str
        ) else reset_config["agent_position"]
        starting_poses = json.loads(reset_config["starting_poses"]) if isinstance(
            reset_config["starting_poses"], str
        ) else reset_config["starting_poses"]
        target_poses = json.loads(reset_config["target_poses"]) if isinstance(
            reset_config["target_poses"], str
        ) else reset_config["target_poses"]
        openable_data = json.loads(reset_config["openable_data"]) if isinstance(
            reset_config["openable_data"], str
        ) else reset_config["openable_data"]

        task_spec = RearrangeTaskSpec(
            scene=reset_config["scene"],
            agent_position=agent_position,
            agent_rotation=int(reset_config["agent_rotation"]),
            starting_poses=starting_poses,
            target_poses=target_poses,
            openable_data=openable_data,
            stage="eval",
        )

        # Reset (walkthrough phase) then immediately shuffle (unshuffle phase)
        env.reset(task_spec=task_spec, force_axis_aligned_start=True)
        env.shuffle()

        # Clear action tracking
        self._actions_taken = []
        self._actions_taken_success = []

        # No-op to get initial state
        return env.step("Pass")

    def _on_step(self, env, action_text):
        action_name = action_text.strip().lower()

        action_success = False

        if action_name == "done":
            # Episode end — compute final metrics
            info = self._build_info(env, action_name, True)
            info.update(self._compute_final_metrics(env))
            rgb = self._get_rgb(env)
            return rgb, 0.0, True, info

        elif action_name in (
            "move_ahead", "move_left", "move_right", "move_back",
            "rotate_right", "rotate_left", "stand", "crouch",
            "look_up", "look_down",
        ):
            action_success = getattr(env, action_name)()

        elif action_name == "drop_held_object_with_snap":
            action_success = env.drop_held_object_with_snap()

        elif action_name.startswith("pickup_"):
            action_success = self._do_pickup(env, action_name)

        elif action_name.startswith("open_by_type_"):
            action_success = self._do_open(env, action_name)

        # Track actions for PuSR/PuLen
        self._actions_taken.append(action_name)
        self._actions_taken_success.append(action_success)

        rgb = self._get_rgb(env)
        info = self._build_info(env, action_name, action_success)
        feedback = "success" if action_success else "action failed"
        info["feedback"] = feedback

        return rgb, 0.0, False, info

    def _do_pickup(self, env, action_name: str) -> bool:
        """Pick up the nearest visible object of the given type."""
        from easi.tasks.ai2thor_rearrangement_2023.vendor.rearrange.environment import (
            include_object_data,
        )

        with include_object_data(env.controller):
            metadata = env.last_event.metadata

            if len(metadata["inventoryObjects"]) != 0:
                return False

            # Convert snake_case action to PascalCase object type
            obj_type_snake = action_name.replace("pickup_", "")
            obj_type = _snake_to_pascal(obj_type_snake)

            candidates = [
                o for o in metadata["objects"]
                if o["visible"] and o["objectType"] == obj_type
            ]
            if not candidates:
                return False

            # Sort by distance, pick closest
            candidates.sort(key=lambda o: (o["distance"], o["name"]))
            obj_id = candidates[0]["objectId"]

            env.controller.step(
                "PickupObject",
                objectId=obj_id,
                **env.physics_step_kwargs,
            )
            return env.controller.last_event.metadata["lastActionSuccess"]

    def _do_open(self, env, action_name: str) -> bool:
        """Toggle openness of the nearest visible openable object."""
        from easi.tasks.ai2thor_rearrangement_2023.vendor.rearrange.environment import (
            include_object_data,
        )

        with include_object_data(env.controller):
            metadata = env.last_event.metadata

            obj_type_snake = action_name.replace("open_by_type_", "")
            obj_type = _snake_to_pascal(obj_type_snake)

            candidates = [
                o for o in metadata["objects"]
                if o["visible"] and o["objectType"] == obj_type and o["openable"]
            ]
            if not candidates:
                return False

            # Sort by distance, pick closest
            candidates.sort(key=lambda o: (o["distance"], o["name"]))
            obj = candidates[0]

            # Toggle: if open -> close, if closed -> open
            if obj["isOpen"] or obj.get("openness", 0) > 0.5:
                env.controller.step(
                    "CloseObject",
                    objectId=obj["objectId"],
                    **env.physics_step_kwargs,
                )
            else:
                env.controller.step(
                    "OpenObject",
                    objectId=obj["objectId"],
                    openness=1.0,
                    **env.physics_step_kwargs,
                )
            return env.controller.last_event.metadata["lastActionSuccess"]

    def _get_rgb(self, env) -> np.ndarray:
        """Extract RGB frame from environment."""
        return env.last_event.frame.copy()

    def _build_info(self, env, action_name: str, action_success: bool) -> dict:
        """Build info dict with sensor data and action feedback."""
        info = {
            "action_name": action_name,
            "action_success": action_success,
        }

        # GPS data (if enabled)
        if self._use_gps:
            loc = env.get_agent_location()
            info["agent_x"] = float(loc["x"])
            info["agent_y"] = float(loc["y"])
            info["agent_z"] = float(loc["z"])
            info["agent_rotation"] = float(loc["rotation"])
            info["agent_horizon"] = float(loc["horizon"])
            info["agent_standing"] = float(loc.get("standing", True))

        # Held object info
        held = env.held_object
        info["held_object"] = held["objectType"] if held else "none"

        return info

    def _compute_final_metrics(self, env) -> dict:
        """Compute rearrangement metrics at episode end (when done is called)."""
        ips, gps, cps = env.poses

        start_energies = env.pose_difference_energy(gps, ips)
        end_energies = env.pose_difference_energy(gps, cps)
        start_energy = float(start_energies.sum())
        end_energy = float(end_energies.sum())

        start_misplaced = start_energies > 0.0
        end_misplaced = end_energies > 0.0

        num_initially_misplaced = int(start_misplaced.sum())
        num_fixed = int(
            num_initially_misplaced - (start_misplaced & end_misplaced).sum()
        )
        num_newly_misplaced = int(
            (end_misplaced & ~start_misplaced).sum()
        )
        num_broken = sum(1 for cp in cps if cp.get("broken", False))

        prop_fixed = (
            1.0
            if num_initially_misplaced == 0
            else num_fixed / num_initially_misplaced
        )

        return {
            "success": float(end_energy == 0),
            "prop_fixed_strict": float(
                (num_newly_misplaced == 0) * prop_fixed
            ),
            "energy_prop": (
                end_energy / start_energy if start_energy > 0 else 0.0
            ),
            "start_energy": start_energy,
            "end_energy": end_energy,
            "num_initially_misplaced": num_initially_misplaced,
            "num_fixed": num_fixed,
            "num_newly_misplaced": num_newly_misplaced,
            "num_broken": num_broken,
        }

    def _extract_image(self, obs):
        """Extract RGB from observation."""
        if isinstance(obs, np.ndarray):
            return obs
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def _extract_info(self, info):
        """Filter info to JSON-serializable types."""
        if not isinstance(info, dict):
            return {}
        return {
            k: v for k, v in info.items()
            if isinstance(v, (int, float, str, bool))
        }


if __name__ == "__main__":
    AI2THORRearrangement2023Bridge.main()
