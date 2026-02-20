"""EB-Manipulation bridge -- wraps vendored EBManEnv via BaseBridge.

This script runs inside the easi_coppeliasim_v4_1_0 conda env (Python 3.10).
Communicates with parent process via filesystem IPC.

The bridge adds vendored AMSolver (easi/tasks/ebmanipulation/vendor/amsolver/)
to sys.path at startup so EBManEnv can import from amsolver.

It also handles:
- Object coordinate extraction (form_object_coord_for_input)
- Image annotation with XYZ axes (draw_xyz_coordinate)
- Image annotation with YOLO bounding boxes (draw_bounding_boxes)

These are passed to the prompt builder via observation metadata.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _setup_amsolver_path(simulator_kwargs: dict, data_dir: str = "") -> None:
    """Add vendored AMSolver to sys.path and set TTMS_FOLDER.

    AMSolver is vendored at easi/tasks/ebmanipulation/vendor/amsolver/.
    The vendor/ directory must be on sys.path so ``import amsolver`` works.
    """
    vendor_dir = str(Path(__file__).parent / "vendor")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    # Allow override via simulator_kwargs (for custom installations)
    amsolver_parent = simulator_kwargs.get("amsolver_parent_path", "")
    if amsolver_parent and amsolver_parent not in sys.path:
        sys.path.insert(0, amsolver_parent)

    # Point AMSolver's TTMS_FOLDER to the HF dataset cache
    if data_dir:
        import amsolver.task_environment as te

        te.TTMS_FOLDER = data_dir + "/"


from easi.simulators.base_bridge import BaseBridge
from easi.tasks.ebmanipulation.actions import deserialize_action
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class EBManipulationBridge(BaseBridge):
    """BaseBridge wrapper around vendored EBManEnv for EB-Manipulation.

    Handles object coordinate extraction and image annotation to match
    the original EmbodiedBench evaluation pipeline.
    """

    # Camera views used for object coordinate computation
    _camera_types = ["front_rgb"]

    def _create_env(self, reset_config, simulator_kwargs):
        # Must set up AMSolver path before importing vendored env
        data_dir = reset_config.get("data_dir", "")
        _setup_amsolver_path(simulator_kwargs, data_dir=data_dir)

        from easi.tasks.ebmanipulation.vendor.EBManEnv import EBManEnv

        resolution = simulator_kwargs.get("screen_height", 500)
        max_steps = simulator_kwargs.get("max_steps", 15)
        voxel_size = simulator_kwargs.get("voxel_size", 100)
        rotation_resolution = simulator_kwargs.get("rotation_resolution", 3)
        scene_bounds = simulator_kwargs.get(
            "scene_bounds", [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        )

        self._resolution = resolution
        self._detection_box = simulator_kwargs.get("detection_box", True)

        return EBManEnv(
            img_size=(resolution, resolution),
            max_steps=max_steps,
            scene_bounds=scene_bounds,
            voxel_size=voxel_size,
            rotation_resolution=rotation_resolution,
        )

    def _on_reset(self, env, reset_config):
        # Construct episode file paths from HF dataset cache
        data_dir = reset_config.get("data_dir", "")
        split = reset_config.get("split", "base")
        task_name = reset_config["task_name"]
        variation = reset_config.get("variation", 0)
        episode_num = reset_config.get("episode_num", 0)

        episode_path = (
            Path(data_dir)
            / "data"
            / split
            / "eval"
            / task_name
            / f"variation{variation}"
            / "episodes"
            / f"episode{episode_num}"
        )

        episode = {
            "task_file": task_name,
            "task_base": str(episode_path / "task_base.ttm"),
            "waypoint_sets": str(episode_path / "waypoint_sets.ttm"),
            "configs": str(episode_path / "configs.pkl"),
        }
        description, obs_dict = env.reset(episode=episode)

        # Store reset_config for metadata
        self._current_reset_config = reset_config

        return obs_dict

    def _on_step(self, env, action_text):
        """Parse 7D action from action_text string and execute."""
        action = deserialize_action(action_text)
        if not action or len(action) != 7:
            # Random action fallback (matches ManipPlanner behavior)
            voxel_size = self.simulator_kwargs.get("voxel_size", 100)
            rotation_resolution = self.simulator_kwargs.get("rotation_resolution", 3)
            rotation_bins = int(360 / rotation_resolution)
            action = [np.random.randint(0, voxel_size) for _ in range(3)] + [
                np.random.randint(0, rotation_bins) for _ in range(3)
            ] + [1.0]
        return env.step(action)

    def _extract_image(self, obs):
        """Extract front camera RGB from observation dict."""
        return obs["front_rgb"]

    def _extract_info(self, info):
        return {
            "task_success": float(info.get("task_success", 0.0)),
            "action_success": float(info.get("action_success", 0.0)),
            "feedback": str(info.get("env_feedback", "")),
            "env_step": int(info.get("env_step", 0)),
            "episode_elapsed_seconds": float(
                info.get("episode_elapsed_seconds", 0.0)
            ),
        }

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to compute object coordinates and annotate images.

        This matches the EmbodiedBench evaluation pipeline:
        1. Save raw front_rgb image
        2. Compute object coordinates via form_object_coord_for_input()
        3. Annotate image with XYZ coordinate axes
        4. Annotate image with YOLO bounding boxes
        5. Pass avg_obj_coord, task_variation, task_class in metadata
        """
        # First, build the standard response (saves raw image)
        response = super()._make_response(obs, reward, done, info)

        # Compute object coordinates from the full observation dict
        avg_obj_coord = "{}"
        task_variation = ""
        task_class = ""

        if self.env is not None:
            task_variation = getattr(self.env, "current_task_variation", "") or ""
            task_class = getattr(self.env, "task_class", "") or ""

            try:
                from easi.tasks.ebmanipulation.vendor.eb_man_utils import (
                    draw_bounding_boxes,
                    draw_xyz_coordinate,
                    form_object_coord_for_input,
                )

                # Compute object coordinates
                coord_result = form_object_coord_for_input(
                    obs, task_class, self._camera_types
                )
                avg_coord, all_avg_point_list, cam_ext_list, cam_int_list = (
                    coord_result
                )
                avg_obj_coord = str(avg_coord)

                # Annotate the saved image
                rgb_path = response["observation"]["rgb_path"]

                if self._detection_box:
                    # Draw XYZ coordinate axes on the image
                    draw_xyz_coordinate(rgb_path, self._resolution)

                    # Draw YOLO bounding boxes
                    if all_avg_point_list and cam_ext_list and cam_int_list:
                        annotated_paths = draw_bounding_boxes(
                            [rgb_path], all_avg_point_list, cam_ext_list, cam_int_list
                        )
                        if annotated_paths:
                            response["observation"]["rgb_path"] = annotated_paths[0]

            except Exception as e:
                logger.warning("Object coordinate extraction failed: %s", e)

        # Store metadata for prompt builder
        response["observation"]["metadata"]["avg_obj_coord"] = avg_obj_coord
        response["observation"]["metadata"]["task_variation"] = task_variation
        response["observation"]["metadata"]["task_class"] = task_class

        return response


if __name__ == "__main__":
    EBManipulationBridge.main()
