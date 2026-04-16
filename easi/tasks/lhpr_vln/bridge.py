"""LHPR-VLN bridge — wraps vendored SceneSimulator via BaseBridge.

This script runs inside the easi_habitat_sim_v0_3_0 conda env (Python 3.9).
Communicates with parent process via filesystem IPC.

Key difference from other bridges: creates a NEW SceneSimulator per episode
(Habitat-Sim ties one Simulator instance to one scene). The bridge calls
self.env.close() and recreates on each reset.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--data-dir /path/to/data] [--simulator-kwargs '{}']
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from easi.simulators.base_bridge import BaseBridge
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class LHPRVLNBridge(BaseBridge):
    """BaseBridge wrapper for LHPR-VLN SceneSimulator.

    Unlike other bridges, this creates a new SceneSimulator per episode
    because Habitat-Sim binds one simulator instance to one scene file.
    """

    _scene_sim = None  # The vendored SceneSimulator (recreated per episode)
    _prev_position = None  # For collision detection

    def _create_env(self, reset_config, simulator_kwargs):
        """Called once on first reset. We store simulator_kwargs and return
        a placeholder — the real SceneSimulator is created in _on_reset()."""
        # Store config for _on_reset to use
        self._sim_kwargs = simulator_kwargs
        return object()  # placeholder — _on_reset creates the real sim

    def _on_reset(self, env, reset_config):
        """Create a new SceneSimulator for this episode's scene."""
        from easi.tasks.lhpr_vln.vendor.scene_simulator import SceneSimulator

        # Close previous simulator if any
        if self._scene_sim is not None:
            self._scene_sim.close()
            self._scene_sim = None

        data_dir = reset_config.get("data_dir") or self._sim_kwargs.get("data_dir", "")
        scene_base_path = str(Path(data_dir) / "hm3d") + "/"
        scene_dataset_path = str(Path(data_dir) / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json")

        # Parse episode data from reset_config
        targets = json.loads(reset_config["targets"]) if isinstance(reset_config["targets"], str) else reset_config["targets"]
        regions = json.loads(reset_config["regions"]) if isinstance(reset_config["regions"], str) else reset_config["regions"]
        gt_step = json.loads(reset_config.get("gt_step", "[]")) if isinstance(reset_config.get("gt_step"), str) else reset_config.get("gt_step")

        device = self._sim_kwargs.get("device", "cpu")
        if device == "gpu":
            gpu_device_id = self._sim_kwargs.get("_assigned_gpu_id", 0)
        else:
            gpu_device_id = -1
        max_steps = self._sim_kwargs.get("max_steps", 500)
        width = self._sim_kwargs.get("screen_width", 512)
        height = self._sim_kwargs.get("screen_height", 512)
        success_distance = self._sim_kwargs.get("success_distance", 0.25)
        forward_step_size = self._sim_kwargs.get("forward_step_size", 0.25)
        turn_angle = self._sim_kwargs.get("turn_angle", 30.0)
        sensors = self._sim_kwargs.get("sensors", None)

        self._scene_sim = SceneSimulator(
            scene_id=reset_config["scene_id"],
            robot=reset_config.get("robot", "spot"),
            targets=targets,
            regions=regions,
            instruction=reset_config.get("instruction", ""),
            gt_step=gt_step if gt_step else None,
            scene_base_path=scene_base_path,
            scene_dataset_path=scene_dataset_path,
            gpu_device_id=gpu_device_id,
            success_distance=success_distance,
            max_steps=max_steps,
            width=width,
            height=height,
            forward_step_size=forward_step_size,
            turn_angle=turn_angle,
            sensors=sensors,
        )

        # Get initial observation (step -1 in SceneSimulator)
        obs, done, info = self._scene_sim.actor("move_forward")
        # Initialize position tracking for collision detection
        sim_info = self._scene_sim.info or {}
        self._prev_position = sim_info.get("agent_position")
        return obs

    def _on_step(self, env, action_text):
        """Step the SceneSimulator and return (obs, reward, done, info)."""
        # Capture position before action for collision detection
        pre_pos = self._prev_position

        obs, done, info = self._scene_sim.actor(action_text)
        episode_over = self._scene_sim.episode_over

        # Get current position
        sim_info = self._scene_sim.info or {}
        cur_pos = sim_info.get("agent_position")

        # Detect collision: move_forward with no position change
        collided = False
        if action_text == "move_forward" and pre_pos is not None and cur_pos is not None:
            collided = all(
                abs(float(a) - float(b)) < 1e-4
                for a, b in zip(pre_pos, cur_pos)
            )

        self._prev_position = cur_pos

        # Build info dict with subtask state
        step_info = self._build_step_info(done, episode_over, action_text, collided)
        return obs, 0.0, episode_over, step_info

    def _build_step_info(self, all_done: bool, episode_over: bool,
                         action: str = "", collided: bool = False) -> dict:
        """Build info dict with subtask completion state for metrics."""
        sim = self._scene_sim
        results = sim.return_results()

        task_success = 1.0 if all(sim.successes) else 0.0
        geo_dis = sim.info.get("geo_dis", -1)

        # Build feedback reporting action outcome only
        # (subtask progress/distance is shown separately in Environmental Feedback)
        if action == "stop":
            # After stop, sim.stage has already advanced. Check success of the
            # previous subtask via sim.successes[stage - 1] instead of geo_dis,
            # because sim.info now reflects the NEXT subtask's distance.
            prev_stage = sim.stage - 1
            if prev_stage >= 0 and sim.successes[prev_stage]:
                feedback = "Subtask completed successfully."
            else:
                feedback = f"Stop failed: too far from target ({geo_dis:.1f}m away, need < {sim.success_distance}m)."
        elif collided:
            feedback = "Blocked: move_forward hit an obstacle, position unchanged. Try turning."
        elif episode_over and all_done:
            feedback = "All subtasks completed. Navigation finished."
        elif episode_over:
            feedback = f"Maximum steps reached ({sim.stage}/{sim.target_num} subtasks completed)."
        else:
            feedback = "OK"

        return {
            "task_success": task_success,
            "feedback": feedback,
            "subtask_stage": float(sim.stage),
            "subtask_total": float(sim.target_num),
            "current_geo_distance": float(sim.info.get("geo_dis", -1)),
            # Serialized arrays for evaluate_episode to read at episode end
            "subtask_successes": json.dumps([int(s) for s in results["successes"]]),
            "subtask_oracle_successes": json.dumps([int(s) for s in results["oracle_successes"]]),
            "subtask_nav_errors": json.dumps(results["navigation_errors"]),
            "subtask_nav_steps": json.dumps(results["navigation_steps"]),
            "gt_steps": json.dumps(results["gt_step"] or []),
            "gt_paths": json.dumps(results["gt_path"]),
        }

    def _extract_image(self, obs):
        """Return front RGB view as the primary observation image."""
        if "color_sensor_f" in obs:
            return obs["color_sensor_f"][:, :, :3]  # RGBA -> RGB
        # Fallback to any available RGB sensor
        for key in ("color_sensor_l", "color_sensor_r"):
            if key in obs:
                return obs[key][:, :, :3]
        return None

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to save 3 separate RGB views and pass paths in metadata.

        Saves: step_NNNN_left.png, step_NNNN_front.png, step_NNNN_right.png
        Front is the primary rgb_path; left/right paths go in metadata.
        """
        from easi.communication.schemas import make_observation_response

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image

        paths = {}
        for view_name, sensor_key in [("left", "color_sensor_l"),
                                       ("front", "color_sensor_f"),
                                       ("right", "color_sensor_r")]:
            if sensor_key not in obs:
                continue
            rgb = obs[sensor_key][:, :, :3]
            path = save_dir / ("step_%04d_%s.png" % (self.step_count, view_name))
            Image.fromarray(rgb).save(str(path))
            paths[view_name] = str(path)

        # Save depth images when present
        depth_paths = {}
        for view_name, sensor_key in [("left", "depth_sensor_l"),
                                       ("front", "depth_sensor_f"),
                                       ("right", "depth_sensor_r")]:
            if sensor_key not in obs:
                continue
            depth = obs[sensor_key]
            clipped = np.clip(depth, 0, 10.0)
            depth_uint16 = (clipped / 10.0 * 65535).astype(np.uint16)
            # Squeeze to 2D if needed (H, W, 1) -> (H, W)
            if depth_uint16.ndim == 3:
                depth_uint16 = depth_uint16[:, :, 0]
            path = save_dir / ("step_%04d_%s_depth.png" % (self.step_count, view_name))
            Image.fromarray(depth_uint16).save(str(path))
            depth_paths[view_name] = str(path)

        clean_info = self._extract_info(info or {})
        clean_info["step"] = str(self.step_count)

        # Build metadata with image paths + environmental feedback for prompt builder
        metadata = {
            "step": str(self.step_count),
        }
        if "left" in paths:
            metadata["left_rgb_path"] = paths["left"]
        if "front" in paths:
            metadata["front_rgb_path"] = paths["front"]
        if "right" in paths:
            metadata["right_rgb_path"] = paths["right"]
        if "front" in depth_paths:
            metadata["front_depth_path"] = depth_paths["front"]
        if "left" in depth_paths:
            metadata["left_depth_path"] = depth_paths["left"]
        if "right" in depth_paths:
            metadata["right_depth_path"] = depth_paths["right"]

        # Expose environmental feedback in metadata so the prompt builder can use it
        sim = self._scene_sim
        if sim is not None:
            sim_info = sim.info or {}
            metadata["subtask_stage"] = str(float(sim.stage))
            metadata["subtask_total"] = str(float(sim.target_num))
            metadata["current_geo_distance"] = str(float(sim_info.get("geo_dis", -1)))
            metadata["current_target"] = str(sim_info.get("target", ""))
            # Agent position/rotation
            pos = sim_info.get("agent_position")
            if pos is not None:
                metadata["agent_position"] = json.dumps([float(x) for x in pos])
            rot = sim_info.get("agent_rotation")
            if rot is not None:
                metadata["agent_rotation"] = json.dumps([float(rot.w), float(rot.x), float(rot.y), float(rot.z)])
            # Target coordinate
            coord = sim_info.get("target_coord")
            if coord is not None:
                metadata["target_coordinate"] = json.dumps([float(x) for x in coord])

        # Build agent_pose from sim info
        agent_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if sim is not None:
            sim_info = sim.info or {}
            pos = sim_info.get("agent_position")
            if pos is not None:
                agent_pose = [float(pos[0]), float(pos[1]), float(pos[2]), 0.0, 0.0, 0.0]

        # Use front as primary rgb_path, fall back to any available view
        primary_rgb = paths.get("front") or paths.get("left") or paths.get("right") or ""

        return make_observation_response(
            rgb_path=primary_rgb,
            agent_pose=agent_pose,
            metadata=metadata,
            reward=reward,
            done=done,
            info=clean_info,
        )

    def _extract_info(self, info):
        """Pass through our pre-built info dict.
        All values are already int/float/str/bool."""
        return {k: v for k, v in info.items()
                if isinstance(v, (int, float, str, bool))}

    def close(self):
        """Close the SceneSimulator."""
        if self._scene_sim is not None:
            self._scene_sim.close()
            self._scene_sim = None
        super().close()


if __name__ == "__main__":
    LHPRVLNBridge.main()
