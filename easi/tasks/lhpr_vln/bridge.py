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

        gpu_device_id = self._sim_kwargs.get("gpu_device_id", 0)
        if gpu_device_id < 0:
            gpu_device_id = 0  # Habitat-Sim needs a real GPU ID
        max_steps = self._sim_kwargs.get("max_steps", 500)
        width = self._sim_kwargs.get("screen_width", 512)
        height = self._sim_kwargs.get("screen_height", 512)
        success_distance = self._sim_kwargs.get("success_distance", 1.0)

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
        )

        # Get initial observation (step -1 in SceneSimulator)
        obs, done, info = self._scene_sim.actor("move_forward")
        return obs

    def _on_step(self, env, action_text):
        """Step the SceneSimulator and return (obs, reward, done, info)."""
        obs, done, info = self._scene_sim.actor(action_text)
        episode_over = self._scene_sim.episode_over

        # Build info dict with subtask state
        step_info = self._build_step_info(done, episode_over)
        return obs, 0.0, episode_over, step_info

    def _build_step_info(self, all_done: bool, episode_over: bool) -> dict:
        """Build info dict with subtask completion state for metrics."""
        sim = self._scene_sim
        results = sim.return_results()

        task_success = 1.0 if all(sim.successes) else 0.0

        # Build feedback string for the agent
        if episode_over and all_done:
            feedback = "All subtasks completed. Navigation finished."
        elif episode_over:
            feedback = f"Maximum steps reached. Completed {sim.stage}/{sim.target_num} subtasks."
        else:
            current_target = sim.targets[sim.stage] if sim.stage < sim.target_num else "N/A"
            feedback = f"Subtask {sim.stage + 1}/{sim.target_num}: navigate to {current_target}. Distance: {sim.info.get('geo_dis', -1):.1f}m"

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
        return obs["color_sensor_f"][:, :, :3]  # RGBA -> RGB

    def _make_response(self, obs, reward=0.0, done=False, info=None):
        """Override to save 3 separate RGB views and pass paths in metadata.

        Saves: step_NNNN_left.png, step_NNNN_front.png, step_NNNN_right.png
        Front is the primary rgb_path; left/right paths go in metadata.
        """
        from easi.communication.schemas import make_observation_response

        save_dir = Path(self.episode_output_dir) if self.episode_output_dir else self.workspace
        save_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for view_name, sensor_key in [("left", "color_sensor_l"),
                                       ("front", "color_sensor_f"),
                                       ("right", "color_sensor_r")]:
            rgb = obs[sensor_key][:, :, :3]
            path = save_dir / ("step_%04d_%s.png" % (self.step_count, view_name))
            from PIL import Image
            Image.fromarray(rgb).save(str(path))
            paths[view_name] = str(path)

        clean_info = self._extract_info(info or {})
        clean_info["step"] = str(self.step_count)
        return make_observation_response(
            rgb_path=paths["front"],
            agent_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            metadata={
                "step": str(self.step_count),
                "left_rgb_path": paths["left"],
                "front_rgb_path": paths["front"],
                "right_rgb_path": paths["right"],
            },
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
