"""ManipulaTHOR Arm Point Navigation task for EASI."""
from __future__ import annotations

from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.manipulathor.actions import get_action_space
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class ManipulaTHORTask(BaseTask):
    """ManipulaTHOR Arm Point Navigation benchmark.

    6 evaluation splits, 13 discrete actions, 6 paper-aligned metrics.
    Uses AI2-THOR v3.3.5 in arm mode via subprocess bridge.
    """

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def _build_action_space(self) -> list:
        return get_action_space()

    def get_instruction(self, episode: dict) -> str:
        object_type = episode.get("object_type", "object")
        return f"Pick up the {object_type} and move it to the goal location."

    def format_reset_config(self, episode: dict) -> dict:
        """Convert HF episode row to bridge reset config.

        HF dataset fields: id, scene, object_type, object_id,
        source_position (JSON str), target_position (JSON str),
        initial_agent_pose (JSON str), arm_starting_pose (JSON str), ...
        """
        return {
            "episode_id": str(episode.get("id", "unknown")),
            "scene": episode["scene"],
            "object_type": episode.get("object_type", ""),
            "object_id": episode["object_id"],
            "source_position": episode["source_position"],   # JSON str or dict
            "target_position": episode["target_position"],   # JSON str or dict
            "initial_agent_pose": episode["initial_agent_pose"],  # JSON str or dict
            "arm_starting_pose": episode.get("arm_starting_pose"),  # JSON str or dict
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult],
    ) -> dict[str, float]:
        """Extract per-episode metrics from trajectory.

        Returns dict with keys matching the 6 paper metrics:
        - episode_success: 1.0 if task completed successfully
        - pickup_success: 1.0 if object was picked up
        - success_wo_disturb: 1.0 if success without moving other objects
        - eplen_pickup: steps until pickup (0 if not picked up)
        - eplen_success: total steps if successful (0 if not)
        - num_steps: total episode length
        """
        if not trajectory:
            return {
                "episode_success": 0.0,
                "pickup_success": 0.0,
                "success_wo_disturb": 0.0,
                "eplen_pickup": 0.0,
                "eplen_success": 0.0,
                "num_steps": 0.0,
            }

        last = trajectory[-1]
        episode_success = last.info.get("episode_success", 0.0)
        pickup_success = last.info.get("pickup_success", 0.0)
        success_wo_disturb = last.info.get("success_wo_disturb", 0.0)
        eplen_pickup = last.info.get("eplen_pickup", 0.0)
        eplen_success = last.info.get("eplen_success", 0.0)
        num_steps = float(len(trajectory))

        return {
            "episode_success": float(episode_success),
            "pickup_success": float(pickup_success),
            "success_wo_disturb": float(success_wo_disturb),
            "eplen_pickup": float(eplen_pickup),
            "eplen_success": float(eplen_success),
            "num_steps": num_steps,
        }

    def aggregate_results(
        self, records: list[EpisodeRecord],
    ) -> dict[str, float]:
        """Custom aggregation for ManipulaTHOR's 6 paper metrics.

        Three metrics are simple averages across all episodes:
        - Episode Success %
        - PickUp Success %
        - Ep-Len (average steps)

        One metric averages only over successful episodes:
        - Ep-Success w/o Disturbance %: success_wo_disturb / num_successful
          (Note: in the paper this is reported as a rate, not conditional avg.
           We report both: the rate and the conditional average.)

        Two metrics are conditional averages:
        - Ep-Len for PickUp: average eplen_pickup where pickup occurred
        - Ep-Len for Success: average eplen_success where success occurred
        """
        if not records:
            return {}

        n = len(records)

        # Unconditional averages
        ep_success_sum = sum(r.episode_results["episode_success"] for r in records)
        pickup_sum = sum(r.episode_results["pickup_success"] for r in records)
        steps_sum = sum(r.episode_results["num_steps"] for r in records)

        # Ep-Success w/o Disturbance:
        # The paper reports this as a percentage of ALL episodes (not just successes).
        # From the original code: result["metric/average/success_wo_disturb"] = (len(objects_moved) == 1)
        # This is only set when _success is True, otherwise it's 0.
        # So the paper metric = sum(success_wo_disturb) / n
        wo_disturb_sum = sum(r.episode_results["success_wo_disturb"] for r in records)

        # Conditional: Ep-Len for PickUp (only episodes where pickup occurred)
        pickup_episodes = [
            r for r in records if r.episode_results["pickup_success"] > 0
        ]
        avg_eplen_pickup = (
            sum(r.episode_results["eplen_pickup"] for r in pickup_episodes) / len(pickup_episodes)
            if pickup_episodes else 0.0
        )

        # Conditional: Ep-Len for Success (only successful episodes)
        success_episodes = [
            r for r in records if r.episode_results["episode_success"] > 0
        ]
        avg_eplen_success = (
            sum(r.episode_results["eplen_success"] for r in success_episodes) / len(success_episodes)
            if success_episodes else 0.0
        )

        return {
            "episode_success_rate": round(ep_success_sum / n, 4),
            "pickup_success_rate": round(pickup_sum / n, 4),
            "success_wo_disturb_rate": round(wo_disturb_sum / n, 4),
            "avg_eplen_pickup": round(avg_eplen_pickup, 1),
            "avg_eplen_success": round(avg_eplen_success, 1),
            "avg_eplen": round(steps_sum / n, 1),
            # Convenience aliases
            "success_rate": round(ep_success_sum / n, 4),
            "num_episodes": n,
            "num_successful": len(success_episodes),
            "num_picked_up": len(pickup_episodes),
        }
