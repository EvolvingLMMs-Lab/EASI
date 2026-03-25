"""HAZARD bridge — manages disaster rescue scenarios via HAZARD envs.

This script runs inside the easi_tdw_v1_11_23 conda env (Python 3.10).
It communicates with the parent process via filesystem IPC.

HAZARD source is vendored at ``easi/tasks/hazard/vendor/HAZARD/``.
The vendor dir is added to sys.path so ``from HAZARD.*`` imports resolve.

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Add EASI repo root so easi imports work
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Add vendored HAZARD source to sys.path
_vendor_dir = str(Path(__file__).resolve().parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

from easi.simulators.base_bridge import BaseBridge
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Object values (matching actions.py)
HIGH_VALUE = 5
LOW_VALUE = 1


class HAZARDBridge(BaseBridge):
    """Bridge for HAZARD benchmark scenarios (fire/flood/wind)."""

    def __init__(self, workspace, simulator_kwargs=None):
        super().__init__(workspace, simulator_kwargs)

        self.scenario = self.simulator_kwargs.get("scenario", "fire")
        self._max_steps = self.simulator_kwargs.get("max_steps", 1500)
        self._screen_size = self.simulator_kwargs.get("screen_size", 512)
        self._port = self.simulator_kwargs.get("port", 1071)

        # HAZARD state (reset per episode)
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []
        self.target_status = {}
        self.target_ids = []
        self.target_categories = []
        self.object_list = []
        self.current_seen_objects_id = []
        self._last_action_result = False
        self._last_action_info = ""
        self._action_history = []  # list of (matched_plan_text, result, info)
        self._env_change_record = {}  # {str(obj_id): [temp/water_level values]}
        self._cached_plans = []       # plan descriptions from previous step
        self._cached_plan_actions = [] # action tuples from previous step

    @staticmethod
    def _configure_tdw_build_path():
        """Point TDW's Build.BUILD_PATH to our downloaded binary.

        The TDW Controller uses Build.BUILD_PATH to find and launch the
        Unity build when launch_build=True. By default it points to
        ~/tdw_build/TDW/TDW.x86_64. We override it to use the binary
        downloaded by TDWEnvManager (via TDW_BUILD_PATH env var).
        """
        build_dir = os.environ.get("TDW_BUILD_PATH")
        if not build_dir:
            logger.trace("TDW_BUILD_PATH not set, using TDW default build path")
            return

        binary = Path(build_dir) / "TDW.x86_64"
        logger.trace("TDW_BUILD_PATH=%s, binary exists=%s", build_dir, binary.exists())
        if not binary.exists():
            logger.warning(
                "TDW build binary not found at %s — TDW Controller will use its default path. "
                "Run 'easi env install tdw:v1_11_23' to download the build.",
                binary,
            )
            return

        from tdw.release.build import Build
        Build.BUILD_PATH = binary
        logger.info("Set TDW Build.BUILD_PATH to %s", binary)

    def reset(self, reset_config):
        """Reset env and return observation with initial info for prompt."""
        if self.env is None:
            self.env = self._create_env(reset_config, self.simulator_kwargs)
        self.step_count = 0
        obs = self._on_reset(self.env, reset_config)
        return self._make_response(obs, info=self._reset_info)

    def _create_env(self, reset_config, simulator_kwargs):
        """Create the appropriate HAZARD env (Fire/Flood/Wind).

        Point TDW's Build.BUILD_PATH to our downloaded binary (if available),
        then let the TDW Controller handle build launch via launch_build=True.
        """
        scenario = simulator_kwargs.get("scenario", "fire")
        port = simulator_kwargs.get("port", 1071)
        screen_size = simulator_kwargs.get("screen_size", 512)
        use_cached_assets = simulator_kwargs.get("use_cached_assets", False)
        use_gt = simulator_kwargs.get("use_gt", True)

        # Point TDW to our downloaded build binary
        self._configure_tdw_build_path()

        if scenario == "fire":
            from HAZARD.envs.fire import FireEnv
            env = FireEnv(
                launch_build=True, screen_size=screen_size, port=port,
                use_local_resources=use_cached_assets,
                check_version=False, use_gt=use_gt,
            )
        elif scenario == "flood":
            from HAZARD.envs.flood import FloodEnv
            env = FloodEnv(
                launch_build=True, screen_size=screen_size, port=port,
                use_local_resources=use_cached_assets,
                check_version=False, use_gt=use_gt,
            )
        elif scenario == "wind":
            from HAZARD.envs.wind import WindEnv
            env = WindEnv(
                launch_build=True, screen_size=screen_size, port=port,
                use_local_resources=use_cached_assets,
                check_version=False, use_gt=use_gt,
            )
        else:
            raise ValueError(f"Unknown HAZARD scenario: {scenario}")

        logger.info("Created %s env (port=%d, screen=%d)", scenario, port, screen_size)
        return env

    def _on_reset(self, env, reset_config):
        """Reset HAZARD env with episode data.

        Replicates the logic of fire_gym.py/flood_gym.py/wind_gym.py reset()
        with trace logging around each step to aid debugging.
        """
        # Reset bridge state
        self.holding_object = []
        self.nearest_object = None
        self.have_finished_list = []
        self._last_action_result = False
        self._last_action_info = ""
        self._action_history = []
        self._env_change_record = {}
        self._cached_plans = []
        self._cached_plan_actions = []

        source_dir = reset_config["source_dir"]
        logger.info("Resetting HAZARD env with source_dir: %s", source_dir)

        # --- Replicate env.reset() with trace logging ---
        from HAZARD.utils.scene_setup import SceneSetup

        logger.trace("Creating SceneSetup from data_dir=%s (scenario=%s)", source_dir, self.scenario)
        scene_kwargs = {"data_dir": source_dir}
        if self.scenario == "flood":
            scene_kwargs["is_flood"] = True
        env.setup = SceneSetup(**scene_kwargs)
        logger.trace("SceneSetup created (targets=%s)", getattr(env.setup, 'target_names', []))

        # Terminate existing controller if re-using the env
        if env.controller is not None:
            logger.trace("Terminating existing controller")
            env.controller.communicate({"$type": "terminate"})
            env.controller.socket.close()
            logger.trace("Existing controller terminated")

        # Create new controller — this is where TDW build launches + ZMQ connects.
        # TDW Controller.__init__ calls socket.recv() which blocks until the
        # build binary connects. If the build fails to start, this hangs forever.
        logger.trace(
            "Creating controller (scenario=%s, port=%s, launch_build=%s)",
            self.scenario, env.controller_args.get("port"),
            env.controller_args.get("launch_build"),
        )

        controller_cls = self._get_controller_class()
        env.controller = controller_cls(**env.controller_args)
        logger.trace("Controller created and connected to TDW build")

        env.controller.seed(env.RNG.randint(1000000))
        logger.info(
            "Loading scene (this may take several minutes on first run "
            "while TDW downloads 3D assets — subsequent runs will be faster)"
        )

        env.controller.init_scene(env.setup)
        logger.info("Scene loaded successfully")

        env.num_step = 0
        env.last_action = None
        env.last_target = None

        if not getattr(env, 'record_only', False):
            logger.trace("Performing initial turn_by(0)")
            env.controller.do_action(0, "turn_by", {"angle": 0})
            env.controller.next_key_frame()
            logger.trace("Initial turn complete")

        # Initialize target tracking
        self.target_ids = [int(tid) for tid in reset_config.get("target_object_ids", [])]
        self.target_categories = reset_config.get("target_categories", [])
        self.target_status = {tid: False for tid in self.target_ids}

        # Initial communicate + observation
        env.controller.communicate([])
        state = env.controller._obs()
        self._update_seen_objects(state)

        # Compute initial available plans so the first prompt has actions
        available_plans, plan_actions = self._get_available_plans()
        self._cached_plans = available_plans
        self._cached_plan_actions = plan_actions

        object_distances = self._compute_object_distances(state)

        self._reset_info = {
            "task_success": 0.0,
            "frame_count": 0.0,
            "max_steps": float(self._max_steps),
            "max_rescue_frame": 0.0,
            "last_action_success": 1.0,
            "feedback": "episode started",
            "holding_objects": json.dumps(self.holding_object),
            "available_plans": json.dumps(available_plans),
            "plan_actions": json.dumps(plan_actions),
            "targets_rescued": 0.0,
            "targets_total": float(len(self.target_status)),
            "value_score": 0.0,
            "max_value": 0.0,
            "rescued_count": 0.0,
            "damaged_count": 0.0,
            "object_list": json.dumps(self.object_list),
            "current_seen_objects_id": json.dumps(self.current_seen_objects_id),
            "object_distances": json.dumps(object_distances),
            "env_change_record": json.dumps({}),
            "target_categories": json.dumps(self.target_categories),
        }

        return self._wrap_obs(state, is_reset=True)

    def _get_controller_class(self):
        """Return the correct AgentController class for the current scenario."""
        if self.scenario == "fire":
            from HAZARD.envs.fire.fireagent_controller import FireAgentController
            return FireAgentController
        elif self.scenario == "flood":
            from HAZARD.envs.flood.floodagent_controller import FloodAgentController
            return FloodAgentController
        elif self.scenario == "wind":
            from HAZARD.envs.wind.windagent_controller import WindAgentController
            return WindAgentController
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def _on_step(self, env, action_text):
        """Execute a HAZARD action and return (obs, reward, done, info)."""
        from HAZARD.policy.env_actions import (
            agent_walk_to, agent_pickup, agent_drop, agent_explore,
        )

        # Parse action_text against CACHED plans from previous step
        action_tuple, matched_plan = self._parse_action(action_text)

        if action_tuple is None:
            # Failed to parse -> treat as explore
            action_tuple = ("explore", None)
            matched_plan = "look around"

        # Execute the action
        action_type, action_target = action_tuple
        action_result, action_info = self._execute_action(
            env, action_type, action_target
        )
        self._last_action_result = action_result
        self._last_action_info = action_info
        # Store matched plan text (not raw LLM text) for explore condition
        self._action_history.append((matched_plan, action_result, action_info))

        # Handle state updates based on action
        if action_type == "walk_to" and action_result:
            self.nearest_object = int(action_target) if action_target else None
        elif action_type == "walk_to" and not action_result:
            self.nearest_object = None
        elif action_type == "pick_up" and action_result:
            self._do_hold_object(env)
        elif action_type == "drop" and action_result:
            self._do_drop_object(env, action_target)

        # Chain auto-actions
        self._execute_auto_actions(env)

        # Get new observation
        state = env.controller._obs()
        self._update_seen_objects(state)
        self._update_object_status(state)  # Track temperature/water_level

        # Check done
        frame_count = env.controller.frame_count
        all_targets_done = all(self.target_status.values())
        done = frame_count >= self._max_steps or all_targets_done

        # Compute available plans for NEXT step and cache them
        available_plans, plan_actions = self._get_available_plans()
        self._cached_plans = available_plans
        self._cached_plan_actions = plan_actions

        # Compute scoring (Value + Damage metrics from HAZARD paper)
        value_score, max_value, rescued_count, damaged_count = self._compute_scores()

        # Compute max rescue frame (latest frame any target was rescued)
        # Paper "Step" metric uses max(rescue_frame_per_target) per episode
        rescue_frames = [f for f in self.target_status.values() if f]
        max_rescue_frame = float(max(rescue_frames)) if rescue_frames else 0.0

        # Compute per-object distances for prompt builder
        object_distances = self._compute_object_distances(state)

        # Snapshot env_change_record: last value per object for prompt
        env_record_snapshot = {}
        for obj_id, values in self._env_change_record.items():
            if values:
                env_record_snapshot[obj_id] = values[-1]

        # Full history for object state history section
        env_record_history = {}
        for obj_id, values in self._env_change_record.items():
            if values:
                env_record_history[obj_id] = values

        # Build info
        info = {
            "task_success": float(all_targets_done),
            "frame_count": float(frame_count),
            "max_steps": float(self._max_steps),
            "max_rescue_frame": max_rescue_frame,     # for paper "Step" metric
            "last_action_success": float(action_result),
            "feedback": str(action_info),
            "holding_objects": json.dumps(self.holding_object),
            "available_plans": json.dumps(available_plans),
            "plan_actions": json.dumps(plan_actions),
            "targets_rescued": float(sum(1 for v in self.target_status.values() if v)),
            "targets_total": float(len(self.target_status)),
            # HAZARD paper metrics (Value + Damage)
            "value_score": float(value_score),       # sum(base_value * discount)
            "max_value": float(max_value),            # sum(base_value for all targets)
            "rescued_count": float(rescued_count),    # total objects rescued
            "damaged_count": float(damaged_count),    # rescued objects that were damaged
            # Data for prompt builder (STATE section)
            "object_list": json.dumps(self.object_list),
            "current_seen_objects_id": json.dumps(self.current_seen_objects_id),
            "object_distances": json.dumps(object_distances),
            "env_change_record": json.dumps(env_record_snapshot),
            "env_change_record_history": json.dumps(env_record_history),
            "target_categories": json.dumps(self.target_categories),
        }

        obs = self._wrap_obs(state)
        return obs, 0.0, done, info

    def _extract_image(self, obs):
        """Extract RGB numpy array from HAZARD observation."""
        rgb = obs.get("rgb_array")
        if rgb is not None:
            return rgb
        return np.zeros((self._screen_size, self._screen_size, 3), dtype=np.uint8)

    def _extract_info(self, info):
        """Pass through all info keys (already cleaned)."""
        return info or {}

    # --- Action execution ---

    def _execute_action(self, env, action_type, action_target):
        """Execute a single HAZARD action, return (success, info_text)."""
        from HAZARD.policy.env_actions import (
            agent_walk_to, agent_pickup, agent_drop, agent_explore,
        )

        if action_type == "walk_to":
            target_id = self._resolve_target_id(action_target)
            if target_id is None:
                return False, "invalid target id"
            return agent_walk_to(
                env, target=target_id,
                max_steps=100, reset_arms=False, arrived_at=0.5,
                task=self.scenario, effect_on_agents=False,
            )
        elif action_type == "pick_up":
            target_id = action_target
            if target_id is not None:
                self.nearest_object = int(target_id)
            if self.nearest_object is None:
                return False, "no nearest object to pick up"
            real_id = self._id_reverse_renumber(self.nearest_object)
            return agent_pickup(env, real_id, env_type=self.scenario)
        elif action_type == "drop":
            if not self.holding_object:
                return False, "not holding any object"
            if action_target is not None:
                real_id = self._id_reverse_renumber(int(action_target))
                return agent_drop(env, real_id, env_type=self.scenario)
            else:
                return agent_drop(env, env_type=self.scenario)
        elif action_type == "explore":
            return agent_explore(env)
        elif action_type == "stop":
            return True, "stopped"
        else:
            return False, f"unknown action type: {action_type}"

    # --- Auto-actions ---

    def _execute_auto_actions(self, env):
        """Chain automatic pickup/drop actions based on current state."""
        max_chain = 5  # prevent infinite loops
        for _ in range(max_chain):
            auto = self._check_auto_action()
            if auto is None:
                break
            action_type, action_target = auto
            logger.info("Auto-action: %s %s", action_type, action_target)
            result, info = self._execute_action(env, action_type, action_target)
            if action_type == "pick_up" and result:
                self._do_hold_object(env)
            elif action_type == "drop" and result:
                self._do_drop_object(env, action_target)
            self._action_history.append((f"auto_{action_type}", result, info))

    def _check_auto_action(self):
        """Check if an auto-action should fire. Returns (type, target) or None.

        Auto-action rules (from HAZARD/src/HAZARD/policy/llm.py:choose_target):
        - Wind: If not holding + nearest is target -> pick_up
        - Wind: If holding + last walk_to cart succeeded -> drop into cart
        - Fire/Flood: If holding target -> drop (into bag)
        - Fire/Flood: If nearest is target -> pick_up
        """
        if self.scenario == "wind":
            if (not self.holding_object
                    and self.nearest_object is not None
                    and self._is_target_object(self.nearest_object)):
                return ("pick_up", self.nearest_object)
            if (self.holding_object
                    and self._last_action_result
                    and self._action_history
                    and self._is_container(self.nearest_object)):
                return ("drop", self.nearest_object)
        else:
            # Fire/Flood
            if (self.holding_object
                    and self.holding_object[0].get("category") in self.target_categories):
                return ("drop", None)
            if (self.nearest_object is not None
                    and not self.holding_object
                    and self._is_target_object(self.nearest_object)):
                return ("pick_up", self.nearest_object)
        return None

    # --- State management ---

    def _do_hold_object(self, env):
        """Record picking up the nearest object."""
        if self.nearest_object is None:
            return
        real_id = self._id_reverse_renumber(self.nearest_object)
        name = env.controller.manager.segm.names.get(real_id, "unknown")
        category = env.controller.manager.segm.categories.get(real_id, "unknown")
        self.holding_object.append({
            "name": name, "category": category,
            "id": str(self.nearest_object),
        })
        self.nearest_object = None

    def _do_drop_object(self, env, container_target):
        """Record dropping the held object."""
        if not self.holding_object:
            return
        obj_id = int(self.holding_object[0]["id"])
        real_id = self._id_reverse_renumber(obj_id)
        if real_id in self.target_status:
            self.target_status[real_id] = env.controller.frame_count
            self.have_finished_list.append(obj_id)
        else:
            self.nearest_object = obj_id
        self.holding_object = []

    def _is_target_object(self, renumbered_id):
        """Check if a renumbered object ID corresponds to a target."""
        real_id = self._id_reverse_renumber(renumbered_id)
        return real_id in self.target_status

    def _is_container(self, renumbered_id):
        """Check if object is a container (wind scenario)."""
        if renumbered_id is None:
            return False
        real_id = self._id_reverse_renumber(renumbered_id)
        return hasattr(self.env, 'controller') and real_id in getattr(
            self.env.controller, 'containers', []
        )

    def _id_reverse_renumber(self, renumbered_id):
        """Map renumbered ID back to original ID."""
        if not hasattr(self.env.controller, 'manager'):
            return renumbered_id
        id_map = self.env.controller.manager.id_renumbering
        for orig, renum in id_map.items():
            if renum == renumbered_id:
                return orig
        return renumbered_id

    def _resolve_target_id(self, target):
        """Resolve a target (could be renumbered or original ID)."""
        if target is None:
            return None
        return self._id_reverse_renumber(int(target))

    def _update_seen_objects(self, state):
        """Update the list of currently visible object IDs."""
        seg_mask = state["raw"]["seg_mask"]
        self.current_seen_objects_id = [
            str(x) for x in set(seg_mask.flatten()) if int(x) != 0
        ]

    def _update_object_status(self, state):
        """Track per-object temperature/water_level from sensor data.

        Matches original llm.py:update_object_status() exactly.
        For fire/flood: computes average log_temp from segmentation mask.
        For wind: no-op.
        """
        if self.scenario not in ("fire", "flood"):
            return
        for o_id in self.current_seen_objects_id:
            obj_id = int(o_id)
            obj_mask = (state["raw"]["seg_mask"] == obj_id)
            if isinstance(obj_mask, np.ndarray):
                temp = state["raw"]["log_temp"] * obj_mask
                mask_sum = obj_mask.sum()
            else:
                temp = state["raw"]["log_temp"] * obj_mask.cpu().numpy()
                mask_sum = obj_mask.sum().item()
            if mask_sum == 0:
                continue
            avg_temp = float(temp.sum() / mask_sum)
            if str(obj_id) not in self._env_change_record:
                self._env_change_record[str(obj_id)] = [avg_temp]
            else:
                self._env_change_record[str(obj_id)].append(avg_temp)

    def _compute_object_distances(self, state):
        """Compute per-object distances from agent on semantic map.

        Matches original llm.py:get_object_location_description().
        Returns: dict {str(obj_id): float distance}
        """
        distances = {}
        # Get agent position on semantic map
        agent_map = state["goal_map"]
        agent_points = (agent_map == -2).nonzero()
        if isinstance(agent_points[0], np.ndarray):
            agent_pos = (agent_points[0].astype(float).mean(),
                         agent_points[1].astype(float).mean())
        else:
            agent_pos = (agent_points[:, 0].float().mean().item(),
                         agent_points[:, 1].float().mean().item())

        id_map = state["sem_map"]["explored"] * state["sem_map"]["id"]
        for obj in self.object_list:
            idx = int(obj["id"])
            object_points = (id_map == idx).nonzero()
            if isinstance(object_points[0], np.ndarray):
                center = (object_points[0].astype(float).mean(),
                          object_points[1].astype(float).mean())
            else:
                center = (object_points[:, 0].float().mean().item(),
                          object_points[:, 1].float().mean().item())
            dist = float(np.linalg.norm(
                np.array([agent_pos[0] - center[0], agent_pos[1] - center[1]])
            ))
            distances[str(idx)] = round(dist, 2)
        return distances

    # --- Available plans ---

    def _get_available_plans(self):
        """Compute available plans for the current state.

        Reference: HAZARD/src/HAZARD/policy/llm.py:get_available_plans()
        Returns: (plan_descriptions: list[str], plan_actions: list[tuple])
        """
        plans = []
        actions = []

        # Build object list from explored objects
        state = self.env.controller._obs()
        explored = state["sem_map"]["explored"] * state["sem_map"]["id"]
        explored_ids = [int(idx) for idx in set(explored.flatten()) if int(idx) != 0]

        object_list = []
        for idx in explored_ids:
            real_id = self._id_reverse_renumber(idx)
            name = self.env.controller.manager.segm.names.get(real_id, "unknown")
            category = self.env.controller.manager.segm.categories.get(real_id, "unknown")
            object_list.append({"name": name, "category": category, "id": str(idx)})
        self.object_list = object_list

        if not self.holding_object:
            # Can walk to target objects
            for obj in object_list:
                if obj["category"] not in self.target_categories:
                    continue
                if int(obj["id"]) in self.have_finished_list:
                    continue
                plans.append(f"go pick up object <{obj['category']}> ({obj['id']})")
                actions.append(("walk_to", obj["id"]))
        else:
            # Can drop or walk to container
            if self.scenario == "wind":
                for obj in object_list:
                    if obj["category"] != "shopping cart":
                        continue
                    plans.append(f"go put object into <{obj['category']}> ({obj['id']})")
                    actions.append(("walk_to", obj["id"]))
            else:
                plans.append("put the holding object in my bag")
                actions.append(("drop", None))

        # Explore option (if last action wasn't "look around")
        # Matches original: self.action_history[-1] != 'look around'
        if (not self._action_history
                or self._action_history[-1][0] != "look around"):
            plans.append("look around")
            actions.append(("explore", None))

        # Fallback: walk to any visible object
        if not actions:
            for obj in object_list:
                plans.append(f"go to object <{obj['category']}> ({obj['id']})")
                actions.append(("walk_to", obj["id"]))

        if not actions:
            plans.append("look around")
            actions.append(("explore", None))

        # Limit to 10 options
        plans = plans[:10]
        actions = actions[:10]

        return plans, [list(a) for a in actions]

    # --- Scoring (HAZARD paper: Value + Damage) ---

    def _compute_scores(self):
        """Compute Value and Damage metrics per HAZARD paper.

        Reference: HAZARD/src/HAZARD/utils/calc_value.py

        Returns: (value_score, max_value, rescued_count, damaged_count)
        - value_score: sum(base_value * discount) for rescued objects
          where discount=1.0 if undamaged, 0.5 if damaged
        - max_value: sum(base_value) for all target objects
        - rescued_count: number of objects successfully rescued
        - damaged_count: number of rescued objects that were damaged
        """
        value_dict = self._load_value_dict()
        value_score = 0.0
        max_value = 0.0
        rescued_count = 0
        damaged_count = 0

        for target_id in self.target_status:
            # Get object name/category for value lookup
            name = self.env.controller.target_id2name.get(target_id, "unknown")
            base_value = HIGH_VALUE if value_dict.get(name) == 1 else LOW_VALUE
            max_value += base_value

            if not self.target_status[target_id]:
                continue  # Not rescued

            rescued_count += 1

            # Compute discount based on object state at time of rescue
            discount = self._get_damage_discount(target_id)
            value_score += base_value * discount

            if discount < 0.6:  # damaged (discount == 0.5)
                damaged_count += 1

        return value_score, max_value, rescued_count, damaged_count

    def _get_damage_discount(self, target_id):
        """Get damage discount for a rescued object.

        Returns 1.0 if undamaged, 0.5 if damaged.

        Reference: HAZARD/src/HAZARD/utils/calc_value.py:get_values()
        """
        if self.scenario == "fire":
            from HAZARD.envs.fire.fire_utils import ObjectState as FireObjectState
            obj = self.env.controller.manager.objects.get(target_id)
            if obj and obj.state == FireObjectState.NORMAL:
                return 1.0
            return 0.5
        elif self.scenario == "flood":
            from HAZARD.envs.flood.utils import ObjectState as FloodObjectState
            obj = self.env.controller.manager.objects.get(target_id)
            if obj is None:
                return 1.0
            # Check waterproof
            name = self.env.controller.target_id2name.get(target_id, "")
            fluid_dict = self._load_fluid_dict()
            if fluid_dict.get(name, 0) == 1:
                return 1.0  # Waterproof -> always full value
            if obj.state in (FloodObjectState.NORMAL, FloodObjectState.FLOATING):
                return 1.0
            return 0.5
        else:
            # Wind: no damage mechanic
            return 1.0

    def _load_value_dict(self):
        """Load object value metadata (cached)."""
        if not hasattr(self, '_value_dict_cache'):
            config_path = Path(__file__).parent / "config" / "value.json"
            self._value_dict_cache = json.loads(config_path.read_text())
        return self._value_dict_cache

    def _load_fluid_dict(self):
        """Load waterproof metadata (cached)."""
        if not hasattr(self, '_fluid_dict_cache'):
            config_path = Path(__file__).parent / "config" / "fluid.json"
            self._fluid_dict_cache = json.loads(config_path.read_text())
        return self._fluid_dict_cache

    # --- Action parsing ---

    def _parse_action(self, action_text):
        """Parse LLM's action text into (action_tuple, matched_plan_text).

        Uses CACHED plans from the previous step (what the LLM actually saw),
        not freshly recomputed plans (which may have changed due to state drift).

        Returns: (action_tuple, matched_plan_text) or (None, None)
        """
        if not action_text:
            return None, None

        # Match against cached plans (what the LLM was shown)
        available_plans = self._cached_plans
        plan_actions = self._cached_plan_actions

        if not available_plans:
            # First step or no cached plans -- keyword fallback only
            return self._keyword_fallback(action_text)

        # Exact match
        for i, plan in enumerate(available_plans):
            if plan == action_text:
                return tuple(plan_actions[i]), plan

        # Option letter match (A, B, C, ...)
        text = action_text.strip()
        for i, plan in enumerate(available_plans):
            option = chr(ord('A') + i)
            if (text == option or text == f"Option {option}" or text == f"{option}."
                    or f"option {option}" in text.lower()):
                return tuple(plan_actions[i]), plan

        # Fuzzy match: check if action text contains plan keywords
        for i, plan in enumerate(available_plans):
            if plan.lower() in action_text.lower():
                return tuple(plan_actions[i]), plan

        # Keyword-based fallback
        return self._keyword_fallback(action_text)

    def _keyword_fallback(self, action_text):
        """Fallback parsing when no cached plan matches.

        Returns: (action_tuple, plan_text) or (None, None)
        """
        import re

        lower = action_text.lower()
        if "look around" in lower or "explore" in lower:
            return ("explore", None), "look around"
        if "put" in lower and "bag" in lower:
            return ("drop", None), "put the holding object in my bag"
        if "pick" in lower:
            id_match = re.search(r'\((\d+)\)', action_text)
            if id_match:
                return ("walk_to", id_match.group(1)), action_text
        if "go" in lower or "walk" in lower:
            id_match = re.search(r'\((\d+)\)', action_text)
            if id_match:
                return ("walk_to", id_match.group(1)), action_text

        return None, None

    # --- Observation wrapping ---

    def _wrap_obs(self, state, is_reset=False):
        """Wrap HAZARD state dict into a simple obs dict for BaseBridge."""
        rgb = state["raw"]["rgb"]  # (3, H, W) float32 0-1
        rgb_array = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        return {"rgb_array": rgb_array}


if __name__ == "__main__":
    HAZARDBridge.main()
