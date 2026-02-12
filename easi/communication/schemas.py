"""JSON schemas for command/response exchange between parent and bridge subprocess."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from easi.core.episode import Action, Observation, StepResult


# --- Command schemas (parent → child) ---

def make_reset_command(
    episode_id: str,
    reset_config: dict | None = None,
    episode_output_dir: str | None = None,
) -> dict:
    cmd = {
        "type": "reset",
        "episode_id": episode_id,
        "reset_config": reset_config or {},
    }
    if episode_output_dir is not None:
        cmd["episode_output_dir"] = episode_output_dir
    return cmd


def make_step_command(action: Action) -> dict:
    return {
        "type": "step",
        "action": {
            "action_name": action.action_name,
            "params": action.params,
        },
    }


def make_close_command() -> dict:
    return {"type": "close"}


# --- Response schemas (child → parent) ---

def make_observation_response(
    rgb_path: str,
    depth_path: str | None = None,
    agent_pose: list[float] | None = None,
    metadata: dict[str, str] | None = None,
    reward: float = 0.0,
    done: bool = False,
    info: dict[str, float] | None = None,
) -> dict:
    return {
        "status": "ok",
        "observation": {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "agent_pose": agent_pose or [],
            "metadata": metadata or {},
        },
        "reward": reward,
        "done": done,
        "info": info or {},
    }


def make_error_response(error: str) -> dict:
    return {"status": "error", "error": error}


def make_status_response(ready: bool) -> dict:
    return {"ready": ready}


# --- Parsing helpers ---

def parse_observation(data: dict) -> Observation:
    obs = data["observation"]
    return Observation(
        rgb_path=obs["rgb_path"],
        depth_path=obs.get("depth_path"),
        agent_pose=obs.get("agent_pose", []),
        metadata=obs.get("metadata", {}),
    )


def parse_step_result(data: dict) -> StepResult:
    return StepResult(
        observation=parse_observation(data),
        reward=data.get("reward", 0.0),
        done=data.get("done", False),
        info=data.get("info", {}),
    )


def parse_action_from_command(data: dict) -> Action:
    action_data = data["action"]
    return Action(
        action_name=action_data["action_name"],
        params=action_data.get("params", {}),
    )
