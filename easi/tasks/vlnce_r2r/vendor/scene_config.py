"""Habitat-Sim 0.1.7 configuration for VLN-CE R2R.

Single front RGB camera, 15-degree turns, 0.25m forward step.
Adapted from LHPR-VLN's scene_config.py for habitat-sim 0.1.7 API.
"""
from __future__ import annotations

import habitat_sim


def make_cfg(
    scene_path: str,
    gpu_device_id: int = -1,
    width: int = 480,
    height: int = 480,
    hfov: int = 90,
    sensor_height: float = 1.25,
    forward_step_size: float = 0.25,
    turn_angle: float = 15.0,
    allow_sliding: bool = True,
) -> habitat_sim.Configuration:
    """Build habitat-sim 0.1.7 Configuration for VLN-CE R2R."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = gpu_device_id
    sim_cfg.scene_id = scene_path
    sim_cfg.allow_sliding = allow_sliding

    # Single front-facing RGB sensor
    color_sensor = habitat_sim.SensorSpec()
    color_sensor.uuid = "color_sensor"
    color_sensor.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor.resolution = [height, width]
    color_sensor.position = [0.0, sensor_height, 0.0]
    color_sensor.parameters["hfov"] = str(hfov)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=forward_step_size)
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=turn_angle)
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=turn_angle)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
