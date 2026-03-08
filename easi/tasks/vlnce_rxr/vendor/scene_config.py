# easi/tasks/vlnce_rxr/vendor/scene_config.py
"""Habitat-Sim 0.1.7 configuration for VLN-CE RxR.

Single front RGB camera, 30-degree turns/tilts, 0.25m forward step.
6 actions: move_forward, turn_left, turn_right, look_up, look_down.
"""
from __future__ import annotations

import habitat_sim


def make_cfg(
    scene_path: str,
    gpu_device_id: int = -1,
    width: int = 640,
    height: int = 480,
    hfov: int = 79,
    sensor_height: float = 0.88,
    forward_step_size: float = 0.25,
    turn_angle: float = 30.0,
    tilt_angle: float = 30.0,
    allow_sliding: bool = False,
) -> habitat_sim.Configuration:
    """Build habitat-sim 0.1.7 Configuration for VLN-CE RxR."""
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
        "look_up": habitat_sim.ActionSpec(
            "look_up", habitat_sim.ActuationSpec(amount=tilt_angle)
        ),
        "look_down": habitat_sim.ActionSpec(
            "look_down", habitat_sim.ActuationSpec(amount=tilt_angle)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
