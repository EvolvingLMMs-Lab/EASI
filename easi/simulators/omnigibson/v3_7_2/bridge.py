"""Bridge subprocess for OmniGibson v3.7.2 + Isaac Sim 4.5.0.

This script runs inside the easi_omnigibson_v3_7_2 conda env (Python 3.10).
It communicates with the parent process via filesystem IPC.

Provides an OmniGibsonBridge that handles:
- Minimal OmniGibson environment creation (Scene + DummyTask, no dataset)
- Action execution and observation capture
- Main IPC loop (reset/step/close)

Headless mode is enabled via OMNIGIBSON_HEADLESS=1 env var (set by env_manager).

Usage:
    python bridge.py --workspace /tmp/easi_xxx [--simulator-kwargs '{}']
"""
from __future__ import annotations

import numpy as np

from easi.simulators.base_bridge import BaseBridge


class OmniGibsonBridge(BaseBridge):
    """OmniGibson bridge using BaseBridge pattern.

    Creates a minimal OmniGibson environment with Scene (empty floor + skybox)
    and DummyTask for smoke testing. No BEHAVIOR-1K dataset required.
    """

    def _create_env(self, reset_config, simulator_kwargs):
        """Create minimal OmniGibson env (no dataset needed).

        Args:
            reset_config: Episode reset configuration.
            simulator_kwargs: From task YAML's simulator_configs.
                May contain 'og_config' to override the default config.
        """
        import omnigibson as og
        from omnigibson.macros import gm

        gm.ENABLE_FLATCACHE = True
        gm.USE_GPU_DYNAMICS = False

        cfg = {
            "scene": {"type": "Scene"},
            "robots": [{
                "type": "R1",
                "obs_modalities": ["rgb"],
                "action_type": "continuous",
                "action_normalize": True,
            }],
            "task": {"type": "DummyTask"},
        }
        # Allow full config override from simulator_kwargs
        if "og_config" in simulator_kwargs:
            cfg = simulator_kwargs["og_config"]

        env = og.Environment(configs=cfg)
        return env

    def _on_reset(self, env, reset_config):
        """Reset the OmniGibson environment."""
        obs, info = env.reset()
        return obs

    def _on_step(self, env, action_text):
        """Execute zero action (for smoke test) or parse action_text.

        For smoke testing, sends a zero action vector. Task-specific
        subclasses should override for meaningful action parsing.
        """
        import torch as th

        action_dim = env.action_space.shape[0]
        action = th.zeros(action_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        return obs, float(reward), terminated or truncated, info

    def _extract_image(self, obs):
        """Auto-discover first RGB camera from nested OmniGibson observation.

        OmniGibson observations are nested: obs[robot_key][sensor_key].
        We find the first sensor key containing 'rgb' and return it.
        """
        for robot_key in obs:
            if not isinstance(obs[robot_key], dict):
                continue
            for sensor_key in obs[robot_key]:
                if "rgb" in sensor_key.lower():
                    img = obs[robot_key][sensor_key]
                    # Convert torch tensor to numpy if needed
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    # Convert float [0,1] to uint8 [0,255]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    # Return RGB only (drop alpha if present)
                    return img[:, :, :3]
        # Fallback: return black image
        return np.zeros((128, 128, 3), dtype=np.uint8)


if __name__ == "__main__":
    OmniGibsonBridge.main()
