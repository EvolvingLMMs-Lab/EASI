"""Tests for BaseBridge topdown map and episode metadata hooks."""
import json

import numpy as np
import pytest
from pathlib import Path

from easi.simulators.base_bridge import BaseBridge


class FakeEnv:
    def reset(self):
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action):
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}, 0.0, False, {}

    def close(self):
        pass


class BridgeWithMap(BaseBridge):
    """Bridge that returns a topdown map."""

    def _create_env(self, reset_config, simulator_kwargs):
        return FakeEnv()

    def _extract_image(self, obs):
        return obs["rgb"]

    def _get_topdown_map(self):
        rgb = np.full((100, 120, 3), 200, dtype=np.uint8)
        meta = {
            "bounds_lower": [0.0, 0.0, 0.0],
            "bounds_upper": [12.0, 3.0, 10.0],
            "meters_per_pixel": 0.1,
            "height": 1.25,
        }
        return rgb, meta

    def _get_episode_meta(self):
        return {
            "start_position": [1.0, 0.5, -2.0],
            "goal_position": [4.5, 0.5, 1.2],
            "gt_locations": [[1.0, 0.5, -2.0], [2.0, 0.5, -1.0], [4.5, 0.5, 1.2]],
        }


class BridgeWithoutMap(BaseBridge):
    """Bridge that returns no topdown map (default)."""

    def _create_env(self, reset_config, simulator_kwargs):
        return FakeEnv()

    def _extract_image(self, obs):
        return obs["rgb"]


class TestBaseBridgeHooksDefault:
    def test_get_topdown_map_returns_none(self, tmp_path):
        bridge = BridgeWithoutMap(workspace=tmp_path)
        assert bridge._get_topdown_map() is None

    def test_get_episode_meta_returns_none(self, tmp_path):
        bridge = BridgeWithoutMap(workspace=tmp_path)
        assert bridge._get_episode_meta() is None

    def test_reset_no_map_files(self, tmp_path):
        bridge = BridgeWithoutMap(workspace=tmp_path)
        ep_dir = tmp_path / "episode_001"
        ep_dir.mkdir()
        bridge.episode_output_dir = str(ep_dir)
        bridge.reset({})
        assert not (ep_dir / "topdown_map.png").exists()
        assert not (ep_dir / "episode_meta.json").exists()


class TestBaseBridgeHooksWithMap:
    def test_reset_saves_topdown_map(self, tmp_path):
        bridge = BridgeWithMap(workspace=tmp_path)
        ep_dir = tmp_path / "episode_001"
        ep_dir.mkdir()
        bridge.episode_output_dir = str(ep_dir)
        bridge.reset({})
        assert (ep_dir / "topdown_map.png").exists()
        assert (ep_dir / "topdown_map_meta.json").exists()
        meta = json.loads((ep_dir / "topdown_map_meta.json").read_text())
        assert meta["meters_per_pixel"] == 0.1
        assert len(meta["bounds_lower"]) == 3

    def test_reset_saves_episode_meta(self, tmp_path):
        bridge = BridgeWithMap(workspace=tmp_path)
        ep_dir = tmp_path / "episode_001"
        ep_dir.mkdir()
        bridge.episode_output_dir = str(ep_dir)
        bridge.reset({})
        assert (ep_dir / "episode_meta.json").exists()
        meta = json.loads((ep_dir / "episode_meta.json").read_text())
        assert meta["start_position"] == [1.0, 0.5, -2.0]
        assert meta["goal_position"] == [4.5, 0.5, 1.2]
        assert len(meta["gt_locations"]) == 3

    def test_reset_no_output_dir_skips_save(self, tmp_path):
        bridge = BridgeWithMap(workspace=tmp_path)
        bridge.episode_output_dir = None
        bridge.reset({})
        # No crash, no files saved outside workspace
        map_files = list(tmp_path.glob("topdown_map.*"))
        assert len(map_files) == 0
