"""Tests for trajectory video generation."""
import json

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


def _make_episode_dir(parent: Path, name: str, success: float | None = 1.0, num_steps: int = 3):
    """Create a fake episode directory with minimal files."""
    ep_dir = parent / "episodes" / name
    ep_dir.mkdir(parents=True, exist_ok=True)

    # result.json
    result = {"success": success, "navigation_error": 2.5}
    (ep_dir / "result.json").write_text(json.dumps(result))

    # trajectory.jsonl
    lines = []
    # Reset entry
    lines.append(json.dumps({
        "step": 0, "type": "reset",
        "rgb_path": "step_0000.png",
        "agent_pose": [0, 0, 0, 0, 0, 0],
        "reward": 0.0, "done": False, "info": {},
    }))
    # Step entries with agent_position
    for i in range(1, num_steps + 1):
        pos = [float(i), 0.5, float(i) * 0.5]
        lines.append(json.dumps({
            "step": i, "type": "step",
            "action": "move_forward",
            "rgb_path": f"step_{i:04d}.png",
            "agent_pose": [0, 0, 0, 0, 0, 0],
            "reward": 0.0, "done": i == num_steps,
            "info": {"agent_position": json.dumps(pos), "geo_distance": "3.2"},
        }))
    (ep_dir / "trajectory.jsonl").write_text("\n".join(lines) + "\n")

    # Fake step images (8x8 red squares)
    from PIL import Image
    for i in range(num_steps + 1):
        img = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))
        img.save(str(ep_dir / f"step_{i:04d}.png"))

    # episode_meta.json
    meta = {
        "start_position": [0.0, 0.5, 0.0],
        "goal_position": [4.5, 0.5, 1.5],
    }
    (ep_dir / "episode_meta.json").write_text(json.dumps(meta))

    return ep_dir


class TestDiscoverEpisodes:
    def test_discover_all(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        _make_episode_dir(tmp_path, "000_1", success=1.0)
        _make_episode_dir(tmp_path, "001_2", success=0.0)
        eps = discover_episodes(tmp_path)
        assert len(eps) == 2

    def test_filter_success(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        _make_episode_dir(tmp_path, "000_1", success=1.0)
        _make_episode_dir(tmp_path, "001_2", success=0.0)
        _make_episode_dir(tmp_path, "002_3", success=None)
        eps = discover_episodes(tmp_path, filter_by="success")
        assert len(eps) == 1
        assert eps[0].name == "000_1"

    def test_filter_failed(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        _make_episode_dir(tmp_path, "000_1", success=1.0)
        _make_episode_dir(tmp_path, "001_2", success=0.0)
        _make_episode_dir(tmp_path, "002_3", success=None)
        eps = discover_episodes(tmp_path, filter_by="failed")
        assert len(eps) == 2

    def test_sample(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        for i in range(10):
            _make_episode_dir(tmp_path, f"{i:03d}_{i}", success=1.0)
        eps = discover_episodes(tmp_path, sample_n=3, seed=42)
        assert len(eps) == 3

    def test_sample_larger_than_available(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        for i in range(3):
            _make_episode_dir(tmp_path, f"{i:03d}_{i}", success=1.0)
        eps = discover_episodes(tmp_path, sample_n=10, seed=42)
        assert len(eps) == 3  # returns all, no error

    def test_sample_deterministic(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        for i in range(10):
            _make_episode_dir(tmp_path, f"{i:03d}_{i}", success=1.0)
        eps1 = discover_episodes(tmp_path, sample_n=3, seed=42)
        eps2 = discover_episodes(tmp_path, sample_n=3, seed=42)
        assert [e.name for e in eps1] == [e.name for e in eps2]

    def test_no_episodes_dir(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        eps = discover_episodes(tmp_path)
        assert eps == []

    def test_missing_result_json_skipped_when_filtering(self, tmp_path):
        from easi.analysis.trajectory_video import discover_episodes
        _make_episode_dir(tmp_path, "000_1", success=1.0)
        # Create dir with trajectory.jsonl but no result.json
        bad_dir = tmp_path / "episodes" / "001_bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "trajectory.jsonl").write_text('{"step":0}\n')
        eps = discover_episodes(tmp_path, filter_by="success")
        # 001_bad has trajectory.jsonl so it's discovered, but no result.json
        # so it should be skipped during filtering
        assert len(eps) == 1
        assert eps[0].name == "000_1"


class TestCoordinateProjection:
    def test_world_to_pixel(self):
        from easi.analysis.trajectory_video import world_to_pixel
        meta = {
            "bounds_lower": [0.0, 0.0, 0.0],
            "bounds_upper": [10.0, 3.0, 8.0],
            "meters_per_pixel": 0.1,
        }
        # Position at world origin -> pixel (0, 0)
        px, py = world_to_pixel(0.0, 0.0, meta)
        assert px == 0
        assert py == 0

        # Position at (5.0, 4.0) -> pixel (50, 40)
        px, py = world_to_pixel(5.0, 4.0, meta)
        assert px == 50
        assert py == 40

    def test_world_to_pixel_no_meta(self):
        from easi.analysis.trajectory_video import world_to_pixel_fallback
        positions = [[0.0, 0.0], [10.0, 8.0]]
        canvas_size = (200, 200)
        padding = 20
        px, py = world_to_pixel_fallback(5.0, 4.0, positions, canvas_size, padding)
        assert padding <= px <= canvas_size[0] - padding
        assert padding <= py <= canvas_size[1] - padding


class TestRenderEpisodeVideo:
    def test_generates_mp4(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import render_episode_video
        ep_dir = _make_episode_dir(tmp_path, "000_1", success=1.0, num_steps=3)
        output_dir = tmp_path / "analysis" / "videos"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "000_1.mp4"
        render_episode_video(ep_dir, output_path, fps=4)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generates_video_without_map(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import render_episode_video
        ep_dir = _make_episode_dir(tmp_path, "000_1", success=1.0, num_steps=3)
        # Remove topdown map files (should fall back to blank canvas)
        (ep_dir / "topdown_map.png").unlink(missing_ok=True)
        output_path = tmp_path / "out.mp4"
        render_episode_video(ep_dir, output_path, fps=4)
        assert output_path.exists()

    def test_skips_missing_trajectory(self, tmp_path, caplog):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import render_episode_video
        ep_dir = tmp_path / "episodes" / "000_bad"
        ep_dir.mkdir(parents=True)
        output_path = tmp_path / "out.mp4"
        render_episode_video(ep_dir, output_path, fps=4)
        assert not output_path.exists()

    def test_skips_zero_step_episode(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import render_episode_video
        ep_dir = tmp_path / "episodes" / "000_short"
        ep_dir.mkdir(parents=True)
        # Only reset entry, no steps
        (ep_dir / "trajectory.jsonl").write_text(
            json.dumps({"step": 0, "type": "reset", "rgb_path": "step_0000.png",
                         "agent_pose": [0,0,0,0,0,0], "reward": 0.0, "done": False, "info": {}})
            + "\n"
        )
        output_path = tmp_path / "out.mp4"
        render_episode_video(ep_dir, output_path, fps=4)
        assert not output_path.exists()

    def test_handles_missing_step_images(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import render_episode_video
        ep_dir = _make_episode_dir(tmp_path, "000_1", success=1.0, num_steps=3)
        # Delete middle step image
        (ep_dir / "step_0002.png").unlink()
        output_path = tmp_path / "out.mp4"
        render_episode_video(ep_dir, output_path, fps=4)
        assert output_path.exists()  # video still generated with placeholder


class TestCLI:
    def test_analyze_trajectory_generates_videos(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from easi.analysis.trajectory_video import generate_trajectory_videos
        _make_episode_dir(tmp_path, "000_1", success=1.0, num_steps=3)
        _make_episode_dir(tmp_path, "001_2", success=0.0, num_steps=2)
        generate_trajectory_videos(str(tmp_path), filter_by="success", fps=4)
        videos = list((tmp_path / "analysis" / "videos").glob("*.mp4"))
        assert len(videos) == 1
        assert videos[0].name == "000_1.mp4"

    def test_analyze_trajectory_nonexistent_dir(self, tmp_path, caplog):
        from easi.analysis.trajectory_video import generate_trajectory_videos
        generate_trajectory_videos(str(tmp_path / "nonexistent"))
        assert "not found" in caplog.text.lower()
