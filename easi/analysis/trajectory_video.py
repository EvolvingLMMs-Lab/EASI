"""Trajectory video generator for post-evaluation analysis.

Generates per-episode videos showing the robot's path on a top-down map
alongside the agent's camera view. No simulator dependencies — pure
post-processing from episode output directories.

Requires: opencv-python-headless (optional dependency)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _require_cv2():
    """Import cv2 with helpful error if missing."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python-headless is required for trajectory video generation.\n"
            "Install it with: pip install opencv-python-headless"
        )


def discover_episodes(
    run_dir: Path | str,
    filter_by: str | None = None,
    sample_n: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Discover and filter episode directories in a run.

    Args:
        run_dir: Path to evaluation run directory.
        filter_by: "success" or "failed" to filter by outcome.
        sample_n: Randomly sample N episodes after filtering.
        seed: Random seed for sampling.

    Returns:
        Sorted list of episode directory paths.
    """
    run_dir = Path(run_dir)
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.is_dir():
        logger.warning("No episodes/ directory found in %s", run_dir)
        return []

    episode_dirs = sorted(
        d for d in episodes_dir.iterdir()
        if d.is_dir() and (d / "trajectory.jsonl").exists()
    )

    if filter_by:
        filtered = []
        for ep_dir in episode_dirs:
            result_path = ep_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                result = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            success = result.get("success")
            if filter_by == "success" and success == 1.0:
                filtered.append(ep_dir)
            elif filter_by == "failed" and success != 1.0:
                filtered.append(ep_dir)
        episode_dirs = filtered

    if sample_n is not None and sample_n < len(episode_dirs):
        episode_dirs = random.Random(seed).sample(episode_dirs, sample_n)
        episode_dirs.sort()

    return episode_dirs


def world_to_pixel(
    world_x: float, world_z: float, map_meta: dict
) -> tuple[int, int]:
    """Project world [x, z] to pixel coords using map metadata.

    Habitat-Sim uses Y-up coordinates. The floor plane is [x, z].
    """
    bounds_lower = map_meta["bounds_lower"]
    mpp = map_meta["meters_per_pixel"]
    px = int((world_x - bounds_lower[0]) / mpp)
    pz = int((world_z - bounds_lower[2]) / mpp)
    return px, pz


def world_to_pixel_fallback(
    world_x: float,
    world_z: float,
    all_positions: list[list[float]],
    canvas_size: tuple[int, int],
    padding: int = 20,
) -> tuple[int, int]:
    """Project world coords to pixel coords on a blank canvas.

    Computes bounding box from all positions and maps linearly.
    """
    xs = [p[0] for p in all_positions]
    zs = [p[1] for p in all_positions]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)

    # Avoid division by zero for single-point paths
    x_range = max(x_max - x_min, 0.01)
    z_range = max(z_max - z_min, 0.01)

    draw_w = canvas_size[0] - 2 * padding
    draw_h = canvas_size[1] - 2 * padding

    px = int(padding + (world_x - x_min) / x_range * draw_w)
    pz = int(padding + (world_z - z_min) / z_range * draw_h)
    return px, pz


def _load_trajectory(ep_dir: Path) -> list[dict]:
    """Load trajectory.jsonl entries."""
    path = ep_dir / "trajectory.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _parse_positions(entries: list[dict]) -> list[list[float] | None]:
    """Extract [x, z] positions from trajectory entries.

    Returns a list parallel to entries. Reset entry (step 0) has None
    since its info is empty — start position comes from episode_meta.json.
    """
    positions = []
    for entry in entries:
        info = entry.get("info", {})
        raw = info.get("agent_position")
        if raw is not None:
            pos_3d = json.loads(raw) if isinstance(raw, str) else raw
            positions.append([pos_3d[0], pos_3d[2]])  # [x, z] floor plane
        else:
            positions.append(None)
    return positions


def render_episode_video(
    ep_dir: Path,
    output_path: Path,
    fps: int = 4,
) -> None:
    """Render a trajectory video for one episode.

    Args:
        ep_dir: Path to episode directory containing trajectory.jsonl, step_*.png, etc.
        output_path: Where to write the MP4 video.
        fps: Frames per second.
    """
    cv2 = _require_cv2()
    from PIL import Image

    traj_path = ep_dir / "trajectory.jsonl"
    if not traj_path.exists():
        logger.warning("No trajectory.jsonl in %s, skipping", ep_dir)
        return

    entries = _load_trajectory(ep_dir)
    if len(entries) < 2:
        logger.warning("Trajectory too short in %s, skipping", ep_dir)
        return

    positions = _parse_positions(entries)

    # Load episode metadata (start position, goal, gt_locations)
    meta_path = ep_dir / "episode_meta.json"
    ep_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    start_pos = ep_meta.get("start_position")
    goal_pos = ep_meta.get("goal_position")
    gt_locations = ep_meta.get("gt_locations")

    # Start position as [x, z]
    start_xz = [start_pos[0], start_pos[2]] if start_pos else None

    # Goal position as [x, z]
    goal_xz = [goal_pos[0], goal_pos[2]] if goal_pos else None

    # GT path as [[x, z], ...]
    gt_xz = [[p[0], p[2]] for p in gt_locations] if gt_locations else None

    # Load topdown map or create blank canvas
    map_path = ep_dir / "topdown_map.png"
    map_meta_path = ep_dir / "topdown_map_meta.json"
    has_map = map_path.exists() and map_meta_path.exists()

    if has_map:
        map_img = np.array(Image.open(map_path).convert("RGB"))
        map_meta = json.loads(map_meta_path.read_text())
    else:
        map_img = None
        map_meta = None

    # Collect all valid [x, z] positions for bounding box fallback
    all_xz = [p for p in positions if p is not None]
    if start_xz:
        all_xz.insert(0, start_xz)
    if goal_xz:
        all_xz.append(goal_xz)

    if not all_xz:
        logger.warning("No positions found in %s, skipping", ep_dir)
        return

    # Determine panel height from first step image
    first_img_path = ep_dir / entries[0].get("rgb_path", "step_0000.png")
    if first_img_path.exists():
        cam_h = np.array(Image.open(first_img_path)).shape[0]
    else:
        cam_h = 480
    panel_h = cam_h

    # Blank canvas fallback
    if map_img is None:
        map_img = np.full((panel_h, panel_h, 3), 40, dtype=np.uint8)

    # Resize map to match panel height
    scale = panel_h / map_img.shape[0]
    map_w = int(map_img.shape[1] * scale)
    map_base = cv2.resize(map_img, (map_w, panel_h))

    # Load result for final frame overlay
    result_path = ep_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}

    # Helper: project world coord to map pixel
    def to_pixel(x, z):
        if map_meta:
            px, pz = world_to_pixel(x, z, map_meta)
            return int(px * scale), int(pz * scale)
        else:
            return world_to_pixel_fallback(x, z, all_xz, (map_w, panel_h))

    # Set up video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_w = map_w + cam_h  # map_panel + camera_panel (camera is square)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, panel_h))

    if not writer.isOpened():
        logger.error("Failed to open video writer for %s", output_path)
        return

    try:
        path_so_far = []
        if start_xz:
            path_so_far.append(start_xz)

        for i, entry in enumerate(entries):
            # Update path
            if positions[i] is not None:
                path_so_far.append(positions[i])

            # Draw map panel
            map_frame = map_base.copy()

            # Draw GT path (dashed, faint)
            if gt_xz and len(gt_xz) >= 2:
                for j in range(len(gt_xz) - 1):
                    p1 = to_pixel(*gt_xz[j])
                    p2 = to_pixel(*gt_xz[j + 1])
                    # Dashed line: draw every other segment
                    if j % 2 == 0:
                        cv2.line(map_frame, p1, p2, (180, 180, 180), 1)

            # Draw agent path (solid, growing)
            if len(path_so_far) >= 2:
                for j in range(len(path_so_far) - 1):
                    p1 = to_pixel(*path_so_far[j])
                    p2 = to_pixel(*path_so_far[j + 1])
                    cv2.line(map_frame, p1, p2, (0, 200, 0), 2)

            # Draw start (blue circle)
            if start_xz:
                sp = to_pixel(*start_xz)
                cv2.circle(map_frame, sp, 6, (255, 100, 100), -1)

            # Draw goal (red circle)
            if goal_xz:
                gp = to_pixel(*goal_xz)
                cv2.circle(map_frame, gp, 6, (100, 100, 255), -1)

            # Draw current position (green arrowhead)
            if path_so_far:
                cp = to_pixel(*path_so_far[-1])
                cv2.circle(map_frame, cp, 5, (0, 255, 0), -1)

            # Step/distance text on map panel
            step_text = f"Step: {entry.get('step', i)}"
            cv2.putText(map_frame, step_text, (10, panel_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            geo_dist = entry.get("info", {}).get("geo_distance")
            if geo_dist:
                dist_text = f"Dist: {geo_dist}m"
                cv2.putText(map_frame, dist_text, (10, panel_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Load camera panel
            rgb_name = entry.get("rgb_path", f"step_{i:04d}.png")
            cam_path = ep_dir / rgb_name
            if cam_path.exists():
                cam_img = np.array(Image.open(cam_path).convert("RGB"))
                cam_img = cv2.resize(cam_img, (cam_h, panel_h))
            else:
                logger.warning("Missing image %s, using placeholder", cam_path.name)
                cam_img = np.full((panel_h, cam_h, 3), 30, dtype=np.uint8)

            # Action overlay on camera panel
            action = entry.get("action", "")
            if action:
                cv2.putText(cam_img, action, (10, panel_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Final frame: overlay outcome
            if i == len(entries) - 1 and result:
                success = result.get("success")
                if success == 1.0:
                    label = "SUCCESS"
                    color = (0, 255, 0)
                elif success is not None:
                    label = "FAILURE"
                    color = (0, 0, 255)
                else:
                    label = "NO GOAL"
                    color = (200, 200, 200)
                cv2.putText(cam_img, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Concatenate panels
            frame = np.concatenate([map_frame, cam_img], axis=1)

            # OpenCV uses BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    logger.trace("Wrote %s (%d frames)", output_path, len(entries))


def generate_trajectory_videos(
    run_dir: str,
    filter_by: str | None = None,
    sample_n: int | None = None,
    fps: int = 4,
    seed: int = 42,
) -> None:
    """Generate trajectory videos for all matching episodes in a run.

    Args:
        run_dir: Path to evaluation run directory.
        filter_by: "success" or "failed".
        sample_n: Randomly sample N episodes.
        fps: Video frame rate.
        seed: Random seed for sampling.
    """
    _require_cv2()

    run_path = Path(run_dir)
    if not run_path.is_dir():
        logger.error("Run directory not found: %s", run_dir)
        return

    episodes = discover_episodes(run_path, filter_by=filter_by, sample_n=sample_n, seed=seed)
    if not episodes:
        logger.info("No episodes found matching criteria in %s", run_dir)
        return

    output_dir = run_path / "analysis" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d trajectory videos in %s", len(episodes), output_dir)

    from easi.utils.progress import ProgressBar

    failed = 0
    with ProgressBar(total=len(episodes)) as bar:
        for i, ep_dir in enumerate(episodes):
            output_path = output_dir / f"{ep_dir.name}.mp4"
            try:
                render_episode_video(ep_dir, output_path, fps=fps)
            except Exception:
                logger.exception("Failed to render %s", ep_dir.name)
                failed += 1
            bar.update(completed=i + 1, failed=failed)

    logger.info("Done. %d videos saved to %s", len(episodes) - failed, output_dir)
