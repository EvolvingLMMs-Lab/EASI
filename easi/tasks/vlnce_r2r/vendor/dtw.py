"""NDTW and SDTW metric computation.

Vendored from VLN-CE habitat_extensions/measures.py.
Uses fastdtw for dynamic time warping distance.
"""
from __future__ import annotations

import numpy as np


def _euclidean_distance(a, b):
    return np.linalg.norm(np.array(b) - np.array(a))


def compute_ndtw(agent_positions, gt_locations, success_distance=3.0):
    """Compute Normalized Dynamic Time Warping distance.

    Args:
        agent_positions: List of [x, y, z] agent positions at each step.
        gt_locations: List of [x, y, z] ground truth reference path positions.
        success_distance: Success threshold (default 3.0m for VLN-CE R2R).

    Returns:
        NDTW score in [0, 1]. Higher is better.
    """
    from fastdtw import fastdtw

    dtw_distance = fastdtw(
        agent_positions, gt_locations, dist=_euclidean_distance
    )[0]
    return np.exp(-dtw_distance / (len(gt_locations) * success_distance))


def compute_sdtw(ndtw, success):
    """Compute Success weighted by NDTW.

    Args:
        ndtw: NDTW score.
        success: 1.0 if episode succeeded, 0.0 otherwise.

    Returns:
        SDTW score (0.0 if not successful).
    """
    return ndtw * success
