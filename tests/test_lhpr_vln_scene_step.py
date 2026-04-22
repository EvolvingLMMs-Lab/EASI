"""Regression tests for SceneSimulator.actor step counting.

Covers the 2026-04-22 fix that dropped the ``step == -1`` early return in
``easi/tasks/lhpr_vln/vendor/scene_simulator.py``. See
``docs/superpowers/plans/2026-04-22-lhpr-vln-step-offbyone.md``.

These tests stub ``habitat_sim`` at import time so they run offline in the
main venv (the real module only exists inside the ``easi_habitat_sim_v0_3_0``
conda env).
"""
from __future__ import annotations

import sys
import types

import pytest


# ---- habitat_sim stub -------------------------------------------------------

def _install_habitat_sim_stub() -> None:
    """Register a minimal ``habitat_sim`` module in sys.modules.

    scene_simulator.py does ``import habitat_sim`` at top level. We only
    need enough of the surface to let the class body evaluate; the tests
    never hit anything that actually uses these stubs because we bypass
    __init__ via ``object.__new__``.
    """
    if "habitat_sim" in sys.modules:
        return
    mod = types.ModuleType("habitat_sim")
    mod.Simulator = object  # type: ignore[attr-defined]
    mod.AgentState = object  # type: ignore[attr-defined]

    class _NoopNav:
        class ShortestPath:
            pass

    mod.nav = _NoopNav()  # type: ignore[attr-defined]
    sys.modules["habitat_sim"] = mod


_install_habitat_sim_stub()


from easi.tasks.lhpr_vln.vendor.scene_simulator import SceneSimulator  # noqa: E402


# ---- fixture ---------------------------------------------------------------

class _FakeSim:
    """Stand-in for ``habitat_sim.Simulator`` — records every step() call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def step(self, action):  # mimics the observation return
        self.calls.append(action)
        return {"color_sensor_f": None}


def _make_sim(*, target_num: int, max_steps: int, geo_dis_seq: list[float]) -> SceneSimulator:
    """Build a SceneSimulator without running its heavy __init__.

    ``geo_dis_seq`` feeds ``_get_info()`` calls; each call pops the next value.
    When exhausted, returns the last value (so tests don't have to count
    exact info invocations — they only care about when the sim *should*
    be near / far from the current target).
    """
    sim = object.__new__(SceneSimulator)
    sim.sim = _FakeSim()
    sim.targets = [f"obj_{i}" for i in range(target_num)]
    sim.target_num = target_num
    sim.stage = 0
    sim.step = 0
    sim.max_steps = max_steps
    sim.success_distance = 1.0
    sim.successes = [False] * target_num
    sim.oracle_successes = [False] * target_num
    sim.nav_steps = []
    sim.nav_errors = []
    sim.done = False
    sim.episode_over = False
    sim.observations = {"color_sensor_f": None}

    geo_iter = list(geo_dis_seq)

    def _get_info() -> dict:
        val = geo_iter.pop(0) if geo_iter else (geo_dis_seq[-1] if geo_dis_seq else 0.0)
        return {"target": sim.targets[sim.stage], "geo_dis": val}

    sim._get_info = _get_info  # type: ignore[assignment]
    sim.info = _get_info()
    sim.gt_path = [sim.info["geo_dis"]]
    return sim


# ---- tests -----------------------------------------------------------------

def test_timeout_matches_gt_path_length():
    """After max_steps of non-stop actions, timeout branch fires and all
    per-subtask accounting lists end up the same length as gt_path."""
    sim = _make_sim(
        target_num=3,
        max_steps=5,
        geo_dis_seq=[10.0] * 10,   # far from target the whole time
    )
    for _ in range(5):
        sim.actor("move_forward")

    assert sim.episode_over is True
    assert sim.done is False, "episode_over from timeout should NOT mark done"
    assert len(sim.nav_errors) == 1, sim.nav_errors
    assert len(sim.nav_steps) == 1, sim.nav_steps
    assert len(sim.gt_path) == 1, sim.gt_path
    assert len(sim.gt_path) == len(sim.nav_errors)


def test_all_subtasks_success_keeps_arrays_aligned():
    sim = _make_sim(
        target_num=3,
        max_steps=50,
        # geo_dis < 1.0 whenever a stop action is issued below.
        geo_dis_seq=[0.3] * 20,
    )
    # 3 stops in a row (no movement in between — the test cares about bookkeeping,
    # not about pose). Each stop advances the stage.
    sim.actor("stop")
    sim.actor("stop")
    sim.actor("stop")

    assert sim.done is True
    assert sim.episode_over is True
    assert sim.successes == [True, True, True], sim.successes
    assert len(sim.nav_errors) == 3
    assert len(sim.nav_steps) == 3
    # gt_path: initial + one append per non-final stage advance (2 non-final stops),
    # no append on the final stop (returns early).
    assert len(sim.gt_path) == 3


def test_first_action_stop_is_honoured():
    """Used to be silently ignored when step==-1 short-circuited. Now the
    first-action stop must advance the stage and append a nav_error."""
    sim = _make_sim(
        target_num=3,
        max_steps=50,
        geo_dis_seq=[5.0, 5.0, 5.0, 5.0, 5.0],  # far: stop fails but still advances
    )
    sim.actor("stop")

    assert sim.stage == 1, "stage must advance on stop even when failed"
    assert sim.successes == [False, False, False]
    assert len(sim.nav_errors) == 1
    # self.step is the count of actor() calls (including this one).
    assert sim.step == 1, sim.step
    # First nav_steps entry is raw self.step (= 1).
    assert sim.nav_steps == [1]


def test_mixed_partial_success_then_timeout():
    """Agent stops on subtask 0 (near → success, stage→1), then walks into
    timeout during subtask 1. nav_errors / gt_path stay aligned so TAR can
    iterate j over gt_path[i] without overrunning error_length[i]."""
    sim = _make_sim(
        target_num=3,
        max_steps=4,
        # step 1: stop (near)   -> success
        # steps 2-4: move_forward (far)
        geo_dis_seq=[0.2, 0.2, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    sim.actor("stop")
    sim.actor("move_forward")
    sim.actor("move_forward")
    sim.actor("move_forward")

    assert sim.episode_over is True
    assert sim.successes[0] is True
    assert sim.stage == 1
    # One entry from the successful stop; one from the timeout.
    assert len(sim.nav_errors) == 2
    assert len(sim.nav_steps) == 2
    # gt_path: initial + one append from stage-0 stop advancing to stage 1.
    assert len(sim.gt_path) == 2
    assert len(sim.gt_path) == len(sim.nav_errors)
