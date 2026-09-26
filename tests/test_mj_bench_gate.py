"""Scenario builders, metrics and the gate arithmetic of the tracking bench."""

import maker_arm_rs
import numpy as np
import pytest

from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as prof
from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.mj import bench
from maker_arm_lab.mj import bench_gate as bg


@pytest.fixture(scope="module")
def ik():
    return GlobalIK()


def _trace(t, tip, dq=None, force=None):
    n = len(t)
    zeros6 = np.zeros((n, 6))
    return bench.Trace(
        t=np.asarray(t, float),
        q=zeros6,
        dq=zeros6 if dq is None else np.asarray(dq, float),
        tip=np.asarray(tip, float),
        force=np.zeros(n) if force is None else np.asarray(force, float),
        r=zeros6,
        offset=zeros6,
        q_r=zeros6,
        gate=np.ones(n),
    )


def test_sway_is_the_norm_of_the_per_axis_range_in_the_window():
    t = np.arange(1, 401) * 0.005
    tip = np.zeros((400, 3))
    tip[:, 0] = np.where(t >= 1.0, 0.002 * np.sin(40 * t), 0.5)
    tip[:, 1] = np.where(t >= 1.0, 0.001, 0.0)
    assert abs(bg.sway_mm(_trace(t, tip), t_last=1.0) - 4.0) < 0.05


def test_overshoot_projects_on_the_travel_direction_and_clips_at_zero():
    t = np.arange(1, 201) * 0.005
    tip = np.zeros((200, 3))
    tip[:, 0] = np.minimum(t, 0.6) + np.where(t > 0.6, 0.003 * np.exp(-(t - 0.6) * 5), 0.0)
    start, final = np.array([0.0, 0, 0]), np.array([0.6, 0, 0])
    assert abs(bg.overshoot_mm(_trace(t, tip), start, final, t_from=0.0) - 3.0) < 0.1
    assert bg.overshoot_mm(_trace(t, np.zeros((200, 3))), start, final, t_from=0.0) == 0.0


def test_contact_metrics_find_the_first_touch():
    t = np.arange(1, 201) * 0.005
    force = np.where(t >= 0.4 - 1e-9, 10.0 - 10.0 * (t - 0.4), 0.0)
    c = bg.contact(_trace(t, np.zeros((200, 3)), force=force))
    assert abs(c["t_contact"] - 0.4) < 1e-9 and abs(c["peak"] - 10.0) < 1e-9
    assert abs(c["at_0_3"] - 7.0) < 0.06


def test_growing_needs_a_rising_envelope_above_the_floor():
    t = np.arange(1, 401) * 0.005
    grow = np.zeros((400, 6))
    grow[:, 0] = 0.01 * np.exp(3 * t) * np.sin(20 * t)
    decay = np.zeros((400, 6))
    decay[:, 0] = np.exp(-3 * t) * np.sin(20 * t)
    tiny = np.zeros((400, 6))
    tiny[:, 0] = 0.001 * np.exp(3 * t) * np.sin(20 * t) / np.exp(6)
    assert bg.growing(_trace(t, np.zeros((400, 3)), dq=grow))
    assert not bg.growing(_trace(t, np.zeros((400, 3)), dq=decay))
    assert not bg.growing(_trace(t, np.zeros((400, 3)), dq=tiny))


def test_a_scenario_3_only_oscillation_fails_the_growing_row(monkeypatch):
    t = np.arange(1, 401) * 0.005
    grow = np.zeros((400, 6))
    grow[:, 0] = 0.01 * np.exp(3 * t) * np.sin(20 * t)
    quiet, rising = _trace(t, np.zeros((400, 3))), _trace(t, np.zeros((400, 3)), dq=grow)

    def stub(name):
        return lambda go, s: ({}, [rising if name == "3" else quiet])

    monkeypatch.setattr(bg, "SCENARIOS", {n: stub(n) for n in ("1", "2", "3", "4", "5", "6")})
    m = bg._metrics(None, None)
    assert m["growing"] is True
    assert m["growing_by_scenario"] == {n: n == "3" for n in ("1", "2", "3", "4", "5", "6")}


def test_poses_and_walks_are_reachable_and_deterministic(ik):
    p = bg.poses(ik)
    assert set(p) == {"desk", "reach", "hover", "release"}
    rows = bg.transport_rows(ik, p["hover"], p["release"])
    assert np.allclose(rows[0][:6], p["hover"]) and np.allclose(rows[-1][:6], p["release"])
    assert all(len(r) == 7 for r in rows)
    a, b = bg.random_targets(ik, 2, 8), bg.random_targets(ik, 2, 8)
    assert a == b and len(a) == 8
    assert not any(np.allclose(x, y) for x in a for y in bg.random_targets(ik, 1, 8))


class _RefusesFarX:
    """GlobalIK that has no solution for any target past x = 0.2 m."""

    def __init__(self, ik):
        self.ik = ik
        self.refused = 0

    def solve(self, target, **kwargs):
        if target.position[0] > 0.2:
            self.refused += 1
            return None
        return self.ik.solve(target, **kwargs)

    def fk(self, q):
        return self.ik.fk(q)


def test_an_unsolvable_walk_target_is_redrawn_from_the_same_generator(ik):
    a, b = _RefusesFarX(ik), _RefusesFarX(ik)
    qa, qb = bg.random_targets(a, 2, 8), bg.random_targets(b, 2, 8)
    assert a.refused > 0 and a.refused == b.refused and qa == qb and len(qa) == 8
    assert all(float(ik.fk(q)[0][0]) <= 0.2 + 1e-6 for q in qa)


def test_the_recorded_tracking_initials_are_the_rust_defaults(ik):
    """``bg.TRACKING_INITIAL`` mirrors TrackingParams' defaults: passing it changes nothing."""
    plant = bench.build_plant()
    model = dm.dynamics(dm.urdf_bodies())
    p = bg.poses(ik)
    rows = bg.transport_rows(ik, p["desk"], p["hover"])

    def go(params):
        tracking = maker_arm_rs.Tracking(model, params=params, profile=prof.sim_profile())
        return bench.run(plant, "A", p["desk"], [(0.1, 0.1, 1 / 30, rows)], 1.5, tracking=tracking)

    base = {"tau_th": [0.1] * 6}
    a, b = go(base), go({**base, **bg.TRACKING_INITIAL})
    assert np.abs(a.offset).max() > 0.0 and a.gate.min() < 1.0  # the contact side ran
    for field in ("q", "r", "offset", "q_r", "gate"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_calibration_is_deterministic(ik):
    model = dm.dynamics(dm.urdf_bodies())
    a, b = bg.calibrate(ik, model), bg.calibrate(ik, model)
    assert a == b and len(a) == 6 and all(0.0 < v < float("inf") for v in a)


def test_the_gate_arithmetic_reads_relative_and_absolute_bars():
    rows = bg.gate_rows(
        {
            "exact": {
                "A": {
                    "static_sag_mm": 0.5,
                    "sway_mm": 1.0,
                    "overshoot_mm": 0.2,
                    "false_yield_mm": 0.4,
                    "wall_force_03": 2.0,
                    "wall_force_peak": 9.0,
                    "press_force": 5.5,
                    "growing": False,
                },
                "B1": {
                    "sway_mm": 4.0,
                    "overshoot_mm": 1.0,
                    "wall_force_03": 5.0,
                    "wall_force_peak": 10.0,
                    "press_force": 5.0,
                },
            },
            "housing": {"A": {"sway_mm": 2.5, "false_yield_mm": 2.0, "growing": False}},
        }
    )
    by = {(r["model"], r["metric"]): r["pass"] for r in rows}
    assert by[("exact", "sway_mm")] is True  # 1.0 <= 0.3 x 4.0
    assert by[("housing", "sway_mm")] is False  # 2.5 > 0.5 x 4.0
    assert by[("exact", "press_force")] is True  # within 20 % of 5.0
