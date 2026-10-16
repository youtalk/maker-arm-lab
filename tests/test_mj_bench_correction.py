"""MS5 control plan task 11: the two correction layers on the tracking bench, the observer
compensation (``dob``) and the learned residual, before either reaches the arm."""

import maker_arm_rs
import numpy as np
import pytest

from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.ident import excitation
from maker_arm_lab.mj import bench
from maker_arm_lab.models import residual
from test_ident import BENCH_HOLD, bench_runs

SIM = core_profile.sim_profile()
DT = bench.TICK_S
POSE = np.array(excitation.EXCITE_CENTER)
MOVE = np.array([0.3, 0.25, -0.25, 0.2, 0.3, 0.3])
"""The one transport: every joint moves, from the excitation center."""
HOLD_S, MOVE_S, DURATION_S = 2.0, 1.0, 5.0
"""Hold 2 s, one 1 s transport, hold 2 s."""
TAU_TH = 0.5
"""N m on every joint: dob needs a finite deadband, which also bounds the compensation."""
F_Y = 1.5
"""Hz: the yield corner below half the observer bandwidth at the sweep's lowest k_o (25 /s
is 3.98 Hz, half of it 1.99 Hz), so every point of the sweep validates."""
EXACT = dm.dynamics(dm.urdf_bodies())
"""The bench plant's model: the URDF's bodies, without the plant's armature and friction."""


def plan():
    """One row per tick: the hold at POSE, a minimum-jerk transport by MOVE, the hold there."""
    t = np.arange(0.0, DURATION_S + 1e-9, DT)
    s = np.clip((t - HOLD_S) / MOVE_S, 0.0, 1.0)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    q = POSE + s[:, None] * MOVE
    grip = bench.grip_value()
    return [[*row, grip] for row in q.tolist()]


PUSHES = [(0.0, 0.0, DT, plan())]


def make(k_o, dob, dob_f):
    params = {
        "contact": True,
        "k_o": k_o,
        "f_y": F_Y,
        "tau_th": [TAU_TH] * 6,
        "dob": dob,
        "dob_f": dob_f,
    }
    return maker_arm_rs.Tracking(EXACT, params=params, profile=SIM)


def envelope(q, *, last_s=None, first_hold_s=None):
    """The peak-to-peak of the joint error over a window, the largest over the joints. The
    target is constant over both windows, so the error's peak-to-peak is q's. ``last_s``: the
    last seconds of the run; ``first_hold_s``: the last seconds of the first hold (its start
    is the controller's own start-up)."""
    if (last_s is None) == (first_hold_s is None):
        raise ValueError("give one of last_s and first_hold_s")
    q = np.asarray(q)
    if last_s is not None:
        window = q[-round(last_s / DT) :]
    else:
        end = round(HOLD_S / DT)
        window = q[end - round(first_hold_s / DT) : end]
    return float(np.ptp(window, axis=0).max())


def hold_error(trace, last_s=1.0):
    """rad: the RMS over the last ``last_s`` of the joint-space distance to the final target."""
    e = trace.q[-round(last_s / DT) :] - (POSE + MOVE)
    return float(np.sqrt((e**2).sum(axis=1).mean()))


def test_the_bench_trace_carries_the_correction_telemetry():
    plant = bench.build_plant()
    off = bench.run(plant, "A", list(POSE), PUSHES, 0.5, tracking=make(50.0, False, 5.0))
    on = bench.run(plant, "A", list(POSE), PUSHES, 0.5, tracking=make(50.0, True, 5.0))
    assert off.tau_dob.shape == (100, 6) and not off.tau_dob.any()
    assert not off.tau_res.any()
    assert np.abs(on.tau_dob).max() > 0.0
    assert np.abs(on.tau_dob).max() <= TAU_TH
    b0 = bench.run(plant, "B0", list(POSE), [], 0.1)
    assert b0.tau_dob is None and b0.tau_res is None


def test_the_compensation_stays_stable_across_the_sweep():
    # Bench plant, rung A, tau_th 0.5 N m, hold 2 s then one 1 s transport then hold 2 s.
    plant = bench.build_plant()
    ratios = {}
    for k_o in (25.0, 50.0, 100.0):
        for dob_f in (2.0, 5.0, 10.0):
            trace = bench.run(
                plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(k_o, True, dob_f)
            )
            assert np.isfinite(trace.q).all()
            last = envelope(trace.q, last_s=1.0)
            first = envelope(trace.q, first_hold_s=1.0)
            ratios[(k_o, dob_f)] = last / first
            assert last <= first * 1.05, (k_o, dob_f, last, first)
    print("envelope ratios", {k: round(v, 3) for k, v in ratios.items()})


def test_the_compensation_lowers_the_hold_error_under_unmodeled_friction():
    plant = bench.build_plant(extra_coulomb=0.15)
    errors = {}
    for dob in (False, True):
        trace = bench.run(plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(50.0, dob, 5.0))
        errors[dob] = hold_error(trace)
    print("hold error rad: without dob", errors[False], "with dob", errors[True])
    assert errors[True] < errors[False], errors


@pytest.fixture(scope="module")
def bench_residual(tmp_path_factory):
    """A residual trained on three 20 s bench excitations (seed 1, rung A, URDF model) and a
    held-out 10 s excitation of another seed."""
    logs = bench_runs(tmp_path_factory.mktemp("bench_residual"), 1, 3)
    X, Y = residual.build_dataset(logs, EXACT, 4, hold_steps=BENCH_HOLD)
    model, report = residual.train(X, Y, seed=0, epochs=60)
    spec = residual.residual_spec(model, report["stats"], 4)
    (traj,) = excitation.design_excitation(
        np.random.default_rng(99),
        SIM,
        excitation.EXCITE_CENTER,
        count=1,
        wf_hz=0.3,
        v_scale=1.0,
        amplitude_rad=None,
        duration_s=10.0,
    )
    return spec, traj


def test_a_residual_trained_on_bench_logs_lowers_the_free_motion_observer_reading(
    bench_residual,
):
    spec, traj = bench_residual
    rows = np.concatenate([traj["q"], traj["grip"][:, None]], axis=1).tolist()
    plant = bench.build_plant()
    reading = {}
    for name, res in (("none", None), ("mlp", spec)):
        tracking = maker_arm_rs.Tracking(EXACT, params={"contact": True}, profile=SIM, residual=res)
        trace = bench.run(
            plant,
            "A",
            list(traj["q"][0]),
            [(0.0, 0.0, DT, rows)],
            traj["t"][-1],
            tracking=tracking,
            temperature=[30.0] * 7,  # the bench logs' temperature (write_session_log's)
            seed=5,
        )
        free = trace.gate > 0.99
        assert free.mean() > 0.9  # free motion: the gate stays open
        reading[name] = float(np.sqrt(np.mean(trace.r[free] ** 2)))
        if res is not None:
            assert np.abs(trace.tau_res).max() > 0.0
    print("free-motion r rms N m", reading)
    assert reading["mlp"] < 0.95 * reading["none"], reading
