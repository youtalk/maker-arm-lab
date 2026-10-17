"""MS5 control plan task 11: the two correction layers on the tracking bench, the observer
compensation (``dob``) and the learned residual, before either reaches the arm."""

import copy

import maker_arm_rs
import numpy as np
import pytest

from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.ident import excitation
from maker_arm_lab.mj import bench
from maker_arm_lab.models import residual
from maker_arm_lab.real import rig
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


def capped(scale):
    """The simulator's profile at a reduced cap ``scale`` on every joint's kp and kd, as
    ``real.rig.reduced_profile`` scales a rig's caps."""
    profile = copy.deepcopy(SIM)
    for joint in profile["joints"]:
        joint["kp"] *= scale
        joint["kd"] *= scale
    return profile


def make(k_o, dob, dob_f, tau_th=TAU_TH, scale=1.0):
    """The controller of the sweep; at a cap ``scale`` below 1 the gains are scaled and the
    error clamp follows ruling R33 (``real.rig.capped_e_max``)."""
    params = {
        "contact": True,
        "k_o": k_o,
        "f_y": F_Y,
        "tau_th": [tau_th] * 6,
        "dob": dob,
        "dob_f": dob_f,
    }
    if scale != 1.0:
        params["e_max"] = rig.capped_e_max([scale] * 7)
    return maker_arm_rs.Tracking(EXACT, params=params, profile=capped(scale))


def envelope(q, start_s, end_s):
    """The peak-to-peak of the joint error over ``[start_s, end_s)``, the largest over the
    joints. The windows read lie inside a hold, where the target is constant, so the error's
    peak-to-peak is q's."""
    q = np.asarray(q)
    return float(np.ptp(q[round(start_s / DT) : round(end_s / DT)], axis=0).max())


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


GROWTH_MAX = 1.05
"""The second hold's envelope over 4.5-5 s may exceed the one over 3.5-4 s by at most 5 %: a
loop that rings up after the transport fails, one that settles or wanders in place passes."""
DOB_WANDER_MAX = 8.0  # INITIAL: session 6 reads the real hold
"""How many times the dob-off run's last-second envelope the dob run's may be. The bench sweep
reads 2.6 to 5.2 times (5.5 at tau_th 2.0): the compensation chases the 0.2 N m torque noise and
the plant's unmodelled friction and armature in r, hard enough to break the stiction that holds
the dob-off arm still (its first hold reads 0.0 mrad peak to peak, the dob runs 3.9 to 6.8
mrad). The bar catches a wander that grows past today's, not today's."""
SATURATED = 0.9
"""A tick whose |tau_dob| is at least this share of tau_th on some joint counts as clamped."""
SWEEP = [(k_o, dob_f, TAU_TH) for k_o in (25.0, 50.0, 100.0) for dob_f in (2.0, 5.0, 10.0)]
SWEEP.append((50.0, 5.0, 2.0))
"""The 3 x 3 sweep at tau_th 0.5 N m, and one point at 2.0 N m, where the clamp is far out of
reach: the loop is stable without the clamp bounding it."""


def test_the_compensation_stays_stable_across_the_sweep():
    """Bench plant, rung A, hold 2 s then one 1 s transport then hold 2 s. Two checks per
    point: (a) no growth inside the second hold (``GROWTH_MAX``), and (b) the last second's
    envelope against the dob-off run's at the same tau_th (``DOB_WANDER_MAX``). The reference
    is not the first hold with dob on, which the compensation itself inflates. Each point also
    records the share of ticks with |tau_dob| at ``SATURATED`` x tau_th or more on some joint:
    0 to 0.4 %, largest |tau_dob| 0.375 to 0.47 N m, so the clamp does not bound these runs,
    and the tau_th 2.0 point reads like its 0.5 twin. Today's reads: growth 0.47 to 1.004 (the
    tightest is k_o 100, dob_f 2 Hz), wander 2.6 to 5.2 times the dob-off run's."""
    plant = bench.build_plant()
    reference = {}
    for tau_th in sorted({p[2] for p in SWEEP}):
        off = bench.run(
            plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(50.0, False, 5.0, tau_th)
        )
        reference[tau_th] = envelope(off.q, DURATION_S - 1.0, DURATION_S)
        assert reference[tau_th] > 0.0
    rows = {}
    for k_o, dob_f, tau_th in SWEEP:
        trace = bench.run(
            plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(k_o, True, dob_f, tau_th)
        )
        assert np.isfinite(trace.q).all()
        growth = envelope(trace.q, 4.5, 5.0) / envelope(trace.q, 3.5, 4.0)
        wander = envelope(trace.q, DURATION_S - 1.0, DURATION_S) / reference[tau_th]
        saturated = float((np.abs(trace.tau_dob) >= SATURATED * tau_th).any(axis=1).mean())
        rows[(k_o, dob_f, tau_th)] = (round(growth, 3), round(wander, 2), saturated)
        assert growth <= GROWTH_MAX, (k_o, dob_f, tau_th, growth)
        assert wander <= DOB_WANDER_MAX, (k_o, dob_f, tau_th, wander)
    print("(k_o, dob_f, tau_th): (growth, wander vs dob off, saturated share)", rows)


CAP_SCALES = (0.5, 0.25)
"""The reduced caps of the first powered sessions: kp and kd scaled, e_max by R33."""
REDUCED_ENVELOPE_MAX = 0.05  # INITIAL: session 6 reads the real hold at reduced caps
"""rad: the largest last-second peak to peak a dob run at a reduced cap may read on the bench.
A runaway loop leaves it; today's wander does not (see the test)."""


def test_the_compensation_at_reduced_caps_is_bounded_and_recorded():
    """The sweep's 3 x 3 points at tau_th 0.5 N m at the cap scales 0.5 and 0.25 (ruling R33's
    e_max). Recorded, with only a runaway bound asserted (``REDUCED_ENVELOPE_MAX``): at
    reduced gains the profile-gain bars do not hold. Today's reads (bench, seed 0): at 0.5 the
    growth is 0.45 to 1.19 and the wander 2.7 to 5.3 times the dob-off run's (1.5 mrad); at
    0.25 the growth is 0.41 to 3.37 and the wander 3.3 to 9.2 times the dob-off run's (2.2
    mrad), last-second envelopes 7 to 20 mrad, and a 15 s hold wanders 5 to 33 mrad per second
    without a trend. The final hold error still falls with dob: 4.6 -> 2.3 to 3.4 mrad at 0.5,
    13.9 -> 3.8 to 8.9 mrad at 0.25."""
    plant = bench.build_plant()
    rows = {}
    for scale in CAP_SCALES:
        off = bench.run(
            plant, "A", list(POSE), PUSHES, DURATION_S,
            tracking=make(50.0, False, 5.0, TAU_TH, scale),
        )  # fmt: skip
        reference = envelope(off.q, DURATION_S - 1.0, DURATION_S)
        rows[(scale, "off")] = (round(1e3 * reference, 2), round(1e3 * hold_error(off), 2))
        for k_o in (25.0, 50.0, 100.0):
            for dob_f in (2.0, 5.0, 10.0):
                trace = bench.run(
                    plant, "A", list(POSE), PUSHES, DURATION_S,
                    tracking=make(k_o, True, dob_f, TAU_TH, scale),
                )  # fmt: skip
                assert np.isfinite(trace.q).all()
                last = envelope(trace.q, DURATION_S - 1.0, DURATION_S)
                growth = envelope(trace.q, 4.5, 5.0) / envelope(trace.q, 3.5, 4.0)
                rows[(scale, k_o, dob_f)] = (
                    round(growth, 3),
                    round(last / reference, 2),
                    round(1e3 * last, 2),
                    round(1e3 * hold_error(trace), 2),
                )
                assert last <= REDUCED_ENVELOPE_MAX, (scale, k_o, dob_f, last)
    print("(scale, k_o, dob_f): (growth, wander vs dob off, last envelope mrad, hold mrad)", rows)


MASS_ERROR = 0.65
"""The model's mass share of the plant's in the caps test: a light model leaves the spring to
carry a third of gravity."""


def test_a_reduced_kp_with_the_scaled_error_clamp_holds_a_light_model():
    """Ruling R33 on the bench: rung A, 3 s hold at the excitation center, the URDF bodies at
    0.65 of their mass as the model, kp and kd at 0.25 of the simulator profile's. With the
    profile's e_max (0.03 rad) the spring gives at most kp e_max and the arm sinks: j2 reads
    1556 mrad from the start (it runs to its stop), j3 974, j4 274. With e_max = 0.03 / 0.25
    the drift stays at 37, 46 and 43 mrad. Only the bounded case is asserted."""
    light = dm.dynamics(dm.scaled(dm.urdf_bodies(), MASS_ERROR))
    plant = bench.build_plant()
    rows = [[*POSE, bench.grip_value()]]
    drift = {}
    for name, e_max in (("profile", None), ("r33", rig.capped_e_max([0.25] * 7))):
        params = {"contact": True, "tau_th": [2.0] * 6}
        if e_max is not None:
            params["e_max"] = e_max
        tracking = maker_arm_rs.Tracking(light, params=params, profile=capped(0.25))
        trace = bench.run(plant, "A", list(POSE), [(0.0, 0.0, DT, rows)], 3.0, tracking=tracking)
        drift[name] = np.abs(trace.q - POSE).max(axis=0)
    print("drift mrad: profile e_max", np.round(1e3 * drift["profile"], 1))
    print("drift mrad: R33 e_max", np.round(1e3 * drift["r33"], 1))
    assert drift["r33"].max() < 0.08, drift["r33"]


def test_the_compensation_lowers_the_hold_error_under_unmodeled_friction():
    """The plant with 0.15 N m more dry friction on every joint than any model of it, k_o 50,
    dob_f 5 Hz, f_y 1.5 Hz (not the SDK's 2.0, see ``F_Y``). The RMS distance to the target
    over the last second reads 4.16 mrad without dob and 1.68 with it (seed 0; seeds 1 and 2
    read 4.21 / 2.71 and 4.29 / 2.28)."""
    plant = bench.build_plant(extra_coulomb=0.15)
    errors = {}
    for dob in (False, True):
        trace = bench.run(plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(50.0, dob, 5.0))
        errors[dob] = hold_error(trace)
    print("hold error rad: without dob", errors[False], "with dob", errors[True])
    assert errors[True] < errors[False], errors


def test_the_compensation_on_the_default_plant_is_recorded():
    """The companion read on the plant whose friction is only the bench's (``COULOMB_NM``),
    same settings (f_y 1.5 Hz). Here the compensation does not help: the final hold error reads
    1.05 to 1.09 mrad without dob and 1.6 to 2.3 with it (seeds 0 to 2), and dob adds hold
    wander (the sweep's ``DOB_WANDER_MAX``). Recorded, not asserted in its direction: pinning a
    defect would fail the day a better compensation fixes it. The test checks that both reads
    are finite and stay under 5 mrad, and prints them."""
    plant = bench.build_plant()
    errors = {}
    for dob in (False, True):
        trace = bench.run(plant, "A", list(POSE), PUSHES, DURATION_S, tracking=make(50.0, dob, 5.0))
        errors[dob] = hold_error(trace)
    print("default plant hold error rad: without dob", errors[False], "with dob", errors[True])
    assert all(np.isfinite(e) and e < 0.005 for e in errors.values()), errors


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
