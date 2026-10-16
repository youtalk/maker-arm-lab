"""MS5 control plan task 7: the excitation design, the regressor and the identification fit,
proved on synthetic logs of a known model and on the MuJoCo bench plant."""

import ast
import json
import math
import sys

import maker_arm_rs
import numpy as np
import pytest

from helpers import ROOT
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.ident import excitation, fit
from maker_arm_lab.mj import bench
from maker_arm_lab.real import session
from maker_arm_lab.real.telemetry import TelemetryLog
from maker_arm_lab.robots.frames import ARM_ARMATURE

SIM = core_profile.sim_profile()
DT = session.DT
PROFILE_COULOMB = [bench.COULOMB_NM[j["model"]] for j in SIM["joints"][:6]]


def _factory(bodies):
    return dm.dynamics(bodies)


# --- session-format logs ------------------------------------------------------------------


def write_session_log(
    path, t, q, dq, tau, *, gate=None, clamped=None, segments=(), grip=0.0, kp=None, kd=None
):
    """A session log (``real.telemetry``) of six-wide q, dq and tau rows at times ``t``, with a
    gripper column and a tracking row per tick, as ``real_session.py`` writes one. ``kp`` and
    ``kd`` (six) are the commanded gains, zero by default."""
    n = len(t)

    def gains(g):
        return np.zeros((n, 7)) if g is None else np.tile([*g, 0.0], (n, 1))

    gate = np.ones(n) if gate is None else np.asarray(gate, dtype=float)
    clamped = np.zeros(n, dtype=bool) if clamped is None else np.asarray(clamped, dtype=bool)

    def seven(a, last):
        return np.concatenate([np.asarray(a, dtype=float), np.full((n, 1), last)], axis=1)

    ticks = {
        "t": np.asarray(t, dtype=float),
        "tick": np.arange(n),
        "clamped": clamped,
        "q": seven(q, grip),
        "dq": seven(dq, 0.0),
        "tau": seven(tau, 0.0),
        "temp": np.full((n, 7), 30.0),
        "fault_bits": np.zeros((n, 7), dtype=np.uint16),
        "feedback_age": np.zeros((n, 7)),
        "cmd_pos": seven(q, grip),
        "cmd_vel": np.zeros((n, 7)),
        "cmd_kp": gains(kp),
        "cmd_kd": gains(kd),
        "cmd_tau": np.zeros((n, 7)),
    }
    tracking = {
        "tick_t": np.asarray(t, dtype=float),
        "gate": gate,
        "r": np.zeros((n, 6)),
        "offset": np.zeros((n, 6)),
        "q_r": np.asarray(q, dtype=float),
        "tau_ff": np.zeros((n, 6)),
    }
    with TelemetryLog(path, {"dynamics": "urdf", "lab_revision": "test"}) as log:
        log.append(ticks, tracking, mono_ns=0)
        for seg in segments:
            log.add_segment(*seg)
    return path


BENCH_GAINS = {k: [float(j[k]) for j in SIM["joints"][:6]] for k in ("kp", "kd")}
"""Rung A commands the profile's gains (the clamp's caps) on every tick."""
BENCH_HOLD = bench.STEPS_PER_TICK
"""The bench's ``fit.HOLD_STEPS``: five 1 ms physics steps per held command."""


def bench_log(path, trace, grip):
    """The bench trace as a session log. ``Trace.tau[k]`` is the reply of tick k's first
    physics step: the torque applied from the state at the end of tick k - 1 (``Trace.q[k -
    1]``), so the log pairs that state with that torque, as the real arm's reply pairs them."""
    t = trace.t[:-1]
    write_session_log(
        path,
        t,
        trace.q[:-1],
        trace.dq[:-1],
        trace.tau[1:],
        gate=trace.gate[1:],
        grip=grip,
        **BENCH_GAINS,
    )
    return path


def synthetic_log(path, model, traj, *, noise=0.0, seed=0, **kw):
    """tau = the model's inverse dynamics plus its friction along ``traj``, plus noise."""
    rng = np.random.default_rng(seed)
    tau = np.array(
        [
            np.add(model.inverse(list(q), list(dq), list(ddq)), model.friction(list(dq)))
            for q, dq, ddq in zip(traj["q"], traj["dq"], traj["ddq"])
        ]
    )
    tau = tau + rng.normal(0.0, noise, tau.shape)
    return write_session_log(path, traj["t"], traj["q"], traj["dq"], tau, **kw)


def test_the_ident_package_imports_only_what_the_plan_allows():
    allowed = {"torch", "numpy", "h5py", "maker_arm_rs"}
    for path in sorted((ROOT / "source/maker_arm_lab/ident").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                ok = (
                    top in allowed
                    or top in sys.stdlib_module_names
                    or top == "__future__"
                    or name.startswith("maker_arm_lab.core")
                )
                assert ok, (path.name, name)


# --- the regressor ------------------------------------------------------------------------


def test_the_parameters_round_trip_through_the_bodies():
    bodies = dm.urdf_bodies()
    p = fit.linear_params(bodies)
    assert p.shape == (60,)
    back = fit.bodies_from_params(p)
    for (m, c, i), (m2, c2, i2) in zip(bodies, back):
        assert m2 == pytest.approx(m, abs=1e-12)
        assert np.allclose(c2, c, atol=1e-12) and np.allclose(i2, i, atol=1e-12)
    assert fit.physical(back)


def test_the_regressor_times_the_parameters_is_inverse_dynamics():
    p = fit.linear_params(dm.urdf_bodies())
    dyn = dm.dynamics(dm.urdf_bodies())
    rng = np.random.default_rng(3)
    q, dq, ddq = (rng.uniform(-1.0, 1.0, (5, 6)) for _ in range(3))
    y = fit.regressor(_factory, p, q, dq, ddq)
    assert y.shape == (5, 6, 60)
    for k in range(5):
        assert np.allclose(y[k] @ p, dyn.inverse(list(q[k]), list(dq[k]), list(ddq[k])), atol=1e-9)
    # one sample in, one (6, 60) matrix out; and linear: a scaled model is the same matrix
    one = fit.regressor(_factory, p, q[0], dq[0], ddq[0])
    assert one.shape == (6, 60) and np.allclose(one, y[0], atol=1e-9)
    heavy = fit.linear_params(dm.scaled(dm.urdf_bodies(), 1.3))
    tau = dm.dynamics(dm.scaled(dm.urdf_bodies(), 1.3)).inverse(
        list(q[1]), list(dq[1]), list(ddq[1])
    )
    assert np.allclose(y[1] @ heavy, tau, atol=1e-9)


# --- the excitation -----------------------------------------------------------------------


def _rest_coefficients(seed=0, harmonics=5):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0.0, 0.2, (6, harmonics)), rng.normal(0.0, 0.2, (6, harmonics))
    return excitation.rest_coefficients(a, b)


def test_a_fourier_trajectory_starts_and_ends_at_rest_at_q0():
    a, b = _rest_coefficients()
    q0 = list(excitation.EXCITE_CENTER)
    t, q, dq, ddq = excitation.fourier_trajectory(q0, a, b, 0.1, 20.0)
    assert t.shape == (4001,) and q.shape == dq.shape == ddq.shape == (4001, 6)
    assert np.allclose(t, np.arange(4001) * DT)
    for k in (0, -1):
        assert np.allclose(q[k], q0, atol=1e-12)
        assert np.allclose(dq[k], 0.0, atol=1e-12) and np.allclose(ddq[k], 0.0, atol=1e-12)
    # the analytic derivatives agree with the samples
    assert np.allclose(np.gradient(q, DT, axis=0)[1:-1], dq[1:-1], atol=1e-3)
    assert np.allclose(np.gradient(dq, DT, axis=0)[1:-1], ddq[1:-1], atol=1e-3)


def test_a_fourier_trajectory_refuses_coefficients_that_do_not_rest():
    a, b = _rest_coefficients()
    a[2, 0] += 0.1  # the start velocity is no longer zero
    with pytest.raises(ValueError, match="rest"):
        excitation.fourier_trajectory([0.0] * 6, a, b, 0.1, 20.0)
    with pytest.raises(ValueError, match="period"):
        excitation.fourier_trajectory([0.0] * 6, *_rest_coefficients(), 0.1, 15.0)


REAL = core_profile.load_profile()
REAL_MID = [0.5 * (j["q_lo"] + j["q_hi"]) for j in REAL["joints"][:6]]
"""The real profile's limits are still its motor-frame numbers (ruling R20), so its center is
the middle of its own windows, not EXCITE_CENTER."""


@pytest.mark.parametrize(
    "profile,center", [(SIM, excitation.EXCITE_CENTER), (REAL, REAL_MID)], ids=["sim", "real"]
)
def test_an_excitation_stays_inside_the_limits_and_the_velocity_cap(profile, center):
    trajs = excitation.design_excitation(np.random.default_rng(7), profile, center, count=3)
    assert len(trajs) == 3
    cap = 0.5 * float(profile["max_velocity"])
    for traj in trajs:
        session.check_traj(traj, profile, margin_rad=0.1)
        assert np.abs(traj["dq"]).max() <= cap + 1e-12
        assert np.abs(np.diff(traj["q"], axis=0)).max() / DT <= cap + 1e-9
        span = np.abs(traj["q"] - np.asarray(center)).max()
        assert span <= excitation.EXCITE_AMPLITUDE_RAD + 1e-12
        for k in (0, -1):
            assert np.allclose(traj["dq"][k], 0.0, atol=1e-12)
            assert np.allclose(traj["ddq"][k], 0.0, atol=1e-12)
        assert np.allclose(traj["q"][0], traj["q"][-1], atol=1e-12)
        # every joint moves: an idle joint leaves its columns unexcited
        assert (traj["q"].max(axis=0) - traj["q"].min(axis=0) > 0.1).all()
    again = excitation.design_excitation(np.random.default_rng(7), profile, center, count=3)
    assert all(np.array_equal(x["q"], y["q"]) for x, y in zip(trajs, again))


def test_an_excitation_without_an_amplitude_cap_uses_the_soft_limits():
    trajs = excitation.design_excitation(
        np.random.default_rng(1), SIM, excitation.EXCITE_CENTER, count=1, amplitude_rad=None
    )
    session.check_traj(trajs[0], SIM)
    assert np.abs(trajs[0]["q"] - np.asarray(excitation.EXCITE_CENTER)).max() > 0.6


def test_an_excitation_refuses_a_center_outside_the_soft_limits():
    center = list(excitation.EXCITE_CENTER)
    center[1] = SIM["joints"][1]["q_hi"] - 0.05
    with pytest.raises(ValueError, match="j2"):
        excitation.design_excitation(np.random.default_rng(0), SIM, center, count=1)


# --- the fit on synthetic logs ------------------------------------------------------------

TRUE_FRICTION = [(0.12, 0.03, fit.EPS_GRID[1], 0.18, 0.05)] * 3 + [
    (0.08, 0.015, fit.EPS_GRID[1], 0.10, 0.05)
] * 3
TRUE_ARMATURE = [0.02, 0.03, 0.03, 0.01, 0.01, 0.01]


def _true_model():
    return maker_arm_rs.Dynamics(
        tuple(dm.MOUNT_RPY),
        dm.chain_links(dm.ARM_JOINT_NAMES),
        dm.urdf_bodies(),
        [list(f) for f in TRUE_FRICTION],
        dm.GRAVITY,
        TRUE_ARMATURE,
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    trajs = excitation.design_excitation(
        np.random.default_rng(11), SIM, excitation.EXCITE_CENTER, count=3, amplitude_rad=None
    )
    model = _true_model()
    logs = [synthetic_log(root / f"exc_{k:02d}.h5", model, tr) for k, tr in enumerate(trajs)]
    return root, trajs, model, logs


def test_the_fit_recovers_a_known_model_from_noise_free_logs(synthetic):
    _, _, _, logs = synthetic
    res = fit.fit(logs, dm.urdf_bodies(), hold_steps=None)
    assert np.allclose(res.armature, TRUE_ARMATURE, rtol=0.02, atol=1e-4)
    for got, want in zip(res.friction, TRUE_FRICTION):
        assert got[0] == pytest.approx(want[0], rel=0.05)
        assert got[1] == pytest.approx(want[1], rel=0.05)
        assert got[3] == pytest.approx(want[3], rel=0.05)
        assert got[4] == want[4] and got[2] == fit.EPS_GRID[1]
    assert max(res.torque_sigma) < 1e-3
    assert fit.physical(res.bodies)
    assert max(fit.residual_rms(logs, res.dynamics(), hold_steps=None)) < 1e-3


def test_the_fit_reads_only_free_motion_samples(synthetic, tmp_path):
    _, trajs, model, logs = synthetic
    clean = fit.fit(logs[:1], dm.urdf_bodies(), hold_steps=None)
    tr = trajs[0]
    n = len(tr["t"])
    tau = np.array(
        [
            np.add(model.inverse(list(q), list(dq), list(ddq)), model.friction(list(dq)))
            for q, dq, ddq in zip(tr["q"], tr["dq"], tr["ddq"])
        ]
    )
    gate, clamped = np.ones(n), np.zeros(n, dtype=bool)
    bad = np.zeros(n, dtype=bool)
    bad[500:600] = True  # a contact: the gate fell
    gate[500:600] = 0.0
    bad[1500:1550] = True  # a clamped command
    clamped[1500:1550] = True
    prompt = (tr["t"] >= 2.5) & (tr["t"] <= 3.0)  # an operator prompt
    bad |= prompt
    tau[bad] += 5.0  # whatever the arm did there must not reach the fit
    dirty = write_session_log(
        tmp_path / "dirty.h5",
        tr["t"],
        tr["q"],
        tr["dq"],
        tau,
        gate=gate,
        clamped=clamped,
        segments=[(2.5, 3.0, "prompt", "tag")],
    )
    res = fit.fit([dirty], dm.urdf_bodies(), hold_steps=None)
    assert np.allclose(res.armature, clean.armature, rtol=0.02, atol=1e-4)
    assert res.report["samples"] < clean.report["samples"] - 150
    assert max(res.torque_sigma) < 1e-3


def test_from_ident_round_trips_the_file(synthetic, tmp_path):
    _, trajs, _, logs = synthetic
    res = fit.fit(logs, dm.urdf_bodies(), hold_steps=None)
    path = tmp_path / "ident.json"
    res.to_json(path)
    data = json.loads(path.read_text())
    assert set(data) >= {"bodies", "friction", "armature", "torque_sigma", "provenance"}
    assert all(len(f) == 5 for f in data["friction"])
    assert data["provenance"]["logs"] == [str(p) for p in logs]
    assert data["provenance"]["sdk_revision"] and data["provenance"]["date"]
    loaded = dm.from_ident(path)
    direct = res.dynamics()
    q, dq, ddq = trajs[0]["q"][700], trajs[0]["dq"][700], trajs[0]["ddq"][700]
    args = (list(q), list(dq), list(ddq))
    assert np.allclose(loaded.inverse(*args), direct.inverse(*args), atol=1e-12)
    assert np.allclose(loaded.friction(list(dq)), direct.friction(list(dq)), atol=1e-12)


def test_from_ident_refuses_a_file_without_bodies(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"friction": [], "armature": []}))
    with pytest.raises(ValueError, match="bodies"):
        dm.from_ident(path)


def test_the_ridge_grows_only_for_the_body_that_is_unphysical(synthetic, monkeypatch):
    _, _, _, logs = synthetic
    calls = []
    real = fit.physical

    def strict(bodies):
        calls.append(1)
        # body 1 (link_003) reads unphysical in the first two rounds of six checks
        return real(bodies) and not (len(calls) <= 12 and len(calls) % 6 == 2)

    monkeypatch.setattr(fit, "physical", strict)
    res = fit.fit(logs[:1], dm.urdf_bodies(), hold_steps=None)
    assert res.report["ridge_growths"] == [0, 2, 0, 0, 0, 0]
    assert res.report["lam"][1] == pytest.approx(1e-3 * 100)
    assert res.report["lam"][0] == pytest.approx(1e-3)
    monkeypatch.setattr(fit, "physical", lambda bodies: False)
    with pytest.raises(fit.IdentError, match="physical"):
        fit.fit(logs[:1], dm.urdf_bodies(), hold_steps=None)


def _payload_bodies():
    """The URDF bodies with 0.2 kg held 5 cm out from link_007's origin."""
    bodies = dm.urdf_bodies()
    tiny = ((1e-6, 0.0, 0.0), (0.0, 1e-6, 0.0), (0.0, 0.0, 1e-6))
    bodies[5] = dm.lump([bodies[5], (0.2, (0.05, 0.0, 0.0), tiny)])
    return bodies


@pytest.mark.parametrize(
    "truth",
    [dm.scaled(dm.urdf_bodies(), 1.25), dm.scaled(dm.urdf_bodies(), 0.8), _payload_bodies()],
    ids=["mass_1.25", "mass_0.8", "payload"],
)
def test_the_fit_recovers_an_arm_that_differs_from_the_urdf(truth, tmp_path):
    """Review round 1: a fixed URDF prior turned these bodies unphysical and the ridge growth
    that followed put the error into j2's armature. Nominal stays the URDF, logs noise-free."""
    model = maker_arm_rs.Dynamics(
        tuple(dm.MOUNT_RPY),
        dm.chain_links(dm.ARM_JOINT_NAMES),
        truth,
        [list(f) for f in TRUE_FRICTION],
        dm.GRAVITY,
        TRUE_ARMATURE,
    )
    trajs = excitation.design_excitation(
        np.random.default_rng(11), SIM, excitation.EXCITE_CENTER, count=3, amplitude_rad=None
    )
    logs = [synthetic_log(tmp_path / f"e{k}.h5", model, tr) for k, tr in enumerate(trajs)]
    res = fit.fit(logs, dm.urdf_bodies(), hold_steps=None)
    assert fit.physical(res.bodies) and res.report["ridge_growths"] == [0] * 6
    # joint 1's armature shares its column with link_002's inertia about the same axis: only
    # their sum is in the data, and the prior splits it
    arm = np.asarray(res.armature)
    assert np.allclose(arm[1:], TRUE_ARMATURE[1:], rtol=0.10), arm
    assert abs(arm[0] - TRUE_ARMATURE[0]) / TRUE_ARMATURE[0] < 0.15, arm
    assert max(res.torque_sigma) < 5e-3, res.torque_sigma
    # The masses and first moments the data sees are combinations (a link's mass at its own
    # origin is its parent's), so they are read through what they do: the gravity torques
    # and the inverse dynamics at random states anywhere in the soft limits.
    ident = res.dynamics()
    rng = np.random.default_rng(0)
    lo = np.array([j["q_lo"] for j in SIM["joints"][:6]])
    hi = np.array([j["q_hi"] for j in SIM["joints"][:6]])
    for q in rng.uniform(lo + 0.1, hi - 0.1, (20, 6)).tolist():
        g_true, g_fit = np.asarray(model.gravity(q)), np.asarray(ident.gravity(q))
        assert np.abs(g_fit - g_true).max() < 0.01 * max(np.abs(g_true).max(), 0.1), q
        dq, ddq = rng.uniform(-2.0, 2.0, 6).tolist(), rng.uniform(-5.0, 5.0, 6).tolist()
        t_true = np.asarray(model.inverse(q, dq, ddq))
        t_fit = np.asarray(ident.inverse(q, dq, ddq))
        assert np.abs(t_fit - t_true).max() < 0.02 * max(np.abs(t_true).max(), 0.1), q


# --- the bench plant ----------------------------------------------------------------------


def test_the_bench_trace_carries_the_measured_torque():
    plant = bench.build_plant()
    trace = bench.run(plant, "B0", [0.0, -1.2, 0.6, -0.5, 0.0, 0.0], [], 0.2, seed=1)
    assert trace.tau.shape == (40, 6) and np.isfinite(trace.tau).all()
    assert np.abs(trace.tau).max() > 0.0


def test_extra_coulomb_adds_to_every_joints_friction_loss():
    base, more = bench.build_plant(), bench.build_plant(extra_coulomb=0.15)
    lost = more.model.dof_frictionloss[more.dof] - base.model.dof_frictionloss[base.dof]
    assert np.allclose(lost, 0.15)
    assert np.allclose(base.model.dof_frictionloss[base.dof], PROFILE_COULOMB)


BENCH_SEEDS = (1, 2, 3)
"""Fixed before reading any result of this configuration. The first round's single seed
(20261016) was the one the excitation knobs were tuned on; the review asked for several."""


def _bench_runs(root, seed, count):
    """``count`` 20 s excitations on the bench plant, rung A, URDF model, as session logs.
    Deviations from the excitation defaults, recorded: they swing to the soft limits (the bench
    has no desk) at a 0.3 Hz fundamental and up to the full velocity cap. At the defaults (0.1
    Hz, half the cap) the joints stay slow and the 0.2 N m torque noise swamps the 0.02 N m
    s/rad viscous term."""
    trajs = excitation.design_excitation(
        np.random.default_rng(seed),
        SIM,
        excitation.EXCITE_CENTER,
        count=count,
        wf_hz=0.3,
        v_scale=1.0,
        amplitude_rad=None,
    )
    plant = bench.build_plant()
    exact = dm.dynamics(dm.urdf_bodies())
    logs = []
    for k, traj in enumerate(trajs):
        tracking = maker_arm_rs.Tracking(exact, params={"contact": True}, profile=SIM)
        rows = np.concatenate([traj["q"], traj["grip"][:, None]], axis=1).tolist()
        trace = bench.run(
            plant,
            "A",
            list(traj["q"][0]),
            [(0.0, 0.0, DT, rows)],
            traj["t"][-1],
            tracking=tracking,
            seed=k,
        )
        logs.append(bench_log(root / f"s{seed}_exc_{k:02d}.h5", trace, float(traj["grip"][0])))
    return logs


@pytest.fixture(scope="module", params=BENCH_SEEDS, ids=lambda s: f"seed{s}")
def bench_seed(request, tmp_path_factory):
    """The realistic bench plant (COULOMB_NM, VISCOUS_NMS, ARM_ARMATURE): four fit and two
    held-out excitations of one seed, and the fit."""
    logs = _bench_runs(tmp_path_factory.mktemp("bench"), request.param, 6)
    return logs[:4], logs[4:], fit.fit(logs[:4], dm.urdf_bodies(), hold_steps=BENCH_HOLD)


def test_the_fit_recovers_the_bench_plants_armature(bench_seed):
    _, _, res = bench_seed
    assert all(abs(a - ARM_ARMATURE) / ARM_ARMATURE < 0.15 for a in res.armature), res.armature
    # the measurement noise is the bench's: the fit explains everything else
    assert all(s < 1.5 * bench.TAU_NOISE_SIGMA for s in res.torque_sigma), res.torque_sigma
    assert math.isfinite(res.report["cond"])


def test_the_identified_model_beats_the_urdf_model_on_held_out_bench_data(bench_seed):
    _, held, res = bench_seed
    ident = fit.residual_rms(held, res.dynamics(), hold_steps=BENCH_HOLD)
    urdf = fit.residual_rms(held, dm.dynamics(dm.urdf_bodies()), hold_steps=BENCH_HOLD)
    assert len(ident) == len(urdf) == 6
    assert all(i < u for i, u in zip(ident, urdf)), (ident, urdf)


def test_the_fit_recovers_viscous_friction_on_a_plant_whose_friction_is_the_models(
    tmp_path_factory,
):
    """The realistic plant's MuJoCo friction loss eases in below 0.1 rad/s, a curve the rs law
    cannot draw, and the fit reads part of it as viscous friction (up to +40 % on one seed).
    With no dry friction in the plant the law is exact; the 0.2 N m noise still leaves one
    seed's viscous term 15 to 50 % off, so the bar is read on the three seeds pooled."""
    root = tmp_path_factory.mktemp("viscous")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bench, "COULOMB_NM", {k: 0.0 for k in bench.COULOMB_NM})
        logs = [log for seed in BENCH_SEEDS for log in _bench_runs(root, seed, 4)]
    res = fit.fit(logs, dm.urdf_bodies(), hold_steps=BENCH_HOLD)
    viscous = [f[1] for f in res.friction]
    assert all(abs(v - bench.VISCOUS_NMS) / bench.VISCOUS_NMS < 0.30 for v in viscous), viscous
    assert all(abs(a - ARM_ARMATURE) / ARM_ARMATURE < 0.10 for a in res.armature), res.armature
