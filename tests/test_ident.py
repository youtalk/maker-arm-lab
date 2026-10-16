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


def test_the_ridge_grows_until_the_bodies_are_physical(synthetic, monkeypatch):
    _, _, _, logs = synthetic
    seen = []
    real = fit.physical

    def strict(bodies):
        seen.append(1)
        return len(seen) > 2 and real(bodies)  # the first two solves read as unphysical

    monkeypatch.setattr(fit, "physical", strict)
    res = fit.fit(logs[:1], dm.urdf_bodies(), hold_steps=None)
    assert res.report["ridge_growths"] == 2
    assert res.report["lam"] == pytest.approx(1e-3 * 100)
    monkeypatch.setattr(fit, "physical", lambda bodies: False)
    with pytest.raises(fit.IdentError, match="physical"):
        fit.fit(logs[:1], dm.urdf_bodies(), hold_steps=None)


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


BENCH_FIT, BENCH_HELD = 4, 2


@pytest.fixture(scope="module")
def bench_logs(tmp_path_factory):
    """Four fit and two held-out 20 s excitations on the bench plant, rung A, URDF model. They
    swing to the soft limits (the bench has no desk) at a 0.3 Hz fundamental and up to the
    full velocity cap: at the defaults (0.1 Hz, half the cap) the joints stay slow and the
    0.2 N m torque noise swamps the 0.02 N m s/rad viscous term."""
    root = tmp_path_factory.mktemp("bench")
    trajs = excitation.design_excitation(
        np.random.default_rng(20261016),
        SIM,
        excitation.EXCITE_CENTER,
        count=BENCH_FIT + BENCH_HELD,
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
        logs.append(bench_log(root / f"exc_{k:02d}.h5", trace, float(traj["grip"][0])))
    return logs[:BENCH_FIT], logs[BENCH_FIT:]


@pytest.fixture(scope="module")
def bench_fit(bench_logs):
    logs, _ = bench_logs
    return fit.fit(logs, dm.urdf_bodies(), hold_steps=BENCH_HOLD)


def test_the_fit_recovers_the_bench_plants_armature_and_friction(bench_fit):
    res = bench_fit
    assert all(abs(a - ARM_ARMATURE) / ARM_ARMATURE < 0.15 for a in res.armature), res.armature
    assert all(abs(f[1] - bench.VISCOUS_NMS) / bench.VISCOUS_NMS < 0.30 for f in res.friction), [
        f[1] for f in res.friction
    ]
    # the measurement noise is the bench's: the fit explains everything else
    assert all(s < 1.5 * bench.TAU_NOISE_SIGMA for s in res.torque_sigma), res.torque_sigma
    assert math.isfinite(res.report["cond"])


def test_the_identified_model_beats_the_urdf_model_on_held_out_bench_data(bench_logs, bench_fit):
    _, held = bench_logs
    res = bench_fit
    ident = fit.residual_rms(held, res.dynamics(), hold_steps=BENCH_HOLD)
    urdf = fit.residual_rms(held, dm.dynamics(dm.urdf_bodies()), hold_steps=BENCH_HOLD)
    assert len(ident) == len(urdf) == 6
    assert all(i < u for i, u in zip(ident, urdf)), (ident, urdf)
