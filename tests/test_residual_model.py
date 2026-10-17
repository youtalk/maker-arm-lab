"""MS5 control plan task 10: the residual torque model, its dataset from session logs, its
training and its export to the ``maker_arm_rs`` residual format."""

import ast
import hashlib
import json
import os
import subprocess
import sys

import maker_arm_rs
import numpy as np
import pytest
import torch

from helpers import ROOT
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as core_profile
from maker_arm_lab.ident import excitation, fit
from maker_arm_lab.models import residual
from test_ident import write_session_log

SIM = core_profile.sim_profile()
SCRIPT = ROOT / "scripts" / "real_residual_train.py"
FRICTION = [[0.1, 0.02, 0.01, 0.15, 0.05]] * 6
ARMATURE = [0.02] * 6


def _model():
    return dm.dynamics(dm.urdf_bodies(), FRICTION, armature=ARMATURE)


def _traj(seed=3, seconds=5.0):
    (tr,) = excitation.design_excitation(
        np.random.default_rng(seed),
        SIM,
        excitation.EXCITE_CENTER,
        count=1,
        wf_hz=0.2,
        duration_s=seconds,
        amplitude_rad=None,
    )
    return tr


def _temps(n, seed=0):
    rng = np.random.default_rng(seed)
    return 35.0 + np.cumsum(rng.normal(0.0, 0.01, (n, 6)), axis=0)


def _log(path, model, tr, *, extra=None, temp=None, keep=None, **kw):
    """tau = the model's inverse dynamics and friction along ``tr`` plus ``extra(dq, temp)``;
    ``keep`` a mask of the ticks to log (a dropped tick is a gap)."""
    n = len(tr["t"])
    temp = _temps(n) if temp is None else temp
    tau = np.array(
        [
            np.add(model.inverse(list(q), list(dq), list(ddq)), model.friction(list(dq)))
            for q, dq, ddq in zip(tr["q"], tr["dq"], tr["ddq"])
        ]
    )
    if extra is not None:
        tau = tau + extra(tr["dq"], temp)
    cols = [tr["t"], tr["q"], tr["dq"], tau, temp]
    if keep is not None:
        cols = [c[keep] for c in cols]
    t, q, dq, tau, temp = cols
    return write_session_log(path, t, q, dq, tau, temp=temp, **kw)


# --- the package boundary -----------------------------------------------------------------


def test_the_models_package_imports_only_what_the_plan_allows():
    allowed = {"torch", "numpy", "h5py", "maker_arm_rs"}
    paths = sorted((ROOT / "source/maker_arm_lab/models").glob("*.py"))
    assert paths
    for path in paths:
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
                    or name.startswith(("maker_arm_lab.core", "maker_arm_lab.ident"))
                )
                assert ok, (path.name, name)


# --- the dataset --------------------------------------------------------------------------


def test_the_target_is_zero_on_a_log_of_the_model_and_the_input_is_the_history(tmp_path):
    model, tr = _model(), _traj()
    n = len(tr["t"])
    temp = _temps(n)
    log = _log(tmp_path / "a.h5", model, tr, temp=temp)
    info = {}
    X, Y = residual.build_dataset([log], model, 4, hold_steps=None, info=info)
    assert X.shape == (len(Y), 12 * 4 + 6) and Y.shape[1] == 6 and len(Y) > 0.9 * n
    # the smoothed ddq estimate, nothing else
    assert np.sqrt(np.mean(Y**2)) < 2e-3 and np.max(np.abs(Y)) < 2e-2
    k = np.searchsorted(tr["t"], info["t"])
    for h in range(4):  # block h is tick k - 3 + h: oldest first
        assert np.array_equal(X[:, 12 * h : 12 * h + 6], tr["q"][k - 3 + h])
        assert np.array_equal(X[:, 12 * h + 6 : 12 * h + 12], tr["dq"][k - 3 + h])
    assert np.array_equal(X[:, 48:], temp[k])


def test_the_target_is_the_fits_rows_less_the_model_under_the_hold_model(tmp_path):
    model, tr = _model(), _traj()
    log = _log(tmp_path / "a.h5", model, tr, kp=[20.0] * 6, kd=[1.0] * 6)
    info = {}
    _, Y = residual.build_dataset([log], model, 10, hold_steps=0, info=info)
    d = fit.load_free_motion(log, hold_steps=0)
    assert info["no_history"] > 0 and info["straddled"] == 0
    assert len(Y) == d["samples"] - info["no_history"]
    want = d["tau"][-len(Y) :] - np.array(
        [
            np.add(model.inverse(list(q), list(dq), list(a)), model.friction(list(dq)))
            for q, dq, a in zip(d["q"][-len(Y) :], d["dq"][-len(Y) :], d["ddq"][-len(Y) :])
        ]
    )
    assert np.allclose(Y, want, atol=1e-12)
    # the friction is not counted twice: inverse alone is rigid body plus armature
    q, dq, a = [0.1] * 6, [0.3] * 6, [0.0] * 6
    bare = dm.dynamics(dm.urdf_bodies(), armature=ARMATURE)
    assert np.allclose(model.inverse(q, dq, a), bare.inverse(q, dq, a))


def test_a_history_window_never_straddles_a_gap_or_a_segment_boundary(tmp_path):
    model, tr = _model(), _traj()
    n, dt = len(tr["t"]), float(tr["t"][1] - tr["t"][0])
    keep = np.ones(n, dtype=bool)
    keep[300] = False  # one lost tick: a gap
    t_gap = tr["t"][300]
    t_seg = float(tr["t"][650])
    segs = [(0.0, t_seg - 0.5 * dt, "traj", "a"), (t_seg, float(tr["t"][-1]), "traj", "b")]
    log = _log(tmp_path / "a.h5", model, tr, keep=keep, segments=segs)
    info = {}
    _, Y = residual.build_dataset([log], model, 10, hold_steps=None, info=info)
    # a window of ten ticks reaches back 9 dt, or 10 dt across the one lost tick
    assert not np.any((info["t"] - 9.5 * dt < t_gap) & (info["t"] > t_gap))
    assert not np.any((info["t"] - 9 * dt < t_seg - 0.25 * dt) & (info["t"] >= t_seg))
    assert info["straddled"] > 0
    assert len(Y) == info["rows"]


def test_rows_with_a_nan_are_dropped_and_counted(tmp_path):
    model, tr = _model(), _traj()
    n = len(tr["t"])
    clean = _log(tmp_path / "clean.h5", model, tr)
    temp = _temps(n)
    temp[300, 2] = np.nan

    def extra(dq, _temp):
        out = np.zeros((n, 6))
        out[600, 4] = np.nan
        return out

    dirty = _log(tmp_path / "dirty.h5", model, tr, temp=temp, extra=extra)
    info0, info1 = {}, {}
    _, y0 = residual.build_dataset([clean], model, 4, hold_steps=None, info=info0)
    x1, y1 = residual.build_dataset([dirty], model, 4, hold_steps=None, info=info1)
    assert info0["non_finite"] == 0 and info1["non_finite"] == 2
    assert len(y1) == len(y0) - 2
    assert np.isfinite(x1).all() and np.isfinite(y1).all()


# --- the model, the training and the export -----------------------------------------------


def _stats(n_in, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "x_mean": rng.normal(0.0, 1.0, n_in),
        "x_std": rng.uniform(0.1, 2.0, n_in),
        "y_mean": rng.normal(0.0, 0.5, 6),
        "y_std": rng.uniform(0.05, 1.0, 6),
    }


def test_the_export_matches_the_rust_mlp_to_1e_9(tmp_path):
    torch.manual_seed(0)
    n_in = 12 * 4 + 6
    model = residual.ResidualMlp(n_in)
    stats = _stats(n_in)
    residual.export(model, stats, 4, [1.0] * 6, tmp_path / "m.json")
    spec = json.loads((tmp_path / "m.json").read_text())
    assert set(spec) == {"history", "layers", "x_mean", "x_std", "y_scale", "out_max"}
    assert spec["history"] == 4 and spec["out_max"] == [1.0] * 6
    rng = np.random.default_rng(2)
    mlp = maker_arm_rs.Mlp(spec["layers"])
    for _ in range(20):
        x = rng.normal(size=n_in) * stats["x_std"] + stats["x_mean"]
        xn = (x - np.asarray(spec["x_mean"])) / np.asarray(spec["x_std"])
        rust = np.asarray(mlp.forward(list(xn))) * np.asarray(spec["y_scale"])
        want = residual.predict(model, stats, x[None])[0]
        with torch.no_grad():
            raw = model(torch.from_numpy((x - stats["x_mean"]) / stats["x_std"]))
        assert np.allclose(want, raw.numpy() * stats["y_std"] + stats["y_mean"], atol=1e-12)
        assert np.allclose(rust, want, atol=1e-9)
    maker_arm_rs.Tracking(_model(), residual=spec)  # the SDK takes the file as it is
    with pytest.raises(FileExistsError):
        residual.export(model, stats, 4, [1.0] * 6, tmp_path / "m.json")


def test_out_max_defaults_to_three_standard_deviations_of_the_target(tmp_path):
    model = residual.ResidualMlp(12 + 6, hidden=(8,))
    stats = _stats(18)
    residual.export(model, stats, 1, None, tmp_path / "m.json")
    spec = json.loads((tmp_path / "m.json").read_text())
    assert residual.OUT_MAX_SIGMA == 3.0
    assert np.allclose(spec["out_max"], 3.0 * stats["y_std"])
    assert len(spec["layers"]) == 2 and len(spec["layers"][0][0][0]) == 18


def _synthetic(f, history=4, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(history):
        blocks += [rng.normal(0.5, 0.4, (n, 6)), rng.normal(0.0, 0.2, (n, 6))]
    temp = rng.uniform(30.0, 50.0, (n, 6))
    X = np.concatenate([*blocks, temp], axis=1)
    return X, f(blocks[-1], temp)


def test_a_learnable_residual_is_learned():
    X, Y = _synthetic(lambda dq, temp: 0.3 * np.tanh(dq / 0.05) + 0.01 * (temp - 40))
    model, rep = residual.train(X, Y)
    assert all(v < 0.3 * z for v, z in zip(rep["val_rmse"], rep["zero_rmse"])), rep
    assert rep["n_val"] == 600 and rep["n_train"] == 2400 - 4 and rep["embargo"] == 4
    assert 0 <= rep["best_epoch"] < 300
    assert next(model.parameters()).dtype == torch.float64


def test_the_validation_split_is_the_last_rows_in_time():
    X, Y = _synthetic(lambda dq, temp: 0.0 * dq)
    Y[-600:] = 1.0  # only the last fifth carries a torque: the zero model's val error is 1
    _, rep = residual.train(X, Y, epochs=2)
    assert np.allclose(rep["zero_rmse"], 1.0)


def test_the_input_scale_is_floored_and_the_far_rows_are_counted():
    X, Y = _synthetic(lambda dq, temp: 0.1 * dq)
    X[:, :6] = 0.3  # the oldest q block never moves
    X[:, -6:] = 40.0  # nor the temperature
    X[-1, 6] = 100.0  # a reading far outside: past X_BOUND
    _, rep = residual.train(X, Y, epochs=1)
    std = np.asarray(rep["stats"]["x_std"])
    assert np.allclose(std[:6], residual.Q_STD_FLOOR)
    assert np.allclose(std[-6:], residual.TEMP_STD_FLOOR)
    assert np.all(std[6:12] >= residual.DQ_STD_FLOOR)
    assert rep["beyond_x_bound"] == {"train": 0, "val": 1}


def test_train_refuses_an_input_that_is_not_a_history():
    with pytest.raises(ValueError):
        residual.train(np.zeros((100, 20)), np.zeros((100, 6)))


def _sdk_residual(spec, model, t, q, dq, temp):
    """The SDK's ``tau_res`` per tick: every tick fed to ``Tracking.update`` in order."""
    trk = maker_arm_rs.Tracking(model, residual=spec)
    out = []
    for k in range(len(t)):
        trk.update(float(t[k]), [*q[k], 0.0], [*dq[k], 0.0], [0.0] * 7, [*map(float, temp[k]), 0.0])
        out.append(trk.telemetry()["tau_res"])
    return np.asarray(out), trk.telemetry()["residual_skips"]


def test_the_sdk_reads_the_same_window_as_the_dataset(tmp_path):
    model, tr = _model(), _traj()
    temp = _temps(len(tr["t"]))
    log = _log(tmp_path / "a.h5", model, tr, temp=temp)
    info = {}
    X, Y = residual.build_dataset([log], model, 4, hold_steps=None, info=info)
    torch.manual_seed(0)
    net = residual.ResidualMlp(residual.n_inputs(4), hidden=(16,))
    stats = residual.fit_stats(X, Y, 4)
    stats["y_std"] = np.full(6, 0.5)  # a visible output; out_max at tau_max never clamps it
    tau_max = [float(j["tau_max"]) for j in core_profile.load_profile()["joints"][:6]]
    spec = residual.export(net, stats, 4, tau_max, tmp_path / "m.json")
    sdk, _ = _sdk_residual(spec, model, tr["t"], tr["q"], tr["dq"], temp)
    k = np.searchsorted(tr["t"], info["t"])
    want = residual.predict(net, stats, X)
    assert len(k) > 900 and np.abs(sdk[k] - want).max() < 1e-9
    assert np.abs(sdk[k + 1] - want).max() > 1e-4  # one tick off is visible


def test_x_bound_is_the_sdks():
    net = residual.ResidualMlp(residual.n_inputs(1), hidden=(4,))
    n_in = residual.n_inputs(1)
    stats = {"x_mean": np.zeros(n_in), "x_std": np.ones(n_in), "y_mean": np.zeros(6)}
    stats["y_std"] = np.ones(6)
    spec = residual.residual_spec(net, stats, 1, [1.0] * 6)  # no floors: unit scale
    model = _model()
    for value, skips in ((residual.X_BOUND, 0), (residual.X_BOUND + 1e-9, 1)):
        q = np.zeros((1, 6))
        q[0, 2] = value
        _, got = _sdk_residual(spec, model, [0.0], q, np.zeros((1, 6)), np.zeros((1, 6)))
        assert got == skips, value


# --- the script ---------------------------------------------------------------------------


def _run(args, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    tokens = [s for s in out.stdout.splitlines() if s.startswith(("RESID_OK", "RESID_FAIL"))]
    return out.returncode, tokens, out.stderr


def test_the_train_script_usage_error_prints_one_fail_line():
    code, tokens, _ = _run(["--history", "10"])
    assert code == 2 and len(tokens) == 1
    assert json.loads(tokens[0].split(" ", 1)[1])["reason"] == "usage"


def _ident_file(path):
    data = {
        "bodies": [[m, list(c), [list(r) for r in i]] for m, c, i in dm.urdf_bodies()],
        "friction": FRICTION,
        "armature": ARMATURE,
        "torque_sigma": [0.1] * 6,
        "provenance": {"logs": []},
        "report": {"hold_steps": None},
    }
    path.write_text(json.dumps(data))
    return path


def test_the_train_script_learns_a_residual_and_never_overwrites(tmp_path):
    model = _model()
    logs = [
        _log(
            tmp_path / f"s{k}.h5",
            model,
            _traj(seed=10 + k, seconds=10.0),
            extra=lambda dq, temp: 0.2 * np.tanh(dq / 0.05),
        )
        for k in range(2)
    ]
    ident = _ident_file(tmp_path / "ident.json")
    out, rep = tmp_path / "m.json", tmp_path / "rep.json"
    args = ["--logs", *logs, "--ident", ident, "--history", "4", "--out", out, "--report", rep]
    code, tokens, err = _run([*args, "--epochs", "40"])
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("RESID_OK"), err
    payload = json.loads(tokens[0].split(" ", 1)[1])
    assert all(v < 0.5 * z for v, z in zip(payload["val_rmse"], payload["zero_rmse"]))
    spec = json.loads(out.read_text())
    maker_arm_rs.Tracking(model, residual=spec)
    report = json.loads(rep.read_text())
    assert report["hold_steps"] is None  # read from the ident file
    split = report["train"]
    assert report["dataset"]["rows"] == split["n_train"] + split["embargo"] + split["n_val"]
    assert "beyond_x_bound" in report["train"] and report["provenance"]["ident"] == str(ident)
    # ruling R35: the report ties the model to its ident file and to the model file itself
    sha = hashlib.sha256
    assert report["provenance"]["ident_sha256"] == sha(ident.read_bytes()).hexdigest()
    assert report["model_sha256"] == sha(out.read_bytes()).hexdigest()
    before = out.read_text()
    code, tokens, _ = _run([*args, "--epochs", "1"])
    assert code == 1 and len(tokens) == 1
    assert json.loads(tokens[0].split(" ", 1)[1])["reason"] == "out_exists"
    assert out.read_text() == before


def test_the_train_script_fails_on_a_bad_ident_file(tmp_path):
    bad = tmp_path / "ident.json"
    bad.write_text("{}")
    log = _log(tmp_path / "s.h5", _model(), _traj())
    code, tokens, _ = _run(
        ["--logs", log, "--ident", bad, "--out", tmp_path / "m.json", "--report", tmp_path / "r"]
    )
    assert code == 1 and len(tokens) == 1
    assert json.loads(tokens[0].split(" ", 1)[1])["reason"] == "ident"
    assert not (tmp_path / "m.json").exists()


def test_the_train_script_refuses_a_hold_model_other_than_the_idents(tmp_path):
    log = _log(tmp_path / "s.h5", _model(), _traj())
    ident = _ident_file(tmp_path / "ident.json")  # fitted under hold_steps none
    out = tmp_path / "m.json"
    code, tokens, _ = _run(
        ["--logs", log, "--ident", ident, "--hold_steps", "0", "--out", out]
        + ["--report", tmp_path / "r.json"]
    )
    assert code == 1 and len(tokens) == 1
    assert json.loads(tokens[0].split(" ", 1)[1])["reason"] == "hold_steps"
    assert not out.exists()


def test_an_out_max_past_the_joints_torque_cap_is_refused(tmp_path):
    torch.manual_seed(0)
    n_in = residual.n_inputs(2)
    stats = {
        "x_mean": np.zeros(n_in),
        "x_std": np.ones(n_in),
        "y_mean": np.zeros(6),
        "y_std": np.full(6, 0.01),
    }
    model = residual.ResidualMlp(n_in, hidden=(8,))
    tau_max = [float(j["tau_max"]) for j in core_profile.load_profile()["joints"][:6]]
    at_cap = residual.residual_spec(model, stats, 2, tau_max)
    assert at_cap["out_max"] == tau_max
    over = list(tau_max)
    over[2] *= 1.01
    with pytest.raises(ValueError, match=r"out_max\[2\]"):
        residual.residual_spec(model, stats, 2, over)
    with pytest.raises(ValueError, match="tau_max"):
        residual.residual_spec(model, {**stats, "y_std": np.full(6, 100.0)}, 2)  # 3 sigma: 300
    assert not list(tmp_path.iterdir())
