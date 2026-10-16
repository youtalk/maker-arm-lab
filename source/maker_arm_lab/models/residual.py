"""The learned residual torque (MS5 control plan task 10, the MA3 recipe): a small MLP gives
per joint the torque the identified dynamics model misses, from a short history of the
measured joint state and the motor temperatures. It is trained here in torch and runs inside
``maker_arm_rs``' 200 Hz loop (``Tracking(residual=...)``).

The input of tick k is ``history`` blocks ``[q (6), dq (6)]`` of the measured ticks k -
history + 1 .. k, oldest first, then the six arm motors' temperatures at tick k: ``12 history
+ 6`` values. It is never the position error: the residual must be a function of the arm's
state, not of what the controller asked for, or it would learn to cancel the tracking. The
target is the measured torque less ``dynamics.inverse(q, dq, ddq)`` (the rigid body and the
armature) less ``dynamics.friction(dq)``, on the free-motion rows the identification fits
(``ident.fit.free_motion_ticks`` and ``free_motion_rows``, the same selector and the same
``hold_model``, so the residual is what the model misses on exactly the rows it was fitted
to). Under a hold model row k is the hold that follows tick k, which is the hold the SDK's
``tau_res`` of tick k acts over; its input is still the raw readings up to tick k, as the SDK
sees them.

The export is the SDK's residual dict, exactly ``history``, ``layers``, ``x_mean``, ``x_std``,
``y_scale`` and ``out_max``: ``layers`` is ``[[w, b], ...]`` (torch's ``weight.tolist()``,
out rows of ``in`` values), tanh between layers and none after the last. The SDK normalizes
the input by ``x_mean`` and ``x_std``, scales the MLP's output by ``y_scale`` and clamps it to
``+-out_max``. It has no output mean, so the export folds the target mean into the last
layer's bias, ``b + y_mean / y_scale``. The SDK skips (gives zero for) a tick with a
normalized input past ``X_BOUND``, so the input scale is floored per feature
(``Q_STD_FLOOR``, ``DQ_STD_FLOOR``, ``TEMP_STD_FLOOR``): a session that held still, or ran at
one temperature, would otherwise give a scale so narrow that every later reading skips.
``out_max`` never exceeds the joint's ``tau_max`` of ``maker_arm_rs.profile()``: the export
refuses one that does (``residual_spec``), since a residual past the motor's torque cap is a
model gone wrong, not a correction.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from maker_arm_lab.core.profile import load_profile
from maker_arm_lab.ident import fit

ARM_DOF = 6
BLOCK = 2 * ARM_DOF
"""One tick of the history: q (6) then dq (6)."""
X_BOUND = 10.0
"""``maker_arm::residual::X_BOUND`` of the pinned SDK: the SDK skips a tick whose normalized
input is past it."""
Q_STD_FLOOR = 0.05  # INITIAL: session 8
"""rad: the smallest input scale of a joint angle. Ten of them, 0.5 rad, is about the reach
of one excitation's swing about its center: a model trained on one pose keeps acting over
the neighbourhood the arm works in and skips far from it, where it has seen nothing."""
DQ_STD_FLOOR = 0.05  # INITIAL: session 8
"""rad/s: the smallest input scale of a joint velocity. At rest the velocity reading is
noise of about 0.01 rad/s; ten of these, 0.5 rad/s, is a slow approach, so a model trained
on holds still runs (and is clamped by ``out_max``) through the approach to them."""
TEMP_STD_FLOOR = 5.0  # INITIAL: session 8
"""degC: the smallest input scale of a motor temperature. A session runs at one temperature
to within a few degrees; ten of these, 50 degC, covers the motor's whole range from a cold
start to its 70 degC limit, so a temperature never skips a tick on its own."""
Y_STD_FLOOR = 1e-3
"""N m: the smallest output scale, to keep a joint whose target is constant from dividing by
zero. Far below any torque the arm's motors resolve."""
OUT_MAX_SIGMA = 3.0  # INITIAL: written in docs/ms5-foundations.md before session 8
"""The default ``out_max``: this many standard deviations of the training target, per joint."""
BATCH = 256
HIDDEN = (64, 64)


# --- the dataset --------------------------------------------------------------------------


def n_inputs(history: int) -> int:
    return BLOCK * int(history) + ARM_DOF


def history_of(n_in: int) -> int:
    """The history of an input ``n_in`` wide. ValueError for a width that is not one."""
    h, rest = divmod(int(n_in) - ARM_DOF, BLOCK)
    if h < 1 or rest:
        raise ValueError(f"an input of {n_in} is not 12 history + 6 values")
    return h


def _segment_ids(t: np.ndarray, segments) -> np.ndarray:
    """(N,) per tick the index of the last segment that holds it, -1 outside every one."""
    ids = np.full(t.shape[0], -1, dtype=np.int64)
    for i, seg in enumerate(segments):
        ids[(t >= seg["start_t"]) & (t <= seg["end_t"])] = i
    return ids


def build_dataset(
    logs: Sequence[Path],
    dynamics,
    history: int,
    hold_steps: int | None = fit.HOLD_STEPS,
    info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """X (N, 12 history + 6) and Y (N, 6) over the free-motion rows of ``logs`` (see the module
    docstring), the logs in order and each in time. A row is kept only when its whole history
    window lies in one run of regular ticks inside one segment: a window across a tick gap or
    a segment boundary is not the motion the SDK would have seen (it restarts its history at
    a gap). The step rule is ``fit``'s (within half a period of the median), stricter than the
    SDK's ``GAP_S`` (20 ms), so the lab drops some rows the SDK still evaluates. ``info``, if given, receives the counts: per log and in total the ``candidates``
    (the rows ``fit`` uses), the ``rows`` kept, and the rows dropped for ``no_history`` (too
    close to the log's start), ``straddled`` (a gap or a boundary) and ``non_finite`` (a
    non-finite input or target), and ``t`` (N,) the time of each kept row's tick."""
    history = int(history)
    if history < 1:
        raise ValueError(f"history must be at least 1, got {history}")
    xs, ys, ts, per_log = [], [], [], []
    for path in logs:
        ticks = fit.free_motion_ticks(path, columns=("temp",))
        q, dq, ddq, tau, use = fit.free_motion_rows(ticks, hold_steps)
        rows = np.arange(use.shape[0])
        # a run of ticks: no irregular step and no change of segment between its neighbours
        seg = _segment_ids(ticks["t"], ticks["segments"])
        cut = ~ticks["regular"] | (np.diff(seg) != 0)
        run = np.concatenate([[0], np.cumsum(cut)])
        early = rows < history - 1
        first = np.maximum(rows - history + 1, 0)
        same = run[rows] == run[first]
        keep = use & ~early & same
        idx = rows[keep]
        tq, tdq = ticks["q"], ticks["dq"]
        blocks = []
        for h in range(history):
            at = idx - history + 1 + h
            blocks += [tq[at], tdq[at]]
        temp = np.asarray(ticks["temp"], dtype=np.float64)[:, :ARM_DOF]
        x = np.concatenate([*blocks, temp[idx]], axis=1).reshape(idx.size, n_inputs(history))
        model = np.full((idx.size, ARM_DOF), np.nan)
        for i, k in enumerate(idx):
            a, v, acc = q[k], dq[k], ddq[k]
            if np.isfinite(a).all() and np.isfinite(v).all() and np.isfinite(acc).all():
                model[i] = np.add(
                    dynamics.inverse(a.tolist(), v.tolist(), acc.tolist()),
                    dynamics.friction(v.tolist()),
                )
        y = tau[idx] - model
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        xs.append(x[finite])
        ys.append(y[finite])
        ts.append(ticks["t"][idx][finite])
        per_log.append(
            {
                "log": str(path),
                "ticks": int(ticks["t"].shape[0]),
                "candidates": int(use.sum()),
                "no_history": int((use & early).sum()),
                "straddled": int((use & ~early & ~same).sum()),
                "non_finite": int((~finite).sum()),
                "rows": int(finite.sum()),
            }
        )
    width = n_inputs(history)
    X = np.concatenate(xs) if xs else np.zeros((0, width))
    Y = np.concatenate(ys) if ys else np.zeros((0, ARM_DOF))
    if info is not None:
        info.clear()
        for key in ("candidates", "no_history", "straddled", "non_finite", "rows"):
            info[key] = sum(p[key] for p in per_log)
        info.update(
            {
                "history": history,
                "hold_steps": hold_steps,
                "per_log": per_log,
                "t": np.concatenate(ts) if ts else np.zeros(0),
            }
        )
    return X, Y


# --- the model ----------------------------------------------------------------------------


class ResidualMlp(nn.Module):
    """Tanh hidden layers and a linear output of six, in float64 (the SDK runs f64)."""

    def __init__(self, n_in: int, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        widths = [int(n_in), *map(int, hidden), ARM_DOF]
        self.linears = nn.ModuleList(
            nn.Linear(a, b, dtype=torch.float64) for a, b in itertools.pairwise(widths)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears[:-1]:
            x = torch.tanh(layer(x))
        return self.linears[-1](x)


def input_floor(history: int) -> np.ndarray:
    """(12 history + 6,) the floor of each input's scale."""
    block = [Q_STD_FLOOR] * ARM_DOF + [DQ_STD_FLOOR] * ARM_DOF
    return np.asarray(block * int(history) + [TEMP_STD_FLOOR] * ARM_DOF, dtype=np.float64)


def fit_stats(X: np.ndarray, Y: np.ndarray, history: int) -> dict:
    """The normalization: ``x_mean``, ``x_std`` (floored by ``input_floor``), ``y_mean`` and
    ``y_std`` (floored at ``Y_STD_FLOOR``)."""
    return {
        "x_mean": X.mean(axis=0),
        "x_std": np.maximum(X.std(axis=0), input_floor(history)),
        "y_mean": Y.mean(axis=0),
        "y_std": np.maximum(Y.std(axis=0), Y_STD_FLOOR),
    }


def _arrays(stats: dict) -> dict:
    return {
        k: np.asarray(stats[k], dtype=np.float64) for k in ("x_mean", "x_std", "y_mean", "y_std")
    }


def beyond_bound(X: np.ndarray, stats: dict) -> int:
    """The rows of ``X`` with a normalized input past ``X_BOUND``: the ticks the SDK skips."""
    s = _arrays(stats)
    if X.shape[0] == 0:
        return 0
    return int((np.abs((X - s["x_mean"]) / s["x_std"]) > X_BOUND).any(axis=1).sum())


def predict(model: ResidualMlp, stats: dict, X: np.ndarray) -> np.ndarray:
    """(N, 6) N m: the model's residual torque of the raw inputs ``X`` (before ``out_max``)."""
    s = _arrays(stats)
    with torch.no_grad():
        out = model(torch.from_numpy((np.asarray(X, dtype=np.float64) - s["x_mean"]) / s["x_std"]))
    return out.numpy() * s["y_std"] + s["y_mean"]


def _rmse(err: np.ndarray) -> list[float]:
    return [float(v) for v in np.sqrt(np.mean(err**2, axis=0))]


def train(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 1e-3,
    val_share: float = 0.2,
    *,
    hidden: tuple[int, ...] = HIDDEN,
    batch_size: int = BATCH,
) -> tuple[ResidualMlp, dict]:
    """Adam on the normalized X and Y in shuffled minibatches. The validation rows are the
    last ``val_share`` of the rows (``build_dataset`` keeps them in time order), the
    ``history`` rows before them are dropped (``embargo``: a window of a training row never
    reaches into the validation rows' ticks), the normalization comes from the training rows
    alone, and the model kept is the one of the epoch with the lowest validation loss. The
    report holds per joint the train and validation RMSE and the zero model's (no residual)
    validation RMSE, in N m, the normalization (``stats``) and the rows past ``X_BOUND``. The
    validation rows chose the epoch, so ``val_rmse`` is on the selection set, not a held-out
    read: a held-out read needs logs the training never saw."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    history = history_of(X.shape[1])
    if Y.shape != (X.shape[0], ARM_DOF):
        raise ValueError(f"Y needs shape ({X.shape[0]}, {ARM_DOF}), got {Y.shape}")
    n = X.shape[0]
    n_val = round(val_share * n)
    if not 1 <= n_val < n:
        raise ValueError(f"{n} rows leave no train or no validation split at {val_share}")
    # the embargo: the last ``history`` training rows are dropped, so no input window holds
    # ticks of both splits
    n_train = n - n_val - history
    if n_train < 1:
        raise ValueError(f"{n} rows leave no train split after the {history}-row embargo")
    stats = fit_stats(X[:n_train], Y[:n_train], history)

    def norm(x, y):
        xn = (x - stats["x_mean"]) / stats["x_std"]
        yn = (y - stats["y_mean"]) / stats["y_std"]
        return torch.from_numpy(xn), torch.from_numpy(yn)

    xt, yt = norm(X[:n_train], Y[:n_train])
    xv, yv = norm(X[n - n_val :], Y[n - n_val :])
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ResidualMlp(X.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_epoch, best_state = math.inf, -1, copy.deepcopy(model.state_dict())
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, batch_size):
            b = perm[start : start + batch_size]
            loss = torch.mean((model(xt[b]) - yt[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.mean((model(xv) - yv) ** 2))
        if val < best_loss:
            best_loss, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    y_train, y_val = Y[:n_train], Y[n - n_val :]
    report = {
        "history": history,
        "rows": n,
        "n_train": n_train,
        "n_val": n_val,
        "embargo": history,
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "seed": int(seed),
        "lr": float(lr),
        "hidden": [int(h) for h in hidden],
        "batch_size": int(batch_size),
        "train_rmse": _rmse(predict(model, stats, X[:n_train]) - y_train),
        "val_rmse": _rmse(predict(model, stats, X[n - n_val :]) - y_val),
        "zero_rmse": _rmse(y_val),
        "zero_train_rmse": _rmse(y_train),
        "y_std": [float(v) for v in stats["y_std"]],
        "x_bound": X_BOUND,
        "x_std_floored": int((stats["x_std"] == input_floor(history)).sum()),
        "beyond_x_bound": {
            "train": beyond_bound(X[:n_train], stats),
            "val": beyond_bound(X[n - n_val :], stats),
        },
        "stats": {k: [float(v) for v in a] for k, a in stats.items()},
    }
    return model, report


# --- the export ---------------------------------------------------------------------------


def residual_spec(
    model: ResidualMlp,
    stats: dict,
    history: int,
    out_max: Sequence[float] | None = None,
    tau_max: Sequence[float] | None = None,
) -> dict:
    """The SDK's residual dict (see the module docstring). ``out_max`` None is
    ``OUT_MAX_SIGMA`` times ``stats["y_std"]``. ValueError, naming the joint, for an
    ``out_max`` past ``tau_max`` (default: the six arm joints' of ``load_profile()``)."""
    s = _arrays(stats)
    width = n_inputs(history)
    if s["x_mean"].shape != (width,) or s["x_std"].shape != (width,):
        raise ValueError(f"history {history} needs {width} input statistics")
    limit = OUT_MAX_SIGMA * s["y_std"] if out_max is None else np.asarray(out_max, dtype=float)
    if limit.shape != (ARM_DOF,) or not (np.isfinite(limit).all() and (limit >= 0).all()):
        raise ValueError(f"out_max needs six finite non-negative values, got {limit.tolist()}")
    if tau_max is None:
        tau_max = [float(j["tau_max"]) for j in load_profile()["joints"][:ARM_DOF]]
    for j, (v, cap) in enumerate(zip(limit, tau_max)):
        if v > float(cap):
            raise ValueError(
                f"out_max[{j}] = {v:.6g} N m is past the joint's tau_max {cap} N m: a residual "
                "that large is a model gone wrong, not a correction"
            )
    layers = []
    for k, layer in enumerate(model.linears):
        w = layer.weight.detach().to(torch.float64).numpy()
        b = layer.bias.detach().to(torch.float64).numpy()
        if k == len(model.linears) - 1:  # the SDK has no output mean: fold it into the bias
            b = b + s["y_mean"] / s["y_std"]
        layers.append([w.tolist(), b.tolist()])
    if len(layers[0][0][0]) != width:
        raise ValueError(
            f"the model takes {len(layers[0][0][0])} inputs, history {history} gives {width}"
        )
    return {
        "history": int(history),
        "layers": layers,
        "x_mean": s["x_mean"].tolist(),
        "x_std": s["x_std"].tolist(),
        "y_scale": s["y_std"].tolist(),
        "out_max": [float(v) for v in limit],
    }


def export(
    model: ResidualMlp,
    stats: dict,
    history: int,
    out_max: Sequence[float] | None,
    path: Path,
) -> dict:
    """Write ``residual_spec`` to ``path`` as JSON, which must not exist yet: a model file is
    never overwritten. Returns the dict."""
    spec = residual_spec(model, stats, history, out_max)
    with open(path, "x") as f:
        f.write(spec_json(spec))
    return spec


def spec_json(spec: dict) -> str:
    """The exact text ``export`` writes for ``spec``: its sha256 is the model's
    (``scripts/real_residual_train.py``'s report, ruling R35)."""
    return json.dumps(spec) + "\n"
