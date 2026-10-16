"""The identification fit (MS5 control plan task 7): the arm's rigid-body parameters, and per
joint its Coulomb, viscous and Stribeck friction and its armature, from session logs.

The model is ``tau = Y(q, dq, ddq) p + F(dq) + armature * ddq``. ``maker_arm_rs.Dynamics``'s
``inverse`` is the rigid-body part plus the armature term, friction not; ``friction(dq)`` is
``F(dq) = (fc + (fs - fc) exp(-(dq / vs)^2)) tanh(dq / eps) + fv dq``. The rigid-body part is
exactly linear in ten parameters per body (``linear_params``: the mass, the first moment ``m c``
and the inertia about the joint frame's origin), so ``regressor`` reads ``Y`` off the SDK's own
inverse dynamics, one model per parameter, and the fit can never disagree with the model the
controller runs. For a fixed Stribeck velocity ``vs`` and tanh width ``eps`` the friction and
the armature are linear too: per joint the columns ``tanh(dq / eps)`` (fc), ``dq`` (fv),
``exp(-(dq / vs)^2) tanh(dq / eps)`` (fs - fc) and ``ddq`` (armature). ``vs`` comes from
``STRIBECK_GRID`` and ``eps`` from ``EPS_GRID``, the pair with the lowest residual.

The solve is ridge least squares toward a prior that starts at the nominal (URDF)
parameters: a rigid-body parameter moved by its own scale costs ``lam`` times the summed
squared torque, so a direction the data cannot see (a body's inertia about an axis its joint
never turns, or joint 1's armature against link_002's inertia about the same axis) stays at
the prior and the friction and armature columns, which carry almost no ridge, take what the
data asks for. A fixed nominal prior goes wrong when the arm differs from it: the data moves
``m`` and ``m c`` while the ridge holds the inertia about the joint origin, and the inertia
about the center of mass that results is no body's (the first review's repro: truth 1.25x the
URDF masses read unphysical, and the ridge growth that followed put the error into j2's
armature, 8x). So the prior is moved to the solution and the solve repeated a fixed
``PRIOR_PASSES`` times (``consistent_prior``): the prior takes the solution's ``m`` and ``m c`` and the nominal
inertia shape about the center of mass, scaled to the solution's inertia, so the directions
the data does not see follow a physical body. Friction and armature are non-negative (and
``fs >= fc``): a parameter the solve drives negative is held at zero and the rest solved
again. A body that still comes out unphysical gets its own ridge grown by 10 (the others
keep theirs), at most ``MAX_GROWTHS`` times.

Only free-motion samples enter: the controller's gate is 1 (``trk/gate``, joined on ``t ==
trk/tick_t``), the command was not clamped, and the tick is outside every prompt segment.
The acceleration is the centred difference of the velocity after a ``SMOOTH_TICKS`` moving
average; a sample whose window crosses a gap in the log is dropped. The torque is the
torque that moved the arm through each held command, paired with the state averaged over
that hold (``hold_model``).

torch, numpy, h5py, ``maker_arm_rs`` and ``maker_arm_lab.core`` only (and the stdlib).
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import maker_arm_rs
import numpy as np

from maker_arm_lab.core import dynamics_model

ARM_DOF = 6
PARAMS_PER_BODY = 10
N_RIGID = ARM_DOF * PARAMS_PER_BODY
FRICTION_COLUMNS = 4
"""Per joint: Coulomb, viscous, Stribeck static excess (fs - fc), armature."""
DELTA = 1e-4
"""The regressor's step, as a fraction of each parameter's scale."""
K_DEADBAND = 4.0  # INITIAL: written in docs/ms5-foundations.md before session 5
"""The admittance deadband as a multiple of the observer's free-motion residual RMS."""
STRIBECK_GRID = (0.02, 0.05, 0.1)
"""rad/s: the Stribeck velocities the fit tries."""
EPS_GRID = (0.005, 0.01, 0.02, 0.05)  # INITIAL: session 3
"""rad/s: the tanh widths of the Coulomb term the fit tries, with each Stribeck velocity. The
law can only rise toward zero velocity (``static_ >= coulomb``); a dry friction that eases in
over the first 0.1 rad/s instead (the bench's MuJoCo friction loss reaches 70 to 80 % of its
level by 0.01 rad/s) needs a wider tanh, or the fit reads the ease-in as viscous friction."""
HOLD_STEPS: int | None = 0  # INITIAL: session 3
"""How the motor runs its PD law through a held command, for ``hold_model``: 0 continuously
(the real motor's fast inner loop), ``n`` in n integration steps (the bench: 5), None not at
all (a log whose torque is the instantaneous torque at its own sample, as a synthetic one).
The assumption under it, that the reply's torque is the one the motor applies at the start of
the hold that follows it, is unverified on the real arm. Session 3 checks it with a
constant-velocity sweep of one joint at two ``kp`` caps (ddq = 0, so the hold correction is
``kp v T / 2`` and nothing else depends on ``kp``): with the right ``HOLD_STEPS`` the fitted
viscous term is the same at both caps; with ``None`` it moves by ``(kp_1 - kp_2) T / 2``.
``scripts/real_ident.py`` takes ``--hold_steps`` with no default and reports the fit under
both ``none`` and the chosen value."""
SMOOTH_TICKS = 9
MAX_GROWTHS = 5
RIDGE_GROWTH = 10.0
PRIOR_MIN_RATIO = 0.1
"""The smallest scale of the nominal inertia the prior pulls toward."""
PRIOR_PASSES = 6  # INITIAL: session 3
"""How many solves the fit makes, the first toward the nominal parameters and each later one
toward ``consistent_prior`` of the one before. The count is part of the regularization, not a
convergence cap: every pass moves the anchor to the data, so more passes take more of the
URDF's bias out of the directions the data sees (on noise-free 1.25x masses j2's armature
reads 0.0406, 0.0326, 0.0333, 0.0330, 0.0327 for 1, 2, 3, 4, 6 passes, j5's 0.0137, 0.0121,
0.0113, 0.0110, 0.0108) and let the weakly seen directions drift further with the noise (the
first review round's 30 to 1000 passes moved link_004's mass by 15 %). From six to eight
passes the noisy 1.25x read moves by under 3 % (a test pins it); session 3 reads the same
table on the arm's data."""
PRIOR_TOL = 1e-2
"""``report["prior_converged"]``: one more pass would move the prior by less than this
fraction of every parameter's scale. A diagnostic only; the fit always makes
``PRIOR_PASSES`` passes."""
FRICTION_RIDGE = 1e-9
"""The friction and armature columns' ridge, relative to each column's own squared norm: only
enough to keep a column the data never excites (a joint that never moved) at zero."""
COND_TOL = 1e-3
"""A unit-norm column whose part outside the columns already kept is shorter than this is a
combination of them, not a base parameter (``condition_number``). The structural
combinations read about 1e-8 (the regressor's rounding); the first moments of link_002 and
link_003 along their shared direction read about 1e-4 on every design tried (a near tie the
ridge settles, not the data), so the cut sits above both, and a kept column can at most reach
a condition number of about 1e3."""

Body = dynamics_model.Body
Factory = Callable[[Sequence[Body]], maker_arm_rs.Dynamics]


class IdentError(Exception):
    """The fit cannot produce a model: no samples, or no physical bodies."""


# --- the parameters -----------------------------------------------------------------------


def _skew_term(m: float, c: np.ndarray) -> np.ndarray:
    """m (|c|^2 E - c c^T): the parallel-axis shift from the center of mass to the origin."""
    return m * (float(c @ c) * np.eye(3) - np.outer(c, c))


def linear_params(bodies: Sequence[Body]) -> np.ndarray:
    """(60,): per body ``m, m cx, m cy, m cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz``, the inertia
    about the joint frame's origin."""
    out = []
    for m, c, inertia in bodies:
        c = np.asarray(c, dtype=np.float64)
        i_o = np.asarray(inertia, dtype=np.float64) + _skew_term(m, c)
        out += [m, *(m * c), i_o[0, 0], i_o[0, 1], i_o[0, 2], i_o[1, 1], i_o[1, 2], i_o[2, 2]]
    return np.asarray(out, dtype=np.float64)


def bodies_from_params(p: np.ndarray) -> list[Body]:
    """``linear_params``' inverse: ``(mass, com, inertia about the com)`` per body."""
    p = np.asarray(p, dtype=np.float64).reshape(ARM_DOF, PARAMS_PER_BODY)
    out = []
    for m, mx, my, mz, ixx, ixy, ixz, iyy, iyz, izz in p:
        c = np.array([mx, my, mz]) / m
        i_o = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        i_c = i_o - _skew_term(m, c)
        out.append(dynamics_model._freeze(m, c, i_c))
    return out


def physical(bodies: Sequence[Body]) -> bool:
    """Positive masses, and inertias about the center of mass that are symmetric positive
    definite and meet the triangle inequality."""
    for m, _, inertia in bodies:
        i = np.asarray(inertia, dtype=np.float64)
        if not (np.isfinite(m) and m > 0.0 and np.isfinite(i).all()):
            return False
        if not np.allclose(i, i.T, rtol=0.0, atol=1e-12 * max(np.abs(i).max(), 1e-12)):
            return False
        ev = np.linalg.eigvalsh(0.5 * (i + i.T))
        if ev[0] <= 0.0 or ev[2] > ev[0] + ev[1]:
            return False
    return True


def param_scale(p: np.ndarray) -> np.ndarray:
    """(60,) per-parameter scales: the mass, the mass times the larger of |c| and 1 cm for
    the first moment, the mean diagonal inertia about the origin for the six inertias."""
    p = np.asarray(p, dtype=np.float64).reshape(ARM_DOF, PARAMS_PER_BODY)
    out = np.empty_like(p)
    for b, row in enumerate(p):
        m = abs(row[0])
        lever = max(float(np.linalg.norm(row[1:4])) / m, 0.01)
        out[b, 0] = m
        out[b, 1:4] = m * lever
        out[b, 4:] = max((row[4] + row[7] + row[9]) / 3.0, 1e-9)
    return out.reshape(-1)


def default_factory(bodies: Sequence[Body]) -> maker_arm_rs.Dynamics:
    return dynamics_model.dynamics(bodies)


def regressor(dyn_factory: Factory, p0: np.ndarray, q, dq, ddq) -> np.ndarray:
    """(6, 60) per sample (or (N, 6, 60) for (N, 6) inputs): column k is the change of
    ``inverse`` per unit of parameter k, read between the model at ``p0`` and one at ``p0 +
    delta e_k``, ``delta = DELTA`` times the parameter's scale, so every body stays physical.
    Inverse dynamics is linear in the parameters, so the difference is exact up to rounding.
    ``dyn_factory(bodies)`` builds a model without friction or armature."""
    p0 = np.asarray(p0, dtype=np.float64)
    q, dq, ddq = (np.asarray(x, dtype=np.float64) for x in (q, dq, ddq))
    single = q.ndim == 1
    q, dq, ddq = (np.atleast_2d(x) for x in (q, dq, ddq))
    delta = DELTA * param_scale(p0)
    base = dyn_factory(bodies_from_params(p0))
    moved = []
    for k in range(p0.size):
        p = p0.copy()
        p[k] += delta[k]
        moved.append(dyn_factory(bodies_from_params(p)))
    out = np.empty((q.shape[0], ARM_DOF, p0.size))
    for n, (qn, dqn, ddqn) in enumerate(zip(q.tolist(), dq.tolist(), ddq.tolist())):
        tau0 = np.asarray(base.inverse(qn, dqn, ddqn))
        for k, dyn in enumerate(moved):
            out[n, :, k] = (np.asarray(dyn.inverse(qn, dqn, ddqn)) - tau0) / delta[k]
    return out[0] if single else out


def friction_columns(
    dq: np.ndarray, ddq: np.ndarray, stribeck_v: float, eps: float = EPS_GRID[1]
) -> np.ndarray:
    """(N, 6, 24): per joint j, row j carries columns 4 j .. 4 j + 3, Coulomb, viscous,
    Stribeck excess and armature."""
    dq, ddq = np.atleast_2d(dq), np.atleast_2d(ddq)
    out = np.zeros((dq.shape[0], ARM_DOF, ARM_DOF * FRICTION_COLUMNS))
    sign = np.tanh(dq / eps)
    for j in range(ARM_DOF):
        c = FRICTION_COLUMNS * j
        out[:, j, c] = sign[:, j]
        out[:, j, c + 1] = dq[:, j]
        out[:, j, c + 2] = np.exp(-((dq[:, j] / stribeck_v) ** 2)) * sign[:, j]
        out[:, j, c + 3] = ddq[:, j]
    return out


RIGID_NAMES = ("m", "mcx", "mcy", "mcz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz")
FRICTION_NAMES = ("coulomb", "viscous", "stribeck", "armature")
COLUMN_NAMES = tuple(
    [f"{link}.{n}" for link in dynamics_model.ARM_LINKS for n in RIGID_NAMES]
    + [f"j{j + 1}.{n}" for j in range(ARM_DOF) for n in FRICTION_NAMES]
)
"""The fit's 84 columns, in order."""


def condition_number(a: np.ndarray, tol: float = COND_TOL) -> tuple[float, int, list]:
    """The condition number of the base columns of ``a`` (M, K), each scaled to unit norm,
    how many there are, and the columns left out as ``(index, residual)``, largest residual
    first. The base columns are picked greedily (Gram-Schmidt with column pivoting); a column
    within ``tol`` of the span of those kept is a combination of them, as some rigid-body
    parameters always are, and is left out (a column the data never reaches has residual 0).
    A residual well above the rounding level (1e-8) is a weakly excited direction that the
    reported condition number does not show."""
    norms = np.linalg.norm(a, axis=0)
    keep = np.flatnonzero(norms > 1e-9 * norms.max())  # a column the data never reaches
    an = a[:, keep] / norms[keep]
    r = np.linalg.qr(an, mode="r")
    x = r.copy()
    sel: list[int] = []
    res = np.linalg.norm(x, axis=0)
    for _ in range(x.shape[1]):
        res = np.linalg.norm(x, axis=0)
        res[sel] = 0.0
        k = int(np.argmax(res))
        if res[k] < tol:
            break
        sel.append(k)
        u = x[:, k] / res[k]
        x -= np.outer(u, u @ x)
    res = np.linalg.norm(x, axis=0)
    dropped = [(int(keep[i]), float(res[i])) for i in range(keep.size) if i not in sel]
    dropped += [(int(i), 0.0) for i in range(a.shape[1]) if i not in set(keep.tolist())]
    dropped.sort(key=lambda d: -d[1])
    return float(np.linalg.cond(r[:, sel])), len(sel), dropped


def excitation_condition(
    trajs: Sequence[dict],
    nominal_bodies: Sequence[Body],
    stride: int = 10,
    stribeck_v: float = STRIBECK_GRID[1],
    eps: float = EPS_GRID[1],
    dyn_factory: Factory = default_factory,
) -> tuple[float, int, list]:
    """``condition_number`` of the regressor the fit would stack from ``trajs`` (each with
    ``q``, ``dq`` and ``ddq``), every ``stride``-th sample, the friction columns at one point
    of the grids. Returns ``condition_number``'s triple."""
    p0 = linear_params(nominal_bodies)
    q, dq, ddq = (np.concatenate([tr[k][::stride] for tr in trajs]) for k in ("q", "dq", "ddq"))
    yr = regressor(dyn_factory, p0, q, dq, ddq).reshape(-1, N_RIGID)
    fr = friction_columns(dq, ddq, stribeck_v, eps).reshape(-1, ARM_DOF * FRICTION_COLUMNS)
    return condition_number(np.concatenate([yr, fr], axis=1))


# --- the logs -----------------------------------------------------------------------------


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def hold_model(t, q, dq, ddq, tau, kp, kd, hold_steps: int | None):
    """The mean torque over each held command and the mean state it moved the arm through.

    A command is held for its tick while the motor runs ``kp (pos - q) + kd (vel - dq) +
    tau_ff`` against it, so the torque changes through the tick by ``-kp (q - q_k) - kd (dq -
    dq_k)``, and the reply samples it at the tick's start. Uncorrected, the fit reads ``kp``
    times about half a tick as viscous friction and ``kd`` times it as armature, plus a term
    that depends on how fast the joint answers inside the tick. With the next sample's ``q``
    the mean over the hold is exact in the velocity: ``mean(dq) = (q_k+1 - q_k) / T``
    (continuous) or that plus ``(dq_k - dq_k+1) / n`` (n steps of semi-implicit Euler, the
    states before each step), and ``mean(q) - q_k`` is ``(n - 1) / 2n`` of the step (1/2
    continuous). The returned rows k = 0 .. N - 2 are the hold after sample k: the mean
    state, the acceleration midway, the torque less its change. ``hold_steps`` None returns
    the samples as they are (``HOLD_STEPS``)."""
    if hold_steps is None:
        return q, dq, ddq, tau
    step = q[1:] - q[:-1]
    period = (t[1:] - t[:-1])[:, None]
    frac = 0.5 if hold_steps == 0 else (hold_steps - 1) / (2.0 * hold_steps)
    v_mean = step / period
    if hold_steps:
        v_mean = v_mean + (dq[:-1] - dq[1:]) / hold_steps
    q_mean = q[:-1] + frac * step
    drive = tau[:-1] - kp[:-1] * (q_mean - q[:-1]) - kd[:-1] * (v_mean - dq[:-1])
    a_mean = 0.5 * (ddq[:-1] + ddq[1:])
    return q_mean, v_mean, a_mean, drive


def load_free_motion(path: Path, hold_steps: int | None = HOLD_STEPS) -> dict:
    """The free-motion samples of one session log: ``q``, ``dq``, ``ddq``, ``tau`` (N, 6) and
    the counts behind them, through ``hold_model``."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        t = f["t"][:]
        q, dq, tau = (f[k][:, :ARM_DOF].astype(np.float64) for k in ("q", "dq", "tau"))
        kp, kd = (f[k][:, :ARM_DOF].astype(np.float64) for k in ("cmd_kp", "cmd_kd"))
        clamped = np.asarray(f["clamped"][:], dtype=bool)
        trk_t = f["trk/tick_t"][:]
        gate = f["trk/gate"][:]
        segments = f["segments"][:]
        attrs = {k: _decode(f.attrs[k]) for k in ("lab_revision", "sdk_revision") if k in f.attrs}
    if clamped.ndim > 1:
        clamped = clamped.any(axis=1)
    n = t.shape[0]
    free = np.zeros(n, dtype=bool)
    if trk_t.shape[0]:
        # join on t == tick_t, not on the index: a fault-hold tick has no tracking row
        order = np.argsort(trk_t, kind="stable")
        trk_t, gate = trk_t[order], gate[order]
        idx = np.minimum(np.searchsorted(trk_t, t), trk_t.shape[0] - 1)
        free = (trk_t[idx] == t) & (gate[idx] == 1.0) & ~clamped
    for seg in segments:
        if _decode(seg["kind"]) == "prompt":
            free &= ~((t >= seg["start_t"]) & (t <= seg["end_t"]))
    # The acceleration: a centred moving average of dq, then a centred difference.
    half = SMOOTH_TICKS // 2
    valid = np.zeros(n, dtype=bool)
    ddq = np.zeros_like(dq)
    if n > SMOOTH_TICKS + 2:
        step = np.diff(t)
        dt = float(np.median(step))
        regular = np.abs(step - dt) < 0.5 * dt
        kernel = np.ones(SMOOTH_TICKS) / SMOOTH_TICKS
        smooth = np.stack([np.convolve(dq[:, j], kernel, mode="same") for j in range(ARM_DOF)], 1)
        ddq[1:-1] = (smooth[2:] - smooth[:-2]) / (t[2:] - t[:-2])[:, None]
        # sample i reads dq over i - half - 1 .. i + half + 1: every step in that span regular
        span = SMOOTH_TICKS + 1
        bad = np.concatenate([[0], np.cumsum(~regular)])
        ok = (bad[span:] - bad[:-span]) == 0  # steps i - half - 1 .. i + half, for i = half + 1..
        valid[half + 1 : half + 1 + ok.size] = ok
    q, dq, ddq, tau = hold_model(t, q, dq, ddq, tau, kp, kd, hold_steps)
    use = free & valid
    if hold_steps is not None:  # row k is the hold from sample k to sample k + 1
        use = use[:-1] & valid[1:]
    return {
        "q": q[use],
        "dq": dq[use],
        "ddq": ddq[use],
        "tau": tau[use],
        "ticks": int(n),
        "free": int(free.sum()),
        "samples": int(use.sum()),
        "attrs": attrs,
    }


# --- the fit ------------------------------------------------------------------------------


@dataclass
class IdentResult:
    bodies: list
    friction: list
    """Six (coulomb, viscous, eps, static_, stribeck_v) tuples."""
    armature: list
    torque_sigma: list
    """Per joint, N m: the RMS of the fit's residual on its own samples."""
    report: dict = field(default_factory=dict)

    def dynamics(self) -> maker_arm_rs.Dynamics:
        return dynamics_model.dynamics(self.bodies, self.friction, armature=self.armature)

    def to_json(self, path: Path, provenance: dict | None = None) -> None:
        """Write the file ``core.dynamics_model.from_ident`` reads. ``provenance`` adds to the
        recorded one (the logs, their revisions, the SDK revision, the date)."""
        prov = dict(self.report.get("provenance", {}))
        prov.update(provenance or {})
        data = {
            "bodies": [[m, list(c), [list(r) for r in i]] for m, c, i in self.bodies],
            "friction": [list(f) for f in self.friction],
            "armature": list(self.armature),
            "torque_sigma": list(self.torque_sigma),
            "provenance": prov,
            "report": {k: v for k, v in self.report.items() if k != "provenance"},
        }
        with open(path, "x") as f:  # an ident file is never overwritten
            f.write(json.dumps(data, indent=2) + "\n")


def sdk_revision() -> str:
    """The git commit ``maker_arm_rs`` was built from, or "unknown"."""
    try:
        raw = importlib.metadata.distribution("maker-arm-rs").read_text("direct_url.json")
        return json.loads(raw or "{}").get("vcs_info", {}).get("commit_id", "unknown")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def consistent_prior(p: np.ndarray, nominal_bodies: Sequence[Body]) -> np.ndarray:
    """(60,) the prior for the next solve, from the solution ``p``: per body its own ``m``
    and ``m c``, and the nominal inertia about the center of mass scaled to the solution's
    (the projection of the solution's inertia about ``c = (m c) / m`` onto the nominal one,
    at least ``PRIOR_MIN_RATIO``), placed at ``c``. Whatever ``m > 0`` and ``c`` the data
    chooses, the inertia the prior pulls toward is a physical body's. A body with ``m <= 0``
    gets its nominal parameters."""
    p = np.asarray(p, dtype=np.float64).reshape(ARM_DOF, PARAMS_PER_BODY)
    out = linear_params(nominal_bodies).reshape(ARM_DOF, PARAMS_PER_BODY)
    for b, (m_nom, c_nom, i_nom) in enumerate(nominal_bodies):
        m = p[b, 0]
        if not (np.isfinite(m) and m > 0.0):
            continue
        c = p[b, 1:4] / m
        i_n = np.asarray(i_nom, dtype=np.float64)
        row = p[b, 4:]
        i_fit = np.array(
            [[row[0], row[1], row[2]], [row[1], row[3], row[4]], [row[2], row[4], row[5]]]
        ) - _skew_term(m, c)
        ratio = float(np.sum(i_fit * i_n) / np.sum(i_n * i_n))
        i_o = max(ratio, PRIOR_MIN_RATIO) * i_n + _skew_term(m, c)
        out[b, 0:4] = p[b, 0:4]
        out[b, 4:] = [i_o[0, 0], i_o[0, 1], i_o[0, 2], i_o[1, 1], i_o[1, 2], i_o[2, 2]]
    return out.reshape(-1)


def _solve_consistent(ata, aty, d, theta0, constrained, nominal_bodies, scale):
    """``PRIOR_PASSES`` ridge solves, the first toward ``theta0`` and each later one toward
    ``consistent_prior`` of the solution before it. Returns the last solution, its held
    indices, and ``last_step``: how far (in fractions of ``scale``) one more pass would move
    the prior."""
    prior = theta0.copy()
    x, held = _solve(ata, aty, d, prior, constrained)
    for _ in range(PRIOR_PASSES - 1):
        prior[:N_RIGID] = consistent_prior(x[:N_RIGID], nominal_bodies)
        x, held = _solve(ata, aty, d, prior, constrained)
    step = np.abs(consistent_prior(x[:N_RIGID], nominal_bodies) - prior[:N_RIGID]) / scale
    return x, held, float(np.max(step))


def _solve(ata, aty, d, theta0, constrained):
    """Ridge least squares ``min |a x - y|^2 + sum d (x - theta0)^2`` from the normal
    equations ``ata = a^T a``, ``aty = a^T y``, with ``x >= 0`` on the ``constrained``
    indices: a constrained entry the solve drives negative is held at zero."""
    k = ata.shape[0]
    fixed = np.zeros(k, dtype=bool)
    while True:
        free = ~fixed
        lhs = ata[np.ix_(free, free)] + np.diag(d[free])
        rhs = aty[free] + d[free] * theta0[free]
        x = np.zeros(k)
        x[free] = np.linalg.solve(lhs, rhs)
        neg = [i for i in constrained if not fixed[i] and x[i] < 0.0]
        if not neg:
            return x, np.flatnonzero(fixed).tolist()
        fixed[min(neg, key=lambda i: x[i])] = True


def fit(
    logs: Sequence[Path],
    nominal_bodies: Sequence[Body],
    lam: float = 1e-3,
    *,
    dyn_factory: Factory = default_factory,
    hold_steps: int | None = HOLD_STEPS,
) -> IdentResult:
    """Fit the model to the free-motion samples of ``logs`` (see the module docstring)."""
    data = [load_free_motion(p, hold_steps) for p in logs]
    samples = sum(d["samples"] for d in data)
    if samples < ARM_DOF * (N_RIGID + ARM_DOF * FRICTION_COLUMNS):
        raise IdentError(f"{samples} free-motion samples in {len(data)} logs: too few to fit")
    q, dq, ddq, tau = (np.concatenate([d[k] for d in data]) for k in ("q", "dq", "ddq", "tau"))
    p0 = linear_params(nominal_bodies)
    yr = regressor(dyn_factory, p0, q, dq, ddq).reshape(-1, N_RIGID)
    y = tau.reshape(-1)
    n_fric = ARM_DOF * FRICTION_COLUMNS
    theta0 = np.concatenate([p0, np.zeros(n_fric)])
    constrained = list(range(N_RIGID, N_RIGID + n_fric))
    energy = float(y @ y)
    # One normal-equation system per (eps, vs) grid point; the rigid-body blocks are shared.
    rr, ry = yr.T @ yr, yr.T @ y
    systems = {}
    for eps in EPS_GRID:
        for vs in STRIBECK_GRID:
            f = friction_columns(dq, ddq, vs, eps).reshape(-1, n_fric)
            fr_ = f.T @ yr
            ata = np.block([[rr, fr_.T], [fr_, f.T @ f]])
            systems[(eps, vs)] = (ata, np.concatenate([ry, f.T @ y]))
    sse_of: dict[tuple[float, float], float] = {}
    scale2 = param_scale(p0) ** 2
    growths = np.zeros(ARM_DOF, dtype=int)
    while True:
        lam_b = lam * RIDGE_GROWTH ** np.repeat(growths, PARAMS_PER_BODY)
        best, skipped = None, []
        for key, (ata, aty) in systems.items():
            d = np.concatenate(
                [
                    lam_b * energy / scale2,
                    FRICTION_RIDGE * np.maximum(np.diag(ata)[N_RIGID:], 1e-12),
                ]
            )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    x, held, last_step = _solve_consistent(
                        ata, aty, d, theta0, constrained, nominal_bodies, param_scale(p0)
                    )
                    sse = float(energy - 2.0 * x @ aty + x @ ata @ x)
            except np.linalg.LinAlgError:
                x, sse, last_step = np.full(1, np.nan), np.nan, np.nan
            if not (np.isfinite(x).all() and np.isfinite(sse) and np.isfinite(last_step)):
                skipped.append(key)  # an overflow at this grid point: it cannot be chosen
                continue
            sse_of[key] = sse
            if best is None or sse < best[0]:
                best = (sse, key, x, held, last_step)
        if best is None:
            raise IdentError(f"no grid point gave a finite solution ({len(skipped)} skipped)")
        sse, (eps, vs), x, held, last_step = best
        bodies = bodies_from_params(x[:N_RIGID])
        bad = [b for b in range(ARM_DOF) if not physical([bodies[b]])]
        if not bad:
            break
        if (growths[bad] >= MAX_GROWTHS).any():
            raise IdentError(
                f"bodies {bad} are not physical after {MAX_GROWTHS} ridge growths "
                f"(lam {lam * RIDGE_GROWTH**MAX_GROWTHS:g})"
            )
        growths[bad] += 1
    fr = x[N_RIGID:].reshape(ARM_DOF, FRICTION_COLUMNS)
    friction = [
        (float(fc), float(fv), float(eps), float(fc + dfs), float(vs)) for fc, fv, dfs, _ in fr
    ]
    armature = [float(v) for v in fr[:, 3]]
    a = np.concatenate([yr, friction_columns(dq, ddq, vs, eps).reshape(-1, n_fric)], axis=1)
    resid = (a @ x - y).reshape(-1, ARM_DOF)
    sigma = np.sqrt(np.mean(resid**2, axis=0))
    cond, rank, dropped = condition_number(a[:: max(1, a.shape[0] // 60000)])
    report = {
        "samples": samples,
        "per_log": [
            {"log": str(p), "ticks": d["ticks"], "free": d["free"], "samples": d["samples"]}
            for p, d in zip(logs, data)
        ],
        "lam": [float(lam * RIDGE_GROWTH**g) for g in growths],
        "hold_steps": hold_steps,
        "ridge_growths": [int(g) for g in growths],
        "prior_passes": PRIOR_PASSES,
        "prior_last_step": last_step,
        "prior_converged": bool(last_step < PRIOR_TOL),
        "grid_skipped": [f"eps {k[0]} vs {k[1]}" for k in skipped],
        "eps": eps,
        "stribeck_v": vs,
        "grid_sse": {f"eps {k[0]} vs {k[1]}": v for k, v in sse_of.items()},
        "held_at_zero": [int(i - N_RIGID) for i in held],
        "cond": cond,
        "base_columns": rank,
        "dropped_columns": [[COLUMN_NAMES[i], r] for i, r in dropped],
        "provenance": {
            "logs": [str(p) for p in logs],
            "log_revisions": [d["attrs"] for d in data],
            "sdk_revision": sdk_revision(),
            "date": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    }
    return IdentResult(
        bodies=bodies,
        friction=friction,
        armature=armature,
        torque_sigma=[float(s) for s in sigma],
        report=report,
    )


def residual_rms(
    logs: Sequence[Path], dynamics: maker_arm_rs.Dynamics, hold_steps: int | None = HOLD_STEPS
) -> list[float]:
    """Per joint, N m: the RMS of the torque (``load_free_motion``) less ``inverse +
    friction`` of ``dynamics``, over the free-motion samples of ``logs``."""
    sq, n = np.zeros(ARM_DOF), 0
    for path in logs:
        d = load_free_motion(path, hold_steps)
        for q, dq, ddq, tau in zip(*(d[k].tolist() for k in ("q", "dq", "ddq", "tau"))):
            pred = np.add(dynamics.inverse(q, dq, ddq), dynamics.friction(dq))
            sq += (np.asarray(tau) - pred) ** 2
        n += d["samples"]
    if n == 0:
        raise IdentError("no free-motion samples in the logs")
    return [float(v) for v in np.sqrt(sq / n)]


def deadband(r_free: np.ndarray, multiple: float = K_DEADBAND) -> list[float]:
    """Per joint, N m: ``multiple`` times the RMS of the observer's residual ``r_free`` (rows
    of ``ARM_DOF`` joints) over free motion, the level below which the yield admittance does
    not react. Raises ValueError for an empty or non-finite ``r_free`` or a ``multiple`` that
    is not a positive number."""
    r = np.asarray(r_free, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[1] != ARM_DOF:
        raise ValueError(f"r_free needs shape (N > 0, {ARM_DOF}), got {r.shape}")
    if not np.isfinite(r).all():
        raise ValueError("r_free must be finite")
    if not (np.isfinite(multiple) and multiple > 0.0):
        raise ValueError(f"multiple must be positive, got {multiple!r}")
    return [float(v) for v in multiple * np.sqrt(np.mean(r**2, axis=0))]
