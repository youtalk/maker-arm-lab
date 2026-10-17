"""Excitation trajectories for the identification runs (MS5 control plan task 7).

Each trajectory is a finite Fourier series per joint (Swevers et al. 1997) at the fundamental
``wf_hz``, ``N`` harmonics:

    q_j(t) = q0_j + sum_l [a_jl / (w l) sin(w l t) - b_jl / (w l) (cos(w l t) - 1)]

with ``w = 2 pi wf_hz``. The ``- 1`` is the series' offset chosen so that ``q(0) = q0``. The
trajectory starts and ends at rest: ``sum_l a_jl = 0`` makes the velocity zero at t = 0 and
``sum_l l b_jl = 0`` the acceleration, and a duration of a whole number of periods repeats
both at the end, where ``q`` is back at ``q0``. This is the constrained Swevers form rather
than a window: a window would change the spectrum the coefficients were drawn for. A session
walks the arm to row 0 with a minimum-jerk approach that arrives at rest (``real.session``),
so neither end steps the velocity. ``rest_coefficients`` projects drawn coefficients onto the
two constraints; ``fourier_trajectory`` refuses coefficients that miss them.

``design_excitation`` draws the coefficients and scales each joint into its soft-limit window
(less ``margin_rad``, and at most ``amplitude_rad`` from the center) and under ``v_scale`` of
the profile's velocity cap, so that every trajectory passes ``real.session.check_traj``.

numpy and ``maker_arm_lab.core`` only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from maker_arm_lab.core import dynamics_model
from maker_arm_lab.core import profile as _profile

DT = 0.005
"""200 Hz, the session's row period (``real.session.DT``)."""
ARM_DOF = 6
EXCITE_CENTER = (0.0, -1.6, 2.2, 0.4, 0.0, 0.0)  # INITIAL: session 3
"""The pose the excitations swing about, in URDF joint coordinates: the upper arm near
vertical, the forearm folded forward. Swung by ``EXCITE_AMPLITUDE_RAD`` on every joint, every
moving link origin stays at least 0.15 m above the desk on the bench plant (the bench's reach
pose, by contrast, already holds the fingertip at desk height)."""
EXCITE_AMPLITUDE_RAD = 0.5  # INITIAL: session 3
"""The farthest any joint swings from the center."""
REST_TOL = 1e-9
FILL = 1.0 - 1e-6
"""A joint fills its window or its velocity cap to this fraction, so rounding never crosses one."""


def rest_coefficients(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The nearest (least squares, per joint) coefficients that start and end at rest:
    ``a`` loses its mean over the harmonics, ``b`` its component along ``l``."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    harm = np.arange(1, a.shape[1] + 1, dtype=np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= np.outer(b @ harm / (harm @ harm), harm)
    return a, b


def fourier_trajectory(
    q0: Sequence[float],
    a: np.ndarray,
    b: np.ndarray,
    wf_hz: float,
    duration_s: float,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``t`` (n,) from 0 to ``duration_s`` inclusive, and ``q``, ``dq``, ``ddq`` (n, 6). ``a``
    and ``b`` are (6, N). Raises ValueError unless the coefficients rest at both ends and
    ``duration_s`` is a whole number of periods."""
    q0 = np.asarray(q0, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if q0.shape != (ARM_DOF,) or a.ndim != 2 or a.shape[0] != ARM_DOF or a.shape != b.shape:
        raise ValueError(f"need q0 (6,) and a, b (6, N), got {q0.shape}, {a.shape}, {b.shape}")
    if not (wf_hz > 0.0 and duration_s > 0.0 and dt > 0.0):
        raise ValueError("wf_hz, duration_s and dt must be positive")
    harm = np.arange(1, a.shape[1] + 1, dtype=np.float64)
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1.0)
    if np.abs(a.sum(axis=1)).max() > REST_TOL * scale or np.abs(b @ harm).max() > (
        REST_TOL * scale * harm.sum()
    ):
        raise ValueError(
            "the coefficients do not start at rest: need sum_l a_jl = 0 and sum_l l b_jl = 0 "
            "(rest_coefficients)"
        )
    periods = duration_s * wf_hz
    if abs(periods - round(periods)) > 1e-9 or round(periods) < 1:
        raise ValueError(
            f"duration_s {duration_s} is not a whole number of periods of {wf_hz} Hz: the "
            "trajectory would not end at rest"
        )
    n = round(duration_s / dt) + 1
    t = np.arange(n, dtype=np.float64) * dt
    w = 2.0 * math.pi * wf_hz
    wl = w * harm  # (N,)
    s, c = np.sin(np.outer(t, wl)), np.cos(np.outer(t, wl))  # (n, N)
    q = q0 + s @ (a / wl).T - (c - 1.0) @ (b / wl).T
    dq = c @ a.T + s @ b.T
    ddq = -s @ (a * wl).T + c @ (b * wl).T
    return t, q, dq, ddq


def _windows(profile: dict, q_center, margin_rad: float, amplitude_rad) -> np.ndarray:
    """(6,) half widths: how far each joint may swing either side of the center."""
    out = []
    for j, joint in enumerate(profile["joints"][:ARM_DOF]):
        lo, hi = float(joint["q_lo"]) + margin_rad, float(joint["q_hi"]) - margin_rad
        c = float(q_center[j])
        half = min(c - lo, hi - c)
        if amplitude_rad is not None:
            half = min(half, float(amplitude_rad))
        if not half > 0.0:
            raise ValueError(
                f"{joint['name']}: the center {c:.4f} rad is not inside [{lo:.4f}, {hi:.4f}] "
                f"(the soft limits less {margin_rad} rad)"
            )
        out.append(half)
    return np.asarray(out)


def design_excitation(
    rng: np.random.Generator,
    profile: dict,
    q_center: Sequence[float],
    count: int = 10,
    wf_hz: float = 0.1,
    harmonics: int = 5,
    margin_rad: float = 0.1,
    v_scale: float = 0.5,
    *,
    duration_s: float = 20.0,
    amplitude_rad: float | None = EXCITE_AMPLITUDE_RAD,
) -> list[dict]:
    """``count`` excitations about ``q_center``, each a dict of ``t``, ``q``, ``dq``, ``ddq``,
    ``grip`` (the gripper held at the model's lumped opening), ``a``, ``b`` and ``q0``. Every
    joint is scaled to fill the smaller of its window (``_windows``) and ``v_scale`` of the
    profile's ``max_velocity``; the swing is centred on ``q_center``. ``amplitude_rad`` None
    swings to the soft limits less ``margin_rad``."""
    if count < 1 or harmonics < 2:
        raise ValueError("need count >= 1 and harmonics >= 2 (one harmonic cannot rest)")
    if not 0.0 < v_scale <= 1.0:
        raise ValueError(f"v_scale must be in (0, 1], got {v_scale}")
    q_center = np.asarray(q_center, dtype=np.float64)
    half = _windows(profile, q_center, margin_rad, amplitude_rad)
    v_cap = v_scale * float(profile["max_velocity"])
    grip = _profile.gripper_angle(profile, dynamics_model.JAW_FRACTION)
    zero = np.zeros(ARM_DOF)
    out = []
    for _ in range(count):
        a, b = rest_coefficients(
            rng.normal(0.0, 1.0, (ARM_DOF, harmonics)), rng.normal(0.0, 1.0, (ARM_DOF, harmonics))
        )
        _, f, df, _ = fourier_trajectory(zero, a, b, wf_hz, duration_s)
        hi, lo = f.max(axis=0), f.min(axis=0)
        k = FILL * np.minimum(half / (0.5 * (hi - lo)), v_cap / np.abs(df).max(axis=0))
        a, b = a * k[:, None], b * k[:, None]
        q0 = q_center - k * 0.5 * (hi + lo)
        t, q, dq, ddq = fourier_trajectory(q0, a, b, wf_hz, duration_s)
        out.append(
            {
                "t": t,
                "q": q,
                "dq": dq,
                "ddq": ddq,
                "grip": np.full(t.shape, grip),
                "a": a,
                "b": b,
                "q0": q0,
                "wf_hz": float(wf_hz),
            }
        )
    return out
