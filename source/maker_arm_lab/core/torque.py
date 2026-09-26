"""The MS7d pick-point force: what the arm's own motors say about a push on the fingers.

MS7d design, L6 and L7, after the QDD contact spike (2026-09-25). The Maker Arm's
quasi-direct-drive motors report a torque estimate from their current. A real host subtracts
what it models, gravity and the dynamics of the commanded motion, and keeps a residual that a
push on the arm moves. Torch and stdlib only.

The truth input is the constraint torque on the six arm joints, which the spike found equal to
the contact torques J^T f. The MuJoCo env's ``arm_torques`` member gives it, with the gravity
torque, the actuator torque and the pick point's translational Jacobian; it stands for the
motors' own reports and the arm's FK. The measured residual is that constraint torque, plus a
model error ``k (a_j G_j + b_j tau_dyn_j)`` with ``tau_dyn`` the actuator plus the constraint
torque and ``a``, ``b`` drawn per joint per episode from N(0, 1), plus white noise per joint per
step. ``k`` is ``TORQUE_MODEL_K`` times the ``torque_k`` knob, the noise sigma
``TORQUE_NOISE_SIGMA_NM`` times the ``torque`` knob.

The force at the pick point is the least-squares image of the tared residual through the
Jacobian (``pick_force``), and its norm is the force the levers read. ``ForceSensor`` keeps two
tares per env. The rolling tare is the spike's, the median residual over steps t-15 to t-6, so a
steady press fades into it within about ten steps. The snap tare is the median residual over the
10 steps before each entry into the snap window (``loop.snap_window``), taken in free motion, so
a steady press on the final approach stays visible. Inside the window the force reads against
the snap tare of the last entry, elsewhere against the rolling tare; before a tare has its
history, the force is 0.

The generator is seeded like the ``Perceiver``'s, ``torque:{run_seed}:{env_index}``, and its
draws are paired across the knobs: 12 at the episode's reset, 6 per step, whatever the scales.
"""

from __future__ import annotations

import random
from collections import deque

import torch

TORQUE_NOISE_SIGMA_NM = 0.05  # INITIAL: MS5a measures it on the real arm
"""Nm: the white noise per joint per step on the measured torque at a ``torque`` scale of 1, the
QDD spike's point for a good host. The spike detects a 5 N push at 0.99 up to 0.2 Nm."""
TORQUE_MODEL_K = 0.05  # INITIAL: MS5a measures it on the real arm
"""The relative model error of the gravity and dynamics torques at a ``torque_k`` scale of 1,
the spike's point for the same host."""
HOLD_FORCE_MAX_N = 5.0  # INITIAL: the hold calibration of MS7d design section 7.2 reads it
"""N: ``hold_force`` (L6) lets the hold detector enter a hold only on a step whose pick-point
force is under this. In the spike's traces the false holds pressed a finger on a static surface
at 12 N median, the true holds 0.6 N. The calibration may raise it to 8 N, then to 12 N."""
FORCE_CAP_N = 15.0  # INITIAL: the MS7d isolation row R4 reads it, MS5a sets it on the real arm
"""N: ``force_cap`` (L7) holds the arm at its measured joint positions on a step whose pick-point
force is over this. The policy pressed its fingers into static surfaces with up to 45 N (the
spike's 90th percentile of those steps); the teacher touched them in 0.27 % of its approach."""
TARE_STEPS = 10
"""Residuals in the median of either tare, the spike's."""
ROLLING_TARE_LAG = 15
"""The rolling tare is the median of the ``TARE_STEPS`` oldest of this many steps before this
one: steps t-15 to t-6, the spike's."""


def pick_force(jac, residual) -> torch.Tensor:
    """(3,) N: the force at the pick point that best explains the joint torque ``residual``
    (6,) Nm, the least-squares solution of ``jac.T f = residual`` with ``jac`` the pick point's
    (3, 6) translational Jacobian. Pure."""
    jac = torch.as_tensor(jac, dtype=torch.float64)
    residual = torch.as_tensor(residual, dtype=torch.float64)
    return torch.linalg.lstsq(jac.T, residual.unsqueeze(-1)).solution.squeeze(-1)


def _median(rows: list[torch.Tensor]) -> torch.Tensor:
    """The per-joint median of (6,) rows, the mean of the middle two on an even count."""
    return torch.stack(rows).quantile(0.5, dim=0)


class ForceSensor:
    """One env's pick-point force. ``update`` once per control step; ``force`` is the last
    step's, in N."""

    def __init__(self, run_seed: int, env_index: int, scales: dict[str, float] | None = None):
        scales = scales or {}
        self.k = TORQUE_MODEL_K * float(scales.get("torque_k", 1.0))
        self.sigma = TORQUE_NOISE_SIGMA_NM * float(scales.get("torque", 1.0))
        self.rng = random.Random(f"torque:{run_seed}:{env_index}")
        self.reset_episode()

    def _gauss(self, n: int) -> torch.Tensor:
        return torch.tensor([self.rng.gauss(0.0, 1.0) for _ in range(n)], dtype=torch.float64)

    def reset_episode(self) -> None:
        """Draw the episode's model error scales, ``a`` then ``b``, and forget the history."""
        self.a, self.b = self._gauss(6), self._gauss(6)
        self.history: deque = deque(maxlen=ROLLING_TARE_LAG)  # the residuals before this step
        self.snap_tare: torch.Tensor | None = None  # the tare of the last window entry
        self.inside = False  # the last step was inside the snap window
        self.force = 0.0

    def residual(self, constraint, gravity, actuator) -> torch.Tensor:
        """(6,) Nm: the measured residual of one step, the arm's (6,) constraint, gravity and
        actuator torques in. Draws the step's six noise values."""
        noise = self._gauss(6)
        constraint = torch.as_tensor(constraint, dtype=torch.float64)
        gravity = torch.as_tensor(gravity, dtype=torch.float64)
        dynamics = torch.as_tensor(actuator, dtype=torch.float64) + constraint
        return constraint + self.k * (self.a * gravity + self.b * dynamics) + self.sigma * noise

    def update(self, constraint, gravity, actuator, jac, window: bool) -> float:
        """One step: the ``arm_torques`` read and whether the pick point is in the snap window.
        Returns the force (N) against the snap tare inside the window, else the rolling tare."""
        r = self.residual(constraint, gravity, actuator)
        past = list(self.history)
        if window and not self.inside:
            self.snap_tare = _median(past[-TARE_STEPS:]) if len(past) >= TARE_STEPS else None
        self.inside = window
        if window:
            tare = self.snap_tare
        else:
            full = len(past) == ROLLING_TARE_LAG
            tare = _median(past[:TARE_STEPS]) if full else None
        self.force = 0.0 if tare is None else float(pick_force(jac, r - tare).norm())
        self.history.append(r)
        return self.force


def force_summary(forces: list[float]) -> dict:
    """The approach windows' force read: the steps, and the median, 90th percentile and
    maximum in N (None with no step), rounded to 0.01 N. Pure."""
    if not forces:
        return {"steps": 0, "p50": None, "p90": None, "max": None}
    f = torch.tensor(forces, dtype=torch.float64)
    p50, p90 = (round(float(f.quantile(q)), 2) for q in (0.5, 0.9))
    return {"steps": len(forces), "p50": p50, "p90": p90, "max": round(float(f.max()), 2)}
