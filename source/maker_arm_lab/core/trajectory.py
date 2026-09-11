"""A leg's walk in time: the minimum-jerk law along its waypoint polyline, plus the
commanded-motion statistics the smooth-oracle probe reads.

The staircase this replaces held each joint waypoint for ``WAYPOINT_HOLD_STEPS`` control
steps and jumped to the next, so the commanded target stood still on half of all frames and
its acceleration was a train of spikes. Here one C2 time law spans the whole leg: the
minimum-jerk quintic ``s = 10 t^3 - 15 t^4 + 6 t^5``, at rest at both ends, peaking at
1.875x the mean speed mid-leg. The target at each control step is the polyline point at
fraction ``s``. The waypoints are the IK images of equally spaced task-space targets, so a
uniform fraction is uniform task-space progress.
"""

from __future__ import annotations

import array
import math

import torch

CONTROL_HZ = 30.0
"""The control rate the walk is sampled at (sim dt 1/120 with decimation 4)."""

TRANSPORT_SPEED = 4.0
"""The adopted multiplier on the transport legs' pace (HOVER, LIFT, RAISE, EXTEND, RETREAT).

Read off the pre-registered speed ladder on MuJoCo, 2026-09-10 (12 envs, 50 attempts, cameras
off, the same seeds per rung): yield 0.960 / 1.000 / 1.000 / 0.980 and 935 / 665 / 539 / 484
control steps per brick at 1x / 2x / 3x / 4x, against the staircase's 815; the rule adopted
the fastest rung with yield >= 0.95 and a cycle at most 0.85x the 1x control.
"""

MIN_JERK_PEAK_VELOCITY = 1.875
"""max s'(t) over [0, 1]: the peak speed is this times the mean speed."""

MIN_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)
"""max s''(t): the peak acceleration is this times length / duration^2."""


def min_jerk(tau: float) -> float:
    """Fraction of the leg covered at normalised time ``tau`` (clamped to [0, 1])."""
    tau = min(max(tau, 0.0), 1.0)
    return tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))


def walk_seconds(length: float, mean_speed: float, accel_per_peak_speed: float) -> float:
    """Duration of a leg of ``length`` under the law.

    ``length / mean_speed``, stretched on short legs where the peak acceleration would
    otherwise exceed ``accel_per_peak_speed`` (1/s) times the peak speed.
    """
    if length <= 0.0:
        return 0.0
    by_speed = length / mean_speed
    peak_speed = MIN_JERK_PEAK_VELOCITY * mean_speed
    by_accel = math.sqrt(MIN_JERK_PEAK_ACCELERATION * length / (accel_per_peak_speed * peak_speed))
    return max(by_speed, by_accel)


def polyline_point(waypoints: list[torch.Tensor], fraction: float) -> torch.Tensor:
    """The point at ``fraction`` of the way along the polyline, by segment count."""
    last = len(waypoints) - 1
    if last <= 0:
        return waypoints[0]
    x = min(max(fraction, 0.0), 1.0) * last
    i = min(int(x), last - 1)
    return waypoints[i] + (waypoints[i + 1] - waypoints[i]) * (x - i)


class MotionStats:
    """Static-frame fraction and acceleration quantiles of a commanded joint target over
    every env of a batch. Call ``reset(i)`` at an episode boundary so the jump to the next
    start pose is not read as motion."""

    STATIC_RAD = 1e-6

    def __init__(self, num_envs: int, control_hz: float = CONTROL_HZ):
        self._hz2 = control_hz * control_hz
        self._prev: list[torch.Tensor | None] = [None] * num_envs
        self._prev_delta: list[torch.Tensor | None] = [None] * num_envs
        self.frames = 0
        self.static = 0
        self._accel = array.array("d")

    def reset(self, i: int) -> None:
        self._prev[i] = None
        self._prev_delta[i] = None

    def observe(self, i: int, q: torch.Tensor) -> None:
        q = q.detach().to("cpu", torch.float64)
        prev = self._prev[i]
        self._prev[i] = q
        if prev is None:
            return
        delta = q - prev
        self.frames += 1
        if float(delta.abs().max()) < self.STATIC_RAD:
            self.static += 1
        if self._prev_delta[i] is not None:
            self._accel.append(float((delta - self._prev_delta[i]).abs().max()) * self._hz2)
        self._prev_delta[i] = delta

    def summary(self) -> dict:
        acc = torch.tensor(self._accel, dtype=torch.float64)
        quantile = (lambda p: float(acc.quantile(p))) if len(acc) else (lambda p: 0.0)
        return {
            "frames": self.frames,
            "near_static_fraction": self.static / self.frames if self.frames else 0.0,
            "accel_p50_rad_s2": quantile(0.5),
            "accel_p90_rad_s2": quantile(0.9),
            "accel_max_rad_s2": float(acc.max()) if len(acc) else 0.0,
        }
