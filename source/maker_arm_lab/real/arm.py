"""The arm backend over ``maker_arm_rs`` (MS5 design sections 4.3 and 8).

``SdkArm`` owns one ``maker_arm_rs.Arm``: it connects over SocketCAN (or to the SDK's own
simulator), enables, and starts the ``Tracking`` controller in the SDK's 200 Hz loop, at
SCHED_FIFO and pinned when ``rt`` is given. The env pushes one row a control step (six joint
targets and the gripper motor angle), due ahead on the loop clock; the controller
interpolates to it and then holds it. ``Arm`` is unsendable: build and use it on one thread.

Only ``Tracking`` runs through ``Arm``, the contact-aware rung A of the tracking design: the
SDK refuses rung F, and B0 and B1 are bench rungs that need a Python command every tick. The
gripper motor angle maps to the jaw opening through ``core.profile.gripper_angle``'s
provisional linear map and its inverse ``jaw_fraction`` until A1 measures the transmission.

The contact residual is what the arm's model does not explain: ``Dynamics.inverse`` at the
measured joint state and the reference acceleration, minus the motors' torque reports. Under
a push f at the pick point it is J^T f, the constraint torque the simulator's
``arm_torques`` gives (MS7d design, L6 and L7).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import maker_arm_rs
import numpy as np
import torch

from maker_arm_lab.core import profile as _profile
from maker_arm_lab.robots.frames import PICK_OFFSET_MEASURED

FIRST_TICK_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ArmReading:
    t: float  # s, the SDK loop clock (since connect)
    q: tuple[float, ...]  # (6,) rad, URDF joint coordinates
    dq: tuple[float, ...]  # (6,) rad/s
    tau: tuple[float, ...]  # (6,) N m, the motors' torque reports
    gripper: float  # rad, the gripper motor's angle
    state: str  # "enabled" while the loop runs the controller
    alive: bool  # the SDK loop thread still ticks
    fault: str | None


def jaw_fraction(
    profile: dict, angle: float, gripper_map: tuple[float, float] | None = None
) -> float:
    """The jaw opening fraction of gripper motor angle ``angle``, clipped to [0, 1].

    Without ``gripper_map`` it inverts ``core.profile.gripper_angle``'s provisional linear
    map (INITIAL: the gripper map script replaces it). With ``gripper_map = (a, b)``, the
    caliper line ``gap = a * angle + b`` (m, the full gap between the jaws), the fraction is
    the gap over the gap at the open limit, the profile gripper's ``q_hi`` put through the
    same line. ValueError if that gap is not positive."""
    g = profile["joints"][_profile.GRIPPER_INDEX]
    lo, hi = float(g["q_lo"]), float(g["q_hi"])
    if gripper_map is None:
        return min(1.0, max(0.0, (float(angle) - lo) / (hi - lo)))
    a, b = float(gripper_map[0]), float(gripper_map[1])
    full = a * hi + b
    if not full > 0.0:
        raise ValueError(f"the gripper map gives a gap of {full} m at the open limit {hi} rad")
    return min(1.0, max(0.0, (a * float(angle) + b) / full))


def gripper_angle_of(
    profile: dict, fraction: float, gripper_map: tuple[float, float] | None = None
) -> float:
    """The gripper motor angle that commands jaw opening ``fraction`` (clipped to [0, 1]): the
    inverse of ``jaw_fraction``. Without ``gripper_map`` it is ``core.profile.gripper_angle``.
    With ``(a, b)``, ``angle = (fraction * full_gap - b) / a``, ``full_gap`` the gap at the
    profile gripper's ``q_hi``, clamped to the gripper's soft limits. ValueError if the map
    is flat (a = 0) or gives no positive gap at the open limit."""
    f = min(1.0, max(0.0, float(fraction)))
    if gripper_map is None:
        return _profile.gripper_angle(profile, f)
    g = profile["joints"][_profile.GRIPPER_INDEX]
    lo, hi = float(g["q_lo"]), float(g["q_hi"])
    a, b = float(gripper_map[0]), float(gripper_map[1])
    full = a * hi + b
    if a == 0.0 or not full > 0.0:
        raise ValueError(f"the gripper map ({a}, {b}) cannot be inverted to a command")
    return min(hi, max(lo, (f * full - b) / a))


def pick_point(ik, q) -> np.ndarray:
    """(3,) the FK pick point, ``grasp_center`` plus ``PICK_OFFSET_MEASURED`` in its frame."""
    pos, rot = ik.fk(torch.as_tensor(np.asarray(q, dtype=np.float64)))
    return pos.numpy() + rot.numpy() @ np.asarray(PICK_OFFSET_MEASURED)


def pick_jacobian(ik, q, eps: float = 1e-6) -> np.ndarray:
    """(3, 6) translational Jacobian of the FK pick point, central differences."""
    q = np.asarray(q, dtype=np.float64)
    cols = []
    for j in range(6):
        step = np.zeros(6)
        step[j] = eps
        cols.append((pick_point(ik, q + step) - pick_point(ik, q - step)) / (2.0 * eps))
    return np.stack(cols, axis=1)


def reference_acceleration(targets: Sequence[Sequence[float]], dt: float) -> tuple[float, ...]:
    """The second difference of the last three commanded arm targets over ``dt``; zeros with
    fewer than three."""
    if len(targets) < 3:
        return (0.0,) * 6
    a, b, c = (np.asarray(t, dtype=np.float64) for t in list(targets)[-3:])
    return tuple(float(v) for v in (c - 2.0 * b + a) / (dt * dt))


def contact_residual(dynamics, q, dq, ddq, tau) -> tuple[float, ...]:
    """(6,) N m: ``dynamics.inverse(q, dq, ddq)`` minus the measured torque ``tau``."""
    model = dynamics.inverse(list(map(float, q)), list(map(float, dq)), list(map(float, ddq)))
    return tuple(float(m) - float(t) for m, t in zip(model, tau))


def _reading(snap: dict | None) -> ArmReading | None:
    if snap is None:
        return None
    return ArmReading(
        t=float(snap["t"]),
        q=tuple(float(v) for v in snap["positions"][:6]),
        dq=tuple(float(v) for v in snap["velocities"][:6]),
        tau=tuple(float(v) for v in snap["torques"][:6]),
        gripper=float(snap["positions"][6]),
        state=str(snap["state"]),
        alive=bool(snap["loop_alive"]),
        fault=snap["fault"],
    )


class SdkArm:
    """One arm session: ``start`` enables and starts tracking, then ``push`` one row a step."""

    controller = "tracking"

    def __init__(
        self,
        arm,
        dynamics,
        profile: dict | None = None,
        params: dict | None = None,
        rt=None,
    ):
        self.arm = arm
        self.profile = profile if profile is not None else maker_arm_rs.profile()
        self.tracking = maker_arm_rs.Tracking(dynamics, params=params, profile=self.profile)
        self.rt = None if rt is None else (int(rt[0]), int(rt[1]))

    @classmethod
    def connect(cls, interface: str, dynamics, profile: dict | None = None, **kwargs) -> SdkArm:
        """``interface`` "sim" is the SDK's built-in simulator, anything else a SocketCAN
        interface such as "can0" or "vcan0". ``profile`` (a reduced ``maker_arm_rs.profile()``,
        ``real.rig.reduced_profile``) reaches both the arm and the ``Tracking`` controller; the
        SDK refuses one that loosens a limit with ValueError before it connects."""
        if interface == "sim":
            arm = maker_arm_rs.Arm.sim(profile)
        else:
            arm = maker_arm_rs.Arm.socketcan(interface, profile)
        return cls(arm, dynamics, profile=profile, **kwargs)

    def start(self) -> ArmReading:
        """Enable, start tracking and wait for the loop's first snapshot.

        If ``start_tracking`` raises after ``enable`` succeeds, the session is left enabled
        with no loop running it; the caller must still call ``stop()`` to disable the motors.
        """
        self.arm.enable()
        self.arm.start_tracking(self.tracking, rt=self.rt)
        deadline = time.monotonic() + FIRST_TICK_TIMEOUT_S
        while time.monotonic() < deadline:
            reading = self.read()
            if reading is not None:
                return reading
            time.sleep(0.005)
        raise RuntimeError(f"no snapshot within {FIRST_TICK_TIMEOUT_S} s of start_tracking")

    def read(self) -> ArmReading | None:
        return _reading(self.arm.snapshot())

    def now(self) -> float | None:
        """The loop clock, None once the loop has not ticked for 0.2 s."""
        return self.tracking.now()

    def push(self, due: float, dt: float, row: Sequence[float]) -> None:
        self.tracking.push(float(due), float(dt), [[float(v) for v in row]])

    def push_rows(self, t0: float, dt: float, rows: Sequence[Sequence[float]]) -> None:
        """Rows k of ``rows`` (six joints and the gripper motor angle) due at ``t0 + k dt``."""
        self.tracking.push(float(t0), float(dt), [[float(v) for v in row] for row in rows])

    def telemetry(self) -> dict | None:
        return self.tracking.telemetry()

    def drain(self) -> tuple[dict | None, dict]:
        """Every tick since the last drain: the loop's columns (``Arm.drain``, None with no
        loop) and the controller's (``Tracking.drain``). Join them on ``t`` and ``tick_t``."""
        return self.arm.drain(), self.tracking.drain()

    def hold(self) -> None:
        """Swap in the SDK's hold at the current pose (torque stays on).

        Ends this session's use of ``self.tracking``: from the loop's next tick, ``push``
        raises RuntimeError and ``now`` returns None. There is no resuming this ``Tracking``
        short of ``stop()`` and a fresh session.
        """
        self.arm.hold_now()

    def stop(self) -> None:
        """Disable every motor: the arm goes limp."""
        self.arm.stop()


HOLD_STALE_S = 0.2
"""The motors' CAN_TIMEOUT: a loop whose tick has not changed for this long no longer runs."""


class SdkHoldArm:
    """One arm session under the SDK's hold controller (``Arm.start_hold``): the arm holds the
    pose it had at ``start`` on the profile's PD gains, with no model, no feed-forward and no
    ``Tracking``. ``real_session``'s surface (``read``, ``now``, ``telemetry``, ``drain``,
    ``hold``, ``stop``); ``push_rows`` raises, since this controller takes no target. Under the
    reduced caps a PD hold sags by G / kp, which is what a session with no trusted model
    gets (ruling R32: the model's joint frame is not yet the arm's)."""

    controller = "hold"

    def __init__(self, arm, profile: dict | None = None, rt=None):
        self.arm = arm
        self.profile = profile if profile is not None else maker_arm_rs.profile()
        self.rt = None if rt is None else (int(rt[0]), int(rt[1]))
        self._tick = None
        self._tick_mono = 0.0

    @classmethod
    def connect(cls, interface: str, profile: dict | None = None, rt=None) -> SdkHoldArm:
        """As ``SdkArm.connect``, with no controller to build."""
        if interface == "sim":
            arm = maker_arm_rs.Arm.sim(profile)
        else:
            arm = maker_arm_rs.Arm.socketcan(interface, profile)
        return cls(arm, profile=profile, rt=rt)

    def start(self) -> ArmReading:
        """Enable, start the hold loop and wait for its first snapshot; as ``SdkArm.start``,
        a failure after ``enable`` leaves the motors to the caller's ``stop()``."""
        self.arm.enable()
        self.arm.start_hold(rt=self.rt)
        deadline = time.monotonic() + FIRST_TICK_TIMEOUT_S
        while time.monotonic() < deadline:
            reading = self.read()
            if reading is not None:
                return reading
            time.sleep(0.005)
        raise RuntimeError(f"no snapshot within {FIRST_TICK_TIMEOUT_S} s of start_hold")

    def read(self) -> ArmReading | None:
        return _reading(self.arm.snapshot())

    def now(self) -> float | None:
        """The loop clock as ``Tracking.now`` reads it: the last tick's ``t`` plus the wall
        time since that tick changed. None before the first tick, and None once the tick has
        not changed for ``HOLD_STALE_S``."""
        snap = self.arm.snapshot()
        if snap is None:
            return None
        mono = time.monotonic()
        if snap["tick"] != self._tick:
            self._tick, self._tick_mono = snap["tick"], mono
        elif mono - self._tick_mono > HOLD_STALE_S:
            return None
        return float(snap["t"]) + (mono - self._tick_mono)

    def telemetry(self) -> None:
        return None

    def push_rows(self, t0: float, dt: float, rows) -> None:
        raise RuntimeError("the hold controller takes no target; use the tracking controller")

    def drain(self) -> tuple[dict | None, None]:
        """The loop's columns (``Arm.drain``); there is no controller telemetry."""
        return self.arm.drain(), None

    def hold(self) -> None:
        """Nothing to swap: the hold controller already holds. ``hold_now`` would re-target
        the sagged pose and let the arm sink by one more sag."""

    def stop(self) -> None:
        """Disable every motor: the arm goes limp."""
        self.arm.stop()
