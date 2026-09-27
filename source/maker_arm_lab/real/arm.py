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


def jaw_fraction(profile: dict, angle: float) -> float:
    """The jaw opening fraction of gripper motor angle ``angle``, clipped to [0, 1]: the inverse
    of ``core.profile.gripper_angle``. INITIAL: A1 replaces both with the measured
    transmission (MS5 design section 6)."""
    g = profile["joints"][_profile.GRIPPER_INDEX]
    lo, hi = float(g["q_lo"]), float(g["q_hi"])
    return min(1.0, max(0.0, (float(angle) - lo) / (hi - lo)))


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


class SdkArm:
    """One arm session: ``start`` enables and starts tracking, then ``push`` one row a step."""

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
    def connect(cls, interface: str, dynamics, **kwargs) -> SdkArm:
        """``interface`` "sim" is the SDK's built-in simulator, anything else a SocketCAN
        interface such as "can0" or "vcan0"."""
        if interface == "sim":
            arm = maker_arm_rs.Arm.sim()
        else:
            arm = maker_arm_rs.Arm.socketcan(interface)
        return cls(arm, dynamics, **kwargs)

    def start(self) -> ArmReading:
        """Enable, start tracking and wait for the loop's first snapshot."""
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
        snap = self.arm.snapshot()
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

    def now(self) -> float | None:
        """The loop clock, None once the loop has not ticked for 0.2 s."""
        return self.tracking.now()

    def push(self, due: float, dt: float, row: Sequence[float]) -> None:
        self.tracking.push(float(due), float(dt), [[float(v) for v in row]])

    def telemetry(self) -> dict | None:
        return self.tracking.telemetry()

    def hold(self) -> None:
        """Swap in the SDK's hold at the current pose (torque stays on)."""
        self.arm.hold_now()

    def stop(self) -> None:
        """Disable every motor: the arm goes limp."""
        self.arm.stop()
