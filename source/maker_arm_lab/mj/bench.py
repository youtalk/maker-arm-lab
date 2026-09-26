"""Arm-only MuJoCo bench for the compliant tracking controller (tracking design section 8).

The plant is the vendored URDF on the scene's compiler block and Z-up mount: gravity on, no
gravcomp, the jaws held at the adapter's lumped opening (their joints removed), a 1 ms
implicitfast step, and the RobStride MIT law applied at every physics step from a command
held for its 5 ms tick. Only a massless fingertip sphere at grasp_center collides. Its
contacts take the sim's CONTACT_SOLREF and MuJoCo's defaults for everything else (not the
sim's desk solimp and solmix, elliptic cone, impratio or noslip). The controller at tick k sees the state and the applied torque 1 ms
after tick k-1's command went out (the MIT reply), with Gaussian torque noise.

Four rungs drive it: B0 (the profile PD, the upstream behavior), B1 (B0 plus the exact
gravity, what the training sim's gravcomp assumes), F and A (``maker_arm_rs.Tracking``
without and with its contact side). Every command goes through ``maker_arm_rs.clamp_command``
with the sim profile, as the loop's output stage does.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import maker_arm_rs
import mujoco
import numpy as np

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.core import dynamics_model
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.mj import scene
from maker_arm_lab.robots.frames import ARM_ARMATURE

PHYSICS_DT = 0.001
TICK_S = 0.005
STEPS_PER_TICK = 5
TAU_NOISE_SIGMA = 0.2  # INITIAL: N m, the QDD spike's noise ceiling
COULOMB_NM = {"RS00": 0.1, "RS02": 0.2}  # INITIAL: MA2 fits the real friction
VISCOUS_NMS = 0.02  # INITIAL: N m s/rad
TIP_RADIUS = 0.008  # INITIAL: a fingertip-sized sphere at grasp_center
WALL_HALF = (0.004, 0.05, 0.05)  # m: thin along its normal
RUNGS = ("B0", "B1", "F", "A")
ARM_DOF = 6


@dataclass
class Plant:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos: np.ndarray
    dof: np.ndarray
    tip_body: int
    tip_geom: int
    rating: np.ndarray


@dataclass
class Trace:
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tip: np.ndarray
    force: np.ndarray
    r: np.ndarray
    offset: np.ndarray
    q_r: np.ndarray
    gate: np.ndarray


def grip_value() -> float:
    """The gripper entry of every plan row: the profile angle at the lumped opening."""
    return _profile.gripper_angle(_profile.sim_profile(), dynamics_model.JAW_FRACTION)


def build_plant(
    wall: tuple[Sequence[float], Sequence[float]] | None = None, desk: bool = False
) -> Plant:
    profile = _profile.sim_profile()
    spec = mujoco.MjSpec.from_string(scene._urdf_with_compiler())
    spec.compiler.fusestatic = False
    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.body("base_link").quat = scene._mount_quat()
    opening = dynamics_model.JAW_FRACTION * urdf_prep.gripper_travel()
    # MuJoCo 3.13's URDF importer turns the jaw mimic into a joint equality; it goes with them.
    for eq in list(spec.equalities):
        if {eq.name1, eq.name2} & set(urdf_prep.GRIPPER_JOINT_NAMES):
            spec.delete(eq)
    for joint_name, body_name in zip(urdf_prep.GRIPPER_JOINT_NAMES, dynamics_model.JAW_LINKS):
        joint, body = spec.joint(joint_name), spec.body(body_name)
        shift = np.zeros(3)
        mujoco.mju_rotVecQuat(shift, np.asarray(joint.axis, float), np.asarray(body.quat, float))
        body.pos = np.asarray(body.pos, float) + opening * shift
        spec.delete(joint)
    for geom in spec.geoms:
        geom.contype, geom.conaffinity = 0, 0
    tip = spec.body("grasp_center").add_geom(
        name="tip", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TIP_RADIUS, 0.0, 0.0]
    )
    tip.density = 0.0  # massless: the plant stays the adapter's exact model
    tip.contype, tip.conaffinity = 1, 1
    tip.solref = list(scene.CONTACT_SOLREF)
    for name, j in zip(urdf_prep.ARM_JOINT_NAMES, profile["joints"][:ARM_DOF]):
        joint = spec.joint(name)
        joint.armature = ARM_ARMATURE
        joint.damping = [VISCOUS_NMS, 0.0, 0.0]  # MuJoCo 3.13: linear, then polynomial terms
        joint.frictionloss = COULOMB_NM[j["model"]]
    if desk:
        plane = spec.worldbody.add_geom(
            name="desk", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.01]
        )
        plane.contype, plane.conaffinity = 1, 1
        plane.solref = list(scene.CONTACT_SOLREF)
    if wall is not None:
        center, normal = wall
        yaw = math.atan2(normal[1], normal[0])
        box = spec.worldbody.add_geom(
            name="wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(WALL_HALF),
            pos=[float(v) for v in center],
            quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        )
        box.contype, box.conaffinity = 1, 1
        box.solref = list(scene.CONTACT_SOLREF)
    model = spec.compile()
    data = mujoco.MjData(model)
    names = urdf_prep.ARM_JOINT_NAMES
    return Plant(
        model=model,
        data=data,
        qpos=np.array([model.joint(n).qposadr[0] for n in names]),
        dof=np.array([model.joint(n).dofadr[0] for n in names]),
        tip_body=model.body("grasp_center").id,
        tip_geom=model.geom("tip").id,
        rating=np.array(
            [_profile.MOTOR_RATING_NM[j["model"]] for j in profile["joints"][:ARM_DOF]]
        ),
    )


def tip_of(plant: Plant, q: Sequence[float]) -> np.ndarray:
    """grasp_center for joint vector ``q``, on scratch data (the plant's state is untouched)."""
    scratch = mujoco.MjData(plant.model)
    scratch.qpos[plant.qpos] = q
    mujoco.mj_kinematics(plant.model, scratch)
    return scratch.xpos[plant.tip_body].copy()


def _tip_force(plant: Plant) -> float:
    total, f = 0.0, np.zeros(6)
    for i in range(plant.data.ncon):
        c = plant.data.contact[i]
        if plant.tip_geom in (c.geom1, c.geom2):
            mujoco.mj_contactForce(plant.model, plant.data, i, f)
            total += abs(f[0])
    return total


class HeldPlan:
    """B0 and B1's raw target: the latest row at or before t (30 Hz targets held, as the
    training sim does), the first row before the plan starts, ``hold`` with no plan."""

    def __init__(self, hold: Sequence[float]):
        self.hold = list(hold)
        self.times: list[float] = []
        self.rows: list[list[float]] = []

    def push(self, t0: float, dt: float, rows: Sequence[Sequence[float]]) -> None:
        keep = bisect.bisect_left(self.times, t0)
        self.times = self.times[:keep] + [t0 + k * dt for k in range(len(rows))]
        self.rows = self.rows[:keep] + [list(r) for r in rows]

    def sample(self, t: float) -> list[float]:
        if not self.rows:
            return self.hold
        return self.rows[max(bisect.bisect_right(self.times, t) - 1, 0)]


def run(
    plant: Plant,
    rung: str,
    q0: Sequence[float],
    pushes: Sequence[tuple[float, float, float, Sequence[Sequence[float]]]],
    duration: float,
    *,
    tracking: maker_arm_rs.Tracking | None = None,
    exact: maker_arm_rs.Dynamics | None = None,
    seed: int = 0,
) -> Trace:
    """One run of ``rung`` from rest at ``q0``. Each push is ``(t_push, t0, dt, rows)``."""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung}")
    if rung in ("F", "A") and tracking is None:
        raise ValueError(f"rung {rung} needs a Tracking")
    if rung == "B1" and exact is None:
        raise ValueError("rung B1 needs the exact Dynamics")
    profile = _profile.sim_profile()
    grip = grip_value()
    joints = profile["joints"]
    kp = [float(j["kp"]) for j in joints]
    kd = [float(j["kd"]) for j in joints]
    m, d = plant.model, plant.data
    mujoco.mj_resetData(m, d)
    d.qpos[plant.qpos] = q0
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    plan = HeldPlan([*q0, grip])
    pending = sorted(pushes, key=lambda p: p[0])
    q_m, dq_m, tau_m = np.array(q0, float), np.zeros(ARM_DOF), np.zeros(ARM_DOF)
    ticks = round(duration / TICK_S)
    rec: dict[str, list] = {k: [] for k in Trace.__dataclass_fields__}
    for k in range(ticks):
        t = k * TICK_S
        while pending and pending[0][0] <= t + 1e-12:
            _, t0, dt, rows = pending.pop(0)
            (tracking.push if rung in ("F", "A") else plan.push)(t0, dt, [list(r) for r in rows])
        if rung in ("F", "A"):
            cmds = tracking.update(t, [*q_m, grip], [*dq_m, 0.0], [*tau_m, 0.0])
            tel = tracking.telemetry()
            r, offset, q_r, gate = tel["r"], tel["offset"], tel["q_r"], tel["gate"]
        else:
            target = plan.sample(t)
            tau = exact.gravity(list(q_m)) if rung == "B1" else [0.0] * ARM_DOF
            cmds = [(target[j], 0.0, kp[j], kd[j], tau[j]) for j in range(ARM_DOF)]
            cmds.append((grip, 0.0, kp[ARM_DOF], kd[ARM_DOF], 0.0))
            r, offset, q_r, gate = [0.0] * ARM_DOF, [0.0] * ARM_DOF, target[:ARM_DOF], 1.0
        cmds, _ = maker_arm_rs.clamp_command([tuple(c) for c in cmds], profile)
        pos, vel, kps, kds, tff = (np.array([c[i] for c in cmds[:ARM_DOF]]) for i in range(5))
        force = 0.0
        for s in range(STEPS_PER_TICK):
            q, dq = d.qpos[plant.qpos], d.qvel[plant.dof]
            applied = np.clip(kps * (pos - q) + kds * (vel - dq) + tff, -plant.rating, plant.rating)
            d.qfrc_applied[plant.dof] = applied
            mujoco.mj_step(m, d)
            force = max(force, _tip_force(plant))
            if s == 0:
                q_m = d.qpos[plant.qpos].copy()
                dq_m = d.qvel[plant.dof].copy()
                tau_m = applied + rng.normal(0.0, TAU_NOISE_SIGMA, ARM_DOF)
        rec["t"].append((k + 1) * TICK_S)
        rec["q"].append(d.qpos[plant.qpos].copy())
        rec["dq"].append(d.qvel[plant.dof].copy())
        rec["tip"].append(d.xpos[plant.tip_body].copy())
        rec["force"].append(force)
        rec["r"].append(list(r))
        rec["offset"].append(list(offset))
        rec["q_r"].append(list(q_r))
        rec["gate"].append(gate)
    return Trace(**{k: np.asarray(v, dtype=float) for k, v in rec.items()})
