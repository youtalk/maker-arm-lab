"""Global inverse kinematics for the Maker Arm: solve the whole grasp pose at once.

Why this exists, in one paragraph. Isaac Lab's differential IK is a damped-least-squares
*local* solver with no joint-limit avoidance and no null-space term. Task D3 measured that
its Jacobian is correctly wired -- the row it slices is `grasp_center`'s own, to within
0.077 mm/rad of a kinematic finite difference -- and that it nevertheless walks
``link_005_joint`` into its stop rather than turning the base, after which the arm is frozen
and every later phase servos a dead mechanism. It never once closed the jaws on a brick. A
global solver has no such failure mode because it does not walk anywhere: it solves the pose
in one shot and can be asked for the branch that sits furthest from the stops. Measured, on
the unchanged task definition, it carries the brick 8 times in 9 against that 0.

The task posed here is the one that ships: `grasp_center` at a point, the tool axis tilted
``GRASP_TILT_RAD`` off world -z, the tool lean pointed along a commanded azimuth, and the
jaw axis square to the brick. That is six scalar constraints on six joints, so the solution
set is generically DISCRETE -- there is no redundancy to resolve and no null space to
project into. What multi-start buys is therefore not optimisation but *branch selection*:
different seeds converge to different elbow/wrist configurations, and this module returns
the one whose tightest joint is furthest from its limit.

Freeing the lean azimuth (a 5-DOF task, one degree of redundancy) was measured too and is
deliberately NOT implemented: it more than doubles the worst-case joint margin band-wide
(0.1428 -> 0.3336 rad) and covers band corners the 6-DOF task cannot reach, but it grasps
WORSE (4/9 against 8/9), because a redundancy resolved for joint margin picks poses that are
bad for the grasp. It is a coverage patch for a later task, not an alternative to this one.

isaaclab-free and torch-only, per this package's contract, so the solver, its limit handling
and its continuity constraint are all pinned by CPU tests -- the coverage this task lacked
throughout, when every defect it found needed a GPU round to see.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch

from maker_arm_lab.assets.urdf_prep import (
    ARM_JOINT_NAMES,
    MOUNT_RPY,
    VENDORED_URDF,
    arm_joint_limits,
)
from maker_arm_lab.robots.frames import JAW_AXIS_BODY, TOOL_AXIS_BODY

ARM_DOF = 6

# The kinematic chain from the mount to the IK target, in order. `grasp_center_joint` is a
# fixed joint, so it contributes a transform but no coordinate.
_CHAIN = (*ARM_JOINT_NAMES, "grasp_center_joint")

# Per-waypoint joint-step ceiling for `solve_path`. A Cartesian leg is walked in ~4-5 mm
# waypoints and the Jacobian columns measured in Task D3 part 7 are 0.23-0.27 m/rad, so the
# geometry needs about 0.02 rad per waypoint; 0.12 leaves 5x headroom. That headroom is the
# point. The cap is not there to shape the path -- measured, it never binds, and the
# constrained and unconstrained chains agree to four decimals -- it is there so that a
# solver jumping to a different IK branch mid-approach fails loudly instead of silently
# commanding a 0.5 rad lurch through the scene.
MAX_WAYPOINT_STEP = 0.12

# A solution counts as feasible only if it actually satisfies the task. These are far tighter
# than the grasp needs (the brick is 15.8 mm wide); a Levenberg-Marquardt root find on a
# well-posed 6x6 system converges to machine precision or not at all, so a loose tolerance
# would only let non-convergence through as success.
POS_TOL = 1e-5  # m
ANG_TOL = math.radians(0.05)

_DEFAULT_RESTARTS = 24
# Fixed so that two runs of the same task return the same joint vector. An oracle that
# silently picks a different IK branch run to run would make datasets irreproducible.
_RNG_SEED = 20260906


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def wrap_to_pi(angle: torch.Tensor | float) -> torch.Tensor | float:
    """Fold an angle into [-pi, pi) -- exactly +pi comes back as -pi."""
    if isinstance(angle, torch.Tensor):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fold_to_half_turn(angle: torch.Tensor | float) -> torch.Tensor | float:
    """Fold an angular error modulo pi -- a jaw is symmetric under a half turn."""
    return wrap_to_pi(2.0 * angle) / 2.0


@dataclass(frozen=True)
class GraspTarget:
    """The six scalars the grasp task constrains, all in world coordinates."""

    position: tuple[float, float, float]
    """Where `grasp_center` must sit (m)."""
    tilt: float
    """Angle of the tool axis off world -z (rad). Magnitude only."""
    jaw_heading: float
    """World heading of the jaw closing axis (rad), taken modulo pi."""
    lean_azimuth: float
    """World azimuth the tool leans toward (rad). 0.0 is the shipping convention, +x."""


# --- what descend the jaw plates actually allow -------------------------------------
# The jaw-plate collision box relative to `grasp_center`, from the vendored STLs (quoted in
# robots/frames.py's wrist-camera block). Two of its three extents gate the descend:
JAW_TIP_REACH = 0.0342
"""How far the plates reach PAST grasp_center along the tool axis (m)."""
JAW_FACE_FAR = 0.0625
"""How far the plate face reaches ACROSS the tool axis, on the far side (m)."""

# A tilted plate dips its far corner `JAW_FACE_FAR * sin(tilt)` below its own axial tip, and
# the plate face is nearly twice as long across the tool axis as along it. So tilt costs far
# more desk clearance than it buys, and above this ceiling NO pick offset both clears the
# desk and keeps the brick inside the tip: the condition is `JAW_FACE_FAR sin(t) <
# brick_half_height`, measured at 4.40 deg for a 9.6 mm brick.
#
# The floor is a different problem entirely. At EXACTLY zero tilt the tool axis is vertical,
# so the lean azimuth `atan2(a_y, a_x)` is undefined and the azimuth constraint row of the
# solver goes singular. Measured (Task D3 part 10): with the descend interpolating tilt to
# exactly 0.0, the continuity-constrained path failed at waypoint 39 OF 40 -- the last one,
# the only one at zero tilt -- on five of six bricks tested, including one whose endpoint was
# solvable standalone at 0.534 rad of margin. A descend must therefore be NEAR level, never
# level.
MIN_DESCEND_TILT = math.radians(0.5)
MAX_DESCEND_TILT = math.radians(3.0)


def descend_tilt_ceiling(brick_half_height: float) -> float:
    """Largest tilt (rad) at which some pick offset still clears the desk AND the tip."""
    return math.asin(min(1.0, brick_half_height / JAW_FACE_FAR))


def descend_pick_offset(tilt: float, brick_half_height: float) -> tuple[float, float] | None:
    """The pick offset for a descend at `tilt`, and the margin it leaves.

    Returns `(offset_along_tool_axis, margin)` where the offset is NEGATIVE in the
    `PICK_OFFSET_MEASURED` sign convention, or None if the plates cannot clear the desk at
    this tilt at all.

    Two constraints pull against each other and the offset is placed to balance them:

      * deep enough that the brick stays inside the plate tip -- `|offset| <= JAW_TIP_REACH`;
      * shallow enough that the plates' lowest corner clears the desk --
        `|offset| >= JAW_TIP_REACH + JAW_FACE_FAR tan(t) - brick_half_height / cos(t)`.

    Equalising the two margins is what this returns. The window is small by construction on
    this gripper -- about 2.1 mm either side at 0.5 deg -- because `grasp_center` sits only
    34.2 mm inside a plate that has to reach past a brick resting on a table.
    """
    lower = JAW_TIP_REACH + JAW_FACE_FAR * math.tan(tilt) - brick_half_height / math.cos(tilt)
    if lower >= JAW_TIP_REACH:
        return None
    offset = (JAW_TIP_REACH + lower * math.cos(tilt)) / (1.0 + math.cos(tilt))
    return -offset, JAW_TIP_REACH - offset


def descend_tilt_ladder(brick_half_height: float, steps: int = 6) -> list[float]:
    """Tilts to try for a descend, smallest first, all inside the fouling window.

    Ruling D3-11 candidate (2): level where level solves, otherwise the minimum tilt that
    brick admits. "Level" is `MIN_DESCEND_TILT`, not zero, for the singularity above.
    """
    top = min(MAX_DESCEND_TILT, descend_tilt_ceiling(brick_half_height))
    if top <= MIN_DESCEND_TILT:
        return [MIN_DESCEND_TILT]
    return [MIN_DESCEND_TILT + (top - MIN_DESCEND_TILT) * i / (steps - 1) for i in range(steps)]


def tool_axis_world(tilt: float, lean_azimuth: float) -> torch.Tensor:
    """World-frame unit vector the tool axis points along for a commanded attitude.

    Shared rather than re-derived by each caller: the descend has to place `grasp_center`
    an offset along this axis from the brick, and this task has already paid once for a
    convention drifting between the check script and the oracle.
    """
    return torch.tensor(
        [
            math.sin(tilt) * math.cos(lean_azimuth),
            math.sin(tilt) * math.sin(lean_azimuth),
            -math.cos(tilt),
        ],
        dtype=torch.float64,
    )


@dataclass(frozen=True)
class GraspPose:
    """A pose read back off the arm, in the same four quantities `GraspTarget` commands."""

    position: torch.Tensor
    tilt: float
    jaw_heading: float
    lean_azimuth: float


@dataclass(frozen=True)
class IKSolution:
    q: torch.Tensor
    """The six joint coordinates (rad), URDF order, float64."""
    margin: float
    """Distance from its nearer limit of the tightest joint (rad). Larger is safer."""
    position_error: float
    tilt_error: float
    heading_error: float
    azimuth_error: float
    step: float
    """Inf-norm joint distance from the seed pose the caller asked continuity against."""

    def as_list(self) -> list[float]:
        return [float(v) for v in self.q]


class GlobalIK:
    """Forward kinematics and a global solver for the Maker Arm's six arm joints.

    Construction parses the vendored URDF once; solving is allocation-light after that.
    Everything is float64 on CPU: this is a 6x6 root find run a few dozen times per grasp,
    not a per-step control loop, and float32 would put the convergence tolerance in the same
    range as the residual.
    """

    def __init__(self, urdf: Path = VENDORED_URDF) -> None:
        self._links = self._load_chain(urdf)
        self._mount = _rpy_matrix(*MOUNT_RPY)
        limits = arm_joint_limits(urdf)
        self.lower = torch.tensor([lo for lo, _ in limits], dtype=torch.float64)
        self.upper = torch.tensor([hi for _, hi in limits], dtype=torch.float64)
        self._tool_axis = torch.tensor(TOOL_AXIS_BODY, dtype=torch.float64)
        self._jaw_axis = torch.tensor(JAW_AXIS_BODY, dtype=torch.float64)

    @staticmethod
    def _load_chain(urdf: Path) -> list[dict]:
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        out = []
        for name in _CHAIN:
            joint = joints[name]
            origin, axis = joint.find("origin"), joint.find("axis")
            out.append(
                {
                    "xyz": torch.tensor(
                        [float(v) for v in origin.get("xyz").split()], dtype=torch.float64
                    ),
                    "rot": _rpy_matrix(*[float(v) for v in origin.get("rpy").split()]),
                    "axis": (
                        torch.tensor(
                            [float(v) for v in axis.get("xyz").split()], dtype=torch.float64
                        )
                        if axis is not None
                        else None
                    ),
                }
            )
        return out

    # -- forward kinematics ---------------------------------------------------
    def fk_batch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 6) joint coordinates -> (N, 3) positions and (N, 3, 3) rotations."""
        q = torch.as_tensor(q, dtype=torch.float64).reshape(-1, ARM_DOF)
        n = q.shape[0]
        rot = self._mount.expand(n, 3, 3).clone()
        pos = torch.zeros(n, 3, dtype=torch.float64)
        index = 0
        for link in self._links:
            pos = pos + torch.einsum("nij,j->ni", rot, link["xyz"])
            rot = rot @ link["rot"]
            if link["axis"] is not None:
                rot = rot @ self._axis_rotation(link["axis"], q[:, index])
                index += 1
        return pos, rot

    def fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(6,) joint coordinates -> (3,) position and (3, 3) rotation of `grasp_center`."""
        pos, rot = self.fk_batch(q)
        return pos[0], rot[0]

    @staticmethod
    def _axis_rotation(axis: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Rodrigues, batched over `angles`."""
        unit = axis / torch.linalg.norm(axis)
        skew = torch.tensor(
            [
                [0.0, -unit[2], unit[1]],
                [unit[2], 0.0, -unit[0]],
                [-unit[1], unit[0], 0.0],
            ],
            dtype=torch.float64,
        )
        sin = torch.sin(angles).reshape(-1, 1, 1)
        one_minus_cos = (1.0 - torch.cos(angles)).reshape(-1, 1, 1)
        eye = torch.eye(3, dtype=torch.float64)
        return eye + sin * skew + one_minus_cos * (skew @ skew)

    def pose_of(self, q: torch.Tensor) -> GraspPose:
        """The four commanded quantities, read off a joint vector."""
        pos, rot = self.fk(q)
        tool = rot @ self._tool_axis
        jaw = rot @ self._jaw_axis
        return GraspPose(
            position=pos,
            tilt=float(torch.arccos(torch.clamp(-tool[2], -1.0, 1.0))),
            jaw_heading=float(torch.atan2(jaw[1], jaw[0])),
            lean_azimuth=float(torch.atan2(tool[1], tool[0])),
        )

    def margins(self, q: torch.Tensor) -> torch.Tensor:
        """Per-joint distance to the nearer URDF stop (rad). Negative means past it."""
        q = torch.as_tensor(q, dtype=torch.float64)
        return torch.minimum(q - self.lower, self.upper - q)

    # -- the solver -----------------------------------------------------------
    def _residual_batch(self, q: torch.Tensor, target: GraspTarget) -> torch.Tensor:
        """(N, 6) joint coordinates -> (N, 6) task-space residual."""
        pos, rot = self.fk_batch(q)
        tool = torch.einsum("nij,j->ni", rot, self._tool_axis)
        jaw = torch.einsum("nij,j->ni", rot, self._jaw_axis)
        want = torch.tensor(target.position, dtype=torch.float64)
        tilt = torch.arccos(torch.clamp(-tool[:, 2], -1.0, 1.0))
        heading = torch.atan2(jaw[:, 1], jaw[:, 0])
        azimuth = torch.atan2(tool[:, 1], tool[:, 0])
        return torch.stack(
            [
                pos[:, 0] - want[0],
                pos[:, 1] - want[1],
                pos[:, 2] - want[2],
                tilt - target.tilt,
                fold_to_half_turn(heading - target.jaw_heading),
                wrap_to_pi(azimuth - target.lean_azimuth),
            ],
            dim=1,
        )

    def _residual(self, q: torch.Tensor, target: GraspTarget) -> torch.Tensor:
        return self._residual_batch(q.reshape(1, ARM_DOF), target)[0]

    def _residual_jacobian(
        self, q: torch.Tensor, target: GraspTarget, eps: float = 1e-7
    ) -> torch.Tensor:
        """d(residual)/dq by central differences, as one batched FK call."""
        steps = torch.eye(ARM_DOF, dtype=torch.float64) * eps
        probe = torch.cat([q.reshape(1, -1) + steps, q.reshape(1, -1) - steps], dim=0)
        res = self._residual_batch(probe, target)
        return ((res[:ARM_DOF] - res[ARM_DOF:]) / (2.0 * eps)).T

    def _refine(self, seed: torch.Tensor, target: GraspTarget, max_iter: int = 120) -> torch.Tensor:
        """Levenberg-Marquardt onto the constraint manifold, clamped to the joint box."""
        q = torch.clamp(seed.clone(), self.lower, self.upper)
        residual = self._residual(q, target)
        cost = float(residual @ residual)
        damping = 1e-4
        for _ in range(max_iter):
            if cost < 1e-24:
                break
            jac = self._residual_jacobian(q, target)
            normal = jac.T @ jac + damping * torch.eye(ARM_DOF, dtype=torch.float64)
            try:
                delta = torch.linalg.solve(normal, -(jac.T @ residual))
            except RuntimeError:  # singular normal matrix at a kinematic singularity
                break
            candidate = torch.clamp(q + delta, self.lower, self.upper)
            trial = self._residual(candidate, target)
            trial_cost = float(trial @ trial)
            if trial_cost < cost:
                q, residual, cost = candidate, trial, trial_cost
                damping = max(damping * 0.5, 1e-12)
            else:
                damping *= 4.0
                if damping > 1e12:
                    break
        return q

    def _seeds(self, seed: torch.Tensor | None, restarts: int) -> list[torch.Tensor]:
        """The caller's seed first, then a deterministic spread across the joint box."""
        out: list[torch.Tensor] = []
        if seed is not None:
            out.append(torch.as_tensor(seed, dtype=torch.float64).reshape(ARM_DOF).clone())
        generator = torch.Generator().manual_seed(_RNG_SEED)
        if restarts > 0:
            spread = torch.rand((restarts, ARM_DOF), generator=generator, dtype=torch.float64)
            out.extend(self.lower + spread * (self.upper - self.lower))
        return out

    def _grade(
        self, q: torch.Tensor, target: GraspTarget, reference: torch.Tensor | None
    ) -> IKSolution:
        residual = self._residual(q, target)
        step = float(torch.max(torch.abs(q - reference))) if reference is not None else 0.0
        return IKSolution(
            q=q,
            margin=float(torch.min(self.margins(q))),
            position_error=float(torch.linalg.norm(residual[:3])),
            tilt_error=abs(float(residual[3])),
            heading_error=abs(float(residual[4])),
            azimuth_error=abs(float(residual[5])),
            step=step,
        )

    @staticmethod
    def _is_feasible(solution: IKSolution) -> bool:
        return (
            solution.position_error < POS_TOL
            and solution.tilt_error < ANG_TOL
            and solution.heading_error < ANG_TOL
            and solution.azimuth_error < ANG_TOL
            and solution.margin >= 0.0
        )

    def solve(
        self,
        target: GraspTarget,
        seed: torch.Tensor | None = None,
        restarts: int = _DEFAULT_RESTARTS,
        q_prev: torch.Tensor | None = None,
        max_step: float | None = None,
    ) -> IKSolution | None:
        """Solve `target` globally; return the branch furthest from the joint stops.

        `seed` is tried first and is what makes a seeded walk along a Cartesian path cheap.
        `q_prev` + `max_step` impose continuity: candidates that move any joint further than
        `max_step` from `q_prev` are rejected outright, so a branch flip surfaces as None
        rather than as a lurch. Returns None if nothing feasible was found, which is the
        honest answer for a pose the arm cannot reach in this task's six constraints.
        """
        reference = (
            torch.as_tensor(q_prev, dtype=torch.float64).reshape(ARM_DOF)
            if q_prev is not None
            else None
        )
        best: IKSolution | None = None
        for candidate_seed in self._seeds(seed, restarts):
            solution = self._grade(self._refine(candidate_seed, target), target, reference)
            if not self._is_feasible(solution):
                continue
            if max_step is not None and reference is not None and solution.step > max_step:
                continue
            if best is None or solution.margin > best.margin:
                best = solution
        return best

    def solve_path(
        self,
        targets: list[GraspTarget],
        seed: torch.Tensor,
        max_step: float = MAX_WAYPOINT_STEP,
        restarts: int = _DEFAULT_RESTARTS,
    ) -> list[IKSolution] | None:
        """Solve a waypoint list with continuity enforced between adjacent waypoints.

        Returns None as soon as one waypoint has no solution within `max_step` of the last:
        a path that cannot be walked continuously is not a path, and silently returning the
        reachable prefix would hand the caller a trajectory that stops mid-descend.
        """
        out: list[IKSolution] = []
        previous = torch.as_tensor(seed, dtype=torch.float64).reshape(ARM_DOF)
        for target in targets:
            # Escalate rather than always paying for the restart spread. Continuity means
            # the previous waypoint is a seed a few millimetres away, so the seeded solve
            # almost always converges; measured, a 24-waypoint descend costs 0.06 s that way
            # against 19.8 s with the full spread at every waypoint. The spread is still
            # there for the waypoint where the cheap seed does not converge, so this is a
            # speed-up and not a weaker search.
            solution = self.solve(
                target, seed=previous, restarts=0, q_prev=previous, max_step=max_step
            )
            if solution is None and restarts > 0:
                solution = self.solve(
                    target,
                    seed=previous,
                    restarts=restarts,
                    q_prev=previous,
                    max_step=max_step,
                )
            if solution is None:
                return None
            out.append(solution)
            previous = solution.q
        return out


class JointTrim:
    """Integral trim on an open-loop joint-position command.

    The Maker Arm's actuator DOES apply gravity feed-forward
    (`actuators/mit_mode.py` through `GravityCompensatedArticulation`), and Task D3 part 9
    confirmed it was active -- the guard that warns when the flag is set but nothing fills
    the buffer never fired. The residual error is not missing gravity compensation. It is
    EFFORT-LIMIT SATURATION: holding the home pose, `link_004_joint` demands 22.0 Nm against
    a 17.0 Nm motor rating and `link_007_joint` -16.3 Nm against 14.0 Nm, so the applied
    torque is truncated and those joints cannot close their error. Measured open-loop droop
    at home: 0.155 rad on `link_003_joint` and 0.171 rad on `link_007_joint`, about 40 mm of
    `grasp_center`.

    This is the correction a real joint controller's integrator supplies, and it is what a
    saturating joint needs: bias the command so the unsaturated part of the loop carries the
    load. It stays entirely in joint space, so callers still drive through the action
    pipeline and the maker_arm_rs clamp.

    The gain is load-bearing and is deliberately small. The steady-state recursion is
    ``trim <- (1 - g) trim + g e``, stable for any 0 < g < 2 IF the arm reached steady state
    within one control step -- which it does not. Measured at g = 0.25 the integrator wound
    up while the arm was still moving and drove `grasp_center` 107 mm low, worse than the
    uncorrected droop. g = 0.03 has a ~33-step time constant, comfortably slower than the arm.
    """

    def __init__(self, gain: float = 0.03, limit: float = 0.35) -> None:
        if not 0.0 <= gain < 2.0:
            raise ValueError(f"gain must be in [0, 2), got {gain}")
        self.gain = gain
        self.limit = limit
        self.value = torch.zeros(ARM_DOF, dtype=torch.float64)

    def reset(self) -> None:
        self.value = torch.zeros(ARM_DOF, dtype=torch.float64)

    def command(self, desired: torch.Tensor) -> torch.Tensor:
        """The biased target to send for `desired`."""
        return torch.as_tensor(desired, dtype=torch.float64).reshape(ARM_DOF) + self.value

    def update(self, desired: torch.Tensor, measured: torch.Tensor) -> None:
        """Fold one control step's tracking error into the trim."""
        desired = torch.as_tensor(desired, dtype=torch.float64).reshape(ARM_DOF)
        measured = torch.as_tensor(measured, dtype=torch.float64).reshape(ARM_DOF)
        self.value = torch.clamp(
            self.value + self.gain * (desired - measured), -self.limit, self.limit
        )
