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
from maker_arm_rs import Kinematics

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
# would only let non-convergence through as success. The values live with the solver in
# maker-arm-rs; they are re-exported here for the callers that read them.
POS_TOL = Kinematics.POS_TOL  # m
ANG_TOL = Kinematics.ANG_TOL  # rad

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

# A flat part (a plate or a tile, 3.2 mm tall) leaves the offset window no room: the tips
# have to hold 2.1 mm off the desk AND reach down a 3.2 mm side face, and the pinch then
# closes over the part's top edge and presses it into the desk (lone-part reads, MuJoCo,
# 2026-09-16: every plate and tile lost that way at the level descend). Below this half
# height the descend instead asks for the tips FLAT_TIP_PRESS under the desk, so the desk
# stops them at zero height and the DESCEND stall is the intended end of the leg. Measured
# on the same reads: 2 mm of press pinches the 2-wide plates and tiles (3.8 to 5.7 N per jaw),
# 1 mm does not, 0 mm only the 1-wide ones.
FLAT_HALF_HEIGHT = 0.002
FLAT_TIP_PRESS = 0.002


def descend_tilt_ceiling(brick_half_height: float) -> float:
    """Largest tilt (rad) at which some pick offset still clears the desk AND the tip."""
    return math.asin(min(1.0, brick_half_height / JAW_FACE_FAR))


class DescendInfeasibleError(ValueError):
    """No pick offset exists at the requested descend tilt.

    Deliberately a ``ValueError`` and therefore an ``Exception``, not a ``SystemExit``.
    Callers that promise a machine-readable outcome on every path -- notably
    ``scripts/scripted_grasp_check.py``, whose ``GRASP_OK`` / ``GRASP_FAIL`` token is parsed
    by the D6/D7 supervisor -- wrap their work in ``except Exception`` and turn a failure
    into a token. A ``SystemExit`` is a ``BaseException``, so it would sail straight through
    that handler and terminate with no token at all, which reads to a supervisor as a hang
    rather than a failure.
    """


def descend_pick_offset_or_raise(tilt: float, brick_half_height: float) -> tuple[float, float]:
    """`descend_pick_offset`, with the None turned into a diagnosable exception.

    The None return is correct for callers searching a ladder of tilts, where "this rung is
    infeasible" is an ordinary answer. It is a trap for callers using a single caller-supplied
    tilt, which indexed the result and got a TypeError far from the cause.
    """
    solved = descend_pick_offset(tilt, brick_half_height)
    if solved is None:
        raise DescendInfeasibleError(
            f"no pick offset exists at a descend tilt of {math.degrees(tilt):.2f} deg for a "
            f"brick of half-height {brick_half_height * 1000:.1f} mm: above roughly "
            f"{math.degrees(descend_tilt_ceiling(brick_half_height)):.2f} deg the jaw plates "
            f"cannot clear the desk at any offset. Use a smaller tilt."
        )
    return solved


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
    if brick_half_height < FLAT_HALF_HEIGHT:
        # The margin is negative on purpose: the tips are asked under the desk.
        offset = JAW_TIP_REACH - brick_half_height - FLAT_TIP_PRESS
        return -offset, -FLAT_TIP_PRESS
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
    if brick_half_height < FLAT_HALF_HEIGHT:
        return [MIN_DESCEND_TILT]  # a pressed tip gains nothing from tilt
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

    The solver itself lives in maker-arm-rs (``maker_arm_rs.Kinematics``, from
    ``crates/maker-arm/src/kinematics.rs``): FK over the URDF chain, the six-scalar grasp
    residual, the central-difference Jacobian, the Levenberg-Marquardt refine, multi-seed
    branch selection, and the continuity-constrained path walk. It moved there because the
    torch version spent its time in dispatch overhead on 3x3 matrices: a 24-restart solve
    cost 1.18 s. This class parses the vendored URDF once, hands the chain to Rust, and
    converts between the lab's tensors and dataclasses and the plain Python data the binding
    takes. The restart spread is still drawn here, with torch's generator under the fixed
    seed, so a solve tries the same seeds MS1 and MS2 tried. Everything is float64.
    """

    def __init__(self, urdf: Path = VENDORED_URDF) -> None:
        limits = arm_joint_limits(urdf)
        self.lower = torch.tensor([lo for lo, _ in limits], dtype=torch.float64)
        self.upper = torch.tensor([hi for _, hi in limits], dtype=torch.float64)
        self._kin = Kinematics(
            mount_rpy=tuple(MOUNT_RPY),
            links=self._load_chain(urdf),
            lower=self.lower.tolist(),
            upper=self.upper.tolist(),
            tool_axis=tuple(TOOL_AXIS_BODY),
            jaw_axis=tuple(JAW_AXIS_BODY),
        )

    @staticmethod
    def _load_chain(urdf: Path) -> list[tuple[tuple, tuple, tuple | None]]:
        """The chain's joints as ``(xyz, rpy, axis)`` per URDF joint, ``axis`` None when fixed."""
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        out = []
        for name in _CHAIN:
            joint = joints[name]
            origin, axis = joint.find("origin"), joint.find("axis")
            out.append(
                (
                    tuple(float(v) for v in origin.get("xyz").split()),
                    tuple(float(v) for v in origin.get("rpy").split()),
                    tuple(float(v) for v in axis.get("xyz").split()) if axis is not None else None,
                )
            )
        return out

    # -- forward kinematics ---------------------------------------------------
    def fk_batch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 6) joint coordinates -> (N, 3) positions and (N, 3, 3) rotations."""
        q = torch.as_tensor(q, dtype=torch.float64).reshape(-1, ARM_DOF)
        pos, rot = self._kin.fk_batch(q.tolist())
        return torch.tensor(pos, dtype=torch.float64), torch.tensor(rot, dtype=torch.float64)

    def fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(6,) joint coordinates -> (3,) position and (3, 3) rotation of `grasp_center`."""
        pos, rot = self._kin.fk(_joints(q))
        return torch.tensor(pos, dtype=torch.float64), torch.tensor(rot, dtype=torch.float64)

    def pose_of(self, q: torch.Tensor) -> GraspPose:
        """The four commanded quantities, read off a joint vector."""
        position, tilt, jaw_heading, lean_azimuth = self._kin.pose_of(_joints(q))
        return GraspPose(
            position=torch.tensor(position, dtype=torch.float64),
            tilt=tilt,
            jaw_heading=jaw_heading,
            lean_azimuth=lean_azimuth,
        )

    def margins(self, q: torch.Tensor) -> torch.Tensor:
        """Per-joint distance to the nearer URDF stop (rad). Negative means past it."""
        return torch.tensor(self._kin.margins(_joints(q)), dtype=torch.float64)

    # -- the solver -----------------------------------------------------------
    def _seeds(self, seed: torch.Tensor | None, restarts: int) -> list[list[float]]:
        """The caller's seed first, then a deterministic spread across the joint box."""
        out: list[list[float]] = []
        if seed is not None:
            out.append(_joints(seed))
        generator = torch.Generator().manual_seed(_RNG_SEED)
        if restarts > 0:
            spread = torch.rand((restarts, ARM_DOF), generator=generator, dtype=torch.float64)
            out.extend((self.lower + spread * (self.upper - self.lower)).tolist())
        return out

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
        found = self._kin.solve(
            _target(target),
            self._seeds(seed, restarts),
            q_prev=None if q_prev is None else _joints(q_prev),
            max_step=max_step,
        )
        return None if found is None else _solution(found)

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

        Each waypoint is seeded from the previous solution and the restart spread is only
        tried where that seed does not converge: measured on the torch solver, a 24-waypoint
        descend cost 0.06 s that way against 19.8 s with the spread at every waypoint, and
        the spread is still there for the waypoint where the cheap seed fails, so this is a
        speed-up and not a weaker search.
        """
        path = self._kin.solve_path(
            [_target(t) for t in targets], _joints(seed), self._seeds(None, restarts), max_step
        )
        return None if path is None else [_solution(s) for s in path]


def _joints(q: torch.Tensor) -> list[float]:
    return torch.as_tensor(q, dtype=torch.float64).reshape(ARM_DOF).tolist()


def _target(target: GraspTarget) -> tuple:
    return (
        tuple(float(v) for v in target.position),
        target.tilt,
        target.jaw_heading,
        target.lean_azimuth,
    )


def _solution(found: dict) -> IKSolution:
    return IKSolution(
        q=torch.tensor(found["q"], dtype=torch.float64),
        margin=found["margin"],
        position_error=found["position_error"],
        tilt_error=found["tilt_error"],
        heading_error=found["heading_error"],
        azimuth_error=found["azimuth_error"],
        step=found["step"],
    )


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

    def update(
        self,
        desired: torch.Tensor,
        measured: torch.Tensor,
        saturated: torch.Tensor | None = None,
    ) -> None:
        """Fold one control step's tracking error into the trim.

        `saturated` is a per-joint mask of actuators whose effort limit is currently
        binding. On those joints the trim does NOT integrate, which is the anti-windup this
        class was missing.

        Why it matters is ARITHMETIC, not a measurement, and that distinction is deliberate:
        a trim at its 0.35 rad clamp on `link_003_joint`, whose kp is 150, commands
        150 x 0.35 = 52.5 N.m into a joint rated 17 N.m. A saturated joint does not move in
        response to more command, so integrating against it only inflates a demand that is
        already being refused, and the integrator then has to unwind before the joint can
        track again.

        WHAT THIS DOES *NOT* REST ON, because the evidence is dead. An earlier version cited
        "56 to 79 N.m of demand past the rating, sustained (Task D3 part 11)". That number
        was sampled from ArticulationData while the arm was in the Nyquist-rate chatter that
        `robots.maker_arm`'s ARM_ARMATURE comment describes, so it was reading opposite
        phases of a period-2 oscillation; every parts 11-13 torque sample is a casualty. On
        the quiet arm the lever is INERT: re-running the 10-trial gate with this mask on and
        off (Task D3 part 15) gave 9/10 both ways with 44 of 45 phase lines byte-identical,
        because nothing saturates during a normal grasp.

        It is kept anyway, and the honest reason is that an integrator without anti-windup is
        a latent failure whatever a passing gate says -- the arithmetic above is true whenever
        a joint does saturate. It costs nothing measurable today. Do not re-justify it with a
        torque trace from parts 11-13.

        Clamping alone is not anti-windup: it bounds the trim's magnitude but still lets it
        run all the way to the bound while the joint is immobile. Conditional integration
        is what stops it.
        """
        desired = torch.as_tensor(desired, dtype=torch.float64).reshape(ARM_DOF)
        measured = torch.as_tensor(measured, dtype=torch.float64).reshape(ARM_DOF)
        stepped = torch.clamp(
            self.value + self.gain * (desired - measured), -self.limit, self.limit
        )
        if saturated is None:
            self.value = stepped
            return
        saturated = torch.as_tensor(saturated, dtype=torch.bool).reshape(ARM_DOF)
        self.value = torch.where(saturated, self.value, stepped)


def steepest_jaw_roll(jaw_axis: torch.Tensor, tool_axis: torch.Tensor) -> float:
    """Rotation about the tool axis (rad) that stands the jaw plane up as far as it goes.

    MS6 rung 2. RELEASE holds the pose EXTEND left, and the wrist roll is the one degree of
    freedom nothing commands there: ``extend_goal`` passes no jaw heading, so the plane the
    jaws open in is inherited from the heading the PICK happened to need. Measured over one
    instrumented episode it stood at 5.3, 56.4, 78.0 and 71.6 degrees off horizontal on four
    consecutive releases. The near-horizontal one dropped its part 13.3 mm in ONE control
    step, a free fall from the whole release height; the steep ones moved 1.3 mm, the part
    sliding off the lower plate into the bin.

    The roll cannot be asked for through ``GraspTarget.jaw_heading``. That row constrains the
    jaw axis's HORIZONTAL heading, and a steep axis has almost no horizontal component, so
    the row goes singular exactly where it is needed -- the same failure the lean azimuth has
    at zero tilt. Driven, commanding the lean azimuth as the jaw heading produced a 2.6 degree
    plane and commanding its quarter turn produced 0.00007 degrees. This function answers the
    question in the non-singular parameter instead, the roll itself, which the wrist's last
    joint drives exactly: ``link_007_joint`` turns about the tool axis, so rolling it moves
    `grasp_center` by 0.00 mm and the tool axis by 0.00 degrees while the jaw axis follows
    one for one (measured 2026-09-17, ratio -1.000).

    Both signs of the jaw axis are the same jaw, so the nearer of the two is returned and the
    result is always within a quarter turn.
    """
    t = tool_axis / tool_axis.norm()
    e1 = jaw_axis - (jaw_axis @ t) * t
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(t, e1)
    # v(d) = cos(d) e1 + sin(d) e2 sweeps the plane the jaw axis lives in; its z component is
    # extreme where its derivative vanishes, which is this angle.
    delta = math.atan2(float(e2[2]), float(e1[2]))
    # Fold in plain float: routing through a default-dtype tensor costs float32 and puts the
    # quarter-turn bound a rounding step out of reach.
    if delta > math.pi / 2:
        delta -= math.pi
    elif delta <= -math.pi / 2:
        delta += math.pi
    return delta
