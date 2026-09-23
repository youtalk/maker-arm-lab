"""GPU binding of the core oracle: per-env global-IK plans driven as joint targets.

`core.oracle` is isaaclab-free by design and only says WHAT to do next (a Goal). This
module owns HOW: it turns that Goal into a joint-space plan through `core.global_ik`,
commands one waypoint of that plan per control step onto the env's joint-position action
term, reads the frames (the tool body (`grasp_center`) pose, brick poses, the filtered
finger<->brick contact sensors) the oracle's SceneSnapshot needs, and applies the two
measured constants core.oracle deliberately leaves to the runtime
(DESCEND_CLEARANCE_MEASURED, PICK_OFFSET_MEASURED).

WHY A PLANNER AND NOT A SERVO -- this module used to servo `Goal.target_pos` through the
IK-Rel action term's differential IK, and Task D5 moved it onto `core.global_ik` together
with `MakerArm-LegoSort-Datagen-v0`'s action space (datagen.LegoSortDatagenEnvCfg). Two
reasons, and neither of them is "the differential IK never works" -- Task D4 drove this same
class through a full pick and place at a fixed easy pose, closing the jaws 40/40 with
`pinch_ok=True`:

  1. ACTION-SPACE CONSISTENCY. The dataset's recorded action stream is the post-IK
     joint-position target (`mdp.recorders.JointTargetRecorder`), so the oracle now emits
     exactly what a policy must output, with no solver between the demonstration and the
     label. The global solver emits JOINT TARGETS and the IK-Rel term reads its input as
     CARTESIAN DELTAS, so the module and the env cfg had to move as one change.
  2. BAND-WIDE PERFORMANCE. Over the randomised scatter band Task D3 measured the
     differential IK spending `link_005_joint`'s margin instead of turning the base, after
     which the arm is jammed and every later phase servos a dead mechanism; its grasp gate
     never passed until the check script moved to `core.global_ik`, which then took it to
     10/10. A global solve has no such failure mode: it solves the pose in one shot and
     returns the branch furthest from the stops.

The recipe here is the one `scripts/scripted_grasp_check.py` validated, applied to every
phase rather than to the pick alone:

  * the free-space repositioning leg (HOVER) is a GLOBAL solve of the endpoint plus a
    joint-space interpolation to it -- that is what turns the base, because the branch is
    chosen by joint margin rather than walked into;
  * every leg that approaches or carries (DESCEND/LIFT/RAISE/EXTEND/RETREAT) is a Cartesian
    waypoint path through `GlobalIK.solve_path`, whose continuity constraint is the guard
    against an IK branch flip mid-carry;
  * the descend runs `core.global_ik`'s tilt ladder near level, per ruling D3-10: at
    GRASP_TILT_RAD the jaw plates foul the desk at every pick offset, and each rung carries
    its own pick offset;
  * a `core.global_ik.JointTrim` per env biases the command against the arm's open-loop
    droop, which is effort-limit saturation on `link_004_joint`/`link_007_joint` and not
    missing gravity compensation (Task D3 part 9).

There is no attitude servo any more. `core.attitude_hold.AttitudeHold` existed because the
IK-Rel term re-bases the desired orientation on the measured one every step, so a zero
rotation delta does not hold an attitude; the solver constrains tilt, lean azimuth and jaw
heading directly, so the attitude is a property of the joint vector being commanded. For the
same reason a hold phase now genuinely holds: it keeps commanding the same joint target,
where a zero IK-Rel delta ratcheted gravity sag into the setpoint at ~0.9 mm/step.

Frame convention: every position this class hands to core.oracle (SceneSnapshot fields,
Goal.target_pos) and every position it reads back for planning (``pick_point``) is
env-origin-subtracted -- SceneSnapshot's own docstring ("world frame, env-origin
subtracted"). That is what makes a leg error (target - measured) identical to the same
subtraction done in raw world coordinates: the per-env origin cancels in the difference
either way, so picking the origin-local convention throughout is free, and it is required
for the goals core.oracle builds from constants that are the same for every env (bin
centers, RETREAT_POINT) to mean the same physical pose in every env.
"""

import math
import random

import torch

from maker_arm_lab.core import grasp, layout, oracle, trajectory
from maker_arm_lab.core.action_scale import IK_SCALE
from maker_arm_lab.core.checker import (
    HELD_MIN_FORCE,
    PINCH_MAX_ASYMMETRY,
    PINCH_MIN_FORCE,
    PINCH_MIN_STALL,
    held_brick_from_forces,
    pinch_min_force,
    pinch_min_stall,
)
from maker_arm_lab.core.global_ik import (
    ARM_DOF,
    GlobalIK,
    GraspTarget,
    JointTrim,
    descend_pick_offset,
    descend_tilt_ladder,
    fold_to_half_turn,
    steepest_jaw_roll,
    tool_axis_world,
)
from maker_arm_lab.core.heap_oracle import target_context
from maker_arm_lab.core.oracle import LIFT_MIN_RISE, PHASE_TO_INT
from maker_arm_lab.core.quat import quat_apply
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.robots.frames import (
    ARM_JOINT_NAMES,
    BASE_YAW_SIGN,
    DESCEND_CLEARANCE_MEASURED,
    GRASP_TILT_RAD,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    JAW_AXIS_BODY,
    JAW_FULL_GAP,
    PICK_OFFSET_MEASURED,
    TOOL_AXIS_BODY,
    TOOL_BODY,
    WRIST_ROLL_INDEX,
    WRIST_ROLL_SIGN,
)

# --- The grasp attitude ------------------------------------------------------
# The tool leans toward world +x. Task D3 part 8 measured that this shipping convention is
# also the one to keep: freeing the lean azimuth covers band corners it cannot reach but
# GRASPS WORSE (4/9 against 8/9), because a redundancy resolved for joint margin picks poses
# that are bad for the grasp. Stated here rather than imported because
# `scripts/scripted_grasp_check.py` states its own; both cite core.global_ik, which carries
# the measurement.
LEAN_AZIMUTH = 0.0

BRICK_HALF_HEIGHT = layout.BRICK_SIZE[2] / 2
# Ruling D3-10 / D3-11 candidate 2: the descend is near level, and rather than one fixed
# tilt the runtime tries the ladder in order and takes the smallest tilt whose descend path
# actually solves. Every rung is inside the jaw-plate fouling window by construction
# (core.global_ik.descend_tilt_ladder), so buying path feasibility with tilt can never buy
# it by putting the plates into the desk.
DESCEND_TILT_LADDER = descend_tilt_ladder(BRICK_HALF_HEIGHT)

# Restart spread for the one solve per pick attempt that has no previous waypoint to seed
# from (the HOVER endpoint). core.global_ik's own default: it is the number that decides
# which BRANCH the approach lands on, and Task D3 measured that a spread of this size is
# what makes the solver turn the base instead of walking a joint into its stop.
# `scripts/scripted_grasp_check.py` uses 64 for the same solve because THERE the same call
# also decides whether a brick is reported UNREACHABLE, and calling a pose infeasible
# because the search was thin would blame the layout for the solver's own weakness. The
# oracle renders no such verdict -- a hover it cannot solve fails the pick and retries -- and
# 24 costs about a third of the wall clock, paid once per pick attempt in every generated
# episode.
HOVER_RESTARTS = 24

# Hover heights above the brick (m) the HOVER leg falls back to, in order, when the drawn
# height gives no pick: no IK solution for the hover itself, or a hover pose from which the
# descend ladder has no continuous path. The drawn height is always tried first, so a pick
# that plans at it is bit-for-bit unaffected.
#
# MEASURED on GPU (2026-09-07, instrumented 12-env batch): both grasp-retry abandons of the
# batch were picks the arm could not plan at the DRAWN height and retried three times
# against the same target -- a band-edge brick (radius 0.274 m, azimuth +40 deg) with no
# hover solution at 0.138 m at any restart count, and a near brick (radius 0.146 m) whose
# hover solved at 0.129 m only on a branch with the wrist 0.18 rad from its stop, from which
# no descend rung was walkable. Replayed on CPU with the runtime's own planner, both plan at
# 0.10 m (hover margin 0.34 rad and 0.28 rad; descend on the first rung) and at every lower
# rung. The hover height is a via pose and a dataset-diversity draw
# (`core.oracle.OracleNoise.hover_height`), not a grasp parameter: the descend commands the
# brick's own z whatever the hover was, so lowering it costs the dataset a little variety
# on the affected picks and nothing else. The floor of 0.06 m keeps the jaw plates -- which
# hang JAW_FACE_FAR * sin(GRASP_TILT_RAD) = 21 mm below the tool tip at the hover tilt --
# well clear of neighbouring bricks (MIN_SEPARATION 0.06 m).
HOVER_HEIGHT_LADDER_M = (0.10, 0.08, 0.06)

# --- Leg resolution ----------------------------------------------------------
# How finely a leg is cut into waypoints, and how many control steps each waypoint is
# commanded for. 3 mm at 2 steps is ~1.5 mm per control step, the pace the IK-Rel servo
# actually achieved on the same legs (D4 measured EXTEND converging in 203-222 steps over
# ~0.3 m), so the phase budgets in core.oracle -- which are NOT modified, spec section 5
# item 5 -- still fit. The tilt and joint entries are the same idea for legs that rotate the
# tool or move in joint space: RAISE applies the whole of RELEASE_TILT_RAD, and a leg cut
# for position alone would exceed core.global_ik's continuity cap on the rotation.
WAYPOINT_STEP_M = 0.003
WAYPOINT_TILT_STEP_RAD = 0.03
WAYPOINT_JOINT_STEP_RAD = 0.02
WAYPOINT_HOLD_STEPS = 2
# Fraction of a phase's own step budget the walk may consume, leaving the rest for the arm
# to close the tracking error the walk leaves behind (the trim's time constant is ~33 steps;
# see core.global_ik.JointTrim).
WALK_BUDGET_FRACTION = 0.6
# The staircase walked one waypoint spacing per WAYPOINT_HOLD_STEPS control steps; these are
# that pace in physical units per second, the unit `transport_speed` multiplies. At 1.0 a
# leg takes the time the staircase took, but under core.trajectory's minimum-jerk law: the
# target moves on every step, from rest to rest, peaking at 1.875x this speed mid-leg.
# DESCEND always walks at 1.0: it is stall-gated and grasp-critical.
LINEAR_PACE_M_PER_S = WAYPOINT_STEP_M * trajectory.CONTROL_HZ / WAYPOINT_HOLD_STEPS
TILT_PACE_RAD_PER_S = WAYPOINT_TILT_STEP_RAD * trajectory.CONTROL_HZ / WAYPOINT_HOLD_STEPS
JOINT_PACE_RAD_PER_S = WAYPOINT_JOINT_STEP_RAD * trajectory.CONTROL_HZ / WAYPOINT_HOLD_STEPS
# Acceleration cap as a multiple of the leg's peak speed (1/s): the ramp to peak speed takes
# about a quarter of a second at 4.0. It binds only on legs of a few centimetres.
WALK_ACCEL_PER_PEAK_SPEED = 4.0
TRANSPORT_PHASES = frozenset(
    {
        oracle.Phase.HOVER,
        oracle.Phase.LIFT,
        oracle.Phase.RAISE,
        oracle.Phase.EXTEND,
        oracle.Phase.RETREAT,
    }
)

# Jaw-heading offsets a CARRY leg may end at when the held heading gives no walkable path,
# tried in this order (see `OracleRig._plan_carry` for the measurement and for why moving the
# heading with a brick in the jaws costs nothing). The held heading is always first, so a leg
# that solves as commanded is bit-for-bit unaffected by this ladder; the widest rung is 60
# deg because the measured failures needed up to 60 deg and none needed more.
HEADING_RELAXATION_RAD = (
    0.0,
    *(sign * math.radians(deg) for deg in (10.0, 20.0, 30.0, 45.0, 60.0) for sign in (-1.0, 1.0)),
)

# Largest joint jump a leg may take when the Cartesian path fails and the fallback solves
# the endpoint directly. That fallback is seeded from the current pose, so it normally stays
# on the same branch; this rejects the case where it does not, because a branch flip with a
# brick in the jaws flings the brick. A whole leg legitimately moves the base and shoulder by
# a few tenths of a radian (D4: EXTEND's base swing to the outer bins is under 0.5 rad); an
# elbow or wrist inversion is a multiple of that.
MAX_LEG_JOINT_JUMP = 1.5

# Per-step motion of the pick point below which a stall-gated phase (DESCEND) is treated as
# blocked. A walked waypoint delivers ~1.5 mm/step, so this is far below any real motion and
# above solver noise. Carried over unchanged from the servo version, where full deflection
# delivered ~1.7 mm/step.
STALL_EPS = 5e-5

# Attitude error (rad) at which a phase's attitude counts as arrived. NOT a phase-end gate
# any more, and that is a deliberate change: the servo version gated convergence on a
# rotation channel steered independently of the position channel, which could lag it
# arbitrarily, while under the global solver the attitude and the position come out of the
# SAME joint vector -- arriving at the commanded joint target in position IS arriving in
# attitude, up to joint tracking error. It is kept as the threshold the check scripts report
# `rot_err` against (`scripts/release_envelope_check.py`'s `extend_ok`), where it still says
# whether the arm held the attitude it was asked for.
ROT_CONVERGED = 0.02

# Pinch evidence (gated after LIFT, in free air, so the readings are pure finger<->brick and
# not the desk press): a real pinch stalls the jaws short of GRIPPER_CLOSED_POS and puts
# equal, non-trivial filtered force on both fingers; entrainment (brick merely cammed in the
# jaw pocket) rises and centers identically but does neither.
#
# All three thresholds are imported from core.checker above, and THIS REPO is now their
# authority -- not OpenYAM's script. PINCH_MIN_STALL in particular is no longer OpenYAM's
# 0.0087 m nor the 0.003 m this module used to define locally: it is derived from THIS brick
# as min(BRICK_SIZE)/2 = 0.0048 m, and core.checker carries the derivation. PINCH_MIN_FORCE
# was validated against Task D3's 10/10 gate on this arm and closed; PINCH_MAX_ASYMMETRY
# remains inherited from OpenYAM. Read core.checker for which is which.
# LIFT_MIN_RISE (0.05 m) lives in core.oracle, next to LIFT_HEIGHT, so the torch-only heap
# loop reads the same rise gate (phase B design 5.5).
LIFT_MAX_XY_ERROR = 0.03  # m

# `resume` calls a brick pinched at core.checker.HELD_MIN_FORCE (imported above), the
# attribution threshold the eval loop and the privileged channel read it at. It is under
# PINCH_MIN_FORCE on purpose: a takeover on a weak pinch still enters LIFT, whose gate then
# decides.

# Idle action for an env whose oracle has nothing left to do this step (DONE or ABANDON):
# jaws open, arm holding its last commanded joint target. The arm channel is NOT zeroed --
# on the joint-position action space a zero arm action commands every joint to fold to zero,
# it does not ask the arm to stand still.
GRIPPER_OPEN_ACTION = 1.0

# These legs hold the jaws at the grasp's own opening instead of driving them fully open, for
# an oracle that asks for it (`EnvOracle.matched_jaw_opening`). CLOSE already closes, and
# REOPEN must open the whole way because its retry has to clear a part the jaws may still be
# fouling.
#
# HOVER and DESCEND bring the tips down over the part (MS6 rung 1): the fully open landing
# spots 52.4 mm out refused 84.1 % of in-dish part sightings on the three read heaps.
#
# RELEASE lets the part go (MS6 rung 2). Opening the whole way there is about 44 mm of travel
# per jaw from a pinched grip, at roughly 3.3 mm per control step, with the tool tilted
# 80.2 deg over the bin, and it THROWS the part: 18 of 73 carried parts never reached the bin
# on the rung 1 heaps, every traced one landing on the desk 60-200 mm short of the release
# point. The part only needs `JAW_APPROACH_CLEARANCE` per side to fall out.
MATCHED_OPENING_PHASES = (oracle.Phase.HOVER, oracle.Phase.DESCEND, oracle.Phase.RELEASE)

# The wrist roll EXTEND ends at, so RELEASE opens in a plane the part slides off rather than
# one it free-falls from (`OracleRig._roll_waypoints`). The deadband keeps a roll that is
# already steep from spending waypoints; the step cap keeps the whole turn, up to a quarter
# of a revolution, from arriving inside one continuity step, the way every other leg is
# walked rather than stepped.
ROLL_DEADBAND_RAD = math.radians(5.0)
ROLL_MAX_STEP_RAD = math.radians(6.0)
# The continuous gripper releases with the jaw plane LEVEL instead (phase B release rung): a
# position-held jaw opens smoothly, and a part resting on the lower jaw of a stood-up plane
# stays there -- the jaw face slopes about 10 degrees against a sliding friction of 1.0.
# Traced on the corpus heaps, 20 to 26 of about 49 releases per heap still held their part.
# A level plane has no jaw under the part. On the corpus seeds p0..p3 (K=50) it stows 180 of
# 200 against 158 on the stood-up plane; p4..p7 read 188 of 200, binary 190. A first probe
# that rolled level without the unroll lost RETREATs: planned from the rolled wrist, 3 to 13
# per heap came back unplanned, and two episodes were abandoned. RETREAT now turns the wrist
# back along the same waypoints before it plans. The margin keeps the level roll off the wrist
# stop; 10 degrees is chosen, not swept.
LEVEL_ROLL_MARGIN_RAD = math.radians(10.0)

# MS6 rung 3. The legs that CARRY hold the jaws at the part's own width instead of driving
# them to `GRIPPER_CLOSED_POS`, for an oracle that asks for it.
#
# A binary close command runs the jaws all the way shut, and nothing stops them once a thin
# part starts to slip. A 9.6 mm brick stalls them at 5.9 mm on a 15.8 mm grasp and is held.
# A 3.2 mm plate is not: traced per control step on a LONE plate_2x4 with nothing around it,
# CLOSE ends with 2.9 N on both jaws and the part in the grip, and LIFT then keeps closing --
# 5.18, 4.81, 4.21, 3.70, 3.19, 2.69 mm -- while the force bleeds off 2.9, 1.0, 0.9, 0.33,
# 0.00 N. The plate is squeezed out like a pip and the jaws end at 0.02 mm holding nothing.
# RELEASE is not here: it opens, and MATCHED_OPENING_PHASES already bounds that.
MATCHED_CLOSE_PHASES = (
    oracle.Phase.CLOSE,
    oracle.Phase.LIFT,
    oracle.Phase.RAISE,
    oracle.Phase.EXTEND,
)
# Per-step joint jitter (rad) that one raw unit of `EpisodeDraw.action_noise_std` buys.
# core.oracle calibrates its noise in the raw units of the action space this module used to
# drive, where a raw unit was IK_SCALE metres of tool motion in one control step; the joint
# targets that replaced it are in radians, so a literal reading of "raw units" would inflate
# the disturbance by two orders of magnitude. This converts through the tool instead, at the
# Jacobian column magnitude Task D3 part 7 measured on this arm (0.23-0.27 m/rad), so the
# per-step disturbance keeps the PHYSICAL size the noise draw was tuned at -- 0.1 mm of tool
# motion per step at the default std of 0.05, times core.oracle's induction boost.
JACOBIAN_M_PER_RAD = 0.25
JOINT_NOISE_PER_RAW_UNIT = IK_SCALE[0] / JACOBIAN_M_PER_RAD

# Consecutive watchdog-flagged control steps before an env is abandoned. 30 steps at 30 Hz
# is 1 s -- comfortably longer than the transient joint-limit contact a converging arm can
# produce, short enough not to burn an episode's budget on a genuinely runaway env. Carried
# over unchanged from OpenYAM; the reasoning below is OpenYAM's own measurements, on
# OpenYAM's arm, not yet re-validated on the Maker Arm.
#
# Since OpenYAM's Task 8, the strike counter resets to 0 on every phase entry (see
# _enter_phase below), not just once per episode. Two consequences worth knowing before
# retuning either this or a phase's own step budget:
#
# * core.oracle.RAISE_MAX_STEPS (60) depends on this value: RAISE is only guaranteed to exit
#   on "budget" -- not get abandoned by the watchdog mid-phase -- if RAISE_MAX_STEPS <=
#   WATCHDOG_TRIP_STRIKES + first_contact_step, where first_contact_step (measured 30-40
#   control steps into RAISE, on OpenYAM's Task 8 GPU runs) is how soon a genuine
#   reach-ceiling contact starts flagging "near" a joint limit. Raising
#   WATCHDOG_TRIP_STRIKES without re-checking this bound is safe (more margin); lowering it,
#   or raising RAISE_MAX_STEPS, is not.
# * core.oracle.REOPEN (hold_steps=20) and RELEASE (hold_steps=20) both run fewer control
#   steps than this threshold, so with a per-phase reset neither phase can EVER trip the
#   watchdog on its own, regardless of how close to a joint limit the arm is throughout -- a
#   real gap versus OpenYAM's pre-Task-8 persistent counter, not just a smaller window.
#   Measured (OpenYAM's Task 8) this is latent, not exercised. Left undefended deliberately
#   rather than given special-cased carry-over, because REOPEN/RELEASE are pure-hold phases
#   (Goal.target_pos is None) -- they never themselves command motion toward a limit, only
#   hold whatever joint target the PRECEDING active phase left -- and the alternative
#   (carrying strikes over for hold phases only, resetting for active ones) is more state to
#   keep consistent for a case measurement shows is not currently live there.
WATCHDOG_TRIP_STRIKES = 30
# A step near a joint stop counts as a strike only if the arm is NOT where it was told to be:
# the solver accepts an end pose with any non-negative margin, and the minimum-jerk walk
# decelerates into such a pose over more than WATCHDOG_TRIP_STRIKES steps (measured on
# MuJoCo: a HOVER ending 0.0008 rad from link_003_joint's stop was abandoned on the last step
# of its walk under the bare near-limit rule). Tracking error at rest is under 0.002 rad; an
# arm pinned on a stop by a command it cannot follow is off by far more. Speed and past-the-
# limit strikes are unchanged.
WATCHDOG_TRACKING_TOL_RAD = 0.02


def grasp_center_for(pick, tilt: float, heading: float, offset_z: float) -> tuple:
    """The `grasp_center` position that puts the pick point at ``pick``.

    ``pick = grasp_center + R @ (ox, 0, oz)``, so
    ``grasp_center = pick - ox * (body x in world) - oz * (body z in world)``. Body z maps to
    the negated tool axis, which the commanded attitude gives exactly. Body x is the
    plate-face direction: it is ``body y x body z``, and since body y is the jaw axis at
    world heading ``heading``, body x points at ``heading - 90 deg`` in the horizontal plane,
    exactly at zero tilt and to within the tilt otherwise. At the near-level descend tilts
    that approximation is worth ``|ox| * sin(tilt)``, which is 0.0 on this arm because
    ``PICK_OFFSET_MEASURED[0]`` is 0.0 (robots.frames records why that is a measurement and
    not an unset default).

    Module-level, and not a method, because it is pure geometry: the inverse of
    ``OracleRig.pick_point``, and a sign error in it would put the jaws a tool-length off the
    brick in a way only a GPU run could see. Same derivation as
    ``scripted_grasp_check.GraspRig.grasp_center_for``, which is where it was measured.
    """
    axis = tool_axis_world(tilt, LEAN_AZIMUTH)
    face = torch.tensor([math.sin(heading), -math.cos(heading), 0.0], dtype=torch.float64)
    point = torch.as_tensor(pick, dtype=torch.float64).reshape(3)
    return tuple(float(v) for v in point + offset_z * axis - PICK_OFFSET_MEASURED[0] * face)


def leg_resolution(
    max_steps: int,
    gain_scale: float,
    distance: float = 0.0,
    tilt_change: float = 0.0,
    heading_change: float = 0.0,
    joint_change: float = 0.0,
    speed_scale: float = 1.0,
) -> tuple[int, int]:
    """How many waypoints a leg gets, and how many control steps its walk takes.

    Resolution comes from the geometry (position, tilt, joint travel) so the waypoints stay
    inside core.global_ik's continuity cap. Pace comes from the staircase's own (the
    ``*_PACE_*`` constants) times ``EpisodeDraw.gain_scale`` -- the per-episode approach-speed
    draw, which used to scale the servo's proportional gain -- times ``speed_scale``, under
    core.trajectory's minimum-jerk law with its acceleration cap. Both are held to
    ``WALK_BUDGET_FRACTION`` of the phase's ``max_steps``, so a slow draw cannot spend a
    budget core.oracle sized for a faster one: a leg too long for its budget is cut coarser
    and walked faster rather than truncated, because the arm has to reach the end of the
    leg, and a truncated plan would leave every later phase starting from the wrong pose.
    """
    count = max(
        2,
        math.ceil(distance / (WAYPOINT_STEP_M * gain_scale)),
        math.ceil(tilt_change / (WAYPOINT_TILT_STEP_RAD * gain_scale)),
        math.ceil(heading_change / (WAYPOINT_TILT_STEP_RAD * gain_scale)),
        math.ceil(joint_change / (WAYPOINT_JOINT_STEP_RAD * gain_scale)),
    )
    budget = max(2, int(WALK_BUDGET_FRACTION * max_steps))
    pace = gain_scale * speed_scale
    seconds = max(
        trajectory.walk_seconds(quantity, unit * pace, WALK_ACCEL_PER_PEAK_SPEED)
        for quantity, unit in (
            (distance, LINEAR_PACE_M_PER_S),
            (tilt_change, TILT_PACE_RAD_PER_S),
            (heading_change, TILT_PACE_RAD_PER_S),
            (joint_change, JOINT_PACE_RAD_PER_S),
        )
    )
    # `- 1e-6`: 0.12 m at 0.045 m/s is 80 steps, not the 81 float rounding would give.
    steps = max(2, math.ceil(seconds * trajectory.CONTROL_HZ - 1e-6))
    return min(count, budget), min(steps, budget)


class OracleRig:
    """Per-env GPU binding of ``core.oracle.EnvOracle`` onto batched joint-position actions.

    One ``core.oracle.EnvOracle`` and one ``core.global_ik.JointTrim`` per env; one shared
    ``core.global_ik.GlobalIK``, which keeps no state between solves. ``step()`` advances
    every env by one control step; ``rearm()`` re-seeds the envs a caller has just reset.

    A phase that commands motion is PLANNED once, at phase entry, into a list of joint
    waypoints, which ``step()`` then walks. Solving is therefore paid per phase and not per
    control step: a seeded waypoint solve is a couple of milliseconds and the one global
    HOVER solve a few hundred, against roughly fifty phase entries in a six-brick episode.
    """

    def __init__(
        self,
        env,
        bl: layout.BinLayout,
        noise: oracle.OracleNoise,
        rng: random.Random,
        induction: oracle.FailureInduction | None = None,
        transport_speed: float = trajectory.TRANSPORT_SPEED,
        oracle_factory=None,
    ):
        self.env = env
        self.gripper_mode = getattr(env, "gripper_mode", "binary")
        # Multiplier on the pace of the transport legs (TRANSPORT_PHASES); see leg_resolution.
        self.transport_speed = transport_speed
        self.bl = bl
        self.noise = noise
        self.rng = rng
        self.induction = induction
        self.robot = env.scene["robot"]
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_bricks = env.cfg.num_bricks
        self.tool = self.robot.find_bodies(TOOL_BODY)[0][0]
        self.finger_ids = self.robot.find_joints(list(GRIPPER_JOINT_NAMES))[0]
        # URDF order (link_002_joint .. link_007_joint), preserved rather than accepted from
        # the articulation, because every joint vector crossing this boundary -- the solver's
        # `q`, the joint limits, WRIST_ROLL_INDEX, the action term's own columns -- is in
        # that order, and a permutation between them would be silent.
        self.arm_ids = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        self._check_action_space(env)
        self.ex = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.jaw_axis = torch.tensor(JAW_AXIS_BODY, device=self.device)
        self.tool_axis = torch.tensor(TOOL_AXIS_BODY, device=self.device)
        self.tool_axis_body = torch.tensor(TOOL_AXIS_BODY, dtype=torch.float64)
        self.ik = GlobalIK()
        # Joint limits read once: identical for every env (same robot asset replicated), so
        # env 0's row is every env's row. The articulation's, not the URDF's, so the command
        # clamp and the watchdog agree with what the simulator will accept.
        self.limits = self.robot.data.joint_pos_limits[0, self.arm_ids].clone()  # (6,2)
        # Plain CPU floats, read once: the jaw-heading fold needs them on every phase entry
        # and a per-call .tolist() would be a device sync.
        self.wrist_roll_limits = tuple(self.limits[WRIST_ROLL_INDEX].tolist())
        self.left_brick = env.scene["finger_left_brick"]
        self.right_brick = env.scene["finger_right_brick"]

        self.oracles: list[oracle.EnvOracle | None] = [None] * self.num_envs
        self.goals: list[oracle.Goal | None] = [None] * self.num_envs
        self.trims = [JointTrim() for _ in range(self.num_envs)]
        # The joint target being commanded right now, float64 CPU, URDF order. This is the
        # arm's setpoint between waypoints and through every hold phase.
        self.q_cmd = [torch.zeros(ARM_DOF, dtype=torch.float64) for _ in range(self.num_envs)]
        # The current phase's plan (joint waypoints, emptied when the walk completes), how
        # many control steps its walk takes, and how many it has taken.
        self.plans: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self.walk_steps = [1] * self.num_envs
        self.walk_step = [0] * self.num_envs
        self.unroll: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_envs
        # Where a desk-resting brick sits in `grasp_center` coordinates. Per env because the
        # descend ladder picks its own offset per rung (the offset balancing desk clearance
        # against tip margin moves with tilt), and that choice then has to survive into the
        # LIFT/EXTEND legs, which read `pick_point` in the same frame.
        if PICK_OFFSET_MEASURED[1] != 0.0:
            # `_grasp_center_for` inverts the offset along the tool axis and the plate-face
            # direction only; a jaw-axis component would need the jaw axis too. It is 0.0 by
            # measurement (robots.frames), so this guards a future edit, not a live case.
            raise NotImplementedError(
                "PICK_OFFSET_MEASURED has a jaw-axis component; _grasp_center_for inverts "
                "only its tool-axis and plate-face components"
            )
        self.pick_offset = torch.tensor(PICK_OFFSET_MEASURED, device=self.device).repeat(
            self.num_envs, 1
        )
        # Last explicit jaw-heading target per env (the folded closing-axis heading), reused
        # whenever a goal's yaw is None so a held phase (CLOSE/LIFT/RAISE/...) keeps the
        # heading DESCEND last commanded -- there is no brick reference once the brick has
        # left the desk. `commanded_tilt` is the same for the tool-axis tilt.
        self.commanded_yaw = [0.0] * self.num_envs
        self.commanded_tilt = [GRASP_TILT_RAD] * self.num_envs
        # Brick pose at the HOVER this pick attempt started from (env-origin subtracted), for
        # the LIFT rise/xy gate -- mirrors grasp_one's `brick_start` in
        # scripts/scripted_grasp_check.py.
        self.brick_start: list[torch.Tensor | None] = [None] * self.num_envs
        # Pick point one step ago, for DESCEND's stall detector.
        self.previous_pick_point: list[torch.Tensor | None] = [None] * self.num_envs
        self.steps_in_phase = [0] * self.num_envs
        self.quiet = [0] * self.num_envs
        # Post-arrival settle, LIFT only (Goal.hold_steps > 0 with a target_pos set):
        # scripts/scripted_grasp_check.py reads pinch/rise after a hold, not at the instant
        # the arm arrives, so the contact forces have settled. `holding[i]` marks that this
        # env has arrived/stalled/budgeted-out and is now coasting the extra
        # `goal.hold_steps` steps before the gate is read; `pending_outcome` is the outcome
        # the leg actually ended on.
        self.holding = [False] * self.num_envs
        self.hold_counter = [0] * self.num_envs
        self.pending_outcome: list[str | None] = [None] * self.num_envs
        self.watchdog_strikes = [0] * self.num_envs
        # Set when a leg could not be planned at all, so `_phase_result` can end the phase at
        # once rather than let the arm hold still until the budget expires.
        self.unplanned = [False] * self.num_envs
        # (N, B) bool: which bricks were at rest in their own colour's bin at the start of the
        # most recent control step. Refreshed in `step()`; read by `episode_stats`.
        self._stowed_sample = torch.zeros(
            self.num_envs, self.num_bricks, dtype=torch.bool, device=self.device
        )
        # Per-env outcome, frozen the step an episode ends. The two fields in it are read
        # from state the ENV owns and overwrites on the next step -- for every env in the
        # batch, whether or not that env was rearmed -- so unlike the counters in
        # `self.stats` they cannot simply be read later. See `episode_stats`.
        self._outcome: list[dict | None] = [None] * self.num_envs
        self.abandoned = torch.zeros(self.num_envs, dtype=torch.bool)
        self.stats = [self._fresh_stats() for _ in range(self.num_envs)]
        # MS4 c2 design 6.1: an idle env is under another actor (the DAgger policy). Its row in
        # `actions()` is the coast row, its stash reads -1, and `observe()` leaves it alone.
        self.idle = [False] * self.num_envs
        # MS4 c2 design 6.2: a `resume()` takeover runs exactly one brick cycle. `single_cycle`
        # marks an env mid-takeover so `_advance_phases`/`_enter_phase` know to hand it back
        # at RETREAT instead of entering the next HOVER; `takeover_result` is None while that
        # takeover runs, then whether the cycle's brick reached its bin.
        self.single_cycle = [False] * self.num_envs
        self.takeover_result: list[bool | None] = [None] * self.num_envs
        self._planned: list[torch.Tensor] = []
        # MS6 design 7.5: who builds an env's oracle. The default is the six-brick EnvOracle;
        # the heap script installs HeapOracle without a rig subclass. Called by rearm and resume.
        self.oracle_factory = oracle_factory or self._default_oracle
        self.rearm(list(range(self.num_envs)))

    @staticmethod
    def _fresh_stats() -> dict:
        return {
            "retries": 0,
            # Pick-and-place cycles the oracle carried through to the end of RETREAT. An
            # ATTEMPT count, which is why it is not called anything like "sorted": a brick
            # dropped in transit completes its RETREAT and is counted here exactly like a
            # delivered one. `bricks_delivered` below is the count read off the bricks
            # themselves, and it is the one a consumer wants.
            "sort_attempts": 0,
            "abandon": None,
            "unplanned_legs": 0,
            # Legs that fell through the Cartesian planner to `_plan_direct`, whose path is
            # joint-space and so gives up the straight-line guarantee on the tool. Counted
            # because comments elsewhere (core.oracle's RAISE_CLEARANCE_Z) reason about what a
            # straight EXTEND leg clears, and that reasoning holds only for the legs that
            # actually plan straight.
            "direct_legs": 0,
            # RAISE legs that planned but were refused because EXTEND could not be planned
            # from where they end (see `_plan_leg`). Each is also counted in
            # `unplanned_legs`, since that is how the phase then ends.
            "raise_vetoed": 0,
            # HOVER legs that had to take a lower rung of HOVER_HEIGHT_LADDER_M than the
            # episode's drawn hover height, because the drawn one gave no pick.
            "hover_lowered": 0,
        }

    def _default_oracle(self, i: int, bl: layout.BinLayout, draw: oracle.EpisodeDraw):
        return oracle.EnvOracle(bl, self.num_bricks, draw, induction=self.induction, rng=self.rng)

    def _check_action_space(self, env) -> None:
        """Fail loudly unless this env exposes the 7-element joint-position action.

        What this rig emits is six ABSOLUTE joint targets plus a binary gripper. Fed to the
        IK-Rel term instead, those six numbers would be read as Cartesian deltas -- silently,
        with no error, producing a dataset of nonsense. That is exactly why
        `MakerArm-LegoSort-Datagen-v0` moved to the joint-position action space in the same
        change as this module (see datagen.LegoSortDatagenEnvCfg), so it is worth a check
        rather than a comment.
        """
        action_dim = env.action_manager.total_action_dim
        if action_dim != 7:
            raise RuntimeError(f"expected a 7-element joint-position action, got {action_dim}")
        term = env.action_manager.get_term("arm_action")
        names = list(getattr(term, "_joint_names", []))
        if names != list(ARM_JOINT_NAMES):
            raise RuntimeError(
                "the arm action term does not drive the six arm joints in URDF order "
                f"({list(ARM_JOINT_NAMES)}); got {names}. The joint targets this rig emits "
                "are ordered, so a permutation here would silently drive the wrong joints."
            )

    # -- control loop ----------------------------------------------------

    def actions(self) -> torch.Tensor:
        """The batch action for this control step. Refreshes the cycle stash and the stowed
        sample, so it must run once per step, before ``env.step``."""
        action = torch.zeros(self.num_envs, 7, device=self.device)
        self._planned = []
        for i in range(self.num_envs):
            goal = self.goals[i]
            if goal is None:
                # DONE/ABANDON/idle envs coast: jaws open, arm holding its last joint target.
                action[i, 6] = self._gripper_command(i, None)
            else:
                action[i, 6] = self._gripper_command(i, goal)
                self._advance_plan(i)
            target = self.q_cmd[i]
            self._planned.append(target)
            command = self.trims[i].command(target) + self._command_noise(i)
            # NOT clamped to the joint limits here, deliberately. The action term clamps the
            # target through maker_arm_rs and RECORDS that it did
            # (`mdp.actions.ClampedJointPositionAction.clamped` -> `obs/clamped`), and
            # `core.profile.sim_profile` gives that clamp the same URDF bounds this rig would
            # have used -- so a pre-clamp here changes nothing about what the arm is
            # commanded and everything about what the dataset can tell you. It made the
            # spec's clamp-fraction criterion (section 4.3: "a non-zero fraction means the
            # oracle demands targets the real arm would refuse") unable to fail: the trim's
            # bias and the per-step noise can both push a target past a stop, and with a
            # pre-clamp that happened silently, with nothing recording it.
            action[i, :6] = command.to(device=self.device, dtype=torch.float32)
        self._stash_cycle_state()
        # Ground truth for `episode_stats`, sampled BEFORE the step because env.step()
        # auto-resets a finished env before it returns -- reading brick poses afterwards
        # describes the NEXT episode's scatter. Device-side and never synced here; the one
        # host read is in episode_stats.
        self._stowed_sample = per_brick_stowed(self.env)
        return action

    def observe(self, dones: torch.Tensor) -> dict:
        """Read the env after ``env.step``: trims, the NaN guard, phase transitions, the
        watchdog, the outcome freeze. ``dones`` is ``(terminated | truncated).cpu()``."""
        self._update_trims(self._planned, dones)
        self._abandon_nan_envs(dones)
        self._advance_phases(dones)
        self._run_watchdog(dones)
        self._freeze_outcomes(dones)
        return {"dones": dones, "abandoned": self.abandoned.clone()}

    def step(self) -> dict:
        """One control step for the whole batch. Returns ``{"dones", "abandoned"}``."""
        action = self.actions()
        _, _, terminated, truncated, _ = self.env.step(action)
        # Isaac Lab's ManagerBasedRLEnv.step() resets terminated|truncated envs to their NEXT
        # episode's initial state before returning, so for those envs every read below
        # (joint_pos, brick_pos, pick_point, ...) would be post-reset noise unrelated to the
        # goal still sitting in self.goals[i]. dones is computed here, before the trim update
        # and _advance_phases/_run_watchdog, precisely so all three can skip those envs for
        # this step; their stats/goal freeze at the completed episode's values and stay
        # readable via episode_stats() until the caller calls rearm().
        return self.observe((terminated | truncated).cpu())

    def _freeze_outcomes(self, dones: torch.Tensor) -> None:
        """Snapshot the outcome of every env that just finished, before the env moves on.

        Last thing in ``step()``, so a watchdog abandon raised earlier in this same step is
        included. Both fields it captures are perishable in a way ``self.stats``'s counters
        are not: the success term and the brick sample are recomputed by the very next
        ``env.step()`` for the whole batch, regardless of which envs have been rearmed. With
        one env and an immediate read that is invisible; with a batch, the natural pattern --
        finish env A, keep stepping so env B can finish, then read A -- would have reported
        A's NEXT episode.
        """
        for i in range(self.num_envs):
            if self._outcome[i] is not None:
                continue
            if dones[i] or self.abandoned[i]:
                self._outcome[i] = self._episode_outcome(i)

    def _episode_outcome(self, i: int) -> dict:
        """How the episode went, read from the env's own state (see ``episode_stats``)."""
        succeeded = self._succeeded(i)
        delivered = self.num_bricks if succeeded else int(self._stowed_sample[i].sum().item())
        return {"bricks_delivered": delivered, "succeeded": succeeded}

    def _matched_opening_command(self, i: int, fully_open: float) -> float:
        """Hold the jaws at this grasp's own opening instead of driving them fully open.

        The gripper channel is BINARY by contract (raw >= 0 opens, < 0 closes), so an
        intermediate opening is held by flipping the command around it. The jaw target ramps
        at the URDF's jaw velocity, about 3.3 mm per control step, which is its own
        hysteresis, so no further band is needed here: what this emits stays a binary gripper
        command the real arm accepts. ``grasp.JAW_HOLD_BAND`` is what the MODEL adds on top
        of this set point to follow the jaws rather than the command.

        An oracle that does not ask for it keeps the fully open descend, so the six-brick
        task and its gated corpus are untouched (MS6 rung 1). Under the continuous gripper the
        set point is returned as a gap instead (``_gripper_command``).
        """
        orc = self.oracles[i]
        binary = self.gripper_mode == "binary"
        if not getattr(orc, "matched_jaw_opening", False):
            return fully_open if binary else JAW_FULL_GAP
        target = grasp.landing_half_gap(float(orc.grasp_width)) - grasp.JAW_HOLD_BAND
        if not binary:
            return 2.0 * target
        measured = min(self.robot.data.joint_pos[i, self.finger_ids].tolist())
        return fully_open if measured <= target else -fully_open

    def _gripper_command(self, i: int, goal: oracle.Goal | None) -> float:
        """The seventh action column for env ``i`` this step, in the env's gripper mode.

        Binary (every corpus before MS6 phase B): the goal's own +1/-1, held around a matched
        opening or closing set point by the two flip helpers. Continuous (phase B design
        5.4): the set point itself as a jaw gap in meters, so the recorded label is the
        opening the teacher wants and not a chatter around it.
        """
        binary = self.gripper_mode == "binary"
        if goal is None:
            return GRIPPER_OPEN_ACTION if binary else JAW_FULL_GAP
        g = goal.gripper
        if g > 0.0 and goal.phase in MATCHED_OPENING_PHASES:
            return self._matched_opening_command(i, g)
        if g < 0.0 and goal.phase in MATCHED_CLOSE_PHASES:
            return self._matched_close_command(i, g)
        if binary:
            return g
        return JAW_FULL_GAP if g > 0.0 else 0.0

    def _advance_plan(self, i: int) -> None:
        """Move the commanded joint target one control step along this phase's plan: the
        polyline point at the minimum-jerk fraction of the walk, the last waypoint exactly
        on the final step, after which the plan is exhausted and the target holds there."""
        plan = self.plans[i]
        if not plan:
            return
        self.walk_step[i] += 1
        if self.walk_step[i] >= self.walk_steps[i]:
            self.q_cmd[i] = plan[-1]
            self.plans[i] = []
            return
        fraction = trajectory.min_jerk(self.walk_step[i] / self.walk_steps[i])
        self.q_cmd[i] = trajectory.polyline_point(plan, fraction)

    def _command_noise(self, i: int) -> torch.Tensor | float:
        """Per-step joint jitter for this env's episode draw (spec section 2.3, Datagen-v0 DR).

        Applied to the COMMAND and not to the plan, so the disturbance reaches the arm and
        the recorded action exactly as the servo version's action noise did, without the
        trim integrating against it (see ``_update_trims``). Hold phases command no noise
        unless ``noise_factor()`` boosts it -- CLOSE for a noise-induced first attempt, which
        has no motion for the noise to ride on but still needs the boost its docstring
        promises. Gated strictly on factor > 1.0 so that with induction disabled, and for
        every uninduced pick, no RNG is drawn on those phases at all.
        """
        orc = self.oracles[i]
        goal = self.goals[i]
        if orc is None or goal is None:
            return 0.0
        factor = orc.noise_factor()
        if goal.target_pos is None and factor <= 1.0:
            return 0.0
        std = orc.draw.action_noise_std * factor * JOINT_NOISE_PER_RAW_UNIT
        if std <= 0.0:
            return 0.0
        return std * torch.randn(ARM_DOF, dtype=torch.float64)

    def _update_trims(self, planned: list[torch.Tensor], dones: torch.Tensor) -> None:
        """Fold this step's joint tracking error into each env's trim.

        The trim closes the arm's open-loop droop -- effort-limit saturation on
        `link_004_joint` and `link_007_joint`, about 40 mm of `grasp_center` at the home pose
        (Task D3 part 9) -- which nothing else does now that no task-space servo re-references
        from the measured pose every step. It integrates against ``planned`` and not against
        the command actually sent, so it neither chases its own bias nor the per-step noise.
        """
        measured = self.robot.data.joint_pos[:, self.arm_ids].double().cpu()
        saturated = self._joint_saturated()
        for i in range(self.num_envs):
            if dones[i] or self.idle[i]:
                continue  # post-auto-reset state this step; rearm() resets the trim
            self.trims[i].update(planned[i], measured[i], saturated=saturated[i])

    def _joint_saturated(self) -> torch.Tensor:
        """(N, 6) bool: is each arm actuator's effort limit binding this step?

        ``computed_torque`` is the PD+feed-forward demand before the clamp and
        ``applied_torque`` is what reached the sim, so they differ exactly when the effort
        limit bound (see actuators/mit_mode.py). Feeds ``JointTrim``'s anti-windup.
        """
        computed = self.robot.data.computed_torque[:, self.arm_ids]
        applied = self.robot.data.applied_torque[:, self.arm_ids]
        return ((computed - applied).abs() > 1e-6).cpu()

    def _stash_cycle_state(self) -> None:
        """Pre-step phase/brick for CyclePhaseRecorder/CycleBrickRecorder.

        Those recorders fire inside env.step(), so the stash is refreshed in place before it.
        NOT ``target_brick or -1`` -- brick 0 is falsy and must survive.

        ``oracle_cycle_brick`` is the ACCOUNTING id: the batch loop counts cycles by it and
        the converter's ``--split_cycles`` cuts on it, so it names the retreating cycle's part
        until RETREAT ends. ``oracle_marker_part`` and ``oracle_cycle_target`` are what the
        policy sees, and under the 2026-09-22 protocol they switch to the pre-selected next
        part the moment RELEASE ends (``EnvOracle.preselects_next_target``).
        """
        if not hasattr(self.env, "oracle_cycle_phase"):
            self.env.oracle_cycle_phase = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device
            )
            self.env.oracle_cycle_brick = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device
            )
            self.env.oracle_marker_part = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device
            )
            self.env.oracle_cycle_target = torch.zeros(
                (self.num_envs, 7), dtype=torch.float32, device=self.device
            )
        for i in range(self.num_envs):
            goal_i, orc_i = self.goals[i], self.oracles[i]
            if self.idle[i] or orc_i is None:
                phase_int = -1
            elif goal_i is not None:
                phase_int = PHASE_TO_INT[goal_i.phase.name]
            else:
                phase_int = PHASE_TO_INT[orc_i.phase.name]  # DONE/ABANDON coast
            self.env.oracle_cycle_phase[i] = phase_int
            brick_i = orc_i.target_brick if orc_i is not None and not self.idle[i] else None
            self.env.oracle_cycle_brick[i] = brick_i if brick_i is not None else -1
            # The target channel (MS6 heap lever): the heap teacher's grasp on its target,
            # zeros on the six-brick oracle, between cycles and on an idle env.
            marker_i = brick_i
            start = getattr(orc_i, "_brick_start", None) if brick_i is not None else None
            chosen = getattr(orc_i, "grasp", None) if start is not None else None
            preselected = brick_i is not None and orc_i.preselects_next_target
            if preselected and orc_i.next_brick is not None:
                marker_i, start, chosen = orc_i.next_brick, orc_i.next_brick_start, orc_i.next_grasp
            self.env.oracle_marker_part[i] = marker_i if marker_i is not None else -1
            self.env.oracle_cycle_target[i] = torch.tensor(target_context(start, chosen))

    def _advance_phases(self, dones: torch.Tensor) -> None:
        """Per env: check the current goal's end condition and transition on end."""
        for i in range(self.num_envs):
            if dones[i]:
                continue  # post-auto-reset state this step; goal is stale until rearm()
            goal = self.goals[i]
            if goal is None:
                continue
            self.steps_in_phase[i] += 1
            result = self._phase_result(i, goal)
            if result is None:
                continue
            if goal.phase == oracle.Phase.RETREAT:
                # Mirrors EnvOracle.report()'s RETREAT branch, which is where the brick is
                # actually marked sorted -- tracked here too because that set is private to
                # EnvOracle.
                self.stats[i]["sort_attempts"] += 1
            orc = self.oracles[i]
            scene = self.snapshot(i)
            brick = orc.target_brick
            orc.report(result, scene)
            if self.single_cycle[i] and goal.phase == oracle.Phase.RETREAT:
                # MS4 c2 design 6.2: a takeover is one brick cycle. RETREAT's report marked
                # the brick sorted if it arrived; hand the env back instead of the next HOVER.
                self._go_idle(i, brick in orc.sorted_bricks)
                continue
            self._enter_phase(i, scene, ending_phase=goal.phase)

    def _phase_result(self, i: int, goal: oracle.Goal) -> oracle.PhaseResult | None:
        """None while the phase continues; a PhaseResult the step it ends."""
        if goal.target_pos is None:
            # Pure-hold goal (SETTLE/CLOSE/REOPEN/RELEASE): ends at hold_steps.
            if self.steps_in_phase[i] < goal.hold_steps:
                return None
            return oracle.PhaseResult(outcome="budget")

        if self.unplanned[i]:
            # This leg could not be planned, so the arm is not going to arrive: end the phase
            # now. Holding still until the budget expires writes those control steps into the
            # dataset as a dead segment, which teaches a policy to pause -- openyam-lab's M6
            # campaign measured a dead tail over 36 % of eval steps and paid to escape it.
            # For RAISE, which is the leg this actually happens on (core.oracle's
            # RAISE_CLEARANCE_Z), ending at once also leaves EXTEND starting from the LIFT
            # pose, which Task D5 measured as the BEST pose to start it from.
            if goal.phase == oracle.Phase.LIFT:
                return self._lift_result(i, "unplanned")
            return oracle.PhaseResult(outcome="unplanned")

        if self.holding[i]:
            # Post-arrival settle (LIFT): the leg already ended this phase; keep commanding
            # the same joint target for goal.hold_steps more steps, so the gate below sees
            # settled contact forces rather than the instant the position error first dipped
            # under stop_pos.
            self.hold_counter[i] += 1
            if self.hold_counter[i] < goal.hold_steps:
                return None
            self.holding[i] = False
            outcome = self.pending_outcome[i]
            if goal.phase == oracle.Phase.LIFT:
                return self._lift_result(i, outcome)
            return oracle.PhaseResult(outcome=outcome)

        pos_err = (goal.target_pos.to(self.device) - self.pick_point(i)).norm().item()
        # Arrival means the whole planned leg was walked AND the arm is inside the phase's
        # own position tolerance. Requiring the plan to be exhausted is what makes
        # "converged" mean "walked the leg and arrived" rather than "passed within tolerance
        # while still moving": the stop tolerances run up to 20 mm (RAISE) against a leg that
        # walks ~1.5 mm per step, so several phases would otherwise end mid-leg.
        converged = goal.stop_pos is not None and not self.plans[i] and pos_err < goal.stop_pos
        stalled = False
        if goal.stall_steps:
            now = self.pick_point(i)
            moved = (now - self.previous_pick_point[i]).norm().item()
            self.quiet[i] = self.quiet[i] + 1 if moved < STALL_EPS else 0
            self.previous_pick_point[i] = now.clone()
            stalled = self.quiet[i] >= goal.stall_steps
        budget = self.steps_in_phase[i] >= goal.max_steps
        if not (converged or stalled or budget):
            return None
        outcome = "converged" if converged else ("stalled" if stalled else "budget")
        if goal.hold_steps > 0:
            self.holding[i] = True
            self.hold_counter[i] = 0
            self.pending_outcome[i] = outcome
            return None
        if goal.phase == oracle.Phase.LIFT:
            return self._lift_result(i, outcome)
        return oracle.PhaseResult(outcome=outcome)

    def _lift_result(self, i: int, outcome: str) -> oracle.PhaseResult:
        """LIFT end: pinch evidence + rise/xy, exactly the check script's gate (grasp_one)."""
        orc = self.oracles[i]
        brick = orc.target_brick
        brick_end = self._brick_pos(i, brick)
        rise = (brick_end[2] - self.brick_start[i][2]).item()
        xy_error = (brick_end[:2] - self.pick_point(i)[:2]).norm().item()
        lifted = rise > LIFT_MIN_RISE and xy_error < LIFT_MAX_XY_ERROR
        if not lifted and orc.accepts_any_held_part:
            held = self._held_part(i, brick)
            if held is not None:
                # The target was missed but the jaws are not empty, and on a heap one bin
                # takes every part (core.heap_oracle: 16 of 154 LIFT reports of 2026-09-17).
                # The cycle follows the part actually held; REOPEN would drop it.
                orc.adopt_held_part(held)
                brick, lifted = held, True
        # MS6 design 7.4: the stall floor follows the width the jaws close across, and so
        # does the force floor (the jaw drive's force is its stall times its stiffness). The
        # floors stay the ones the CHOSEN grasp set -- `pinch_ok` is telemetry here, not the
        # decision (core.oracle's LIFT branch) -- but they are read against the held part.
        width = orc.grasp_width
        min_size = getattr(orc, "grasp_min_size", min(layout.BRICK_SIZE))
        pinch = self.pinch_evidence(
            i, brick, pinch_min_stall(width, min_size), pinch_min_force(width, min_size)
        )
        return oracle.PhaseResult(outcome=outcome, pinch_ok=pinch["ok"], lifted=lifted)

    def _held_part(self, i: int, target: int) -> int | None:
        """The part other than ``target`` that is up in the jaws, nearest the pick point.

        A part counts as held on the same two numbers the target's gate uses: it stands clear
        of the desk (layout.py: desk top at z = 0) by at least LIFT_MIN_RISE, and it sits
        inside LIFT_MAX_XY_ERROR of the pick point. A part already in the bin is skipped --
        it is resting there, not riding in the jaws, and it can be within reach of the pick
        point once the arm has come back out.
        """
        pick = self.pick_point(i)
        stowed = self.oracles[i].sorted_bricks
        best, best_error = None, LIFT_MAX_XY_ERROR
        for b in range(self.num_bricks):
            if b == target or b in stowed:
                continue
            pos = self._brick_pos(i, b)
            if pos[2].item() <= LIFT_MIN_RISE:
                continue
            error = (pos[:2] - pick[:2]).norm().item()
            if error < best_error:
                best, best_error = b, error
        return best

    def _enter_phase(
        self, i: int, scene: oracle.SceneSnapshot, ending_phase: oracle.Phase | None = None
    ) -> None:
        """Fetch the next goal, plan the leg it asks for, zero the per-phase counters."""
        orc = self.oracles[i]
        goal = orc.next_goal(scene)
        self.goals[i] = goal
        self.steps_in_phase[i] = 0
        self.quiet[i] = 0
        self.holding[i] = False
        # Per-phase, not per-episode: OpenYAM measured (its Task 8) that RAISE's climb
        # legitimately nears a joint limit and that EXTEND's own approach then starts from
        # that same physical pose -- common, and not itself evidence of a stuck env. A
        # persistent counter double-counts the same joint-limit contact across both phases
        # and can trip before EITHER phase's own budget would. See WATCHDOG_TRIP_STRIKES's
        # comment for the margin dependency this creates for RAISE_MAX_STEPS, and for the
        # REOPEN/RELEASE gap it trades in for it.
        self.watchdog_strikes[i] = 0
        if goal is None:
            if self.single_cycle[i]:
                # A takeover that ends in DONE or ABANDON hands the env back; the cycle did
                # not stow its brick, and the DAgger loop drops its frames.
                self._go_idle(i, False)
                return
            if orc.phase == oracle.Phase.ABANDON:
                self.abandoned[i] = True
                if self.stats[i]["abandon"] is None:
                    self.stats[i]["abandon"] = (
                        "grasp_retries_exhausted"
                        if ending_phase == oracle.Phase.LIFT
                        else "no_reachable_brick"
                    )
            return
        if goal.phase == oracle.Phase.REOPEN:
            # EnvOracle.retries_used resets to 0 at the next brick's HOVER (core/oracle.py),
            # so it cannot be read live for an episode total -- accumulate it here, once per
            # REOPEN entry, the same treatment "sort_attempts" gets for the same reason.
            self.stats[i]["retries"] += 1
        if goal.phase == oracle.Phase.HOVER:
            # Fresh brick target selected inside next_goal(); capture its start pose for the
            # LIFT rise/xy gate (grasp_one's brick_start). Recomputed via _brick_pos rather
            # than read off `scene` (CPU -- see snapshot()) so it stays on self.device,
            # matching brick_end/pick_point in _lift_result.
            self.brick_start[i] = self._brick_pos(i, orc.target_brick)
        if goal.phase == oracle.Phase.DESCEND:
            # core.oracle deliberately leaves this to the runtime (it owns the
            # maker_arm_lab.robots import) -- the core target is the brick's resting z. It is
            # 0.0 on the global-IK path, where the descend commands a POSITION and the whole
            # depth is expressed by the pick offset; see robots.frames.
            goal.target_pos[2] = goal.target_pos[2] + DESCEND_CLEARANCE_MEASURED
        self._plan_leg(i, goal)
        if goal.stall_steps:
            self.previous_pick_point[i] = self.pick_point(i).clone()

    # -- planning ------------------------------------------------------------

    def _plan_leg(self, i: int, goal: oracle.Goal) -> None:
        """Turn ``goal`` into the joint waypoints ``step()`` will walk.

        A jaw is symmetric under a half turn, so an explicit ``goal.yaw`` is folded to one of
        ``yaw``/``yaw + pi`` by ``oracle.fold_jaw_heading`` -- which picks on the wrist
        roll's REMAINING TRAVEL over every waypoint in ``goal.yaw_hold_xy``, not on raw
        proximity to the current wrist heading. Note what the fold does and does not decide
        HERE: ``core.global_ik``'s heading residual is folded modulo pi, so the solver treats
        both representatives as one constraint and picks the wrist branch itself, by joint
        margin. The fold still decides which representative this class RECORDS as commanded
        (and so what ``attitude_error`` measures against), and it stays the place to look if
        a leg ever runs the wrist roll onto its stop: it is the only machinery here that
        reasons about the whole excursion a heading is held across, which the solver, seeing
        one waypoint at a time, does not.

        ``goal.yaw`` is None for every phase after DESCEND -- there is no brick reference
        once the brick has left the desk -- and those legs start from the heading last
        commanded and may end at a relaxed one; see ``_plan_carry``.
        """
        self.plans[i] = []
        self.walk_steps[i] = 1
        self.walk_step[i] = 0
        self.unplanned[i] = False
        if goal.yaw is not None:
            self.commanded_yaw[i] = oracle.fold_jaw_heading(
                goal.yaw + math.pi / 2,
                self.jaw_yaw(i),
                self.wrist_roll(i),
                self.wrist_roll_limits,
                base_yaw_range=self._base_yaw_range(i, goal.yaw_hold_xy),
                roll_sign=WRIST_ROLL_SIGN,
                base_sign=BASE_YAW_SIGN,
            )
        if goal.target_pos is None:
            return  # hold phase: keep commanding the joint target the last leg left
        start = self.q_cmd[i]
        if goal.phase is not oracle.Phase.RETREAT:
            self.unroll[i] = None
        elif self.unroll[i] is not None:
            start = self.unroll[i][1]
        pick_from, tilt_from = self._commanded_pick_pose(i, start)
        pick_to = goal.target_pos.double().cpu()
        heading = self.commanded_yaw[i]
        if goal.phase == oracle.Phase.DESCEND or goal.phase in (
            oracle.PUSH_PHASES | oracle.TIP_PHASES
        ) - {oracle.Phase.PUSH_HOVER, oracle.Phase.TIP_HOVER}:
            # PUSH_DESCEND, PUSH_SWEEP (MS6 design 7.3) and TIP_DESCEND, TIP_SWEEP run near
            # level like a descend: at the grasp tilt the plate corner dips 21 mm under the tip
            # and ploughs the desk.
            plan = self._plan_descend(i, goal, start, pick_from, tilt_from, pick_to, heading)
        elif goal.phase == oracle.Phase.HOVER:
            plan = self._plan_hover_ladder(i, goal, start, pick_to, heading)
        else:
            # GRASP_TILT_RAD is the FLOOR, not an offset. core.oracle is the ported,
            # arm-agnostic state machine and is not modified (spec section 5 item 5), so it
            # still asks for tilt_rad=0.0 on the carrying phases, meaning "hold the tool
            # level". This arm cannot hold a level tool over the scatter band -- measured
            # minimum 18.42 deg at the band's outer corners at hover height, Task D3 -- and
            # demanding one jams link_005_joint against its stop. So 0.0 is read here as "as
            # level as this arm gets", which is frames.GRASP_TILT_RAD. RELEASE is unaffected:
            # its oracle.RELEASE_TILT_RAD of 1.40 rad is far past the floor. The DESCEND
            # branch above is the exception, and the reason the floor is not applied
            # everywhere: at the grasp tilt the jaw plates foul the desk at every pick offset
            # (ruling D3-10), so the descend runs the near-level ladder instead.
            tilt_to = max(goal.tilt_rad, GRASP_TILT_RAD)
            plan, heading, direct = self._plan_carry(
                i, goal, start, pick_from, tilt_from, pick_to, tilt_to, heading
            )
            if (
                plan is not None
                and goal.phase == oracle.Phase.RAISE
                and not self._extend_plannable_from(i, plan[-1], heading)
            ):
                # LOOKAHEAD. RAISE is the one leg the episode can do without -- the runtime
                # already advances straight to EXTEND when it cannot be planned, and Task D5
                # measured the LIFT pose as EXTEND's best start -- so it is only worth
                # walking if EXTEND can still be planned from where it ends. It cannot
                # always: MEASURED on GPU (2026-09-07, instrumented 12-env batch), a near
                # pick (radius ~0.15 m, azimuth ~-11 deg) bound for an outer bin had its
                # RAISE fall through to `_plan_direct`, which solved the endpoint 1.25 rad
                # away in base yaw -- inside MAX_LEG_JOINT_JUMP -- on a branch from which
                # EXTEND had no continuous path at any heading rung and no direct fallback
                # either. RELEASE then opened the jaws over the scatter band and the episode
                # timed out one brick short; the same EXTEND planned at once from the LIFT
                # pose. Both of the first round's failures were this, so the check is made
                # here, with the runtime's own planner, rather than by tightening the jump
                # cap by guesswork. Costs one EXTEND plan per walkable RAISE.
                self.stats[i]["raise_vetoed"] += 1
                plan = None
                direct = False
            if plan is not None:
                self.commanded_tilt[i] = tilt_to
                self.commanded_yaw[i] = heading
                if direct:
                    self.stats[i]["direct_legs"] += 1
        if plan is None:
            # No plan: the phase ends on the next step (see `_phase_result`) rather than the
            # arm holding still until its budget expires. WHAT CATCHES IT DEPENDS ON THE
            # PHASE, and only one of the two is a ladder:
            #
            # * before CLOSE (HOVER/DESCEND), the pick simply does not happen, LIFT's
            #   pinch/rise gate fails, and core.oracle's retry -> retry -> ABANDON ladder
            #   takes over. Nothing is lost.
            # * on a CARRY leg (LIFT/RAISE/EXTEND/RETREAT) there is NO gate and NO ladder:
            #   core.oracle sequences straight on, so an unplanned EXTEND means RELEASE opens
            #   the jaws wherever the arm is standing and the brick is not delivered. RETREAT
            #   then counts it in `stats["sort_attempts"]` regardless (see `_fresh_stats`). That
            #   episode does not reach the dataset -- `all_bricks_stowed` grades the bricks'
            #   own poses, so it never fires, the episode ends on its time_out and
            #   EXPORT_SUCCEEDED_ONLY drops it (measured twice: /tmp/oracle3.log trial 0 and
            #   /tmp/oracle2_sink.log trial 0, both `sorted=6` with a brick on the desk and
            #   `exported_count=0`) -- but it is a lost episode, not a recovered one.
            #
            # Counted either way, because a run where this is common has a reach problem, not
            # a grasp problem.
            self.stats[i]["unplanned_legs"] += 1
            self.unplanned[i] = True
            return
        if goal.phase == oracle.Phase.EXTEND and getattr(
            self.oracles[i], "matched_jaw_opening", False
        ):
            roll = self._roll_waypoints(plan[-1])
            if roll and self.gripper_mode != "binary":
                roll = self._paced_roll(i, plan, plan[-1], roll[-1])
                # RETREAT turns the wrist back from where RELEASE holds to the carry's end.
                self.unroll[i] = (roll[-1], plan[-1])
            plan = plan + roll
        elif goal.phase == oracle.Phase.RETREAT and self.unroll[i] is not None:
            (rolled, arrival), self.unroll[i] = self.unroll[i], None
            # The carry plan starts at `arrival`; the walk starts where RELEASE holds.
            plan = [rolled] + self._paced_roll(i, plan, rolled, arrival) + plan[1:]
        self.plans[i] = plan

    def _paced_roll(
        self, i: int, plan: list[torch.Tensor], start: torch.Tensor, end: torch.Tensor
    ) -> list[torch.Tensor]:
        """Wrist-only waypoints after ``start`` up to ``end``, walked with ``plan``.

        The walk gives every segment the same share of its steps (``polyline_point``), and the
        carry's steps come from its Cartesian length, which a wrist roll does not add to. So
        the roll is cut into as many segments as keep the wrist under ``ROLL_MAX_STEP_RAD``
        per control step at the minimum-jerk peak, and the walk grows by the carry's own
        steps per segment for each one: the carry keeps its pace.
        """
        per_segment = self.walk_steps[i] / len(plan)
        turn = abs(float(end[WRIST_ROLL_INDEX] - start[WRIST_ROLL_INDEX]))
        peak = trajectory.MIN_JERK_PEAK_VELOCITY * turn / (ROLL_MAX_STEP_RAD * per_segment)
        n = max(1, math.ceil(peak))
        self.walk_steps[i] = math.ceil(self.walk_steps[i] * (len(plan) + n) / len(plan))
        return [start + (end - start) * k / n for k in range(1, n + 1)]

    def _matched_close_command(self, i: int, fully_closed: float) -> float:
        """Hold the jaws at the part's own width instead of driving them shut.

        The floor is the grasp's half width less ``grasp.CLOSE_SQUEEZE_M``, which is where a brick
        of the same grasp width settles on its own. Bang-bang around it, the same way
        ``_matched_opening_command`` holds an opening: what this emits stays a binary gripper
        command the real arm accepts.

        An oracle that does not ask for it keeps driving the jaws shut, so the six-brick task
        and its gated corpus are untouched (MS6 rung 3). Under the continuous gripper the set
        point is returned as a gap instead (``_gripper_command``).
        """
        orc = self.oracles[i]
        binary = self.gripper_mode == "binary"
        if not getattr(orc, "matched_jaw_opening", False):
            return fully_closed if binary else 0.0
        floor = 0.5 * grasp.close_gap(float(orc.grasp_width))
        if not binary:
            return 2.0 * floor
        measured = min(self.robot.data.joint_pos[i, self.finger_ids].tolist())
        return fully_closed if measured > floor else -fully_closed

    def _roll_waypoints(self, arrival: torch.Tensor) -> list[torch.Tensor]:
        """Extra EXTEND waypoints that stand the jaw plane up, turning the wrist and nothing else.

        MS6 rung 2. RELEASE has no pose of its own -- it HOLDS what EXTEND left -- so the
        plane the jaws open in is whatever roll the carry happened to end at, and a
        near-horizontal one drops the part in free fall from the release height instead of
        letting it slide off the lower plate (``core.global_ik.steepest_jaw_roll``).

        Only ``WRIST_ROLL_INDEX`` moves. That joint turns about the tool axis, so these
        waypoints move `grasp_center` by 0.00 mm and the tool axis by 0.00 degrees, and
        ``PICK_OFFSET_MEASURED`` lies on that axis so the pick point does not move either.
        Nothing is solved here and nothing can fail to solve, which is why the roll is NOT
        asked of ``_plan_carry``: that method's own docstring records three six-brick
        episodes lost to an EXTEND held at a heading the solver could not walk.

        The wrist's travel is finite, so a roll that would leave the joint's range is dropped
        rather than clamped: half a stood-up plane is not worth a servo against its stop.

        Under the continuous gripper the roll levels the plane instead: of the two level rolls,
        a quarter turn either way off the steepest one, the shorter that keeps
        ``LEVEL_ROLL_MARGIN_RAD`` off both wrist stops, and none if neither does.
        """
        _, rotation = self.ik.fk(arrival)
        rotation = torch.as_tensor(rotation, dtype=torch.float64)
        jaw = rotation @ torch.tensor(JAW_AXIS_BODY, dtype=torch.float64)
        tool = rotation @ torch.tensor(TOOL_AXIS_BODY, dtype=torch.float64)
        # `link_007_joint` turns the jaw axis by MINUS its own angle about the tool axis
        # (measured 2026-09-17, ratio -1.000), so the joint moves against the roll.
        delta = -steepest_jaw_roll(jaw, tool)
        lo, hi = self.wrist_roll_limits
        if self.gripper_mode != "binary":
            # Level: a quarter turn off the steepest roll, either way. Take the shorter one
            # that keeps LEVEL_ROLL_MARGIN_RAD off both wrist stops.
            here = float(arrival[WRIST_ROLL_INDEX])
            options = [
                d
                for d in (delta + math.pi / 2, delta - math.pi / 2)
                if lo + LEVEL_ROLL_MARGIN_RAD <= here + d <= hi - LEVEL_ROLL_MARGIN_RAD
            ]
            if not options:
                return []
            delta = min(options, key=abs)
        target = float(arrival[WRIST_ROLL_INDEX]) + delta
        if not lo <= target <= hi or abs(delta) < ROLL_DEADBAND_RAD:
            return []
        steps = max(1, math.ceil(abs(delta) / ROLL_MAX_STEP_RAD))
        out = []
        for k in range(1, steps + 1):
            q = arrival.clone()
            q[WRIST_ROLL_INDEX] = float(arrival[WRIST_ROLL_INDEX]) + delta * k / steps
            out.append(q)
        return out

    def _plan_carry(
        self, i, goal, start, pick_from, tilt_from, pick_to, tilt_to, heading
    ) -> tuple[list[torch.Tensor] | None, float, bool]:
        """A leg with the brick in the jaws (LIFT/RAISE/EXTEND/RETREAT), and the one place
        the jaw heading is allowed to move.

        Returns the plan, the heading it ends at, and whether the plan came from the
        joint-space fallback (`_plan_direct`). The caller counts that last one into
        `stats["direct_legs"]`, not this method: `_plan_leg`'s RAISE lookahead plans an
        EXTEND it never walks, and a leg that is not walked must not be counted.

        WHY THE HEADING LADDER, measured rather than assumed. `core.oracle` sets
        ``Goal.yaw`` only at HOVER and DESCEND -- where the jaws must be square across the
        brick, which is the grasp -- and leaves it None from CLOSE to RETREAT, meaning "no
        brick reference any more, hold what you have". Holding it EXACTLY is over-constrained
        for the solver in a way the differential-IK servo never was, because the servo could
        sit partway to an attitude it could not reach while the solver refuses the whole leg:
        the brick's own desk orientation fixes the jaw heading modulo pi, and for some
        orientations there is no CONTINUOUS path from the pick to the release attitude at that
        heading. Measured on this arm (CPU sweep over the shipping EXTEND leg): a brick at
        (0.16, 0.13) carried to either outer bin has no walkable leg at a held heading of
        +-90 deg -- the break is at a waypoint that is itself REACHABLE, so it is a branch
        change the continuity cap rightly refuses, not a reach limit -- while the same leg
        solves at every heading tested from -60 to +60 deg. Live, an unplanned EXTEND meant
        the arm stood still for its whole 300-step budget and then opened the jaws over the
        scatter band. Three of seven six-brick episodes lost a brick that way; the mechanism
        was traced end to end on one of them (EXTEND 300 steps, 471 mm short, tool never
        tilted off the grasp attitude, brick found on the desk under the leg's start), and
        the other two carry the same signature -- a brick that never entered a bin and no
        stow ever lost.

        Rotating the wrist while a brick is gripped rotates the BRICK, which costs nothing --
        the release drops it into a bin at whatever yaw it is holding, and nothing downstream
        reads brick yaw. So the ladder tries the held heading first and then widening
        offsets, and the first that yields a continuous path wins. The heading is swept ALONG
        the leg like the tilt, never stepped: a heading offset applied at the first waypoint
        would demand the whole wrist rotation inside one continuity step.

        The ladder is NOT screened by an endpoint solve first, and that was measured rather
        than assumed: a failed `solve_path` stops at the first unwalkable waypoint, so it
        already costs about what one restart-spread solve costs, and adding a spread per rung
        made the same two-episode gate run 6.0 -> 8.1 minutes. The obvious optimisation is a
        pessimisation here; do not re-add it without timing it.
        """
        for offset in HEADING_RELAXATION_RAD:
            plan = self._plan_cartesian(
                i, goal, start, pick_from, tilt_from, pick_to, tilt_to, heading, heading + offset
            )
            if plan is not None:
                return plan, heading + offset, False
        # Last resort: solve the endpoint and interpolate in joint space, at the held heading.
        plan = self._plan_direct(i, goal, start, pick_to, tilt_to, heading)
        return plan, heading, plan is not None

    def _extend_plannable_from(self, i: int, q: torch.Tensor, heading: float) -> bool:
        """Would the EXTEND leg plan if the arm were commanded to ``q`` holding ``heading``?

        A dry run of exactly what `_plan_leg` will do at EXTEND entry -- the same goal
        (`EnvOracle.extend_goal`), the same heading ladder and the same fallback -- from the
        pose a candidate RAISE plan ends at. Nothing it computes is kept: the plan is thrown
        away, `direct_legs` is not touched, and the walk length `_plan_cartesian` sets as
        a side effect is restored, so a vetoed RAISE leaves the env exactly as an unplanned
        one does.
        """
        goal = self.oracles[i].extend_goal()
        pick_from, tilt_from = self._commanded_pick_pose(i, q)
        walk_steps = self.walk_steps[i]
        plan, _, _ = self._plan_carry(
            i,
            goal,
            q,
            pick_from,
            tilt_from,
            goal.target_pos.double().cpu(),
            max(goal.tilt_rad, GRASP_TILT_RAD),
            heading,
        )
        self.walk_steps[i] = walk_steps
        return plan is not None

    def _plan_hover_ladder(self, i, goal, start, pick_to, heading) -> list[torch.Tensor] | None:
        """HOVER at the drawn height, or failing that at each rung of HOVER_HEIGHT_LADDER_M.

        A rung counts only if the hover solves AND the descend ladder can be planned from
        the pose it ends at -- the same lookahead `_plan_leg` runs for RAISE, and for the
        same reason: a leg that strands the next one is not a leg. When a lower rung is
        taken, ``goal.target_pos[2]`` is moved down to it, because `_phase_result` grades
        HOVER's arrival against the goal and a hover walked to 0.10 m can never converge on a
        target at 0.13 m.
        """
        brick_z = float(self.brick_start[i][2].item())
        drawn = float(pick_to[2])
        heights = [drawn] + [brick_z + h for h in HOVER_HEIGHT_LADDER_M if brick_z + h < drawn]
        for rung, z in enumerate(heights):
            target = pick_to.clone()
            target[2] = z
            plan = self._plan_hover(i, goal, start, target, heading)
            if plan is None or not self._descend_plannable_from(i, plan[-1], goal, heading):
                continue
            if rung > 0:
                self.stats[i]["hover_lowered"] += 1
                goal.target_pos[2] = z
            return plan
        return None

    def _descend_plannable_from(self, i: int, q: torch.Tensor, hover: oracle.Goal, heading) -> bool:
        """Would the DESCEND leg plan if the arm were commanded to ``q`` after ``hover``?

        A dry run of what `_plan_leg` will do at DESCEND entry: the same goal
        (`EnvOracle.descend_goal`, at the yaw the HOVER goal carries, with the runtime's
        DESCEND_CLEARANCE_MEASURED applied as `_enter_phase` applies it), the same tilt
        ladder. Nothing it computes is kept: the plan is dropped and the per-env state the
        ladder writes -- pick offset, commanded tilt, walk length -- is restored.
        """
        goal = self.oracles[i].descend_goal(hover.yaw)
        goal.target_pos[2] = goal.target_pos[2] + DESCEND_CLEARANCE_MEASURED
        pick_offset = self.pick_offset[i].clone()
        tilt, walk_steps = self.commanded_tilt[i], self.walk_steps[i]
        pick_from, tilt_from = self._commanded_pick_pose(i, q)
        plan = self._plan_descend(
            i, goal, q, pick_from, tilt_from, goal.target_pos.double().cpu(), heading
        )
        self.pick_offset[i] = pick_offset
        self.commanded_tilt[i], self.walk_steps[i] = tilt, walk_steps
        return plan is not None

    def _plan_hover(self, i, goal, start, pick_to, heading) -> list[torch.Tensor] | None:
        """The approach: solve the endpoint GLOBALLY, then interpolate in joint space.

        This is the one solve per pick attempt that CHOOSES a branch rather than following
        one, and it is what makes the arm turn its base for a base-shaped motion instead of
        spending a wrist joint on it. The path itself is joint-space and therefore continuous
        by construction; the jaws are open and the arm is above the desk for all of it.
        """
        self.pick_offset[i] = torch.tensor(PICK_OFFSET_MEASURED, device=self.device)
        target = self._target_for(pick_to, GRASP_TILT_RAD, heading, PICK_OFFSET_MEASURED[2])
        solution = self.ik.solve(target, seed=start, restarts=HOVER_RESTARTS)
        if solution is None:
            return None
        self.commanded_tilt[i] = GRASP_TILT_RAD
        return self._interpolate(i, goal, start, solution.q)

    def _plan_descend(
        self, i, goal, start, pick_from, tilt_from, pick_to, heading
    ) -> list[torch.Tensor] | None:
        """The descend: the tilt ladder, smallest tilt first, each rung with its own offset.

        Level where level solves, otherwise the least tilt this brick admits (ruling D3-11
        candidate 2). The rung's pick offset becomes this env's, so ``pick_point`` -- which
        the DESCEND tolerance, the LIFT gate and every later leg's target are expressed in --
        names the point the jaws will actually close on. The ladder and the offset follow
        ``goal.half_height``, the 2x4's on the six-brick task and the part's own on a heap.
        """
        for rung in descend_tilt_ladder(goal.half_height):
            geometry = descend_pick_offset(rung, goal.half_height)
            if geometry is None:
                continue
            offset_z = geometry[0]
            plan = self._plan_cartesian(
                i, goal, start, pick_from, tilt_from, pick_to, rung, heading, offset_z=offset_z
            )
            if plan is not None:
                self.pick_offset[i] = torch.tensor(
                    (PICK_OFFSET_MEASURED[0], PICK_OFFSET_MEASURED[1], offset_z),
                    device=self.device,
                )
                self.commanded_tilt[i] = rung
                return plan
        return None

    def _plan_cartesian(
        self,
        i,
        goal,
        start,
        pick_from,
        tilt_from,
        pick_to,
        tilt_to,
        heading_from,
        heading_to=None,
        offset_z=None,
    ) -> list[torch.Tensor] | None:
        """A straight Cartesian leg for the pick point, with the tilt (and, on a carry leg,
        the jaw heading) swept along it.

        ``GlobalIK.solve_path``'s continuity constraint is what makes this safe to walk with
        a brick in the jaws: a solver jumping to another IK branch mid-leg fails the path
        outright instead of commanding a lurch through the scene.
        """
        if heading_to is None:
            heading_to = heading_from
        if offset_z is None:
            offset_z = float(self.pick_offset[i, 2].item())
        count, steps = self._leg_resolution(
            i,
            goal,
            distance=float((pick_to - pick_from).norm()),
            tilt_change=abs(tilt_to - tilt_from),
            heading_change=abs(heading_to - heading_from),
        )
        targets = [
            self._target_for(
                pick_from + (pick_to - pick_from) * t,
                float(tilt_from + (tilt_to - tilt_from) * float(t)),
                float(heading_from + (heading_to - heading_from) * float(t)),
                offset_z,
            )
            for t in torch.linspace(0.0, 1.0, count, dtype=torch.float64)
        ]
        chain = self.ik.solve_path(targets, seed=start)
        if chain is None:
            return None
        self.walk_steps[i] = steps
        return [step.q for step in chain]

    def _plan_direct(self, i, goal, start, pick_to, tilt_to, heading) -> list[torch.Tensor] | None:
        """Fallback for a leg with no continuous Cartesian solution: solve the endpoint.

        Seeded from the current pose and rejected outright if it lands further than
        ``MAX_LEG_JOINT_JUMP`` away, so this cannot answer a failed carry with a branch flip.
        The path is then joint-space, which gives up the straight-line guarantee on the tool
        -- acceptable in a fallback, where the alternative is not moving at all.
        """
        target = self._target_for(pick_to, tilt_to, heading, float(self.pick_offset[i, 2].item()))
        solution = self.ik.solve(
            target, seed=start, restarts=0, q_prev=start, max_step=MAX_LEG_JOINT_JUMP
        )
        if solution is None:
            return None
        return self._interpolate(i, goal, start, solution.q)

    def _interpolate(self, i, goal, start, end) -> list[torch.Tensor]:
        """Joint-space waypoints from ``start`` to ``end``, cut to this phase's budget."""
        count, steps = self._leg_resolution(
            i, goal, joint_change=float(torch.max(torch.abs(end - start)))
        )
        self.walk_steps[i] = steps
        return [start + (end - start) * t for t in torch.linspace(0.0, 1.0, count).double()]

    def _leg_resolution(self, i: int, goal: oracle.Goal, **geometry: float) -> tuple[int, int]:
        """``leg_resolution`` for this env's episode draw, this phase's budget and pace."""
        speed_scale = self.transport_speed if goal.phase in TRANSPORT_PHASES else 1.0
        return leg_resolution(
            goal.max_steps, self.oracles[i].draw.gain_scale, speed_scale=speed_scale, **geometry
        )

    def _target_for(self, pick, tilt: float, heading: float, offset_z: float) -> GraspTarget:
        """The six-scalar grasp task that puts the PICK point at ``pick``."""
        return GraspTarget(
            position=grasp_center_for(pick, tilt, heading, offset_z),
            tilt=tilt,
            jaw_heading=heading,
            lean_azimuth=LEAN_AZIMUTH,
        )

    def _commanded_pick_pose(self, i: int, q: torch.Tensor) -> tuple[torch.Tensor, float]:
        """The pick point and tool tilt that the joint target ``q`` asks for.

        A leg is planned from the COMMANDED pose, not the measured one, so that the command
        stream stays continuous: the arm lags its target by whatever the trim has not yet
        closed, and starting each leg from the measured pose would command a step backwards
        every time.
        """
        position, rotation = self.ik.fk(q)
        offset = self.pick_offset[i].double().cpu()
        tool = rotation @ self.tool_axis_body
        tilt = float(torch.arccos(torch.clamp(-tool[2], -1.0, 1.0)))
        return position + rotation @ offset, tilt

    # -- episode lifecycle -------------------------------------------------

    def rearm(self, env_ids: list[int], idle: bool = False) -> None:
        """Re-seed the envs in ``env_ids`` after their caller has reset them.

        ``idle=True`` leaves the env under another actor (the DAgger policy): no oracle, no
        goal, and ``actions()``/``_stash_cycle_state`` treat it as coasting.
        """
        for i in env_ids:
            self.idle[i] = idle
            if idle:
                self.oracles[i] = None
                self.goals[i] = None
            else:
                draw = oracle.sample_episode_draw(self.rng, self.noise)
                # MuJoCo with per-episode draws moves the bins at every reset: the oracle for
                # env i must plan for env i's bins. Every other env carries the rig's one
                # layout.
                layouts = getattr(self.env, "bin_layouts", None)
                bl = self.bl if layouts is None else layouts[i]
                self.oracles[i] = self.oracle_factory(i, bl, draw)
            self.commanded_yaw[i] = self.jaw_yaw(i)
            self.commanded_tilt[i] = self.tool_tilt(i)
            self.pick_offset[i] = torch.tensor(PICK_OFFSET_MEASURED, device=self.device)
            # The reset pose is the first thing commanded, so SETTLE holds the arm where the
            # reset left it (start-pose randomisation included) instead of driving it
            # anywhere. The trim starts clean: its accumulated bias belongs to the previous
            # episode's poses.
            self.q_cmd[i] = self.q_arm(i)
            self.trims[i].reset()
            self.plans[i] = []
            self.walk_steps[i] = 1
            self.walk_step[i] = 0
            self.brick_start[i] = None
            self.previous_pick_point[i] = None
            self.steps_in_phase[i] = 0
            self.quiet[i] = 0
            self.holding[i] = False
            self.watchdog_strikes[i] = 0
            self.unplanned[i] = False
            self.abandoned[i] = False
            self._outcome[i] = None
            self.stats[i] = self._fresh_stats()
            self.single_cycle[i] = False
            self.takeover_result[i] = None
            if not idle:
                self.goals[i] = self.oracles[i].next_goal(self.snapshot(i))  # SETTLE

    def resume(self, i: int, target: int | None = None) -> bool:
        """Take idle env ``i`` over from the policy for one brick cycle (MS4 c2 design 6.2).

        Returns False, and leaves the env idle, when the oracle answers DONE or ABANDON at
        entry or the first leg cannot be planned. With ``target`` (the heap outer loop, phase
        B design 5.5) a takeover that holds nothing works that part instead of letting HOVER
        select; it is declined when no grasp ranks on the part any more.
        """
        if not self.idle[i]:
            raise RuntimeError(f"env {i} is not idle")
        layouts = getattr(self.env, "bin_layouts", None)
        bl = self.bl if layouts is None else layouts[i]
        orc = self.oracle_factory(i, bl, oracle.nominal_draw())
        self.oracles[i] = orc
        self.commanded_yaw[i] = self.jaw_yaw(i)
        self.commanded_tilt[i] = self.tool_tilt(i)
        self.pick_offset[i] = torch.tensor(PICK_OFFSET_MEASURED, device=self.device)
        self.q_cmd[i] = self.q_arm(i)
        self.trims[i].reset()
        self.plans[i] = []
        self.walk_steps[i] = 1
        self.walk_step[i] = 0
        self.previous_pick_point[i] = None
        self.steps_in_phase[i] = 0
        self.quiet[i] = 0
        self.holding[i] = False
        self.watchdog_strikes[i] = 0
        self.unplanned[i] = False
        self.abandoned[i] = False
        self.stats[i] = self._fresh_stats()
        left = self.left_brick.data.force_matrix_w[i, 0].norm(dim=-1)
        right = self.right_brick.data.force_matrix_w[i, 0].norm(dim=-1)
        held = int(held_brick_from_forces(left[None], right[None], HELD_MIN_FORCE)[0])
        scene = self.snapshot(i)
        if held >= 0 and float(scene.brick_pos[held, 2]) > oracle.LIFT_HEIGHT - LIFT_MIN_RISE:
            # RULING P32, tightened on re-review: decline a carry the clamped LIFT target
            # cannot clear LIFT_MIN_RISE from. A takeover may only enter LIFT from a carry low
            # enough that the clamped target still clears the rise gate -- the LIFT entry this
            # resume would take reads its target as an absolute clearance of LIFT_HEIGHT above
            # the desk, so a carry any higher plans a walk DOWN with the jaws closed. Above
            # LIFT_HEIGHT that walk is a negative rise outright; in the band between
            # LIFT_HEIGHT - LIFT_MIN_RISE and LIFT_HEIGHT it still clamps to LIFT_HEIGHT, so
            # the commanded rise from the live `brick_start` lands under LIFT_MIN_RISE and
            # `_lift_result` returns `lifted=False` with `pinch_ok=True` -- core.oracle routes
            # that to REOPEN, whose jaws open and drop a brick that was genuinely held. Either
            # way the normal retry usually re-picks it off the desk, so the cycle scores as a
            # stowed takeover and every one of its frames is KEPT: an expert demonstration of
            # opening the jaws on a held brick, which is far worse than a missed takeover. It
            # is learned rather than crashed on. Declining costs a takeover and nothing else --
            # the policy keeps the env and a later trigger catches it once the brick is back
            # down. Design 6.2's own escape hatch: return False, leave the env idle.
            self._go_idle(i, None)
            return False
        orc.resume(scene, held)
        if target is not None and held < 0 and not orc.aim_hover_at(scene, target):
            self._go_idle(i, None)
            return False
        self.brick_start[i] = self._brick_pos(i, held) if held >= 0 else None
        self.idle[i] = False
        self.single_cycle[i] = True
        self.takeover_result[i] = None
        self._enter_phase(i, scene)
        if self.goals[i] is None or self.unplanned[i]:
            self._go_idle(i, None)
            return False
        return True

    def _go_idle(self, i: int, stowed: bool | None) -> None:
        """End env ``i``'s takeover: the policy has it again."""
        self.idle[i] = True
        self.single_cycle[i] = False
        self.goals[i] = None
        self.takeover_result[i] = stowed

    def episode_stats(self, env_id: int) -> dict:
        """Everything a metadata sidecar needs about the finished episode.

        Counters from ``self.stats`` (not ``EnvOracle.retries_used``, which resets to 0 at
        every new brick's HOVER -- see ``_enter_phase``'s REOPEN branch), so those stay
        episode-totals. ``induced_offset``/``induced_noise`` read from the ``EnvOracle``
        object instead, because they ARE already episode-cumulative there.

        EVERY field here is readable until the caller calls ``rearm()`` on that env, and for
        two of them that took work rather than luck. The counters and the ``EnvOracle`` are
        plain Python state this class owns, so they simply survive. ``succeeded`` and
        ``bricks_delivered`` are not: they come from the env's success term and from a sample
        of the bricks, and the next ``env.step()`` overwrites both for the WHOLE batch,
        rearmed or not. So they are frozen the step the episode ends (``_freeze_outcomes``)
        and served from that snapshot afterwards. Without it the batch pattern -- finish env
        A, keep stepping so env B can finish, then read A -- would have reported A's next
        episode under A's finished one's counters.

        THREE FIELDS DESCRIBE HOW THE EPISODE WENT AND THEY ARE NOT THE SAME NUMBER. Reach
        for the right one:

        * ``succeeded`` -- what the recorder's export filter itself gates on
          (``termination_manager``'s ``success`` term, i.e. ``mdp.all_bricks_stowed``). This
          is the LABEL. Like the recorder's own read it is the terminating step's value, so
          read it before stepping the env again.
        * ``bricks_delivered`` -- how many bricks were at rest in their own colour's bin.
          Partial credit, for the episodes that did not succeed. The sample it comes from is
          one control step stale by construction (see ``step()``), so on a SUCCESSFUL episode
          -- where the last brick often lands on the very step that ends it -- it is taken
          from ``succeeded`` instead, which is not a substitution but the predicate's own
          meaning: ``all_bricks_stowed`` IS "every brick is in its bin".
        * ``sort_attempts`` -- pick-and-place cycles the ORACLE completed. Counts a brick it
          dropped in transit. Never a delivery count.

        A heap oracle's ``cycle_outcomes`` (from ``extra_stats``) has the same staleness as
        ``bricks_delivered`` and for the same reason: the brick that ends a successful episode
        is marked sorted by ``RETREAT``'s own report, but ``dones`` gates ``_advance_phases``
        off the very step termination fires, which for that brick lands mid-``RELEASE`` hold,
        before ``RETREAT`` ever starts. So ``cycle_outcomes`` is one entry short of
        ``bricks_delivered`` on every successful episode, structurally, not by chance. Padded
        below with the same trust ``bricks_delivered`` already places in ``succeeded``: on a
        successful episode, every brick that has no outcome yet arrived.
        """
        orc = self.oracles[env_id]
        outcome = self._outcome[env_id]
        if outcome is None:  # still running: the live values are this episode's own
            outcome = self._episode_outcome(env_id)
        # The brick indices this episode targeted, in order (retries are not repeated
        # entries). The acceptance discriminator for --randomize_pick_order: run_meta can
        # only say the flag was set, while a campaign whose episodes all read
        # [0, 1, 2, 3, 4, 5] here shipped the fixed R,B,Y,G,R,B schedule regardless. Also
        # readable with the flag off, as the positive record that it WAS fixed. An idle env
        # has no oracle, so it reads zeros and an empty pick order.
        induced = {"induced_offset": 0, "induced_noise": 0, "hover_jitters": 0, "pick_order": []}
        if orc is not None:
            induced = {
                "induced_offset": orc.induced_offset,
                "induced_noise": orc.induced_noise,
                "hover_jitters": orc.hover_jitters,
                "pick_order": list(orc.pick_order),
            }
            extra = getattr(orc, "extra_stats", None)
            if extra is not None:
                induced.update(extra())
        result = {**self.stats[env_id], **outcome, **induced}
        if outcome["succeeded"] and "cycle_outcomes" in result:
            shortfall = result["bricks_delivered"] - len(result["cycle_outcomes"])
            if shortfall > 0:
                result["cycle_outcomes"] = result["cycle_outcomes"] + [True] * shortfall
        return result

    def _succeeded(self, env_id: int) -> bool | None:
        """The env's own ``success`` termination for the step that ended the episode.

        The same tensor ``RecorderManager.record_pre_reset`` grades an episode from, and it
        is not recomputed by the reset, so it survives long enough for a caller that reads it
        before stepping again. ``None`` when the env has no ``success`` term at all, rather
        than a fabricated False -- a cfg without one is not a failed episode.
        """
        manager = getattr(self.env, "termination_manager", None)
        if manager is None or "success" not in manager.active_terms:
            return None
        return bool(manager.get_term("success")[env_id].item())

    def _abandon_nan_envs(self, dones: torch.Tensor) -> None:
        """Abandon any env whose robot articulation OR brick rotation state has gone NaN
        (physics divergence: seen live as a flood of USD OrthogonalizeBasis warnings followed
        by a NaN jaw heading crashing fold_jaw_heading -- either from a NaN robot joint/tool
        pose directly, or from a NaN brick root_quat_w flowing into goal.yaw via
        _brick_headings/scene.brick_yaw with the robot state itself still clean). Abandoning
        routes the episode to the export filter's discard pile, exactly like a watchdog trip,
        instead of killing the whole batch process.

        Batched into one `.cpu()` transfer for the whole env-and-brick check (mirrors
        `_brick_headings`'s reasoning) instead of two `bool(...)` device syncs per env.
        """
        joint_nan = torch.isnan(self.robot.data.joint_pos[:, self.arm_ids]).any(dim=1)
        tool_nan = torch.isnan(self.robot.data.body_quat_w[:, self.tool]).any(dim=1)
        brick_quats = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_quat_w for b in range(self.num_bricks)]
        )
        brick_nan = torch.isnan(brick_quats).any(dim=-1).any(dim=0)
        nan_env = (joint_nan | tool_nan | brick_nan).cpu()
        for i in range(self.num_envs):
            if dones[i] or self.goals[i] is None:
                continue
            if nan_env[i]:
                self.abandoned[i] = True
                self.goals[i] = None
                if self.stats[i]["abandon"] is None:
                    self.stats[i]["abandon"] = "nan_state"

    def _run_watchdog(self, dones: torch.Tensor) -> None:
        """Joint-space watchdog: 30 consecutive flagged steps abandons the env."""
        joint_pos = self.robot.data.joint_pos[:, self.arm_ids]
        joint_vel = self.robot.data.joint_vel[:, self.arm_ids]
        for i in range(self.num_envs):
            if dones[i]:
                continue  # post-auto-reset state this step; joint_pos/vel are the next episode's
            if self.goals[i] is None:
                continue
            tripped = oracle.watchdog_trips(joint_pos[i], joint_vel[i], self.limits)
            if tripped and not oracle.watchdog_trips(
                joint_pos[i], joint_vel[i], self.limits, guard_rad=0.0
            ):
                # Near a stop, not past it, not fast: a jam only if off the commanded target.
                tracking = (joint_pos[i].double().cpu() - self.q_cmd[i]).abs().max()
                tripped = float(tracking) >= WATCHDOG_TRACKING_TOL_RAD
            self.watchdog_strikes[i] = self.watchdog_strikes[i] + 1 if tripped else 0
            if self.watchdog_strikes[i] >= WATCHDOG_TRIP_STRIKES:
                self.abandoned[i] = True
                self.goals[i] = None
                if self.stats[i]["abandon"] is None:
                    self.stats[i]["abandon"] = "watchdog"

    # -- frames --------------------------------------------------------------
    #
    # Every position below is env-origin subtracted (see the module docstring): it is what
    # `SceneSnapshot` documents and what makes `pick_point` directly comparable to
    # `Goal.target_pos`, whatever env produced it.

    def q_arm(self, i: int) -> torch.Tensor:
        """This env's six arm joint positions, float64 CPU, URDF order."""
        return self.robot.data.joint_pos[i, self.arm_ids].double().cpu()

    def tool_quat(self, i: int) -> torch.Tensor:
        return self.robot.data.body_quat_w[i, self.tool]

    def tool_tilt(self, i: int) -> float:
        """Measured tool-axis tilt off world -z, rad. 0 is a level (straight-down) tool."""
        axis = quat_apply(self.tool_quat(i), self.tool_axis)
        return math.acos(max(-1.0, min(1.0, -axis[2].item())))

    def attitude_reference(self, i: int) -> tuple[float, float]:
        """The (tool tilt, jaw heading) this env is currently commanding, rad.

        CAPTURE THIS AT PHASE ENTRY if you mean to grade a phase at its END. A caller
        watching ``goals[i]`` for a transition only sees the change after ``_enter_phase``
        has already planned the NEXT leg, and planning sets these -- so reading them at that
        moment reads the SUCCESSOR's commanded attitude, not the ended phase's. The check
        scripts capture at entry for exactly this reason (it is the same discipline the
        ``AttitudeHold`` reference needed before Task D5).
        """
        return (self.commanded_tilt[i], self.commanded_yaw[i])

    def attitude_error(self, i: int, reference: tuple[float, float] | None = None) -> float:
        """How far the measured tool attitude is from the commanded one, rad.

        The larger of the tilt error and the jaw-heading error, the heading folded modulo pi
        because a jaw is symmetric under a half turn (the same fold ``core.global_ik`` puts
        on its own heading residual). Two scalars rather than one rotation angle because the
        commanded attitude IS two scalars plus a pinned lean azimuth -- there is no commanded
        quaternion left to difference against.

        ``reference`` defaults to what is commanded right now; pass an ``attitude_reference``
        captured earlier to grade against what was commanded then.
        """
        tilt_ref, yaw_ref = reference if reference is not None else self.attitude_reference(i)
        tilt_error = abs(self.tool_tilt(i) - tilt_ref)
        heading_error = abs(float(fold_to_half_turn(self.jaw_yaw(i) - yaw_ref)))
        return max(tilt_error, heading_error)

    def pick_point(self, i: int) -> torch.Tensor:
        """Env-local position of the point on the tool axis a desk-resting brick occupies."""
        origin = self.env.scene.env_origins[i]
        return (
            self.robot.data.body_pos_w[i, self.tool]
            - origin
            + quat_apply(self.tool_quat(i), self.pick_offset[i])
        )

    def jaw_yaw(self, i: int) -> float:
        """World heading of the jaw closing axis (JAW_AXIS_BODY in the tool frame), flat."""
        v = quat_apply(self.tool_quat(i), self.jaw_axis)
        return math.atan2(v[1].item(), v[0].item())

    def _brick_pos(self, i: int, brick: int) -> torch.Tensor:
        return self.env.scene[f"brick_{brick}"].data.root_pos_w[i] - self.env.scene.env_origins[i]

    def _brick_headings(self, i: int) -> torch.Tensor:
        """(B,) CPU: every brick's long-axis heading, as the jaw-heading fold folds against.

        Batched into one ``quat_apply`` plus one ``.cpu()`` transfer for the whole env
        instead of a per-brick ``.item()`` sync, since ``snapshot()`` needs this on CPU
        anyway (see its docstring).
        """
        quats = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_quat_w[i] for b in range(self.num_bricks)]
        )
        long_axis = quat_apply(quats, self.ex.expand(self.num_bricks, 3)).cpu()
        return torch.atan2(long_axis[:, 1], long_axis[:, 0])

    def wrist_roll(self, i: int) -> float:
        """Wrist-roll joint position, rad -- the joint the jaw-heading fold spends."""
        return self.robot.data.joint_pos[i, self.arm_ids[WRIST_ROLL_INDEX]].item()

    def _base_yaw_range(
        self, i: int, hold_xy: tuple[tuple[float, float], ...]
    ) -> tuple[float, float]:
        """Signed joint1 excursion, relative to now, over ``hold_xy``.

        Each waypoint's joint1 is approximated by its horizontal bearing, which OpenYAM's
        arm's joint1 tracked directly: MEASURED, on OpenYAM's arm,
        ``atan2(pick_y, pick_x) - joint1 = 0.006 +- 0.097 rad`` over a full 13437-step
        instrumented run there. Not yet re-validated on the Maker Arm. 0.0 is always in the
        range because the arm is at the current pick point now. Empty ``hold_xy`` (every
        phase that does not command a yaw) collapses to no swing.
        """
        if not hold_xy:
            return (0.0, 0.0)
        pick = self.pick_point(i).tolist()
        here = math.atan2(pick[1], pick[0])
        swings = [0.0] + [oracle.wrap_to_pi(math.atan2(y, x) - here) for x, y in hold_xy]
        return (min(swings), max(swings))

    def snapshot(self, i: int) -> oracle.SceneSnapshot:
        """Built on CPU: ``EnvOracle`` (core.oracle, deliberately isaaclab/device-free)
        compares this against ``layout.bin_interior_centers()``/``bin_half_extents()``
        inside ``checker.bricks_at_rest_in_bins`` -- plain CPU tensors -- so a CUDA snapshot
        raises at the first mixed-device comparison (the first HOVER-entry brick lookup).
        Position/attitude reads still go through ``pick_point``/``_brick_pos``/``tool_quat``
        directly (self.device) -- only the values handed to ``core.oracle`` move to CPU here.
        """
        brick_pos = torch.stack([self._brick_pos(i, b) for b in range(self.num_bricks)]).cpu()
        brick_lin_vel = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_lin_vel_w[i] for b in range(self.num_bricks)]
        ).cpu()
        brick_quat = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_quat_w[i] for b in range(self.num_bricks)]
        ).cpu()
        return oracle.SceneSnapshot(
            brick_pos=brick_pos,
            brick_yaw=self._brick_headings(i),
            brick_lin_vel=brick_lin_vel,
            pick_point=self.pick_point(i).cpu(),
            brick_quat=brick_quat,
        )

    def pinch_evidence(
        self,
        i: int,
        brick: int,
        min_stall: float = PINCH_MIN_STALL,
        min_force: float = PINCH_MIN_FORCE,
    ) -> dict:
        """Is the brick pinched between the jaws, or merely entrained in the pocket?

        ``force_matrix_w`` is (num_envs, num_bodies, num_filters, 3); each sensor prim is a
        single finger body and the filters are the bricks in scene order, so ``[i, 0, brick]``
        isolates this env's this brick.
        """
        left = self.left_brick.data.force_matrix_w[i, 0, brick].norm().item()
        right = self.right_brick.data.force_matrix_w[i, 0, brick].norm().item()
        joints = self.robot.data.joint_pos[i, self.finger_ids].tolist()
        stall = min(abs(value - GRIPPER_CLOSED_POS) for value in joints)
        weakest = min(left, right)
        strongest = max(left, right)
        asymmetry = abs(left - right) / strongest if strongest > 0 else 1.0
        return {
            "stall": stall,
            "left": left,
            "right": right,
            "asymmetry": asymmetry,
            "ok": bool(
                stall >= min_stall and weakest >= min_force and asymmetry <= PINCH_MAX_ASYMMETRY
            ),
        }
