"""Scripted grasp + brick contact-stability validation (GPU).

The physics proof this milestone hangs on (design spec section 5.3): real-size 31.8 x 15.8
x 9.6 mm bricks must rest stably on the desk, and the arm must be able to pick one up
autonomously.

**Approved spec deviation (owner, 2026-09-06).** Spec section 5 item 4 specifies this check
runs on ``MakerArm-LegoSort-IK-Rel-v0``. It now runs on ``MakerArm-LegoSort-v0``, the
joint-position action space, and computes joint targets with ``core.global_ik`` instead of
asking Isaac Lab's differential IK to discover them. **The dataset contract is unaffected**:
``mdp/recorders.py`` always recorded the post-IK joint-position target, so the action stream
is the same quantity in the same units either way -- what changed is how the oracle produces
them. This must be carried into the F1 milestone report.

Why the change. Task D3 measured Isaac Lab's differential IK to be correctly wired -- the
Jacobian row it slices is `grasp_center`'s own to within 0.077 mm/rad of a kinematic finite
difference -- and nonetheless unable to do this job: as a damped-least-squares local solver
with no joint-limit avoidance it spends ``link_005_joint``'s 0.30 rad of home margin instead
of turning the base, jams against the stop, and every later phase then servos a dead
mechanism. Over six rounds it never once closed the jaws on a brick; ``finger_total_force``
and ``peak_brick_force`` read 0.00/0.00 N throughout. The global solver carries the brick.

Run:  $ISAACLAB_PATH/isaaclab.sh -p scripts/scripted_grasp_check.py --headless

Every phase, including the settle, is driven through the 7-element joint-position action
``[q1..q6 gripper]``. Nothing is written around the action pipeline, so ``arm_action`` still
passes every target through maker_arm_rs's clamp.

  0. settle   240 steps with the arm's joint targets pinned at HOME_JOINT_POS, jaws open.
              Gates rest stability: every brick's drift over the second half of the settle
              must be < 5 mm and its final speed < 0.01 m/s.
  1. hover    approach home -> the globally-solved hover pose, interpolated IN JOINT SPACE
              (continuous by construction, no IK branch to flip), putting the PICK point
              0.12 m above the target brick with the tool axis at frames.GRASP_TILT_DEG off
              world -z, leaning toward world +x, and the jaws squared to the brick.
  2. descend  a straight Cartesian line down to the brick, every waypoint solved globally
              with continuity enforced (|dq| <= global_ik.MAX_WAYPOINT_STEP per waypoint).
  3. close    gripper action -1.0 while holding the descend's joint solution.
  4. lift     a straight Cartesian line 0.10 m up, jaws closed, same waypoint treatment.

Success needs the brick to have risen more than 0.05 m and finished within 0.03 m
(horizontally) of the tool axis, AND to be genuinely pinched: the jaws stalled short of
``GRIPPER_CLOSED_POS``, and equal, non-trivial force between each finger and that brick
specifically (filtered per-pair contact sensors, so the desk press cannot be mistaken for a
grip). Position alone cannot separate a pinch from the entrainment the brief's amendment 3
warns about -- fingers fully closed with the brick cammed into the jaw pocket rises just as
far.

This script also has two measurement-only modes, run before any grasp trial and gated on
their own criteria (Task D3):

* ``--gripper_probe`` holds the gripper open then closed and reports the raw joint travel,
  to catch an inverted jaw before anything downstream trusts the open/close convention.
* ``--signs`` nudges the base and wrist-roll joints one at a time and reads the jaw heading
  before/after, to measure ``WRIST_ROLL_SIGN`` / ``BASE_YAW_SIGN``
  (``core.oracle.fold_jaw_heading``'s kinematic sign relations) on this arm.

Same AppLauncher/``os._exit()`` skeleton as smoke_lego_sort.py and smoke_usd.py.
"""

import argparse
import itertools
import json
import math
import os
import random
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-v0")
parser.add_argument("--num_bricks", type=int, default=6)
parser.add_argument(
    "--brick",
    type=str,
    default="auto",
    help="Index of the brick to pick, or 'auto' (default) for a seeded-random "
    "reachable brick. Random rather than nearest so the gate samples the whole "
    "reachable band instead of always testing the easiest brick in the scene.",
)
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--seed", type=int, default=0, help="First trial's seed; trials use seed+i.")
parser.add_argument("--settle_steps", type=int, default=240)
parser.add_argument(
    "--x_sweep",
    type=str,
    default="",
    help="Comma-separated brick x positions. Teleports a single brick to each "
    "(y=0) and grasps it, mapping the top-down reachable region. Gated: every "
    "listed x must be grasped, so sweeping a failing range exits non-zero.",
)
parser.add_argument("--contact_sensor", type=int, default=1)
parser.add_argument(
    "--descend_clearance",
    type=float,
    default=None,
    help="Override DESCEND_CLEARANCE (m): extra depth the descend commands the pick "
    "point past the brick centre. See the constant's comment.",
)
parser.add_argument(
    "--pick_offset_z",
    type=float,
    default=None,
    help="Override PICK_OFFSET_MEASURED[2] (m), along the tool axis. Used to probe "
    "the offset before it is measured and pinned in robots/frames.py.",
)
parser.add_argument(
    "--descend_tilt_deg",
    type=float,
    default=None,
    help="Pin the DESCEND/LIFT tool tilt (deg off world -z) instead of searching for the "
    "smallest one that brick admits. Default (unset) is ruling D3-11 candidate 2: try "
    "core.global_ik.descend_tilt_ladder in order and take the first tilt whose descend "
    "path solves. The tilt is near level because at the hover tilt the jaw plates foul the "
    "desk at EVERY pick offset -- the plate face is 82.5 mm across the tool axis against "
    "34.2 mm along it, so 20 deg drops its far corner 21.4 mm below the axial tip -- and it "
    "is never EXACTLY level because at vertical the lean azimuth is undefined.",
)
parser.add_argument(
    "--gripper_probe",
    action="store_true",
    help="Print jaw travel direction under open/close actions, then exit "
    "(Task D3 step 2 -- must be run and confirmed before --signs or a trial).",
)
parser.add_argument(
    "--signs",
    action="store_true",
    help="Print WRIST_ROLL_SIGN/BASE_YAW_SIGN measured on this arm's home pose, "
    "then exit (Task D3 step 3).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab (and friends) are only importable once AppLauncher has launched Kit;
# the deferral is deliberate, as in smoke_usd.py.
import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.core import layout
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.core.checker import PINCH_MIN_FORCE, PINCH_MIN_STALL
from maker_arm_lab.core.global_ik import (
    JAW_TIP_REACH,
    MAX_WAYPOINT_STEP,
    GlobalIK,
    GraspTarget,
    JointTrim,
    descend_pick_offset,
    descend_tilt_ladder,
    tool_axis_world,
    wrap_to_pi,
)
from maker_arm_lab.robots.frames import (
    ARM_JOINT_NAMES,
    DESCEND_CLEARANCE_MEASURED,
    GRASP_TILT_RAD,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_POS,
    JAW_AXIS_BODY,
    PICK_OFFSET_MEASURED,
    TOOL_AXIS_BODY,
    TOOL_BODY,
)

# --- Geometry, measured on THIS arm by this script (Task D3) -----------------
#
# Measured geometry is imported, never retyped -- the pick offset and the descend
# clearance live next to PICK_OFFSET_MEASURED/DESCEND_CLEARANCE_MEASURED in
# maker_arm_lab.robots.frames, and the top-down pick limit lives next to SCATTER_X
# in maker_arm_lab.core.layout, so the spawn bound and the pick bound cannot
# silently drift apart. The mechanism (not this arm's numbers) carried over from
# OpenYAM: the jaws reach some distance past the tool frame along the tool axis,
# so a brick resting on the desk is gripped near the jaw tips rather than at the
# tool frame itself. ``--pick_offset_z``/``--descend_clearance`` exist so this
# script can probe both before they are pinned.
PICK_OFFSET = (
    PICK_OFFSET_MEASURED[0],
    PICK_OFFSET_MEASURED[1],
    args.pick_offset_z if args.pick_offset_z is not None else PICK_OFFSET_MEASURED[2],
)
DESCEND_CLEARANCE = (
    DESCEND_CLEARANCE_MEASURED if args.descend_clearance is None else args.descend_clearance
)
TOP_DOWN_X_LIMIT = layout.TOP_DOWN_X_LIMIT

# --- The grasp attitude ------------------------------------------------------
# The tool leans toward world +x. This is the convention that ships, and Task D3 part 8
# measured that it is also the one to keep: freeing the lean azimuth (a 5-DOF relaxation)
# more than doubles the band-wide worst-case joint margin and covers band corners this
# convention cannot reach, but it GRASPS WORSE -- 4/9 against 8/9 -- because a redundancy
# resolved for joint margin picks poses that are bad for the grasp. See core.global_ik.
LEAN_AZIMUTH = 0.0

# Ruling D3-10: the descend is near level by default. GRASP_TILT_DEG keeps its meaning as
# the APPROACH tilt and is untouched; these are different constants for different phases.
# Ruling D3-11 candidate 2: rather than one fixed descend tilt, try the ladder in order and
# take the smallest tilt whose descend path actually solves. Every rung is inside the
# fouling window by construction (core.global_ik.descend_tilt_ladder), so raising the tilt
# to buy path feasibility can never buy it by putting the plates into the desk.
BRICK_HALF_HEIGHT = layout.BRICK_SIZE[2] / 2
DESCEND_TILT_LADDER = (
    [math.radians(args.descend_tilt_deg)]
    if args.descend_tilt_deg is not None
    else descend_tilt_ladder(BRICK_HALF_HEIGHT)
)
# The offset the HOVER is solved at. The descend picks its own per rung, but the hover has
# to commit before the ladder runs; the rungs differ by ~1-2 mm of offset, far inside the
# continuity cap, so the first descend waypoint reconverges from the hover pose without
# trouble. --pick_offset_z overrides it, for probing.
NOMINAL_OFFSET_Z = (
    PICK_OFFSET[2]
    if args.pick_offset_z is not None
    else descend_pick_offset(DESCEND_TILT_LADDER[0], BRICK_HALF_HEIGHT)[0]
)
# The brick's position ACROSS the plate face (body x), which is the axis the plate is
# asymmetric on: it spans -62.5..+20.5 mm about grasp_center, so a brick centred on
# grasp_center sits 21 mm off the plate's own centre, toward the short edge, with only
# 4.6 mm to spare once its own 15.9 mm half-length is counted. That is what the 10-trial
# gate's single DROPPED was -- lateral +20.5 mm against a plate ending at +20.5 mm.
#
# Ruling D3-11 candidate 3 asked for the jaw-heading FOLD to be chosen by plate side. It
# cannot help: a half-turn roll about the tool axis maps the brick's long axis onto itself,
# so both representatives put the brick at the same body x. Measured, both folds ->
# +8.63 mm, identical to 2 dp. What the ruling actually wants -- a rule that cannot place a
# brick off the end of its own plate -- is this offset, which is the constant designed for
# it and which was an unmeasured 0.0 until now. Centring the brick on the plate face turns
# 4.6 mm of edge margin into 25.6 mm on BOTH edges.
PICK_OFFSET_LATERAL = PICK_OFFSET[0]

# --- Path resolution ---------------------------------------------------------
# The approach is joint-space, so its waypoint count only sets how gently the arm leaves
# home. The Cartesian legs are what the continuity constraint acts on: 24 waypoints over a
# ~0.09-0.12 m leg is ~4-5 mm apiece, and the Jacobian columns measured in part 7 are
# 0.23-0.27 m/rad, so the geometry needs ~0.02 rad per waypoint against the 0.12 rad cap.
APPROACH_WAYPOINTS = 40
# 40, not 24, because the descend now sweeps the TILT as well as the position: it rotates
# the tool from the hover tilt to the descend tilt over the same leg, and that rotation adds
# joint motion per waypoint on top of the translation. Measured at 24 waypoints with a level
# descend: "NO CONTINUOUS SOLUTION along the descend" on 1 trial in 4, where the combined
# translate-and-rotate step could not fit inside the 0.12 rad continuity cap.
CARTESIAN_WAYPOINTS = 40
# Restart spread for the ONE solve that has no previous waypoint to seed from. Deliberately
# larger than core.global_ik's default: this solve is also what decides whether a brick is
# reported UNREACHABLE, and calling a pose infeasible because the search was thin would
# blame the layout for the solver's own weakness. Measured at ~1.7 s, once per trial.
HOVER_RESTARTS = 64
# Control steps held at each intermediate waypoint, and at the end of a leg. The end-of-leg
# hold must outlast the trim's ~33-step time constant several times over or the trim's own
# transient is reported as a pose error; see core.global_ik.JointTrim.
WAYPOINT_STEPS = 4
SETTLE_AT_WAYPOINT = 200

# --- Acceptance thresholds ---------------------------------------------------
REST_MAX_DRIFT = 0.005  # m, over the second half of the settle
REST_MAX_SPEED = 0.01  # m/s, at the end of the settle
LIFT_MIN_RISE = 0.05  # m
LIFT_MAX_XY_ERROR = 0.03  # m

# Pinch evidence, gated after the lift so the readings are pure finger<->brick
# (in free air there is no desk contact to confuse them). Position alone cannot
# tell a pinch from the entrainment the brief's amendment 3 warns about: fingers
# fully closed at GRIPPER_CLOSED_POS with the brick merely cammed into the jaw
# pocket also rises 98 mm and also finishes on the tool axis. The two signals
# that do separate them are (a) the jaws stalling short of their target because
# something is between them and (b) equal force on both fingers.
#
# PINCH_MIN_FORCE (2.0 N on the weaker finger) is imported from core.checker
# rather than redefined here: this measurement set is where the OpenYAM number
# came from, but the oracle's runtime check and the eval harness's strict grasp
# grading use the same number, and it was three copies of 2.0 there until it
# was consolidated.
PINCH_MAX_ASYMMETRY = 0.30  # |L - R| / max(L, R)

# Jaw-plate extent relative to grasp_center, on the three axes that gate a grasp, read off
# the vendored collision geometry quoted in robots/frames.py's wrist-camera block. These are
# what the descend telemetry prints against, so a miss says WHICH way it missed.
#   jaw travel (body y): the jaws close along +-GRIPPER_OPEN_POS, so a brick further out
#     than half the opening minus half its own width can never be reached by a closing jaw.
#   lateral (body x): the plate face spans -0.0625 .. +0.0200; past that the brick is off
#     the end of the plate.
#   tool axis: the plate reaches 0.0342 m past grasp_center and 0.0664 m back toward the
#     wrist. A desk-resting brick necessarily sits near the far end of that span, because
#     the tips have to clear the desk -- which is exactly what PICK_OFFSET_MEASURED pins.
JAW_CLOSURE_BUDGET = GRIPPER_OPEN_POS - 0.5 * layout.BRICK_SIZE[1]
JAW_PLATE_LATERAL = (-0.0625, 0.0200)
JAW_TIP_BEYOND_GRASP_CENTER = JAW_TIP_REACH
JAW_PLATE_BEHIND_GRASP_CENTER = 0.0664
# The eight corners of the jaw-plate collision box in grasp_center coordinates, from the
# same vendored AABBs. Used to report how far the plates actually are from the desk, which
# the axial tip alone does not tell you.
JAW_PLATE_CORNERS = torch.tensor(
    [
        [x, y, z]
        for x in JAW_PLATE_LATERAL
        for y in (-0.0303, 0.0302)
        for z in (-JAW_TIP_BEYOND_GRASP_CENTER, JAW_PLATE_BEHIND_GRASP_CENTER)
    ],
    dtype=torch.float32,
)

HOVER_HEIGHT = 0.12
LIFT_HEIGHT = 0.10
CLOSE_STEPS = 60
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0

# Parking spots for the bricks the x-sweep is not picking: on the desk (which
# spans x in [-0.15, 0.75], y in [-0.45, 0.45]), beyond the shelf and well clear
# of the arm. Parking them off the desk edge instead would leave them free-falling
# for the rest of the run, which silently makes the neighbour-disturbance figure
# meaningless.
PARK_X = 0.66
PARK_Y0 = -0.40
PARK_Y_SPACING = 0.04


class GraspRig:
    """Everything the phases need: frames, the global solver, the driver, and telemetry."""

    def __init__(self, env, sensor):
        self.env = env
        self.sensor = sensor
        self.robot = env.scene["robot"]
        self.device = env.device
        self.ik = GlobalIK()
        self.trim = JointTrim()
        self.tool = self.robot.find_bodies(TOOL_BODY)[0][0]
        # Joint order resolved from the articulation, never hard-coded (the
        # pattern check_success_teleport.py established on OpenYAM).
        self.finger_ids = self.robot.find_joints(list(GRIPPER_JOINT_NAMES))[0]
        self.arm_joint_ids = self.robot.find_joints(list(ARM_JOINT_NAMES))[0]
        self.action_dim = env.action_manager.total_action_dim
        if self.action_dim != 7:
            raise RuntimeError(f"expected a 7-element joint-position action, got {self.action_dim}")
        self.pick_offset = torch.tensor(PICK_OFFSET, device=self.device).unsqueeze(0)
        self.ex = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.jaw_axis = torch.tensor([JAW_AXIS_BODY], device=self.device)
        self.tool_axis = torch.tensor([TOOL_AXIS_BODY], device=self.device)
        limits = self.robot.data.joint_pos_limits[0, self.arm_joint_ids]
        self.q_lower = limits[:, 0].double().cpu()
        self.q_upper = limits[:, 1].double().cpu()
        self.finger_sensor_ids = (
            [sensor.body_names.index("gripper_left"), sensor.body_names.index("gripper_right")]
            if sensor is not None
            else []
        )
        # Filtered per-pair sensors: finger <-> brick force only. The unfiltered
        # net_forces_w above sums EVERY contact on the finger body, so during the
        # descend it is dominated by any desk press and cannot be read as a grip
        # force. The two instruments are kept separate and labelled separately.
        self.brick_pair = (
            (env.scene["finger_left_brick"], env.scene["finger_right_brick"])
            if sensor is not None
            else None
        )

    # -- driving, always through the action pipeline ---------------------------
    def drive(self, q, gripper: float, steps: int = 1) -> None:
        """Command an absolute joint target for ``steps`` control steps.

        ``self.trim`` biases the command to cancel the arm's open-loop droop; see
        ``core.global_ik.JointTrim`` for what that droop actually is (effort-limit
        saturation on link_004_joint and link_007_joint, NOT missing gravity
        compensation -- the actuator does apply G(q) and Task D3 part 9 confirmed it
        was active). The bias is joint space only, so this still drives the action
        pipeline and maker_arm_rs's clamp.
        """
        q = torch.as_tensor(q, dtype=torch.float64).reshape(6).cpu()
        action = torch.zeros(1, 7, device=self.device)
        action[0, 6] = gripper
        for _ in range(steps):
            command = torch.clamp(self.trim.command(q), self.q_lower, self.q_upper)
            action[0, :6] = command.to(device=self.device, dtype=torch.float32)
            self.env.step(action)
            self.trim.update(q, self.q_arm())

    def hold(self, steps: int, gripper: float) -> None:
        """Hold wherever the arm currently is, for ``steps`` control steps."""
        self.drive(self.q_arm(), gripper, steps)

    def q_arm(self) -> torch.Tensor:
        return self.robot.data.joint_pos[0, self.arm_joint_ids].double().cpu()

    def home_q(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos[0, self.arm_joint_ids].double().cpu()

    # -- frames ---------------------------------------------------------------
    def tool_quat(self):
        return self.robot.data.body_quat_w[:, self.tool]

    def pick_point(self):
        """World position of the point on the tool axis that a desk-resting brick occupies."""
        return self.robot.data.body_pos_w[:, self.tool] + math_utils.quat_apply(
            self.tool_quat(), self.pick_offset
        )

    def tool_angle_deg(self) -> float:
        axis = math_utils.quat_apply(self.tool_quat(), self.tool_axis)[0]
        return math.degrees(math.acos(max(-1.0, min(1.0, (-axis[2]).item()))))

    def jaw_yaw(self) -> float:
        """World heading of the jaw closing axis (JAW_AXIS_BODY), projected flat."""
        v = math_utils.quat_apply(self.tool_quat(), self.jaw_axis)[0]
        return math.atan2(v[1].item(), v[0].item())

    def brick_pos(self, index: int):
        return self.env.scene[f"brick_{index}"].data.root_pos_w[0]

    def jaw_heading_target(self, brick_index: int) -> float:
        """World heading the jaw axis must take to be square across the brick.

        A jaw is symmetric under a half turn, so the target is folded to whichever of
        ``yaw`` / ``yaw + pi`` the wrist is already nearer -- ``link_007_joint`` only spans
        +-2.7 rad and the far representative can be unreachable.

        This replaces the quaternion ``desired_quat`` the differential IK needed. The solver
        constrains the jaw heading directly, so the attitude is now three scalars (tilt
        magnitude, lean azimuth, jaw heading) rather than a rotation matrix that has to be
        assembled and re-orthogonalised. ``oracle_runtime`` must build its target the same
        way, through the same ``core.global_ik``, so the check and the oracle cannot drift
        apart -- this task has already paid once for exactly that.
        """
        brick_quat = self.env.scene[f"brick_{brick_index}"].data.root_quat_w[0].unsqueeze(0)
        long_axis = math_utils.quat_apply(brick_quat, self.ex)[0]
        want = math.atan2(long_axis[1].item(), long_axis[0].item()) + math.pi / 2
        return want + math.pi * round(wrap_to_pi(self.jaw_yaw() - want) / math.pi)

    def grasp_center_for(self, pick_point, tilt: float, jaw_heading: float, offset_z: float):
        """The `grasp_center` position that puts the PICK point at ``pick_point``.

        ``pick_point = grasp_center + R @ (ox, 0, oz)``, so
        ``grasp_center = pick_point - ox * (body x in world) - oz * (body z in world)``.
        Body z maps to the negated tool axis, which the commanded attitude gives exactly.
        Body x is the plate-face direction: it is ``body y x body z``, and since body y is
        the jaw axis at world heading ``jaw_heading``, body x points at
        ``jaw_heading - 90 deg`` in the horizontal plane, exactly at zero tilt and to within
        the descend tilt otherwise. At the ladder's near-level tilts that approximation is
        worth ``|ox| * sin(tilt)`` = 0.2 mm, which is an order of magnitude below the
        smallest margin it is used to protect.
        """
        axis = tool_axis_world(tilt, LEAN_AZIMUTH)
        face = torch.tensor(
            [math.sin(jaw_heading), -math.cos(jaw_heading), 0.0], dtype=torch.float64
        )
        point = torch.as_tensor(pick_point, dtype=torch.float64).reshape(3).cpu()
        return tuple(float(v) for v in point + offset_z * axis - PICK_OFFSET_LATERAL * face)

    def target_for(
        self,
        pick_point,
        jaw_heading: float,
        tilt: float = GRASP_TILT_RAD,
        offset_z: float = NOMINAL_OFFSET_Z,
    ) -> GraspTarget:
        return GraspTarget(
            position=self.grasp_center_for(pick_point, tilt, jaw_heading, offset_z),
            tilt=tilt,
            jaw_heading=jaw_heading,
            lean_azimuth=LEAN_AZIMUTH,
        )

    def set_pick_offset(self, offset_z: float) -> None:
        """Point the telemetry's PICK frame at the offset this descend actually used."""
        self.pick_offset = torch.tensor(
            [PICK_OFFSET_LATERAL, PICK_OFFSET[1], offset_z], device=self.device
        ).unsqueeze(0)

    def lowest_jaw_corner(self) -> float:
        """World z of the lowest corner of the jaw-plate collision box (desk top is 0).

        The gate that the pick offset is actually pinned by. The axial tip is NOT the
        lowest point: the plate box spans 82.5 mm across the tool axis, so any tool tilt
        drops its far corner well below the tip -- 0.0625 * sin(tilt) of it.
        """
        origin = self.robot.data.body_pos_w[0, self.tool]
        world = origin.unsqueeze(0) + math_utils.quat_apply(
            self.tool_quat().expand(JAW_PLATE_CORNERS.shape[0], -1),
            JAW_PLATE_CORNERS.to(self.device, dtype=origin.dtype),
        )
        return float(world[:, 2].min() - self.env.scene.env_origins[0, 2])

    # -- telemetry ------------------------------------------------------------
    def finger_joints(self) -> list[float]:
        return self.robot.data.joint_pos[0, self.finger_ids].tolist()

    def joint_margins(self) -> list[float]:
        """Each arm joint's distance (rad) to its nearer URDF stop, motor-id order.

        Printed per phase because a jammed joint and a slow servo look identical in a
        position error alone: Task D3 spent two rounds distinguishing them by hand.
        """
        q = self.robot.data.joint_pos[0, self.arm_joint_ids]
        lo = self.robot.data.joint_pos_limits[0, self.arm_joint_ids, 0]
        hi = self.robot.data.joint_pos_limits[0, self.arm_joint_ids, 1]
        return torch.minimum(q - lo, hi - q).tolist()

    def joint_tracking_error(self, q_des) -> float:
        """Worst joint's distance from the commanded target (rad)."""
        q_des = torch.as_tensor(q_des, dtype=torch.float64).reshape(6).cpu()
        return float(torch.max(torch.abs(self.q_arm() - q_des)))

    def finger_total_forces(self) -> tuple[float, float]:
        """Total contact force on each finger body, from EVERY contact.

        During the descend this includes any desk press, not just a grip force. Use
        ``finger_brick_forces`` for the grip.
        """
        if self.sensor is None:
            return (float("nan"), float("nan"))
        forces = self.sensor.data.net_forces_w[0].norm(dim=-1)
        return (
            forces[self.finger_sensor_ids[0]].item(),
            forces[self.finger_sensor_ids[1]].item(),
        )

    def finger_brick_forces(self, brick_index: int) -> tuple[float, float]:
        """Force between each finger and ``brick_{brick_index}`` alone.

        ``force_matrix_w`` is (num_envs, num_bodies, num_filters, 3); the sensor
        prim is a single finger body and the filters are the bricks in scene
        order, so column ``brick_index`` isolates this brick.
        """
        if self.brick_pair is None:
            return (float("nan"), float("nan"))
        left, right = self.brick_pair
        return (
            left.data.force_matrix_w[0, 0, brick_index].norm().item(),
            right.data.force_matrix_w[0, 0, brick_index].norm().item(),
        )

    def pinch_evidence(self, brick_index: int) -> dict:
        """Is the brick pinched between the jaws, or merely entrained in the pocket?"""
        joints = self.finger_joints()
        stall = min(abs(value - GRIPPER_CLOSED_POS) for value in joints)
        left, right = self.finger_brick_forces(brick_index)
        weakest = min(left, right)
        strongest = max(left, right)
        asymmetry = abs(left - right) / strongest if strongest > 0 else 1.0
        return {
            "stall": stall,
            "left": left,
            "right": right,
            "asymmetry": asymmetry,
            "ok": bool(
                stall >= PINCH_MIN_STALL
                and weakest >= PINCH_MIN_FORCE
                and asymmetry <= PINCH_MAX_ASYMMETRY
            ),
        }

    def brick_offsets(self, index: int) -> dict:
        """The brick relative to `grasp_center`, on the axes that actually gate the grasp.

        Resolved and NAMED rather than returned as a raw body-frame triple. In this frame
        the jaws slide along body **y** (``frames.JAW_AXIS_BODY``; the URDF's
        ``gripper_left_joint`` axis is ``(0, 1, 0)`` and ``gripper_right_joint`` is its
        negative), so the component the jaw closure must cover is [1], and [0] is the
        plate's lateral direction. Task D3 part 6 read [0] as the jaw axis and reported the
        wrong mechanism for a failure as a result; naming them is the fix.
        """
        origin = self.robot.data.body_pos_w[0, self.tool]
        body = math_utils.quat_apply_inverse(
            self.tool_quat(), (self.brick_pos(index) - origin).unsqueeze(0)
        )[0]
        return {
            "jaw": body[1].item(),
            "lateral": body[0].item(),
            # Reported as distance ALONG the tool axis (TOOL_AXIS_BODY is body -z), so it
            # compares directly against the plate span.
            "tool": -body[2].item(),
        }


def _margin_str(margins: list[float]) -> str:
    """``j4:+0.00`` -- the tightest arm joint and how far it is from its stop."""
    index = min(range(len(margins)), key=lambda i: margins[i])
    return f"j{index + 1}:{margins[index]:+.2f}"


def _offset_str(offsets: dict) -> str:
    return (
        f"jaw={offsets['jaw'] * 1000:+.1f}mm(budget+-{JAW_CLOSURE_BUDGET * 1000:.1f}) "
        f"lateral={offsets['lateral'] * 1000:+.1f}mm(plate "
        f"{JAW_PLATE_LATERAL[0] * 1000:+.1f}..{JAW_PLATE_LATERAL[1] * 1000:+.1f}) "
        f"tool={offsets['tool'] * 1000:+.1f}mm(plate "
        f"{-JAW_PLATE_BEHIND_GRASP_CENTER * 1000:+.1f}..{JAW_TIP_BEYOND_GRASP_CENTER * 1000:+.1f})"
    )


def _path_str(label: str, path, points) -> str:
    steps = [step.step for step in path[1:]]
    worst = max(range(len(steps)), key=lambda i: steps[i]) if steps else 0
    cartesian = (
        float(torch.linalg.norm(points[worst + 1] - points[worst])) if len(points) > 1 else 0.0
    )
    return (
        f"PATH {label}: {len(path)} waypoints, continuous  "
        f"max_joint_step={max(steps) if steps else 0.0:.4f}rad "
        f"(waypoint {worst + 1}, cartesian step {cartesian * 1000:.1f}mm, cap "
        f"{MAX_WAYPOINT_STEP:.2f})  worst_margin={min(s.margin for s in path):.3f}rad  "
        f"worst_pos_resid={max(s.position_error for s in path) * 1e6:.2f}um"
    )


def check_rest_stability(rig, num_bricks: int, steps: int) -> dict:
    """Phase 0. Hold the home pose with the jaws open and watch the bricks settle.

    Drift is measured over the *second half* of the window. The whole-window
    displacement is dominated by the reset scatter's 15.2 mm drop from
    ``layout.SPAWN_Z`` (0.02 m) to the resting height (half the brick's 9.6 mm
    thickness), which is the brick falling into place, not instability. What the
    spec's rest-stability gate is actually about is whether a settled brick then
    stays put instead of jittering or sinking, so the tail is the honest window.

    On the joint-position action space this hold goes through the action pipeline like
    every other phase. It could not before: the IK-Rel term re-references from the measured
    pose each control step, so a zero delta ratchets gravity sag into the setpoint at about
    0.9 mm/step, and the settle had to bypass the action term entirely to stand still. That
    whole exception is gone. The arm's own drift is still reported on the same second-half
    window as the bricks', so "the settle does not move the arm" stays a logged number.
    """
    half = steps // 2
    home = rig.home_q()
    rig.trim.reset()
    arm_start = rig.pick_point()[0].clone()
    angle_start = rig.tool_angle_deg()
    rig.drive(home, GRIPPER_OPEN_ACTION, half)
    mid = torch.stack([rig.brick_pos(i).clone() for i in range(num_bricks)])
    arm_mid = rig.pick_point()[0].clone()
    angle_mid = rig.tool_angle_deg()
    rig.drive(home, GRIPPER_OPEN_ACTION, steps - half)
    end = torch.stack([rig.brick_pos(i).clone() for i in range(num_bricks)])
    tail = max(steps - half, 1)
    arm_drift = (rig.pick_point()[0] - arm_mid).norm().item() / tail
    angle_drift = (rig.tool_angle_deg() - angle_mid) / tail
    speed = torch.stack(
        [rig.env.scene[f"brick_{i}"].data.root_lin_vel_w[0] for i in range(num_bricks)]
    ).norm(dim=-1)
    drift = (end - mid).norm(dim=-1)
    rest_z = end[:, 2] - rig.env.scene.env_origins[0, 2]
    expected_z = layout.BRICK_SIZE[2] / 2
    return {
        "drift": drift.tolist(),
        "speed": speed.tolist(),
        "rest_z": rest_z.tolist(),
        "z_error": (rest_z - expected_z).abs().max().item(),
        "max_drift": drift.max().item(),
        "max_speed": speed.max().item(),
        # Arm-side drift, over the same second-half window as the bricks'.
        "arm_drift_per_step": arm_drift,
        "arm_total_drift": (rig.pick_point()[0] - arm_start).norm().item(),
        "angle_drift_per_step": angle_drift,
        "angle_start": angle_start,
        "angle_end": rig.tool_angle_deg(),
        "ok": bool(drift.max() < REST_MAX_DRIFT and speed.max() < REST_MAX_SPEED),
    }


def pick_target_brick(rig, num_bricks: int, requested: str, rng: random.Random) -> int:
    """Choose the brick to grasp.

    ``auto`` picks a *uniformly random* reachable brick, not the nearest one.
    Nearest-first sounds sensible and is a trap for a gate: it systematically
    tests the easiest brick in the scene. On OpenYAM, over eight gated trials
    the nearest reachable brick was always at x = 0.18-0.24, never within 50 mm
    of ``TOP_DOWN_X_LIMIT``, even in trials where a reachable brick sat at
    0.288 -- so a regression that broke picks over the top half of the band
    would have left this gate green. Random selection costs nothing and covers
    the band.

    ``rng`` is seeded per trial from the same seed as the scatter, so a failure
    is reproducible from the printed seed alone.
    """
    if requested != "auto":
        return int(requested)
    xs = [rig.brick_pos(i)[0].item() for i in range(num_bricks)]
    reachable = [i for i in range(num_bricks) if xs[i] <= TOP_DOWN_X_LIMIT]
    # No reachable brick should be possible now that layout.SCATTER_X is bounded
    # by TOP_DOWN_X_LIMIT, but fall back to the full set rather than crashing --
    # and let the grasp fail honestly if the scene ever changes underneath.
    return rng.choice(reachable or list(range(num_bricks)))


def grasp_one(rig, brick_index: int, label: str, num_bricks: int) -> bool:
    """Phases 1-4 on ``brick_index``. Prints per-phase telemetry; returns success."""
    brick_start = rig.brick_pos(brick_index).clone()
    others_start = torch.stack(
        [rig.brick_pos(i).clone() for i in range(num_bricks) if i != brick_index]
    )
    heading = rig.jaw_heading_target(brick_index)
    brick_xyz = brick_start.double().cpu()
    hover_pick = brick_xyz + torch.tensor([0.0, 0.0, HOVER_HEIGHT], dtype=torch.float64)

    # -- solve the hover pose globally, from a spread of seeds ----------------
    hover_target = rig.target_for(hover_pick, heading)
    hover = rig.ik.solve(hover_target, seed=rig.q_arm(), restarts=HOVER_RESTARTS)
    if hover is None:
        # Not a grasp failure: the SHIPPING 6-DOF task (position + tilt magnitude + world
        # +x lean + jaw heading) genuinely has no solution at some band corners, measured
        # in Task D3 part 8 and mapped in part 9. Printed with its own token so the gate's
        # failures can be separated into "the arm could not do it" and "the task as posed
        # has no answer here", which are different problems with different owners --
        # the second is a layout question (spec section 5 item 5), i.e. Task D4's lever.
        print(
            f"PHASE hover [{label}]: UNREACHABLE_TASK at "
            f"brick=({float(brick_xyz[0]):.4f}, {float(brick_xyz[1]):.4f}) "
            f"radius={math.hypot(float(brick_xyz[0]), float(brick_xyz[1])):.4f}m "
            f"azimuth={math.degrees(math.atan2(float(brick_xyz[1]), float(brick_xyz[0]))):+.1f}deg "
            f"jaw_heading={math.degrees(heading):+.1f}deg after {HOVER_RESTARTS} restarts",
            flush=True,
        )
        return False
    print(
        f"SOLVE hover [{label}]: margin={hover.margin:.3f}rad "
        f"pos_resid={hover.position_error * 1e6:.2f}um "
        f"tilt_resid={math.degrees(hover.tilt_error):.3f}deg "
        f"head_resid={math.degrees(hover.heading_error):.3f}deg "
        f"azim_resid={math.degrees(hover.azimuth_error):.3f}deg "
        f"jaw_heading={math.degrees(heading):+.1f}deg",
        flush=True,
    )

    # -- approach + hover: joint-space interpolation, continuous by construction --
    start = rig.q_arm()
    path = [start + (hover.q - start) * t for t in torch.linspace(0.0, 1.0, APPROACH_WAYPOINTS)]
    heights = [float(rig.ik.fk(q)[0][2]) for q in path]
    approach_step = max(float(torch.max(torch.abs(b - a))) for a, b in itertools.pairwise(path))
    for q in path:
        rig.drive(q, GRIPPER_OPEN_ACTION, 2)
    rig.drive(hover.q, GRIPPER_OPEN_ACTION, SETTLE_AT_WAYPOINT)
    pos_err = (rig.pick_point()[0].double().cpu() - hover_pick).norm().item()
    print(
        f"PHASE hover [{label}]: {APPROACH_WAYPOINTS} joint-space waypoints "
        f"max_joint_step={approach_step:.4f}rad "
        f"min_grasp_center_height={min(heights) * 1000:.1f}mm "
        f"pos_err={pos_err * 1000:.2f}mm tool_angle={rig.tool_angle_deg():.2f}deg "
        f"joint_track_err={rig.joint_tracking_error(hover.q):.4f}rad "
        f"min_margin={_margin_str(rig.joint_margins())}",
        flush=True,
    )

    # -- descend: the tilt ladder, smallest tilt first (ruling D3-11 candidate 2) ------
    # Level where level solves; otherwise the least tilt this brick admits. Every rung is
    # inside the jaw-plate fouling window, and each rung carries its OWN pick offset,
    # because the offset that balances desk clearance against tip margin moves with tilt.
    descend_pick = brick_xyz + torch.tensor([0.0, 0.0, DESCEND_CLEARANCE], dtype=torch.float64)
    points = [
        hover_pick + (descend_pick - hover_pick) * t
        for t in torch.linspace(0.0, 1.0, CARTESIAN_WAYPOINTS)
    ]
    chain, descend_tilt, descend_offset_z, tip_margin, rungs = None, None, None, None, 0
    for rung in DESCEND_TILT_LADDER:
        geometry = descend_pick_offset(rung, BRICK_HALF_HEIGHT)
        if geometry is None:
            continue
        rungs += 1
        offset_z, margin = geometry
        tilts = torch.linspace(GRASP_TILT_RAD, rung, CARTESIAN_WAYPOINTS, dtype=torch.float64)
        chain = rig.ik.solve_path(
            [
                rig.target_for(p, heading, float(t), offset_z)
                for p, t in zip(points, tilts, strict=True)
            ],
            seed=hover.q,
        )
        if chain is not None:
            descend_tilt, descend_offset_z, tip_margin = rung, offset_z, margin
            break
    if chain is None:
        print(
            f"PHASE descend [{label}]: NO CONTINUOUS SOLUTION at any of the {rungs} tilts in "
            f"the ladder ({math.degrees(DESCEND_TILT_LADDER[0]):.2f}"
            f"..{math.degrees(DESCEND_TILT_LADDER[-1]):.2f}deg)",
            flush=True,
        )
        return False
    rig.set_pick_offset(descend_offset_z)
    print(
        f"SOLVE descend [{label}]: tilt={math.degrees(descend_tilt):.2f}deg "
        f"(rung {DESCEND_TILT_LADDER.index(descend_tilt) + 1}/{len(DESCEND_TILT_LADDER)}) "
        f"pick_offset_z={descend_offset_z:+.4f}m tip_margin={tip_margin * 1000:.2f}mm",
        flush=True,
    )
    print(_path_str("descend", chain, points), flush=True)
    for step in chain:
        rig.drive(step.q, GRIPPER_OPEN_ACTION, WAYPOINT_STEPS)
    rig.drive(chain[-1].q, GRIPPER_OPEN_ACTION, SETTLE_AT_WAYPOINT)
    desk_left, desk_right = rig.finger_total_forces()
    pos_err = (rig.pick_point()[0].double().cpu() - descend_pick).norm().item()
    print(
        f"PHASE descend [{label}]: pos_err={pos_err * 1000:.2f}mm "
        f"tool_angle={rig.tool_angle_deg():.2f}deg {_offset_str(rig.brick_offsets(brick_index))} "
        f"finger_total_force={desk_left:.2f}/{desk_right:.2f}N "
        f"lowest_jaw_corner={rig.lowest_jaw_corner() * 1000:+.1f}mm "
        f"joint_track_err={rig.joint_tracking_error(chain[-1].q):.4f}rad "
        f"min_margin={_margin_str(rig.joint_margins())}",
        flush=True,
    )

    # -- close: hold the descend solution, jaws in ----------------------------
    peak = [0.0, 0.0]
    for _ in range(CLOSE_STEPS):
        rig.drive(chain[-1].q, GRIPPER_CLOSE_ACTION, 1)
        left, right = rig.finger_brick_forces(brick_index)
        peak = [max(peak[0], left), max(peak[1], right)]
    joints = rig.finger_joints()
    print(
        f"PHASE close [{label}]: finger_joints=({joints[0]:+.5f},{joints[1]:+.5f}) "
        f"peak_brick_force={peak[0]:.2f}/{peak[1]:.2f}N "
        f"{_offset_str(rig.brick_offsets(brick_index))}",
        flush=True,
    )

    # -- lift: straight Cartesian line up -------------------------------------
    # Anchored on the descend chain's OWN final pick point, not the measured one: the lift
    # waypoints are seeded from chain[-1].q, so anchoring on a pose the arm may not have
    # reached makes waypoint 0 unreachable inside the continuity cap and reports a spurious
    # "no continuous solution". Measured, Task D3 part 8: that failed all three trials.
    lift_points = [
        descend_pick + torch.tensor([0.0, 0.0, LIFT_HEIGHT], dtype=torch.float64) * t
        for t in torch.linspace(0.0, 1.0, CARTESIAN_WAYPOINTS)
    ]
    # The tilt is interpolated BACK to the hover tilt as the tool rises, mirroring the
    # descend. Holding the descend tilt all the way up does not work and is not a solver
    # limitation: frames.GRASP_TILT_DEG's own comment records that the arm can only be level
    # AT the brick, and needs 10-18 deg of tilt (by radius) at hover height. Measured, with
    # a level lift: "NO CONTINUOUS SOLUTION along the lift" on the very first trial.
    lift_tilts = torch.linspace(
        descend_tilt, GRASP_TILT_RAD, CARTESIAN_WAYPOINTS, dtype=torch.float64
    )
    lift = rig.ik.solve_path(
        [
            rig.target_for(p, heading, float(t), descend_offset_z)
            for p, t in zip(lift_points, lift_tilts, strict=True)
        ],
        seed=chain[-1].q,
    )
    if lift is None:
        print(f"PHASE lift [{label}]: NO CONTINUOUS SOLUTION along the lift", flush=True)
        return False
    print(_path_str("lift", lift, lift_points), flush=True)
    for step in lift:
        rig.drive(step.q, GRIPPER_CLOSE_ACTION, WAYPOINT_STEPS)
    rig.drive(lift[-1].q, GRIPPER_CLOSE_ACTION, SETTLE_AT_WAYPOINT)

    brick_end = rig.brick_pos(brick_index)
    rise = (brick_end[2] - brick_start[2]).item()
    xy_error = (brick_end[:2] - rig.pick_point()[0, :2]).norm().item()
    # Read the pinch evidence here, in free air: with the brick 100 mm off the
    # desk there is no other contact for the filtered sensors to pick up.
    pinch = rig.pinch_evidence(brick_index)
    lifted = rise > LIFT_MIN_RISE and xy_error < LIFT_MAX_XY_ERROR
    ok = lifted and pinch["ok"]
    others_end = torch.stack(
        [rig.brick_pos(i).clone() for i in range(num_bricks) if i != brick_index]
    )
    disturbance = (others_end - others_start).norm(dim=-1).max().item()
    verdict = "CARRIED" if ok else ("ENTRAINED" if lifted else "DROPPED")
    print(
        f"PHASE lift [{label}]: rise={rise * 1000:+.2f}mm xy_error={xy_error * 1000:.2f}mm "
        f"grip_force={pinch['left']:.2f}/{pinch['right']:.2f}N "
        f"jaw_stall={pinch['stall'] * 1000:.3f}mm asymmetry={pinch['asymmetry']:.3f} "
        f"pinch_ok={pinch['ok']} min_margin={_margin_str(rig.joint_margins())} "
        f"max_other_brick_moved={disturbance * 1000:.2f}mm -> {verdict}",
        flush=True,
    )
    return ok


def isolate_brick(rig, keep: int, num_bricks: int, position) -> None:
    """Park every brick except ``keep`` at rest off to the side, and put ``keep`` at ``position``."""
    env_ids = torch.tensor([0], device=rig.device)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=rig.device)
    zero_vel = torch.zeros(1, 6, device=rig.device)
    for index in range(num_bricks):
        brick = rig.env.scene[f"brick_{index}"]
        if index == keep:
            place = list(position)
        else:
            place = [PARK_X, PARK_Y0 + PARK_Y_SPACING * index, layout.BRICK_SIZE[2] / 2]
        pose = torch.cat([torch.tensor([place], device=rig.device), identity], dim=-1)
        brick.write_root_pose_to_sim(pose, env_ids=env_ids)
        brick.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


def run() -> bool:
    env_cfg_cls = type(parse_env_cfg(args.task, device=args.device, num_envs=1))
    env_cfg = env_cfg_cls(num_bricks=args.num_bricks)
    # --device (AppLauncher's flag) selects the Kit app's GPU but never reaches
    # SimulationCfg.device, which the cfg class leaves at "cuda:0" -- hence
    # passing it to parse_env_cfg above. The cfg is rebuilt through the
    # constructor, which discards that, so re-apply it here.
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    if args.contact_sensor:
        # Contact reporting only -- it adds a read-out, not a force. Requires the
        # spawn flag, which the registered cfg leaves off.
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        # Unfiltered, every robot body: total force per body.
        env_cfg.scene.contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1
        )
        # Filtered per-pair, one sensor per finger: isolates finger <-> brick
        # force from everything else, which is what the pinch gate needs. The
        # filter list is in brick index order so force_matrix_w's filter axis can
        # be indexed by brick. Filtering requires the sensor prim to resolve to a
        # single prim, hence one sensor per finger rather than one regex.
        brick_filters = [f"{{ENV_REGEX_NS}}/brick_{i}" for i in range(args.num_bricks)]
        env_cfg.scene.finger_left_brick = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
            history_length=1,
            filter_prim_paths_expr=brick_filters,
        )
        env_cfg.scene.finger_right_brick = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
            history_length=1,
            filter_prim_paths_expr=brick_filters,
        )
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset(seed=args.seed)
    rig = GraspRig(env, env.scene["contact"] if args.contact_sensor else None)
    num_bricks = env.cfg.num_bricks

    # Provenance for every number below: which maker_arm_rs profile this run actually
    # loaded. Gains and torque caps come from the Rust SDK (core.profile, no fallback).
    print(
        "PROFILE_GAINS "
        + json.dumps(
            {
                name: {"kp": g["kp"], "kd": g["kd"], "tau_max": g["tau_max"]}
                for name, g in _profile.arm_gains(_profile.load_profile()).items()
            }
        ),
        flush=True,
    )
    print(
        f"GEOMETRY pick_offset_z={PICK_OFFSET[2]:+.4f} descend_clearance={DESCEND_CLEARANCE:+.4f} "
        f"grasp_tilt={math.degrees(GRASP_TILT_RAD):.1f}deg "
        f"descend_tilt_ladder=["
        + ",".join(f"{math.degrees(t):.2f}" for t in DESCEND_TILT_LADDER)
        + "]deg "
        f"lean_azimuth={LEAN_AZIMUTH:.3f}rad",
        flush=True,
    )

    if args.gripper_probe:
        # Direction check (spec section 3.3): open action must INCREASE the jaw joint values.
        rig.hold(40, GRIPPER_OPEN_ACTION)
        open_joints = rig.finger_joints()
        rig.hold(40, GRIPPER_CLOSE_ACTION)
        close_joints = rig.finger_joints()
        print(
            "GRIPPER_PROBE "
            + json.dumps(
                {
                    "open_action_joints_m": open_joints,
                    "close_action_joints_m": close_joints,
                    "open_expected_m": GRIPPER_OPEN_POS,
                    "closed_expected_m": GRIPPER_CLOSED_POS,
                }
            ),
            flush=True,
        )
        env.close()
        return all(abs(v - GRIPPER_OPEN_POS) < 0.003 for v in open_joints) and all(
            abs(v - GRIPPER_CLOSED_POS) < 0.003 for v in close_joints
        )

    if args.signs:
        # d(jaw_yaw)/d(wrist roll) and d(jaw_yaw)/d(base yaw), by nudging one joint at a time
        # through the joint-state buffer and reading the jaw heading straight back out of
        # forward kinematics: write the joint state, refresh, read. NO physics step and no
        # servo between the two readings, so nothing but the nudged joint can move.
        #
        # These are kinematic constants of the arm, so a static two-point difference is the
        # right instrument. Holding env steps after the nudge is not: measured on this arm
        # (Task D3), a held reading INVERTED the measured base sign (+0.77 held vs. an exact
        # -1.00 static), so it is not a noisier version of the right answer, it is the wrong
        # answer. See the Task D3 report.
        #
        # Each measurement restarts from the pinned home joint state rather than from
        # wherever the previous one left the arm, so the two are independent and both are
        # taken at the pose robots.frames.HOME_JOINT_POS documents.
        robot = rig.robot
        base_id = robot.joint_names.index(ARM_JOINT_NAMES[0])
        roll_id = robot.joint_names.index(ARM_JOINT_NAMES[5])
        home = robot.data.default_joint_pos.clone()

        def d_yaw_per_joint(joint_id: int, delta: float = 0.1) -> float:
            robot.write_joint_state_to_sim(home, torch.zeros_like(home))
            robot.update(0.0)
            y0 = rig.jaw_yaw()
            nudged = home.clone()
            nudged[0, joint_id] += delta
            robot.write_joint_state_to_sim(nudged, torch.zeros_like(nudged))
            robot.update(0.0)
            return float(wrap_to_pi(rig.jaw_yaw() - y0)) / delta

        d_roll = d_yaw_per_joint(roll_id)
        d_base = d_yaw_per_joint(base_id)
        robot.write_joint_state_to_sim(home, torch.zeros_like(home))
        robot.update(0.0)
        signs = {
            "roll_sign": math.copysign(1.0, d_roll),
            "base_sign": math.copysign(1.0, d_base),
            "d_yaw_per_roll": d_roll,
            "d_yaw_per_base": d_base,
            "home_tool_angle_deg": rig.tool_angle_deg(),
        }
        print("SIGNS " + json.dumps(signs), flush=True)
        env.close()
        # The magnitudes are only exactly 1 when the nudged joint's axis is world-vertical.
        # The base axis is (Task D3: measured -1.0000, and the URDF plus the Z-up mount
        # rotation put it on world -z exactly), so it is held to a tight band; the wrist
        # roll axis is off vertical by the home pose's tool angle, so it gets the brief's
        # original 0.2 band. Both bands exist to catch a heading that is not a rotation
        # about a near-vertical axis at all, which is what would make the sign meaningless.
        return abs(abs(d_base) - 1.0) < 0.01 and abs(abs(d_roll) - 1.0) < 0.2

    if args.x_sweep:
        # Maps where in the scatter zone a top-down pick is possible. This is a
        # gated mode, not a printout: every requested x must be grasped, so
        # sweeping a range known to fail correctly exits non-zero.
        print("X_SWEEP: single brick, y=0, identity yaw", flush=True)
        results = []
        for value in (float(v) for v in args.x_sweep.split(",")):
            env.reset(seed=args.seed)
            rig.trim.reset()
            rig.drive(rig.home_q(), GRIPPER_OPEN_ACTION, 20)
            isolate_brick(rig, 0, num_bricks, [value, 0.0, layout.BRICK_SIZE[2] / 2])
            rig.drive(rig.home_q(), GRIPPER_OPEN_ACTION, 40)
            results.append((value, grasp_one(rig, 0, f"x={value:.3f}", num_bricks)))
        env.close()
        carried = [f"{x:.3f}" for x, ok in results if ok]
        failed = [f"{x:.3f}" for x, ok in results if not ok]
        print(
            f"X_SWEEP: {len(carried)}/{len(results)} carried; carried={carried} failed={failed}",
            flush=True,
        )
        return bool(results) and all(ok for _, ok in results)

    outcomes = []
    rest_ok = True
    for trial in range(args.trials):
        seed = args.seed + trial
        env.reset(seed=seed)
        rest = check_rest_stability(rig, num_bricks, args.settle_steps)
        rest_ok = rest_ok and rest["ok"]
        print(
            f"PHASE settle [trial {trial} seed {seed}]: "
            f"max_drift={rest['max_drift'] * 1000:.3f}mm (limit {REST_MAX_DRIFT * 1000:.0f}) "
            f"max_speed={rest['max_speed'] * 1000:.4f}mm/s (limit {REST_MAX_SPEED * 1000:.0f}) "
            f"max_rest_z_error={rest['z_error'] * 1000:.3f}mm -> "
            f"{'REST_OK' if rest['ok'] else 'REST_FAIL'}",
            flush=True,
        )
        print(
            f"  arm_hold: tool_angle {rest['angle_start']:.2f}->{rest['angle_end']:.2f}deg "
            f"drift={rest['arm_drift_per_step'] * 1000:.5f}mm/step "
            f"({rest['angle_drift_per_step']:+.5f}deg/step) "
            f"total_grasp_center_drift={rest['arm_total_drift'] * 1000:.3f}mm",
            flush=True,
        )
        print(
            f"  per-brick drift(mm)={[round(v * 1000, 4) for v in rest['drift']]} "
            f"speed(mm/s)={[round(v * 1000, 4) for v in rest['speed']]} "
            f"rest_z(mm)={[round(v * 1000, 3) for v in rest['rest_z']]}",
            flush=True,
        )

        positions = [
            (i, round(rig.brick_pos(i)[0].item(), 4), round(rig.brick_pos(i)[1].item(), 4))
            for i in range(num_bricks)
        ]
        target = pick_target_brick(rig, num_bricks, args.brick, random.Random(seed))
        print(
            f"  brick xy={positions} top_down_x_limit={TOP_DOWN_X_LIMIT} -> target=brick_{target}"
            f" (x={rig.brick_pos(target)[0].item():.4f})",
            flush=True,
        )
        outcomes.append(grasp_one(rig, target, f"trial {trial}", num_bricks))

    env.close()
    print(
        f"TRIALS: {sum(outcomes)}/{len(outcomes)} carried; rest_stability_ok={rest_ok}", flush=True
    )
    return bool(rest_ok and all(outcomes) and outcomes)


try:
    ok = run()
except Exception:  # noqa: BLE001 -- any failure must become GRASP_FAIL, not a crash.
    traceback.print_exc()
    ok = False

print("GRASP_OK" if ok else "GRASP_FAIL", flush=True)
os._exit(0 if ok else 1)
