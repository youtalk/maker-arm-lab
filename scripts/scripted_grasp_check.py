"""Scripted grasp + brick contact-stability validation (GPU).

The physics proof this milestone hangs on (design spec section 5.3): real-size 31.8 x 15.8
x 9.6 mm bricks must rest stably on the desk, and the arm must be able to pick one up
autonomously through the ``MakerArm-LegoSort-IK-Rel-v0`` action space -- the same action
space ``oracle_runtime.py`` drives.

Run:  $ISAACLAB_PATH/isaaclab.sh -p scripts/scripted_grasp_check.py --headless

Every phase that MOVES the arm is driven through the 7-element IK-Rel action
``[dx dy dz drx dry drz gripper]``, never by writing joint targets. Phase 0 is the
one exception and does not move the arm at all -- see ``GraspRig.hold_home_joints``
for the measurements that forced it:

  0. settle   240 steps with the arm's joint targets pinned at HOME_JOINT_POS,
              jaws open. Gates rest stability: every brick's drift over the second
              half of the settle must be < 5 mm and its final speed < 0.01 m/s.
  1. hover    servo the PICK point (see below) to 0.12 m above the target brick,
              with the tool axis at frames.GRASP_TILT_DEG off world -z and the jaws
              squared to the brick.
  2. descend  servo the PICK point onto the brick and slightly past it, so the
              jaw plates come to rest on the desk. Stops on stall.
  3. close    45 steps of gripper action -1.0, holding the descend's pose
              closed-loop.
  4. lift     servo the PICK point 0.10 m straight up, jaws still closed.

Success needs the brick to have risen more than 0.05 m and finished within
0.03 m (horizontally) of the tool axis, AND to be genuinely pinched: the jaws
stalled short of ``GRIPPER_CLOSED_POS``, and equal, non-trivial force between
each finger and that brick specifically (filtered per-pair contact sensors, so
the desk press cannot be mistaken for a grip). Position alone cannot separate a
pinch from the entrainment the brief's amendment 3 warns about -- fingers fully
closed with the brick cammed into the jaw pocket rises just as far.

This script also has two measurement-only modes, run before any grasp trial and
gated on their own criteria (Task D3):

* ``--gripper_probe`` holds the gripper open then closed and reports the raw
  joint travel, to catch an inverted jaw before anything downstream trusts the
  open/close convention.
* ``--signs`` nudges the base and wrist-roll joints one at a time and reads the
  jaw heading before/after, to measure ``WRIST_ROLL_SIGN`` / ``BASE_YAW_SIGN``
  (``core.oracle.fold_jaw_heading``'s kinematic sign relations) on this arm.

Same AppLauncher/``os._exit()`` skeleton as smoke_lego_sort.py and smoke_usd.py.
"""

import argparse
import json
import math
import os
import random
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-IK-Rel-v0")
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
    help="Override DESCEND_CLEARANCE (m). Negative commands the jaw plates into "
    "the desk; the desk is what stops them. See the constant's comment.",
)
parser.add_argument(
    "--pick_offset_z",
    type=float,
    default=None,
    help="Override PICK_OFFSET_MEASURED[2] (m), along the tool axis. Used to probe "
    "the offset before it is measured and pinned in robots/frames.py.",
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
from maker_arm_lab.core.action_scale import IK_SCALE, IK_SCALE_ROT
from maker_arm_lab.core.attitude_hold import AttitudeHold
from maker_arm_lab.core.checker import PINCH_MIN_FORCE
from maker_arm_lab.robots.frames import (
    ARM_JOINT_NAMES,
    CANON_IN_BODY,
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
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import canonical_to_body

# --- Geometry, measured on THIS arm by this script (Task D3) -----------------
#
# Measured geometry is imported, never retyped -- the pick offset and the descend
# clearance live next to PICK_OFFSET_MEASURED/DESCEND_CLEARANCE_MEASURED in
# maker_arm_lab.robots.frames, and the top-down pick limit lives next to SCATTER_X
# in maker_arm_lab.core.layout, so the spawn bound and the pick bound cannot
# silently drift apart. The mechanism (not this arm's numbers) carried over from
# OpenYAM: the jaws reach some distance past the tool frame along the tool axis,
# so a brick resting on the desk is gripped near the jaw tips rather than at the
# tool frame itself, and the descend has to be commanded close to (or slightly
# past) the desk to reach it. ``--pick_offset_z``/``--descend_clearance`` exist so
# this script can probe both before they are pinned.
PICK_OFFSET = (
    PICK_OFFSET_MEASURED[0],
    PICK_OFFSET_MEASURED[1],
    args.pick_offset_z if args.pick_offset_z is not None else PICK_OFFSET_MEASURED[2],
)
DESCEND_CLEARANCE = DESCEND_CLEARANCE_MEASURED
TOP_DOWN_X_LIMIT = layout.TOP_DOWN_X_LIMIT

# --- Servo gains -------------------------------------------------------------
# Position: the brief's ``clamp(0.8 * err, -0.01, 0.01)`` expressed against the
# action scale -- IK_SCALE's 0.002 m/unit means the 0.01 m clamp is already
# beyond full deflection, so this is exactly ``clip(0.8 * err / IK_SCALE, +-1)``,
# the same closed-loop Cartesian driver OpenYAM's Task 9 used to reach all four
# bins.
SERVO_GAIN_POS = 0.8
# Rotation: handed to AttitudeHold rather than applied with an inline clamp. The
# clamp this replaced was componentwise -- ``clamp(gain * err / scale, +-1)`` --
# which preserves each component's sign but NOT the correction's axis once any
# component saturates. Measured on OpenYAM's own seeds before that change (not
# yet re-measured on the Maker Arm): the hover phase started 13.6-25.0 deg off
# target and saturated the rotation channel on 29% (trial 0) and 97% (trial 1)
# of its steps, during which the commanded rotation sat up to 39-44 deg away
# from the true error axis. It still converged, because a componentwise clamp
# keeps every sign right so the loop is always heading roughly the correct way
# and the 58-84 step hover budget absorbed the detour -- but that margin was
# accidental. AttitudeHold bounds the demand by magnitude instead, which keeps
# the axis. OpenYAM's A/B at identical seeds: 2/2 CARRIED either way, rise
# 98.43/98.44 -> 98.43/98.45 mm, grip 8.78/8.69 -> 8.78/8.70 N.
SERVO_GAIN_ROT = 1.0
# Per-step pick-point motion below which the descend is treated as blocked.
# Full deflection commands 2 mm/step and delivers ~1.7 mm on OpenYAM's arm, so
# 0.05 mm is ~3% of a moving step -- comfortably below any real motion and
# above solver noise.
STALL_EPS = 5e-5
# Rotation error (rad) below which a servo phase counts as converged.
ROT_CONVERGED = 0.02
# Per-joint torque gap (N*m) above which computed_torque != applied_torque
# counts as saturated for this step (core.mit_law's effort_limit clamp). See
# actuators/mit_mode.py's docstring point 2 for why this idiom, rather than
# comparing applied_torque to effort_limit directly, is the one that survives
# tau_ff_limit clamping being invisible to it.
SATURATION_EPS = 1e-6

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
# Thresholds carried over from OpenYAM's LegoSort checker, set well between a
# real pinch and a fully-closed empty jaw on THAT arm; this script's own
# trials (Task D3) are what re-validates them on the Maker Arm's gripper.
#
# PINCH_MIN_FORCE (2.0 N on the weaker finger) is imported from core.checker
# rather than redefined here: this measurement set is where the OpenYAM number
# came from, but the oracle's runtime check and the eval harness's strict grasp
# grading use the same number, and it was three copies of 2.0 there until it
# was consolidated.
PINCH_MIN_STALL = 0.003  # m short of GRIPPER_CLOSED_POS
PINCH_MAX_ASYMMETRY = 0.30  # |L - R| / max(L, R)

HOVER_HEIGHT = 0.12
LIFT_HEIGHT = 0.10
MAX_PHASE_STEPS = 300
CLOSE_STEPS = 45
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class GraspRig:
    """Everything the phases need: frames, the servo, and telemetry."""

    def __init__(self, env, sensor):
        self.env = env
        self.sensor = sensor
        self.robot = env.scene["robot"]
        self.device = env.device
        self.tool = self.robot.find_bodies(TOOL_BODY)[0][0]
        # Joint order resolved from the articulation, never hard-coded (the
        # pattern check_success_teleport.py established on OpenYAM).
        self.finger_ids = self.robot.find_joints(list(GRIPPER_JOINT_NAMES))[0]
        self.arm_joint_ids = self.robot.find_joints(list(ARM_JOINT_NAMES))[0]
        self.action_dim = env.action_manager.total_action_dim
        if self.action_dim != 7:
            raise RuntimeError(f"expected a 7-element IK-Rel action, got {self.action_dim}")
        self.pick_offset = torch.tensor(PICK_OFFSET, device=self.device).unsqueeze(0)
        self.ex = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.jaw_axis = torch.tensor([JAW_AXIS_BODY], device=self.device)
        self.tool_axis = torch.tensor([TOOL_AXIS_BODY], device=self.device)
        self.canon_in_body = torch.tensor(CANON_IN_BODY, device=self.device)
        self.scale_pos = torch.tensor(IK_SCALE[0:3], device=self.device)
        self.scale_rot = torch.tensor(IK_SCALE_ROT, device=self.device)
        self.finger_sensor_ids = (
            [sensor.body_names.index("gripper_left"), sensor.body_names.index("gripper_right")]
            if sensor is not None
            else []
        )
        # Filtered per-pair sensors: finger <-> brick force only. The unfiltered
        # net_forces_w above sums EVERY contact on the finger body, so during the
        # descend it is dominated by the desk press and cannot be read as a grip
        # force. The two instruments are kept separate and labelled separately.
        self.brick_pair = (
            (env.scene["finger_left_brick"], env.scene["finger_right_brick"])
            if sensor is not None
            else None
        )

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

    def desired_quat(self, brick_index: int):
        """Tool axis GRASP_TILT_DEG off world -z toward +x, jaws square across the brick.

        A jaw is symmetric under a half turn, so the target heading is folded to
        whichever of ``yaw`` / ``yaw + pi`` the wrist is already nearer -- joint6
        only spans +-2.0944 rad and the far representative can be unreachable.

        The tilt is the arm-specific part. The OpenYAM port asked for the tool exactly
        on world -z; this arm cannot hold that over the scatter band at hover height
        (measured minimum 13.01 deg at the far edge, ``frames.GRASP_TILT_DEG``), and
        demanding it jams link_005_joint against its stop. What the grasp actually
        needs is the jaw axis square to the brick -- which is unchanged here -- plus
        the tool pointed down enough to reach the brick between the jaws.

        Built identically to ``oracle_runtime.OracleRig.desired_quat``, deliberately:
        this script is the oracle's rehearsal, so if the two disagreed about grasp
        attitude the oracle would fail in a way that looks like an oracle bug. Both
        assemble the frame in canonical tool coordinates (x = jaw axis, z = tool axis)
        and convert to the ``grasp_center`` body frame with ``canonical_to_body``.
        """
        brick_quat = self.env.scene[f"brick_{brick_index}"].data.root_quat_w[0].unsqueeze(0)
        long_axis = math_utils.quat_apply(brick_quat, self.ex)[0]
        want = math.atan2(long_axis[1].item(), long_axis[0].item()) + math.pi / 2
        want += math.pi * round(_wrap_to_pi(self.jaw_yaw() - want) / math.pi)
        c, s = math.cos(want), math.sin(want)
        # A rotation matrix's columns are the world-frame images of the CANONICAL axes,
        # so build it from the axes as columns via torch.stack(dim=1):
        #   canon x -> (c, s, 0)                     the jaw closing direction
        #   canon z -> (sin tilt, 0, -cos tilt)      the tool axis, leaning toward +x
        #   canon y -> z x x, normalized             keeps the frame right-handed
        # then re-orthogonalize x against the tilted z (a no-op at tilt = 0, which is
        # why this reduces exactly to the OpenYAM closed form when the tilt is zero).
        tilt = GRASP_TILT_RAD
        axis_x = torch.tensor([c, s, 0.0], device=self.device)
        axis_z = torch.tensor([math.sin(tilt), 0.0, -math.cos(tilt)], device=self.device)
        axis_y = torch.cross(axis_z, axis_x, dim=-1)
        axis_y = axis_y / axis_y.norm()
        axis_x = torch.cross(axis_y, axis_z, dim=-1)
        rot = torch.stack([axis_x, axis_y, axis_z], dim=1)
        rot = canonical_to_body(rot, self.canon_in_body)
        return math_utils.quat_from_matrix(rot.unsqueeze(0))

    def rot_error(self, quat_des):
        """World-frame axis-angle rotation taking the current tool frame to ``quat_des``.

        ``apply_delta_pose`` pre-multiplies the commanded rotation onto the
        measured orientation, so the delta this action term expects is expressed
        in the world frame, not the tool frame.
        """
        err = math_utils.quat_mul(quat_des, math_utils.quat_conjugate(self.tool_quat()))
        return math_utils.axis_angle_from_quat(err)[0]

    # -- actuation ------------------------------------------------------------
    def step(self, delta_pos, delta_rot, gripper: float):
        action = torch.zeros(1, self.action_dim, device=self.device)
        action[0, 0:3] = delta_pos
        action[0, 3:6] = delta_rot
        action[0, 6] = gripper
        self.env.step(action)

    def hold(self, steps: int, gripper: float):
        """Send a zero position/rotation delta for ``steps`` steps.

        This does NOT hold the arm's pose -- ``hold_pose`` does. Use this only where
        the arm's pose is not what is being measured (the gripper-direction probe).
        """
        zero = torch.zeros(3, device=self.device)
        for _ in range(steps):
            self.step(zero, zero, gripper)

    def capture_pose(self):
        """``(pick point, tool quaternion)`` right now, to hand to ``hold_pose``."""
        return (self.pick_point()[0].clone(), self.tool_quat()[0].clone())

    def hold_pose(self, steps: int, gripper: float, reference=None):
        """Hold a pose closed-loop for exactly ``steps`` steps, jaws at ``gripper``.

        The difference from ``hold`` is load-bearing. A zero delta through
        ``DifferentialIKController`` in relative mode does not mean "stay where you
        are": ``process_actions`` re-references the desired pose from the *measured*
        pose every control step, so a steady-state tracking error is never corrected
        -- it is adopted as the next setpoint, and the pose ratchets away under
        gravity without bound. Measured on this arm (Task D3) at roughly 0.003 rad
        per step, which over the 240-step settle walked the tool angle from the
        pinned 23.85 deg to 71.69 deg and handed the hover servo a 280 mm starting
        error it had no budget left to close.

        Holding against a reference captured ONCE feeds that error back instead of
        forgetting it. Same servo law as ``servo`` -- P on position, ``AttitudeHold``
        on attitude -- but with no convergence exit, because the job is to stay put
        for a fixed number of steps rather than to arrive.

        ``reference`` is a ``capture_pose()`` pair, defaulting to the pose at the
        moment of the call.
        """
        target, quat_des = self.capture_pose() if reference is None else reference
        attitude = AttitudeHold(
            rot_scale=self.scale_rot, gain=SERVO_GAIN_ROT, give_up_error=float("inf")
        )
        attitude.capture(quat_des)
        zero_rot_command = torch.zeros(3, device=self.device)
        for _ in range(steps):
            pos_err = target - self.pick_point()[0]
            self.step(
                torch.clamp(SERVO_GAIN_POS * pos_err / self.scale_pos, -1.0, 1.0),
                attitude.advance(self.tool_quat()[0], zero_rot_command),
                gripper,
            )

    def hold_home_joints(self, steps: int):
        """Step ``steps`` control steps with every joint target pinned at the home pose.

        The settle's job is for the arm to change nothing for 240 control steps while the
        bricks come to rest, and on this arm NO hold driven through the IK-Rel action can
        do that -- not even the closed-loop ``hold_pose``. The action term's authority is
        ``IK_SCALE[0:3]`` = 2 mm per control step, while the arm's own sag away from the
        home pose runs at about 4.5 mm per control step over the first 20 steps (measured,
        Task D3). A servo that cannot command as fast as the arm falls cannot converge:
        measured, ``hold_pose`` drifts 111 mm and then runs ``link_005_joint`` into its
        URDF stop at step 153 and ``link_004_joint`` at step 173, after which the arm is
        frozen and every later phase is servoing a dead mechanism.

        So this phase alone is held in JOINT space, at ``robot.data.default_joint_pos``
        (which is ``robots.frames.HOME_JOINT_POS``), and the IK-Rel action term takes over
        for the grasp phases with the arm exactly where the home pose says it should be.
        Bypassing the action term is sound precisely here and nowhere else: the settle is a
        measurement phase about the BRICKS, it commands no arm motion, and it contributes
        no action to any dataset.

        ``steps`` is counted in control steps and expanded by ``cfg.decimation``, so a
        given ``--settle_steps`` covers the same simulated time as the other phases.
        """
        robot, scene, sim = self.robot, self.env.scene, self.env.sim
        home = robot.data.default_joint_pos.clone()
        dt = sim.get_physics_dt()
        for _ in range(steps * self.env.cfg.decimation):
            robot.set_joint_position_target(home)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)

    def finger_joints(self) -> list[float]:
        return self.robot.data.joint_pos[0, self.finger_ids].tolist()

    def joint_saturated(self) -> list[bool]:
        """Which arm joints hit ``effort_limit`` this step.

        ``computed_torque`` is the PD+feedforward demand before the actuator's
        effort clamp; ``applied_torque`` is what actually reached the sim. They
        differ iff effort_limit bound (see actuators/mit_mode.py's docstring).
        """
        computed = self.robot.data.computed_torque[0, self.arm_joint_ids]
        applied = self.robot.data.applied_torque[0, self.arm_joint_ids]
        return ((computed - applied).abs() > SATURATION_EPS).tolist()

    def servo(
        self,
        target,
        quat_des,
        max_steps: int,
        gripper: float,
        stop_pos: float | None,
        stall_steps: int = 0,
    ):
        """P-servo the pick point to ``target`` and the tool to ``quat_des``.

        ``stall_steps`` > 0 also stops once the pick point has moved less than
        ``STALL_EPS`` per step for that many consecutive steps. The descend phase
        needs this: its target is deliberately below the desk, so the position
        error never converges -- the plates land on the desk and the servo would
        otherwise spend the rest of its budget leaning on it.

        Returns (steps_used, final position error, final rotation error, stalled,
        per-joint torque-saturation fraction over the steps actually taken).
        """
        previous = self.pick_point()[0].clone()
        quiet = 0
        pos_norm = (target - previous).norm().item()
        rot_norm = self.rot_error(quat_des).norm().item()
        sat_counts = [0] * len(self.arm_joint_ids)
        # One hold per phase, holding this phase's target attitude. A zero
        # rotation command each step makes it a pure servo to quat_des: the
        # reference only ever moves by command, so it stays put. give_up is
        # disabled because this driver legitimately starts a phase tens of
        # degrees off target -- far above the teleop-tuned default threshold --
        # and has its own step budget and stall detection to end a phase that is
        # going nowhere. See AttitudeHold's "Giving up" docstring section.
        attitude = AttitudeHold(
            rot_scale=self.scale_rot, gain=SERVO_GAIN_ROT, give_up_error=float("inf")
        )
        attitude.capture(quat_des[0])
        zero_rot_command = torch.zeros(3, device=self.device)
        for index in range(max_steps):
            pos_err = target - self.pick_point()[0]
            self.step(
                torch.clamp(SERVO_GAIN_POS * pos_err / self.scale_pos, -1.0, 1.0),
                attitude.advance(self.tool_quat()[0], zero_rot_command),
                gripper,
            )
            for joint, saturated in enumerate(self.joint_saturated()):
                sat_counts[joint] += int(saturated)
            # One post-step read of each error, reused by every branch below.
            now = self.pick_point()[0]
            pos_norm = (target - now).norm().item()
            rot_norm = self.rot_error(quat_des).norm().item()
            steps_taken = index + 1
            sat_frac = [count / steps_taken for count in sat_counts]
            if stop_pos is not None and pos_norm < stop_pos and rot_norm < ROT_CONVERGED:
                return steps_taken, pos_norm, rot_norm, False, sat_frac
            if stall_steps:
                quiet = quiet + 1 if (now - previous).norm().item() < STALL_EPS else 0
                if quiet >= stall_steps:
                    return steps_taken, pos_norm, rot_norm, True, sat_frac
            previous = now.clone()
        return max_steps, pos_norm, rot_norm, False, sat_frac

    # -- telemetry ------------------------------------------------------------
    def finger_total_forces(self) -> tuple[float, float]:
        """Total contact force on each finger body, from EVERY contact.

        During the descend this is the desk press, not a grip force -- the plates
        are deliberately driven onto the desk. Use ``finger_brick_forces`` for the
        grip.
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

    def brick_in_tool_frame(self, index: int):
        origin = self.robot.data.body_pos_w[0, self.tool]
        return math_utils.quat_apply_inverse(
            self.tool_quat(), (self.brick_pos(index) - origin).unsqueeze(0)
        )[0]


def _saturation_str(sat_frac: list[float]) -> str:
    """``j1:0% j2:12% ...`` -- ARM_JOINT_NAMES in motor-id order (1-6)."""
    return " ".join(f"j{i + 1}:{frac * 100:.0f}%" for i, frac in enumerate(sat_frac))


def check_rest_stability(rig, num_bricks: int, steps: int) -> dict:
    """Phase 0. Hold the home pose with the jaws open and watch the bricks settle.

    Drift is measured over the *second half* of the window. The whole-window
    displacement is dominated by the reset scatter's 15.2 mm drop from
    ``layout.SPAWN_Z`` (0.02 m) to the resting height (half the brick's 9.6 mm
    thickness), which is the brick falling into place, not instability. What the
    spec's rest-stability gate is actually about is whether a settled brick then
    stays put instead of jittering or sinking, so the tail is the honest window.

    The arm is held with ``hold_home_joints``, not ``hold`` and not ``hold_pose``;
    that method's docstring has the measurements ruling the other two out. The arm's
    own drift is reported on the same second-half window as the bricks', so "the
    settle does not move the arm" is a logged number rather than an inference.
    """
    half = steps // 2
    arm_start = rig.pick_point()[0].clone()
    angle_start = rig.tool_angle_deg()
    rig.hold_home_joints(half)
    mid = torch.stack([rig.brick_pos(i).clone() for i in range(num_bricks)])
    arm_mid = rig.pick_point()[0].clone()
    angle_mid = rig.tool_angle_deg()
    rig.hold_home_joints(steps - half)
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
    quat_des = rig.desired_quat(brick_index)

    hover_target = brick_start.clone()
    hover_target[2] += HOVER_HEIGHT
    steps, perr, rerr, _, sat = rig.servo(
        hover_target, quat_des, MAX_PHASE_STEPS, GRIPPER_OPEN_ACTION, 0.005
    )
    print(
        f"PHASE hover [{label}]: steps={steps} pos_err={perr * 1000:.2f}mm "
        f"rot_err={math.degrees(rerr):.2f}deg tool_angle={rig.tool_angle_deg():.2f}deg "
        f"torque_sat=({_saturation_str(sat)})",
        flush=True,
    )

    clearance = DESCEND_CLEARANCE if args.descend_clearance is None else args.descend_clearance
    descend_target = brick_start.clone()
    descend_target[2] += clearance
    steps, perr, rerr, stalled, sat = rig.servo(
        descend_target, quat_des, MAX_PHASE_STEPS, GRIPPER_OPEN_ACTION, 0.0008, stall_steps=12
    )
    tool_brick = rig.brick_in_tool_frame(brick_index)
    desk_left, desk_right = rig.finger_total_forces()
    print(
        f"PHASE descend [{label}]: steps={steps} stalled={stalled} pos_err={perr * 1000:.2f}mm "
        f"rot_err={math.degrees(rerr):.2f}deg tool_angle={rig.tool_angle_deg():.2f}deg "
        f"brick_in_tool=({tool_brick[0].item():+.4f},{tool_brick[1].item():+.4f},"
        f"{tool_brick[2].item():.4f}) finger_total_force={desk_left:.2f}/{desk_right:.2f}N"
        " (mostly the desk press -- the plates are driven onto the desk on purpose) "
        f"torque_sat=({_saturation_str(sat)})",
        flush=True,
    )

    # Held closed-loop against the pose the descend ended at: the jaws must close
    # around the brick where the descend left them, and 45 steps of zero delta is
    # long enough for the ratchet in ``hold_pose``'s docstring to drag the plates
    # off the brick before they ever touch it.
    close_reference = rig.capture_pose()
    peak = [0.0, 0.0]
    for _ in range(CLOSE_STEPS):
        rig.hold_pose(1, GRIPPER_CLOSE_ACTION, close_reference)
        left, right = rig.finger_brick_forces(brick_index)
        peak = [max(peak[0], left), max(peak[1], right)]
    joints = rig.finger_joints()
    tool_brick = rig.brick_in_tool_frame(brick_index)
    print(
        f"PHASE close [{label}]: finger_joints=({joints[0]:+.5f},{joints[1]:+.5f}) "
        f"peak_brick_force={peak[0]:.2f}/{peak[1]:.2f}N "
        f"brick_in_tool=({tool_brick[0].item():+.4f},{tool_brick[1].item():+.4f},"
        f"{tool_brick[2].item():.4f}) tool_angle={rig.tool_angle_deg():.2f}deg",
        flush=True,
    )

    lift_target = rig.pick_point()[0].clone()
    lift_target[2] += LIFT_HEIGHT
    steps, perr, rerr, _, _ = rig.servo(
        lift_target, quat_des, MAX_PHASE_STEPS, GRIPPER_CLOSE_ACTION, 0.002
    )
    # Closed-loop again: the pinch evidence below is read after this settle, so the
    # arm must actually be holding the brick up rather than sinking with it.
    rig.hold_pose(30, GRIPPER_CLOSE_ACTION)

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
        f"PHASE lift [{label}]: steps={steps} pos_err={perr * 1000:.2f}mm "
        f"rise={rise * 1000:+.2f}mm xy_error={xy_error * 1000:.2f}mm "
        f"grip_force={pinch['left']:.2f}/{pinch['right']:.2f}N "
        f"jaw_stall={pinch['stall'] * 1000:.3f}mm asymmetry={pinch['asymmetry']:.3f} "
        f"pinch_ok={pinch['ok']} "
        f"max_other_brick_moved={disturbance * 1000:.2f}mm -> {verdict}",
        flush=True,
    )
    return ok


# Parking spots for the bricks the x-sweep is not picking: on the desk (which
# spans x in [-0.15, 0.75], y in [-0.45, 0.45]), beyond the shelf and well clear
# of the arm. Parking them off the desk edge instead would leave them free-falling
# for the rest of the run, which silently makes the neighbour-disturbance figure
# meaningless.
PARK_X = 0.66
PARK_Y0 = -0.40
PARK_Y_SPACING = 0.04


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
    # constructor, which discards that, so re-apply it here. This line used to
    # hard-code "cuda:0", i.e. --device was accepted and silently ignored.
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    if args.contact_sensor:
        # Contact reporting only -- it adds a read-out, not a force. Requires the
        # spawn flag, which the registered cfg leaves off.
        env_cfg.scene.robot.spawn.activate_contact_sensors = True
        # Unfiltered, every robot body: total force per body. During the
        # descend this is the desk press.
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
    # loaded. Gains and torque caps come from the Rust SDK (core.profile, no fallback), and
    # that SDK is installed into the Isaac venv separately from the one uv manages -- so a
    # run can execute against a different profile than the repo's pin implies. Print the
    # caps so the log says which, instead of leaving it to be inferred from the pin.
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
        # (Task D3), holding the IK-Rel action at a zero delta lets the pose drift further
        # than the nudge itself -- 0.054 rad of jaw yaw over 5 steps with NO nudge at all,
        # and the tool angle wandering from the pinned 23.6 deg to 43 deg and on to 66 deg.
        # That drift INVERTED the measured base sign (+0.77 held vs. an exact -1.00 static),
        # so the held reading is not a noisier version of the right answer, it is the wrong
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
            return _wrap_to_pi(rig.jaw_yaw() - y0) / delta

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
            rig.hold(20, GRIPPER_OPEN_ACTION)
            isolate_brick(rig, 0, num_bricks, [value, 0.0, layout.BRICK_SIZE[2] / 2])
            rig.hold(40, GRIPPER_OPEN_ACTION)
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
