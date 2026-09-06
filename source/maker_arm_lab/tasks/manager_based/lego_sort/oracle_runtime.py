"""GPU binding of the core oracle: per-env servo loops over the IK-Rel action.

Adapted from scripts/scripted_grasp_check.py: same P-servo, same AttitudeHold
usage, same pinch-evidence gate. Differences: N envs stepped together (one
env.step per control step, per-env phase state), goals come from
core.oracle.EnvOracle, and noise comes from EpisodeDraw.

core.oracle is isaaclab-free by design and only says WHAT to do next (a Goal).
This module owns HOW: it turns that Goal into a 7-element IK-Rel action every
control step, reads the frames (the tool body (`grasp_center`) pose, brick poses, the
filtered finger<->brick contact sensors) the oracle's SceneSnapshot needs, and
applies the two measured constants core.oracle deliberately leaves to the
runtime (DESCEND_CLEARANCE_MEASURED, PICK_OFFSET_MEASURED).

Frame convention: every position this class hands to core.oracle (SceneSnapshot
fields, Goal.target_pos) and every position it reads back for the P-servo
(``pick_point``) is env-origin-subtracted -- SceneSnapshot's own docstring
("world frame, env-origin subtracted"). That is what makes a servo error
(target - measured) identical to the same subtraction done in raw world
coordinates: the per-env origin cancels in the difference either way, so
picking the origin-local convention throughout is free, and it is required for
the goals core.oracle builds from constants that are the same for every env
(bin centers, RETREAT_POINT) to mean the same physical pose in every env.
"""

import math
import random

import isaaclab.utils.math as math_utils
import torch

from maker_arm_lab.core import layout, oracle
from maker_arm_lab.core.action_scale import IK_SCALE
from maker_arm_lab.core.attitude_hold import AttitudeHold
from maker_arm_lab.core.checker import PINCH_MIN_FORCE
from maker_arm_lab.robots.frames import (
    ARM_JOINT_REGEX,
    BASE_YAW_SIGN,
    CANON_IN_BODY,
    DESCEND_CLEARANCE_MEASURED,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    JAW_AXIS_BODY,
    PICK_OFFSET_MEASURED,
    TOOL_BODY,
    WRIST_ROLL_INDEX,
    WRIST_ROLL_SIGN,
)

from .mdp.recorders import PHASE_TO_INT

# --- Servo gains and thresholds, carried over unchanged from OpenYAM's
# scripts/scripted_grasp_check.py (OpenYAM's M1 Task 10/11 validated code). See
# that script's module docstring for the full measurement-backed derivation on
# OpenYAM's arm; only the headline numbers are repeated here. Not yet
# re-validated on the Maker Arm -- this milestone's own
# scripts/scripted_grasp_check.py (Task D3) is what re-measures them.
SERVO_GAIN_POS = 0.8
SERVO_GAIN_ROT = 1.0
# Per-step pick-point motion below which a stall-gated phase (DESCEND) is
# treated as blocked -- full deflection delivers ~1.7 mm/step, so this is far
# below any real motion and above solver noise.
STALL_EPS = 5e-5
# Rotation error (rad) below which a servo phase counts as converged.
ROT_CONVERGED = 0.02

# Pinch evidence (gated after LIFT, in free air, so the readings are pure
# finger<->brick and not the desk press): a real pinch stalls the jaws short of
# GRIPPER_CLOSED_POS and puts equal, non-trivial filtered force on both
# fingers; entrainment (brick merely cammed in the jaw pocket) rises and
# centers identically but does neither. See OpenYAM's scripted_grasp_check.py's
# PINCH_MIN_STALL/PINCH_MIN_FORCE/PINCH_MAX_ASYMMETRY comments for the
# measurements (8.71/8.71 N, 0.0087 m stall), on OpenYAM's arm, that set these
# thresholds -- not yet re-validated on the Maker Arm.
# PINCH_MIN_FORCE (2.0 N on the weaker finger) is imported from
# core.checker rather than redefined: it is one threshold with three consumers.
PINCH_MIN_STALL = 0.003  # m short of GRIPPER_CLOSED_POS
PINCH_MAX_ASYMMETRY = 0.30  # |L - R| / max(L, R)
LIFT_MIN_RISE = 0.05  # m
LIFT_MAX_XY_ERROR = 0.03  # m

# Idle action for an env whose oracle has nothing left to do this step (DONE or
# ABANDON): zero position/rotation delta, jaws open, same as OpenYAM's
# scripted_grasp_check.py's `hold(..., GRIPPER_OPEN_ACTION)` convention.
GRIPPER_OPEN_ACTION = 1.0

# Consecutive watchdog-flagged control steps before an env is abandoned. 30
# steps at 30 Hz is 1 s -- comfortably longer than the transient joint-limit
# contact a converging servo can produce, short enough not to burn an episode's
# budget on a genuinely runaway env. Carried over unchanged from OpenYAM; the
# reasoning and the numbers below are OpenYAM's own measurements, on OpenYAM's
# arm, not yet re-validated on the Maker Arm.
#
# Since OpenYAM's Task 8, the strike counter resets to 0 on every phase entry
# (see _enter_phase below), not just once per episode. Two consequences worth
# knowing before retuning either this or a phase's own step budget:
#
# * core.oracle.RAISE_MAX_STEPS (60) depends on this value: RAISE is only
#   guaranteed to exit on "budget" -- not get abandoned by the watchdog mid-
#   phase -- if RAISE_MAX_STEPS <= WATCHDOG_TRIP_STRIKES + first_contact_step,
#   where first_contact_step (measured 30-40 control steps into RAISE, on
#   OpenYAM's Task 8 GPU runs) is how soon a genuine reach-ceiling contact
#   starts flagging "near" a joint limit. Raising WATCHDOG_TRIP_STRIKES
#   without re-checking this bound is safe (more margin); lowering it, or
#   raising RAISE_MAX_STEPS, is not -- re-run
#   scripts/release_envelope_check.py's full sweep before doing either.
# * core.oracle.REOPEN (hold_steps=20) and RELEASE (hold_steps=20) both run
#   fewer control steps than this threshold, so with a per-phase reset
#   neither phase can EVER trip the watchdog on its own, regardless of how
#   close to a joint limit the arm is throughout -- a real gap versus
#   OpenYAM's pre-Task-8 persistent counter, not just a smaller window.
#   Measured (OpenYAM's Task 8, 40 release-envelope trials + 2 full 6-brick
#   episodes) this is latent, not exercised: RELEASE runs with joint4_margin
#   0.08-0.45 rad, comfortably clear of the 0.05 rad guard band, in every
#   case tried on OpenYAM's arm. Left undefended deliberately rather than
#   given special-cased carry-over, because REOPEN/RELEASE are pure-hold
#   phases (Goal.target_pos is None) -- they never themselves command motion
#   toward a limit, only inherit whatever pose the PRECEDING active phase
#   converged to -- and the alternative (carrying strikes over for hold
#   phases only, resetting for active ones) is more state to keep consistent
#   for a case measurement shows is not currently live there. Revisit if a
#   future release pose runs RELEASE genuinely near a limit on the Maker Arm.
WATCHDOG_TRIP_STRIKES = 30


def canonical_to_body(rot_canon: torch.Tensor, canon_in_body: torch.Tensor) -> torch.Tensor:
    """World rotation of the tool BODY given the desired world rotation of the canonical
    tool frame (columns: jaw axis, y, tool axis) and CANON_IN_BODY (rows: those axes in
    body coordinates): R_world_body = R_world_canon @ C^T."""
    return rot_canon @ canon_in_body.T


class OracleRig:
    """Per-env GPU binding of ``core.oracle.EnvOracle`` onto batched IK-Rel actions.

    One ``core.oracle.EnvOracle`` and one single-env ``AttitudeHold`` per env
    (``give_up_error=float("inf")``: as in OpenYAM's scripted_grasp_check.py
    servo, a phase legitimately starts tens of degrees off target and this
    driver has its own phase budgets -- see ``core.attitude_hold``'s "Giving
    up" docstring section). ``step()`` advances every env by one control step;
    ``rearm()`` re-seeds the envs a caller has just reset.
    """

    def __init__(
        self,
        env,
        bl: layout.BinLayout,
        noise: oracle.OracleNoise,
        rng: random.Random,
        induction: oracle.FailureInduction | None = None,
    ):
        self.env = env
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
        self.arm_ids = self.robot.find_joints([ARM_JOINT_REGEX])[0]
        action_dim = env.action_manager.total_action_dim
        if action_dim != 7:
            raise RuntimeError(f"expected a 7-element IK-Rel action, got {action_dim}")
        self.pick_offset = torch.tensor(PICK_OFFSET_MEASURED, device=self.device)
        self.ex = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.jaw_axis = torch.tensor(JAW_AXIS_BODY, device=self.device)
        self.canon_in_body = torch.tensor(CANON_IN_BODY, device=self.device)
        self.scale_pos = torch.tensor(IK_SCALE[0:3], device=self.device)
        self.scale_rot = torch.tensor(IK_SCALE[3:6], device=self.device)
        # Joint limits read once: identical for every env (same robot asset
        # replicated), so env 0's row is every env's row.
        self.limits = self.robot.data.joint_pos_limits[0, self.arm_ids].clone()  # (6,2)
        # Plain CPU floats, read once: desired_quat's fold needs them on every
        # phase entry and a per-call .tolist() would be a device sync.
        self.wrist_roll_limits = tuple(self.limits[WRIST_ROLL_INDEX].tolist())
        self.left_brick = env.scene["finger_left_brick"]
        self.right_brick = env.scene["finger_right_brick"]

        self.oracles: list[oracle.EnvOracle | None] = [None] * self.num_envs
        self.holds: list[AttitudeHold | None] = [None] * self.num_envs
        self.goals: list[oracle.Goal | None] = [None] * self.num_envs
        # Last explicit jaw-heading target per env (the `want` angle from
        # desired_quat), reused whenever a goal's yaw is None so a held phase
        # (CLOSE/LIFT/RAISE/...) keeps the heading DESCEND last commanded
        # instead of re-deriving one -- there is no brick reference once the
        # brick has left the desk.
        self.commanded_yaw = [0.0] * self.num_envs
        # Brick pose at the HOVER this pick attempt started from (env-origin
        # subtracted), for the LIFT rise/xy gate -- mirrors grasp_one's
        # `brick_start` in OpenYAM's scripted_grasp_check.py.
        self.brick_start: list[torch.Tensor | None] = [None] * self.num_envs
        # Pick point one step ago, for DESCEND's stall detector.
        self.previous_pick_point: list[torch.Tensor | None] = [None] * self.num_envs
        self.steps_in_phase = [0] * self.num_envs
        self.quiet = [0] * self.num_envs
        # Post-convergence settle, LIFT only (Goal.hold_steps > 0 with a
        # target_pos set): OpenYAM's scripted_grasp_check.py reads pinch/rise after
        # `rig.hold(30, GRIPPER_CLOSE_ACTION)`, not at the instant the position
        # servo converges, so the contact forces have settled. `holding[i]`
        # marks that this env has converged/stalled/budgeted-out and is now
        # coasting the extra `goal.hold_steps` steps before the gate is read;
        # `pending_outcome` is the outcome the servo actually ended on.
        self.holding = [False] * self.num_envs
        self.hold_counter = [0] * self.num_envs
        self.pending_outcome: list[str | None] = [None] * self.num_envs
        self.watchdog_strikes = [0] * self.num_envs
        self.abandoned = torch.zeros(self.num_envs, dtype=torch.bool)
        self.stats = [{"retries": 0, "sorted": 0, "abandon": None} for _ in range(self.num_envs)]
        self.rearm(list(range(self.num_envs)))

    # -- control loop ----------------------------------------------------

    def step(self) -> dict:
        """One control step for the whole batch. Returns ``{"dones", "abandoned"}``."""
        action = torch.zeros(self.num_envs, 7, device=self.device)
        zero_rot_command = torch.zeros(3, device=self.device)
        for i in range(self.num_envs):
            goal = self.goals[i]
            if goal is None:
                # DONE/ABANDON envs coast: zero deltas, jaws open.
                action[i, 6] = GRIPPER_OPEN_ACTION
                continue
            action[i, 6] = goal.gripper
            if goal.target_pos is not None:
                err = goal.target_pos.to(self.device) - self.pick_point(i)
                gain = SERVO_GAIN_POS * self.oracles[i].draw.gain_scale
                delta = torch.clamp(gain * err / self.scale_pos, -1.0, 1.0)
                noise_std = self.oracles[i].draw.action_noise_std * self.oracles[i].noise_factor()
                delta = delta + noise_std * torch.randn(3, device=self.device)
                action[i, 0:3] = torch.clamp(delta, -1.0, 1.0)
            else:
                # Hold phases (SETTLE/CLOSE/REOPEN/RELEASE) command no
                # translation -- except CLOSE for a noise-induced first
                # attempt, which has no servo error for the noise term above
                # to ride on but still needs the boosted noise noise_factor()
                # promises for CLOSE (see its docstring). Gated strictly on
                # factor > 1.0: noise_factor() is 1.0 whenever induction is
                # disabled, for uninduced picks, and for every other hold
                # phase (SETTLE/REOPEN/RELEASE), so action[i, 0:3] then stays
                # at its zero-initialized default and no torch.randn call is
                # made at all -- bit-for-bit identical to before this fix,
                # including the torch RNG stream.
                factor = self.oracles[i].noise_factor()
                if factor > 1.0:
                    delta = (
                        self.oracles[i].draw.action_noise_std
                        * factor
                        * torch.randn(3, device=self.device)
                    )
                    action[i, 0:3] = torch.clamp(delta, -1.0, 1.0)
            # Pure-hold goals (target_pos is None) and LIFT's post-convergence
            # settle (self.holding[i]) send a hard zero rotation delta --
            # action[i, 3:6] stays at its zero-initialized default, never
            # touching AttitudeHold -- exactly what OpenYAM's scripted_grasp_check.py's
            # `hold()` does (scripted_grasp_check.py:289-292 sends delta_rot=zero
            # directly, bypassing AttitudeHold entirely). Servoing rotation
            # through those steps would actively correct the tool's attitude
            # while CLOSE clamps the jaws onto the brick with the plates still
            # resting on the desk, and during LIFT's settle coast -- exactly
            # the legs OpenYAM measured with rotation left alone.
            if goal.target_pos is not None and not self.holding[i]:
                action[i, 3:6] = self.holds[i].advance(self.tool_quat(i), zero_rot_command)
        # Cycle-state stash for CyclePhaseRecorder/CycleBrickRecorder: pre-step
        # values (the recorders fire inside env.step()), refreshed in place.
        # NOT `target_brick or -1` -- brick 0 is falsy and must survive.
        if not hasattr(self.env, "oracle_cycle_phase"):
            self.env.oracle_cycle_phase = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device
            )
            self.env.oracle_cycle_brick = torch.full(
                (self.num_envs,), -1, dtype=torch.int8, device=self.device
            )
        for i in range(self.num_envs):
            goal_i, orc_i = self.goals[i], self.oracles[i]
            if goal_i is not None:
                phase_int = PHASE_TO_INT[goal_i.phase.name]
            elif orc_i is not None:
                phase_int = PHASE_TO_INT[orc_i.phase.name]  # DONE/ABANDON coast
            else:
                phase_int = -1
            self.env.oracle_cycle_phase[i] = phase_int
            brick_i = orc_i.target_brick if orc_i is not None else None
            self.env.oracle_cycle_brick[i] = brick_i if brick_i is not None else -1
        _, _, terminated, truncated, _ = self.env.step(action)
        # Isaac Lab's ManagerBasedRLEnv.step() resets terminated|truncated envs
        # to their NEXT episode's initial state before returning, so for those
        # envs every read below (joint_pos, brick_pos, pick_point, ...) would
        # be post-reset noise unrelated to the goal still sitting in
        # self.goals[i]. dones is computed here, before _advance_phases/
        # _run_watchdog, precisely so both can skip those envs for this step;
        # their stats/goal freeze at the completed episode's values and stay
        # readable via episode_stats() until the caller calls rearm().
        dones = (terminated | truncated).cpu()
        self._abandon_nan_envs(dones)
        self._advance_phases(dones)
        self._run_watchdog(dones)
        return {"dones": dones, "abandoned": self.abandoned.clone()}

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
                # Mirrors EnvOracle.report()'s RETREAT branch, which is where
                # the brick is actually marked sorted -- tracked here too
                # because that set is private to EnvOracle.
                self.stats[i]["sorted"] += 1
            orc = self.oracles[i]
            scene = self.snapshot(i)
            orc.report(result, scene)
            self._enter_phase(i, scene, ending_phase=goal.phase)

    def _phase_result(self, i: int, goal: oracle.Goal) -> oracle.PhaseResult | None:
        """None while the phase continues; a PhaseResult the step it ends."""
        if goal.target_pos is None:
            # Pure-hold goal (SETTLE/CLOSE/REOPEN/RELEASE): ends at hold_steps.
            if self.steps_in_phase[i] < goal.hold_steps:
                return None
            return oracle.PhaseResult(outcome="budget")

        if self.holding[i]:
            # Post-convergence settle (LIFT): the servo already ended this
            # phase: keep coasting (still commanding the same converged
            # target, so the residual delta is ~0) for goal.hold_steps more
            # steps -- matches OpenYAM's scripted_grasp_check.py's `rig.hold(30, ...)` before it
            # reads pinch/rise, so the gate below sees settled contact forces
            # rather than the instant the position error first dipped under
            # stop_pos.
            self.hold_counter[i] += 1
            if self.hold_counter[i] < goal.hold_steps:
                return None
            self.holding[i] = False
            outcome = self.pending_outcome[i]
            if goal.phase == oracle.Phase.LIFT:
                return self._lift_result(i, outcome)
            return oracle.PhaseResult(outcome=outcome)

        pos_err = (goal.target_pos.to(self.device) - self.pick_point(i)).norm().item()
        rot_err = self._rot_error(i).norm().item()
        converged = (
            goal.stop_pos is not None and pos_err < goal.stop_pos and rot_err < ROT_CONVERGED
        )
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
        """LIFT end: pinch evidence + rise/xy, exactly OpenYAM's gate (grasp_one)."""
        brick = self.oracles[i].target_brick
        brick_end = self._brick_pos(i, brick)
        rise = (brick_end[2] - self.brick_start[i][2]).item()
        xy_error = (brick_end[:2] - self.pick_point(i)[:2]).norm().item()
        pinch = self.pinch_evidence(i, brick)
        lifted = rise > LIFT_MIN_RISE and xy_error < LIFT_MAX_XY_ERROR
        return oracle.PhaseResult(outcome=outcome, pinch_ok=pinch["ok"], lifted=lifted)

    def _enter_phase(
        self, i: int, scene: oracle.SceneSnapshot, ending_phase: oracle.Phase | None = None
    ) -> None:
        """Fetch the next goal, re-capture the hold to it, zero the per-phase counters."""
        orc = self.oracles[i]
        goal = orc.next_goal(scene)
        self.goals[i] = goal
        self.steps_in_phase[i] = 0
        self.quiet[i] = 0
        self.holding[i] = False
        # Per-phase, not per-episode: OpenYAM measured (its Task 8) that RAISE's
        # climb legitimately nears joint4's limit (see core.oracle.RAISE_MAX_STEPS's
        # comment), and EXTEND's own approach then starts from that same
        # physical pose -- common, and not itself evidence of a stuck env. A
        # persistent counter double-counts the same joint-limit contact across
        # both phases and can trip before EITHER phase's own budget would.
        # Resetting here mirrors steps_in_phase/quiet/holding just above, but
        # is NOT a strict improvement -- see WATCHDOG_TRIP_STRIKES's comment
        # for the margin dependency this creates for RAISE_MAX_STEPS, and for
        # the REOPEN/RELEASE gap (both run fewer steps than
        # WATCHDOG_TRIP_STRIKES, so a per-phase reset means neither can ever
        # trip the watchdog on its own) this trades in for it.
        self.watchdog_strikes[i] = 0
        if goal is None:
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
            # EnvOracle.retries_used resets to 0 at the next brick's HOVER
            # (core/oracle.py), so it cannot be read live for an episode
            # total -- accumulate it here, once per REOPEN entry, the same
            # treatment "sorted" gets for the same reason.
            self.stats[i]["retries"] += 1
        if goal.phase == oracle.Phase.HOVER:
            # Fresh brick target selected inside next_goal(); capture its
            # start pose for the LIFT rise/xy gate (grasp_one's brick_start).
            # Recomputed via _brick_pos rather than read off `scene` (CPU --
            # see snapshot()) so it stays on self.device, matching
            # brick_end/pick_point in _lift_result.
            self.brick_start[i] = self._brick_pos(i, orc.target_brick)
        if goal.phase == oracle.Phase.DESCEND:
            # core.oracle deliberately leaves this to the runtime (it owns the
            # maker_arm_lab.robots import) -- the core target is the brick's
            # resting z.
            goal.target_pos[2] = goal.target_pos[2] + DESCEND_CLEARANCE_MEASURED
        # A review of OpenYAM's Task 6 (its own review finding I4) asked
        # whether commanded_yaw[i] -- frozen at whatever half-turn DESCEND's
        # fold picked, relative to the wrist heading AT THE DESK -- could be
        # the FAR representative by the time RAISE/EXTEND swing the wrist
        # toward a bin at y=+-0.21, running the wrist roll (+-2.0944 rad) onto
        # its stop. A one-time re-fold at RAISE entry (re-picking whichever of
        # commanded_yaw[i]/+-pi is nearer the CURRENT jaw_yaw) was tried and
        # instrumented (log the delta whenever it is non-zero) against every
        # case OpenYAM's Task 8 exercises: two full 6-brick episodes (12
        # picks) and the specific isolated pose whose recovery had been
        # credited to it. The fold never fired once -- by the time RAISE is
        # entered, LIFT has already actively servoed rotation to convergence
        # (its own telemetry: rot_err ~0.1 deg), so jaw_yaw(i) is already
        # within the near half-turn of commanded_yaw[i] by construction, not
        # by chance. Removed rather than kept dead: it carried a real
        # mid-carry 180-degree wrist-flip hazard (re-folding after the wrist
        # has swung partway across the +-pi/2 boundary would grab the OTHER
        # representative while a brick is in the jaw) for a branch that
        # OpenYAM's measurement shows never executes there.
        #
        # The wrist-roll-onto-its-stop scenario OpenYAM's I4 asked about IS
        # real -- OpenYAM's Task 9 batch probes reproduced it there -- but not
        # in RAISE/EXTEND, and not as a stale held heading. It happens on the
        # FRESH fold at the next brick's HOVER, whose starting wrist pose
        # RETREAT deliberately leaves 1.5 rad round; core.oracle.fold_jaw_heading
        # (called below) is the fix, and carries the measurement (on OpenYAM's
        # arm). A re-fold here would still be wrong for the same wrist-flip
        # reason.
        self.holds[i].capture(self.desired_quat(i, goal.yaw, goal.tilt_rad, goal.yaw_hold_xy))
        if goal.stall_steps:
            self.previous_pick_point[i] = self.pick_point(i).clone()

    def _abandon_nan_envs(self, dones: torch.Tensor) -> None:
        """Abandon any env whose robot articulation OR brick rotation state has
        gone NaN (physics divergence: seen live as a flood of USD
        OrthogonalizeBasis warnings followed by a NaN jaw heading crashing
        fold_jaw_heading -- either from a NaN robot joint/tool pose directly,
        or from a NaN brick root_quat_w flowing into goal.yaw via
        _brick_headings/scene.brick_yaw with the robot state itself still
        clean). Abandoning routes the episode to the export filter's discard
        pile, exactly like a watchdog trip, instead of killing the whole batch
        process.

        Batched into one `.cpu()` transfer for the whole env-and-brick check
        (mirrors `_brick_headings`'s reasoning) instead of two `bool(...)`
        device syncs per env.
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
            self.watchdog_strikes[i] = self.watchdog_strikes[i] + 1 if tripped else 0
            if self.watchdog_strikes[i] >= WATCHDOG_TRIP_STRIKES:
                self.abandoned[i] = True
                self.goals[i] = None
                if self.stats[i]["abandon"] is None:
                    self.stats[i]["abandon"] = "watchdog"

    # -- episode lifecycle -------------------------------------------------

    def rearm(self, env_ids: list[int]) -> None:
        """Re-seed the envs in ``env_ids`` after their caller has reset them."""
        for i in env_ids:
            draw = oracle.sample_episode_draw(self.rng, self.noise)
            self.oracles[i] = oracle.EnvOracle(
                self.bl, self.num_bricks, draw, induction=self.induction, rng=self.rng
            )
            self.holds[i] = AttitudeHold(
                rot_scale=self.scale_rot, gain=SERVO_GAIN_ROT, give_up_error=float("inf")
            )
            self.commanded_yaw[i] = self.jaw_yaw(i)
            self.holds[i].capture(self.tool_quat(i))
            self.brick_start[i] = None
            self.previous_pick_point[i] = None
            self.steps_in_phase[i] = 0
            self.quiet[i] = 0
            self.holding[i] = False
            self.watchdog_strikes[i] = 0
            self.abandoned[i] = False
            self.stats[i] = {"retries": 0, "sorted": 0, "abandon": None}
            self.goals[i] = self.oracles[i].next_goal(self.snapshot(i))  # SETTLE

    def episode_stats(self, env_id: int) -> dict:
        """Retries used, bricks sorted, abandon reason, induced counters -- for the metadata sidecar.

        Retries/sorted/abandon read from ``self.stats`` (not
        ``EnvOracle.retries_used``, which resets to 0 at every new brick's HOVER
        -- see ``_enter_phase``'s REOPEN branch), so those stay episode-totals.
        ``induced_offset``/``induced_noise`` read from the ``EnvOracle`` object
        instead, because they ARE already episode-cumulative there (see
        ``EnvOracle``'s own counters). Both sources rely on the same lifetime
        guarantee: the just-finished env's ``EnvOracle`` and ``self.stats``
        entry both survive, and stay readable, until the caller calls
        ``rearm()`` on it.
        """
        orc = self.oracles[env_id]
        return {
            **self.stats[env_id],
            "induced_offset": orc.induced_offset,
            "induced_noise": orc.induced_noise,
        }

    # -- frames --------------------------------------------------------------
    #
    # Every position below is env-origin subtracted (see the module docstring):
    # it is what `SceneSnapshot` documents and what makes `pick_point` directly
    # comparable to `Goal.target_pos`, whatever env produced it.

    def tool_quat(self, i: int) -> torch.Tensor:
        return self.robot.data.body_quat_w[i, self.tool]

    def pick_point(self, i: int) -> torch.Tensor:
        """Env-local position of the point on the tool axis a desk-resting brick occupies."""
        origin = self.env.scene.env_origins[i]
        return (
            self.robot.data.body_pos_w[i, self.tool]
            - origin
            + math_utils.quat_apply(self.tool_quat(i), self.pick_offset)
        )

    def jaw_yaw(self, i: int) -> float:
        """World heading of the jaw closing axis (JAW_AXIS_BODY in the tool frame), projected flat."""
        v = math_utils.quat_apply(self.tool_quat(i), self.jaw_axis)
        return math.atan2(v[1].item(), v[0].item())

    def _brick_pos(self, i: int, brick: int) -> torch.Tensor:
        return self.env.scene[f"brick_{brick}"].data.root_pos_w[i] - self.env.scene.env_origins[i]

    def _brick_headings(self, i: int) -> torch.Tensor:
        """(B,) CPU: every brick's long-axis heading, as ``desired_quat`` folds against.

        Batched into one ``quat_apply`` plus one ``.cpu()`` transfer for the
        whole env instead of a per-brick ``.item()`` sync, since ``snapshot()``
        needs this on CPU anyway (see its docstring).
        """
        quats = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_quat_w[i] for b in range(self.num_bricks)]
        )
        long_axis = math_utils.quat_apply(quats, self.ex.expand(self.num_bricks, 3)).cpu()
        return torch.atan2(long_axis[:, 1], long_axis[:, 0])

    def wrist_roll(self, i: int) -> float:
        """Wrist-roll joint position, rad -- the joint the jaw-heading fold spends."""
        return self.robot.data.joint_pos[i, self.arm_ids[WRIST_ROLL_INDEX]].item()

    def _base_yaw_range(
        self, i: int, hold_xy: tuple[tuple[float, float], ...]
    ) -> tuple[float, float]:
        """Signed joint1 excursion, relative to now, over ``hold_xy``.

        Each waypoint's joint1 is approximated by its horizontal bearing, which
        OpenYAM's arm's joint1 tracked directly: MEASURED, on OpenYAM's arm,
        ``atan2(pick_y, pick_x) - joint1 = 0.006 +- 0.097 rad`` over a full
        13437-step instrumented run there. Not yet re-validated on the Maker
        Arm. 0.0 is always in the range because the arm is at the current pick
        point now. Empty ``hold_xy`` (every phase that does not command a yaw)
        collapses to no swing.
        """
        if not hold_xy:
            return (0.0, 0.0)
        pick = self.pick_point(i).tolist()
        here = math.atan2(pick[1], pick[0])
        swings = [0.0] + [oracle.wrap_to_pi(math.atan2(y, x) - here) for x, y in hold_xy]
        return (min(swings), max(swings))

    def desired_quat(
        self,
        i: int,
        yaw: float | None,
        tilt_rad: float,
        hold_xy: tuple[tuple[float, float], ...] = (),
    ) -> torch.Tensor:
        """Tool z-axis tilted ``tilt_rad`` toward +x; jaws squared to ``yaw`` (or held).

        Ported from OpenYAM's ``GraspRig.desired_quat`` (scripts/scripted_grasp_check.py)
        and generalized from a fixed tool-axis-straight-down (tilt=0) to an arbitrary
        tilt, so EXTEND/RELEASE can angle the tool toward the bin rim
        (``oracle.RELEASE_TILT_RAD``) while every other phase keeps OpenYAM's
        script's geometry exactly -- at tilt=0 the Gram-Schmidt step below is a
        no-op and reduces to the script's closed-form ``axis_y = (s, -c, 0)``.
        The result is assembled in the oracle's canonical tool frame (x = jaw
        axis, z = tool axis) and then converted into the ``grasp_center`` body
        frame by ``canonical_to_body`` before it is returned.

        A jaw is symmetric under a half turn, so an explicit ``yaw`` is folded
        to one of ``yaw``/``yaw + pi`` by ``oracle.fold_jaw_heading`` -- which
        picks on the wrist roll's REMAINING TRAVEL over every waypoint in
        ``hold_xy`` (``Goal.yaw_hold_xy``), not on raw proximity to the current
        wrist heading. See that function for the measurement (on OpenYAM's
        arm) and for the two watchdog abandons proximity alone caused there.
        ``yaw=None`` (every phase after DESCEND: there is no brick reference
        once it has left the desk) instead holds the heading last commanded --
        ``tilt_rad`` is never None and is always applied, so RETREAT can also
        relevel while still holding that heading.
        """
        if yaw is not None:
            want = oracle.fold_jaw_heading(
                yaw + math.pi / 2,
                self.jaw_yaw(i),
                self.wrist_roll(i),
                self.wrist_roll_limits,
                base_yaw_range=self._base_yaw_range(i, hold_xy),
                roll_sign=WRIST_ROLL_SIGN,
                base_sign=BASE_YAW_SIGN,
            )
            self.commanded_yaw[i] = want
        else:
            want = self.commanded_yaw[i]
        c, s = math.cos(want), math.sin(want)
        axis_x = torch.tensor([c, s, 0.0], device=self.device)
        axis_z = torch.tensor([math.sin(tilt_rad), 0.0, -math.cos(tilt_rad)], device=self.device)
        axis_y = torch.cross(axis_z, axis_x, dim=-1)
        axis_y = axis_y / axis_y.norm()
        axis_x = torch.cross(axis_y, axis_z, dim=-1)  # re-orthogonalize; a no-op at tilt=0
        rot = torch.stack([axis_x, axis_y, axis_z], dim=1)
        rot = canonical_to_body(rot, self.canon_in_body)
        return math_utils.quat_from_matrix(rot)

    def _rot_error(self, i: int) -> torch.Tensor:
        """World-frame axis-angle rotation taking the tool body (`grasp_center`) to this
        phase's held reference."""
        quat_des = self.holds[i].reference
        err = math_utils.quat_mul(quat_des, math_utils.quat_conjugate(self.tool_quat(i)))
        return math_utils.axis_angle_from_quat(err)

    def snapshot(self, i: int) -> oracle.SceneSnapshot:
        """Built on CPU: ``EnvOracle`` (core.oracle, deliberately isaaclab/device-free)
        compares this against ``layout.bin_interior_centers()``/``bin_half_extents()``
        inside ``checker.bricks_at_rest_in_bins`` -- plain CPU tensors -- so a
        CUDA snapshot raises at the first mixed-device comparison (the first
        HOVER-entry brick lookup). Position/attitude servoing still reads
        ``pick_point``/``_brick_pos``/``tool_quat`` directly (self.device) --
        only the values handed to ``core.oracle`` move to CPU here.
        """
        brick_pos = torch.stack([self._brick_pos(i, b) for b in range(self.num_bricks)]).cpu()
        brick_lin_vel = torch.stack(
            [self.env.scene[f"brick_{b}"].data.root_lin_vel_w[i] for b in range(self.num_bricks)]
        ).cpu()
        return oracle.SceneSnapshot(
            brick_pos=brick_pos,
            brick_yaw=self._brick_headings(i),
            brick_lin_vel=brick_lin_vel,
            pick_point=self.pick_point(i).cpu(),
        )

    def pinch_evidence(self, i: int, brick: int) -> dict:
        """Is the brick pinched between the jaws, or merely entrained in the pocket?

        ``force_matrix_w`` is (num_envs, num_bodies, num_filters, 3); each
        sensor prim is a single finger body and the filters are the bricks in
        scene order, so ``[i, 0, brick]`` isolates this env's this brick.
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
                stall >= PINCH_MIN_STALL
                and weakest >= PINCH_MIN_FORCE
                and asymmetry <= PINCH_MAX_ASYMMETRY
            ),
        }
