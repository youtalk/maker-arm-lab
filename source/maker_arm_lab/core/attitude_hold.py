"""Closed-loop attitude hold for the differential-IK action term's open rotation channel.

Torch-only, no isaaclab: this module is in the isaaclab-free zone and is
CPU-unit-tested (``tests/test_attitude_hold.py``).

``AttitudeHold`` is the fix for a behaviour of
``DifferentialInverseKinematicsAction`` that would otherwise leave any caller
unable to hold a fixed tool orientation, described next.

Provenance of the numbers below: they were measured on openyam-lab's arm at
30 Hz (task-9-report.md, task-10-report.md, and openyam-lab's teleoperation
self-check harness and ``scripts/scripted_grasp_check.py``) -- none of it on
the Maker Arm. The mechanism the numbers document -- the rotation channel is a
free integrator, drift compounds with translation, an integrated reference can
run away, a give-up is needed against poses the arm cannot escape -- is a
property of ``DifferentialInverseKinematicsAction`` and applies unchanged; the
specific angles and step counts do not carry over automatically. This
milestone's own ``scripts/scripted_grasp_check.py`` (Task D3) and the
release-envelope check are what will produce Maker-Arm-specific versions;
until then, read the figures below as illustrating the mechanism, not as
Maker Arm bounds.

Why an attitude hold is needed at all
-------------------------------------
``DifferentialInverseKinematicsAction.process_actions`` reads the **measured**
end-effector pose every control step and calls
``isaaclab.utils.math.apply_delta_pose``, which computes
``target_rot = quat_mul(rot_delta_quat, source_rot)`` -- the commanded rotation
pre-multiplied onto the *measured* attitude. So a zero rotation command does not
mean "hold the attitude you were asked for", it means "adopt whatever attitude
you happen to have". Orientation error is never fed back: the rotation channel is
a free integrator.

Position does not behave this way in practice, because the position channel is
also re-based on measurement but openyam-lab's arm has gravity compensation
and holds its pose to ~1e-6 rad under a zero command -- not yet re-checked on
the Maker Arm. Orientation drifts because *translating* the tool through the
arm's redundancy quietly rotates it, and nothing pulls it back. Measured on
openyam-lab's arm (its task-10-report.md): a position-only closed-loop driver
translating ~0.1 m and descending walked the tool axis from 6 deg off world -z
to **36-64 deg** over a few hundred control steps, with no rotation ever
commanded. A caller commanding only translation deltas would watch the gripper
slowly tip over.

The hold, and its reference-capture policy
------------------------------------------
``AttitudeHold`` keeps a **reference attitude** and each step emits the rotation
action that drives the measured attitude onto it. The reference is:

* **captured** from the measured attitude at env reset, and whenever the
  caller wants to re-arm it (any call to ``capture``);
* **advanced** by exactly the operator's commanded rotation delta -- so commanding
  rotation moves the thing being held rather than switching the hold off;
* **never** silently re-captured from measurement.

That last point is the deliberate choice, and it is what makes the hold actually
remove drift rather than relocate it. The obvious cheaper policy -- "pass the
operator's rotation command straight through while they are commanding, and
re-capture the reference to the measured attitude when they let go" -- holds
attitude while idle but *bakes in* every degree of parasitic drift that
accumulated while the operator was rotating, because the recapture cannot tell
intended rotation from drift. At the measured drift rate that is a real cost: a
rotation command held for a hundred control steps is a hundred steps during which
the tool is also being translated, and the tip that accrues becomes the new
"upright". Advancing the reference by the command instead means the reference is
a pure integral of operator intent and drift is always an error to be corrected.

Anti-windup, and the failure mode it leaves
-------------------------------------------
An integrated reference can run away from a tool that cannot follow it (joint
stop, obstruction, singularity), and a runaway reference means a saturated
command that persists long after the caller's rotation command stops.
``max_lead`` bounds it: per axis, the reference stops being advanced once it
leads the measured attitude by ``max_lead`` radians on that axis *and* the
command would push it further out. The bound is a *freeze*, never a pull-back
-- the reference is never moved toward the measurement, because that is
exactly the drift absorption this design exists to avoid.

``max_lead`` must be at least one full-deflection step (``rot_scale / gain``) or
it would cap the caller's sustained rotation rate; the constructor rejects
smaller values. The default is three steps -- derived from
``core.action_scale.IK_SCALE_ROT`` rather than typed, so it stays three steps
if the rotation scale is ever re-measured -- so the tool coasts at most
``max_lead`` (0.03 rad = 1.7 deg at today's 0.01 rad/step, ~4 control steps at
30 Hz) past where the commanded rotation stopped.

**Failure mode, stated plainly.** If a caller commands rotation into a limit
the arm genuinely cannot pass, the reference ends up ``max_lead`` beyond reach
and the hold keeps commanding up to full rotation deflection into that limit
until the caller rotates back or re-captures the reference. This is the same
thing that happens with the stock action term while a command holds against a
stop -- the hold extends it past release by ``max_lead`` worth of command. It
is bounded and it is visible (the arm does not move and the joint saturates);
any caller can clear it by calling ``capture`` again. The alternative --
shrinking ``max_lead`` below one step -- trades this for a permanently
throttled rotation rate, which is worse.

Giving up, and why it is not optional
-------------------------------------
There is a second, worse failure available: a pose where the arm cannot rotate
about the axis the correction needs at all. On openyam-lab's arm that axis was
pitch, supplied by a single joint that ran onto a lower stop as the tool was
raised and retracted; whether the Maker Arm exposes the same coupling on any
of its own joints has not been checked. Measured there, on a self-check
harness this port does not carry over: that joint sat on its stop -- per-phase
minimum margin at or below 0.005 rad -- for 440 steps across the reach path,
and the servo demanded full pitch deflection against a joint that could not
move. It recovered to 0.072 rad only over the run's last 180 steps, by which
time the damage was done.

An earlier build of this class, with no give-up and with the componentwise
saturation described on the magnitude clip below, drifted the tool axis
**62.64 deg** on that path -- where leaving the rotation channel open drifts only
**24.21 deg**. A hold that is *worse than no hold* in a pose the caller can
reach is not shippable, so the hold gives up: once the error has stayed above
``give_up_error`` for ``give_up_steps`` consecutive steps *without shrinking*, it
**disengages** -- ``advance`` stops servoing, returns the caller's rotation
command (still magnitude-bounded, still a fresh tensor), and keeps the reference
glued to the measurement so re-arming is bumpless. ``gave_up`` counts the
concessions and ``engaged`` reports the current state per env, so the caller can
say so out loud. Only ``capture`` re-arms it, which is on the caller to invoke
-- at env reset, and again whenever it wants to re-arm a hold that gave up.

Disengaging, rather than merely re-capturing and carrying on, is what the
measurements forced. Re-capturing alone bounded the servo *error* (62.68 deg to
27.41 deg) but not the damage: the hold simply re-engaged against the same
unwinnable pose and the tool axis still drifted 60.63 deg. A hold that has lost
has to stop pulling, not aim again. With the give-up as it now stands the same
path measures **28.88 deg**, i.e. 4.7 deg worse than having no hold at all, which
is what it costs to work out that it has lost.

The threshold is not a tuning knob picked by feel, **for the teleoperation case
openyam-lab tuned it for**. A legitimate teleop error there was bounded: the
anti-windup caps the reference's lead at ``max_lead`` per axis, and the arm's
rotation tracking adds about one more step of lag, so anything an operator
could produce stayed near ``max_lead``. Measured over a full in-envelope pick
cycle on that arm, the worst hold error was 2.29 deg against a ``max_lead`` of
1.72 deg and a ``give_up_error`` of 14.3 deg -- under teleoperation there, the
defaults could not fire.

**They can fire, legitimately and unhelpfully, under the callers this
milestone actually has -- read this before reusing the defaults.** A scripted
driver that captures a *desired* attitude and servos to it starts with
whatever error the previous phase left: measured on openyam-lab's
``scripts/scripted_grasp_check.py``, a hover phase began 13.6-25.0 deg off
target, i.e. at or above ``give_up_error`` from the first step. Whether this
milestone's own scripted grasp check (Task D3) starts legs that far off
target has not been checked. The "without shrinking" clause is what keeps
that safe -- a converging servo posts a falling error every step and never
trips -- but a leg that *stalls* for ``give_up_steps`` while still far from
target (contact, a wrist joint momentarily on a limit) will disengage and let
the tool tip freely for the rest of the leg. Such a caller should either pass
a ``give_up_steps`` matching its own stall tolerance, or pass
``give_up_error=float("inf")`` to disable the give-up entirely and rely on
its own phase budgets -- which is what openyam-lab's
``scripts/scripted_grasp_check.py`` did. Either way, check ``engaged`` at the
end of a leg rather than assuming the attitude was held.

After giving up, the hold holds the attitude the tool ended at. The tip is not
undone; the caller has to correct it. That is the stock action term's behaviour,
which is the floor this guarantees rather than a feature.

Batching
--------
Every tensor here is generic in the leading dimensions, so one instance can hold
N attitudes: pass ``(N, 4)`` and ``(N, 3)`` and get ``(N, 3)`` back. The give-up
state is per env too -- one env being dragged away disengages only that env.
``gave_up`` is a scalar total across envs and across time; ``engaged`` is the
per-env tensor.
"""

import torch

from .action_scale import IK_SCALE_ROT

# Rotation servo gain, raw-action units per (radian of error / rot_scale). 1.0
# is unity feedback: an error of exactly one step's worth of commanded rotation
# produces exactly one step's worth of action, so a command issued from rest
# reaches the action term unchanged. Same value as the rotation servo gain
# openyam-lab's scripted grasp driver used (its scripts/scripted_grasp_check.py);
# not re-validated against this milestone's own driver once Task D3 builds it.
DEFAULT_HOLD_GAIN = 1.0

# Radians the reference may lead the measured attitude by before it stops being
# advanced (anti-windup). Three full-deflection rotation steps at the env's
# IK_SCALE (0.01 rad/step today, so 0.03 rad). See the module docstring for the
# trade this sets.
#
# DERIVED, not typed: "three steps" is the invariant, and it is the quantity
# the module docstring describes. Hard-coding 0.03 meant a re-measurement of
# the rotation scale silently changed the band's width in steps -- 0.02
# rad/step would have made this 1.5 steps -- while every test kept passing
# and every message kept naming DEFAULT_MAX_LEAD without naming the cause.
_MAX_LEAD_STEPS = 3.0
DEFAULT_MAX_LEAD = _MAX_LEAD_STEPS * max(IK_SCALE_ROT)

# Attitude error (rad) and consecutive-step count at which the hold concedes that
# the reference is unreachable. 0.25 rad is 14.3 deg, ~8x DEFAULT_MAX_LEAD and
# ~6x the worst error openyam-lab measured over a full in-envelope pick cycle on
# its arm (2.29 deg, not re-validated on the Maker Arm), so **teleoperation**
# could not reach it there -- which is the only case these defaults are tuned
# for. A scripted caller servoing to a commanded attitude routinely starts a leg
# above this (13.6-25.0 deg on openyam-lab's scripts/scripted_grasp_check.py)
# and should pass its own value, or float("inf") to disable the give-up; see
# the "Giving up" section of the module docstring before reusing these.
# 15 steps is 0.5 s at the env's 30 Hz.
#
# 15 is short only because the condition also requires the error to be *not
# shrinking*. Time alone cannot separate "losing" from "recovering": full
# deflection corrects 0.573 deg per step, so clawing back a 15 deg knock
# legitimately takes 26 steps, and a purely time-based rule long enough to allow
# that is also long enough to do several degrees of damage in the pose where the
# hold cannot win. Requiring the error to hold or grow separates them directly --
# a recovering hold shows a falling error every step and never trips.
DEFAULT_GIVE_UP_ERROR = 0.25
DEFAULT_GIVE_UP_STEPS = 15

# Radians per step of error reduction that counts as making progress. One
# thousandth of a full-deflection rotation step: real recovery moves two orders
# of magnitude faster, and solver noise moves less.
_GIVE_UP_PROGRESS = 1.0e-5

# Below this angle a quaternion's vector part carries no usable axis, so the
# small-angle limits are substituted instead of dividing by ~0.
_ANGLE_EPS = 1.0e-7


# --------------------------------------------------------------------------
# Quaternion helpers, (w, x, y, z) -- isaaclab.utils.math's convention.
#
# Duplicated here rather than imported from isaaclab.utils.math because this
# module is in the isaaclab-free zone (maker_arm_lab.core) so that the attitude
# hold can be unit-tested on CPU without a simulator. They are a handful of
# lines each and are round-trip tested against each other.
# --------------------------------------------------------------------------


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a * b``. Shapes broadcast over all but the last dim."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion."""
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def quat_from_axis_angle(vec: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle, radians) -> unit quaternion."""
    angle = vec.norm(dim=-1, keepdim=True)
    # sin(angle/2)/angle, with its angle->0 limit of 1/2 substituted near zero.
    half_sinc = torch.where(
        angle > _ANGLE_EPS,
        torch.sin(0.5 * angle) / angle.clamp_min(_ANGLE_EPS),
        torch.full_like(angle, 0.5),
    )
    return torch.cat([torch.cos(0.5 * angle), vec * half_sinc], dim=-1)


def axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion -> rotation vector, always the short way round.

    ``q`` and ``-q`` are the same rotation but read as ``theta`` and
    ``2*pi - theta``; the sign is normalised so a 350 deg error is reported as
    -10 deg. Without that the hold would unwind a small error the long way at
    full deflection.
    """
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    real = quat[..., :1].clamp(-1.0, 1.0)
    imaginary = quat[..., 1:]
    sin_half = imaginary.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, real)
    # angle/sin(angle/2), with its angle->0 limit of 2 substituted near zero.
    scale = torch.where(
        sin_half > _ANGLE_EPS,
        angle / sin_half.clamp_min(_ANGLE_EPS),
        torch.full_like(angle, 2.0),
    )
    return imaginary * scale


class AttitudeHold:
    """Closes the differential-IK action term's open rotation loop.

    See the module docstring for why this is needed, what the reference-capture
    policy is, and what failure mode the anti-windup bound leaves behind.

    Stateful: one instance per env batch, holding the reference attitude between
    steps. ``capture`` must be called before the first ``advance``.

    All tensors are in the frame the action term works in -- the robot's **root**
    frame, since ``_compute_frame_pose`` subtracts the root transform before
    calling ``apply_delta_pose``. For a base at identity that equals the world
    frame, but the caller must convert explicitly rather than relying on it.
    """

    def __init__(
        self,
        rot_scale: torch.Tensor,
        gain: float = DEFAULT_HOLD_GAIN,
        max_lead: float = DEFAULT_MAX_LEAD,
        give_up_error: float = DEFAULT_GIVE_UP_ERROR,
        give_up_steps: int = DEFAULT_GIVE_UP_STEPS,
    ):
        """
        Args:
            rot_scale: Radians per unit raw action, per axis, shape (3,). This is
                the env's effective ``IK_SCALE[3:6]`` -- including any extra
                scale factor a caller applies on top of it. Getting this wrong
                de-tunes the servo gain by exactly that factor.
            gain: Servo gain, in raw-action units per unit of scaled error.
            max_lead: Radians the reference may lead the measurement by before it
                stops being advanced.
            give_up_error: Attitude error (rad) above which the hold may be
                losing. The default is tuned for openyam-lab's teleoperation
                case, where a legitimate error never left the ``max_lead``
                band. Pass ``float("inf")`` to disable the give-up entirely --
                the right choice for a scripted caller that servos to a
                commanded attitude from a large initial error and has its own
                phase budgets.
            give_up_steps: Consecutive steps above ``give_up_error`` *without the
                error shrinking* before the hold disengages. Raise it to match a
                caller's own tolerance for a stalled approach.
        """
        rot_scale = torch.as_tensor(rot_scale, dtype=torch.float32)
        if rot_scale.shape != (3,):
            raise ValueError(f"rot_scale must be (3,), got {tuple(rot_scale.shape)}")
        if not bool(torch.all(rot_scale > 0.0)):
            raise ValueError(f"rot_scale must be positive on every axis, got {rot_scale.tolist()}")
        if gain <= 0.0:
            raise ValueError(f"gain must be positive, got {gain}")
        # Below one full-deflection step the bound would bind before a single
        # command could be applied, silently capping the operator's sustained
        # rotation rate at less than the arm can do. Fail rather than de-tune.
        one_step = float(rot_scale.max().item()) / gain
        if max_lead < one_step:
            raise ValueError(
                f"max_lead={max_lead} is below one full-deflection step ({one_step}); "
                "that would throttle the operator's sustained rotation rate"
            )
        # Anything the operator can legitimately produce sits near max_lead, so a
        # give-up threshold at or below it would fire during normal use.
        if give_up_error <= 2.0 * max_lead:
            raise ValueError(
                f"give_up_error={give_up_error} is not clear of max_lead={max_lead}; "
                "it would concede during ordinary commanded rotation"
            )
        if give_up_steps < 1:
            raise ValueError(f"give_up_steps must be >= 1, got {give_up_steps}")
        self.rot_scale = rot_scale
        self.gain = gain
        self.max_lead = max_lead
        self.give_up_error = give_up_error
        self.give_up_steps = give_up_steps
        # Total concessions, summed over envs and over time. A caller should
        # watch it, so a tilted tool is never inherited silently. `engaged` is
        # the per-env state.
        self.gave_up = 0
        self._reference: torch.Tensor | None = None
        # All three are per-env, shape (..., 1), allocated by capture(). Scalars
        # would make one env's bad pose disengage the hold for the whole batch.
        self._losing: torch.Tensor | None = None
        self._previous_error: torch.Tensor | None = None
        self._engaged: torch.Tensor | None = None

    @property
    def reference(self) -> torch.Tensor:
        """The held attitude. Raises if nothing has been captured yet."""
        if self._reference is None:
            raise RuntimeError("AttitudeHold.capture() must be called before advance()")
        return self._reference

    @property
    def engaged(self) -> torch.Tensor:
        """Per-env: False where the hold has conceded, until ``capture`` re-arms it.

        Shape ``(...)`` -- the leading dimensions of whatever was captured, so a
        single-env hold gives a one-element tensor and ``bool(hold.engaged)``
        works. For a batch, reduce explicitly (``.all()`` / ``.any()``).
        """
        if self._engaged is None:
            raise RuntimeError("AttitudeHold.capture() must be called before engaged is defined")
        return self._engaged.squeeze(-1)

    def capture(self, quat: torch.Tensor) -> None:
        """Adopt ``quat`` as the attitude to hold, and (re-)engage every env.

        Called at reset, and again whenever the caller wants to re-arm it.
        """
        self._reference = _normalize(quat.detach().clone().to(dtype=torch.float32))
        state = self._reference.shape[:-1] + (1,)
        device = self._reference.device
        self._losing = torch.zeros(state, dtype=torch.int64, device=device)
        self._previous_error = torch.full(state, float("inf"), device=device)
        self._engaged = torch.ones(state, dtype=torch.bool, device=device)

    def advance(self, quat_measured: torch.Tensor, rot_command: torch.Tensor) -> torch.Tensor:
        """One control step of the hold.

        Args:
            quat_measured: Measured end-effector attitude, shape (..., 4).
            rot_command: The raw rotation action being requested, shape (..., 3) --
                the ``[drx dry drz]`` slice of the caller's command, in [-1, 1].

        Returns:
            The raw rotation action to send, shape (..., 3), always a fresh
            tensor and always bounded to unit magnitude. With the tool on its
            reference and a single-axis command this equals ``rot_command``
            exactly, so the hold is transparent from a settled pose.
        """
        # Never-captured and gave-up are different states and must not be
        # conflated: the second passes commands through, the first is a bug.
        if self._reference is None:
            raise RuntimeError("AttitudeHold.capture() must be called before advance()")
        quat_measured = quat_measured.to(dtype=torch.float32)
        rot_scale = self.rot_scale.to(quat_measured.device)

        # Concede first, per env, on the evidence of the previous steps, so
        # everything below runs against a reference still worth servoing to.
        conceding = self._engaged & (self._losing >= self.give_up_steps)
        if bool(conceding.any()):
            self.gave_up += int(conceding.sum())
            self._engaged = self._engaged & ~conceding
        engaged = self._engaged

        # Advance the reference by the command, except on axes where it already
        # leads the measurement by max_lead AND the command would push it further
        # out (anti-windup). Per axis and directional, both deliberately: a
        # norm-wide freeze lets an error the arm cannot correct on one axis
        # silently swallow the input on the others, and a non-directional freeze
        # also blocks the command that would relieve it. The test is on the lead
        # BEFORE advancing, so a command issued from a settled pose is never
        # dropped. Disengaged envs instead glue the reference to the measurement,
        # which both stops the servo and makes a later capture() bumpless.
        lead = axis_angle_from_quat(quat_mul(self._reference, quat_conjugate(quat_measured)))
        blocked = (lead.abs() >= self.max_lead) & ((rot_command * lead) > 0.0)
        applied = torch.where(blocked, torch.zeros_like(rot_command), rot_command)
        advanced = _normalize(quat_mul(quat_from_axis_angle(applied * rot_scale), self._reference))
        self._reference = torch.where(engaged, advanced, _normalize(quat_measured))

        # Servo the measurement onto the (possibly advanced) reference, bounding
        # the demand by MAGNITUDE rather than per component. Componentwise
        # clipping preserves each axis' sign but not the correction's *axis*: a
        # 45 deg error of (0.05, 0.78, 0.02) rad clips to (1, 1, 1), a rotation
        # about (1,1,1) instead of about y, injecting two large corrections
        # nobody asked for. Measured on openyam-lab's arm, that turned the hold
        # into a divergence: 62.64 deg of tool-axis drift on the `reach` self
        # check, where leaving the rotation channel open drifts 24.21 deg. Not
        # re-validated on the Maker Arm.
        #
        # NOTE the bound is applied in ACTION space, so what it preserves is the
        # direction of the *demand*, which coincides with the direction of the
        # *error* only because IK_SCALE[3:6] is isotropic (0.01 on all three
        # axes). Under an anisotropic rotation scale this would still be a sane
        # bound -- it is the actuator envelope that matters -- but it would no
        # longer rotate about the error axis, and neither this comment nor
        # test_hold_correction_keeps_the_error_axis_when_it_saturates would mean
        # what they say. Revisit both if IK_SCALE's rotation entries ever differ.
        #
        # Consequence, and it is a deviation from an unclipped multi-axis
        # command: deflecting all three rotation axes at once yields one
        # calibrated step of total rotation rather than up to sqrt(3) of them.
        # An unclipped command does not bound that; IK_SCALE was calibrated
        # per axis, so bounding it is the conservative reading. Single-axis
        # commands -- what a single-axis rotation request produces -- are
        # untouched.
        lead = axis_angle_from_quat(quat_mul(self._reference, quat_conjugate(quat_measured)))
        demand = torch.where(engaged, self.gain * lead / rot_scale, rot_command)
        norm = demand.norm(dim=-1, keepdim=True)
        action = demand * (norm.clamp(max=1.0) / norm.clamp_min(_ANGLE_EPS))

        # Bookkeeping for the next step's concession test. A disengaged env is
        # reset rather than accumulated, so re-arming starts from a clean slate.
        error = lead.norm(dim=-1, keepdim=True)
        losing = (
            engaged
            & (error > self.give_up_error)
            & (error > self._previous_error - _GIVE_UP_PROGRESS)
        )
        self._losing = torch.where(losing, self._losing + 1, torch.zeros_like(self._losing))
        self._previous_error = torch.where(engaged, error, torch.full_like(error, float("inf")))
        return action


def _normalize(quat: torch.Tensor) -> torch.Tensor:
    """Renormalise. The reference is built by repeated multiplication, which creeps."""
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(_ANGLE_EPS)
