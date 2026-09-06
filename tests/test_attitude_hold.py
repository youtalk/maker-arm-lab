"""Tests for maker_arm_lab.core.attitude_hold (CPU-only).

``DifferentialInverseKinematicsAction`` re-bases its desired orientation on the
*measured* end-effector attitude every control step (``apply_delta_pose``
pre-multiplies the delta), so a zero rotation command means "keep whatever
attitude you have right now" and the rotation channel is a free integrator: the
tool tips over on its own while a caller commands only translation deltas.
``AttitudeHold`` closes that loop. Its behaviour is pure quaternion algebra, so
it is testable on CPU without a simulator, and these tests are the only place
the algebra is checked.
"""

import math

import pytest
import torch

from maker_arm_lab.core.action_scale import IK_SCALE, IK_SCALE_ROT
from maker_arm_lab.core.attitude_hold import (
    DEFAULT_HOLD_GAIN,
    DEFAULT_MAX_LEAD,
    AttitudeHold,
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_axis_angle,
    quat_mul,
)

# The env's rotation scale itself, not a copy of it. This used to be a retyped
# `torch.tensor([0.01, 0.01, 0.01])` with a comment conceding that nothing here
# could catch the two drifting apart; Task 13's review moved the constant into
# the isaaclab-free `core` zone precisely so this line can bind the real object.
# The numbers in these tests therefore describe the real arm by construction.
ROT_SCALE = torch.tensor(IK_SCALE_ROT)


# --------------------------------------------------------------------------
# Constant contract: the anti-windup band is measured in action steps
# --------------------------------------------------------------------------


def test_max_lead_is_three_full_deflection_rotation_steps():
    """``DEFAULT_MAX_LEAD`` must stay three steps wide, whatever ``IK_SCALE`` is.

    "Three steps" is what the module docstring promises. The failure this pins
    is silent: raise the rotation entries of ``IK_SCALE`` to 0.02 with
    ``DEFAULT_MAX_LEAD`` left at a literal 0.03 and the band becomes 1.5 steps
    -- still above the constructor's one-step floor, so nothing raises, while
    the anti-windup band the hold exists to provide is silently halved.
    """
    assert IK_SCALE_ROT == IK_SCALE[3:6]
    assert DEFAULT_MAX_LEAD == pytest.approx(3.0 * max(IK_SCALE_ROT))
    # The constructor's own floor, restated against the shared constant: the
    # band must admit at least one full-deflection step or it caps the
    # operator's sustained rotation rate.
    assert DEFAULT_MAX_LEAD >= max(IK_SCALE_ROT) / DEFAULT_HOLD_GAIN


# --------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z), matching isaaclab.utils.math's convention
# --------------------------------------------------------------------------


def _rot_z(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def _rot_y(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])


def test_axis_angle_round_trip():
    for vec in [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1e-9, 0.0, 0.0]),
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([0.0, 0.0, 3.0]),
    ]:
        got = axis_angle_from_quat(quat_from_axis_angle(vec))
        assert torch.allclose(got, vec, atol=1e-6), f"{vec} -> {got}"


def test_axis_angle_takes_the_short_way_round():
    """A 350 deg rotation must read as -10 deg, or the hold would unwind the long way."""
    got = axis_angle_from_quat(_rot_z(math.radians(350.0)))
    assert math.isclose(got[2].item(), math.radians(-10.0), abs_tol=1e-6)


def test_quat_mul_is_the_world_frame_pre_multiplication():
    """``apply_delta_pose`` does ``quat_mul(delta, source)`` -- world frame, not tool frame."""
    # A quarter turn about world z applied to a frame already pitched about y.
    base = _rot_y(math.pi / 2)
    delta = _rot_z(math.pi / 2)
    composed = quat_mul(delta, base)
    # local +z of ``base`` is world +x; after a +90 deg world-z turn it is world +y.
    local_z = torch.tensor([0.0, 0.0, 1.0])
    rotated = _apply(composed, local_z)
    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def _apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """q * (0, v) * q^-1, written out so the test does not depend on the module under test."""
    point = torch.cat([torch.zeros(1), vec])
    return quat_mul(quat_mul(quat, point), quat_conjugate(quat))[1:]


# --------------------------------------------------------------------------
# AttitudeHold
# --------------------------------------------------------------------------


def _hold(**kwargs) -> AttitudeHold:
    hold = AttitudeHold(rot_scale=ROT_SCALE, **kwargs)
    hold.capture(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    return hold


def test_hold_is_silent_when_the_tool_is_on_its_reference():
    hold = _hold()
    action = hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(3))
    assert torch.allclose(action, torch.zeros(3), atol=1e-7)


def test_hold_passes_an_operator_command_through_unchanged():
    """From rest the hold must be transparent: same action the caller commanded.

    Single-axis, which is what a single-axis rotation request actually produces.
    """
    for command in [
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([-0.8, 0.0, 0.0]),
        torch.tensor([0.0, 0.35, 0.0]),
    ]:
        hold = _hold()
        action = hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), command)
        assert torch.allclose(action, command, atol=1e-5), f"{command} -> {action}"


def test_hold_bounds_a_three_axis_command_to_one_calibrated_step():
    """Deliberate deviation from an unclipped command -- see the magnitude-clip comment."""
    hold = _hold()
    command = torch.tensor([1.0, 1.0, 1.0])
    action = hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), command)
    assert action.norm().item() == pytest.approx(1.0, abs=1e-5)
    # Direction is preserved, only the magnitude is capped.
    assert torch.allclose(action / action.norm(), command / command.norm(), atol=1e-5)


def test_hold_opposes_uncommanded_drift():
    """The whole point: zero command, tool tipped 5 deg, hold pushes back."""
    hold = _hold()
    measured = _rot_y(math.radians(5.0))
    action = hold.advance(measured, torch.zeros(3))
    # Error is -5 deg about y; full deflection is 0.01 rad, so this saturates.
    assert action[1].item() == pytest.approx(-1.0)
    assert torch.allclose(action[[0, 2]], torch.zeros(2), atol=1e-6)


def test_hold_does_not_absorb_drift_into_the_reference():
    """A drifting tool must never redefine 'upright' -- otherwise the tip is permanent."""
    hold = _hold()
    reference = hold.reference.clone()
    for _ in range(50):
        hold.advance(_rot_y(math.radians(5.0)), torch.zeros(3))
    assert torch.allclose(hold.reference, reference, atol=1e-7)


def test_hold_reference_tracks_a_perfectly_following_tool():
    """With a tool that reaches its target every step, N commands integrate exactly."""
    hold = _hold()
    command = torch.tensor([0.0, 0.0, 1.0])
    measured = torch.tensor([1.0, 0.0, 0.0, 0.0])
    steps = 40
    for _ in range(steps):
        hold.advance(measured, command)
        measured = hold.reference.clone()  # ideal tracker
    total = axis_angle_from_quat(measured)
    assert total[2].item() == pytest.approx(steps * ROT_SCALE[2].item(), rel=1e-4)


@pytest.mark.parametrize("magnitude", [0.09, 0.79, 2.09, 3.12])
def test_hold_correction_keeps_the_error_axis_when_it_saturates(magnitude):
    """The bug this replaced: componentwise clipping rotates about (1,1,1), not about y.

    A componentwise clamp turns a mostly-pitch error into a full-deflection
    command on all three axes, which injects roll and yaw nobody asked for. On
    the arm that turned the hold into a divergence.

    Swept from 5 deg to 179 deg because the failure this guards against gets
    *worse* with error magnitude, and because ``axis_angle_from_quat`` changes
    branch near 180 deg -- a rule pinned only at 45 deg would not notice either.

    NOTE this only tests the error axis because ``ROT_SCALE`` is isotropic. The
    bound is applied in action space, so it preserves the demand's direction; the
    two coincide only while ``IK_SCALE[3:6]`` has equal entries. See the NOTE on
    the magnitude clip in ``attitude_hold.advance``.
    """
    hold = _hold()
    # Overwhelmingly about y, with small x and z components, scaled to the angle.
    error = torch.tensor([0.05, 0.78, 0.02])
    error = error / error.norm() * magnitude
    action = hold.advance(quat_from_axis_angle(-error), torch.zeros(3))
    assert action.norm().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(action / action.norm(), error / error.norm(), atol=1e-3)


def test_hold_lead_is_bounded_when_the_tool_cannot_follow():
    """Anti-windup: a stuck tool must not let the reference run away."""
    max_lead = 0.03
    hold = _hold(max_lead=max_lead)
    stuck = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(500):
        action = hold.advance(stuck, torch.tensor([0.0, 0.0, 1.0]))
        assert action.norm().item() <= 1.0 + 1e-6
    lead = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(stuck)))
    # One command may be applied from just below the threshold, hence the +scale.
    assert lead[2].item() <= max_lead + ROT_SCALE[2].item() + 1e-6


def test_hold_freeze_on_one_axis_does_not_block_the_others():
    """A roll error the arm cannot correct must not swallow the operator's yaw stick.

    The bound stays per axis while the other axis runs freely. It is only
    approximate -- the axis-angle components of a composed rotation are not
    independent, so winding yaw redistributes a little of the roll error and can
    unblock roll for a step. That chatter is what the ``+ one step`` allows for;
    it does not accumulate, which is what the 2000 steps here check.
    """
    hold = _hold(max_lead=0.03)
    stuck = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        hold.advance(stuck, torch.tensor([1.0, 0.0, 0.0]))
    before = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(stuck)))
    lead = before
    for _ in range(2000):
        hold.advance(stuck, torch.tensor([1.0, 0.0, 1.0]))
        lead = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(stuck)))
        assert lead[0].item() <= 0.03 + ROT_SCALE[0].item() + 1e-3  # roll still bounded
    assert lead[2].item() > before[2].item() + 0.01  # yaw got through


def test_hold_freeze_still_accepts_a_command_that_relieves_it():
    """The freeze is directional -- rotating back out of a wound-up axis must work."""
    hold = _hold(max_lead=0.03)
    stuck = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        hold.advance(stuck, torch.tensor([0.0, 0.0, 1.0]))
    wound = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(stuck)))[2].item()
    for _ in range(5):
        hold.advance(stuck, torch.tensor([0.0, 0.0, -1.0]))
    relieved = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(stuck)))[2].item()
    assert relieved < wound - 0.04


def test_hold_reference_stays_a_unit_quaternion():
    hold = _hold()
    measured = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(2000):
        hold.advance(measured, torch.tensor([0.0, 0.0, 1.0]))
        measured = hold.reference.clone()
    assert hold.reference.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_hold_gives_up_on_an_attitude_the_tool_is_being_carried_away_from():
    """Graceful degradation: a hold that cannot win must stop fighting.

    Measured on the arm: with the tool pinned by joint4's stop the servo demanded
    full pitch and got nothing, and the tool axis drifted 62.64 deg -- worse than
    the 24.21 deg that leaving the rotation channel open produces on the same
    path. Conceding puts the floor back near "no worse than the stock action
    term".
    """
    hold = _hold(give_up_steps=30)
    conceded_at = _run_until_concession(hold, limit=200)
    assert hold.gave_up == 1
    assert not hold.engaged
    # It conceded soon after crossing the threshold, not hundreds of steps later.
    assert conceded_at < 30 + math.degrees(0.25) / 0.5 + 2
    # Disengaged means no attitude feedback: the command comes back as given
    # (bounded, but this one is well inside the unit ball) however far the tool
    # has since wandered. Re-capturing alone was not enough -- the hold
    # re-engaged against the same unwinnable pose and kept pushing, drifting the
    # tool axis 60.63 deg where having no hold at all drifts 24.21 deg.
    for extra in range(2, 200):
        command = torch.tensor([0.0, 0.0, 0.4])
        action = hold.advance(_rot_y(math.radians(0.5 * (conceded_at + extra))), command)
        assert torch.allclose(action, command)
    assert hold.gave_up == 1


def test_disengaged_advance_neither_aliases_nor_exceeds_the_envelope():
    """A disengaged hold is still a controller: it owns its output and bounds it.

    Returning the caller's tensor would let the env's in-flight action alias
    whatever the caller does with that tensor next; and an unbounded
    pass-through would break the one-calibrated-step envelope the engaged path
    guarantees.
    """
    hold = _hold(give_up_steps=5)
    _run_until_concession(hold, limit=200)
    assert not hold.engaged

    command = torch.tensor([0.0, 0.0, 0.4])
    action = hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), command)
    assert action is not command
    command[2] = 99.0
    assert action[2].item() == pytest.approx(0.4)

    huge = hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([5.0, 0.0, 0.0]))
    assert huge.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_capture_re_arms_a_hold_that_gave_up():
    hold = _hold(give_up_steps=5)
    _run_until_concession(hold, limit=200)
    assert not hold.engaged
    hold.capture(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert hold.engaged
    action = hold.advance(_rot_y(math.radians(5.0)), torch.zeros(3))
    assert action[1].item() == pytest.approx(-1.0)


def _run_until_concession(hold: AttitudeHold, limit: int) -> int:
    """Feed a tool that keeps tipping away whatever it is told. Returns the step index."""
    for step in range(limit):
        hold.advance(_rot_y(math.radians(0.5 * (step + 1))), torch.zeros(3))
        if hold.gave_up:
            return step
    raise AssertionError(f"the hold never conceded in {limit} steps")


def test_hold_does_not_give_up_while_it_is_winning():
    """A big but recoverable knock must be recovered, not conceded.

    Full deflection corrects 0.573 deg per step, so a 25 deg disturbance takes
    ~44 steps to undo -- three times ``give_up_steps``. What keeps the hold at it
    is that the error shrinks every step; only a stalled error concedes.
    """
    hold = _hold()
    tipped = 25.0
    for _ in range(200):
        hold.advance(_rot_y(math.radians(tipped)), torch.zeros(3))
        tipped = max(0.0, tipped - math.degrees(ROT_SCALE[1].item()))
    assert hold.gave_up == 0
    assert hold.engaged


def test_hold_does_not_give_up_during_ordinary_commanded_rotation():
    """The measured worst in-envelope error is 2.29 deg; the threshold is 14.3 deg."""
    hold = _hold()
    measured = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(3000):
        hold.advance(measured, torch.tensor([0.0, 0.0, 1.0]))
        # An arm that tracks 0.8 of command, i.e. always lagging its reference.
        lag = axis_angle_from_quat(quat_mul(hold.reference, quat_conjugate(measured)))
        measured = quat_mul(quat_from_axis_angle(0.8 * lag), measured)
        measured = measured / measured.norm()
    assert hold.gave_up == 0


def test_give_up_disabled_by_an_infinite_threshold():
    """What a scripted caller passes when it wants the servo to keep trying.

    ``scripts/scripted_grasp_check.py`` does this: it starts each phase 13.6-25.0
    deg off target, well above the default threshold, and manages its own step
    budgets and stall detection.
    """
    hold = _hold(give_up_error=float("inf"))
    for step in range(500):
        hold.advance(_rot_y(math.radians(0.5 * (step + 1))), torch.zeros(3))
    assert hold.gave_up == 0
    assert hold.engaged


def test_rejects_a_give_up_threshold_inside_the_normal_error_band():
    with pytest.raises(ValueError):
        AttitudeHold(rot_scale=ROT_SCALE, max_lead=0.03, give_up_error=0.05)


# --------------------------------------------------------------------------
# Batching. One instance holds N attitudes, and the give-up state is per env --
# a scalar give-up would let one env's unwinnable pose silently switch the hold
# off for its healthy neighbours, which is invisible from the outside because a
# disengaged hold just stops correcting.
# --------------------------------------------------------------------------


def test_batched_hold_servos_each_env_independently():
    hold = AttitudeHold(rot_scale=ROT_SCALE)
    hold.capture(torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 4).clone())
    measured = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),  # on reference
            _rot_y(math.radians(5.0)),  # tipped in pitch
            _rot_z(math.radians(-5.0)),  # tipped in yaw
        ]
    )
    action = hold.advance(measured, torch.zeros(3, 3))
    assert action.shape == (3, 3)
    assert torch.allclose(action[0], torch.zeros(3), atol=1e-4)
    assert action[1, 1].item() == pytest.approx(-1.0)
    assert action[2, 2].item() == pytest.approx(1.0)


def test_batched_give_up_disengages_only_the_env_that_is_losing():
    hold = AttitudeHold(rot_scale=ROT_SCALE, give_up_steps=15)
    hold.capture(torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 4).clone())
    upright = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for step in range(200):
        # Env 1 is dragged away; envs 0 and 2 sit on their reference.
        measured = torch.stack([upright, _rot_y(math.radians(0.5 * (step + 1))), upright])
        hold.advance(measured, torch.zeros(3, 3))
        if hold.gave_up:
            break
    assert hold.gave_up == 1
    assert hold.engaged.tolist() == [True, False, True]

    # The healthy envs must still be holding: tip them and they must push back.
    measured = torch.stack([_rot_y(math.radians(5.0)), upright, _rot_y(math.radians(5.0))])
    action = hold.advance(measured, torch.zeros(3, 3))
    assert action[0, 1].item() == pytest.approx(-1.0)
    assert action[2, 1].item() == pytest.approx(-1.0)
    assert torch.allclose(action[1], torch.zeros(3), atol=1e-6)


def test_batched_capture_re_arms_every_env():
    hold = AttitudeHold(rot_scale=ROT_SCALE, give_up_steps=5)
    hold.capture(torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 4).clone())
    upright = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for step in range(200):
        measured = torch.stack([upright, _rot_y(math.radians(0.5 * (step + 1)))])
        hold.advance(measured, torch.zeros(2, 3))
        if hold.gave_up:
            break
    assert hold.engaged.tolist() == [True, False]
    hold.capture(upright.expand(2, 4).clone())
    assert hold.engaged.tolist() == [True, True]


def test_capture_moves_the_reference_to_the_measured_attitude():
    hold = _hold()
    tipped = _rot_y(math.radians(30.0))
    hold.capture(tipped)
    action = hold.advance(tipped, torch.zeros(3))
    # Not exactly zero: dividing by rot_scale amplifies float32 quaternion
    # round-off by 1/0.01 = 100x. 1e-4 of full deflection is 1e-6 rad/step --
    # four orders below the 0.01 rad/step the operator commands, and two orders
    # below the ~1e-4 rad of solver noise the arm sits in anyway.
    assert torch.all(action.abs() < 1e-4)


def test_capture_is_required_before_use():
    hold = AttitudeHold(rot_scale=ROT_SCALE)
    with pytest.raises(RuntimeError):
        hold.advance(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(3))


def test_rejects_a_lead_bound_that_would_throttle_the_operator():
    """max_lead below one full-deflection step caps the sustained rotation rate."""
    with pytest.raises(ValueError):
        AttitudeHold(rot_scale=ROT_SCALE, max_lead=0.005)


def test_rejects_a_non_positive_rotation_scale():
    with pytest.raises(ValueError):
        AttitudeHold(rot_scale=torch.tensor([0.01, 0.0, 0.01]))
