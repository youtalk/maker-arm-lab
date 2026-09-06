"""CPU tests for core.global_ik -- no isaaclab, no GPU.

Task D3 found six defects and every one of them needed a GPU round to see. The solver is
pure kinematics, so its correctness, its limit handling and its continuity constraint can all
be pinned here instead. That is the point of this file.
"""

import itertools
import math

import pytest
import torch

from maker_arm_lab.assets.urdf_prep import ARM_JOINT_NAMES
from maker_arm_lab.core.global_ik import (
    ARM_DOF,
    JAW_FACE_FAR,
    JAW_TIP_REACH,
    MAX_DESCEND_TILT,
    MAX_WAYPOINT_STEP,
    MIN_DESCEND_TILT,
    GlobalIK,
    GraspTarget,
    JointTrim,
    descend_pick_offset,
    descend_tilt_ceiling,
    descend_tilt_ladder,
    fold_to_half_turn,
    tool_axis_world,
    wrap_to_pi,
)
from maker_arm_lab.robots.frames import GRASP_TILT_RAD, HOME_JOINT_POS, TOOL_AXIS_BODY


@pytest.fixture(scope="module")
def ik() -> GlobalIK:
    return GlobalIK()


@pytest.fixture(scope="module")
def home() -> torch.Tensor:
    return torch.tensor([HOME_JOINT_POS[n] for n in ARM_JOINT_NAMES], dtype=torch.float64)


def test_fk_matches_the_pinned_isaac_reading_at_home(ik, home):
    """The FK must reproduce what the simulator actually puts on the arm.

    (0.2749, 0.0371, 0.1646) at 23.85 deg off world -z is the live Isaac reading at
    HOME_JOINT_POS, recorded in frames.HOME_JOINT_POS's own comment and re-measured in Task
    D3. Every number this solver produces rests on this agreement, and until now it was
    only ever checked by a throwaway script in /tmp.
    """
    pose = ik.pose_of(home)
    assert pose.position[0].item() == pytest.approx(0.2749, abs=5e-4)
    assert pose.position[1].item() == pytest.approx(0.0371, abs=5e-4)
    assert pose.position[2].item() == pytest.approx(0.1646, abs=5e-4)
    assert math.degrees(pose.tilt) == pytest.approx(23.85, abs=0.1)


def test_home_margin_matches_the_pinned_value(ik, home):
    """HOME_JOINT_POS documents a 0.30 rad minimum margin on link_005_joint."""
    margins = ik.margins(home)
    assert float(margins.min()) == pytest.approx(0.30, abs=1e-6)
    assert int(margins.argmin()) == ARM_JOINT_NAMES.index("link_005_joint")


def test_fk_batch_agrees_with_single_fk(ik, home):
    q = torch.stack([home, home * 0.5, torch.zeros(ARM_DOF, dtype=torch.float64)])
    pos, rot = ik.fk_batch(q)
    for i in range(q.shape[0]):
        one_pos, one_rot = ik.fk(q[i])
        assert torch.allclose(pos[i], one_pos, atol=1e-12)
        assert torch.allclose(rot[i], one_rot, atol=1e-12)


def _target_from(ik: GlobalIK, q: torch.Tensor) -> GraspTarget:
    pose = ik.pose_of(q)
    return GraspTarget(
        position=tuple(float(v) for v in pose.position),
        tilt=pose.tilt,
        jaw_heading=pose.jaw_heading,
        lean_azimuth=pose.lean_azimuth,
    )


def test_solve_recovers_a_pose_the_arm_is_already_in(ik, home):
    """Round trip: read a pose off a joint vector, solve for it, get that pose back.

    Not necessarily the SAME joint vector -- the task is six constraints on six joints, so
    the solution set is discrete and another branch is a legitimate answer. What must hold
    is that the returned joints reproduce the commanded pose.
    """
    target = _target_from(ik, home)
    solution = ik.solve(target, seed=home)
    assert solution is not None
    pose = ik.pose_of(solution.q)
    assert torch.allclose(
        pose.position, torch.tensor(target.position, dtype=torch.float64), atol=1e-5
    )
    assert pose.tilt == pytest.approx(target.tilt, abs=1e-4)
    assert float(fold_to_half_turn(pose.jaw_heading - target.jaw_heading)) == pytest.approx(
        0.0, abs=1e-4
    )
    assert float(wrap_to_pi(pose.lean_azimuth - target.lean_azimuth)) == pytest.approx(
        0.0, abs=1e-4
    )


def test_solve_reports_residuals_that_match_its_own_solution(ik, home):
    target = _target_from(ik, home)
    solution = ik.solve(target, seed=home)
    assert solution is not None
    assert solution.position_error < 1e-5
    assert solution.tilt_error < math.radians(0.05)
    assert solution.heading_error < math.radians(0.05)
    assert solution.azimuth_error < math.radians(0.05)


def test_solutions_never_leave_the_joint_box(ik):
    """Every returned joint vector is inside the URDF limits, margin >= 0 by construction."""
    target = GraspTarget(
        position=(0.21, 0.0, 0.1248), tilt=GRASP_TILT_RAD, jaw_heading=0.4, lean_azimuth=0.0
    )
    solution = ik.solve(target)
    assert solution is not None
    assert torch.all(solution.q >= ik.lower - 1e-12)
    assert torch.all(solution.q <= ik.upper + 1e-12)
    assert solution.margin >= 0.0


def test_solve_picks_the_branch_furthest_from_the_stops(ik):
    """Multi-start is branch SELECTION here, not optimisation -- it must actually select.

    Solving with only the caller's seed and no restarts can land on a tight branch; the
    default restart spread must do at least as well, never worse.
    """
    target = GraspTarget(
        position=(0.21, 0.0, 0.1248), tilt=GRASP_TILT_RAD, jaw_heading=0.0, lean_azimuth=0.0
    )
    seeded_only = ik.solve(target, seed=torch.zeros(ARM_DOF, dtype=torch.float64), restarts=0)
    with_restarts = ik.solve(target)
    assert with_restarts is not None
    if seeded_only is not None:
        assert with_restarts.margin >= seeded_only.margin - 1e-12


def test_unreachable_target_returns_none(ik):
    """A metre away is outside the arm entirely. None, not a nearest-point fudge."""
    target = GraspTarget(
        position=(1.5, 0.0, 0.5), tilt=GRASP_TILT_RAD, jaw_heading=0.0, lean_azimuth=0.0
    )
    assert ik.solve(target) is None


def test_solve_is_deterministic(ik):
    """Same task, same joints -- an oracle that picked a different branch per run would
    make datasets irreproducible."""
    target = GraspTarget(
        position=(0.24, 0.05, 0.1248), tilt=GRASP_TILT_RAD, jaw_heading=0.3, lean_azimuth=0.0
    )
    first, second = ik.solve(target), ik.solve(target)
    assert first is not None and second is not None
    assert torch.allclose(first.q, second.q, atol=0.0, rtol=0.0)


def _descend_targets(ik: GlobalIK, waypoints: int = 12) -> list[GraspTarget]:
    start = torch.tensor([0.22, 0.02, 0.1248], dtype=torch.float64)
    end = torch.tensor([0.22, 0.02, 0.0341], dtype=torch.float64)
    return [
        GraspTarget(
            position=tuple(float(v) for v in start + (end - start) * t),
            tilt=GRASP_TILT_RAD,
            jaw_heading=0.2,
            lean_azimuth=0.0,
        )
        for t in torch.linspace(0.0, 1.0, waypoints, dtype=torch.float64)
    ]


def test_solve_path_is_continuous_and_within_limits(ik):
    """A straight Cartesian descend must come back as a smooth joint path."""
    targets = _descend_targets(ik)
    first = ik.solve(targets[0])
    assert first is not None
    path = ik.solve_path(targets, seed=first.q)
    assert path is not None and len(path) == len(targets)
    for previous, current in itertools.pairwise(path):
        assert float(torch.max(torch.abs(current.q - previous.q))) <= MAX_WAYPOINT_STEP + 1e-9
    assert all(step.margin >= 0.0 for step in path)


def test_solve_path_step_cap_is_enforced_not_merely_reported(ik):
    """An impossibly tight cap must fail the path, not quietly exceed itself.

    This is the branch-flip guard. If the cap were advisory, a solver jumping IK branches
    mid-approach would return a path whose steps simply exceed it.
    """
    targets = _descend_targets(ik)
    first = ik.solve(targets[0])
    assert first is not None
    assert ik.solve_path(targets, seed=first.q, max_step=1e-6) is None


def test_solve_path_returns_none_rather_than_a_reachable_prefix(ik):
    """Half a descend is not a descend."""
    targets = _descend_targets(ik)[:4] + [
        GraspTarget(
            position=(1.5, 0.0, 0.5), tilt=GRASP_TILT_RAD, jaw_heading=0.2, lean_azimuth=0.0
        )
    ]
    first = ik.solve(targets[0])
    assert first is not None
    assert ik.solve_path(targets, seed=first.q) is None


def test_continuity_rejection_reports_the_step_it_rejected(ik, home):
    """`step` must be measured against q_prev so callers can see how close they were."""
    target = _target_from(ik, home)
    solution = ik.solve(target, seed=home, q_prev=home)
    assert solution is not None
    assert solution.step == pytest.approx(float(torch.max(torch.abs(solution.q - home))), abs=1e-12)


def test_wrap_and_fold_conventions():
    # The range is [-pi, pi), so an exact half turn lands on -pi, not +pi.
    assert wrap_to_pi(3.0 * math.pi) == pytest.approx(-math.pi, abs=1e-12)
    assert wrap_to_pi(-3.0 * math.pi) == pytest.approx(-math.pi, abs=1e-12)
    assert wrap_to_pi(0.4) == pytest.approx(0.4, abs=1e-12)
    assert wrap_to_pi(2.0 * math.pi + 0.4) == pytest.approx(0.4, abs=1e-12)
    # A jaw is symmetric under a half turn: pi of heading error is zero error.
    assert fold_to_half_turn(math.pi) == pytest.approx(0.0, abs=1e-12)
    assert fold_to_half_turn(0.1) == pytest.approx(0.1, abs=1e-12)
    assert fold_to_half_turn(math.pi - 0.1) == pytest.approx(-0.1, abs=1e-12)
    # Exactly a quarter turn is the fold's antipode: +pi/2 and -pi/2 are the same error and
    # the sign there is arbitrary. Only the magnitude is meaningful.
    assert abs(fold_to_half_turn(math.pi / 2)) == pytest.approx(math.pi / 2, abs=1e-12)


def test_jaw_heading_folds_by_half_a_turn_in_the_solver(ik, home):
    """Commanding heading + pi must be satisfied by the same poses as commanding heading."""
    base = _target_from(ik, home)
    flipped = GraspTarget(
        position=base.position,
        tilt=base.tilt,
        jaw_heading=base.jaw_heading + math.pi,
        lean_azimuth=base.lean_azimuth,
    )
    solution = ik.solve(flipped, seed=home)
    assert solution is not None
    assert solution.heading_error < math.radians(0.05)


class TestJointTrim:
    def test_starts_neutral(self):
        trim = JointTrim()
        desired = torch.ones(ARM_DOF, dtype=torch.float64)
        assert torch.allclose(trim.command(desired), desired)

    def test_converges_to_the_droop_against_a_responding_plant(self):
        """Against a plant that responds to the trim, the trim converges to the droop.

        The plant matters. Model it as `measured = desired + trim - droop`, which is what a
        saturating joint does: the commanded bias moves it, minus a load-induced offset.
        Then the error is `droop - trim` and the recursion really is
        trim <- (1 - g) trim + g droop, converging to droop from below without overshoot.

        Feeding a CONSTANT error instead -- a measured value that ignores the trim -- is an
        open loop and winds up to the clamp no matter how small the gain is. That was this
        test's first version, and it failed, which is the distinction worth pinning: the
        gain is only safe because the plant closes the loop.
        """
        trim = JointTrim(gain=0.3)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        droop = torch.full((ARM_DOF,), 0.1, dtype=torch.float64)
        for _ in range(200):
            trim.update(desired, trim.command(desired) - droop)
        assert torch.allclose(trim.value, droop, atol=1e-6)
        assert float(trim.value.max()) <= float(droop.max()) + 1e-9

    def test_an_open_loop_error_winds_up_to_the_clamp(self):
        """The clamp is the only thing bounding a trim whose plant never responds.

        Worth a test of its own: if a joint is so saturated that the command cannot move it
        at all, the integrator has nothing to converge to, and the limit is what stops it
        from commanding an arbitrary joint angle.
        """
        trim = JointTrim(gain=0.3, limit=0.2)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        for _ in range(200):
            trim.update(desired, desired - 0.1)
        assert float(trim.value.max()) == pytest.approx(0.2)

    def test_is_clamped(self):
        trim = JointTrim(gain=1.0, limit=0.05)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        for _ in range(50):
            trim.update(desired, desired - 10.0)
        assert float(trim.value.max()) == pytest.approx(0.05)

    def test_reset_clears_it(self):
        trim = JointTrim(gain=0.5)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        trim.update(desired, desired - 1.0)
        assert float(trim.value.abs().max()) > 0.0
        trim.reset()
        assert float(trim.value.abs().max()) == 0.0

    def test_rejects_an_unstable_gain(self):
        """g >= 2 diverges. Task D3 measured that even g = 0.25 winds up on this arm."""
        with pytest.raises(ValueError):
            JointTrim(gain=2.0)


def test_tool_axis_world_matches_the_fk(ik, home):
    """The shared attitude helper must agree with the FK it is a shortcut for."""
    pose = ik.pose_of(home)
    axis = tool_axis_world(pose.tilt, pose.lean_azimuth)
    _, rot = ik.fk(home)
    measured = rot @ torch.tensor(TOOL_AXIS_BODY, dtype=torch.float64)
    assert torch.allclose(axis, measured, atol=1e-12)


def test_tool_axis_world_is_a_unit_vector_pointing_down(ik):
    axis = tool_axis_world(GRASP_TILT_RAD, 0.0)
    assert float(torch.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-12)
    assert float(axis[2]) < 0.0
    # Zero azimuth is the shipping convention: lean toward world +x.
    assert float(axis[0]) > 0.0
    assert float(axis[1]) == pytest.approx(0.0, abs=1e-12)


class TestDescendGeometry:
    """The jaw plates decide what descend is possible; these pin that arithmetic."""

    HALF = 0.0048  # half a 9.6 mm brick

    def test_tilt_ceiling_is_where_the_plate_corner_eats_the_clearance(self):
        """Above the ceiling no offset both clears the desk and keeps the brick inside."""
        ceiling = descend_tilt_ceiling(self.HALF)
        assert math.degrees(ceiling) == pytest.approx(4.40, abs=0.02)
        assert descend_pick_offset(ceiling + math.radians(0.5), self.HALF) is None

    def test_offset_balances_desk_clearance_against_tip_margin(self):
        offset, margin = descend_pick_offset(MIN_DESCEND_TILT, self.HALF)
        assert offset < 0.0, "the offset is negative in PICK_OFFSET_MEASURED's convention"
        # The brick must sit inside the plate tip...
        assert -offset < JAW_TIP_REACH
        assert margin == pytest.approx(JAW_TIP_REACH + offset, abs=1e-12)
        # ...and the plates' lowest corner must clear the desk by the same margin.
        lowest = (
            -offset * math.cos(MIN_DESCEND_TILT)
            - JAW_TIP_REACH * math.cos(MIN_DESCEND_TILT)
            - JAW_FACE_FAR * math.sin(MIN_DESCEND_TILT)
            + self.HALF
        )
        # The balance point sets the desk clearance EQUAL to the tip margin, so the lowest
        # corner sits exactly `margin` above the desk -- no cosine, that is what balancing
        # means.
        assert lowest == pytest.approx(margin, abs=1e-9)

    def test_margin_shrinks_as_the_tilt_grows(self):
        """More tilt is strictly worse for the plates -- that is the whole finding."""
        margins = [descend_pick_offset(math.radians(d), self.HALF)[1] for d in (0.5, 1.0, 2.0, 3.0)]
        assert margins == sorted(margins, reverse=True)
        assert all(m > 0.0 for m in margins)

    def test_the_ladder_starts_near_level_but_never_at_level(self):
        """Exactly vertical makes the lean azimuth undefined; the ladder must avoid it."""
        ladder = descend_tilt_ladder(self.HALF)
        assert ladder[0] == MIN_DESCEND_TILT > 0.0
        assert ladder == sorted(ladder)
        assert all(descend_pick_offset(t, self.HALF) is not None for t in ladder)
        assert max(ladder) <= descend_tilt_ceiling(self.HALF)

    def test_a_thin_brick_collapses_the_ladder_to_one_rung(self):
        """It is a THIN brick that leaves no room to search, not a tall one.

        The brick's own height is what buys the tilt: the plates may dip by up to
        `brick_half_height` before they reach the desk. A vanishingly thin brick has no
        budget at all, so the ceiling collapses onto the floor and only the near-level rung
        survives. A tall brick has the opposite effect and the ladder runs to
        MAX_DESCEND_TILT -- worth pinning too, since I had this backwards first time.
        """
        assert descend_tilt_ladder(1e-6) == [MIN_DESCEND_TILT]
        assert max(descend_tilt_ladder(1.0)) == pytest.approx(MAX_DESCEND_TILT)

    def test_the_azimuth_constraint_is_singular_at_exactly_zero_tilt(self, ik):
        """Why MIN_DESCEND_TILT is not 0: at vertical the azimuth row carries no information.

        Two different commanded azimuths at zero tilt must produce the same tool axis, which
        is what makes the constraint degenerate there -- and non-degenerate just off it.
        """
        straight_down = tool_axis_world(0.0, 0.0)
        assert torch.allclose(straight_down, tool_axis_world(0.0, 2.0), atol=1e-12)
        assert not torch.allclose(
            tool_axis_world(MIN_DESCEND_TILT, 0.0),
            tool_axis_world(MIN_DESCEND_TILT, 2.0),
            atol=1e-6,
        )

    def test_a_saturated_joint_does_not_wind_the_trim(self):
        """Anti-windup: no integration on a joint whose actuator is already refusing.

        The failure this prevents, measured on the arm: a trim at its clamp on a joint with
        kp 150 commands 52.5 Nm into a 17 Nm actuator. Clamping alone does not prevent it --
        the trim still runs to the bound while the joint sits immobile.
        """
        trim = JointTrim(gain=0.5)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        saturated = torch.zeros(ARM_DOF, dtype=torch.bool)
        saturated[1] = True
        for _ in range(50):
            trim.update(desired, desired - 0.1, saturated=saturated)
        assert float(trim.value[1]) == 0.0, "the saturated joint must not accumulate"
        assert float(trim.value[0]) > 0.0, "unsaturated joints must still integrate"

    def test_saturation_freezes_rather_than_resets_the_trim(self):
        """A joint that saturates keeps the trim it had earned, it does not lose it."""
        trim = JointTrim(gain=0.5)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        none_saturated = torch.zeros(ARM_DOF, dtype=torch.bool)
        for _ in range(10):
            trim.update(desired, desired - 0.1, saturated=none_saturated)
        earned = trim.value.clone()
        assert float(earned[1]) > 0.0
        all_saturated = torch.ones(ARM_DOF, dtype=torch.bool)
        for _ in range(50):
            trim.update(desired, desired - 0.1, saturated=all_saturated)
        assert torch.allclose(trim.value, earned)

    def test_the_mask_is_optional_and_omitting_it_keeps_the_old_behaviour(self):
        trim = JointTrim(gain=0.5, limit=0.2)
        desired = torch.zeros(ARM_DOF, dtype=torch.float64)
        for _ in range(50):
            trim.update(desired, desired - 0.1)
        assert float(trim.value.max()) == pytest.approx(0.2)
