"""CPU tests for the oracle phase state machine (no isaaclab)."""

import math
import random

import pytest
import torch

from maker_arm_lab.core import global_ik, layout, oracle


def _snapshot(brick_xy=(0.20, 0.05), stowed=(), brick_yaw=0.0):
    """One red brick on the desk (or teleported into its bin if in ``stowed``)."""
    bl = layout.DEFAULT_BIN_LAYOUT
    pos = torch.zeros(1, 3)
    if 0 in stowed:
        pos[0] = layout.bin_interior_centers(bl)[0]  # red bin center
    else:
        pos[0] = torch.tensor([brick_xy[0], brick_xy[1], layout.BRICK_SIZE[2] / 2])
    return oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.full((1,), brick_yaw),
        brick_lin_vel=torch.zeros(1, 3),
        pick_point=torch.tensor([0.28, 0.0, 0.20]),
    )


def _draw():
    return oracle.sample_episode_draw(random.Random(0), oracle.OracleNoise())


def _oracle():
    return oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, num_bricks=1, draw=_draw())


def _run_phase(orc, scene, outcome="converged", pinch_ok=None, lifted=None):
    goal = orc.next_goal(scene)
    orc.report(oracle.PhaseResult(outcome=outcome, pinch_ok=pinch_ok, lifted=lifted), scene)
    return goal


def test_happy_path_visits_all_phases_in_order():
    orc = _oracle()
    scene = _snapshot()
    seen = [
        _run_phase(orc, scene).phase,  # SETTLE
        _run_phase(orc, scene).phase,  # HOVER
        _run_phase(orc, scene, outcome="stalled").phase,  # DESCEND (stall = desk hit, ok)
        _run_phase(orc, scene).phase,  # CLOSE
        _run_phase(orc, scene, pinch_ok=True, lifted=True).phase,  # LIFT (verified here)
        _run_phase(orc, scene).phase,  # RAISE
        _run_phase(orc, scene).phase,  # EXTEND
        _run_phase(orc, scene).phase,  # RELEASE
    ]
    assert seen == [
        oracle.Phase.SETTLE,
        oracle.Phase.HOVER,
        oracle.Phase.DESCEND,
        oracle.Phase.CLOSE,
        oracle.Phase.LIFT,
        oracle.Phase.RAISE,
        oracle.Phase.EXTEND,
        oracle.Phase.RELEASE,
    ]
    # After releasing the only brick and retreating, a snapshot showing it
    # stowed must end the episode.
    _run_phase(orc, _snapshot(stowed=(0,)))  # RETREAT
    assert orc.next_goal(_snapshot(stowed=(0,))) is None
    assert orc.phase == oracle.Phase.DONE


def test_failed_pinch_retries_from_hover_then_abandons():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    for attempt in range(oracle.MAX_GRASP_RETRIES + 1):  # 3 grasp attempts total
        assert _run_phase(orc, scene).phase == oracle.Phase.HOVER
        _run_phase(orc, scene, outcome="stalled")  # DESCEND
        _run_phase(orc, scene)  # CLOSE
        goal = _run_phase(orc, scene, pinch_ok=False, lifted=False)  # LIFT, no pinch
        assert goal.phase == oracle.Phase.LIFT
        if attempt < oracle.MAX_GRASP_RETRIES:
            assert _run_phase(orc, scene).phase == oracle.Phase.REOPEN
    assert orc.next_goal(scene) is None
    assert orc.abandoned
    assert orc.phase == oracle.Phase.ABANDON


def test_a_lift_that_rose_advances_even_when_the_pinch_factors_refuse_it():
    """MS6 rework section 4: 8 of 26 real holds on heap 0 failed a pinch factor
    while the part rose 90 mm or more in the jaws."""
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=False, lifted=True)  # LIFT
    assert orc.phase is oracle.Phase.RAISE
    assert orc.retries_used == 0


def test_a_lift_that_did_not_rise_still_retries():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, outcome="budget", pinch_ok=True, lifted=False)  # LIFT
    assert orc.phase is oracle.Phase.REOPEN
    assert orc.retries_used == 1


def test_release_targets_track_the_permuted_bin():
    bl = layout.BinLayout(bin_x=0.45, bin_ys=(-0.20, -0.06, 0.08, 0.20), slot_colors=(3, 1, 2, 0))
    orc = oracle.EnvOracle(bl, num_bricks=1, draw=_draw())
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT
    _run_phase(orc, scene)  # RAISE
    extend = orc.next_goal(scene)  # EXTEND
    assert extend.phase == oracle.Phase.EXTEND
    # Brick 0 is red (color 0); red occupies slot 3 -> y = +0.20, x = bin_x - setback.
    assert extend.target_pos[0].item() == pytest.approx(0.45 - oracle.RELEASE_X_SETBACK)
    assert extend.target_pos[1].item() == pytest.approx(0.20)
    assert extend.target_pos[2].item() == pytest.approx(oracle.RELEASE_Z)
    assert extend.tilt_rad == pytest.approx(oracle.RELEASE_TILT_RAD)


def test_release_height_clears_the_rim_at_the_release_tilt():
    """RELEASE_Z's floor, re-derived for THIS arm by Task D4.

    Replaces the carried-over "no collision-free open-gripper pose below z ~= 0.24"
    (OpenYAM's M1 Task 9), which bounded OpenYAM's gripper against OpenYAM's bins.

    Written as the derivation rather than as a number ON PURPOSE: the floor is a function of
    RELEASE_TILT_RAD, because tilting the tool sinks the jaw plate's far corner
    JAW_FACE_FAR * sin(tilt) below the tool tip -- the same geometry
    global_ik.descend_tilt_ceiling applies against the desk, applied here against the bin
    rim. D4 raised the tilt from 0.70 to 1.40 rad, which raised this floor from 0.235 to
    0.257 m, so an edit of EITHER constant is now checked against the other.
    """
    bin_rim_z = layout._BIN_FLOOR_TOP_Z + layout.BIN_INTERIOR[2]
    plate_dip = global_ik.JAW_FACE_FAR * math.sin(oracle.RELEASE_TILT_RAD)
    assert oracle.RELEASE_Z > bin_rim_z + plate_dip


def test_raise_targets_clearance_height_not_release_height():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT -> RAISE
    assert orc.phase == oracle.Phase.RAISE
    raise_goal = orc.next_goal(scene)
    assert raise_goal.phase == oracle.Phase.RAISE
    # RAISE only needs to clear the bin rim before EXTEND starts translating
    # over the bins -- it must NOT target RELEASE_Z (the reach-ceiling height
    # that, combined with tilt=0.0, OpenYAM measured pinning joint4 before the
    # climb finished; see the RAISE goal's comment in core/oracle.py).
    assert raise_goal.target_pos[2].item() == pytest.approx(oracle.RAISE_CLEARANCE_Z)
    assert raise_goal.target_pos[2].item() != pytest.approx(oracle.RELEASE_Z)
    # The climb is tilted from the start, not level: that is what gave the wrist the margin
    # the release pose needs (OpenYAM measured). But it is NOT the release tilt, and the gap
    # is load-bearing rather than cosmetic -- Task D5 measured that leaving RAISE at
    # RELEASE_TILT_RAD puts the arm in a pose from which no continuous EXTEND to ANY bin
    # exists, driven as a 300-step EXTEND ending 312 mm short at both outer bins. Both bounds
    # are asserted, so neither reverting to the release tilt nor flattening the climb to level
    # can pass.
    assert raise_goal.tilt_rad == pytest.approx(oracle.RAISE_TILT_RAD)
    assert 0.0 < oracle.RAISE_TILT_RAD < oracle.RELEASE_TILT_RAD
    assert raise_goal.stop_pos == pytest.approx(oracle.RAISE_STOP_POS)
    assert raise_goal.max_steps == oracle.RAISE_MAX_STEPS


def test_raise_clearance_height_clears_the_bin_rim():
    # RAISE_CLEARANCE_Z must sit above the bin rim (floor + wall height) --
    # otherwise EXTEND would start translating laterally over the bins while
    # the brick is still low enough to clip a wall.
    bin_rim_z = layout._BIN_FLOOR_TOP_Z + layout.BIN_INTERIOR[2]
    assert oracle.RAISE_CLEARANCE_Z > bin_rim_z


def test_raise_advances_to_extend_on_budget_outcome():
    # report()'s RAISE branch takes no outcome branch (unlike LIFT) -- it
    # always advances to EXTEND -- but RAISE's own short budget
    # (RAISE_MAX_STEPS) means "budget" is the ONLY outcome _phase_result ever
    # produces for it in production (RAISE never converges within its budget
    # by design; see the RAISE goal's comment in core/oracle.py). Pinned
    # explicitly since the happy-path test above only exercises the default
    # "converged" outcome for every phase.
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT -> RAISE
    assert orc.phase == oracle.Phase.RAISE
    raise_goal = _run_phase(orc, scene, outcome="budget")  # RAISE (budget) -> EXTEND
    assert raise_goal.phase == oracle.Phase.RAISE
    assert orc.phase == oracle.Phase.EXTEND


def test_retreat_carries_the_release_tilt():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT -> RAISE
    _run_phase(orc, scene)  # RAISE -> EXTEND
    _run_phase(orc, scene)  # EXTEND -> RELEASE
    _run_phase(orc, scene)  # RELEASE -> RETREAT
    retreat_goal = orc.next_goal(scene)
    assert retreat_goal.phase == oracle.Phase.RETREAT
    # tilt_rad=RELEASE_TILT_RAD, not 0.0 (level): releveling straight back to
    # level, combined with the big position change to RETREAT_POINT, was
    # OpenYAM measured exhausting the same joints RAISE's tilt fix above
    # addresses; see RETREAT's comment in core/oracle.py.
    assert retreat_goal.tilt_rad == pytest.approx(oracle.RELEASE_TILT_RAD)


def test_every_pick_phase_carries_the_pick_tilt():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE -> HOVER
    orc.next_goal(scene)  # HOVER's entry selects the target brick
    for phase in (
        oracle.Phase.HOVER,
        oracle.Phase.DESCEND,
        oracle.Phase.CLOSE,
        oracle.Phase.LIFT,
    ):
        orc.phase = phase
        goal = orc.next_goal(scene)
        assert goal.tilt_rad == oracle.PICK_TILT_RAD


def test_every_pick_phase_carries_a_raised_pick_tilt(monkeypatch):
    """PICK_TILT_RAD == 0.0 makes the assertion above unable to fail; this pins the wiring."""
    monkeypatch.setattr(oracle, "PICK_TILT_RAD", 0.4)
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE -> HOVER
    orc.next_goal(scene)  # HOVER's entry selects the target brick
    for phase in (
        oracle.Phase.HOVER,
        oracle.Phase.DESCEND,
        oracle.Phase.CLOSE,
        oracle.Phase.LIFT,
    ):
        orc.phase = phase
        goal = orc.next_goal(scene)
        assert goal.tilt_rad == 0.4


def test_unreachable_brick_is_skipped_and_reachable_one_sorted():
    bl = layout.DEFAULT_BIN_LAYOUT
    pos = torch.tensor(
        [
            [layout.top_down_x_limit(bl) + 0.02, 0.0, layout.BRICK_SIZE[2] / 2],  # beyond limit
            [0.20, -0.05, layout.BRICK_SIZE[2] / 2],
        ]
    )
    scene = oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.zeros(2),
        brick_lin_vel=torch.zeros(2, 3),
        pick_point=torch.tensor([0.28, 0.0, 0.20]),
    )
    orc = oracle.EnvOracle(bl, num_bricks=2, draw=_draw())
    _run_phase(orc, scene)  # SETTLE
    hover = orc.next_goal(scene)
    # Target must be brick 1 (the reachable one): hover directly above it.
    assert hover.target_pos[0].item() == pytest.approx(0.20, abs=0.03)
    assert hover.target_pos[1].item() == pytest.approx(-0.05, abs=0.02)


def test_hover_goal_carries_the_episode_draw_noise():
    noise = oracle.OracleNoise()
    d1 = oracle.sample_episode_draw(random.Random(1), noise)
    d2 = oracle.sample_episode_draw(random.Random(2), noise)
    assert d1 != d2  # different seeds, different draws
    assert noise.hover_height[0] <= d1.hover_height <= noise.hover_height[1]
    assert abs(d1.lateral_dx) <= noise.lateral_offset
    assert abs(d1.yaw_offset) <= noise.yaw_offset
    assert noise.gain_scale[0] <= d1.gain_scale <= noise.gain_scale[1]


def test_watchdog_trips_near_limits_and_on_speed():
    limits = torch.tensor(
        [
            [-2.61799, 3.13],
            [0.0, 3.65],
            [0.0, 3.13],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
            [-2.0944, 2.0944],
        ]
    )
    ok_pos = torch.tensor([0.0, 1.54, 1.28, -1.20, 0.0, 0.0])
    ok_vel = torch.zeros(6)
    assert not oracle.watchdog_trips(ok_pos, ok_vel, limits)
    near_stop = ok_pos.clone()
    near_stop[3] = -1.5708 + 0.03  # inside the 0.05 rad guard band
    assert oracle.watchdog_trips(near_stop, ok_vel, limits)
    fast = ok_vel.clone()
    fast[2] = 6.5
    assert oracle.watchdog_trips(ok_pos, fast, limits)


JOINT6_LIMITS = (-2.0944, 2.0944)


def test_jaw_fold_keeps_the_near_representative_when_joint6_can_hold_it():
    """132 of the 139 folds in OpenYAM's measured 8-env run take this branch, unchanged.

    These are the default (``roll_sign=-1``, ``base_sign=1``) OpenYAM sign
    relations under which this fold's behavior is byte-for-byte carried over;
    the numbers below are OpenYAM's GPU trace, ``layout_seed=0 run_seed=0
    num_envs=8``, control step 240 -- a first brick straight out of SETTLE.
    The near representative needs joint6 in [0.197, 0.998] over the whole
    pick, nowhere near a stop, and the far one would need [-2.945, -2.144].
    """
    assert oracle.fold_jaw_heading(
        want=0.9314,
        jaw_yaw=1.5264,
        joint6=0.0297,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=(-0.428, 0.373),
    ) == pytest.approx(0.9314)


def test_jaw_fold_vetoes_a_prediction_that_only_just_clears_the_stop():
    """Pins the third of OpenYAM's measured abandons, the one the 0.05 rad band let through.

    OpenYAM's GPU trace (default sign relations), ``num_envs=8``, control step
    6115: ``jaw_yaw=0.267``, ``joint6=2.019`` -- the wrist arrives at this pick
    with 0.075 rad of travel left. The near representative is predicted to
    need up to +2.030, which "clears" a 0.05 rad band by 0.014; OpenYAM
    measured that leg reaching +2.0944 and pinning there through RAISE until
    the watchdog abandoned. The far one is predicted at [-1.855, -1.111].
    ``_FOLD_GUARD_RAD`` (0.10) is what makes this a veto, and it is sized from
    that under-prediction -- see its comment.
    """
    args = {
        "want": 3.408682,
        "jaw_yaw": 0.267442,
        "joint6": 2.019390,
        "joint6_limits": JOINT6_LIMITS,
        "base_yaw_range": (-0.732677, 0.010693),
    }
    near = 3.408682 - math.pi
    assert oracle.fold_jaw_heading(**args) == pytest.approx(near + math.pi)
    # The watchdog's own 0.05 band -- what this guard used to be -- lets it through.
    assert oracle.fold_jaw_heading(**args, guard_rad=0.05) == pytest.approx(near)


def test_jaw_fold_takes_the_far_representative_when_the_near_one_runs_past_joint6():
    """Pins OpenYAM's measured HOVER watchdog abandon (OpenYAM's task-9-report.md, fix round 2).

    Same OpenYAM run, episode 3, control step 13380 -- the second brick's
    RETREAT -> HOVER, the transition that abandoned the episode. The near
    representative is only 0.736 rad of heading away but needs joint6 in
    [-2.957, -2.254], a full rad past its -2.0944 stop; joint6 duly walked onto
    that stop in 29 control steps and stayed pinned until the watchdog abandoned
    at HOVER step 56 of 300. The far one needs [0.184, 0.885].
    """
    chosen = oracle.fold_jaw_heading(
        want=3.77903,
        jaw_yaw=3.04343,
        joint6=-1.51795,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=(-0.703, 0.0),
    )
    assert chosen == pytest.approx(3.77903 + math.pi)


def test_jaw_fold_budgets_joint6_for_the_bin_swing_not_just_the_reach():
    """Pins the SECOND of OpenYAM's measured abandons, the one a reach-only budget missed.

    Fixing only the reach to the brick moved the failure from the second
    brick's HOVER to the third brick's EXTEND, joint6 pinned at the OTHER
    (+2.0944) stop: control step 14402, ``jaw_yaw=0.90237``, ``joint6=0.8236``.
    Over the reach alone the near representative is harmless, but held out to
    the bin it needs joint6 up to +2.123 -- past the stop -- against the far
    one's [-1.484, -1.019]. This is why ``Goal.yaw_hold_xy`` carries the bin and
    the retreat point and not just the brick.
    """
    reach_only = (-0.289, 0.0)
    assert oracle.fold_jaw_heading(
        want=-0.22089,
        jaw_yaw=0.90237,
        joint6=0.8236,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=reach_only,
    ) == pytest.approx(-0.22089)
    whole_pick = (-0.289, 0.176)
    assert oracle.fold_jaw_heading(
        want=-0.22089,
        jaw_yaw=0.90237,
        joint6=0.8236,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=whole_pick,
    ) == pytest.approx(-0.22089 + math.pi)


def test_jaw_fold_tolerates_a_leg_that_only_grazes_the_stop():
    """Never flip to something worse, however marginal the near option is.

    OpenYAM's ``num_envs=1`` run, control step 2844: the near representative
    needs joint6 up to 2.028 -- inside the stop but inside ``_FOLD_GUARD_RAD``
    too, so it does count as a violation -- and joint6 was measured to touch
    2.0944 transiently later in that pick, yet the episode succeeded (the position
    servo keeps moving, so OracleRig's strike counter resets). The far
    representative reaches -2.360, a bigger violation, so there is nothing to
    switch to and the near one is kept.
    """
    assert oracle.fold_jaw_heading(
        want=0.3981,
        jaw_yaw=1.4209,
        joint6=0.3005,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=(-0.542, 0.705),
    ) == pytest.approx(0.3981)


def test_jaw_fold_keeps_the_near_representative_when_neither_fits():
    """No reachable side: prefer the near one rather than adding a wrist flip.

    joint6 already on its upper stop and the base about to swing further the
    same way -- at the default ``_FOLD_GUARD_RAD`` (0.10) the near
    representative overshoots by 0.60 rad and the far one by 3.74, so there is
    nothing to gain by flipping. The phase budget and the watchdog handle it
    from there; this function never invents a third option.
    """
    assert oracle.fold_jaw_heading(
        want=0.0,
        jaw_yaw=0.0,
        joint6=2.0944,
        joint6_limits=JOINT6_LIMITS,
        base_yaw_range=(0.0, 0.5),
    ) == pytest.approx(0.0)


def test_hover_and_descend_goals_carry_the_same_yaw_hold_waypoints():
    """The fold must not flip between HOVER and DESCEND: same brick, same bin.

    DESCEND re-runs the fold from a wrist that HOVER has already turned, so the
    two have to agree by construction rather than by luck -- if DESCEND saw a
    different joint6 budget it could flip the heading back with the jaws already
    over the brick.
    """
    orc = _oracle()
    scene = _snapshot()
    hover = _run_phase(orc, scene)  # SETTLE
    hover = _run_phase(orc, scene)  # HOVER
    descend = orc.next_goal(scene)
    assert descend.phase == oracle.Phase.DESCEND
    assert len(hover.yaw_hold_xy) == 3
    assert hover.yaw_hold_xy == descend.yaw_hold_xy


def test_yaw_hold_waypoints_span_brick_bin_and_retreat():
    orc = _oracle()
    scene = _snapshot(brick_xy=(0.20, 0.05))
    _run_phase(orc, scene)  # SETTLE
    hover = orc.next_goal(scene)
    brick_xy, release_xy, retreat_xy = hover.yaw_hold_xy
    assert brick_xy == pytest.approx((float(hover.target_pos[0]), float(hover.target_pos[1])))
    expected = orc._release_target(orc.target_brick)
    assert release_xy == pytest.approx((float(expected[0]), float(expected[1])))
    assert retreat_xy[0] == pytest.approx(oracle.RETREAT_POINT[0] + orc.draw.via_dx)
    assert retreat_xy[1] == pytest.approx(oracle.RETREAT_POINT[1] + orc.draw.via_dy)


def test_wrap_to_pi_round_trips():
    for angle in (0.0, 3.0, -3.0, 4.0, -4.0, 7.5):
        assert oracle.wrap_to_pi(angle) == pytest.approx(
            math.atan2(math.sin(angle), math.cos(angle))
        )


def _induced_oracle(prob=1.0, offset_weight=1.0, seed=7):
    ind = oracle.FailureInduction(prob=prob, offset_weight=offset_weight)
    return oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT,
        num_bricks=1,
        draw=_draw(),
        induction=ind,
        rng=random.Random(seed),
    )


def test_offset_induction_offsets_first_attempt_and_retry_is_clean():
    orc = _induced_oracle()
    clean = _oracle()  # same _draw() (seed 0), no induction
    # Non-zero brick_yaw: at brick_yaw=0 a fixed world-frame +/-y offset would
    # pass this test by coincidence, without actually pinning yaw-relativity.
    scene = _snapshot(brick_yaw=0.9)
    _run_phase(orc, scene)  # SETTLE
    _run_phase(clean, scene)
    hover_induced = _run_phase(orc, scene)  # HOVER (induction drawn here)
    hover_clean = _run_phase(clean, scene)
    dx = float(hover_induced.target_pos[0] - hover_clean.target_pos[0])
    dy = float(hover_induced.target_pos[1] - hover_clean.target_pos[1])
    r = math.hypot(dx, dy)
    assert 0.015 <= r <= 0.025
    # Goal.yaw is the jaw-LINE heading (brick long axis); the runtime commands
    # the CLOSING axis at yaw + pi/2 (oracle_runtime.desired_quat). The offset
    # must stay small along the closing axis, not along yaw itself.
    close = hover_induced.yaw + math.pi / 2.0  # closing axis at HOVER
    along = abs(dx * math.cos(close) + dy * math.sin(close)) / r
    assert along <= math.sin(math.pi / 6.0) + 1e-9  # within perpendicular +/- 30 deg
    descend_induced = _run_phase(orc, scene, outcome="stalled")  # DESCEND
    descend_clean = _run_phase(clean, scene, outcome="stalled")
    ddx = float(descend_induced.target_pos[0] - descend_clean.target_pos[0])
    ddy = float(descend_induced.target_pos[1] - descend_clean.target_pos[1])
    assert math.hypot(ddx, ddy) == pytest.approx(r, abs=1e-9)
    dclose = descend_induced.yaw + math.pi / 2.0  # closing axis at DESCEND
    dalong = abs(ddx * math.cos(dclose) + ddy * math.sin(dclose)) / r
    assert dalong <= math.sin(math.pi / 6.0) + 1e-9  # within perpendicular +/- 30 deg
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=False, lifted=False)  # LIFT fails -> REOPEN
    _run_phase(orc, scene)  # REOPEN -> HOVER, same brick
    hover_retry = orc.next_goal(scene)  # retry HOVER
    assert torch.allclose(hover_retry.target_pos[:2], hover_clean.target_pos[:2], atol=1e-6)


def test_disabled_induction_consumes_no_rng_and_matches_baseline():
    rng = random.Random(7)
    orc = oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT, num_bricks=1, draw=_draw(), induction=None, rng=rng
    )
    clean = _oracle()
    scene = _snapshot()
    state_before = rng.getstate()
    for kwargs in (
        {},  # SETTLE
        {},  # HOVER
        {"outcome": "stalled"},  # DESCEND
        {},  # CLOSE
        {"pinch_ok": True, "lifted": True},  # LIFT
    ):
        g1 = _run_phase(orc, scene, **kwargs)
        g2 = _run_phase(clean, scene, **kwargs)
        if g1.target_pos is not None:
            assert torch.allclose(g1.target_pos, g2.target_pos)
    assert rng.getstate() == state_before


def test_zero_prob_induction_consumes_no_rng_and_matches_baseline():
    # Mirrors the ``induction=None`` case above, but with induction present and
    # its probability pinned at 0.0 -- pins the ``prob > 0.0`` short-circuit in
    # HOVER's draw so a configured-but-disabled induction never touches the rng.
    rng = random.Random(7)
    orc = oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT,
        num_bricks=1,
        draw=_draw(),
        induction=oracle.FailureInduction(prob=0.0),
        rng=rng,
    )
    clean = _oracle()
    scene = _snapshot()
    state_before = rng.getstate()
    for kwargs in (
        {},  # SETTLE
        {},  # HOVER
        {"outcome": "stalled"},  # DESCEND
        {},  # CLOSE
        {"pinch_ok": True, "lifted": True},  # LIFT
    ):
        g1 = _run_phase(orc, scene, **kwargs)
        g2 = _run_phase(clean, scene, **kwargs)
        if g1.target_pos is not None:
            assert torch.allclose(g1.target_pos, g2.target_pos)
    assert rng.getstate() == state_before


def test_noise_boost_scoped_to_first_attempt_descend_and_close():
    orc = _induced_oracle(offset_weight=0.0)  # always noise style
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    orc.next_goal(scene)  # HOVER goal; induction drawn; phase == HOVER
    assert orc.noise_factor() == 1.0
    orc.report(oracle.PhaseResult(outcome="converged"), scene)  # -> DESCEND
    assert orc.noise_factor() == pytest.approx(5.0)
    orc.next_goal(scene)
    orc.report(oracle.PhaseResult(outcome="stalled"), scene)  # -> CLOSE
    assert orc.noise_factor() == pytest.approx(5.0)
    orc.next_goal(scene)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)  # -> LIFT
    assert orc.noise_factor() == 1.0
    # fail the lift; the retry's DESCEND must NOT be boosted
    orc.next_goal(scene)
    orc.report(
        oracle.PhaseResult(outcome="converged", pinch_ok=False, lifted=False), scene
    )  # -> REOPEN
    _run_phase(orc, scene)  # REOPEN -> HOVER
    orc.next_goal(scene)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)  # -> DESCEND (retry)
    assert orc.noise_factor() == 1.0


def test_induction_counters_count_per_style():
    orc = _induced_oracle(offset_weight=1.0)
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    assert (orc.induced_offset, orc.induced_noise) == (1, 0)
    orc2 = _induced_oracle(offset_weight=0.0)
    _run_phase(orc2, scene)
    _run_phase(orc2, scene)
    assert (orc2.induced_offset, orc2.induced_noise) == (0, 1)


def test_uninduced_oracle_noise_factor_is_one():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER -> DESCEND next
    assert orc.noise_factor() == 1.0


def test_jaw_fold_sign_parameters_mirror_the_relation():
    """With roll_sign=+1 the wrist must move WITH the heading, so the representative that
    was unreachable under the OpenYAM relation (-1) becomes the reachable one."""
    limits = (-2.0, 2.0)
    # Heading +1.5 rad away; wrist at +1.0 with 1.0 rad of travel left toward +2.0.
    default = oracle.fold_jaw_heading(
        1.5, 0.0, 1.0, limits
    )  # roll_sign -1: needs 1.0 - 1.5 = -0.5 -> near is fine
    assert default == pytest.approx(1.5)
    flipped = oracle.fold_jaw_heading(1.5, 0.0, 1.0, limits, roll_sign=1.0)
    # roll_sign +1: near needs 1.0 + 1.5 = 2.5 > 2.0 - guard -> far (-1.64 rad away) is chosen.
    assert flipped == pytest.approx(1.5 + math.pi)


def test_jaw_fold_base_sign_flips_the_swing_budget():
    limits = (-2.0, 2.0)  # usable [-1.9, 1.9] after the 0.10 guard
    # Near representative -0.5 needs the wrist at 1.0 - (-0.5) = 1.5; the far one needs
    # 1.5 - pi = -1.64. A +1.2 rad base swing to come adds +1.2 to the wrist under the
    # default relation (k = -base_sign/roll_sign = +1): near would reach 2.7 (out), far
    # stays at -0.44 (in) -> far. Flip base_sign and the swing subtracts (k = -1): near
    # spans [0.3, 1.5] (in) -> near.
    assert oracle.fold_jaw_heading(
        -0.5, 0.0, 1.0, limits, base_yaw_range=(0.0, 1.2)
    ) == pytest.approx(-0.5 + math.pi)
    assert oracle.fold_jaw_heading(
        -0.5, 0.0, 1.0, limits, base_yaw_range=(0.0, 1.2), base_sign=-1.0
    ) == pytest.approx(-0.5)


# --- randomized pick order -------------------------------------------------
#
# Brick color is ``index % 4`` and ``_next_brick`` takes the lowest index, so every
# demonstration in a lowest-index corpus picks R,B,Y,G,R,B in that order: a schedule a
# policy can follow without looking at what it carries, and openyam-lab measured its policy
# doing exactly that (8.3% wrong-bin on-schedule vs 42.0% off-schedule). These tests pin the
# knob that removes the schedule (MS2 spec section 3.1), and pin that removing it costs none
# of the guarantees the fixed order gave: same brick on a retry, every brick exactly once,
# reachability filter intact, and the MS1 sequence byte for byte with the knob off.


def _desk_snapshot(num_bricks, unreachable=()):
    """``num_bricks`` bricks in a reachable row on the desk."""
    bl = layout.DEFAULT_BIN_LAYOUT
    pos = torch.zeros(num_bricks, 3)
    for i in range(num_bricks):
        x = layout.top_down_x_limit(bl) + 0.02 if i in unreachable else 0.20
        pos[i] = torch.tensor([x, -0.10 + 0.04 * i, layout.BRICK_SIZE[2] / 2])
    return oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.zeros(num_bricks),
        brick_lin_vel=torch.zeros(num_bricks, 3),
        pick_point=torch.tensor([0.28, 0.0, 0.20]),
    )


def _order_oracle(num_bricks=6, seed=0, randomize=True, rng=None):
    noise = oracle.OracleNoise(randomize_pick_order=randomize)
    return oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT,
        num_bricks=num_bricks,
        draw=oracle.sample_episode_draw(random.Random(0), noise),
        rng=random.Random(seed) if rng is None else rng,
    )


def _stow(scene, brick):
    """Teleport ``brick`` into its color's bin, in place: RETREAT marks a brick sorted only
    when the scene shows it stowed (the MS2 re-pick rule), so a clean cycle has to land."""
    bl = layout.DEFAULT_BIN_LAYOUT
    slot = bl.slot_colors.index(brick % len(layout.BRICK_COLOR_ORDER))
    scene.brick_pos[brick] = layout.bin_interior_centers(bl)[slot]
    scene.brick_lin_vel[brick] = 0.0


def _run_pick(orc, scene):
    """One clean pick->place cycle from HOVER; returns the brick it targeted."""
    if orc.next_goal(scene) is None:  # HOVER's entry selects the target
        return None
    brick = orc.target_brick
    orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=None, lifted=None), scene)
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT
    for _ in range(3):  # RAISE, EXTEND, RELEASE
        _run_phase(orc, scene)
    _stow(scene, brick)
    _run_phase(orc, scene)  # RETREAT sees the brick stowed
    return brick


def _run_episode(orc, scene):
    """Drive an episode of clean picks to DONE (or ABANDON); returns the order."""
    _run_phase(orc, scene)  # SETTLE
    while _run_pick(orc, scene) is not None:
        pass
    return list(orc.pick_order)


def test_pick_order_defaults_to_lowest_index_and_consumes_no_randomness():
    """Default off must be byte-identical: the MS1 order AND the same rng stream, or
    every MS1 shard stops being comparable with the MS2 corpus."""
    assert oracle.OracleNoise().randomize_pick_order is False
    assert _draw().randomize_pick_order is False
    rng = random.Random(3)
    orc = _order_oracle(randomize=False, rng=rng)
    before = rng.getstate()
    assert _run_episode(orc, _desk_snapshot(6)) == [0, 1, 2, 3, 4, 5]
    assert rng.getstate() == before


def test_pick_order_is_recorded_with_the_knob_off_too():
    """The sidecar's positive record that a run's schedule WAS fixed."""
    orc = oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, num_bricks=3, draw=_draw())
    assert _run_episode(orc, _desk_snapshot(3)) == [0, 1, 2]


def test_randomized_pick_order_varies_and_leaves_the_fixed_schedule():
    """The point of the knob: neither the brick order nor the color order it implies may
    stay predictable across episodes."""
    orders = [
        tuple(_run_episode(_order_oracle(seed=seed), _desk_snapshot(6))) for seed in range(20)
    ]
    assert len(set(orders)) >= 10
    assert sum(order == (0, 1, 2, 3, 4, 5) for order in orders) <= 1
    colors = {tuple(b % len(layout.BRICK_COLOR_ORDER) for b in order) for order in orders}
    assert len(colors) >= 10


def test_randomized_pick_order_eventually_chooses_every_reachable_brick_first():
    """Uniform over the candidates: over many episodes every brick is the FIRST pick."""
    firsts = {_run_episode(_order_oracle(seed=seed), _desk_snapshot(6))[0] for seed in range(60)}
    assert firsts == set(range(6))


def test_randomized_pick_order_still_stows_every_brick_exactly_once():
    """Randomizing the schedule must not cost coverage: no repeats, no skips."""
    for seed in range(20):
        orc = _order_oracle(seed=seed)
        order = _run_episode(orc, _desk_snapshot(6))
        assert sorted(order) == list(range(6))
        assert orc.phase == oracle.Phase.DONE


def test_randomized_pick_order_keeps_a_retry_on_the_same_brick():
    """Randomization selects a NEW target; a failed grasp still re-attempts the brick it
    just missed, and does not count as a second pick."""
    orc = _order_oracle(seed=4)
    scene = _desk_snapshot(6)
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER selects
    first = orc.target_brick
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=False, lifted=False)  # LIFT fails -> REOPEN
    _run_phase(orc, scene)  # REOPEN -> HOVER
    _run_phase(orc, scene)  # HOVER again
    assert orc.target_brick == first
    assert orc.pick_order == [first]


def test_randomized_pick_order_is_reproducible_from_the_seed():
    """A shard's order has to be re-derivable from its recorded seeds."""
    assert _run_episode(_order_oracle(seed=17), _desk_snapshot(6)) == _run_episode(
        _order_oracle(seed=17), _desk_snapshot(6)
    )
    assert _run_episode(_order_oracle(seed=17), _desk_snapshot(6)) != _run_episode(
        _order_oracle(seed=18), _desk_snapshot(6)
    )


def test_randomized_pick_order_still_skips_unreachable_bricks():
    """The reachability filter is the oracle's, not the ordering's."""
    for seed in range(8):
        orc = _order_oracle(seed=seed)
        order = _run_episode(orc, _desk_snapshot(6, unreachable=(2,)))
        assert 2 not in order
        assert sorted(order) == [0, 1, 3, 4, 5]
        assert orc.phase == oracle.Phase.ABANDON


def test_randomize_pick_order_without_an_rng_is_refused():
    """A run_meta advertising a randomized schedule over a corpus that silently kept the
    fixed one is worse than a crash."""
    noise = oracle.OracleNoise(randomize_pick_order=True)
    with pytest.raises(ValueError, match="rng"):
        oracle.EnvOracle(
            layout.DEFAULT_BIN_LAYOUT,
            num_bricks=6,
            draw=oracle.sample_episode_draw(random.Random(0), noise),
        )


def _full_cycle_to_retreat(orc, scene):
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=True, lifted=True)  # LIFT
    _run_phase(orc, scene)  # RAISE
    _run_phase(orc, scene)  # EXTEND
    _run_phase(orc, scene)  # RELEASE


def test_a_brick_lost_on_the_desk_is_picked_again_after_retreat():
    """RETREAT used to mark its brick sorted unconditionally, so a brick that never reached
    the bin -- MEASURED on GPU (2026-09-07): an EXTEND with no plan, RELEASE opening over
    the scatter band, the brick flat on the desk at radius 0.20 m -- was written off, the
    oracle went DONE with a brick on the desk, and the episode idled ~1500 control steps to
    its time_out. A brick still inside the top-down band is a brick to pick again.
    """
    orc = _oracle()
    _full_cycle_to_retreat(orc, _snapshot())
    # RETREAT ends with the brick back on the desk, inside the band.
    dropped = _snapshot(brick_xy=(0.19, -0.04))
    _run_phase(orc, dropped)  # RETREAT
    hover = orc.next_goal(dropped)
    assert hover is not None and hover.phase == oracle.Phase.HOVER
    assert orc.target_brick == 0
    assert orc.retries_used == 0
    assert hover.target_pos[0].item() == pytest.approx(0.19, abs=0.03)
    assert hover.target_pos[1].item() == pytest.approx(-0.04, abs=0.02)


def test_a_brick_lost_out_of_reach_abandons_at_once_instead_of_idling():
    orc = _oracle()
    _full_cycle_to_retreat(orc, _snapshot())
    beyond = layout.top_down_x_limit(layout.DEFAULT_BIN_LAYOUT) + 0.02
    lost = _snapshot(brick_xy=(beyond, 0.0))
    _run_phase(orc, lost)  # RETREAT
    assert orc.next_goal(lost) is None
    assert orc.phase == oracle.Phase.ABANDON


def test_a_stowed_brick_stays_sorted_even_if_it_is_later_jostled():
    """`_sorted` still exists for the brick that DID arrive: a later delivery can knock it
    so it is momentarily not at rest, and that must not un-sort it."""
    orc = _oracle()
    _full_cycle_to_retreat(orc, _snapshot())
    _run_phase(orc, _snapshot(stowed=(0,)))  # RETREAT, brick in its bin
    moving = _snapshot(stowed=(0,))
    moving.brick_lin_vel[0, 0] = 0.5  # jostled: not at rest this instant
    assert orc.next_goal(moving) is None
    assert orc.phase == oracle.Phase.DONE


# --- hover_jitter: corrective approaches (MS3 recipe design 5.1, ported from openyam-lab) ---


def _jittered_oracle(seed=3, jitter=(0.005, 0.035)):
    noise = oracle.OracleNoise(hover_jitter=jitter)
    draw = oracle.sample_episode_draw(random.Random(seed), noise)
    return oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT, num_bricks=1, draw=draw, rng=random.Random(seed)
    )


def test_hover_jitter_is_off_by_default_and_needs_no_rng():
    """With the knob off nothing changes: the HOVER target is the brick plus the episode's
    lateral offset, no jitter is counted, and an oracle without an rng still runs."""
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    goal = orc.next_goal(scene)
    assert goal.phase == oracle.Phase.HOVER
    assert float(goal.target_pos[0]) == pytest.approx(0.20 + orc.draw.lateral_dx)
    assert float(goal.target_pos[1]) == pytest.approx(0.05 + orc.draw.lateral_dy)
    assert orc.hover_jitters == 0 and orc.draw.hover_jitter_range == (0.0, 0.0)


def test_hover_jitter_offsets_hover_only_inside_the_annulus():
    orc = _jittered_oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    hover = _run_phase(orc, scene)  # HOVER
    dx = float(hover.target_pos[0]) - (0.20 + orc.draw.lateral_dx)
    dy = float(hover.target_pos[1]) - (0.05 + orc.draw.lateral_dy)
    assert 0.005 <= math.hypot(dx, dy) <= 0.035
    assert orc.hover_jitters == 1
    descend = orc.next_goal(scene)
    assert descend.phase == oracle.Phase.DESCEND
    assert float(descend.target_pos[0]) == pytest.approx(0.20 + orc.draw.lateral_dx)
    assert float(descend.target_pos[1]) == pytest.approx(0.05 + orc.draw.lateral_dy)


def test_hover_jitter_is_redrawn_for_a_retry():
    """A retry re-enters HOVER and deserves its own correction to demonstrate."""
    orc = _jittered_oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    first = _run_phase(orc, scene).target_pos.clone()  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE
    _run_phase(orc, scene, pinch_ok=False, lifted=False)  # LIFT, no pinch
    _run_phase(orc, scene)  # REOPEN
    second = _run_phase(orc, scene).target_pos  # HOVER again, same brick
    assert orc.hover_jitters == 2
    assert not torch.allclose(first[:2], second[:2])


def test_hover_jitter_without_an_rng_is_refused():
    noise = oracle.OracleNoise(hover_jitter=(0.005, 0.035))
    with pytest.raises(ValueError, match="rng"):
        oracle.EnvOracle(
            layout.DEFAULT_BIN_LAYOUT,
            num_bricks=1,
            draw=oracle.sample_episode_draw(random.Random(0), noise),
        )


# --- nearest-first pick order ----------------------------------------------
#
# The third order (MS3 visual-inspection plan, lever 5). The randomized order removes the
# color schedule, but it also leaves the next target unreadable from the image: the
# 2026-09-11 probes found 30-45% of the closes 2 cm or further from any brick sitting
# BETWEEN two bricks under 10 cm apart, the signature of a policy averaging over candidates
# it cannot disambiguate. Nearest-first keeps the schedule broken -- the desk layout is
# random, so the color order still varies per episode -- and makes the target a function of
# the scene the cameras see. These tests pin that both properties hold.


def _scatter_snapshot(xys, pick_point=(0.28, 0.0, 0.20)):
    """Bricks at the given desk xy positions, tool at ``pick_point``."""
    pos = torch.zeros(len(xys), 3)
    for i, (x, y) in enumerate(xys):
        pos[i] = torch.tensor([x, y, layout.BRICK_SIZE[2] / 2])
    return oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.zeros(len(xys)),
        brick_lin_vel=torch.zeros(len(xys), 3),
        pick_point=torch.tensor(list(pick_point)),
    )


def _nearest_oracle(num_bricks=6, rng=None):
    noise = oracle.OracleNoise(nearest_first_pick_order=True)
    return oracle.EnvOracle(
        layout.DEFAULT_BIN_LAYOUT,
        num_bricks=num_bricks,
        draw=oracle.sample_episode_draw(random.Random(0), noise),
        rng=rng,
    )


def test_nearest_first_takes_the_closest_brick_and_consumes_no_randomness():
    """The rule is the scene's, not the seed's: a deterministic order that spends no
    randomness, so a nearest-first shard stays re-derivable from its layout alone."""
    assert oracle.OracleNoise().nearest_first_pick_order is False
    assert _draw().nearest_first_pick_order is False
    rng = random.Random(3)
    orc = _nearest_oracle(num_bricks=4, rng=rng)
    before = rng.getstate()
    # Distances from the tool at x 0.28: brick 1 is nearest, then 3, 0, 2.
    scene = _scatter_snapshot([(0.20, -0.10), (0.26, 0.02), (0.18, 0.14), (0.24, -0.05)])
    assert _run_episode(orc, scene) == [1, 3, 0, 2]
    assert rng.getstate() == before


def test_nearest_first_leaves_the_fixed_color_schedule():
    """Nearest-first must not hand back the R,B,Y,G shortcut the randomized order removed:
    the desk is scattered at random, so the color order has to vary across episodes."""
    rng = random.Random(11)
    colors = set()
    for _ in range(20):
        xys = [(rng.uniform(0.14, 0.26), rng.uniform(-0.14, 0.14)) for _ in range(6)]
        order = _run_episode(_nearest_oracle(), _scatter_snapshot(xys))
        colors.add(tuple(b % len(layout.BRICK_COLOR_ORDER) for b in order))
    assert len(colors) >= 10
    assert (0, 1, 2, 3, 0, 1) not in colors


def test_nearest_first_still_stows_every_brick_exactly_once():
    """Changing the order must not cost coverage: no repeats, no skips, and the
    reachability filter stays the oracle's."""
    orc = _nearest_oracle()
    order = _run_episode(orc, _desk_snapshot(6))
    assert sorted(order) == list(range(6))
    assert orc.phase == oracle.Phase.DONE
    orc = _nearest_oracle()
    assert 2 not in _run_episode(orc, _desk_snapshot(6, unreachable=(2,)))
    assert orc.phase == oracle.Phase.ABANDON


def test_nearest_first_and_randomize_pick_order_together_are_refused():
    """Two orders cannot both be the order. A run_meta advertising both would leave no
    reader able to say which one its episodes carried."""
    noise = oracle.OracleNoise(randomize_pick_order=True, nearest_first_pick_order=True)
    with pytest.raises(ValueError, match="one pick order"):
        oracle.EnvOracle(
            layout.DEFAULT_BIN_LAYOUT,
            num_bricks=3,
            draw=oracle.sample_episode_draw(random.Random(0), noise),
            rng=random.Random(0),
        )


def test_resume_with_a_held_brick_enters_lift_then_raise():
    orc = _oracle()
    scene = _snapshot()
    orc.resume(scene, held=0)
    assert orc.phase is oracle.Phase.LIFT and orc.target_brick == 0
    goal = _run_phase(orc, scene, pinch_ok=True, lifted=True)
    assert goal.phase is oracle.Phase.LIFT and goal.gripper == -1.0
    assert orc.phase is oracle.Phase.RAISE and orc.pick_order == [0]


def test_resume_without_a_held_brick_enters_hover_on_the_nearest_candidate():
    orc = _oracle()
    orc.resume(_snapshot(), held=-1)
    assert orc.phase is oracle.Phase.HOVER and orc.target_brick is None
    goal = orc.next_goal(_snapshot())
    assert goal.phase is oracle.Phase.HOVER and orc.target_brick == 0


def test_resume_on_a_finished_desk_is_done():
    orc = _oracle()
    scene = _snapshot(stowed=(0,))
    orc.resume(scene, held=-1)
    assert orc.sorted_bricks == frozenset({0})
    assert orc.next_goal(scene) is None and orc.phase is oracle.Phase.DONE


def test_resume_replaces_the_sorted_set_instead_of_merging_it():
    """Design 6.2: `resume()` sets `self._sorted = self._stowed(scene)`, a replace. A brick
    the oracle itself watched arrive and marked sorted, earlier in this same episode, must
    drop back out of `sorted_bricks` if the resumed scene no longer shows it at rest in its
    bin -- the case a policy takeover can leave behind, for instance a later knock that sends
    a delivered brick back onto the desk while the policy held the env. A union
    (`self._sorted |= self._stowed(scene)`) would keep the stale membership, and this test
    would not catch it.
    """
    orc = oracle.EnvOracle(layout.DEFAULT_BIN_LAYOUT, num_bricks=2, draw=_draw())
    scene = _desk_snapshot(2)
    _full_cycle_to_retreat(orc, scene)  # picks brick 0, the lowest-index default target
    _stow(scene, 0)
    _run_phase(orc, scene)  # RETREAT sees brick 0 stowed -> `_sorted` gains 0
    assert orc.sorted_bricks == frozenset({0})
    knocked_out = _desk_snapshot(2)  # brick 0 back on the desk, out of its bin
    orc.resume(knocked_out, held=-1)
    assert orc.sorted_bricks == frozenset()


def test_normal_lift_target_is_the_live_height_plus_lift_height_uncapped():
    """The resume-only clamp (MS4 c2 fix round) must leave a normal CLOSE->LIFT transition
    bit-for-bit unchanged: target z is `scene.pick_point[2] + LIFT_HEIGHT`, with no ceiling
    at `LIFT_HEIGHT` itself. `_snapshot()`'s fixed `pick_point` z (0.20) sits well above
    `LIFT_HEIGHT` (0.10) on its own, so a target that was wrongly clamped to `LIFT_HEIGHT`
    would read 0.10 here instead of 0.30, and this assertion would catch it.
    """
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    _run_phase(orc, scene)  # HOVER
    _run_phase(orc, scene, outcome="stalled")  # DESCEND
    _run_phase(orc, scene)  # CLOSE -> LIFT
    lift = orc.next_goal(scene)
    assert lift.phase is oracle.Phase.LIFT
    assert lift.target_pos[2].item() == pytest.approx(
        scene.pick_point[2].item() + oracle.LIFT_HEIGHT
    )


def test_nominal_draw_has_no_noise_and_nearest_first():
    draw = oracle.nominal_draw()
    assert draw.hover_jitter_range == (0.0, 0.0) and draw.action_noise_std == 0.0
    assert draw.lateral_dx == 0.0 and draw.yaw_offset == 0.0 and draw.nearest_first_pick_order


def test_push_phases_are_pinned_after_abandon():
    assert oracle.PHASE_TO_INT["PUSH_HOVER"] == 12
    assert oracle.PHASE_TO_INT["PUSH_DESCEND"] == 13
    assert oracle.PHASE_TO_INT["PUSH_SWEEP"] == 14
    assert oracle.PUSH_PHASES == frozenset(
        {oracle.Phase.PUSH_HOVER, oracle.Phase.PUSH_DESCEND, oracle.Phase.PUSH_SWEEP}
    )
    assert len(oracle.PHASE_TO_INT) == len(oracle.Phase)


def test_the_tip_phases_are_appended_and_never_renumbered():
    assert oracle.TIP_PHASES == frozenset(
        {oracle.Phase.TIP_HOVER, oracle.Phase.TIP_DESCEND, oracle.Phase.TIP_SWEEP}
    )
    assert oracle.PHASE_TO_INT["TIP_HOVER"] == 15
    assert oracle.PHASE_TO_INT["TIP_DESCEND"] == 16
    assert oracle.PHASE_TO_INT["TIP_SWEEP"] == 17
    assert oracle.PHASE_TO_INT["PUSH_SWEEP"] == 14  # nothing renumbered
    assert len(set(oracle.PHASE_TO_INT.values())) == len(oracle.PHASE_TO_INT)
    assert set(oracle.PHASE_TO_INT) == {p.name for p in oracle.Phase}


def test_goals_carry_the_brick_half_height_by_default():
    orc = _oracle()
    scene = _snapshot()
    _run_phase(orc, scene)  # SETTLE
    hover = orc.next_goal(scene)
    assert hover.half_height == pytest.approx(layout.BRICK_SIZE[2] / 2)
    assert orc.grasp_width == layout.BRICK_SIZE[1]
    assert scene.brick_quat is None


def test_retreat_appends_the_cycle_outcome():
    orc = _oracle()
    scene = _snapshot()
    for outcome, pinch, lifted in (
        ("converged", None, None),  # SETTLE
        ("converged", None, None),  # HOVER
        ("stalled", None, None),  # DESCEND
        ("converged", None, None),  # CLOSE
        ("converged", True, True),  # LIFT
        ("converged", None, None),  # RAISE
        ("converged", None, None),  # EXTEND
        ("converged", None, None),  # RELEASE
    ):
        _run_phase(orc, scene, outcome=outcome, pinch_ok=pinch, lifted=lifted)
    assert orc.phase is oracle.Phase.RETREAT and orc.cycle_outcomes == []
    _run_phase(orc, _snapshot(stowed=(0,)))  # RETREAT with the brick in its bin
    assert orc.cycle_outcomes == [True]
    orc2 = _oracle()
    for _ in range(9):
        _run_phase(orc2, scene, outcome="stalled", pinch_ok=True, lifted=True)
    assert orc2.cycle_outcomes == [False]


def test_the_hooks_reproduce_the_legacy_choices():
    orc = _oracle()
    scene = _snapshot(brick_yaw=0.4)
    _run_phase(orc, scene)
    hover = orc.next_goal(scene)
    assert hover.yaw == pytest.approx(0.4 + orc.draw.yaw_offset)
    assert orc._no_target_phase(_snapshot(stowed=(0,))) is oracle.Phase.DONE
    assert orc._no_target_phase(_snapshot()) is oracle.Phase.ABANDON
