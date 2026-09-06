"""CPU tests for the oracle phase state machine (no isaaclab)."""

import math
import random

import pytest
import torch

from maker_arm_lab.core import layout, oracle


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
    # OpenYAM's measured release floor: no collision-free open-gripper pose
    # below z ~= 0.24 (OpenYAM's M1 Task 9); carried over here, not yet
    # re-measured for the Maker Arm -- the constant may be re-measured upward
    # by Task D4 but never below this.
    assert oracle.RELEASE_Z >= 0.24


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
    # tilt is RELEASE_TILT_RAD from the start, not 0.0: gives joint4 the same
    # margin during the climb that the final release pose needs (OpenYAM
    # measured; same comment).
    assert raise_goal.tilt_rad == pytest.approx(oracle.RELEASE_TILT_RAD)
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
