"""The heap teacher on stub snapshots (design section 7)."""

import math
import random

import pytest
import torch

from maker_arm_lab.core import grasp, heap_oracle, layout, oracle, parts
from maker_arm_lab.core.quat import quat_from_rpy_deg

BL = layout.DEFAULT_BIN_LAYOUT
BRICK = parts.make_part("brick", 2, 4)
SMALL = parts.make_part("brick", 1, 2)
LONG_PLATE = parts.make_part("plate", 2, 8)  # one closing axis: its 15.8 mm side
TOWER_OFFSET = 0.044
"""How far out a 48 mm tower sits to block the target's landing spot but not another tower.

The jaws now descend at an opening matched to the grasp, so the landing spot of a 15.8 mm
grasp starts 19.9 mm out and the tip hull reaches 43.9 mm out at a tower's height. A tower
blocks the target between 27.8 mm and 51.8 mm; the towers clear EACH OTHER, which sets the
floor, because the nearest pair is sqrt(2) times this apart.
"""
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _snapshot(positions, stowed=()):
    pos = torch.tensor(positions, dtype=torch.float32)
    for b in stowed:
        # HeapOracle._stowed grades against the one heap bin (the MS6 z-centre fix), so a
        # "stowed" stub sits at layout.heap_bin_center, not a per-color slot.
        pos[b] = torch.tensor(layout.heap_bin_center(BL))
    n = pos.shape[0]
    return oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.zeros(n),
        brick_lin_vel=torch.zeros(n, 3),
        pick_point=torch.tensor([0.28, 0.0, 0.20]),
        brick_quat=IDENTITY.expand(n, 4).clone(),
    )


def _oracle(part_list, destination=1, budget=heap_oracle.CYCLE_BUDGET, seed=0):
    return heap_oracle.HeapOracle(
        BL,
        part_list,
        destination,
        oracle.nominal_draw(),
        rng=random.Random(seed),
        cycle_budget=budget,
    )


def _step(orc, scene, outcome="converged", pinch_ok=None, lifted=None):
    goal = orc.next_goal(scene)
    if goal is not None:
        orc.report(oracle.PhaseResult(outcome=outcome, pinch_ok=pinch_ok, lifted=lifted), scene)
    return goal


def _to_hover(orc, scene):
    goal = _step(orc, scene)  # SETTLE
    assert goal.phase is oracle.Phase.SETTLE
    return orc.next_goal(scene)  # HOVER entry selects the target


def _heap_with_parts(parts_list, flat):
    """A heap oracle plus a scene with each part on the desk at the disc center.

    `flat=True` sits each part at the identity quaternion, so a plate lies with its 3.2 mm
    thickness straight up. `flat=False` rolls each part 90 degrees about its long axis, the
    way a tip stands a plate up onto its short edge.
    """
    orc = _oracle(parts_list)
    quat = IDENTITY if flat else torch.tensor(quat_from_rpy_deg((90.0, 0.0, 0.0)))
    positions = [
        [layout.HEAP_CENTER[0] + 0.03 * i, layout.HEAP_CENTER[1], p.size[2] / 2]
        for i, p in enumerate(parts_list)
    ]
    scene = _snapshot(positions)
    scene.brick_quat = quat.expand(len(parts_list), 4).clone()
    return orc, scene


def test_selection_takes_the_highest_clear_part_and_squares_the_jaws_to_its_grasp():
    orc = _oracle([BRICK, SMALL, BRICK])
    scene = _snapshot([[0.195, 0.0, 0.0048], [0.195, 0.0, 0.0144], [0.24, 0.06, 0.0048]])
    hover = _to_hover(orc, scene)
    assert orc.target_brick == 1 and orc.cycles == 1 and orc.cycle_parts == [1]
    assert orc.grasp.part == 1 and orc.grasp_width == pytest.approx(orc.grasp.width)
    assert hover.yaw == pytest.approx(orc.grasp.heading - math.pi / 2)
    assert hover.target_pos[2] == pytest.approx(0.0144 + orc.draw.hover_height)
    assert hover.half_height == pytest.approx(SMALL.size[2] / 2)


def test_release_and_stowed_read_the_destination_bin():
    orc = _oracle([BRICK], destination=3)
    target = orc._release_target(0)
    assert target[1] == pytest.approx(BL.bin_ys[3]) and target[2] == oracle.RELEASE_Z
    # The same release point as the six-brick oracle: a heap setback of 0.035 m opened the
    # jaws 8 mm past the matched ceiling at RELEASE (driven, 2026-09-18), so the backstop
    # wall alone catches the slide (layout.HEAP_BIN_BACKSTOP).
    assert target[0] == pytest.approx(BL.bin_x - oracle.RELEASE_X_SETBACK)
    # _stowed grades against the one heap bin (the MS6 z-centre fix), so it reads the same
    # centre regardless of orc.destination.
    assert orc._stowed(_snapshot([[0.0, 0.0, 0.0]], stowed=(0,))) == {0}
    assert orc._stowed(_snapshot([[0.0, 0.0, 0.0]])) == set()


def _push_phases(orc, scene):
    """Drive one push: PUSH_HOVER, PUSH_DESCEND, PUSH_SWEEP, back at HOVER."""
    goal = orc.next_goal(scene)
    assert goal.phase is oracle.Phase.PUSH_HOVER and orc.target_brick is None
    assert goal.gripper == -1.0 and goal.target_pos[2] == pytest.approx(heap_oracle.PUSH_HOVER_Z)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)
    descend = orc.next_goal(scene)
    assert descend.phase is oracle.Phase.PUSH_DESCEND
    assert descend.target_pos[2] == pytest.approx(heap_oracle.PUSH_Z) and descend.stall_steps > 0
    assert descend.half_height == pytest.approx(heap_oracle.PUSH_HALF_HEIGHT)
    orc.report(oracle.PhaseResult(outcome="stalled"), scene)
    sweep = orc.next_goal(scene)
    assert sweep.phase is oracle.Phase.PUSH_SWEEP and sweep.gripper == -1.0
    assert tuple(sweep.target_pos[:2].tolist()) == pytest.approx(orc.push.end)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.phase is oracle.Phase.HOVER and orc.push is None


def test_a_landing_spot_under_a_stack_falls_back_to_the_densest_cell_push():
    # A tall stack one matched jaw gap out on every closing axis of the target blocks
    # both jaws. The gap follows the grasp width now, so the distance does too.
    x, y = layout.HEAP_CENTER[0], 0.0
    d = TOWER_OFFSET
    orc = _oracle([BRICK, BRICK, BRICK, BRICK, BRICK])
    scene = _snapshot(
        [
            [x, y, 0.0048],
            [x + d, y, 0.0240],
            [x - d, y, 0.0240],
            [x, y + d, 0.0240],
            [x, y - d, 0.0240],
        ]
    )
    orc.failures = {}
    orc.sizes[1:] = torch.tensor([0.0158, 0.0158, 0.0480])  # four 48 mm towers
    goal = _to_hover(orc, scene)
    # The towers themselves rank first and clear, so the target is a tower, not a push.
    assert goal.phase is oracle.Phase.HOVER and orc.target_brick in (1, 2, 3, 4)


def _towered_brick():
    """One brick whose landing spots four 48 mm towers block on every closing axis.

    The towers sit in excluded cells, so they are not candidates themselves, and off the
    50 mm cell boundary so the +y tower does not share the target's own cell.
    """
    d = TOWER_OFFSET
    x, y = 0.19, -0.025
    towers = [[x + d, y], [x - d, y], [x, y + d], [x, y - d]]
    orc = _oracle([BRICK, BRICK, BRICK, BRICK, BRICK])
    orc.sizes[1:] = torch.tensor([0.0158, 0.0158, 0.0480])
    for t in towers:
        orc.failures[heap_oracle.HeapOracle.cell_of(t)] = heap_oracle.CELL_FAILURE_CAP
    scene = _snapshot([[x, y, 0.0048]] + [[tx, ty, 0.0240] for tx, ty in towers])
    return orc, scene


def test_a_blocked_scan_is_a_forced_pick_at_once_and_never_a_push():
    """MAX_PUSHES_IN_A_ROW is 0 on the bare desk: 20 of 21 forced picks stowed on the rA read
    while the pushes before them took a third of every episode's steps."""
    assert heap_oracle.MAX_PUSHES_IN_A_ROW == 0
    orc, scene = _towered_brick()
    hover = _to_hover(orc, scene)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick == 0
    assert orc.forced_picks == 1 and orc.pushes == 0 and orc.push is None
    assert orc.grasp.clearance < 0.0
    assert orc.extra_stats()["forced_picks"] == 1 and orc.extra_stats()["pushes"] == 0


def test_three_pushes_then_a_forced_pick_and_done_only_at_a_budget(monkeypatch):
    monkeypatch.setattr(heap_oracle, "MAX_PUSHES_IN_A_ROW", 3)
    orc, scene = _towered_brick()
    goal = _to_hover(orc, scene)
    assert goal.phase is oracle.Phase.PUSH_HOVER and orc.target_brick is None
    for _ in range(heap_oracle.MAX_PUSHES_IN_A_ROW):
        _push_phases(orc, scene)
    assert orc.pushes == heap_oracle.MAX_PUSHES_IN_A_ROW
    # The pushes moved nothing (stub scene). With no dish there is no wall to land past, so
    # the best grasp is taken anyway once the push cap is reached (fix F2).
    hover = orc.next_goal(scene)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick == 0 and orc.forced_picks == 1
    assert orc.grasp.clearance < 0.0
    # DONE comes from the push budget.
    orc2, scene2 = _towered_brick()
    orc2.pushes = heap_oracle.PUSH_BUDGET
    _to_hover(orc2, scene2)
    assert orc2.phase is oracle.Phase.DONE and orc2.next_goal(scene2) is None


def test_a_part_out_of_reach_is_counted_but_never_a_target():
    near = layout.SCATTER_MIN_RADIUS - 0.02
    far = layout.SCATTER_MAX_RADIUS + 0.01
    orc = _oracle([BRICK, BRICK, BRICK])
    scene = _snapshot([[near, 0.0, 0.0048], [0.19, 0.0, 0.0048], [far, 0.0, 0.0048]])
    _to_hover(orc, scene)
    assert orc.target_brick == 1 and orc.out_of_reach == 2
    alone = _oracle([BRICK])
    _to_hover(alone, _snapshot([[near, 0.0, 0.0048]]))
    assert alone.phase is oracle.Phase.DONE and alone.out_of_reach == 1


def test_a_lift_that_fails_after_the_retries_releases_before_the_next_cycle():
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.19, 0.0, 0.0048], [0.19, 0.04, 0.0048]])
    _to_hover(orc, scene)
    for _ in range(oracle.MAX_GRASP_RETRIES + 1):
        for phase in (oracle.Phase.HOVER, oracle.Phase.DESCEND, oracle.Phase.CLOSE):
            assert orc.phase is phase
            orc.report(oracle.PhaseResult(outcome="converged"), scene)
        assert orc.phase is oracle.Phase.LIFT
        # MS6 rework section 4: the gate reads the measured rise, not the pinch factors, so a
        # retry needs a lift that did not rise -- a failed pinch on a lift that did rise now
        # advances (see test_the_heap_gives_up_on_a_lift_that_did_not_rise_and_keeps_one_that_did).
        orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=False, lifted=False), scene)
        assert orc.phase is oracle.Phase.REOPEN
        orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.target_brick is None and orc.phase is oracle.Phase.HOVER
    assert orc.cycle_outcomes == [False]
    hover = orc.next_goal(scene)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick is not None


def test_the_heap_gives_up_on_a_lift_that_did_not_rise_and_keeps_one_that_did():
    """The heap carries its own copy of the gate; both copies read `lifted` only."""
    orc = _oracle([BRICK])
    scene = _snapshot([[0.19, 0.0, 0.0048]])
    _to_hover(orc, scene)
    for phase in (oracle.Phase.HOVER, oracle.Phase.DESCEND, oracle.Phase.CLOSE):
        assert orc.phase is phase
        orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.phase is oracle.Phase.LIFT
    orc.retries_used = oracle.MAX_GRASP_RETRIES
    orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=False, lifted=True), scene)
    assert orc.phase is oracle.Phase.RAISE
    assert orc.cycle_outcomes == []

    orc = _oracle([BRICK])
    scene = _snapshot([[0.19, 0.0, 0.0048]])
    _to_hover(orc, scene)
    for phase in (oracle.Phase.HOVER, oracle.Phase.DESCEND, oracle.Phase.CLOSE):
        assert orc.phase is phase
        orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.phase is oracle.Phase.LIFT
    orc.retries_used = oracle.MAX_GRASP_RETRIES
    orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=True, lifted=False), scene)
    assert orc.phase is oracle.Phase.REOPEN
    assert orc.cycle_outcomes == [False]


def test_push_heading_matches_the_old_uniform_draw_at_the_disc_center():
    # At the exact centre, radius == 0, so away == 0.0 (radius <= 1e-6) and bias == 0
    # (radius / HEAP_RADIUS), leaving heading == a plain uniform draw, exactly as before.
    orc = heap_oracle.HeapOracle(BL, [BRICK], 1, oracle.nominal_draw(), rng=random.Random(0))
    mirror = random.Random(0)
    for _ in range(20):
        assert orc._push_heading(layout.HEAP_CENTER) == pytest.approx(
            mirror.uniform(-math.pi, math.pi)
        )


def test_push_heading_never_points_out_of_the_disc_at_its_edge():
    at_the_rim = (layout.HEAP_CENTER[0] + layout.HEAP_RADIUS, layout.HEAP_CENTER[1])
    # radius == HEAP_RADIUS, so bias == 1 (max narrowing) and away == pi (points at the centre).
    orc = heap_oracle.HeapOracle(BL, [BRICK], 1, oracle.nominal_draw(), rng=random.Random(1))
    headings = [orc._push_heading(at_the_rim) for _ in range(200)]
    assert all(math.cos(h - math.pi) >= -1e-9 for h in headings)  # never > 90 deg from away (pi)
    assert len({round(h, 6) for h in headings}) > 1  # still random, not collapsed to one value


def test_the_cycle_budget_ends_the_episode_done_with_parts_left():
    orc = _oracle([BRICK, BRICK], budget=1)
    scene = _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]])
    _to_hover(orc, scene)
    for outcome, pinch, lifted in (
        ("converged", None, None),  # HOVER
        ("stalled", None, None),  # DESCEND
        ("converged", None, None),  # CLOSE
        ("converged", True, True),  # LIFT
        ("converged", None, None),  # RAISE
        ("converged", None, None),  # EXTEND
        ("converged", None, None),  # RELEASE
    ):
        orc.report(oracle.PhaseResult(outcome=outcome, pinch_ok=pinch, lifted=lifted), scene)
        orc.next_goal(scene)
    first = orc.target_brick
    orc.report(
        oracle.PhaseResult(outcome="converged"),
        _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]], stowed=(first,)),
    )
    assert orc.cycle_outcomes == [True]
    assert orc.next_goal(scene) is None and orc.phase is oracle.Phase.DONE


def test_a_lift_that_fails_after_the_retries_marks_the_cell_and_takes_a_new_cycle():
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]])
    _to_hover(orc, scene)
    first = orc.target_brick
    for attempt in range(oracle.MAX_GRASP_RETRIES + 1):
        orc.report(oracle.PhaseResult(outcome="converged"), scene)  # HOVER
        orc.next_goal(scene)
        orc.report(oracle.PhaseResult(outcome="stalled"), scene)  # DESCEND
        orc.next_goal(scene)
        orc.report(oracle.PhaseResult(outcome="converged"), scene)  # CLOSE
        orc.next_goal(scene)
        orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=False, lifted=False), scene)
        assert orc.phase is oracle.Phase.REOPEN  # a retry, or the release before moving on
        orc.next_goal(scene)
        orc.report(oracle.PhaseResult(outcome="budget"), scene)
        if attempt < oracle.MAX_GRASP_RETRIES:
            assert orc.phase is oracle.Phase.HOVER and orc.target_brick == first
    assert orc.phase is oracle.Phase.HOVER and orc.target_brick is None
    assert orc.cycle_outcomes == [False]
    cell = heap_oracle.HeapOracle.cell_of(scene.brick_pos[first, :2].tolist())
    assert orc.failures[cell] == 1
    hover = orc.next_goal(scene)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick is not None
    assert orc.cycles == 2 and len(orc.cycle_parts) == 2


def test_a_swapped_hold_carries_the_new_part_through_retreat():
    """The jaws missed the target and came up with the other part (what the runtime's LIFT
    gate reports through `adopt_held_part`). Everything the cycle records afterwards must be
    about the part that was carried: RETREAT grades IT, the cycle log names IT, and the
    target's cell is not marked failed -- a part did leave the heap."""
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]])
    _to_hover(orc, scene)
    target = orc.target_brick
    other = 1 - target
    for outcome in ("converged", "stalled", "converged"):  # HOVER, DESCEND, CLOSE
        orc.report(oracle.PhaseResult(outcome=outcome), scene)
        orc.next_goal(scene)
    assert orc.phase is oracle.Phase.LIFT and orc.cycle_parts == [target]

    orc.adopt_held_part(other)
    orc.report(oracle.PhaseResult(outcome="converged", pinch_ok=False, lifted=True), scene)

    assert orc.phase is oracle.Phase.RAISE and orc.target_brick == other
    for _ in range(3):  # RAISE, EXTEND, RELEASE
        orc.next_goal(scene)
        orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.phase is oracle.Phase.RETREAT
    orc.next_goal(scene)
    orc.report(
        oracle.PhaseResult(outcome="converged"),
        _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]], stowed=(other,)),
    )
    assert orc.cycle_outcomes == [True]
    assert orc.sorted_bricks == frozenset({other})
    assert orc.cycle_parts == [other]
    assert orc.held_swaps == 1 and orc.extra_stats()["held_swaps"] == 1
    assert orc.failures == {}  # nothing was dropped, so no cell is marked


def test_the_six_brick_oracle_never_accepts_another_part():
    """The flag the runtime's LIFT gate reads: off on the base class, on for a heap."""
    assert oracle.EnvOracle(BL, 6, oracle.nominal_draw()).accepts_any_held_part is False
    assert _oracle([BRICK]).accepts_any_held_part is True


def test_three_failures_exclude_a_cell_from_selection():
    orc = _oracle([BRICK])
    scene = _snapshot([[0.16, -0.045, 0.0048]])
    cell = heap_oracle.HeapOracle.cell_of((0.16, -0.045))
    orc.failures[cell] = heap_oracle.CELL_FAILURE_CAP
    _step(orc, scene)  # SETTLE
    assert orc.next_goal(scene) is None and orc.phase is oracle.Phase.DONE


def test_candidates_exclude_out_of_reach_lifted_and_stowed_parts():
    orc = _oracle([BRICK, BRICK, BRICK, BRICK], destination=1)
    scene = _snapshot(
        [[0.16, -0.05, 0.0048], [0.35, 0.0, 0.0048], [0.16, 0.05, 0.15], [0.0, 0.0, 0.0]],
        stowed=(3,),
    )
    mask = orc._candidates(scene)
    assert mask.tolist() == [True, False, False, False]
    assert orc.out_of_reach == 1  # the part at x = 0.35 lies past the reach band


def test_extra_stats_report_the_counters():
    orc = _oracle([BRICK])
    stats = orc.extra_stats()
    assert stats == {
        "cycles": 0,
        "pushes": 0,
        "tips": 0,
        "tips_ok": 0,
        "forced_picks": 0,
        "held_swaps": 0,
        "cycle_parts": [],
        "cycle_outcomes": [],
        "cells_excluded": 0,
        "parts_out_of_reach": 0,
    }


def test_a_flat_plate_is_tipped_and_a_brick_is_picked(monkeypatch):
    """A 3.2 mm plate grips 3.2 mm of face, under TIP_MIN_GRIP_HEIGHT. A brick grips 9.6."""
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    plate = parts.make_part("plate", 2, 4)
    brick = parts.make_part("brick", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    assert orc._next_brick(scene) is None
    assert orc.tip is not None and orc.tip.part == 0
    assert orc._no_target_phase(scene) is oracle.Phase.TIP_HOVER

    orc, scene = _heap_with_parts([brick], flat=True)
    assert orc._next_brick(scene) == 0
    assert orc.tip is None


def test_the_shipped_default_never_tips_a_flat_plate():
    """Pins TIP_MIN_GRIP_HEIGHT's shipped value (0.0): the trigger is unsatisfiable, so even
    a flat plate is picked, not tipped. Re-enabling the tip by mistake would fail this."""
    assert heap_oracle.TIP_MIN_GRIP_HEIGHT == 0.0
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    assert orc._next_brick(scene) == 0
    assert orc.tip is None


def test_the_tip_stroke_starts_beside_the_part_and_stays_in_reach(monkeypatch):
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    orc._next_brick(scene)
    start, end = orc.tip.start, orc.tip.end
    assert layout.in_reach(*start, BL)
    assert layout.in_reach(*end, BL)
    assert math.isclose(math.dist(start, end), heap_oracle.TIP_STROKE_M, rel_tol=0.2)


def test_standing_reads_the_short_axis_against_the_closing_band():
    plate = parts.make_part("plate", 2, 4)
    orc, flat_scene = _heap_with_parts([plate], flat=True)
    assert not orc.standing(flat_scene, 0)
    _, upright_scene = _heap_with_parts([plate], flat=False)  # rolled 90 degrees
    assert orc.standing(upright_scene, 0)


def test_standing_and_candidate_grasps_agree_at_a_non_zero_jaw_tilt():
    """MS6 rework section 7: standing() must read the same jaw plane candidate_grasps ranks
    against, or a tip could pass a jaw plane the pick itself would refuse.

    The plate's 3.2 mm thickness (its short axis) lies along world +x here: in the jaw plane
    when level, 0.342 = sin(20 deg) off it once the jaw plane tilts, over the 0.259 =
    sin(15 deg) = sin(AXIS_MAX_TILT_RAD) band both must refuse it by. World +y stays
    unaffected by an about-y tilt, so +x is the axis that would show a drift.
    """
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    scene.brick_quat = torch.tensor(quat_from_rpy_deg((0.0, 90.0, 0.0))).expand(1, 4).clone()
    tilt = math.radians(20.0)
    short_width = min(plate.size)
    size = torch.tensor(plate.size)

    assert orc.standing(scene, 0, tilt_rad=0.0)
    assert not orc.standing(scene, 0, tilt_rad=tilt)

    level = grasp.candidate_grasps(0, size, scene.brick_pos[0], scene.brick_quat[0], 0.0)
    tilted = grasp.candidate_grasps(0, size, scene.brick_pos[0], scene.brick_quat[0], tilt)
    assert any(g.width == pytest.approx(short_width) for g in level)
    assert not any(g.width == pytest.approx(short_width) for g in tilted)


def test_the_tip_phases_run_in_order_and_a_failed_tip_marks_the_cell(monkeypatch):
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    orc._next_brick(scene)
    orc.phase = oracle.Phase.TIP_HOVER
    for expected in (oracle.Phase.TIP_DESCEND, oracle.Phase.TIP_SWEEP, oracle.Phase.HOVER):
        orc.report(oracle.PhaseResult("converged"), scene)
        assert orc.phase is expected
    assert orc.tips == 1 and orc.tip is None
    assert orc.extra_stats()["tips"] == 1 and orc.extra_stats()["tips_ok"] == 0
    assert orc.failures  # the flat part did not stand up, so its cell took a failure


def _tip_phases(orc, scene):
    """Drive one tip end to end via next_goal/report: TIP_HOVER, TIP_DESCEND, TIP_SWEEP, HOVER."""
    hover = orc.next_goal(scene)
    assert hover.phase is oracle.Phase.TIP_HOVER and orc.target_brick is None
    assert hover.gripper == -1.0 and hover.target_pos[2] == pytest.approx(heap_oracle.TIP_HOVER_Z)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)
    descend = orc.next_goal(scene)
    assert descend.phase is oracle.Phase.TIP_DESCEND
    assert descend.target_pos[2] == pytest.approx(heap_oracle.TIP_Z) and descend.stall_steps > 0
    orc.report(oracle.PhaseResult(outcome="stalled"), scene)
    sweep = orc.next_goal(scene)
    assert sweep.phase is oracle.Phase.TIP_SWEEP and sweep.gripper == -1.0
    assert tuple(sweep.target_pos[:2].tolist()) == pytest.approx(orc.tip.end)
    orc.report(oracle.PhaseResult(outcome="converged"), scene)
    assert orc.phase is oracle.Phase.HOVER and orc.tip is None


def test_next_goal_drives_a_flat_part_through_a_tip_end_to_end(monkeypatch):
    """Correction 1: the widened HOVER guard must hand back the TIP_HOVER goal, not None.

    Before the guard widened, `HeapOracle.next_goal` saw a phase that was not PUSH_HOVER and
    returned None from the HOVER branch, which the rig reads as the episode ending instead of
    tipping.
    """
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    hover = _to_hover(orc, scene)
    assert hover is not None and hover.phase is oracle.Phase.TIP_HOVER
    _tip_phases(orc, scene)
    assert orc.tips == 1


def test_a_forced_pick_that_gets_tipped_does_not_inflate_forced_picks(monkeypatch):
    """Correction 2: a candidate reached through the forced-pick branch that turns out to be
    flat gets tipped, not picked, and `forced_picks` must not count a pick that never ran."""
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    x, y = 0.19, 0.0
    d = grasp.OPEN_HALF_GAP + 0.006
    towers = [[x + d, y], [x - d, y], [x, y + d], [x, y - d]]
    orc = _oracle([BRICK, BRICK, BRICK, BRICK, BRICK])
    orc.pushes_in_a_row = heap_oracle.MAX_PUSHES_IN_A_ROW
    # Only the footprint (x, y) sizes drive the towers' blocking; shrink the target's height
    # to a plate's so its best grasp grips less than TIP_MIN_GRIP_HEIGHT of face.
    orc.sizes[0] = torch.tensor(parts.make_part("plate", 2, 4).size)
    orc.sizes[1:] = torch.tensor([0.0158, 0.0158, 0.0480])
    for t in towers:
        orc.failures[heap_oracle.HeapOracle.cell_of(t)] = heap_oracle.CELL_FAILURE_CAP
    scene = _snapshot([[x, y, 0.0016]] + [[tx, ty, 0.0240] for tx, ty in towers])
    hover = _to_hover(orc, scene)
    assert hover is not None and hover.phase is oracle.Phase.TIP_HOVER
    assert orc.tip is not None and orc.tip.part == 0
    assert orc.forced_picks == 0


def test_select_target_is_the_teachers_rule_with_its_bookkeeping():
    orc = _oracle([BRICK, BRICK, BRICK])
    scene = _snapshot([[0.20, -0.05, 0.005], [0.20, 0.0, 0.005], [0.20, 0.05, 0.005]])
    target = orc.select_target(scene)
    assert target in (0, 1, 2)
    assert orc.cycles == 1 and orc.cycle_parts == [target]
    # The loop re-selects after a failed cycle: the same scene selects again and counts again.
    assert orc.select_target(scene) == target and orc.cycles == 2
    assert orc.target_brick is None  # selection does not enter the machine


def test_select_target_returns_none_at_the_cycle_budget():
    orc = _oracle([BRICK, BRICK], budget=1)
    scene = _snapshot([[0.20, -0.05, 0.005], [0.20, 0.05, 0.005]])
    assert orc.select_target(scene) is not None
    assert orc.select_target(scene) is None


def test_aim_hover_at_points_the_pending_hover_at_the_loops_target():
    orc = _oracle([BRICK, BRICK, BRICK])
    scene = _snapshot([[0.20, -0.05, 0.005], [0.20, 0.0, 0.005], [0.20, 0.05, 0.005]])
    orc.resume(scene, -1)
    assert orc.aim_hover_at(scene, 2) is True
    assert orc.phase is oracle.Phase.HOVER and orc.target_brick == 2
    assert orc.grasp is not None and orc.grasp.part == 2 and orc.pick_order == [2]
    assert orc.cycles == 0  # the loop counted the cycle at selection, not here
    goal = orc.next_goal(scene)
    assert goal.phase is oracle.Phase.HOVER
    assert goal.target_pos[:2].tolist() == pytest.approx(scene.brick_pos[2, :2].tolist(), abs=1e-6)


def test_aim_hover_at_declines_a_part_with_no_grasp():
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.20, -0.05, 0.005], [0.20, 0.05, 0.005]])
    # Corner up: every body axis is more than 15 degrees off the jaw plane, so no axis closes.
    scene.brick_quat[1] = torch.tensor(quat_from_rpy_deg((45.0, 45.0, 0.0)))
    orc.resume(scene, -1)
    assert orc.aim_hover_at(scene, 1) is False
    assert orc.target_brick is None
