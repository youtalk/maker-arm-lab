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
    """A heap oracle plus a scene with each part on the desk at the tray center.

    `flat=True` sits each part at the identity quaternion, so a plate lies with its 3.2 mm
    thickness straight up. `flat=False` rolls each part 90 degrees about its long axis, the
    way a tip stands a plate up onto its short edge.
    """
    orc = _oracle(parts_list)
    quat = IDENTITY if flat else torch.tensor(quat_from_rpy_deg((90.0, 0.0, 0.0)))
    positions = [
        [layout.TRAY_CENTER[0] + 0.03 * i, layout.TRAY_CENTER[1], p.size[2] / 2]
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


def test_a_wall_side_part_is_pushed_toward_the_tray_center_then_tipped(monkeypatch):
    # A long plate 22 mm off the rim, short side along y: its one closing axis lands a jaw
    # past the rim. 20 mm put the push start exactly on the dish's clamp radius (TRAY_RADIUS
    # - TIP_HALF_FACE), where float32 scene positions tip the comparison either way; 22 mm
    # keeps it clear of that boundary.
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    orc = _oracle([LONG_PLATE])
    scene = _snapshot([[layout.TRAY_CENTER[0], y1 - 0.022, 0.0016]])
    goal = _to_hover(orc, scene)
    assert goal.phase is oracle.Phase.PUSH_HOVER and orc.push_parts == [0]
    cx, cy = layout.TRAY_CENTER
    start, end = orc.push.start, orc.push.end
    assert x0 < start[0] < x1 and y0 < start[1] < y1
    assert math.hypot(end[0] - cx, end[1] - cy) < math.hypot(start[0] - cx, start[1] - cy)
    _push_phases(orc, scene)
    assert orc.pushes == 1 and orc.pushes_in_a_row == 1
    moved = _snapshot([[layout.TRAY_CENTER[0], 0.0, 0.0016]])
    # A LONG_PLATE is 3.2 mm thick, under TIP_MIN_GRIP_HEIGHT, so the centered plate is
    # tipped, not picked directly (task 8: the teacher tips a flat part before it picks it).
    hover = orc.next_goal(moved)
    assert hover.phase is oracle.Phase.TIP_HOVER and orc.tip is not None and orc.tip.part == 0
    assert orc.forced_picks == 0
    # The tip is a corrective like the push, and counts against the same forced-pick
    # grace period: it raises pushes_in_a_row again once it completes.
    _tip_phases(orc, moved)
    assert orc.pushes_in_a_row == 2


def test_a_landing_spot_under_a_stack_falls_back_to_the_densest_cell_push():
    # A tall stack OPEN_HALF_GAP out on every closing axis of the target blocks both jaws.
    x, y = layout.TRAY_CENTER[0], 0.0
    d = grasp.OPEN_HALF_GAP + 0.006
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


def test_three_pushes_then_a_forced_pick_and_done_only_at_a_budget():
    (_x0, _x1), (_y0, y1) = layout.tray_bounds()
    orc = _oracle([LONG_PLATE, BRICK])
    # One plate wall-blocked on its only axis, one brick clear but in an excluded cell.
    scene = _snapshot([[layout.TRAY_CENTER[0], y1 - 0.020, 0.0016], [0.19, 0.0, 0.0048]])
    orc.failures[heap_oracle.HeapOracle.cell_of((0.19, 0.0))] = heap_oracle.CELL_FAILURE_CAP
    _to_hover(orc, scene)
    for _ in range(heap_oracle.MAX_PUSHES_IN_A_ROW):
        _push_phases(orc, scene)
    assert orc.pushes == heap_oracle.MAX_PUSHES_IN_A_ROW
    # The pushes moved nothing (stub scene), and the only candidate lands a jaw past the wall,
    # so no forced pick exists: the teacher keeps pushing instead of giving up.
    goal = orc.next_goal(scene)
    assert goal.phase is oracle.Phase.PUSH_HOVER and orc.forced_picks == 0
    # A candidate whose tips land inside the tray but whose landing spots towers block on
    # every closing axis becomes the forced pick after the push cap (the towers sit in
    # excluded cells, so they are not candidates themselves).
    orc.failures = {}
    orc.push = None
    orc.phase = oracle.Phase.HOVER
    orc.pushes_in_a_row = heap_oracle.MAX_PUSHES_IN_A_ROW
    d = grasp.OPEN_HALF_GAP + 0.006
    x, y = 0.19, 0.0
    towers = [[x + d, y], [x - d, y], [x, y + d], [x, y - d]]
    orc = _oracle([BRICK, BRICK, BRICK, BRICK, BRICK])
    orc.pushes_in_a_row = heap_oracle.MAX_PUSHES_IN_A_ROW
    orc.sizes[1:] = torch.tensor([0.0158, 0.0158, 0.0480])
    for t in towers:
        orc.failures[heap_oracle.HeapOracle.cell_of(t)] = heap_oracle.CELL_FAILURE_CAP
    blocked = _snapshot([[x, y, 0.0048]] + [[tx, ty, 0.0240] for tx, ty in towers])
    hover = _to_hover(orc, blocked)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick == 0 and orc.forced_picks == 1
    assert orc.grasp.clearance < 0.0 <= orc.grasp.wall
    # DONE comes from the push budget.
    orc2 = _oracle([LONG_PLATE])
    orc2.pushes = heap_oracle.PUSH_BUDGET
    wall_side = _snapshot([[layout.TRAY_CENTER[0], y1 - 0.020, 0.0016]])
    _to_hover(orc2, wall_side)
    assert orc2.phase is oracle.Phase.DONE and orc2.next_goal(wall_side) is None


def test_a_part_outside_the_tray_is_counted_but_never_a_target():
    (x0, _x1), _ = layout.tray_bounds()
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[x0 - 0.02, 0.0, 0.0048], [0.19, 0.0, 0.0048]])
    _to_hover(orc, scene)
    assert orc.target_brick == 1 and orc.outside_tray == 1
    alone = _oracle([BRICK])
    _to_hover(alone, _snapshot([[x0 - 0.02, 0.0, 0.0048]]))
    assert alone.phase is oracle.Phase.DONE and alone.outside_tray == 1


def test_a_part_outside_the_circle_is_counted_and_never_a_target():
    cx, cy = layout.TRAY_CENTER
    corner = (cx + layout.TRAY_RADIUS * 0.9, cy + layout.TRAY_RADIUS * 0.9)
    assert not layout.inside_tray(*corner)  # inside the old rectangle, outside the dish
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[cx, cy, 0.0048], [corner[0], corner[1], 0.0048]])
    _to_hover(orc, scene)
    assert orc.target_brick == 0 and orc.outside_tray == 1


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


def test_push_heading_matches_the_old_uniform_draw_at_the_tray_center():
    # At the exact centre, radius == 0, so away == 0.0 (radius <= 1e-6) and bias == 0
    # (radius / TRAY_RADIUS), leaving heading == a plain uniform draw, exactly as before.
    orc = heap_oracle.HeapOracle(BL, [BRICK], 1, oracle.nominal_draw(), rng=random.Random(0))
    mirror = random.Random(0)
    for _ in range(20):
        assert orc._push_heading(layout.TRAY_CENTER) == pytest.approx(
            mirror.uniform(-math.pi, math.pi)
        )


def test_push_heading_never_points_into_the_rim_it_sits_against():
    at_the_rim = (layout.TRAY_CENTER[0] + layout.TRAY_RADIUS, layout.TRAY_CENTER[1])
    # radius == TRAY_RADIUS, so bias == 1 (max narrowing) and away == pi (points at the centre).
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
    assert orc.outside_tray == 1  # the part at x = 0.35 lies outside the tray


def test_extra_stats_report_the_counters():
    orc = _oracle([BRICK])
    stats = orc.extra_stats()
    assert stats == {
        "cycles": 0,
        "pushes": 0,
        "tips": 0,
        "tips_ok": 0,
        "forced_picks": 0,
        "push_parts": [],
        "cycle_parts": [],
        "cycle_outcomes": [],
        "cells_excluded": 0,
        "parts_outside_tray": 0,
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


def test_the_tip_stroke_starts_beside_the_part_and_stays_in_the_dish(monkeypatch):
    monkeypatch.setattr(heap_oracle, "TIP_MIN_GRIP_HEIGHT", 0.005)
    plate = parts.make_part("plate", 2, 4)
    orc, scene = _heap_with_parts([plate], flat=True)
    orc._next_brick(scene)
    start, end = orc.tip.start, orc.tip.end
    assert layout.inside_tray(*start, margin=0.0)
    assert layout.inside_tray(*end, margin=0.0)
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
