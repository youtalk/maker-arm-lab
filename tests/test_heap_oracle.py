"""The heap teacher on stub snapshots (design section 7)."""

import math
import random

import pytest
import torch

from maker_arm_lab.core import grasp, heap_oracle, layout, oracle, parts

BL = layout.DEFAULT_BIN_LAYOUT
BRICK = parts.make_part("brick", 2, 4)
SMALL = parts.make_part("brick", 1, 2)
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _snapshot(positions, stowed=(), destination=1):
    pos = torch.tensor(positions, dtype=torch.float32)
    for b in stowed:
        pos[b] = layout.bin_interior_centers(BL)[destination]
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
    assert orc._stowed(_snapshot([[0.0, 0.0, 0.0]], stowed=(0,), destination=3)) == {0}
    assert orc._stowed(_snapshot([[0.0, 0.0, 0.0]], stowed=(0,), destination=0)) == set()


def test_a_blocked_heap_pushes_through_the_densest_cell_then_selects_again():
    # Two bricks side by side, touching along y: every closing axis has a neighbor in a jaw box.
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.20, -0.006, 0.0048], [0.20, 0.006, 0.0048]])
    goal = _to_hover(orc, scene)
    assert goal.phase is oracle.Phase.PUSH_HOVER and orc.target_brick is None
    assert goal.gripper == -1.0 and goal.target_pos[2] == pytest.approx(heap_oracle.PUSH_HOVER_Z)
    center = grasp.densest_cell(scene.brick_pos[:, :2])
    start, end = orc.push.start, orc.push.end
    assert math.hypot(start[0] - center[0], start[1] - center[1]) == pytest.approx(
        grasp.PUSH_HALF_LENGTH_M
    )
    assert math.hypot(end[0] - start[0], end[1] - start[1]) == pytest.approx(
        2 * grasp.PUSH_HALF_LENGTH_M
    )
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
    assert orc.phase is oracle.Phase.HOVER and orc.pushes == 1 and orc.push is None
    apart = _snapshot([[0.185, -0.045, 0.0048], [0.205, 0.045, 0.0048]])
    hover = orc.next_goal(apart)
    assert hover.phase is oracle.Phase.HOVER and orc.target_brick in (0, 1)
    assert orc.pushes_in_a_row == 0


def test_push_heading_matches_the_old_uniform_draw_at_the_tray_center():
    # At TRAY_CENTER, x and y are both exactly half the (smaller) span from their walls, so
    # bias == 0 and away == 0.0 (the x0 candidate), leaving heading == a plain uniform draw.
    orc = heap_oracle.HeapOracle(BL, [BRICK], 1, oracle.nominal_draw(), rng=random.Random(0))
    mirror = random.Random(0)
    for _ in range(20):
        assert orc._push_heading(layout.TRAY_CENTER) == pytest.approx(
            mirror.uniform(-math.pi, math.pi)
        )


def test_push_heading_never_points_into_a_wall_it_sits_against():
    (x0, _), _ = layout.tray_bounds()
    at_the_wall = (x0, layout.TRAY_CENTER[1])  # nearest_dist == 0, so bias == 1 (max narrowing)
    orc = heap_oracle.HeapOracle(BL, [BRICK], 1, oracle.nominal_draw(), rng=random.Random(1))
    headings = [orc._push_heading(at_the_wall) for _ in range(200)]
    assert all(math.cos(h) >= -1e-9 for h in headings)  # never more than 90 deg from away (0.0)
    assert len({round(h, 6) for h in headings}) > 1  # still random, not collapsed to one value


def test_three_pushes_in_a_row_end_the_episode_done():
    orc = _oracle([BRICK, BRICK])
    scene = _snapshot([[0.20, -0.006, 0.0048], [0.20, 0.006, 0.0048]])
    _to_hover(orc, scene)
    for _ in range(3):
        for _ in range(3):  # PUSH_HOVER, PUSH_DESCEND, PUSH_SWEEP
            orc.report(oracle.PhaseResult(outcome="converged"), scene)
            orc.next_goal(scene)
    assert orc.pushes == 3
    assert orc.phase is oracle.Phase.DONE and orc.next_goal(scene) is None


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
        if attempt < oracle.MAX_GRASP_RETRIES:
            assert orc.phase is oracle.Phase.REOPEN
            orc.next_goal(scene)
            orc.report(oracle.PhaseResult(outcome="budget"), scene)
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
        "cycle_parts": [],
        "cycle_outcomes": [],
        "cells_excluded": 0,
        "parts_outside_tray": 0,
    }
