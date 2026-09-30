"""MS7e ledger R12 lever T1, the carry via (design section 10.2 style): under ``--carry_via`` an
EXTEND whose straight line would pass the arm base walks an arc about it; with it off, and on
every carry that clears the base, the plans are the straight ones."""

import itertools
import math
import random

import pytest

from maker_arm_lab.core import layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import HEAP_PARK_X, MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
    CARRY_MIN_RADIUS_M,
    OracleRig,
    base_clearance,
    carry_arc_point,
)
from test_mj_heap_oracle import _place

BRICK = parts.make_part("brick", 2, 4)
BACK = (-0.07, 0.19, 0.3)
"""A box at the back of the B2 band (ledger D7)."""
FRONT = (0.20, 0.19, 0.3)
"""A box at the front of the band: every carry to it clears the base."""
FAR = (0.15, -0.08)
"""A tray point on the side away from BACK: the straight carry passes 0.06 m from the base."""
PARKED = [(-0.6, 0.3 * j, 0.0) for j in (1, 2, 3)]


def _cycle(box, carry_via: bool) -> dict:
    """One lone-part cycle from FAR into ``box``, from SETTLE to the end of RETREAT, under the
    release lean. Records the phase and the commanded joint target of every step, and the
    radius of every EXTEND pick point the solver is asked for."""
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=[BRICK, BRICK],
        cameras=False,
        desk_boxes=4,
        gripper_mode="continuous",
        release_lean=True,
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, [box, *PARKED])
    env.destination_bins[0] = 0
    _place(env, [(*FAR, BRICK.size[2] / 2 + 0.001, 0.3), (HEAP_PARK_X, 0.0, 0.0, 0.0)])
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            boxes=env.desk_box_poses[i],
        )

    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        rng,
        oracle_factory=factory,
        release_lean=True,
        carry_via=carry_via,
    )
    asked = []
    target_for = rig._target_for

    def record(pick, *args, **kw):
        if rig.goals[0].phase == oracle.Phase.EXTEND:
            asked.append(math.hypot(float(pick[0]), float(pick[1])))
        return target_for(pick, *args, **kw)

    rig._target_for = record
    steps = []
    for _ in range(3000):
        phase = rig.goals[0].phase
        out = rig.step()
        steps.append((phase, rig.q_cmd[0].clone()))
        goal = rig.goals[0]
        if phase == oracle.Phase.RETREAT and (goal is None or goal.phase != phase):
            break
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    radii = [
        math.hypot(*rig._commanded_pick_pose(0, q)[0][:2].tolist())
        for phase, q in steps
        if phase == oracle.Phase.EXTEND
    ]
    result = {
        "steps": steps,
        "extend_min_radius": min(radii),
        "asked_min_radius": min(asked),
        "stowed": bool(per_brick_stowed(env)[0, 0]),
        "stats": rig.episode_stats(0),
    }
    env.close()
    return result


@pytest.fixture(scope="module")
def cycles():
    return {(box, via): _cycle(box, via) for box in (BACK, FRONT) for via in (False, True)}


def test_the_arc_keeps_its_end_radii_and_meets_both_ends():
    start, end = (FAR[0], FAR[1], 0.22), (BACK[0], BACK[1], 0.10)
    assert base_clearance(start, end) < 0.07
    for t in (0.0, 1.0):
        want = start if t == 0.0 else end
        assert carry_arc_point(start, end, t).tolist() == pytest.approx(list(want), abs=1e-12)
    floor = min(math.hypot(*start[:2]), math.hypot(*end[:2]))
    radii = [math.hypot(*carry_arc_point(start, end, k / 50)[:2].tolist()) for k in range(51)]
    assert min(radii) >= floor - 1e-12 and floor > CARRY_MIN_RADIUS_M
    # the short way round: the azimuth climbs from the pick's to the box's
    azimuths = [
        math.atan2(p[1], p[0])
        for p in (carry_arc_point(start, end, k / 50).tolist() for k in range(51))
    ]
    assert all(b > a for a, b in itertools.pairwise(azimuths))


def test_the_straight_carry_to_the_back_box_passes_the_base_and_the_arc_does_not(cycles):
    """Off, the solver is asked for the straight line past the base, and from this pick it
    has no walkable path: EXTEND goes unplanned and the part is not delivered. On, every
    point asked for and every command walked keeps CARRY_MIN_RADIUS_M, and the part lands."""
    off, on = cycles[(BACK, False)], cycles[(BACK, True)]
    assert off["asked_min_radius"] < 0.07
    assert off["stats"]["unplanned_legs"] == 1 and not off["stowed"], off["stats"]
    assert on["asked_min_radius"] >= CARRY_MIN_RADIUS_M
    assert on["extend_min_radius"] >= CARRY_MIN_RADIUS_M
    assert on["stowed"] and on["stats"]["unplanned_legs"] == 0, on["stats"]


def test_the_grasp_legs_do_not_change(cycles):
    """Every command before EXTEND, from SETTLE through RAISE, is the same with the switch on."""
    for box in (BACK, FRONT):
        before = []
        for via in (False, True):
            steps = cycles[(box, via)]["steps"]
            first = next(k for k, (phase, _) in enumerate(steps) if phase == oracle.Phase.EXTEND)
            before.append([(phase, q.tolist()) for phase, q in steps[:first]])
        assert {phase for phase, _ in before[1]} >= {oracle.Phase.DESCEND, oracle.Phase.LIFT}
        assert before[0] == before[1]


def test_a_carry_that_clears_the_base_is_straight_under_the_switch(cycles):
    """Off is unchanged by construction (the arc needs the switch); on, a carry that clears
    the base plans exactly the straight leg, every command of the cycle the same."""
    off, on = cycles[(FRONT, False)], cycles[(FRONT, True)]
    assert off["extend_min_radius"] >= CARRY_MIN_RADIUS_M
    assert [(p, q.tolist()) for p, q in off["steps"]] == [(p, q.tolist()) for p, q in on["steps"]]
