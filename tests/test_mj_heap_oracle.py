"""The whole heap teacher on the CPU MuJoCo env: three parts, real physics (design 12)."""

import math
import random

import mujoco
import pytest
import torch

from maker_arm_lab.core import oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("brick", 2, 2),
    parts.make_part("brick", 1, 4),
]


def _env_and_rig(seed: int = 11, part_list=None):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=part_list or PARTS, cameras=False)
    env.reset(seed=seed)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng)

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    return env, rig


def _place(env, poses) -> None:
    """(x, y, z, yaw) per part on the desk, settled, and read back into the surface."""
    m, d, h = env.models[0], env.datas[0], env.handles
    for b, (x, y, z, yaw) in enumerate(poses):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 3] = (x, y, z)
        d.qpos[adr + 3 : adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=120)
    env._refresh([0], images=False)


def _run(rig, budget: int = 9000) -> int:
    steps = 0
    while steps < budget:
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    return steps


def test_the_heap_teacher_stows_three_spread_parts():
    env, rig = _env_and_rig()
    # Spread in y inside the pickable band (owner ruling, 2026-09-16): x=0.195,
    # y at -0.045, 0.0, +0.045, clear of each other. The outer
    # two parts need the full 45 mm offset: at 40 mm the best reachable grasp on those
    # two parts reads 0.00287 m clearance, under a CLEARANCE_MIN of 0.003 m, the value of that time (it is 0.0 now).
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    steps = _run(rig)
    stats = rig.episode_stats(0)
    env.close()
    assert stats["succeeded"] is True, stats
    assert stats["bricks_delivered"] == 3 and stats["cycles"] == 3 and stats["pushes"] == 0
    assert stats["cycle_outcomes"] == [True, True, True]
    assert sorted(stats["cycle_parts"]) == [0, 1, 2]
    assert steps < 4500, steps


def test_the_heap_teacher_takes_the_top_of_a_stack_first():
    env, rig = _env_and_rig()
    # Owner ruling, 2026-09-16: the stack translates into the pickable band with its z
    # offsets unchanged, and the loose third part moves into the band alongside it.
    _place(
        env, [(0.195, -0.03, 0.0048, 0.0), (0.195, -0.03, 0.0149, 0.0), (0.195, 0.03, 0.0048, 0.5)]
    )
    _run(rig)
    stats = rig.episode_stats(0)
    env.close()
    assert stats["cycle_parts"][0] == 1, stats  # the 2x2 resting on the 2x4 goes first
    assert stats["succeeded"] is True and stats["bricks_delivered"] == 3


def test_a_takeover_on_a_heap_resumes_with_the_heap_teacher():
    env, rig = _env_and_rig()
    # Same layout as the spread-parts test, translated the same way (owner ruling, 2026-09-16):
    # the takeover only needs a part with a clear grasp, which the untranslated poses lacked.
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    rig.rearm([0], idle=True)
    assert rig.resume(0)
    assert isinstance(rig.oracles[0], HeapOracle)
    assert rig.oracles[0].destination == int(env.destination_bins[0])
    assert rig.goals[0].phase is oracle.Phase.HOVER
    env.close()


def test_the_snapshot_carries_the_part_quaternions_and_the_pinch_floor_follows_the_width():
    env, rig = _env_and_rig()
    snap = rig.snapshot(0)
    assert snap.brick_quat.shape == (3, 4)
    assert torch.allclose(snap.brick_quat.norm(dim=1), torch.ones(3), atol=1e-5)
    evidence = rig.pinch_evidence(0, 0, min_stall=0.001)
    assert set(evidence) == {"stall", "left", "right", "asymmetry", "ok"}
    env.close()


def test_the_default_factory_builds_the_six_brick_oracle():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    assert type(rig.oracles[0]) is oracle.EnvOracle
    assert rig.oracles[0].grasp_width == pytest.approx(0.0158)
    env.close()


def test_the_pick_approach_holds_the_grasp_width_not_the_full_opening():
    """MS6 rung 1: the jaws come down beside the part, not ``OPEN_HALF_GAP`` out from it.

    The fully open descend put both landing spots 52.4 mm out whatever the part, which
    ``core.grasp`` had to model as required clear floor: it refused 84.1 % of every in-dish
    part sighting on the three read heaps, 67.4 % of them at the rim alone. The jaws have to
    hold the matched opening for the ranking to describe the arm.
    """
    from maker_arm_lab.core import grasp

    env, rig = _env_and_rig()
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    widest = 0.0
    seen_descend = False
    for _ in range(4500):
        goal = rig.goals[0]
        rig.step()
        if goal is None or goal.phase is not oracle.Phase.DESCEND:
            continue
        seen_descend = True
        gap = min(rig.robot.data.joint_pos[0, rig.finger_ids].tolist())
        widest = max(widest, gap)
        if rig.oracles[0].target_brick is None:
            continue
        ceiling = grasp.landing_half_gap(float(rig.oracles[0].grasp_width))
        assert gap <= ceiling, (gap, ceiling)
    env.close()
    assert seen_descend, "the episode never descended on a part"
    assert widest < grasp.OPEN_HALF_GAP - 0.010, widest


def test_the_release_opens_only_far_enough_to_let_the_part_go():
    """MS6 rung 2: the jaws let the part drop, they do not fling it.

    RELEASE opened the jaws the whole way, about 44 mm of travel per jaw from a pinched
    grip at roughly 3.3 mm per control step, with the tool tilted 80.2 deg over the bin. On
    the three rung 1 read heaps 18 of 73 carried parts (25 %) never reached the bin, against
    7 of 56 (12 %) before, and every traced one landed on the DESK 60-200 mm SHORT of the
    release point: thrown back toward the arm, not dropped. One instrumented cycle left the
    jaws at (0.417, 0.002, 0.300) and came to rest at (0.246, 0.018, 0.230).

    Four mm of clearance per side is all the part needs to fall out.
    """
    from maker_arm_lab.core import grasp

    env, rig = _env_and_rig()
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    widest = 0.0
    seen_release = False
    for _ in range(4500):
        goal = rig.goals[0]
        rig.step()
        if goal is None or goal.phase is not oracle.Phase.RELEASE:
            continue
        seen_release = True
        ceiling = grasp.landing_half_gap(float(rig.oracles[0].grasp_width))
        gap = min(rig.robot.data.joint_pos[0, rig.finger_ids].tolist())
        widest = max(widest, gap)
        assert gap <= ceiling, (gap, ceiling)
    stats = rig.episode_stats(0)
    env.close()
    assert seen_release, "the episode never released a part"
    assert widest < grasp.OPEN_HALF_GAP - 0.010, widest
    # The part must still leave the jaws: a capped opening that never lets go is worse.
    assert stats["bricks_delivered"] == 3, stats


def test_a_lone_flat_part_survives_the_lift_instead_of_being_squeezed_out():
    """MS6 rung 3: CLOSE stops at the part's own width, it does not drive the jaws shut.

    The gripper channel is binary, so a close command runs the jaws all the way to
    ``GRIPPER_CLOSED_POS``. A 9.6 mm brick stalls them at 5.9 mm and is held. A 3.2 mm plate
    is not: traced per control step on a LONE plate_2x4 with nothing around it, CLOSE ends
    with 2.9 N on both jaws and the part in the grip, and then LIFT keeps closing --
    5.18, 4.81, 4.21, 3.70, 3.19, 2.69 mm -- while the force bleeds off 2.9, 1.0, 0.9, 0.33,
    0.00 N. The plate is squeezed out like a pip, the jaws end at 0.02 mm holding nothing,
    and the part is back on the desk. That is why the LIFT gate reads 0.00 N: it samples
    AFTER the part has gone.

    The pin is the LIFT itself: the plate rises in the jaws at least once. Whether the
    carry then delivers it is a plate's own RAISE weakness (docs/ms6-matched-jaw-opening.md
    section 5), which is why the flat families left the catalog on 2026-09-18.
    """
    from maker_arm_lab.core import grasp
    from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import CLOSE_SQUEEZE_M

    plate = parts.make_part("plate", 2, 4)
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=[plate], cameras=False)
    env.reset(seed=11)
    rng = random.Random(1)
    lifts: list[bool] = []

    class LiftLog(HeapOracle):
        def report(self, result, scene):
            if self.phase is oracle.Phase.LIFT:
                lifts.append(bool(result.lifted))
            super().report(result, scene)

    def factory(i, bl, episode_draw):
        return LiftLog(bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng)

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    _place(env, [(0.195, 0.0, 0.004, 0.0)])
    tightest = grasp.OPEN_HALF_GAP
    for _ in range(4000):
        goal = rig.goals[0]
        out = rig.step()
        if goal is not None and goal.gripper < 0.0:
            tightest = min(tightest, min(rig.robot.data.joint_pos[0, rig.finger_ids].tolist()))
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    floor = 0.5 * min(plate.size[0], plate.size[1]) - CLOSE_SQUEEZE_M
    assert tightest > floor - 0.002, (tightest, floor)
    assert any(lifts), lifts
