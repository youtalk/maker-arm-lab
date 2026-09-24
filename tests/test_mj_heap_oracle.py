"""The whole heap teacher on the CPU MuJoCo env: three parts, real physics (design 12)."""

import math
import random

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import grasp, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.quat import quat_from_rpy_deg
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.robots import frames
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("brick", 2, 2),
    parts.make_part("brick", 1, 4),
]


def _env_and_rig(seed: int = 11, part_list=None, gripper_mode: str = "binary"):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=part_list or PARTS, cameras=False, gripper_mode=gripper_mode)
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


def _step(rig):
    """``rig.step()`` split in its two halves, so the test sees the one action it sent.

    ``actions()`` advances the plan, so calling it beside ``rig.step()`` would walk twice.
    """
    action = rig.actions()
    _, _, terminated, truncated, _ = rig.env.step(action)
    return action, rig.observe((terminated | truncated).cpu())


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
    floor = 0.5 * min(plate.size[0], plate.size[1]) - grasp.CLOSE_SQUEEZE_M
    assert tightest > floor - 0.002, (tightest, floor)
    assert any(lifts), lifts


def test_a_takeover_with_a_target_works_that_part_and_not_the_teachers_choice():
    env, rig = _env_and_rig()
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    rig.rearm([0], idle=True)
    assert rig.resume(0, target=2)
    orc = rig.oracles[0]
    assert orc.target_brick == 2 and orc.pick_order == [2] and orc.cycles == 0
    assert rig.goals[0].phase is oracle.Phase.HOVER
    hover = rig.goals[0].target_pos
    assert hover[:2].tolist() == pytest.approx(rig.snapshot(0).brick_pos[2, :2].tolist(), abs=0.02)
    env.close()


def test_a_takeover_with_a_target_that_has_no_grasp_is_declined():
    env, rig = _env_and_rig()
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    rig.rearm([0], idle=True)
    # Stand part 1 on a corner: no body axis lies within the closing band.
    m, d, h = env.models[0], env.datas[0], env.handles
    adr = h.brick_qpos[1]
    d.qpos[adr + 3 : adr + 7] = quat_from_rpy_deg((45.0, 45.0, 0.0))
    mujoco.mj_forward(m, d)
    env._refresh([0], images=False)
    assert rig.resume(0, target=1) is False
    assert rig.idle[0] is True and rig.takeover_result[0] is None
    env.close()


SPREAD = [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]


def test_the_continuous_rig_commands_the_matched_gap_without_a_flip():
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    commands = []
    for _ in range(4000):
        goal = rig.goals[0]
        approach = goal is not None and goal.phase in (oracle.Phase.HOVER, oracle.Phase.DESCEND)
        action, out = _step(rig)
        if approach:
            commands.append(float(action[0, 6]))
        if rig.oracles[0].phase in (oracle.Phase.CLOSE, oracle.Phase.LIFT) or bool(out["dones"][0]):
            break
    assert commands, "no HOVER or DESCEND frame was seen"
    width = rig.oracles[0].grasp_width
    expected = 2.0 * (grasp.landing_half_gap(width) - grasp.JAW_HOLD_BAND)
    # No flips: every command is the same set point, and it is a gap, not a sign.
    assert all(c == pytest.approx(expected) for c in commands), (commands[:5], expected)
    assert 0.02 < expected < frames.JAW_FULL_GAP
    jaw = min(rig.robot.data.joint_pos[0, rig.finger_ids].tolist())
    assert abs(2.0 * jaw - expected) <= 2.0 * grasp.JAW_HOLD_BAND + 0.004
    env.close()


def test_the_continuous_rig_closes_holds_and_releases_at_its_set_points():
    """Phase B design 5.8: the close and release gaps value by value, and a matched-gap hold
    that stays within JAW_HOLD_BAND of its set point for 100 steps with no command flip."""
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    held_phases = (oracle.Phase.LIFT, oracle.Phase.RAISE, oracle.Phase.EXTEND)
    closes, releases, held = [], [], []
    width = None
    for _ in range(4000):
        goal = rig.goals[0]
        action, out = _step(rig)
        if goal is None:
            continue
        if goal.phase is oracle.Phase.HOVER and releases:
            break  # the second cycle's approach: the first cycle is over
        width = float(rig.oracles[0].grasp_width)
        command = float(action[0, 6])
        if goal.phase is oracle.Phase.CLOSE:
            closes.append((command, width))
        elif goal.phase in held_phases:
            jaws = rig.robot.data.joint_pos[0, rig.finger_ids].tolist()
            held.append((command, jaws))
        elif goal.phase is oracle.Phase.RELEASE:
            releases.append((command, width))
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    assert closes and releases and len(held) >= 100, (len(closes), len(releases), len(held))
    for command, w in closes:
        assert command == pytest.approx(2.0 * max(0.0, 0.5 * w - grasp.CLOSE_SQUEEZE_M))
    for command, w in releases:
        assert command == pytest.approx(2.0 * (grasp.landing_half_gap(w) - grasp.JAW_HOLD_BAND))
    hold = held[:100]
    set_point = hold[0][0]
    assert all(command == set_point for command, _ in hold)
    worst = max(abs(j - 0.5 * set_point) for _, jaws in hold for j in jaws)
    assert worst <= grasp.JAW_HOLD_BAND, worst


def test_the_continuous_rig_stows_three_spread_parts():
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    _run(rig)
    stats = rig.episode_stats(0)
    env.close()
    assert stats["succeeded"] is True and stats["bricks_delivered"] == 3, stats


def test_the_binary_rig_still_flips_around_the_matched_opening():
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    commands = set()
    for _ in range(600):
        action, _ = _step(rig)
        commands.add(float(action[0, 6]))
        if rig.oracles[0].phase is oracle.Phase.CLOSE:
            break
    assert commands <= {1.0, -1.0} and len(commands) == 2
    env.close()


def test_the_continuous_rig_releases_on_a_level_jaw_plane_and_lets_go():
    """Phase B release rung: a position-held jaw opens smoothly, so a part resting on the lower
    jaw of a stood-up plane rides it down and stays. The continuous rig levels the jaw plane
    for RELEASE, and every part leaves the jaws before RETREAT."""
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    m, d = env.models[0], env.datas[0]
    left, right = m.body("gripper_left").id, m.body("gripper_right").id
    from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
        LEVEL_ROLL_MARGIN_RAD,
        ROLL_MAX_STEP_RAD,
        WRIST_ROLL_INDEX,
    )

    lo, hi = rig.wrist_roll_limits
    slopes, heights, wrist_steps, release_wrist, tool_moves = [], [], [], [], []
    for _ in range(9000):
        goal = rig.goals[0]
        target = rig.oracles[0].target_brick
        q_before = rig.q_cmd[0].clone()
        before = float(q_before[WRIST_ROLL_INDEX])
        out = rig.step()
        if goal is not None and goal.phase is oracle.Phase.RETREAT:
            wrist_steps.append(abs(float(rig.q_cmd[0][WRIST_ROLL_INDEX]) - before))
        turned = abs(float(rig.q_cmd[0][WRIST_ROLL_INDEX]) - before) > 0.01
        if (
            goal is not None
            and goal.phase in (oracle.Phase.EXTEND, oracle.Phase.RETREAT)
            and turned
        ):
            tool_moves.append(float((rig.ik.fk(rig.q_cmd[0])[0] - rig.ik.fk(q_before)[0]).norm()))
        if goal is not None and goal.phase is oracle.Phase.RELEASE:
            release_wrist.append(float(rig.q_cmd[0][WRIST_ROLL_INDEX]))
            axis = d.xpos[left] - d.xpos[right]
            slopes.append(abs(axis[2]) / float((axis**2).sum() ** 0.5))
            if rig.goals[0] is not None and rig.goals[0].phase is oracle.Phase.RETREAT:
                heights.append(float(rig._brick_pos(0, target)[2]))
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    stats = rig.episode_stats(0)
    env.close()
    assert slopes and max(slopes) < math.sin(math.radians(10.0)), max(slopes)
    # The release point is 0.30 m up; a part still in the jaws is within a jaw length of it.
    # The last part's stow ends the episode inside its RELEASE, so it has no RETREAT entry.
    assert len(heights) >= 2 and max(heights) < 0.25, heights
    assert stats["succeeded"] is True and stats["bricks_delivered"] == 3, stats
    # The released wrist sits off its stops, and RETREAT turns it back without a jump.
    assert min(release_wrist) >= lo + LEVEL_ROLL_MARGIN_RAD - 1e-9, (min(release_wrist), lo)
    assert max(release_wrist) <= hi - LEVEL_ROLL_MARGIN_RAD + 1e-9, (max(release_wrist), hi)
    assert wrist_steps and max(wrist_steps) <= ROLL_MAX_STEP_RAD, max(wrist_steps)
    # The roll rides the carry: the tool never stands still over the bin while the wrist turns.
    assert tool_moves and min(tool_moves) > 1e-4, min(tool_moves)


def test_the_row_and_marker_name_the_next_part_from_the_frame_release_ends():
    """The 2026-09-22 protocol lever on the rig: ``oracle_cycle_target`` and the marker switch
    on the first RETREAT frame, while ``oracle_cycle_brick`` (the accounting id) still names
    the part retreating, and the next HOVER carries that same target."""
    env, rig = _env_and_rig()
    _place(
        env, [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]
    )
    frames = []
    for _ in range(9000):
        action = rig.actions()
        pre = (
            int(env.oracle_cycle_phase[0]),
            int(env.oracle_cycle_brick[0]),
            int(env.oracle_marker_part[0]),
            env.oracle_cycle_target[0].numpy().copy(),
        )
        _, _, terminated, truncated, _ = env.step(action)
        frames.append((*pre, int(env.target_part[0]), env.context_target[0].copy()))
        out = rig.observe((terminated | truncated).cpu())
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    hover = oracle.PHASE_TO_INT["HOVER"]
    first_hover = next(t for t, f in enumerate(frames) if f[0] == hover)
    first = frames[first_hover][1]
    row = frames[first_hover][3]
    switch = next(
        t for t, f in enumerate(frames) if t > first_hover and not np.array_equal(f[3], row)
    )
    phase, brick, marker, row2, marked, served = frames[switch]
    assert phase == oracle.PHASE_TO_INT["RETREAT"], phase
    assert frames[switch - 1][0] == oracle.PHASE_TO_INT["RELEASE"]
    assert brick == first and marker != first and marked == marker
    assert np.array_equal(served, row2) and row2[5] > 0.0
    next_hover = next(t for t, f in enumerate(frames) if t > switch and f[0] == hover)
    assert frames[next_hover][1] == marker and frames[next_hover][2] == marker
    assert np.allclose(frames[next_hover][3][:3], row2[:3], atol=0.01)
    assert np.array_equal(frames[next_hover][3][3:], row2[3:])
