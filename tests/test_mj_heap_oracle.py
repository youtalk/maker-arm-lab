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


def _env_and_rig(seed: int = 11):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, parts=PARTS, cameras=False)
    env.reset(seed=seed)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng)

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    return env, rig


def _place(env, poses) -> None:
    """(x, y, z, yaw) per part on the tray floor, settled, and read back into the surface."""
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
    # y at -0.045, 0.0, +0.045, clear of each other and of the tray walls. The outer
    # two parts need the full 45 mm offset: at 40 mm the best reachable grasp on those
    # two parts reads 0.00287 m clearance, under CLEARANCE_MIN (0.003 m).
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
