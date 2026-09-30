"""MS7f track H (design section 5): the deep tray (H1), the refill and its parked parts (H1')
and the box emptying (H2), on the MuJoCo heap env, cameras off."""

import dataclasses
import math
import random

import mujoco
import pytest
import torch

from maker_arm_lab.core import layout, loop, parts, perception
from maker_arm_lab.mj.env import MjLegoSortEnv

FAMILIES = ("round", "ldraw", "flat")


def _draw():
    draw = layout.sample_run_draw(random.Random(4))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


def _heap(k: int, seed: int = 9) -> list[parts.Part]:
    train, heldout = parts.split(parts.catalog(FAMILIES), 20260915)
    return parts.sample_parts(random.Random(seed), train + heldout, k)


def _env(k: int, **kwargs) -> MjLegoSortEnv:
    """The MS7e box and tray configuration (B2, the 30 degree tray) with ``k`` catalog parts."""
    return MjLegoSortEnv(
        1,
        _draw(),
        num_bricks=k,
        parts=_heap(k),
        cameras=False,
        tray=True,
        tray_lean_deg=30.0,
        desk_boxes=8,
        box_draw=8,
        release_lean=True,
        class_names=parts.PART_CLASSES,
        **kwargs,
    )


def _pos(env, b: int) -> list[float]:
    return env.scene[f"brick_{b}"].data.root_pos_w[0].tolist()


def test_with_the_switches_off_the_reset_draws_what_ms7e_drew():
    """Neither switch on: the reset makes the draws of the MS7e env (29a36ef) in its order, so
    the env's generator stands where it stood there (read off 29a36ef, 2026-09-30)."""
    env = _env(20)
    env.reset(seed=2)
    assert env.tray_wall_height == layout.TRAY_WALL_HEIGHT
    assert not env.parked.any() and env.pool == [[]]
    assert env._rng.random() == 0.06683939545281614
    env.close()


def test_a_settled_150_part_heap_stays_inside_the_deep_tray():
    """H1: the walls stand DEEP_TRAY_WALL_HEIGHT high, and a 150-part catalog heap, stacked in
    over its own pile, settles inside the tray: every centre over the rim footprint, none more
    than 2 cm over the rim (the top of the mound), and every part at rest."""
    env = _env(150, deep_tray=True)
    env.reset(seed=2)
    m = env.models[0]
    wall = m.geom("tray_wall_left").id
    assert math.isclose(2 * m.geom_size[wall][2] * math.cos(math.radians(30.0)), 0.045)
    assert env.tray_count(0) == 150 and not env.parked.any()
    grow = layout.tray_rim_growth(30.0, layout.DEEP_TRAY_WALL_HEIGHT)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    for b in range(150):
        x, y, z = _pos(env, b)
        assert x0 - grow <= x <= x1 + grow and y0 - grow <= y <= y1 + grow, (b, x, y)
        assert z < layout.DEEP_TRAY_WALL_HEIGHT + 0.02, (b, z)
        assert env.scene[f"brick_{b}"].data.root_lin_vel_w[0].norm() < 0.05, b
    env.close()


def test_the_deep_tray_and_the_refill_need_the_tray():
    for name in ("deep_tray", "tray_refill"):
        with pytest.raises(ValueError, match="tray"):
            MjLegoSortEnv(1, _draw(), parts=_heap(3), cameras=False, **{name: True})


def _take_out(env, count: int) -> None:
    """Stand in for cycles: move ``count`` of the parts in the tray off it, onto the desk far
    from the tray, as a stow would, and settle."""
    m, d, h = env.models[0], env.datas[0], env.handles
    inside = [b for b in range(env.cfg.num_bricks) if env._in_tray(d.xpos[h.brick_bodies[b]])]
    for n, b in enumerate(inside[:count]):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 7] = (0.40 + 0.02 * (n % 10), -0.35 + 0.07 * (n // 10), 0.02, 1, 0, 0, 0)
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=60)
    env._refresh([0], images=False)


def test_a_refill_keeps_the_tray_count_rule():
    """H1': TRAY_REFILL_START parts start in the tray and the rest wait parked. No refill while
    more than TRAY_REFILL_AT are in; at TRAY_REFILL_AT the next waiting parts drop in until
    TRAY_REFILL_START are in, and the last refill stops when none wait."""
    start, at = layout.TRAY_REFILL_START, layout.TRAY_REFILL_AT
    k = start + (start - at) + 5
    env = _env(k, tray_refill=True)
    env.reset(seed=2)
    assert env.tray_count(0) == start and int(env.parked.sum()) == k - start
    assert env.pool[0] == list(range(start, k))
    _take_out(env, start - at - 1)
    assert env.tray_count(0) == at + 1 and env.refill_tray(0) == 0
    _take_out(env, 1)
    assert env.refill_tray(0) == start - at
    assert env.tray_count(0) == start and env.pool[0] == list(range(k - 5, k))
    assert int(env.parked.sum()) == 5
    _take_out(env, start - at)
    assert env.refill_tray(0) == 5 and env.pool[0] == [] and not env.parked.any()
    assert env.tray_count(0) == at + 5
    _take_out(env, 5)
    assert env.refill_tray(0) == 0
    env.close()


def test_a_parked_part_never_enters_the_perceived_state_or_the_tray_count():
    """A waiting part floats off the desk, touches nothing, is not counted in the tray and is
    never detected; it draws like any other part, so the other detections stay paired."""
    k = layout.TRAY_REFILL_START + 3
    env = _env(k, tray_refill=True)
    env.reset(seed=2)
    parked = list(range(layout.TRAY_REFILL_START, k))
    before = [_pos(env, b) for b in parked]
    for _ in range(5):
        env.step(torch.cat([env.scene["robot"].data.joint_pos[:, :6], torch.ones(1, 1)], dim=1))
    d, h = env.datas[0], env.handles
    for b, p in zip(parked, before):
        assert _pos(env, b) == pytest.approx(p, abs=1e-9) and p[2] > 0.5
        for c in range(d.ncon):
            assert h.brick_bodies[b] not in (
                env.models[0].geom_bodyid[d.contact[c].geom1],
                env.models[0].geom_bodyid[d.contact[c].geom2],
            )
    assert env.tray_count(0) == layout.TRAY_REFILL_START

    class _Rig:
        def pick_point(self, i):
            return torch.tensor([0.5, 0.3, 0.2])

    truth = loop.perception_truth(env, _Rig(), 0)
    assert truth.parked == frozenset(parked)
    scales = perception.parse_scales("miss=0")
    seen = perception.Perceiver(1, 0, scales, parts.PART_CLASSES).perceive(truth)
    free = perception.Perceiver(1, 0, scales, parts.PART_CLASSES).perceive(
        dataclasses.replace(truth, parked=frozenset())
    )
    ids = {det.sim_index for det in seen.detections}
    assert not ids & set(parked)
    assert [det for det in free.detections if det.sim_index not in parked] == seen.detections
    env.close()
