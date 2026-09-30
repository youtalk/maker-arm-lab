"""MS7f track H (design section 5): the deep tray (H1), the refill and its parked parts (H1')
and the box emptying (H2), on the MuJoCo heap env, cameras off."""

import dataclasses
import math
import random
from collections import Counter

import mujoco
import pytest
import torch

from maker_arm_lab.core import batch_loop, desk_boxes, layout, loop, oracle, parts, perception
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.stowed import parts_resting_box
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_desk_boxes import FIXED
from test_perceived_loop import _env_and_rig as _loop_env
from test_perceived_loop import _rollout, _Still

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


def _fill_box(env, j: int, members: list[int]) -> None:
    """Stack ``members`` at rest in box ``j``'s interior."""
    m, d, h = env.models[0], env.datas[0], env.handles
    x, y, _ = env.desk_box_poses[0, j]
    for n, b in enumerate(members):
        adr = h.brick_qpos[b]
        dx = 0.02 * (n % 3 - 1)
        d.qpos[adr : adr + 7] = (x + dx, y, 0.02 + 0.02 * (n // 3), 1, 0, 0, 0)
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=480)
    env._refresh([0], images=False)


def test_an_emptied_boxs_parts_are_graded_once_at_the_emptying(monkeypatch):
    """H2: a box whose parts at rest fill its capacity is emptied between cycles. Each part is
    graded then, its own box or not, and parked; the next call finds the box empty and grades
    nothing again. A box under its capacity stays. The capacity ratio is raised after the
    reset so three parts fill a box."""
    env = _env(20)
    env.reset(seed=2)
    dest = env.destination_bins[0].tolist()
    j = max(set(dest), key=dest.count)
    own = [b for b in range(20) if dest[b] == j][:2]
    other = [b for b in range(20) if dest[b] != j][:2]
    members, spare = own + other[:1], other[1]
    k = next(n for n in range(int(env.box_count[0])) if n != j)
    _fill_box(env, j, members)
    _fill_box(env, k, [spare])
    resting = [int(v) for v in parts_resting_box(env)[0]]
    assert [resting[b] for b in members] == [j] * 3 and resting[spare] == k
    load = sum(math.prod(env.parts[b].size) for b in members)
    ratio = math.prod(env.desk_box_sizes[0, j]) / load
    assert math.prod(env.parts[spare].size) < math.prod(env.desk_box_sizes[0, k]) / ratio
    monkeypatch.setattr(desk_boxes, "BOX_VOLUME_PER_PART_VOLUME", ratio)
    counts = Counter(tray_refill=0, box_empty=0)
    ok = torch.zeros(1, 20, dtype=torch.bool)
    box = torch.full((1, 20), -1, dtype=torch.long)
    assert loop.between_cycles(env, 0, counts, ok, box, True, False)
    assert counts == {"tray_refill": 0, "box_empty": 1}
    assert [b for b in range(20) if int(box[0, b]) >= 0] == sorted(members)
    assert all(int(box[0, b]) == j for b in members)
    assert [bool(ok[0, b]) for b in members] == [True, True, False]
    assert env.parked[0].nonzero().flatten().tolist() == sorted(members)
    assert [int(v) for v in parts_resting_box(env)[0]].count(k) == 1
    graded = ok.clone()
    assert not loop.between_cycles(env, 0, counts, ok, box, True, False)
    assert counts["box_empty"] == 1 and torch.equal(ok, graded)
    env.close()


def test_the_hidden_share_counts_the_covered_parts_in_the_tray():
    """MS7f design 5.5: of the parts in the tray, the share the two structural rules hide.
    A part under another, a part near the tool and one in the open: 2 of 3, and a part off the
    tray (a box, the park) is not in the count."""

    class _Env:
        tray_lean_deg = 30.0
        tray_wall_height = layout.DEEP_TRAY_WALL_HEIGHT
        parts = [parts.make_part("brick", 2, 4)] * 4

    x, y = layout.TRAY_CENTER
    pos = torch.tensor([[x, y, 0.005], [x, y, 0.015], [x - 0.04, y + 0.06, 0.005], [3.0, 0.3, 1.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    tool = torch.tensor([x - 0.04, y + 0.07, 0.12])
    assert loop.hidden_share(_Env(), pos, quat, tool) == pytest.approx(2 / 3)
    far = torch.tensor([0.5, 0.3, 0.12])
    assert loop.hidden_share(_Env(), pos, quat, far) == pytest.approx(1 / 3)


def test_a_stray_part_goes_back_into_the_tray_and_counts_one_intervention():
    """Stretch rule 2 (MS7f design 10.4): between cycles a part at rest on the desk outside the
    tray and every box goes back into the tray, one intervention per part; parts in the tray
    stay where they are."""
    env = _env(6)
    env.reset(seed=2)
    assert env.stray_parts(0) == []
    _take_out(env, 1)
    (stray,) = env.stray_parts(0)
    assert not env._in_tray(_pos(env, stray))
    strays = Counter(strays=0, miss_streak=0)
    ok = torch.zeros(1, 6, dtype=torch.bool)
    box = torch.full((1, 6), -1, dtype=torch.long)
    events = Counter(tray_refill=0, box_empty=0)
    assert loop.between_cycles(env, 0, events, ok, box, False, False, strays)
    assert strays == {"strays": 1, "miss_streak": 0} and env._in_tray(_pos(env, stray))
    assert env.stray_parts(0) == [] and env.tray_count(0) == 6
    assert not loop.between_cycles(env, 0, events, ok, box, False, False, strays)
    env.close()


def test_the_miss_streak_grows_on_a_close_with_no_new_stow_and_resets_on_one():
    assert loop.miss_streak(0, 0, 0) == (0, 1)
    assert loop.miss_streak(0, 0, 9) == (0, 10)
    assert loop.miss_streak(3, 2, 7) == (3, 0)


def test_ten_cycles_with_no_correct_stow_end_a_stretch_episode(monkeypatch):
    """Stretch rule 3: a driver that never stows (the arm held still, jaws open) closes every
    cycle on its budget; the tenth close in a row ends the episode ``miss_streak``, one
    intervention. Without ``stretch`` the episode runs on."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 20)
    five = [parts.make_part("brick", 2, 4)] * 5
    for stretch in (True, False):
        env, rig = _loop_env(part_list=five, budget=15)
        payload = _rollout(env, rig, _Still(), budget=15, stretch=stretch, part_failures=True)
        env.close()
        ep = payload["episodes"][0]
        if stretch:
            assert ep["done_reason"] == "miss_streak", ep["done_reason"]
            assert ep["cycles"] == loop.STRETCH_MISS_STREAK
            assert ep["interventions"] == {"strays": 0, "miss_streak": 1}
        else:
            assert ep["cycles"] > loop.STRETCH_MISS_STREAK and "interventions" not in ep


def test_the_recorder_rig_runs_the_events_between_its_own_cycles(monkeypatch):
    """The teacher's own run (the recorder, ``batch_loop.run_batch``): with the events
    installed, the rig refills the tray and empties full boxes when a RETREAT ends, books both
    in the episode's "events", counts the parts an emptying graded correct as delivered, and
    the episode succeeds once every part is stowed or emptied correct. Two of four parts start
    in the tray and one part fills a box."""
    monkeypatch.setattr(layout, "TRAY_REFILL_START", 2)
    monkeypatch.setattr(layout, "TRAY_REFILL_AT", 1)
    four = [parts.make_part("brick", 2, 4), parts.make_part("brick", 2, 2)] * 2
    env = MjLegoSortEnv(
        1,
        _draw(),
        parts=four,
        cameras=False,
        tray=True,
        tray_lean_deg=30.0,
        desk_boxes=4,
        gripper_mode="continuous",
        tray_refill=True,
        episode_length_s=240.0,
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, FIXED)
    monkeypatch.setattr(desk_boxes, "BOX_VOLUME_PER_PART_VOLUME", 1e4)

    def factory(i, bl, draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            draw,
            rng=random.Random(1),
            boxes=env.desk_box_poses[i],
            box_sizes=env.desk_box_sizes[i],
            tray=True,
            tray_lean_deg=30.0,
        )

    rig = OracleRig(
        env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1), oracle_factory=factory
    )
    batch_loop.install_heap_events(env, rig, box_empty=True)
    ep = batch_loop.run_batch(env, rig, 1, max_wall_minutes=10.0)["episodes"][0]
    assert ep["events"]["tray_refill"] >= 1 and ep["events"]["box_empty"] >= 1, ep
    assert ep["succeeded"] and ep["bricks_delivered"] == 4, ep
