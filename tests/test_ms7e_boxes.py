"""MS7e box draw (design sections 4.2, 4.3 and 10.2, the boxes row): each episode's count,
interiors and poses in the env, the per-box grader, L1 and L9, the perceived box size, the
resized floor and the switches."""

import importlib.util
import math
import pathlib
import random
import sys

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import batch_loop, desk_boxes, parts, perception, stowed
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import BOX_PARK_X, HEAP_PARK_X, MjLegoSortEnv
from test_desk_boxes import _in_box, _stub_env
from test_mj_heap_oracle import PARTS

W = desk_boxes.DESK_BOX_WALL


def _env(num_envs=1, **kwargs):
    draw = sample_run_draw(random.Random(5))
    kwargs = {"desk_boxes": 4, "tray": True, "cameras": False, **kwargs}
    return MjLegoSortEnv(num_envs, draw, parts=PARTS, **kwargs)


def _floor_size(env, i, j):
    m = env.models[i]
    return 2 * m.geom_size[m.geom(f"desk_box_{j}_floor").id]


def test_the_box_draw_draws_each_episodes_count_sizes_and_poses_from_its_seed():
    env = _env(2, box_draw=6, class_names=parts.PART_CLASSES, episode_draws=True)
    assert env.desk_boxes == desk_boxes.DESK_BOX_SLOTS == 8
    counts = set()
    for seed in (3, 4, 5, 6, 7, 8):
        env.reset(seed=seed)
        first = (env.box_count.copy(), env.desk_box_poses.copy(), env.desk_box_sizes.copy())
        for i in range(2):
            n = int(env.box_count[i])
            counts.add(n)
            assert desk_boxes.BOX_COUNT_MIN <= n <= 6
            poses, sizes = env.desk_box_poses[i, :n], env.desk_box_sizes[i, :n]
            assert desk_boxes.clear(poses, sizes)
            (x0, x1), (y0, y1) = desk_boxes.B1_BAND_X, desk_boxes.B1_BAND_Y
            for (x, y, _), (ix, iy, iz), j in zip(poses, sizes, range(n)):
                assert x0 <= x <= x1 and y0 <= abs(y) <= y1
                assert desk_boxes.BOX_MIN <= min(ix, iy) and max(ix, iy) <= desk_boxes.BOX_MAX
                assert desk_boxes.BOX_DEPTH[0] <= iz <= desk_boxes.BOX_DEPTH[1]
                # The scene's geoms carry the drawn interior.
                assert _floor_size(env, i, j)[:2] == pytest.approx((ix + 2 * W, iy + 2 * W))
            assert (env.desk_box_poses[i, n:, 0] == BOX_PARK_X).all()  # the rest wait off it
            assert set(env.class_to_box[i]) == set(range(n))  # eight classes fold onto n
            assert int(env.destination_bins[i].max()) < n
        env.reset(seed=seed)
        again = (env.box_count, env.desk_box_poses, env.desk_box_sizes)
        assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert counts == {4, 5, 6}


def test_with_the_draw_off_the_env_draws_what_the_base_drew():
    """Pinned on feat/ms7e-base 86263d6 (MS7d's draws), 2026-09-29: the same boxes, class
    maps and destinations for the same seed, every box the MS7a interior."""
    env = _env(2, episode_draws=True)
    env.reset(seed=3)
    assert env.class_to_box.tolist() == [[2, 1, 3, 0], [2, 1, 0, 3]]
    assert env.destination_bins.tolist() == [[0, 0, 3], [3, 3, 0]]
    assert env.desk_box_poses[0, 0].tolist() == pytest.approx(
        [0.057190438959, 0.208737619098, -0.265536606493], abs=1e-12
    )
    assert (env.desk_box_sizes == desk_boxes.interior(4)).all() and (env.box_count == 4).all()


@pytest.mark.parametrize(
    "unseen, check",
    [
        ("size", lambda s: max(s[:2]) > desk_boxes.BOX_MAX or max(s[:2]) / min(s[:2]) > 1.4),
        ("rim", lambda s: 0.05 <= s[2] <= 0.06),
    ],
)
def test_the_unseen_rows_draw_only_boxes_training_never_drew(unseen, check):
    env = _env(box_draw=6, box_unseen=unseen)
    for seed in range(6):
        env.reset(seed=seed)
        n = int(env.box_count[0])
        assert n <= desk_boxes.UNSEEN_CEILINGS[unseen]
        assert all(check(s) for s in env.desk_box_sizes[0, :n])


def test_the_grader_reads_each_boxs_own_interior():
    """Two parts at the same offset from their box centres: the one in the 12 cm box is in, the
    one in the 7 cm box is not."""
    poses = [(0.08, 0.22, 0.7), (0.08, -0.22, -0.5)]
    sizes = [(0.12, 0.08, 0.05), (0.07, 0.08, 0.05)]
    floor = W + 0.005
    pos = torch.tensor([[_in_box(poses[0], 0.05, 0.0, floor), _in_box(poses[1], 0.05, 0.0, floor)]])
    env = _stub_env(pos, torch.tensor([[0, 1]]))
    env.desk_box_poses, env.desk_box_sizes = np.array([poses]), np.array([sizes])
    assert stowed.per_brick_stowed(env).tolist() == [[True, False]]
    assert stowed.parts_resting_box(env).tolist() == [[0, -1]]


def test_l1_and_l9_read_each_boxs_own_size():
    poses = [(0.10, 0.20, 0.0), (0.10, -0.20, 0.0)]
    sizes = [(0.10, 0.07, 0.05), (0.07, 0.07, 0.03)]
    # L1: 5 cm along x lies inside the 10 cm box's wall, outside the 7 cm box's footprint.
    assert desk_boxes.in_outer_footprint((0.15, 0.20), poses, sizes, margin=0.0)
    assert not desk_boxes.in_outer_footprint((0.15, -0.20), poses, sizes, margin=0.0)
    # L9: each box's +x wall stands at its own half interior, as tall as its own rim.
    walls = desk_boxes.box_walls(poses, sizes)
    for k, (ix, _, iz) in enumerate(sizes):
        px = walls[4 * k]
        assert min(p[0] for p in px.footprint) == pytest.approx(0.10 + ix / 2)
        assert px.top == pytest.approx(W + iz)


FLOOR_POINTS = [(0.0, 0.0), (0.052, 0.05), (-0.054, -0.052), (0.0, -0.055)]


@pytest.mark.parametrize("size", [(0.12, 0.12, 0.05), (0.07, 0.09, 0.03)])
def test_a_part_dropped_on_the_floor_of_a_resized_box_comes_to_rest_on_it(size):
    """MS7e design section 11: a resized box refits the bounds the collision pipeline caches.
    Grown from the compiled 9 cm box to 12 cm, a 1x1 brick dropped past the old floor's edge
    fell through to the desk (z 4.8 mm) with the geoms alone resized, 2026-09-29."""
    (part,) = parts.by_name(["brick_1x1"])
    env = MjLegoSortEnv(
        1,
        sample_run_draw(random.Random(1)),
        parts=[part, part],
        cameras=False,
        desk_color=False,
        physics_dr=False,
        desk_boxes=4,
    )
    env.reset(seed=1)
    pose = (0.12, 0.19, 0.3)
    env.place_desk_boxes(0, [pose], [size])
    m, d, h = env.models[0], env.datas[0], env.handles
    for u, v in FLOOR_POINTS:
        u = max(-1.0, min(1.0, u / (size[0] / 2 - 0.006))) * (size[0] / 2 - 0.006)
        v = max(-1.0, min(1.0, v / (size[1] / 2 - 0.006))) * (size[1] / 2 - 0.006)
        x, y, _ = _in_box(pose, u, v, 0.0)
        tilt = (math.cos(0.2), 0.3 * math.sin(0.2), 0.0, math.sin(0.2))
        for b, xyz in enumerate(((x, y, W + part.size[2] / 2 + 0.02), (HEAP_PARK_X, 0.0, 0.0))):
            adr = h.brick_qpos[b]
            d.qpos[adr : adr + 3] = xyz
            d.qpos[adr + 3 : adr + 7] = np.asarray(tilt) / np.linalg.norm(tilt)
            d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=480)
        rest = float(d.qpos[h.brick_qpos[0] + 2])
        assert rest == pytest.approx(W + part.size[2] / 2, abs=0.001), (u, v, rest)


def test_the_perceiver_draws_each_boxs_size_after_every_pose():
    poses = [(0.1, 0.2, 0.3), (0.12, -0.2, -0.1)]
    sizes = [(0.07, 0.09, 0.03), (0.1, 0.08, 0.05)]
    plain, sized = perception.Perceiver(7, 2), perception.Perceiver(7, 2)
    plain.reset_episode(poses)
    sized.reset_episode(poses, sizes)
    assert sized.box_poses == plain.box_poses  # the MS7d draws come first, unchanged
    assert plain.box_sizes == desk_boxes.interiors(2)
    for got, true in zip(sized.box_sizes, sizes):
        assert got != true and max(abs(a - b) for a, b in zip(got, true)) < 5 * 0.003
    exact = perception.Perceiver(7, 2, {"box": 0.0})
    exact.reset_episode(poses, sizes)
    assert exact.box_sizes == tuple(sizes)


BATCH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_generate_batch.py"


def _load_batch():
    spec = importlib.util.spec_from_file_location("mj_generate_batch", BATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEAP = ["--out_dir", "/tmp/x", "--heap_k", "3", "--tray", "--desk_boxes", "4"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--box_draw", "7"],  # past B1's ceiling
        ["--box_draw", "3"],
        ["--box_unseen", "rim"],  # needs the draw
    ],
)
def test_the_recorder_refuses_a_box_draw_outside_b1_with_its_fail_token(argv, capsys):
    batch = _load_batch()
    with pytest.raises(SystemExit) as exit_info:
        batch.main(HEAP + argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("BATCH_FAIL usage: "), out


def test_the_box_draw_is_built_and_reaches_every_client():
    assert "box_draw" not in batch_loop.UNBUILT_SWITCHES
    assert "box_unseen" not in batch_loop.UNBUILT_SWITCHES
    args = _load_batch().parse_args(HEAP + ["--box_draw", "6", "--box_unseen", "size"])
    assert (args.box_draw, args.box_unseen) == (6, "size")
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    for name in ("mj_eval_heap.py", "mj_generate_batch.py"):
        text = (root / name).read_text()
        assert "box_draw=args.box_draw" in text and "box_unseen=args.box_unseen" in text
    for name in (
        "mj_eval_heap.py",
        "mj_generate_batch.py",
        "mj_dagger_heap.py",
        "mj_inspect_heap.py",
        "mj_release_envelope_check.py",
    ):
        assert "box_sizes=env.desk_box_sizes[i]" in (root / name).read_text(), name


def test_the_recorded_scene_keeps_the_drawn_boxes():
    env = _env(box_draw=6, class_names=parts.PART_CLASSES)
    env.reset(seed=3)
    state = env._initial_state(0)
    n = int(env.box_count[0])
    assert state["box_count"].tolist() == [n] and state["box_draw_rounds"].tolist() == [0]
    assert np.allclose(state["desk_box_sizes"], env.desk_box_sizes[0])
    assert "desk_box_sizes" not in _env()._initial_state(0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
