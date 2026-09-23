"""MS7a desk boxes on the MuJoCo env: the scene, the per-part destination, the box context
row, the recorded scene and the teacher's release into its own box."""

import random

import h5py
import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, layout
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import MjLegoSortEnv
from test_mj_heap_oracle import PARTS, _place

FIXED = [(0.08, 0.20, 0.3), (0.10, -0.20, -0.5), (0.20, 0.21, 0.0), (0.20, -0.21, 0.6)]
"""Four box poses in the band, clear of the corridor and of one another."""


def _env(n=1, num_envs=1, **kwargs):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    return MjLegoSortEnv(num_envs, draw, parts=PARTS, cameras=False, desk_boxes=n, **kwargs)


def _hold(env):
    q = env.scene["robot"].data.joint_pos[:, :6].clone().float()
    return torch.cat([q, torch.ones(env.num_envs, 1)], dim=1)


def _box_floor_xy(env, i, j):
    m, d = env.models[i], env.datas[i]
    return d.geom_xpos[m.geom(f"desk_box_{j}_floor").id][:2]


@pytest.mark.parametrize("n", [1, 4])
def test_desk_boxes_replace_the_shelf_and_stand_at_the_drawn_poses(n):
    env = _env(n, num_envs=2, episode_draws=True)
    env.reset(seed=3)
    m = env.models[0]
    names = {m.geom(g).name for g in range(m.ngeom)}
    assert "shelf" not in names and "heap_bin_floor" not in names
    assert env.destination_bins.shape == (2, len(PARTS))
    assert env.destination_bins.tolist() == [desk_boxes.part_destinations(len(PARTS), n)] * 2
    assert env.desk_box_poses.shape == (2, n, 3)
    for i in range(2):
        assert desk_boxes.clear(env.desk_box_poses[i])
        for j in range(n):
            assert desk_boxes.in_band(env.desk_box_poses[i, j])
            assert np.allclose(_box_floor_xy(env, i, j), env.desk_box_poses[i, j, :2])
        # The heap drops on the tray footprint beside the boxes.
        (x0, x1), (y0, y1) = layout.tray_bounds()
        for b in range(len(PARTS)):
            x, y, _ = env.scene[f"brick_{b}"].data.root_pos_w[i].tolist()
            assert x0 - 0.03 <= x <= x1 + 0.03 and y0 - 0.03 <= y <= y1 + 0.03
    # Box 0 is served until the first target names its own.
    assert np.allclose(env.context_box[0], env.desk_box_poses[0, 0])
    for _ in range(3):
        env.step(_hold(env))
    assert not per_brick_stowed(env).any()
    env.close()


def test_a_seed_reproduces_the_boxes_and_the_heap():
    a, b = _env(4), _env(4)
    a.reset(seed=9)
    b.reset(seed=9)
    assert np.array_equal(a.desk_box_poses, b.desk_box_poses)
    for k in range(len(PARTS)):
        assert torch.equal(
            a.scene[f"brick_{k}"].data.root_pos_w, b.scene[f"brick_{k}"].data.root_pos_w
        )
    a.close()
    b.close()


def test_the_ms6_scene_keeps_its_shelf_bin_and_one_destination_per_env():
    env = _env(0)
    env.reset(seed=3)
    m = env.models[0]
    assert m.geom("shelf").id >= 0 and m.geom("heap_bin_floor").id >= 0 and m.nmocap == 0
    assert env.destination_bins.shape == (1,) and env.desk_box_poses.shape == (1, 0, 3)
    env.close()
    with pytest.raises(ValueError, match="desk_boxes"):
        draw = sample_run_draw(random.Random(5))
        MjLegoSortEnv(1, draw, num_bricks=6, cameras=False, desk_boxes=1)


def test_a_part_in_its_own_box_is_stowed_and_one_in_another_box_is_not():
    env = _env(4)
    env.reset(seed=3)
    env.place_desk_boxes(0, FIXED)
    floor = desk_boxes.DESK_BOX_WALL + 0.006
    # Parts 0, 1, 2 go to boxes 0, 1, 2. Part 0 lies in box 0, part 1 in box 0 too.
    _place(
        env,
        [
            (*FIXED[0][:2], floor, 0.0),
            (FIXED[0][0] + 0.02, FIXED[0][1], 0.03, 1.0),
            (0.19, 0.0, 0.006, 0.0),
        ],
    )
    for _ in range(30):
        env.step(_hold(env))
    assert per_brick_stowed(env)[0].tolist() == [True, False, False]
    env.close()


def test_the_served_box_follows_the_rigs_marker_part():
    env = _env(4, context="camera_box_target")
    env.reset(seed=3)
    env.place_desk_boxes(0, FIXED)
    for marker in (1, 2, 0):
        env.oracle_cycle_brick = torch.tensor([marker], dtype=torch.int8)
        env.oracle_marker_part = torch.tensor([marker], dtype=torch.int8)
        env.oracle_cycle_target = torch.zeros(1, 7)
        env.step(_hold(env))
        box = FIXED[int(env.destination_bins[0, marker])]
        assert np.allclose(env.context_box[0], box, atol=1e-6)
        assert np.allclose(env.context_camera_box[0, 6:], box, atol=1e-6)
        assert np.allclose(env.obs_buf["context"][0, 6:9].numpy(), box, atol=1e-6)
    env.close()


def test_the_recorded_initial_state_carries_the_boxes_and_each_parts_destination(tmp_path):
    env = _env(4, export_dir=tmp_path, export_all=True, episode_length_s=0.2)
    env.reset(seed=3)
    poses = env.desk_box_poses[0].copy()
    for _ in range(8):
        env.step(_hold(env))
    env.close()
    with h5py.File(tmp_path / "shard.hdf5") as f:
        init = f["data/demo_0/initial_state"]
        assert np.allclose(init["desk_box_poses"][0], poses, atol=1e-6)
        assert init["destination_bin"][()].tolist() == [0, 1, 2]
        assert f["data/demo_0/obs/context_box"].shape[1] == 3
