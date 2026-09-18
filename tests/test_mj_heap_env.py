"""Heap mode of MjLegoSortEnv and the recorder's heap datasets (design sections 6, 8, 9)."""

import json
import random

import h5py
import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import layout, parts
from maker_arm_lab.core.heap_oracle import STOW_HALF_EXTENTS
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import overlay
from maker_arm_lab.mj.context_names import CONTEXT_BOX_NAMES, CONTEXT_DIMS
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.mj.recorder import ShardRecorder

PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("plate", 1, 2),
    parts.make_part("tile", 1, 1),
    parts.make_part("slope", 1, 2),
    parts.make_part("brick", 1, 1),
    parts.make_part("plate", 2, 2),
]


def _draw(seed: int = 5):
    draw = sample_run_draw(random.Random(seed))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


def _hold_action(env) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos[:, :6].clone().float()
    return torch.cat([q, torch.ones(env.num_envs, 1)], dim=1)


def _can_render() -> bool:
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(model, 8, 8).close()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_the_recorder_writes_the_heap_datasets(tmp_path):
    rec = ShardRecorder(tmp_path / "s.hdf5", 1, export_all=True, cameras=True, depth=True)
    state = {
        "robot_joint_position": np.zeros(8, np.float32),
        "robot_joint_velocity": np.zeros(8, np.float32),
        "robot_root_pose": np.zeros(7, np.float32),
        "robot_root_velocity": np.zeros(6, np.float32),
        "brick_root_pose": np.zeros((2, 7), np.float32),
        "brick_root_velocity": np.zeros((2, 6), np.float32),
        "bin_layout": np.zeros(5, np.float32),
        "destination_bin": np.array([2], np.int64),
    }
    rec.begin(0, state, parts=["brick_2x4", "tile_1x1"])
    blank = np.zeros((224, 224, 3), np.uint8)
    for _ in range(3):
        rec.pre_step(
            0,
            np.zeros(8),
            np.zeros(7),
            1,
            0,
            blank,
            blank,
            context_camera=np.zeros(6),
            context_privileged=np.zeros(66),
            actor=1,
            depth=np.full((224, 224), 900, np.uint16),
            target_part=0,
            context_box=np.array([0.44, -0.07, 0.0], np.float32),
        )
        rec.post_step(0, np.zeros(8), False)
    assert rec.end(0, False)
    rec.close()
    with h5py.File(tmp_path / "s.hdf5", "r") as f:
        g = f["data/demo_0"]
        assert g["obs/images/overhead_depth"].shape == (3, 224, 224)
        assert g["obs/images/overhead_depth"].dtype == np.uint16
        assert int(g["obs/images/overhead_depth"][0, 10, 10]) == 900
        assert g["obs/target_part"][:].tolist() == [0, 0, 0]
        assert g["obs/target_part"].dtype == np.int16
        assert g["obs/context_box"].shape == (3, 3)
        assert json.loads(g["obs/context_box"].attrs["names"]) == list(CONTEXT_BOX_NAMES)
        assert json.loads(g.attrs["parts"]) == ["brick_2x4", "tile_1x1"]
        assert g["initial_state/destination_bin"][:].tolist() == [2]
    assert CONTEXT_DIMS["camera_box"] == 9


@pytest.fixture
def heap():
    e = MjLegoSortEnv(2, _draw(), parts=PARTS, cameras=False, episode_length_s=2.0)
    e.reset(seed=11)
    yield e
    e.close()


def test_a_heap_reset_drops_the_parts_inside_the_tray_at_rest(heap):
    (x0, x1), (y0, y1) = layout.tray_bounds()
    for i in range(2):
        for b in range(6):
            p = heap.scene[f"brick_{b}"].data.root_pos_w[i]
            v = heap.scene[f"brick_{b}"].data.root_lin_vel_w[i]
            assert x0 - 0.02 <= p[0] <= x1 + 0.02 and y0 - 0.02 <= p[1] <= y1 + 0.02, (i, b, p)
            assert p[2] < 0.05 and v.norm() < 0.05
    assert heap.cfg.num_bricks == 6 and heap.parts == tuple(PARTS)
    assert heap.destination_bins.shape == (2,) and set(heap.destination_bins.tolist()) <= {
        0,
        1,
        2,
        3,
    }
    assert torch.allclose(heap.stow_half_extents, torch.tensor(STOW_HALF_EXTENTS))
    assert heap.target_part.tolist() == [-1, -1]
    assert "overhead_depth" not in heap.obs_buf["images"]  # cameras off


def test_every_dropped_part_starts_inside_the_dish():
    draw = layout.sample_run_draw(random.Random(3))
    draw["slot_colors"] = (0, 1, 2, 3)
    parts_list = parts.sample_parts(random.Random(7), parts.catalog(), 12)
    env = MjLegoSortEnv(1, draw, num_bricks=12, parts=parts_list, cameras=False)
    env.reset(seed=5)
    for b in range(12):
        x, y = env._brick_pos_np(0, b)[:2] if hasattr(env, "_brick_pos_np") else (None, None)
        if x is None:
            adr = env.handles.brick_qpos[b]
            x, y = env.datas[0].qpos[adr], env.datas[0].qpos[adr + 1]
        assert layout.inside_tray(float(x), float(y), margin=0.01)
    env.close()


def test_a_second_reset_keeps_the_one_destination_and_redraws_the_colors(heap):
    # MS6 rework section 6: heap mode has one bin, at destination 0, on every reset.
    seen_dest, seen_color = set(), set()
    for seed in range(12, 24):
        heap.reset(seed=seed)
        seen_dest.add(int(heap.destination_bins[0]))
        seen_color.add(tuple(heap.models[0].geom_rgba[heap.handles.brick_geoms[0]].round(2)))
    assert seen_dest == {0} and len(seen_color) > 1
    stud = heap.handles.body_geoms[0][1]
    assert (
        heap.models[0].geom_rgba[stud] == heap.models[0].geom_rgba[heap.handles.brick_geoms[0]]
    ).all()


def test_the_box_context_names_the_destination_bin(heap):
    for i in range(2):
        bl = heap.bin_layouts[i]
        dest = int(heap.destination_bins[i])
        assert heap.context_box[i].tolist() == pytest.approx([bl.bin_x, bl.bin_ys[dest], 0.0])
    e = MjLegoSortEnv(1, _draw(), parts=PARTS[:2], cameras=False, context="camera_box")
    e.reset(seed=1)
    ctx = e.obs_buf["context"].numpy()
    assert ctx.shape == (1, 9)
    assert np.allclose(ctx[0, :6], e.context_camera[0]) and np.allclose(
        ctx[0, 6:], e.context_box[0]
    )
    e.close()


def test_the_privileged_channel_is_refused_on_a_heap():
    with pytest.raises(ValueError, match="privileged"):
        MjLegoSortEnv(1, _draw(), parts=PARTS, cameras=False, context="privileged")


def test_success_grades_every_part_against_the_destination(heap):
    i = 0
    dest = int(heap.destination_bins[i])
    centers = layout.bin_interior_centers(heap.bin_layouts[i])
    h, d = heap.handles, heap.datas[i]
    for b, adr in enumerate(h.brick_qpos):
        z = -layout.BIN_INTERIOR[2] / 2 + 0.006 + 0.012 * b
        d.qpos[adr : adr + 3] = centers[dest].numpy() + np.array([0.0, 0.0, z])
        d.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(heap.models[i], d)
    mujoco.mj_step(heap.models[i], d, nstep=120)
    heap._refresh([i], images=False)
    _, _, terminated, _, _ = heap.step(_hold_action(heap))
    assert bool(terminated[0]) and not bool(terminated[1])


def test_the_legacy_env_has_no_heap_members():
    e = MjLegoSortEnv(1, _draw(), num_bricks=2, cameras=False)
    assert e.parts is None
    assert not hasattr(e, "destination_bins") and not hasattr(e, "target_part")
    e.close()


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_the_marker_lands_on_the_targets_projection_and_depth_is_rendered():
    e = MjLegoSortEnv(1, _draw(), parts=PARTS[:3], cameras=True, episode_length_s=1.0)
    e.reset(seed=3)
    before = e.obs_buf["images"]["overhead_rgb"][0].numpy().copy()
    e.target_part[0] = 1
    e._stamp_markers()
    d, h, c = e.datas[0], e.handles, e._cam_overhead
    u, v = overlay.project_point(
        d.cam_xpos[c],
        d.cam_xmat[c],
        float(e.models[0].cam_fovy[c]),
        224,
        224,
        d.xpos[h.brick_bodies[1]],
    )
    image = e.obs_buf["images"]["overhead_rgb"][0].numpy()
    assert tuple(image[round(v), round(u)]) == (255, 255, 255)
    assert (image != before).any()
    depth = e.obs_buf["images"]["overhead_depth"][0].numpy()
    assert depth.dtype == np.uint16 and depth.shape == (224, 224)
    # The arm's home pose intrudes into the overhead frame at about 268mm (measured across
    # seeds 0-9), so the lower bound guards against a units error, not against the arm being
    # in view. A depth image in centimetres would read about 27 here, and one in metres would
    # collapse onto the hole sentinel (0); 200 still catches both.
    assert depth.max() > 500 and depth[depth > 0].min() > 200
    e.close()


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_a_heap_shard_carries_the_new_datasets(tmp_path):
    e = MjLegoSortEnv(
        1,
        _draw(),
        parts=PARTS[:2],
        cameras=True,
        episode_length_s=1.0,
        export_dir=tmp_path,
        export_all=True,
    )
    e.reset(seed=3)
    e.oracle_cycle_phase = torch.tensor([1], dtype=torch.int8)
    e.oracle_cycle_brick = torch.tensor([1], dtype=torch.int8)
    action = _hold_action(e)
    for _ in range(30):
        e.step(action)
    assert int(e.target_part[0]) == 1
    assert e.recorder_manager.exported_episode_count == 1  # time out, exported by export_all
    e.close()
    with h5py.File(tmp_path / "shard.hdf5", "r") as f:
        g = f["data/demo_0"]
        assert g["obs/images/overhead_depth"].shape == (30, 224, 224)
        assert g["obs/images/overhead_depth"].dtype == np.uint16
        assert g["obs/target_part"][:].tolist() == [1] * 30
        assert g["obs/context_box"].shape == (30, 3)
        assert json.loads(g.attrs["parts"]) == ["brick_2x4", "plate_1x2"]
        assert g["initial_state/destination_bin"].shape == (1,)
        assert g["obs/images/overhead_rgb"][5].max() == 255  # the marker is in the frame


def test_heap_mode_builds_one_bin_and_grades_against_it():
    draw = layout.sample_run_draw(random.Random(4))
    draw["slot_colors"] = (0, 1, 2, 3)
    parts_list = parts.sample_parts(random.Random(9), parts.catalog(), 3)
    env = MjLegoSortEnv(1, draw, num_bricks=3, parts=parts_list, cameras=False)
    env.reset(seed=2)
    assert int(env.destination_bins[0]) == 0
    assert env.bin_layouts[0].bin_ys == (0.0, 0.0, 0.0, 0.0)
    names = [
        mujoco.mj_id2name(env.models[0], mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(env.models[0].ngeom)
    ]
    assert sum(1 for n in names if n and n.startswith("heap_bin_")) == 5
    assert not any(n and n.startswith("bin_") for n in names)
    env.close()
