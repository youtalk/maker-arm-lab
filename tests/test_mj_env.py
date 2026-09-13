"""MjLegoSortEnv against the duck-typed surface the rig, the loops and the grader read.

Real MuJoCo on CPU with cameras off (a few hundred control steps run in well under a second),
plus one cameras-on shard round trip so the HDF5 layout is pinned against the converter.
"""

import random

import h5py
import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import layout
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.robots import frames


def _draw(seed: int = 5):
    draw = sample_run_draw(random.Random(seed))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


def _hold_action(env) -> torch.Tensor:
    """Command the current arm pose with the jaws open: a no-op action."""
    q = env.scene["robot"].data.joint_pos[:, :6].clone().float()
    return torch.cat([q, torch.ones(env.num_envs, 1)], dim=1)


def _teleport_all_bricks_into_bins(env, i: int) -> None:
    """Rest every brick on its own color's bin floor, then let the physics settle."""
    h, d = env.handles, env.datas[i]
    centers = layout.bin_interior_centers(env.bin_layouts[i])
    rest_z = -layout.BIN_INTERIOR[2] / 2 + layout.BRICK_SIZE[2] / 2 + 0.0005
    for b, adr in enumerate(h.brick_qpos):
        color = b % 4
        d.qpos[adr : adr + 3] = centers[color].numpy() + np.array([0.0, 0.0, rest_z])
        d.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(env.models[i], d)
    mujoco.mj_step(env.models[i], d, nstep=60)
    env._refresh([i], images=False)


@pytest.fixture
def env():
    e = MjLegoSortEnv(2, _draw(), num_bricks=4, cameras=False, episode_length_s=2.0)
    e.reset(seed=11)
    yield e
    e.close()


def test_surface_shapes(env):
    robot = env.scene["robot"]
    assert robot.data.joint_pos.shape == (2, 8) and robot.data.joint_pos.dtype == torch.float32
    assert robot.data.joint_pos_limits.shape == (2, 8, 2)
    assert robot.data.body_pos_w.shape[0] == 2 and robot.data.body_quat_w.shape[-1] == 4
    assert robot.find_bodies(frames.TOOL_BODY)[0][0] == env.handles.tool_body
    assert robot.find_joints(list(frames.ARM_JOINT_NAMES), preserve_order=True)[0] == list(range(6))
    assert env.scene["brick_0"].data.root_pos_w.shape == (2, 3)
    assert env.scene["finger_left_brick"].data.force_matrix_w.shape == (2, 1, 4, 3)
    assert env.action_manager.total_action_dim == 7
    assert env.action_manager.get_term("arm_action")._joint_names == list(frames.ARM_JOINT_NAMES)
    assert "success" in env.termination_manager.active_terms
    assert env.max_episode_length == 60
    assert env.recorder_manager.exported_successful_episode_count == 0


def test_reset_puts_the_arm_at_home_and_scatters_bricks_inside_the_band(env):
    q = env.scene["robot"].data.joint_pos.numpy()
    home = np.array([frames.HOME_JOINT_POS[n] for n in frames.ARM_JOINT_NAMES])
    assert np.abs(q[:, :6] - home).max() <= env.cfg.start_pose_offset_rad + 1e-6
    assert np.allclose(q[:, 6:], frames.GRIPPER_OPEN_POS)
    bl = env.cfg.bin_layout
    for i in range(2):
        for b in range(4):
            x, y, z = env.scene[f"brick_{b}"].data.root_pos_w[i].tolist()
            assert layout.SCATTER_X[0] - 1e-6 <= x <= layout.top_down_x_limit(bl) + 1e-6
            assert layout.SCATTER_Y[0] - 1e-6 <= y <= layout.SCATTER_Y[1] + 1e-6
            assert np.hypot(x, y) <= layout.SCATTER_MAX_RADIUS + 1e-6
            assert z == pytest.approx(layout.SPAWN_Z, abs=1e-6)
    # Two envs draw different scatters.
    assert not torch.allclose(
        env.scene["brick_0"].data.root_pos_w[0], env.scene["brick_0"].data.root_pos_w[1]
    )


def test_step_holds_the_pose_and_reports_no_clamp(env):
    action = _hold_action(env)
    before = env.scene["robot"].data.joint_pos[:, :6].clone()
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(action)
    assert not terminated.any() and not truncated.any()
    assert (env.scene["robot"].data.joint_pos[:, :6] - before).abs().max() < 0.01
    assert not env.action_manager.get_term("arm_action").clamped.any()
    assert per_brick_stowed(env).shape == (2, 4) and not per_brick_stowed(env).any()


def test_out_of_range_target_is_clamped_and_flagged(env):
    action = _hold_action(env)
    action[0, 0] = 10.0  # far past link_002_joint's +-2.8 rad stop
    env.step(action)
    flags = env.action_manager.get_term("arm_action").clamped
    assert bool(flags[0]) and not bool(flags[1])
    assert float(env.datas[0].ctrl[env.handles.arm_actuators[0]]) <= 2.8 + 1e-9


def test_success_terminates_and_auto_resets(env):
    _teleport_all_bricks_into_bins(env, 0)
    assert per_brick_stowed(env)[0].all()
    _, _, terminated, _truncated, _ = env.step(_hold_action(env))
    assert bool(terminated[0]) and not bool(terminated[1])
    # The success term keeps the terminating step's value while the state shows the next
    # episode: bricks are back on the desk and the step counter restarted.
    assert bool(env.termination_manager.get_term("success")[0])
    assert not per_brick_stowed(env)[0].any()
    assert env.episode_length_buf[0] == 0 and env.episode_length_buf[1] == 1


def test_time_out_truncates_at_the_episode_length(env):
    action = _hold_action(env)
    for _ in range(59):
        _, _, _, truncated, _ = env.step(action)
        assert not truncated.any()
    _, _, _, truncated, _ = env.step(action)
    assert truncated.all()
    assert (env.episode_length_buf == 0).all()


def test_reset_of_one_env_leaves_the_other_alone(env):
    env.step(_hold_action(env))
    other = env.scene["brick_0"].data.root_pos_w[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    assert torch.equal(env.scene["brick_0"].data.root_pos_w[1], other)
    assert env.episode_length_buf[0] == 0 and env.episode_length_buf[1] == 1


def _can_render() -> bool:
    """False where no GL context can be made (CI runs with MUJOCO_GL=disable)."""
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception:  # noqa: BLE001 -- any GL failure means "skip the render test"
        return False
    return True


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_shard_layout_matches_the_converter(tmp_path):
    env = MjLegoSortEnv(
        1, _draw(), num_bricks=2, cameras=True, episode_length_s=1.0, export_dir=tmp_path
    )
    env.reset(seed=3)
    env.oracle_cycle_phase = torch.tensor([1], dtype=torch.int8)
    env.oracle_cycle_brick = torch.tensor([0], dtype=torch.int8)
    action = _hold_action(env)
    for _ in range(5):
        env.step(action)
    _teleport_all_bricks_into_bins(env, 0)
    env.step(action)  # success -> the episode exports with 6 frames
    assert env.recorder_manager.exported_successful_episode_count == 1
    # A failed episode (time out at 30 steps) does not export.
    for _ in range(30):
        env.step(action)
    assert env.recorder_manager.exported_successful_episode_count == 1
    env.close()
    with h5py.File(tmp_path / "shard.hdf5", "r") as f:
        assert sorted(f["data"].keys()) == ["demo_0"]
        g = f["data/demo_0"]
        assert bool(g.attrs["success"]) and int(g.attrs["num_samples"]) == 6
        assert g["actions"].shape == (6, 7) and g["joint_targets"].shape == (6, 8)
        assert g["obs/joint_pos_abs"].shape == (6, 8)
        assert g["obs/images/wrist_rgb"].shape == (6, 224, 224, 3)
        assert g["obs/images/overhead_rgb"].dtype == np.uint8
        assert g["obs/cycle_phase"][:].tolist() == [1] * 6
        assert g["obs/cycle_brick"][:].tolist() == [0] * 6
        assert g["obs/clamped"].shape == (6,)
        assert g["initial_state/articulation/robot/joint_position"].shape == (1, 8)
        assert g["initial_state/rigid_object/brick_1/root_pose"].shape == (1, 7)
        assert g["initial_state/bin_layout"].shape == (1, 5)
        assert "inflight" not in f
        # The action's arm columns are the pre-clamp command; joint_targets the applied ones.
        assert np.allclose(g["actions"][0, :6], action[0, :6].numpy(), atol=1e-6)
        assert g["obs/images/wrist_rgb"][0].mean() > 5.0  # a real render, not zeros


def test_episode_draws_move_bins_lights_and_layouts_per_reset():
    env = MjLegoSortEnv(2, _draw(), num_bricks=2, cameras=False, episode_draws=True)
    env.reset(seed=21)
    assert env.bin_layouts[0] != env.bin_layouts[1]
    assert env.bin_layouts[0].slot_colors == env.run_draw["slot_colors"]
    floor = env.models[0].geom("bin_red_floor").id
    assert not np.allclose(env.models[0].geom_pos[floor], env.models[1].geom_pos[floor])
    dome = env.models[0].light("dome").id
    assert not np.allclose(env.models[0].light_diffuse[dome], env.models[1].light_diffuse[dome])
    before, before1 = env.bin_layouts[0], env.bin_layouts[1]
    env.reset(env_ids=torch.tensor([0]))
    assert env.bin_layouts[0] != before and env.bin_layouts[1] == before1
    env.close()


def test_spatial_scale_widens_the_per_episode_draws():
    env = MjLegoSortEnv(
        8, _draw(), num_bricks=2, cameras=False, episode_draws=True, spatial_scale=3.0
    )
    assert env.cfg.spatial_scale == 3.0
    env.reset(seed=21)
    # At scale 1.0 no draw can leave these bounds, so one env past them proves the scale
    # reaches the per-episode draw and not only the run draw.
    assert any(
        max(abs(v) for v in d["overhead_cam_drot_deg"]) > 2.0
        or abs(d["bin_x"] - layout._BIN_X) > 0.02
        for d in env.episode_draws
    )
    env.close()


def test_without_episode_draws_every_env_keeps_the_run_layout():
    env = MjLegoSortEnv(2, _draw(), num_bricks=2, cameras=False)
    env.reset(seed=21)
    assert env.bin_layouts == [env.cfg.bin_layout, env.cfg.bin_layout]
    env.close()


def test_per_brick_stowed_uses_each_envs_own_bins_under_episode_draws():
    env = MjLegoSortEnv(2, _draw(), num_bricks=4, cameras=False, episode_draws=True)
    env.reset(seed=21)
    _teleport_all_bricks_into_bins(env, 0)
    assert per_brick_stowed(env)[0].all()
    # Move env 1's bins 10 cm away, then rest env 1's bricks where env 0's bins are.
    far = layout.BinLayout(
        bin_x=env.bin_layouts[1].bin_x + 0.10,
        bin_ys=env.bin_layouts[1].bin_ys,
        slot_colors=env.bin_layouts[1].slot_colors,
    )
    env.bin_layouts[1] = far
    from maker_arm_lab.mj import scene as mj_scene

    mj_scene.apply_bins(env.models[1], far)
    h, d = env.handles, env.datas[1]
    centers = layout.bin_interior_centers(env.bin_layouts[0])
    rest_z = -layout.BIN_INTERIOR[2] / 2 + layout.BRICK_SIZE[2] / 2 + 0.0005
    for b, adr in enumerate(h.brick_qpos):
        d.qpos[adr : adr + 3] = centers[b % 4].numpy() + np.array([0.0, 0.0, rest_z])
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(env.models[1], d)
    env._refresh([1], images=False)
    assert not per_brick_stowed(env)[1].any()
    env.close()


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_episode_draws_and_frame_jitter_move_the_cameras(tmp_path):
    env = MjLegoSortEnv(
        2,
        _draw(),
        num_bricks=1,
        cameras=True,
        episode_draws=True,
        frame_jitter=True,
        export_dir=tmp_path,
    )
    env.reset(seed=8)
    cam = env.models[0].camera("overhead").id
    assert not np.allclose(env.models[0].cam_pos[cam], env.models[1].cam_pos[cam])
    first = env.models[0].cam_pos[cam].copy()
    env.step(_hold_action(env))
    second = env.models[0].cam_pos[cam].copy()
    assert not np.allclose(first, second)
    pos, _rot = mj_scene_overhead_pose(env.episode_draws[0])
    assert np.abs(second - np.array(pos)).max() <= 0.005 + 1e-9
    env.close()


def mj_scene_overhead_pose(draw):
    from maker_arm_lab.mj import scene as mj_scene

    return mj_scene.overhead_camera_pose(draw)
