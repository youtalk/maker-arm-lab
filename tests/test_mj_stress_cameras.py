"""MS7j camera knobs: the pose, field-of-view and warp-back stress of the MuJoCo env."""

import math
import random
import types

import mujoco
import numpy as np
import pytest

from maker_arm_lab.core import cameras, layout, overlay
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.quat import quat_mul
from maker_arm_lab.mj import scene
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.mj.stress import WARP_SOURCE_PX, Stress, offset_rotation
from maker_arm_lab.real.camera_model import Pinhole, Pose, plane_homography, project, quat_matrix
from maker_arm_lab.robots.frames import WRIST_CAM_POS_IN_WRIST, WRIST_CAM_ROT_IN_WRIST
from test_mj_heap_env import PARTS, _can_render, _draw

needs_gl = pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
ROS_TO_MJ = np.diag([1.0, -1.0, -1.0])  # a ROS camera's axes to a MuJoCo camera's


def _env(stress):
    """The six-brick env after one reset. ``stress=None`` builds it without the argument."""
    extra = {} if stress is None else {"stress": stress}
    e = MjLegoSortEnv(1, sample_run_draw(random.Random(3)), cameras=True, **extra)
    e.reset(seed=1)
    return e


def render_once(stress):
    e = _env(stress)
    out = types.SimpleNamespace(
        overhead=e._overhead[0].copy(),
        wrist=e._wrist[0].copy(),
        warp_renderers=len(e._warp_renderers),
    )
    e.close()
    return out


def _mean_diff(a, b) -> float:
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def _ring(image) -> np.ndarray:
    return np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_offset_rotation_turns_about_the_optical_axes():
    forward = quat_matrix(offset_rotation((1.0, 0.0, 0.0, 0.0), 10.0, 0.0, 0.0))[:, 2]
    assert forward[1] < 0.0  # rx > 0 tilts the view up: ROS y points down
    forward = quat_matrix(offset_rotation((1.0, 0.0, 0.0, 0.0), 0.0, 10.0, 0.0))[:, 2]
    assert forward[0] > 0.0  # ry > 0 turns it right
    want = quat_matrix(WRIST_CAM_ROT_IN_WRIST) @ _rx(1.0) @ _ry(2.0) @ _rz(3.0)
    got = quat_matrix(offset_rotation(WRIST_CAM_ROT_IN_WRIST, 1.0, 2.0, 3.0))
    np.testing.assert_allclose(got, want, atol=1e-12)


@needs_gl
def test_no_camera_knob_keeps_the_frames_byte_identical():
    a, b = render_once(Stress()), render_once(None)  # None: built without a stress argument
    assert np.array_equal(a.overhead, b.overhead) and np.array_equal(a.wrist, b.wrist)
    assert a.warp_renderers == b.warp_renderers == 0
    # A wrist knob alone leaves the overhead frame alone, and no warp, no second renderer.
    c = render_once(Stress(wrist_offset=(0.0, 3.0, 0.0, 0.0, 0.0, 0.0)))
    assert np.array_equal(c.overhead, b.overhead) and not np.array_equal(c.wrist, b.wrist)
    assert c.warp_renderers == 0


def _dot(stress, x, y):
    """The overhead pixel centroid of a dark brick at (x, y) on the desk, and the camera pose
    the episode draws. The brick is dark because the lit desk renders close to white."""
    e = _env(stress)
    m, d, handles = e.models[0], e.datas[0], e.handles
    m.light_diffuse[m.light("sun").id] = 0.0  # no shadow next to the brick
    m.geom_rgba[handles.body_geoms[0][0]] = (0.0, 0.0, 0.0, 1.0)
    for b, a in enumerate(handles.brick_qpos[1:], start=1):
        d.qpos[a : a + 3] = (3.0, 0.1 * b, 0.1)  # the other bricks off the desk
    shots = []
    for pos in ((x, y), (3.0, 0.0)):
        a = handles.brick_qpos[0]
        d.qpos[a : a + 7] = (*pos, layout.BRICK_SIZE[2] / 2, 1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        e._refresh([0], images=True)
        shots.append(e._overhead[0].astype(int))
    pose = scene.overhead_camera_pose(e.episode_draws[0])
    e.close()
    changed = np.abs(shots[0] - shots[1]).sum(axis=-1) > 60
    assert changed.sum() > 30
    vs, us = np.nonzero(changed)
    return (us.mean() + 0.5, vs.mean() + 0.5), pose


@needs_gl
def test_an_overhead_shift_moves_a_desk_point_by_the_projected_amount():
    # dx = 0.05: a dark brick on the desk moves to where camera_model.project puts it, within
    # 1.5 px, the centre of its top face.
    x, y, top = 0.35, 0.2, layout.BRICK_SIZE[2]
    sim = Pinhole.sim(cameras.OVERHEAD_FOV_DEG)
    still, (pos, rot) = _dot(Stress(), x, y)
    moved, _ = _dot(Stress(overhead_offset=(0.0, 0.0, 0.0, 0.05, 0.0, 0.0)), x, y)
    want_still = project(sim, Pose.from_pos_quat(pos, rot), (x, y, top))
    want_moved = project(sim, Pose.from_pos_quat(np.add(pos, (0.05, 0.0, 0.0)), rot), (x, y, top))
    assert math.dist(still, want_still) < 1.5 and math.dist(moved, want_moved) < 1.5
    assert want_moved[1] - want_still[1] > 10.0  # forward moves the desk down the frame


@needs_gl
def test_warp_back_undoes_a_mount_shift_on_the_desk_plane():
    # offset dz = 0.04, warp_back: mean abs difference to the nominal render < 8 (as in
    # tests/test_real_camera_model.py), and less than half of it without warp_back. The desk is
    # one flat color, so only edges differ, and what the warp leaves is the arm's parallax off
    # the desk plane. Measured 4.6 against 10.8 here, and 6.1 to 6.7 against 13.2 to 14.5 on
    # other draws of the six-brick, heap and tray scenes.
    nominal = render_once(Stress()).overhead
    shift = (0.0, 0.0, 0.0, 0.0, 0.0, 0.04)
    warped = render_once(Stress(overhead_offset=shift, warp_back=True))
    assert warped.warp_renderers == 1
    left = _mean_diff(warped.overhead, nominal)
    assert left < 8.0
    assert _mean_diff(render_once(Stress(overhead_offset=shift)).overhead, nominal) > 2 * left


@needs_gl
def test_warp_back_undoes_a_wrist_turn():
    # A turn about the camera centre warps back exactly. Compare the middle of the frame: the
    # moved view does not cover the nominal one's edge.
    nominal = render_once(Stress()).wrist[40:184, 40:184]
    turn = (5.0, 3.0, 4.0, 0.0, 0.0, 0.0)
    warped = render_once(Stress(wrist_offset=turn, warp_back=True)).wrist[40:184, 40:184]
    uncorrected = render_once(Stress(wrist_offset=turn)).wrist[40:184, 40:184]
    assert _mean_diff(warped, nominal) < 2.0 and _mean_diff(uncorrected, nominal) > 6.0


@needs_gl
def test_a_narrower_fov_leaves_a_black_border_and_a_wider_one_does_not():
    # fov delta -10 with warp_back: the outer ring is black; +10: coverage 1.0
    for delta in (-10.0, 10.0):
        e = _env(Stress(overhead_fov_delta=delta, wrist_fov_delta=delta, warp_back=True))
        warps = [e._warps[0][c] for c in (e._cam_overhead, e._cam_wrist)]
        assert [w.source_size for w in warps] == [(WARP_SOURCE_PX, WARP_SOURCE_PX)] * 2
        if delta < 0:
            assert not _ring(e._overhead[0]).any() and not _ring(e._wrist[0]).any()
            assert all(w.coverage < 0.75 for w in warps)
        else:
            assert [w.coverage for w in warps] == [1.0, 1.0]
            assert _ring(e._overhead[0]).any() and _ring(e._wrist[0]).any()
        e.close()


@needs_gl
def test_the_uncorrected_marker_stays_at_the_nominal_projection():
    e = MjLegoSortEnv(
        1,
        _draw(),
        parts=PARTS[:3],
        cameras=True,
        episode_length_s=1.0,
        stress=Stress(overhead_offset=(0.0, 0.0, 0.0, 0.08, 0.0, 0.0)),
    )
    e.reset(seed=3)
    before = e.obs_buf["images"]["overhead_rgb"][0].numpy().copy()
    d, c = e.datas[0], e._cam_overhead
    point = d.xpos[e.handles.brick_bodies[1]].copy()
    e.target_part[0] = 1
    e.stamp_markers()
    image = e.obs_buf["images"]["overhead_rgb"][0].numpy()
    pos, rot = scene.overhead_camera_pose(e.episode_draws[0])
    u, v = project(Pinhole.sim(cameras.OVERHEAD_FOV_DEG), Pose.from_pos_quat(pos, rot), point)
    mu, mv = overlay.project_point(
        d.cam_xpos[c], d.cam_xmat[c], float(e.models[0].cam_fovy[c]), 224, 224, point
    )
    e.close()
    ring = overlay.MARKER_RADIUS_PX + overlay.MARKER_RING_PX
    assert math.hypot(mu - u, mv - v) > 2 * ring  # the moved camera sees the part elsewhere
    assert tuple(image[round(v), round(u)]) == (255, 255, 255)
    assert (image[round(mv), round(mu)] == before[round(mv), round(mu)]).all()


@needs_gl
def test_the_offset_follows_each_episode_draw():
    # --episode_draws, two resets: the camera pose equals nominal_i composed with the offset
    # each time, and the warp is recomputed (Review Focus 2).
    over, wrist = (1.0, 2.0, 3.0, 0.01, 0.02, 0.03), (2.0, 1.0, -1.0, 0.005, 0.0, 0.01)
    e = MjLegoSortEnv(
        2,
        sample_run_draw(random.Random(3)),
        cameras=True,
        episode_draws=True,
        stress=Stress(overhead_offset=over, wrist_offset=wrist, warp_back=True),
    )
    assert len(e._warp_renderers) == 2
    homographies = []
    for seed in (1, None):  # the second reset draws on from the first one's rng
        e.reset(seed=seed)
        for i in range(2):
            m, d, ep = e.models[i], e.datas[i], e.episode_draws[i]
            pos, rot = scene.overhead_camera_pose(ep)
            rot_ros = quat_matrix(rot) @ _rx(1.0) @ _ry(2.0) @ _rz(3.0)
            c = e._cam_overhead
            np.testing.assert_allclose(d.cam_xpos[c], np.add(pos, over[3:]), atol=1e-9)
            np.testing.assert_allclose(d.cam_xmat[c].reshape(3, 3), rot_ros @ ROS_TO_MJ, atol=1e-9)
            nominal = Pose.from_pos_quat(pos, rot)
            want = plane_homography(
                Pinhole.sim(60.0, WARP_SOURCE_PX),
                Pose(rot_ros, np.add(pos, over[3:])),
                Pinhole.sim(60.0),
                nominal,
            )
            h = e._warps[i][c].h
            np.testing.assert_allclose(h / h[2, 2], want / want[2, 2], atol=1e-9)
            homographies.append(h)
            # The wrist: the shift in the nominal camera frame, on the episode's mount draw.
            w = e._cam_wrist
            mount = quat_matrix(WRIST_CAM_ROT_IN_WRIST)
            body_pos = np.add(WRIST_CAM_POS_IN_WRIST, ep["wrist_cam_dpos"]) + mount @ wrist[3:]
            np.testing.assert_allclose(m.cam_pos[w], body_pos, atol=1e-12)
            want_quat = quat_mul(
                offset_rotation(WRIST_CAM_ROT_IN_WRIST, *wrist[:3]), cameras.ROS_TO_MUJOCO_CAM
            )
            np.testing.assert_allclose(m.cam_quat[w], want_quat, atol=1e-12)
            assert m.cam_fovy[w] == cameras.WRIST_FOV_DEG
    e.close()
    # Four episode poses, four different warps.
    for a in range(4):
        for b in range(a):
            assert not np.allclose(homographies[a], homographies[b])
