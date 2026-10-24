"""MS5 design P2: pinholes, poses, the homographies and the warp to the simulator's views."""

import copy
import math
import random

import numpy as np
import pytest

from maker_arm_lab.core import overlay
from maker_arm_lab.core.cameras import WRIST_FOV_DEG
from maker_arm_lab.real.camera_model import (
    Pinhole,
    Pose,
    ViewWarp,
    matrix_quat,
    overhead_warp,
    plane_homography,
    project,
    quat_matrix,
    rotation_homography,
    sim_overhead,
    wrist_warp,
)
from test_mj_heap_env import _can_render

REAL_F = 240.0 / math.tan(math.radians(37.5))  # a 4:3 camera with a 75 degree vertical view


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _real_camera(roll_deg=90.0, f=REAL_F):
    """A 640x480 camera at the simulator's overhead centre, turned ``roll_deg`` about its own
    optical axis."""
    _, sim_pose = sim_overhead()
    return Pinhole(640, 480, f, f, 320.0, 240.0), Pose(sim_pose.rot @ _rz(roll_deg), sim_pose.pos)


def test_from_k_reads_a_camera_info_k_and_refuses_skew():
    cam = Pinhole.from_k(640, 480, (500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0))
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (500.0, 510.0, 320.0, 240.0)
    with pytest.raises(ValueError):
        Pinhole.from_k(640, 480, (500.0, 1.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0))


def test_the_sim_pinhole_has_the_mujoco_fovy():
    assert Pinhole.sim(60.0).fov_deg() == pytest.approx((60.0, 60.0))


def test_quaternions_round_trip_through_matrices():
    rng = random.Random(3)
    for _ in range(50):
        q = np.array([rng.gauss(0, 1) for _ in range(4)])
        q = q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)
        assert matrix_quat(quat_matrix(q)) == pytest.approx(tuple(q), abs=1e-9)


def test_projection_matches_the_overlays_mujoco_convention():
    cam, pose = sim_overhead()
    mujoco_mat = (pose.rot @ np.diag([1.0, -1.0, -1.0])).reshape(9)
    for point in [(0.3, 0.0, 0.0), (0.25, 0.1, 0.02), (0.1, -0.2, 0.0)]:
        want = overlay.project_point(pose.pos, mujoco_mat, 60.0, 224, 224, np.array(point))
        assert project(cam, pose, point) == pytest.approx(want)


def test_the_sim_overhead_puts_plus_x_up_and_plus_y_left():
    cam, pose = sim_overhead()
    u0, v0 = project(cam, pose, (0.30, 0.0, 0.0))
    _, v_far = project(cam, pose, (0.40, 0.0, 0.0))
    u_left, _ = project(cam, pose, (0.30, 0.10, 0.0))
    assert v_far < v0 and u_left < u0


def test_the_plane_homography_maps_desk_points_between_the_cameras():
    sim, sim_pose = sim_overhead()
    real, real_pose = _real_camera()
    h = plane_homography(real, real_pose, sim, sim_pose)
    for point in [(0.30, 0.05, 0.0), (0.2, -0.1, 0.0), (0.45, 0.2, 0.0)]:
        u, v = project(sim, sim_pose, point)
        x = h @ np.array([u, v, 1.0])
        assert (x[0] / x[2], x[1] / x[2]) == pytest.approx(project(real, real_pose, point))


def test_the_rotation_homography_of_one_camera_is_the_scaled_identity():
    sim = Pinhole.sim(70.0)
    h = rotation_homography(sim, sim)
    assert h / h[2, 2] == pytest.approx(np.eye(3))


def test_the_rotation_homography_follows_a_turned_camera_and_its_transpose_does_not():
    """A real wrist camera turned by ``turn`` (its axes in the simulated camera's frame, what
    ``RigConfig.wrist_rot`` holds) against the simulated one at the same centre.
    ``real_eval_heap.open_sources`` passes ``turn.T`` as ``rot_real_from_sim``; the transposed
    rotation must not map the same points."""
    sim = Pinhole.sim(WRIST_FOV_DEG)
    real = Pinhole(640, 480, 400.0, 410.0, 330.0, 236.0)
    sim_pose = Pose(_rz(-30.0) @ _rx(150.0), np.array([0.2, -0.1, 0.3]))
    turn = _rz(12.0) @ _rx(7.0)
    real_pose = Pose(sim_pose.rot @ turn, sim_pose.pos)
    offsets = [(0.03, -0.02, 0.15), (-0.05, 0.04, 0.25), (0.0, 0.0, 0.1), (0.06, 0.05, 0.2)]
    points = [sim_pose.pos + sim_pose.rot @ np.array(o) for o in offsets]

    def maps(h) -> bool:
        for point in points:
            u, v = project(sim, sim_pose, point)
            x = h @ np.array([u, v, 1.0])
            got, want = np.array(x[:2] / x[2]), np.array(project(real, real_pose, point))
            if not np.allclose(got, want, atol=1e-6):
                return False
        return True

    assert maps(rotation_homography(real, sim, turn.T))
    assert maps(wrist_warp(real, quat_matrix(matrix_quat(turn)).T).h)
    assert not maps(rotation_homography(real, sim, turn))


@pytest.mark.parametrize(
    "shape", [(480, 640), (480, 640, 4), (480, 640, 1), (360, 640, 3), (240, 640, 3)]
)
def test_the_warp_refuses_a_frame_it_would_mis_sample(shape):
    """Not 3-channel, or a height that does not follow the width by the source's aspect."""
    warp = overhead_warp(*_real_camera())
    with pytest.raises(ValueError):
        warp.apply(np.zeros(shape, dtype=np.uint8))


def test_the_warp_takes_a_reduced_decode_that_rounds_up():
    """A JPEG draft rounds each side up: 1001x751 at 1/8 decodes to 126x94."""
    camera = Pinhole(1001, 751, 600.0, 600.0, 500.5, 375.5)
    warp = ViewWarp(np.eye(3), (camera.width, camera.height))
    assert warp.apply(np.zeros((94, 126, 3), dtype=np.uint8)).shape == (224, 224, 3)


def _dot_image(camera, pose, point, value=250):
    image = np.full((camera.height, camera.width, 3), 20, dtype=np.uint8)
    u, v = project(camera, pose, point)
    image[round(v) - 3 : round(v) + 4, round(u) - 3 : round(u) + 4] = value
    return image


def test_the_warp_lands_a_desk_dot_where_the_simulator_sees_it():
    real, real_pose = _real_camera()
    point = (0.30, 0.05, 0.0)
    view = overhead_warp(real, real_pose).apply(_dot_image(real, real_pose, point))
    # The warped dot is a plateau several pixels wide, so its argmax is a corner of it; read
    # its position as the centroid of the intensity above the background (20).
    weight = np.clip(view[..., 0].astype(float) - 20.0, 0.0, None)
    rows, cols = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    u = float((weight * (cols + 0.5)).sum() / weight.sum())
    v = float((weight * (rows + 0.5)).sum() / weight.sum())
    su, sv = project(*sim_overhead(), point)
    assert abs(u - su) <= 1.5 and abs(v - sv) <= 1.5


def test_a_half_resolution_decode_warps_to_the_same_view():
    real, real_pose = _real_camera()
    # A smooth pattern with a 24 px period: a 2x2 block mean keeps it (1 px features would
    # not survive it). The correct scaling reads a mean diff of about 1.8; sampling the half
    # decode half a source pixel off reads about 7.3, forgetting the pixel-centre offset
    # about 14.
    rows, cols = np.mgrid[0:480, 0:640]
    wave = np.sin(2 * np.pi * cols / 24) * np.cos(2 * np.pi * rows / 24)
    image = np.repeat((110 + 90 * wave).astype(np.uint8)[..., None], 3, axis=2)
    half = image.reshape(240, 2, 320, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    warp = overhead_warp(real, real_pose)
    diff = np.abs(warp.apply(image).astype(int) - warp.apply(half).astype(int))
    assert diff.mean() < 4.0


def test_a_narrow_camera_leaves_the_uncovered_view_black():
    real, real_pose = _real_camera(roll_deg=0.0, f=REAL_F * 2.0)
    warp = overhead_warp(real, real_pose)
    view = warp.apply(np.full((480, 640, 3), 128, dtype=np.uint8))
    assert 0.0 < warp.coverage < 1.0
    assert tuple(view[0, 0]) == (0, 0, 0) and tuple(view[112, 112]) == (128, 128, 128)


def _uncached_apply(warp, image):
    """``ViewWarp.apply`` as it was before it cached its gathers per frame size."""
    h_img, w_img = image.shape[:2]
    scale = w_img / warp.source_size[0]
    x = warp._su * scale - 0.5
    y = warp._sv * scale - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    valid = (x >= -0.5) & (x <= w_img - 0.5) & (y >= -0.5) & (y <= h_img - 0.5)
    xa, xb = np.clip(x0, 0, w_img - 1), np.clip(x0 + 1, 0, w_img - 1)
    ya, yb = np.clip(y0, 0, h_img - 1), np.clip(y0 + 1, 0, h_img - 1)

    def at(yy, xx):
        return image[yy, xx].astype(np.float32)

    top = at(ya, xa) * (1.0 - fx) + at(ya, xb) * fx
    bottom = at(yb, xa) * (1.0 - fx) + at(yb, xb) * fx
    out = top * (1.0 - fy) + bottom * fy
    out[~valid] = 0.0
    s, n = warp.supersample, warp.size
    out = out.reshape(n, s, n, s, -1).mean(axis=(1, 3))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def test_the_cached_warp_is_byte_identical_to_the_uncached_one():
    """At full size and at reduced decodes, in any order: the cache keys on the frame size."""
    rng = np.random.default_rng(7)
    warp = overhead_warp(*_real_camera(roll_deg=20.0, f=REAL_F * 2.0))
    assert 0.0 < warp.coverage < 1.0  # some samples fall outside the frame
    for shape in ((480, 640, 3), (240, 320, 3), (480, 640, 3), (60, 80, 3)):
        image = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(warp.apply(image), _uncached_apply(warp, image))


def test_jpeg_reduce_keeps_a_source_pixel_per_sample_point():
    real, real_pose = _real_camera()
    assert overhead_warp(real, real_pose).jpeg_reduce() == 1  # about 360 px across the view
    big = Pinhole(1920, 1440, 3.0 * REAL_F, 3.0 * REAL_F, 960.0, 720.0)
    warp = overhead_warp(big, real_pose)  # about 1080 px across it
    assert warp.jpeg_reduce() == 2
    assert warp.source_span / 2 >= warp.size * warp.supersample > warp.source_span / 4


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_the_warp_of_a_wider_4_3_render_matches_the_native_224_view():
    import mujoco

    from maker_arm_lab.core.layout import sample_run_draw
    from maker_arm_lab.mj import scene
    from maker_arm_lab.mj.env import MjLegoSortEnv

    env = MjLegoSortEnv(1, sample_run_draw(random.Random(3)), cameras=True)
    env.reset(seed=1)
    model, data = env.models[0], env.datas[0]
    cam = model.camera("overhead").id
    native = mujoco.Renderer(model, 224, 224)
    native.update_scene(data, camera=cam)
    want = native.render()
    wide = copy.copy(model)
    wide.cam_fovy[cam] = 75.0
    wide.vis.global_.offwidth, wide.vis.global_.offheight = 640, 480
    big = mujoco.Renderer(wide, 480, 640)
    big.update_scene(data, camera=cam)
    frame = big.render()
    pos, rot = scene.overhead_camera_pose(env.episode_draws[0])
    pose = Pose.from_pos_quat(pos, rot)
    real = Pinhole(640, 480, REAL_F, REAL_F, 320.0, 240.0)
    got = ViewWarp(plane_homography(real, pose, Pinhole.sim(60.0), pose), (640, 480)).apply(frame)
    native.close()
    big.close()
    env.close()
    # One camera centre, so the homography is exact at every depth; what is left is resampling.
    # If this fails, save both images and look before touching the bound.
    assert np.abs(got.astype(int) - want.astype(int)).mean() < 8.0
