"""MS5 sensing: the iPad tag board, the screen pose, the overhead calibration and the base
registration."""

import math

import numpy as np
import pytest

from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.real import tags
from maker_arm_lab.real.camera_model import Pinhole, Pose, project
from maker_arm_lab.real.layouts import IPAD_AIR_11, nominal_screen_pose
from maker_arm_lab.real.rig import RigConfig
from maker_arm_lab.real.tags import IPAD_BOARD


def _rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}
    return np.array(m[axis], dtype=float)


def _angle_deg(a, b):
    return math.degrees(math.acos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def _camera_over_desk(height=0.8, tilt=(4.0, -3.0), roll_deg=0.0):
    """A camera at (0.03, -0.02, height) looking down, world from camera; ``roll_deg`` turns
    it about its optical axis (90: held in portrait)."""
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    rot = _rot("x", tilt[0]) @ _rot("y", tilt[1]) @ down @ roll
    return Pose(rot, np.array([0.03, -0.02, height]))


def _world_corners(board, tag_id, screen_pose):
    xy = tags.tag_corners_screen(board, tag_id)
    return (screen_pose.rot @ np.column_stack([xy, np.zeros(4)]).T).T + screen_pose.pos


def _placements(cam, cam_pose, board, screen_poses, noise_px, rng):
    out = []
    for sp in screen_poses:
        placement = {}
        for tag_id in board.tags:
            px = [project(cam, cam_pose, p) for p in _world_corners(board, tag_id, sp)]
            placement[tag_id] = np.array(px) + rng.normal(0, noise_px, (4, 2))
        out.append(placement)
    return out


def _screen_pose(x, y, yaw_deg, z):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return Pose(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), np.array([x, y, z]))


def test_the_board_fits_the_screen_with_its_quiet_zone():
    b = IPAD_BOARD
    side = 8 * b.cell_px
    assert math.isclose(b.size_m, 0.04926, abs_tol=1e-5)
    us = sorted({u for u, _ in b.tags.values()})
    vs = sorted({v for _, v in b.tags.values()})
    assert sorted(b.tags) == [0, 1, 2, 3, 4, 5] and len(us) == 3 and len(vs) == 2
    gaps = {us[1] - us[0] - side, us[2] - us[1] - side, vs[1] - vs[0] - side}
    assert len(gaps) == 1 and gaps.pop() >= b.cell_px
    assert min(us) >= b.cell_px and min(vs) >= b.cell_px
    assert IPAD_AIR_11.width_px - (max(us) + side) >= b.cell_px
    assert IPAD_AIR_11.height_px - (max(vs) + side) >= b.cell_px


def test_the_board_image_is_the_screen_size_and_white_outside_the_tags():
    pytest.importorskip("cv2")
    img = tags.board_png(IPAD_BOARD)
    assert img.shape == (1640, 2360, 3) and img.dtype == np.uint8 and img[0, 0].min() == 255


def test_tag_corners_follow_the_arucos_order_in_the_screen_frame():
    c = tags.tag_corners_screen(IPAD_BOARD, 0)
    assert c.shape == (4, 2)
    assert c[0, 0] < c[1, 0] and c[0, 1] == c[1, 1] and c[0, 1] > c[3, 1]
    assert math.isclose(c[1, 0] - c[0, 0], IPAD_BOARD.size_m, abs_tol=1e-9)


def test_a_rendered_board_gives_back_its_screen_pose():
    cv2 = pytest.importorskip("cv2")
    cam = Pinhole.from_ros_k(1920, 1440, (1500, 0, 959.5, 0, 1500, 719.5, 0, 0, 1))
    true = Pose(_rot("x", 183.0) @ _rot("y", 4.0), np.array([0.01, 0.02, 0.45]))
    board = tags.board_png(IPAD_BOARD)
    # Board pixel (lab convention) to image pixel (lab), then to OpenCV's index convention.
    s = IPAD_AIR_11
    to_xy = np.array([[s.mm_per_px / 1000, 0, -s.width_px / 2 * s.mm_per_px / 1000],
                      [0, -s.mm_per_px / 1000, s.height_px / 2 * s.mm_per_px / 1000],
                      [0, 0, 1.0]])  # fmt: skip
    plane = cam.matrix() @ np.column_stack([true.rot[:, 0], true.rot[:, 1], true.pos])
    h_lab = plane @ to_xy
    a = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
    h_cv = a @ np.diag([4.0, 4.0, 1.0]) @ h_lab @ np.linalg.inv(a)  # to the 4x frame
    big = cv2.warpPerspective(  # render at 4x and average down, as a real sensor integrates
        board, h_cv, (7680, 5760), borderValue=(255, 255, 255)
    )
    frame = cv2.resize(big, (1920, 1440), interpolation=cv2.INTER_AREA)
    corners = tags.detect_tags(frame)
    assert sorted(corners) == [0, 1, 2, 3, 4, 5]
    pose, rms = tags.screen_in_camera(cam, corners, IPAD_BOARD)
    assert np.linalg.norm(pose.pos - true.pos) < 0.001
    assert _angle_deg(pose.rot, true.rot) < 0.1 and rms < 0.5


def test_detected_corners_use_the_lab_pixel_convention():
    cv2 = pytest.importorskip("cv2")
    img = np.full((400, 400, 3), 255, np.uint8)
    img[100:260, 120:280] = np.repeat(
        cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 3, 160, borderBits=1
        )[..., None],
        3,
        axis=2,
    )
    got = tags.detect_tags(img)[3]
    want = np.array([[120, 100], [280, 100], [280, 260], [120, 260]], float)  # pixel edges
    assert np.abs(got - want).max() < 0.1


def test_the_check_ids_are_the_ones_the_solve_did_not_use():
    calib, check = tags.split_ids([5, 0, 3, 1, 2])
    assert calib == [0, 2, 5] and check == [1, 3]


def test_the_mean_pose_of_one_pose_is_that_pose():
    p = Pose(_rot("x", 30.0), np.array([0.1, 0.2, 0.3]))
    m = tags.mean_pose([p])
    assert np.allclose(m.rot, p.rot) and np.allclose(m.pos, p.pos)
    two = tags.mean_pose([p, Pose(_rot("x", 40.0), np.array([0.3, 0.2, 0.3]))])
    assert np.allclose(two.pos, [0.2, 0.2, 0.3]) and abs(_angle_deg(two.rot, _rot("x", 35))) < 1e-6


WIDE = ((1920, 1440), (1500, 0, 959.5, 0, 1500, 719.5, 0, 0, 1), 0.0)
TALL = ((1440, 1920), (1500, 0, 719.5, 0, 1500, 959.5, 0, 0, 1), 90.0)  # as published


def _calibrate_from(size, k, roll, seed):
    cam = Pinhole.from_ros_k(*size, k)
    true = _camera_over_desk(0.8, roll_deg=roll)
    reference = nominal_screen_pose(RigConfig())
    z = reference.pos[2]
    screens = [reference, _screen_pose(0.28, 0.15, 25.0, z), _screen_pose(0.12, -0.17, -40.0, z)]
    placements = _placements(cam, true, IPAD_BOARD, screens, 0.2, np.random.default_rng(seed))
    for p in placements:  # every corner is a real pixel of the published image
        for c in p.values():
            assert (c >= 0).all() and (c[:, 0] <= size[0]).all() and (c[:, 1] <= size[1]).all()
    res = tags.calibrate(cam, placements, IPAD_BOARD, reference)
    assert np.linalg.norm(res.pose.pos - true.pos) < 0.005  # along the view axis it is soft
    assert res.plane_rms_mm < 0.5 and res.check_rms_mm < 0.5 and res.check_rms_px < 1.0
    assert _angle_deg(res.pose.rot, true.rot) < 0.3


@pytest.mark.parametrize("seed", range(5))
def test_three_placements_recover_the_camera_pose(seed):
    pytest.importorskip("cv2")
    _calibrate_from(*WIDE, seed)


@pytest.mark.parametrize("seed", range(5))
def test_a_tall_camera_calibrates_from_the_image_as_published(seed):
    pytest.importorskip("cv2")
    _calibrate_from(*TALL, seed)


def test_a_tag_of_another_board_changes_nothing():
    pytest.importorskip("cv2")
    cam = Pinhole.from_ros_k(*WIDE[0], WIDE[1])
    true = _camera_over_desk(0.8)
    screen = nominal_screen_pose(RigConfig())
    (placement,) = _placements(cam, true, IPAD_BOARD, [screen], 0.2, np.random.default_rng(0))
    extra = dict(placement) | {9: placement[0] + (40.0, 25.0)}
    a, _ = tags.screen_in_camera(cam, placement, IPAD_BOARD)
    b, _ = tags.screen_in_camera(cam, extra, IPAD_BOARD)
    assert np.allclose(a.pos, b.pos) and np.allclose(a.rot, b.rot)


def test_register_base_recovers_the_camera_and_the_tag_offset():
    ik = GlobalIK()
    rng = np.random.default_rng(3)
    lo, hi = ik.lower.numpy(), ik.upper.numpy()
    mid, span = (lo + hi) / 2, (hi - lo) / 2
    q = mid + rng.uniform(-0.5, 0.5, (10, 6)) * span
    true = _camera_over_desk(0.7)
    off = np.array([0.004, -0.012, 0.02])
    tag_cam = []
    for qi in q:
        p, r = ik.fk(qi)
        tag_cam.append(true.to_camera(p.numpy() + r.numpy() @ off) + rng.normal(0, 0.0005, 3))
    guess = Pose(true.rot @ _rot("x", 5.0), true.pos + np.array([0.02, 0.0, 0.0]))
    pose, got_off, rms, per, converged = tags.register_base(q, np.array(tag_cam), ik, guess)
    assert np.linalg.norm(pose.pos - true.pos) < 0.001 and np.linalg.norm(got_off - off) < 0.001
    assert rms < 1.5 and per.shape == (10,) and converged


def test_the_frame_tap_keeps_the_first_camera_and_the_frames_since_start():
    from types import SimpleNamespace as NS

    tap = tags.FrameTap()
    tap.on_jpeg("early")
    tap.on_info(NS(width=1920, height=1440, k=(1500, 0, 959.5, 0, 1500, 719.5, 0, 0, 1)))
    tap.on_info(NS(width=1, height=1, k=(1, 0, 0, 0, 1, 0, 0, 0, 1)))
    tap.start()
    tap.on_jpeg("a")
    assert tap.camera.width == 1920 and tap.stop() == ["a"]
    tap.on_jpeg("late")
    assert tap.frames == ["a"]


def test_register_base_converges_from_moderately_varied_poses():
    ik = GlobalIK()
    rng = np.random.default_rng(5)
    lo, hi = ik.lower.numpy(), ik.upper.numpy()
    mid, span = (lo + hi) / 2, (hi - lo) / 2
    q = mid + rng.uniform(-0.2, 0.2, (10, 6)) * span
    true = _camera_over_desk(0.8)
    off = np.array([0.004, -0.012, 0.02])
    tag_cam = np.array(
        [true.to_camera(ik.fk(qi)[0].numpy() + ik.fk(qi)[1].numpy() @ off) for qi in q]
    )
    guess = Pose(true.rot @ _rot("x", 5.0), true.pos + np.array([0.02, 0.0, 0.0]))
    pose, got_off, rms, _, converged = tags.register_base(q, tag_cam, ik, guess)
    assert converged and rms < 0.01
    assert np.linalg.norm(pose.pos - true.pos) < 0.0005 and np.linalg.norm(got_off - off) < 0.0005
