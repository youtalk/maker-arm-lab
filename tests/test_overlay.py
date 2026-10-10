"""The target marker: projection through a MuJoCo camera and the stamped disc (design 9)."""

import math

import numpy as np
import pytest

from maker_arm_lab.core import overlay

CAM_POS = np.array([0.0, 0.0, 1.0])
LOOK_DOWN = np.eye(3).reshape(9)  # the camera's -z is the world -z


def test_a_point_on_the_optical_axis_projects_to_the_image_center():
    assert overlay.project_point(CAM_POS, LOOK_DOWN, 60.0, 224, 224, np.zeros(3)) == (112.0, 112.0)


def test_projection_moves_right_for_plus_x_and_down_for_minus_y():
    focal = 112.0 / math.tan(math.radians(30.0))
    u, v = overlay.project_point(CAM_POS, LOOK_DOWN, 60.0, 224, 224, np.array([0.1, 0.0, 0.0]))
    assert u == pytest.approx(112.0 + focal * 0.1) and v == pytest.approx(112.0)
    u, v = overlay.project_point(CAM_POS, LOOK_DOWN, 60.0, 224, 224, np.array([0.0, -0.1, 0.0]))
    assert u == pytest.approx(112.0) and v == pytest.approx(112.0 + focal * 0.1)


def test_a_point_behind_the_camera_is_none():
    assert (
        overlay.project_point(CAM_POS, LOOK_DOWN, 60.0, 224, 224, np.array([0.0, 0.0, 2.0])) is None
    )


def test_stamp_marker_paints_a_white_disc_with_a_black_ring():
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    overlay.stamp_marker(image, 100.4, 50.2)
    assert tuple(image[50, 100]) == (255, 255, 255)
    assert tuple(image[50, 105]) == (255, 255, 255)
    assert tuple(image[50, 106]) == (0, 0, 0)
    assert tuple(image[50, 107]) == (128, 128, 128)
    assert tuple(image[56, 100]) == (0, 0, 0)


def test_stamp_marker_clips_at_the_border_and_skips_an_off_image_point():
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    overlay.stamp_marker(image, 2.0, 2.0)
    assert tuple(image[0, 0]) == (255, 255, 255)
    untouched = np.full((224, 224, 3), 128, dtype=np.uint8)
    overlay.stamp_marker(untouched, -50.0, -50.0)
    assert (untouched == 128).all()


def test_projection_with_non_identity_rotation_pins_the_transpose_direction():
    """Rotate 45 degrees around z-axis to pin the camera-to-world transform direction.

    Rotation matrix (45 degrees, counterclockwise when viewed from +z):
    [  cos(45°)  sin(45°)  0 ]   [  0.707  0.707  0 ]
    [ -sin(45°)  cos(45°)  0 ] = [ -0.707  0.707  0 ]
    [      0         0     1 ]   [    0      0     1 ]

    A point at world (0.1, 0, 0) with camera at (0, 0, 1) looking through this rotated camera:
    - Direction from camera to point in world: (0.1, 0, -1)
    - M.T @ direction = (0.0707, +0.0707, -1)
    - Projection: u = cx + focal * x / -z, v = cy - focal * y / -z
    - Result: u = 112 + focal * 0.0707, v = 112 - focal * 0.0707
    - Note: u is identical under both correct and transposed paths. The v assertion catches
      the transpose error: using M instead of M.T gives v = 112 + focal * 0.0707.
    """
    cos45 = math.cos(math.radians(45.0))
    sin45 = math.sin(math.radians(45.0))
    rotated_camera = np.array([cos45, sin45, 0.0, -sin45, cos45, 0.0, 0.0, 0.0, 1.0])
    focal = 112.0 / math.tan(math.radians(30.0))

    u, v = overlay.project_point(CAM_POS, rotated_camera, 60.0, 224, 224, np.array([0.1, 0.0, 0.0]))
    assert u == pytest.approx(112.0 + focal * cos45 * 0.1)
    assert v == pytest.approx(112.0 - focal * sin45 * 0.1)


def test_the_overlay_lives_in_core_so_the_real_env_needs_no_mj():
    import importlib.util

    assert importlib.util.find_spec("maker_arm_lab.core.overlay") is not None
    assert importlib.util.find_spec("maker_arm_lab.mj.overlay") is None
