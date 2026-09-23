"""MS7a desk boxes: the side-band draw, the yawed-box grader and the overhead coverage."""

import itertools
import math
import random
import types

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import (
    cameras,
    checker,
    desk_boxes,
    layout,
    stowed,
)
from maker_arm_lab.core.grasp import polygon_distance
from maker_arm_lab.core.quat import compose_offset_rot, quat_mul
from maker_arm_lab.mj import overlay


def _tray_outer():
    (x0, x1), (y0, y1) = layout.tray_bounds()
    w = layout.TRAY_WALL
    return [(x0 - w, y0 - w), (x1 + w, y0 - w), (x1 + w, y1 + w), (x0 - w, y1 + w)]


def test_the_band_lies_beside_the_arm_never_behind_it_or_in_front_of_the_tray():
    (x0, x1), (y0, y1) = desk_boxes.DESK_BOX_BAND_X, desk_boxes.DESK_BOX_BAND_Y
    tray_far = layout.tray_bounds()[0][1] + layout.TRAY_WALL
    assert 0.0 <= x0 <= x1 <= tray_far
    assert 0.0 < y0 <= y1


@pytest.mark.parametrize("n", [1, 4])
def test_every_drawn_box_is_in_the_band_on_its_side_and_clear(n):
    rng = random.Random(20267000)
    corridor = desk_boxes.pick_corridor()
    sides = set()
    for _ in range(300):
        poses = desk_boxes.sample_boxes(rng, n)
        assert len(poses) == n
        for pose in poses:
            assert desk_boxes.in_band(pose), pose
            assert abs(pose[2]) <= math.radians(45.0) + 1e-12
            print_ = desk_boxes.corners(pose)
            assert polygon_distance(print_, corridor) >= desk_boxes.DESK_BOX_CLEARANCE
            assert polygon_distance(print_, _tray_outer()) >= desk_boxes.DESK_BOX_CLEARANCE
            # Never under the arm base either: every corner stays off the base disc.
            assert min(math.hypot(*c) for c in print_) > desk_boxes.ARM_BASE_RADIUS
        for a, b in itertools.combinations(poses, 2):
            gap = polygon_distance(desk_boxes.corners(a), desk_boxes.corners(b))
            assert gap >= desk_boxes.DESK_BOX_GAP
        if n == 4:
            # Two per side: boxes 0 and 2 on the left (+y), 1 and 3 on the right.
            assert [math.copysign(1.0, p[1]) for p in poses] == [1.0, -1.0, 1.0, -1.0]
        else:
            sides.add(math.copysign(1.0, poses[0][1]))
    if n == 1:
        assert sides == {1.0, -1.0}  # rung 1 draws the side per episode


def test_a_seed_reproduces_the_draw():
    a = desk_boxes.sample_boxes(random.Random(7), 4)
    b = desk_boxes.sample_boxes(random.Random(7), 4)
    assert a == b and desk_boxes.sample_boxes(random.Random(8), 4) != a


def test_one_box_takes_every_part_and_the_placeholder_spreads_them_over_four():
    assert desk_boxes.part_destinations(5, 1) == [0] * 5
    assert desk_boxes.part_destinations(6, 4) == [0, 1, 2, 3, 0, 1]


def test_a_box_footprint_on_the_corridor_is_refused():
    x0, y0 = desk_boxes.DESK_BOX_BAND_X[0], desk_boxes.DESK_BOX_BAND_Y[0]
    assert not desk_boxes.clear([(layout.TRAY_CENTER[0], 0.12, 0.0)])  # on the tray wall
    assert not desk_boxes.clear([(0.1, 0.3, 0.0), (0.1, 0.35, 0.0)])  # two boxes overlap
    assert desk_boxes.clear([(x0 + 0.05, y0 + 0.03, 0.0)])


# -- the grader ------------------------------------------------------------------------------

Z, HALF = desk_boxes.stow_region()
POSES = [(0.08, 0.22, math.radians(40.0)), (0.08, -0.22, math.radians(-30.0))]


def _boxes():
    return torch.tensor([[(x, y, Z, yaw) for x, y, yaw in POSES]])


def _in_box(pose, u, v, z):
    """A world point at (u, v) in the box frame, height z."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * u - s * v, y + s * u + c * v, z)


def test_a_part_at_rest_in_its_own_yawed_box_counts_and_nowhere_else():
    floor = desk_boxes.DESK_BOX_WALL
    half_x = desk_boxes.DESK_BOX_INTERIOR[0] / 2
    pos = torch.tensor(
        [
            [
                _in_box(POSES[0], 0.03, -0.03, floor + 0.005),  # own box (0), near a corner
                _in_box(POSES[0], 0.0, 0.0, floor + 0.005),  # at rest in box 0, not its own
                (0.2, 0.0, 0.005),  # on the desk, in no box
                # inside the box's axis-aligned bounding square, outside the yawed interior
                _in_box(POSES[1], half_x + 0.004, half_x + 0.004, floor + 0.005),
                _in_box(POSES[1], 0.0, 0.01, floor + 2.2 * desk_boxes.DESK_BOX_INTERIOR[2]),
                _in_box(POSES[1], -0.02, 0.0, floor + 0.01),  # own box (1)
            ]
        ]
    )
    ids = torch.tensor([[0, 1, 0, 1, 1, 1]])
    ok = checker.parts_at_rest_in_own_boxes(
        pos, torch.zeros(1, 6, 3), _boxes(), torch.tensor(HALF), ids
    )
    assert ok.tolist() == [[True, False, False, False, False, True]]
    inside = checker.in_boxes(pos, _boxes(), torch.tensor(HALF))[0]
    assert inside.any(-1).tolist() == [True, True, False, False, False, True]
    moving = torch.zeros(1, 6, 3)
    moving[0, 0] = torch.tensor([0.0, 0.0, 0.1])
    ok = checker.parts_at_rest_in_own_boxes(pos, moving, _boxes(), torch.tensor(HALF), ids)
    assert not bool(ok[0, 0])


class _Scene(dict):
    """``env.scene``: parts by name, and the env origins ``core.stowed`` subtracts."""


def _stub_env(pos, destination):
    n, k, _ = pos.shape
    scene = _Scene(
        {
            f"brick_{b}": types.SimpleNamespace(
                data=types.SimpleNamespace(root_pos_w=pos[:, b], root_lin_vel_w=torch.zeros(n, 3))
            )
            for b in range(k)
        }
    )
    scene.env_origins = torch.zeros(n, 3)
    return types.SimpleNamespace(
        device="cpu",
        num_envs=n,
        cfg=types.SimpleNamespace(num_bricks=k, bin_layout=layout.DEFAULT_BIN_LAYOUT),
        scene=scene,
        destination_bins=destination,
        desk_box_poses=np.array([POSES] * n),
    )


def test_per_brick_stowed_grades_each_part_against_its_own_box():
    floor = desk_boxes.DESK_BOX_WALL + 0.005
    pos = torch.tensor([[_in_box(POSES[0], 0.0, 0.0, floor), _in_box(POSES[1], 0.0, 0.0, floor)]])
    assert stowed.per_brick_stowed(_stub_env(pos, torch.tensor([[0, 1]]))).tolist() == [
        [True, True]
    ]
    assert stowed.per_brick_stowed(_stub_env(pos, torch.tensor([[1, 1]]))).tolist() == [
        [False, True]
    ]
    assert stowed.parts_in_desk_boxes(_stub_env(pos, torch.tensor([[1, 1]]))).tolist() == [
        [True, True]
    ]


# -- the overhead coverage -------------------------------------------------------------------


def _camera_poses():
    """The overhead camera at every corner of its per-episode draw, +-0.01 m and +-2 deg per
    axis (``layout.sample_run_draw``), the pose an episode renders with.

    The per-frame jitter (+-0.005 m and +-1 deg, ``mj.env.FRAME_CAM_JITTER_M``) comes on top.
    With every one of the twelve axes at its extreme at once, the near outer corner of a box
    yawed 45 degrees at the band's near x leaves the frame by 7 px; at the draw's own extremes
    it keeps 1.7 px (measured 2026-09-23)."""
    for dpos in itertools.product((-0.01, 0.01), repeat=3):
        for drot in itertools.product((-2.0, 2.0), repeat=3):
            pos = [p + d for p, d in zip(cameras.OVERHEAD_CAM_POS, dpos)]
            rot = quat_mul(
                compose_offset_rot(cameras.OVERHEAD_CAM_ROT, drot), cameras.ROS_TO_MUJOCO_CAM
            )
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.array(rot))
            yield pos, mat


def test_every_box_corner_at_every_band_extreme_is_in_the_overhead_frame():
    (x0, x1), (y0, y1) = desk_boxes.DESK_BOX_BAND_X, desk_boxes.DESK_BOX_BAND_Y
    rim = desk_boxes.rim_z()
    centres = [(x, s * y) for x in (x0, x1) for y in (y0, y1) for s in (1.0, -1.0)]
    points = [
        (*c, z)
        for x, y in centres
        for yaw in (-desk_boxes.DESK_BOX_MAX_YAW, 0.0, desk_boxes.DESK_BOX_MAX_YAW)
        for c in desk_boxes.corners((x, y, yaw))
        for z in (0.0, rim)
    ]
    worst = math.inf
    for pos, mat in _camera_poses():
        for p in points:
            u, v = overlay.project_point(pos, mat, cameras.OVERHEAD_FOV_DEG, 224, 224, p)
            worst = min(worst, u, v, 224 - u, 224 - v)
    assert worst > 0.0, worst
