"""Grasp synthesis on part geometry (design section 7.1)."""

import math

import pytest
import torch

from maker_arm_lab.core import grasp, parts

BRICK = parts.make_part("brick", 2, 4)
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])
ROLL_90 = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
NONE = (torch.zeros(0, 3), torch.zeros(0), torch.zeros(0))


def _size(part):
    return torch.tensor(part.size)


def test_body_axes_of_the_identity_are_the_world_axes():
    assert torch.allclose(grasp.body_axes_world(IDENTITY), torch.eye(3))
    axes = grasp.body_axes_world(ROLL_90)
    assert torch.allclose(axes[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(axes[2], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)


def test_a_flat_brick_offers_its_two_horizontal_axes():
    found = grasp.candidate_grasps(0, _size(BRICK), torch.zeros(3), IDENTITY)
    assert [(round(g.heading, 6), g.width) for g in found] == [
        (0.0, pytest.approx(0.0318)),
        (round(math.pi / 2, 6), pytest.approx(0.0158)),
    ]
    assert all(g.half_height == pytest.approx(0.0048) and math.isnan(g.clearance) for g in found)
    assert all(g.part == 0 and g.height == 0.0 for g in found)


def test_a_brick_on_its_side_offers_its_height_as_a_width():
    found = grasp.candidate_grasps(3, _size(BRICK), torch.tensor([0.2, 0.0, 0.0079]), ROLL_90)
    widths = sorted(g.width for g in found)
    assert widths == [pytest.approx(0.0096), pytest.approx(0.0318)]
    assert all(g.half_height == pytest.approx(0.0079) for g in found)
    thin = next(g for g in found if g.width < 0.01)
    assert thin.heading == pytest.approx(-math.pi / 2)


def test_a_tilted_axis_and_an_oversize_width_are_not_candidates():
    tilt = math.radians(20.0)
    pitched = torch.tensor([math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0])
    found = grasp.candidate_grasps(0, _size(BRICK), torch.zeros(3), pitched)
    assert [g.width for g in found] == [pytest.approx(0.0158)]  # only body y stays level
    wide = torch.tensor([0.0478, 0.0478, 0.0096])
    assert grasp.candidate_grasps(0, wide, torch.zeros(3), IDENTITY) == []


def test_neighbor_extents_follow_the_world_aligned_bounding_box():
    sizes = torch.stack([_size(BRICK), _size(BRICK)])
    rxy, hz = grasp.neighbor_extents(sizes, torch.stack([IDENTITY, ROLL_90]))
    assert rxy[0] == pytest.approx(math.hypot(0.0159, 0.0079)) and hz[0] == pytest.approx(0.0048)
    assert rxy[1] == pytest.approx(math.hypot(0.0159, 0.0048), abs=1e-5)
    assert hz[1] == pytest.approx(0.0079, abs=1e-5)


def _one(x, y, z, rxy, hz):
    return torch.tensor([[x, y, z]]), torch.tensor([rxy]), torch.tensor([hz])


def test_jaw_clearance_is_infinite_without_neighbors_and_negative_inside_a_jaw_box():
    pos = torch.zeros(3)
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *NONE) == math.inf
    inside = _one(0.0158 / 2 + 0.015, 0.0, 0.0, 0.005, 0.005)
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *inside) < 0.0


def test_jaw_clearance_measures_to_the_neighbor_footprint():
    pos = torch.zeros(3)
    expected = 0.10 - (0.0158 / 2 + grasp.JAW_BOX_THICKNESS) - 0.005
    ahead = _one(0.10, 0.0, 0.0, 0.005, 0.005)
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *ahead) == pytest.approx(expected)
    beside = _one(0.0, 0.10, 0.0, 0.01, 0.005)
    dx = 0.0158 / 2
    dy = 0.10 - grasp.JAW_BOX_HALF_FACE
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *beside) == pytest.approx(
        math.hypot(dx, dy) - 0.01
    )


def test_a_neighbor_below_the_tips_is_ignored_and_a_standing_one_counts():
    pos = torch.tensor([0.2, 0.0, 0.0144])  # a 2x2 resting on a flat 2x4
    below = _one(0.2, 0.0, 0.0048, 0.0178, 0.0048)  # its top at 9.6 mm is under the tips
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *below) == math.inf
    standing = _one(0.2, 0.0, 0.0048, 0.0178, 0.0200)  # the same footprint, but 4 cm tall
    assert grasp.jaw_clearance(pos, 0.0, 0.0158, *standing) < 0.0


def test_jaw_clearance_rotates_with_the_heading():
    pos = torch.tensor([0.2, 0.0, 0.0])
    neighbor = _one(0.2, 0.03, 0.0, 0.005, 0.005)  # 3 cm off along +y
    across = grasp.jaw_clearance(pos, 0.0, 0.0158, *neighbor)  # closing along x: beside the face
    along = grasp.jaw_clearance(pos, math.pi / 2, 0.0158, *neighbor)  # closing along y: in a jaw
    assert along < 0.0 < across


def _heap(*rows):
    """rows of (part, x, y, z, quat) -> (sizes, pos, quat) tensors."""
    sizes = torch.stack([_size(p) for p, *_ in rows])
    pos = torch.tensor([[x, y, z] for _, x, y, z, _ in rows])
    quat = torch.stack([q for *_, q in rows])
    return sizes, pos, quat


def test_rank_puts_the_highest_part_first_then_the_clearest():
    sizes, pos, quat = _heap(
        (BRICK, 0.16, 0.00, 0.0048, IDENTITY),
        (BRICK, 0.16, 0.00, 0.0144, IDENTITY),  # resting on the first
        (BRICK, 0.24, 0.08, 0.0048, IDENTITY),  # alone
    )
    mask = torch.ones(3, dtype=torch.bool)
    ranked = grasp.rank_grasps(sizes, pos, quat, mask, *NONE)
    assert ranked[0].part == 1 and ranked[0].height == pytest.approx(0.0144)
    assert ranked[0].clearance > grasp.CLEARANCE_MIN  # the brick under it is below the tips
    lower = [g for g in ranked if g.height < 0.01]
    assert lower[0].part == 2  # the lone brick clears more than the one under a neighbor
    assert all(not math.isnan(g.clearance) for g in ranked)


def test_rank_honors_the_candidate_mask():
    sizes, pos, quat = _heap(
        (BRICK, 0.16, 0.0, 0.0048, IDENTITY), (BRICK, 0.24, 0.0, 0.0048, IDENTITY)
    )
    mask = torch.tensor([False, True])
    assert {g.part for g in grasp.rank_grasps(sizes, pos, quat, mask, *NONE)} == {1}


def test_select_skips_grasps_under_the_clearance_floor():
    clear = grasp.Grasp(0, 0.0, 0.0158, 0.0048, 0.0031, 0.0048)
    blocked = grasp.Grasp(1, 0.0, 0.0158, 0.0048, -0.002, 0.0144)
    assert grasp.select_grasp([blocked, clear]) is clear
    assert grasp.select_grasp([blocked]) is None
    assert grasp.select_grasp([]) is None


def test_densest_cell_and_push_segment():
    xy = torch.tensor([[0.16, 0.01], [0.17, 0.02], [0.18, 0.03], [0.24, -0.08]])
    center = grasp.densest_cell(xy)
    assert center == (pytest.approx(0.175), pytest.approx(0.025))
    start, end = grasp.push_segment(center, math.pi / 2)
    assert start == (pytest.approx(0.175), pytest.approx(0.025 - grasp.PUSH_HALF_LENGTH_M))
    assert end == (pytest.approx(0.175), pytest.approx(0.025 + grasp.PUSH_HALF_LENGTH_M))


def test_tray_wall_obstacles_line_the_four_walls():
    from maker_arm_lab.core import layout

    pos, rxy, hz = grasp.tray_wall_obstacles()
    assert pos.shape[0] == rxy.shape[0] == hz.shape[0] > 30
    assert torch.allclose(pos[:, 2], torch.full((pos.shape[0],), layout.TRAY_WALL_HEIGHT / 2))
    assert torch.all(hz == layout.TRAY_WALL_HEIGHT / 2)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    on_x_walls = (pos[:, 0] < x0) | (pos[:, 0] > x1)
    on_y_walls = (pos[:, 1] < y0) | (pos[:, 1] > y1)
    assert torch.all(on_x_walls | on_y_walls)
    brick_by_wall = torch.tensor([x1 - 0.01, 0.0, 0.0048])
    assert grasp.jaw_clearance(brick_by_wall, math.pi / 2, 0.0158, pos, rxy, hz) < 0.0
    brick_mid_tray = torch.tensor([layout.TRAY_CENTER[0], 0.0, 0.0048])
    assert grasp.jaw_clearance(brick_mid_tray, 0.0, 0.0158, pos, rxy, hz) > grasp.CLEARANCE_MIN
