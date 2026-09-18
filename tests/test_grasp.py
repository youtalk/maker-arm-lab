"""Grasp synthesis on part geometry (design section 7.1) and the open-jaw landing model."""

import math

import pytest
import torch

from maker_arm_lab.core import grasp, layout, parts

BRICK = parts.make_part("brick", 2, 4)
PLATE = parts.make_part("plate", 2, 4)
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])
ROLL_90 = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
DEFAULT_WIDTH = 0.0158  # the 2x4 brick's narrow axis, the width `_clearance` closes across
GAP = grasp.landing_half_gap(DEFAULT_WIDTH)
"""Where the jaw inner faces land for that width, since the rig no longer descends fully open."""


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


def test_a_slope_offers_only_the_axis_across_its_side_faces():
    slope = parts.make_part("slope", 2, 4)
    found = grasp.candidate_grasps(
        0, _size(slope), torch.zeros(3), IDENTITY, 0.0, slope.closing_axes
    )
    assert [(round(g.heading, 6), round(g.width, 4)) for g in found] == [
        (round(math.pi / 2, 6), 0.0158)
    ]
    # Rolled onto a side face, body y points up and the slope has no closing axis at all.
    assert (
        grasp.candidate_grasps(0, _size(slope), torch.zeros(3), ROLL_90, 0.0, slope.closing_axes)
        == []
    )
    # rank_grasps carries one axis tuple per part; a brick beside it still offers both axes.
    sizes, pos, quat = _heap(
        (slope, 0.16, 0.0, 0.0048, IDENTITY), (BRICK, 0.24, 0.0, 0.0048, IDENTITY)
    )
    ranked = grasp.rank_grasps(
        sizes,
        pos,
        quat,
        torch.ones(2, dtype=torch.bool),
        0.0,
        [slope.closing_axes, BRICK.closing_axes],
    )
    assert [g.width for g in ranked if g.part == 0] == [pytest.approx(0.0158)]
    assert len([g for g in ranked if g.part == 1]) == 2


def test_a_tilted_axis_and_an_oversize_width_are_not_candidates():
    tilt = math.radians(20.0)
    pitched = torch.tensor([math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0])
    found = grasp.candidate_grasps(0, _size(BRICK), torch.zeros(3), pitched)
    assert [g.width for g in found] == [pytest.approx(0.0158)]  # only body y stays level
    wide = torch.tensor([0.0478, 0.0478, 0.0096])
    assert grasp.candidate_grasps(0, wide, torch.zeros(3), IDENTITY) == []


def test_a_tilted_approach_opens_the_short_axis_of_a_lying_plate():
    plate = parts.make_part("plate", 2, 4)
    size = torch.tensor(list(plate.size))
    pos = torch.tensor([layout.HEAP_CENTER[0], 0.0, 0.0016])
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    flat = grasp.candidate_grasps(0, size, pos, quat)
    tilted_40 = grasp.candidate_grasps(0, size, pos, quat, tilt_rad=math.radians(40.0))
    tilted_80 = grasp.candidate_grasps(0, size, pos, quat, tilt_rad=math.radians(80.0))
    assert all(g.width > plate.size[2] for g in flat)
    # 40 degrees is short of the ~75 degree floor AXIS_MAX_TILT_RAD sets on this plate:
    # the thickness axis is still refused, only the long in-plane axis survives.
    assert all(g.width > plate.size[2] for g in tilted_40)
    assert any(math.isclose(g.width, plate.size[2], rel_tol=1e-6) for g in tilted_80)


def test_the_tilted_jaw_plane_refuses_an_axis_that_lies_along_the_tilt_direction():
    """Pins the jaw-plane MECHANISM in candidate_grasps, independent of PICK_TILT_RAD's
    shipped value (0.0 as of the R5 revert): world +x is 0.342 = sin(20 deg) off a jaw plane
    tilted 20 deg, over the 0.259 = sin(15 deg) = sin(AXIS_MAX_TILT_RAD) limit; world +y,
    across the tilt, stays at 0. An explicit 20 deg keeps this test meaningful at any
    PICK_TILT_RAD, including the 0.0 it ships at today, so the next attempt at this rung
    starts from a mechanism that is already pinned."""
    size = _size(BRICK)
    world_x_width = float(size[0])
    world_y_width = float(size[1])
    level = grasp.candidate_grasps(0, size, torch.zeros(3), IDENTITY, tilt_rad=0.0)
    tilted = grasp.candidate_grasps(0, size, torch.zeros(3), IDENTITY, tilt_rad=math.radians(20.0))

    def has_axis(found, heading, width):
        return any(
            math.isclose(g.heading, heading, abs_tol=1e-6) and math.isclose(g.width, width)
            for g in found
        )

    assert has_axis(level, 0.0, world_x_width)
    assert not has_axis(tilted, 0.0, world_x_width)
    assert has_axis(tilted, math.pi / 2, world_y_width)


def test_neighbor_extents_follow_the_world_aligned_bounding_box():
    sizes = torch.stack([_size(BRICK), _size(BRICK)])
    rxy, hz = grasp.neighbor_extents(sizes, torch.stack([IDENTITY, ROLL_90]))
    assert rxy[0] == pytest.approx(math.hypot(0.0159, 0.0079)) and hz[0] == pytest.approx(0.0048)
    assert rxy[1] == pytest.approx(math.hypot(0.0159, 0.0048), abs=1e-5)
    assert hz[1] == pytest.approx(0.0079, abs=1e-5)


def test_footprint_is_the_rotated_rectangle():
    yaw = 0.4
    quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    hull = grasp.footprint(_size(BRICK), torch.tensor([0.2, 0.0, 0.0048]), quat)
    assert len(hull) == 4
    radii = [math.hypot(x - 0.2, y) for x, y in hull]
    assert all(r == pytest.approx(math.hypot(0.0159, 0.0079), abs=1e-6) for r in radii)


def test_polygon_distance_is_the_gap_or_minus_the_overlap():
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert grasp.polygon_distance(unit, [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]) == 1.0
    diagonal = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)]
    assert grasp.polygon_distance(unit, diagonal) == pytest.approx(math.sqrt(2.0))
    overlap = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    assert grasp.polygon_distance(unit, overlap) == pytest.approx(-0.5)


def _neighbors(*rows):
    """rows of (part, x, y, z, quat) -> (pos, hz, footprints) for ``jaw_clearance``."""
    if not rows:
        return torch.zeros(0, 3), torch.zeros(0), []
    sizes = torch.stack([_size(p) for p, *_ in rows])
    pos = torch.tensor([[x, y, z] for _, x, y, z, _ in rows])
    quat = torch.stack([q for *_, q in rows])
    _, hz = grasp.neighbor_extents(sizes, quat)
    prints = [grasp.footprint(sizes[b], pos[b], quat[b]) for b in range(len(rows))]
    return pos, hz, prints


def _clearance(pos, heading, width=0.0158, half_height=0.0048, rows=()):
    return grasp.jaw_clearance(pos, heading, width, half_height, *_neighbors(*rows))


def test_clearance_is_infinite_alone_and_the_landing_spot_is_a_matched_jaw_out():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    assert _clearance(pos, 0.0) == (math.inf, 0)
    # A 2x2 brick just past the sweep band, its near face under the landing spot.
    small = parts.make_part("brick", 2, 2)
    under = (small, 0.2 + GAP + 0.010, 0.0, 0.0048, IDENTITY)
    clearance, swept = _clearance(pos, 0.0, rows=[under])
    assert clearance < 0.0 and swept == 0


def test_a_neighbor_in_the_sweep_band_is_swept_not_blocking():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    # A 1x2 brick inside the y stroke and clear of both the landing spot and the x stroke.
    beside = (parts.make_part("brick", 1, 2), 0.2, 0.014, 0.0048, IDENTITY)
    clearance, swept = _clearance(pos, math.pi / 2, rows=[beside])
    assert swept == 1 and clearance > 0.0
    across = _clearance(pos, 0.0, rows=[beside])  # closing along x: the brick is beside a face
    assert across[1] == 0 and across[0] > 0.0


def test_a_neighbor_under_the_grip_band_is_the_floor_not_an_obstacle():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    flat = (PLATE, 0.2 + GAP, 0.0, 0.0016, IDENTITY)  # a plate under the tip
    assert _clearance(pos, 0.0, rows=[flat]) == (math.inf, 0)
    plate_target = torch.tensor([0.2, 0.0, 0.0016])
    # For a plate target every neighbor stands, and a plate under the tip blocks the descend.
    assert _clearance(plate_target, 0.0, half_height=0.0016, rows=[flat])[0] < 0.0


def test_the_landing_spot_grows_with_the_neighbor_height():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    x = 0.2 + GAP + 0.008 + 0.0159  # its near face 8 mm past the tip
    low = (BRICK, x, 0.0, 0.0048, IDENTITY)  # the hull is 6 mm thick at a brick's top
    tall = (parts.make_part("brick", 4, 4), x, 0.0, 0.0300, IDENTITY)  # 18 mm thick up there
    assert _clearance(pos, 0.0, rows=[low])[0] > 0.0
    assert _clearance(pos, 0.0, rows=[tall])[0] < 0.0


def _heap(*rows):
    sizes = torch.stack([_size(p) for p, *_ in rows])
    pos = torch.tensor([[x, y, z] for _, x, y, z, _ in rows])
    quat = torch.stack([q for *_, q in rows])
    return sizes, pos, quat


def test_rank_puts_the_highest_band_first_then_the_least_swept_then_the_clearest():
    sizes, pos, quat = _heap(
        (BRICK, 0.19, 0.00, 0.0048, IDENTITY),
        (BRICK, 0.19, 0.00, 0.0144, IDENTITY),  # resting on the first
        (BRICK, 0.19, 0.017, 0.0048, IDENTITY),  # in the first brick's y sweep band
    )
    mask = torch.ones(3, dtype=torch.bool)
    ranked = grasp.rank_grasps(sizes, pos, quat, mask)
    assert ranked[0].part == 1 and ranked[0].height == pytest.approx(0.0144)
    assert ranked[0].clearance >= grasp.CLEARANCE_MIN  # the brick under it is below the band
    desk = [g for g in ranked if g.height < 0.01]
    assert desk[0].swept == 0 and desk[-1].swept >= 1
    assert all(not math.isnan(g.clearance) for g in ranked)


def test_rank_prefers_the_narrower_grasp_when_nothing_else_separates_them():
    """A lone 2x4 brick offers a 15.8 mm and a 31.8 mm grasp at equal (infinite) clearance.

    The dish rim used to break that tie by accident. With the heap on the bare desk the tie
    is broken on purpose: where a choice existed, the narrow axis lifted 3 of 3 flat parts
    and the wide axis 0 of 4 (docs/ms6-matched-jaw-opening.md section 5).
    """
    sizes, pos, quat = _heap((BRICK, 0.19, 0.0, 0.0048, IDENTITY))
    ranked = grasp.rank_grasps(sizes, pos, quat, torch.tensor([True]))
    assert [round(g.width, 4) for g in ranked] == [0.0158, 0.0318]


def test_rank_honors_the_candidate_mask():
    sizes, pos, quat = _heap(
        (BRICK, 0.16, 0.0, 0.0048, IDENTITY), (BRICK, 0.24, 0.0, 0.0048, IDENTITY)
    )
    mask = torch.tensor([False, True])
    assert {g.part for g in grasp.rank_grasps(sizes, pos, quat, mask)} == {1}


def test_select_skips_grasps_under_the_clearance_floor():
    clear = grasp.Grasp(0, 0.0, 0.0158, 0.0048, grasp.CLEARANCE_MIN + 0.001, 0.0048)
    blocked = grasp.Grasp(1, 0.0, 0.0158, 0.0048, -0.002, 0.0144)
    assert grasp.select_grasp([blocked, clear]) is clear
    assert grasp.select_grasp([blocked]) is None
    assert grasp.select_grasp([]) is None


def test_densest_cell_and_push_segment_stay_in_reach():
    xy = torch.tensor([[0.16, 0.01], [0.17, 0.02], [0.18, 0.03], [0.24, -0.08]])
    center = grasp.densest_cell(xy)
    assert center == (pytest.approx(0.175), pytest.approx(0.025))
    start, end = grasp.push_segment(center, math.pi / 2)
    assert start == (pytest.approx(0.175), pytest.approx(0.025 - grasp.PUSH_HALF_LENGTH_M))
    assert end == (pytest.approx(0.175), pytest.approx(0.025 + grasp.PUSH_HALF_LENGTH_M))
    far = (layout.SCATTER_MAX_RADIUS - 0.005, 0.0)
    near = (layout.SCATTER_MIN_RADIUS + 0.005, 0.0)
    for heading in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        for cell in (far, near):
            for point in grasp.push_segment(cell, heading):
                assert layout.SCATTER_MIN_RADIUS <= math.hypot(*point) <= layout.SCATTER_MAX_RADIUS


def test_clamp_to_reach_pulls_a_point_along_its_radius_from_the_base():
    inside = (0.19, 0.03)
    assert grasp.clamp_to_reach(inside) == inside
    # Diagonal, off both cardinal axes: a box clamp and a radial clamp disagree here.
    far = (0.5, 0.3)
    clamped = grasp.clamp_to_reach(far)
    assert math.isclose(
        math.hypot(*clamped), layout.SCATTER_MAX_RADIUS - grasp.PUSH_REACH_MARGIN_M, abs_tol=1e-9
    )
    assert math.isclose(math.atan2(*far[::-1]), math.atan2(*clamped[::-1]), abs_tol=1e-9)
    near = (0.05, 0.02)
    pushed_out = grasp.clamp_to_reach(near)
    assert math.isclose(
        math.hypot(*pushed_out), layout.SCATTER_MIN_RADIUS + grasp.PUSH_REACH_MARGIN_M, abs_tol=1e-9
    )
    assert grasp.clamp_to_reach((0.0, 0.0)) == (0.0, 0.0)


def test_the_landing_gap_follows_the_grasp_width():
    """MS6 rung 1: the jaws descend at an opening matched to the part, not fully open.

    The shipped model put both landing spots ``OPEN_HALF_GAP`` (52.4 mm) out whatever the
    part, so a 2x4 brick and a 1x1 plate asked the same 105 mm of clear floor. The opening
    the rig now holds is the grasp width plus a fixed jaw clearance, so the model asks for
    the room the jaws actually take.
    """
    narrow = grasp.landing_half_gap(0.0158)  # the 2x4 brick's narrow axis
    wide = grasp.landing_half_gap(0.0318)
    assert narrow < wide < grasp.OPEN_HALF_GAP
    assert narrow == pytest.approx(0.0158 / 2 + grasp.JAW_APPROACH_CLEARANCE + grasp.JAW_HOLD_BAND)
    # A part wider than the jaws can straddle never asks for more than the full opening.
    assert grasp.landing_half_gap(2.0 * grasp.OPEN_HALF_GAP) == pytest.approx(grasp.OPEN_HALF_GAP)
