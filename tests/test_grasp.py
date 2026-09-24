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


def test_a_slope_offers_its_side_faces_upright_and_its_flat_faces_on_its_side():
    slope = parts.make_part("slope", 1, 3)
    found = grasp.candidate_grasps(
        0, _size(slope), torch.zeros(3), IDENTITY, 0.0, slope.closing_axes
    )
    assert [(round(g.heading, 6), round(g.width, 4)) for g in found] == [
        (round(math.pi / 2, 6), 0.0078)
    ]
    # Rolled onto a side face, body y points up and the jaws close along body z, between the
    # flat bottom and the flat top, across the 9.6 mm height. Never along the 23.8 mm run.
    on_side = grasp.candidate_grasps(
        0, _size(slope), torch.zeros(3), ROLL_90, 0.0, slope.closing_axes
    )
    assert [round(g.width, 4) for g in on_side] == [0.0096]
    assert on_side[0].heading == pytest.approx(-math.pi / 2)
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
    assert [g.width for g in ranked if g.part == 0] == [pytest.approx(0.0078)]
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


# -- MS7a design, the tray: the wall term ------------------------------------------------------

WALLS = grasp.tray_walls()
(TX0, TX1), (TY0, TY1) = layout.tray_bounds()
TRAY_MID_X = layout.TRAY_CENTER[0]
YAW_90 = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])


def _yaw(deg: float) -> torch.Tensor:
    half = math.radians(deg) / 2
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])


def _left_wall():
    wall = next(w for w in WALLS if max(y for _, y in w.footprint) <= TY0 + 1e-9)
    return wall


def test_the_tray_walls_run_along_the_interior_edges():
    assert len(WALLS) == 4
    headings = sorted(round(w.heading, 6) for w in WALLS)
    assert headings == [0.0, 0.0, round(math.pi / 2, 6), round(math.pi / 2, 6)]
    left = _left_wall()
    assert left.heading == 0.0 and left.top == pytest.approx(layout.TRAY_WALL_HEIGHT)
    # The interior face: a point on the tray's edge touches it, one 5 mm in clears it by 5 mm.
    assert grasp.wall_distance((TRAY_MID_X, TY0), left) == pytest.approx(0.0, abs=1e-9)
    assert grasp.wall_distance((TRAY_MID_X, TY0 + 0.005), left) == pytest.approx(0.005)


def test_the_heading_is_the_closing_axis_and_near_a_wall_it_runs_along_the_wall():
    """A 2x4 brick 12 mm from the left wall (y = -0.10, running along x).

    ``Grasp.heading`` is the world heading of the CLOSING axis, the line the two fingers
    travel on; the fingers land ``landing_half_gap(width)`` out along it on both sides. At yaw
    0 the brick's long axis (31.8 mm) lies along x, the wall's direction, so closing across
    its length keeps both fingers beside the wall, and closing across its 15.8 mm width
    points a finger into it. Turned 90 degrees, the survivor is the 15.8 mm axis, now on x.
    """
    pos = torch.tensor([TRAY_MID_X, TY0 + 0.012, 0.0048])
    found = grasp.candidate_grasps(0, _size(BRICK), pos, IDENTITY, walls=WALLS)
    assert [(abs(round(g.heading, 6)), g.width) for g in found] == [(0.0, pytest.approx(0.0318))]
    turned = grasp.candidate_grasps(0, _size(BRICK), pos, YAW_90, walls=WALLS)
    assert [(round(abs(g.heading), 6), g.width) for g in turned] == [
        (round(math.pi, 6), pytest.approx(0.0158))
    ]
    # Without walls (the bare desk) both axes are offered, as before.
    assert len(grasp.candidate_grasps(0, _size(BRICK), pos, IDENTITY)) == 2


@pytest.mark.parametrize("gap_mm", [8, 12, 20, 26])
@pytest.mark.parametrize("yaw_deg", range(0, 180, 10))
def test_near_a_wall_every_surviving_heading_lies_inside_the_wall_band(gap_mm, yaw_deg):
    left = _left_wall()
    pos = torch.tensor([TRAY_MID_X, TY0 + gap_mm / 1000, 0.0048])
    for part in (BRICK, parts.make_part("brick", 1, 6), parts.make_part("brick", 2, 3)):
        for g in grasp.candidate_grasps(0, _size(part), pos, _yaw(yaw_deg), walls=WALLS):
            reach = grasp.landing_half_gap(g.width) + grasp.TIP_THICKNESS
            if grasp.wall_distance((float(pos[0]), float(pos[1])), left) < reach:
                assert grasp.along_wall(g.heading, left), (part.name, g)


def test_the_stroke_refuses_a_finger_through_a_wall_and_accepts_one_along_it(monkeypatch):
    # The band off, so only the finger's own path is tested.
    monkeypatch.setattr(grasp, "WALL_BAND_RAD", math.pi / 2)
    width = 0.0158
    reach = grasp.landing_half_gap(width) + grasp.TIP_THICKNESS  # the tip's outer face

    def allows(gap: float, heading: float) -> bool:
        pos = torch.tensor([TRAY_MID_X, TY0 + gap, 0.0048])
        return grasp.wall_allows(pos, heading, width, 0.0048, WALLS)

    across = math.pi / 2  # the closing axis points at the left wall
    assert allows(reach + 0.0005, across)  # the outer face lands half a millimeter inside
    assert not allows(reach - 0.0005, across)  # ... or half a millimeter past the wall face
    assert allows(0.008, 0.0)  # along the wall the fingers never reach it
    # The tip is 8 mm across, so along the wall a centre closer than its half face is refused.
    assert not allows(grasp.TIP_HALF_FACE - 0.0005, 0.0)


def test_the_band_refuses_a_heading_at_the_wall_that_the_stroke_alone_would_pass():
    width = 0.0158
    reach = grasp.landing_half_gap(width) + grasp.TIP_THICKNESS
    pos = torch.tensor([TRAY_MID_X, TY0 + reach + grasp.WALL_BAND_MARGIN / 2, 0.0048])
    assert not grasp.wall_allows(pos, math.pi / 2, width, 0.0048, WALLS)
    far = torch.tensor([TRAY_MID_X, TY0 + reach + 2 * grasp.WALL_BAND_MARGIN, 0.0048])
    assert grasp.wall_allows(far, math.pi / 2, width, 0.0048, WALLS)


def test_a_wall_under_the_grip_band_is_no_obstacle():
    """A part standing on end grips above the 20 mm wall top, as a neighbor under the band."""
    tall = parts.make_part("brick", 1, 2, 5)  # 48 mm high, standing on end at the identity
    pos = torch.tensor([TRAY_MID_X, TY0 + 0.006, 0.024])
    found = grasp.candidate_grasps(0, _size(tall), pos, IDENTITY, walls=WALLS)
    assert sorted(round(g.width, 4) for g in found) == [0.0078, 0.0158]  # both, one at the wall
    # The same footprint one brick high: the wall stands in its grip band, so the axis that
    # points at the wall goes.
    low = torch.tensor([TRAY_MID_X, TY0 + 0.006, 0.0048])
    short = parts.make_part("brick", 1, 2)
    found = grasp.candidate_grasps(0, _size(short), low, IDENTITY, walls=WALLS)
    assert [round(g.width, 4) for g in found] == [0.0158]


def test_a_corner_part_keeps_no_heading_and_ranking_honors_the_walls():
    corner = torch.tensor([TX0 + 0.012, TY0 + 0.012, 0.0048])
    assert not grasp.candidate_grasps(0, _size(BRICK), corner, IDENTITY, walls=WALLS)
    sizes, pos, quat = _heap((BRICK, float(corner[0]), float(corner[1]), 0.0048, IDENTITY))
    mask = torch.ones(1, dtype=torch.bool)
    assert grasp.rank_grasps(sizes, pos, quat, mask, walls=WALLS) == []
    assert len(grasp.rank_grasps(sizes, pos, quat, mask)) == 2


def test_along_a_wall_a_long_part_closes_end_to_end_and_nowhere_else():
    long_brick = parts.make_part("brick", 2, 7)  # 55.8 x 15.8 mm
    flush = torch.tensor([TRAY_MID_X, TY0 + 0.0085, 0.0048])
    found = grasp.candidate_grasps(0, _size(long_brick), flush, IDENTITY, walls=WALLS)
    assert [round(g.width, 4) for g in found] == [0.0558]  # along the wall, end to end
    bare = grasp.candidate_grasps(0, _size(long_brick), flush, IDENTITY)
    assert [round(g.width, 4) for g in bare] == [0.0158]  # the bare desk: across only
    middle = torch.tensor([TRAY_MID_X, layout.TRAY_CENTER[1], 0.0048])
    inside = grasp.candidate_grasps(0, _size(long_brick), middle, IDENTITY, walls=WALLS)
    assert [round(g.width, 4) for g in inside] == [0.0158]  # away from a wall, only across
    too_long = parts.make_part("brick", 2, 8)  # 63.8 mm, past WALL_MAX_GRASP_WIDTH
    assert not grasp.candidate_grasps(0, _size(too_long), flush, IDENTITY, walls=WALLS)


def test_a_flush_part_leaves_no_room_behind_it_so_the_rake_wedges_in_at_the_wall():
    slope = parts.make_part("slope", 1, 3)  # upright, its 7.8 mm side against the left wall
    xy = (TRAY_MID_X, TY0 + 0.0044)
    pos = torch.tensor([*xy, 0.0048])
    part_print = grasp.footprint(_size(slope), pos, IDENTITY)
    left = _left_wall()
    start, end, heading = grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [])
    assert heading == pytest.approx(math.pi / 2)  # straight off the wall
    tips = grasp.rake_footprint(start)
    gap = grasp.polygon_distance(tips, list(left.footprint))  # the tips at the wall face
    assert grasp.RAKE_WEDGE_GAP_M <= gap < grasp.RAKE_WEDGE_GAP_M + 0.001
    assert grasp.polygon_distance(tips, part_print) < 0.0  # on the part's wall-side edge
    assert end[1] == pytest.approx(xy[1] + 0.04)
    # A neighbor under the wedge spot refuses the rake.
    under = grasp.footprint(_size(BRICK), torch.tensor([start[0], start[1], 0.0048]), IDENTITY)
    assert grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [under]) is None


def test_a_crowded_rake_start_lands_at_most_rake_crowd_deep_into_a_neighbour():
    """The flush slope of the wedge test with a brick beside the wedge spot: refused while
    the tips must clear it, taken with --rake_crowd once they land no deeper than
    RAKE_CROWD_M into it, and refused again when the brick sits deeper."""
    slope = parts.make_part("slope", 1, 3)
    xy = (TRAY_MID_X, TY0 + 0.0044)
    part_print = grasp.footprint(_size(slope), torch.tensor([*xy, 0.0048]), IDENTITY)
    left = _left_wall()
    wedge, _, _ = grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [])
    tip_edge = wedge[0] + grasp.TIP_HALF_FACE  # the wedge tips' +x edge
    half = BRICK.size[0] / 2  # the brick lies along x at the identity

    def brick_at(depth):
        """A 2x4 brick whose -x face sits ``depth`` inside the wedge tips' +x edge."""
        pos = torch.tensor([tip_edge - depth + half, wedge[1], 0.0048])
        return grasp.footprint(_size(BRICK), pos, IDENTITY)

    shallow, deep = brick_at(0.002), brick_at(0.006)
    assert grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [shallow]) is None
    crowd = grasp.RAKE_CROWD_M
    got = grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [shallow], crowd)
    assert got is not None
    depth = -grasp.polygon_distance(grasp.rake_footprint(got[0]), shallow)
    assert 0.0 < depth <= crowd + 1e-9
    assert grasp.rake_segment(xy, part_print, 0.04, left, WALLS, [deep], crowd) is None
