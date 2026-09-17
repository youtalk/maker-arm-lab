"""Grasp synthesis on part geometry (design section 7.1) and the open-jaw landing model."""

import math

import pytest
import torch

from maker_arm_lab.core import grasp, layout, parts

BRICK = parts.make_part("brick", 2, 4)
PLATE = parts.make_part("plate", 2, 4)
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])
ROLL_90 = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
WALLS = grasp.tray_wall_obstacles()
NO_WALLS: list = []


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


def test_a_tilted_approach_opens_the_short_axis_of_a_lying_plate():
    plate = parts.make_part("plate", 2, 4)
    size = torch.tensor(list(plate.size))
    pos = torch.tensor([layout.TRAY_CENTER[0], 0.0, 0.0016])
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    flat = grasp.candidate_grasps(0, size, pos, quat)
    tilted_40 = grasp.candidate_grasps(0, size, pos, quat, tilt_rad=math.radians(40.0))
    tilted_80 = grasp.candidate_grasps(0, size, pos, quat, tilt_rad=math.radians(80.0))
    assert all(g.width > plate.size[2] for g in flat)
    # 40 degrees is short of the ~75 degree floor AXIS_MAX_TILT_RAD sets on this plate:
    # the thickness axis is still refused, only the long in-plane axis survives.
    assert all(g.width > plate.size[2] for g in tilted_40)
    assert any(math.isclose(g.width, plate.size[2], rel_tol=1e-6) for g in tilted_80)


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


def _clearance(pos, heading, width=0.0158, half_height=0.0048, rows=(), walls=NO_WALLS):
    return grasp.jaw_clearance(pos, heading, width, half_height, *_neighbors(*rows), walls)


def test_clearance_is_infinite_alone_and_the_landing_spot_is_an_open_jaw_out():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    assert _clearance(pos, 0.0) == (math.inf, 0, math.inf)
    # A 2x2 brick just past the sweep band, its near face under the landing spot.
    small = parts.make_part("brick", 2, 2)
    under = (small, 0.2 + grasp.OPEN_HALF_GAP + 0.010, 0.0, 0.0048, IDENTITY)
    clearance, swept, wall = _clearance(pos, 0.0, rows=[under])
    assert clearance < 0.0 and swept == 0 and wall == math.inf


def test_a_neighbor_in_the_sweep_band_is_swept_not_blocking():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    beside = (BRICK, 0.2, 0.030, 0.0048, IDENTITY)  # 30 mm off along the closing axis
    clearance, swept, _ = _clearance(pos, math.pi / 2, rows=[beside])
    assert swept == 1 and clearance > 0.0
    across = _clearance(pos, 0.0, rows=[beside])  # closing along x: the brick is beside a face
    assert across[1] == 0 and across[0] > 0.0


def test_a_neighbor_under_the_grip_band_is_the_floor_not_an_obstacle():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    flat = (PLATE, 0.2 + grasp.OPEN_HALF_GAP, 0.0, 0.0016, IDENTITY)  # a plate under the tip
    assert _clearance(pos, 0.0, rows=[flat]) == (math.inf, 0, math.inf)
    plate_target = torch.tensor([0.2, 0.0, 0.0016])
    # For a plate target every neighbor stands, and a plate under the tip blocks the descend.
    assert _clearance(plate_target, 0.0, half_height=0.0016, rows=[flat])[0] < 0.0


def test_the_landing_spot_grows_with_the_neighbor_height():
    pos = torch.tensor([0.2, 0.0, 0.0048])
    x = 0.2 + grasp.OPEN_HALF_GAP + 0.008 + 0.0159  # its near face 8 mm past the tip
    low = (BRICK, x, 0.0, 0.0048, IDENTITY)  # the hull is 6 mm thick at a brick's top
    tall = (parts.make_part("brick", 4, 4), x, 0.0, 0.0300, IDENTITY)  # 18 mm thick up there
    assert _clearance(pos, 0.0, rows=[low])[0] > 0.0
    assert _clearance(pos, 0.0, rows=[tall])[0] < 0.0


def test_wall_clearance_matches_the_measured_landing_limit():
    """Wall reads, adapted to the round dish (MS6 rework section 5): a tip landing a couple
    mm inside the rim picks, one a couple mm past it does not, and one well past it has to
    cross the rim on the way in. The dish is rotationally symmetric, so one radius stands
    for every azimuth; the old rectangle's separate short/long wall reads do not carry over.
    """
    cx, cy = layout.TRAY_CENTER
    picks = torch.tensor([cx, cy - 0.032, 0.0048])
    fails = torch.tensor([cx, cy - 0.036, 0.0048])
    beyond = torch.tensor([cx, cy - 0.06, 0.0048])
    assert 0.0 < _clearance(picks, math.pi / 2, walls=WALLS)[2] < 0.003
    assert _clearance(fails, math.pi / 2, walls=WALLS)[2] < 0.0
    assert _clearance(beyond, math.pi / 2, walls=WALLS)[2] < -0.01


def test_a_grasp_at_the_dish_centre_clears_the_rim_and_one_at_the_rim_does_not():
    sizes = torch.tensor([list(layout.BRICK_SIZE)])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    middle = torch.tensor([[layout.TRAY_CENTER[0], layout.TRAY_CENTER[1], 0.005]])
    edge = torch.tensor([[layout.TRAY_CENTER[0] + layout.TRAY_RADIUS - 0.005, 0.0, 0.005]])
    mask = torch.tensor([True])
    best_middle = grasp.rank_grasps(sizes, middle, quat, mask, WALLS)[0]
    best_edge = grasp.rank_grasps(sizes, edge, quat, mask, WALLS)[0]
    assert best_middle.wall > 0.0
    assert best_edge.wall < best_middle.wall


def _heap(*rows):
    sizes = torch.stack([_size(p) for p, *_ in rows])
    pos = torch.tensor([[x, y, z] for _, x, y, z, _ in rows])
    quat = torch.stack([q for *_, q in rows])
    return sizes, pos, quat


def test_rank_puts_the_highest_band_first_then_the_least_swept_then_the_clearest():
    sizes, pos, quat = _heap(
        (BRICK, 0.19, 0.00, 0.0048, IDENTITY),
        (BRICK, 0.19, 0.00, 0.0144, IDENTITY),  # resting on the first
        (BRICK, 0.19, 0.03, 0.0048, IDENTITY),  # in the first brick's y sweep band
    )
    mask = torch.ones(3, dtype=torch.bool)
    ranked = grasp.rank_grasps(sizes, pos, quat, mask, WALLS)
    assert ranked[0].part == 1 and ranked[0].height == pytest.approx(0.0144)
    assert ranked[0].clearance >= grasp.CLEARANCE_MIN  # the brick under it is below the band
    desk = [g for g in ranked if g.height < 0.01]
    assert desk[0].swept == 0 and desk[-1].swept >= 1
    assert all(not math.isnan(g.clearance) for g in ranked)


def test_rank_honors_the_candidate_mask():
    sizes, pos, quat = _heap(
        (BRICK, 0.16, 0.0, 0.0048, IDENTITY), (BRICK, 0.24, 0.0, 0.0048, IDENTITY)
    )
    mask = torch.tensor([False, True])
    assert {g.part for g in grasp.rank_grasps(sizes, pos, quat, mask, WALLS)} == {1}


def test_select_skips_grasps_under_the_clearance_floor():
    clear = grasp.Grasp(0, 0.0, 0.0158, 0.0048, grasp.CLEARANCE_MIN + 0.001, 0.0048)
    blocked = grasp.Grasp(1, 0.0, 0.0158, 0.0048, -0.002, 0.0144)
    assert grasp.select_grasp([blocked, clear]) is clear
    assert grasp.select_grasp([blocked]) is None
    assert grasp.select_grasp([]) is None


def test_densest_cell_and_push_segment_stay_inside_the_tray():
    xy = torch.tensor([[0.16, 0.01], [0.17, 0.02], [0.18, 0.03], [0.24, -0.08]])
    center = grasp.densest_cell(xy)
    assert center == (pytest.approx(0.175), pytest.approx(0.025))
    start, end = grasp.push_segment(center, math.pi / 2)
    assert start == (pytest.approx(0.175), pytest.approx(0.025 - grasp.PUSH_HALF_LENGTH_M))
    assert end == (pytest.approx(0.175), pytest.approx(0.025 + grasp.PUSH_HALF_LENGTH_M))
    (x0, x1), (y0, y1) = layout.tray_bounds()
    for heading in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        for point in grasp.push_segment((x1 - 0.005, y1 - 0.005), heading):
            assert x0 < point[0] < x1 and y0 < point[1] < y1


def test_push_toward_center_lands_behind_the_part_and_sweeps_inward():
    (x0, x1), (y0, y1) = layout.tray_bounds()
    cx, cy = layout.TRAY_CENTER
    part = (cx, cy - 0.06)  # wall-side: 28 mm shy of the rim, room for the sweep to start
    print_ = grasp.footprint(
        _size(parts.make_part("plate", 2, 8)), torch.tensor([*part, 0.0016]), IDENTITY
    )
    start, end = grasp.push_toward_center(part, print_, [])
    assert x0 < start[0] < x1 and y0 < start[1] < y1
    assert math.hypot(start[0] - cx, start[1] - cy) > math.hypot(part[0] - cx, part[1] - cy)
    assert math.hypot(end[0] - cx, end[1] - cy) < math.hypot(part[0] - cx, part[1] - cy)
    assert grasp.polygon_distance(grasp.tip_footprint(start), print_) >= grasp.PUSH_START_GAP_M
    # A start under another part's footprint is refused, so the tips never land on a part.
    blocker = grasp.footprint(_size(BRICK), torch.tensor([*start, 0.0048]), IDENTITY)
    other = grasp.push_toward_center(part, print_, [blocker])
    assert other is None or grasp.polygon_distance(grasp.tip_footprint(other[0]), blocker) >= 0.0


def test_a_part_wedged_against_the_rim_has_no_push():
    """The round dish has no corner; the analog is a part close enough to the rim that no
    swing direction leaves room for the tips to land behind it."""
    cx, cy = layout.TRAY_CENTER
    part = (cx, cy - (layout.TRAY_RADIUS - 0.008))
    print_ = grasp.footprint(
        _size(parts.make_part("brick", 2, 2)), torch.tensor([*part, 0.0048]), IDENTITY
    )
    assert grasp.push_toward_center(part, print_, []) is None


def test_the_wall_obstacles_ring_the_dish():
    prints = grasp.tray_wall_obstacles()
    assert len(prints) == layout.TRAY_SEGMENTS
    cx, cy = layout.TRAY_CENTER
    for poly in prints:
        for x, y in poly:
            assert math.hypot(x - cx, y - cy) > layout.TRAY_RADIUS - 0.006


def test_clamp_to_tray_pulls_a_point_back_along_its_radius():
    cx, cy = layout.TRAY_CENTER
    inside = (cx + 0.01, cy + 0.01)
    assert grasp.clamp_to_tray(inside) == inside
    # Diagonal, off both cardinal axes: a box clamp and a radial clamp disagree here, so
    # this is the direction that actually pins down the radial model.
    far = (cx + 0.5, cy + 0.3)
    clamped = grasp.clamp_to_tray(far)
    assert math.isclose(
        math.hypot(clamped[0] - cx, clamped[1] - cy),
        layout.TRAY_RADIUS - grasp.PUSH_WALL_MARGIN_M,
        abs_tol=1e-9,
    )
    away = math.atan2(far[1] - cy, far[0] - cx)
    clamped_away = math.atan2(clamped[1] - cy, clamped[0] - cx)
    assert math.isclose(away, clamped_away, abs_tol=1e-9)


def test_clamp_to_tray_pins_the_centre_when_the_margin_exceeds_the_radius():
    """A margin past TRAY_RADIUS must pin the centre, not reflect the point through it."""
    cx, cy = layout.TRAY_CENTER
    off_center = (cx + 1e-6, cy)
    clamped = grasp.clamp_to_tray(off_center, margin=layout.TRAY_RADIUS + 0.05)
    assert clamped == (cx, cy)
