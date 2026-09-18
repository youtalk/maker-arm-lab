"""Tests for maker_arm_lab.core.layout (CPU-only)."""

import itertools
import math
import random

import pytest
import torch

from maker_arm_lab.core import layout, oracle


def test_bin_centers_within_reach():
    # OpenYAM reach is ~0.61 m from the base at the origin; require margin.
    centers = layout.bin_interior_centers()
    assert centers.shape == (4, 3)
    assert (centers.norm(dim=-1) <= 0.58).all()


def test_bins_disjoint():
    centers = layout.bin_interior_centers()
    outer = layout.BIN_INTERIOR[0] + 2 * layout.BIN_WALL
    for i, j in itertools.combinations(range(4), 2):
        assert (centers[i] - centers[j]).norm() >= outer


def test_scatter_zone_reachable_and_off_shelf():
    for x in layout.SCATTER_X:
        for y in layout.SCATTER_Y:
            assert (x * x + y * y) ** 0.5 <= 0.50
    shelf_min_x = layout.SHELF_POS[0] - layout.SHELF_SIZE[0] / 2
    assert layout.SCATTER_X[1] < shelf_min_x


def test_scatter_zone_is_pickable():
    """Every brick the scene can spawn must also be reachable by a top-down pick.

    These two bounds were independent until the OpenYAM LegoSort layout's Task 10
    (openyam-lab M1), and disagreed there: bricks spawned out to x = 0.36 while a
    vertical-tool descent was blocked past the OpenYAM-measured 0.290-0.292
    boundary (``layout.TOP_DOWN_X_LIMIT``) by that arm's link_4 fouling the bins'
    back walls, so roughly a third of the zone spawned bricks that could never be
    picked. Both bounds are carried over unchanged onto the Maker Arm, not yet
    re-measured on it; `scripts/release_envelope_check.py` (Task D4) is the
    authority for that. Tying them together here is what stops the spawn bound
    being widened later without re-validating the pick limit.
    """
    assert layout.SCATTER_X[1] <= layout.TOP_DOWN_X_LIMIT
    assert layout.SCATTER_X[0] < layout.SCATTER_X[1]


def test_default_brick_grid_stays_inside_the_scatter_zone():
    """The pre-first-reset grid in LegoSortEnvCfg.__post_init__ must not spill out.

    It insets four columns by ``margin`` at each end of SCATTER_X and spaces them
    evenly, so the grid follows the zone when it moves. Duplicated here rather
    than imported because the env cfg module needs isaaclab, which the CPU test
    suite deliberately does not.
    """
    margin = 0.02
    spacing = (layout.SCATTER_X[1] - layout.SCATTER_X[0] - 2 * margin) / 3
    assert spacing > layout.BRICK_SIZE[0], "columns must clear an unrotated brick's length"
    worst_x = layout.SCATTER_X[0] + margin + spacing * 3
    assert worst_x <= layout.SCATTER_X[1] + 1e-9
    yaw_half_diagonal = ((layout.BRICK_SIZE[0] / 2) ** 2 + (layout.BRICK_SIZE[1] / 2) ** 2) ** 0.5
    shelf_min_x = layout.SHELF_POS[0] - layout.SHELF_SIZE[0] / 2
    assert worst_x + yaw_half_diagonal < shelf_min_x


def test_bin_static_boxes_count_and_naming():
    boxes = layout.bin_static_boxes()
    assert len(boxes) == 20  # 4 bins x (floor + 4 walls)
    names = [name for name, _, _ in boxes]
    assert len(set(names)) == 20
    for color in layout.BRICK_COLOR_ORDER:
        assert sum(color in name for name in names) == 5


def test_brick_dims_match_spec():
    assert layout.BRICK_SIZE == (0.0318, 0.0158, 0.0096)
    assert set(layout.BRICK_RGBA) == set(layout.BRICK_COLOR_ORDER)


def _corners_visible(pose, fov_deg):
    """Every corner of the region, at its own height, inside a straight-down frame.

    Coverage is re-derived here from the camera height and the field of view
    rather than calling back into layout, so the test is an independent check
    and not a restatement of the implementation. A downward camera at height
    ``h`` sees a half-width of ``(h - z) * tan(fov/2)`` on the plane ``z``.
    """
    (x_min, x_max), (y_min, y_max), z_top = layout.overhead_view_region()
    cam_x, cam_y, cam_z = pose
    for z in (0.0, z_top):
        half_width = (cam_z - z) * math.tan(math.radians(fov_deg) / 2.0)
        for x, y in itertools.product((x_min, x_max), (y_min, y_max)):
            if abs(x - cam_x) > half_width or abs(y - cam_y) > half_width:
                return False
    return True


def test_overhead_view_region_contains_scatter_zone_and_bins():
    (x_min, x_max), (y_min, y_max), z_top = layout.overhead_view_region()
    # the scatter zone, widened by a brick's worst-case yaw half-diagonal
    half_diag = 0.5 * math.hypot(layout.BRICK_SIZE[0], layout.BRICK_SIZE[1])
    assert x_min <= layout.SCATTER_X[0] - half_diag
    assert x_max >= layout.SCATTER_X[1] + half_diag
    assert y_min <= layout.SCATTER_Y[0] - half_diag
    assert y_max >= layout.SCATTER_Y[1] + half_diag
    # the bins' OUTER boxes, not their interiors
    centers = layout.bin_interior_centers()
    half = layout.bin_half_extents()
    assert x_min <= float(centers[:, 0].min()) - float(half[0]) - layout.BIN_WALL
    assert x_max >= float(centers[:, 0].max()) + float(half[0]) + layout.BIN_WALL
    assert y_min <= float(centers[:, 1].min()) - float(half[1]) - layout.BIN_WALL
    assert y_max >= float(centers[:, 1].max()) + float(half[1]) + layout.BIN_WALL
    # z_top is the bin rim, the tallest thing that must stay in frame
    assert z_top == pytest.approx(float((centers[:, 2] + half[2]).max()))


def test_overhead_view_pose_frames_the_region():
    fov = 60.0
    pose = layout.overhead_view_pose(fov)
    assert _corners_visible(pose, fov)
    # centred on the region, and above its tallest feature
    (x_min, x_max), (y_min, y_max), z_top = layout.overhead_view_region()
    assert pose[0] == pytest.approx(0.5 * (x_min + x_max))
    assert pose[1] == pytest.approx(0.5 * (y_min + y_max))
    assert pose[2] > z_top


def test_overhead_view_pose_solves_at_the_rim_not_the_desk():
    """Regression guard on the one non-obvious step in the derivation.

    A downward camera magnifies whatever is closest to it, so a height chosen
    to cover the region at the DESK leaves the bin rims' outer corners outside
    the frame. Assert the shipped pose is strictly higher than that naive one,
    and that the naive one genuinely fails -- otherwise this test would pass
    for a derivation that had the bug.
    """
    fov = 60.0
    (x_min, x_max), (y_min, y_max), _ = layout.overhead_view_region()
    half_span = 0.5 * max(x_max - x_min, y_max - y_min) * 1.15
    naive_z = half_span / math.tan(math.radians(fov) / 2.0)  # solved at z = 0
    pose = layout.overhead_view_pose(fov)
    assert pose[2] > naive_z
    assert not _corners_visible((pose[0], pose[1], naive_z), fov)


@pytest.mark.parametrize(
    "scatter_x, scatter_y",
    [
        ((0.13, 0.29), (-0.18, 0.18)),  # today's zone
        ((0.18, 0.36), (-0.18, 0.18)),  # the pre-Task-10 zone
        ((0.05, 0.20), (-0.30, 0.30)),  # wider than the bins in y
        ((0.10, 0.55), (-0.05, 0.05)),  # longer than the bins in x
    ],
)
def test_overhead_view_pose_follows_a_moved_scatter_zone(monkeypatch, scatter_x, scatter_y):
    """The whole reason the pose is derived rather than hard-coded.

    SCATTER_X has already moved once (Task 10, (0.18, 0.36) -> (0.13, 0.29)).
    If it moves again the camera must re-frame itself, and nothing but a CPU
    test can catch a regression here -- the alternative is a GPU render
    silently losing half the zone.
    """
    monkeypatch.setattr(layout, "SCATTER_X", scatter_x)
    monkeypatch.setattr(layout, "SCATTER_Y", scatter_y)
    fov = 60.0
    assert _corners_visible(layout.overhead_view_pose(fov), fov)


def test_default_bin_layout_reproduces_current_constants():
    bl = layout.DEFAULT_BIN_LAYOUT
    assert layout.bin_front_x(bl) == pytest.approx(0.39)
    assert layout.top_down_x_limit(bl) == pytest.approx(layout.TOP_DOWN_X_LIMIT)
    assert layout.scatter_x(bl) == pytest.approx(layout.SCATTER_X)
    # No-arg calls must keep returning exactly what M1 built.
    assert torch.equal(layout.bin_interior_centers(), layout.bin_interior_centers(bl))
    assert layout.bin_static_boxes() == layout.bin_static_boxes(bl)


def test_bin_centers_follow_the_color_permutation():
    # Swap red (0) and green (3): red occupies the rightmost slot, green the leftmost.
    bl = layout.BinLayout(bin_x=0.44, bin_ys=(-0.21, -0.07, 0.07, 0.21), slot_colors=(3, 1, 2, 0))
    centers = layout.bin_interior_centers(bl)
    assert centers[0, 1] == pytest.approx(0.21)  # red -> slot 3 (y = +0.21)
    assert centers[3, 1] == pytest.approx(-0.21)  # green -> slot 0 (y = -0.21)
    boxes = {name: center for name, center, _ in layout.bin_static_boxes(bl)}
    assert boxes["bin_red_floor"][1] == pytest.approx(0.21)
    assert boxes["bin_green_floor"][1] == pytest.approx(-0.21)


def test_scatter_zone_tracks_jittered_bins():
    near = layout.BinLayout(
        bin_x=0.43, bin_ys=layout.DEFAULT_BIN_LAYOUT.bin_ys, slot_colors=(0, 1, 2, 3)
    )
    far = layout.BinLayout(
        bin_x=0.46, bin_ys=layout.DEFAULT_BIN_LAYOUT.bin_ys, slot_colors=(0, 1, 2, 3)
    )
    assert layout.top_down_x_limit(near) == pytest.approx(0.28)
    assert layout.top_down_x_limit(far) == pytest.approx(0.31)
    for bl in (near, far):
        lo, hi = layout.scatter_x(bl)
        assert hi == pytest.approx(layout.top_down_x_limit(bl))
        assert lo == pytest.approx(layout.SCATTER_X[0])


def test_sample_bin_layout_respects_envelope_and_separation():
    rng = random.Random(0)
    for _ in range(200):
        bl = layout.sample_bin_layout(rng, slot_colors=(2, 0, 3, 1))
        assert bl.slot_colors == (2, 0, 3, 1)
        assert -0.01 - 1e-9 <= bl.bin_x - 0.44 <= 0.02 + 1e-9
        for y0, dy in zip(bl.bin_ys, layout.DEFAULT_BIN_LAYOUT.bin_ys):
            assert abs(y0 - dy) <= 0.02 + 1e-9
        gaps = [b - a for a, b in zip(bl.bin_ys, bl.bin_ys[1:])]
        assert min(gaps) >= 0.11


class TestScatterCornerClip:
    """The clip is a TRACKING bound, not an IK one (Task D3, rulings D3-13 then D3-19).

    D3-13 set it to 0.315 from an IK-feasibility sweep. D3-19 replaced it with 0.280, the
    largest radius the arm was MEASURED to actually arrive at, after one trial failed
    identically in four gates at radius 0.3082 -- inside the old clip, with the IK reporting
    a zero residual and the path reporting continuous. These tests pin the second meaning,
    which is strictly stronger: everything the tracking bound admits, the IK bound admitted
    too.
    """

    def test_the_clip_excludes_the_measured_unreachable_corner(self):
        """(0.290, +-0.180) solved 0 of 6 jaw headings with 64 restarts each."""
        corner = math.hypot(layout.SCATTER_X[1], layout.SCATTER_Y[1])
        assert corner == pytest.approx(0.3413, abs=1e-3)
        assert layout.SCATTER_MAX_RADIUS < corner

    def test_the_clip_is_inside_the_ik_feasible_bound_on_purpose(self):
        """It deliberately eats IK-verified band, because IK feasibility was not binding.

        The old 0.315 came from "every fully-covered grid position was inside 0.3199 m".
        That bound is still true and is no longer the operative one: the arm solves poses it
        then fails to reach. So this asserts the ORDERING -- the tracking clip sits at or
        inside the IK bound -- rather than protecting the IK band from being trimmed.
        """
        assert layout.SCATTER_MAX_RADIUS <= 0.3199
        assert layout.SCATTER_MAX_RADIUS < 0.3257

    def test_the_clip_admits_every_measured_trackable_pose(self):
        """0.280 tracked clean at every azimuth measured (Task D3 part 15)."""
        assert layout.SCATTER_MAX_RADIUS >= 0.280
        for az_deg in (-40.0, -25.0, -10.0, 0.0, 10.0, 25.0, 40.0):
            az = math.radians(az_deg)
            x, y = 0.280 * math.cos(az), 0.280 * math.sin(az)
            assert math.hypot(*layout.clip_to_reach(x, y)) == pytest.approx(0.280)

    def test_the_clip_excludes_every_measured_stop_short(self):
        """Each of these was driven for real and the arm stopped 12-50 mm short."""
        for radius, az_deg in (
            (0.290, 0.0),
            (0.295, -10.0),
            (0.295, 10.0),
            (0.3082, -19.8),
            (0.315, -19.8),
        ):
            az = math.radians(az_deg)
            assert math.hypot(radius * math.cos(az), radius * math.sin(az)) > (
                layout.SCATTER_MAX_RADIUS
            )

    def test_the_whole_inner_edge_survives_the_clip(self):
        """The near edge is untouched: this trims reach, it does not shrink the zone."""
        for y in (layout.SCATTER_Y[0], 0.0, layout.SCATTER_Y[1]):
            assert math.hypot(layout.SCATTER_X[0], y) < layout.SCATTER_MAX_RADIUS

    def test_a_box_bound_could_not_do_this_job(self):
        """Why the constant is a radius and not a tighter SCATTER_X or SCATTER_Y.

        At 0.315 the argument was that only the CORNER was out of reach, so neither box
        bound separated it. At 0.280 the clip does reach the far edge at y=0, so that exact
        argument no longer holds -- but the conclusion does, and for a sharper reason: the
        clip excludes points whose x is well inside any x-trim that would keep the rest of
        the band, so no single box bound reproduces it.
        """
        # Excluded by the radius, yet its x sits inside the clip and its |y| inside the box:
        # an x-trim at SCATTER_MAX_RADIUS would wrongly keep it.
        x, y = 0.25, 0.18
        assert x < layout.SCATTER_MAX_RADIUS
        assert abs(y) <= layout.SCATTER_Y[1]
        assert math.hypot(x, y) > layout.SCATTER_MAX_RADIUS

    def test_clip_to_reach_leaves_reachable_points_alone(self):
        for x, y in ((0.13, 0.0), (0.21, 0.0), (0.27, 0.0), (0.13, 0.18), (0.20, 0.10)):
            assert layout.clip_to_reach(x, y) == (x, y)

    def test_clip_to_reach_pulls_in_the_far_edge_at_zero_azimuth(self):
        """New at 0.280 and worth pinning: the cut is no longer corner-only.

        SCATTER_X's upper bound is 0.29, outside the clip, so a brick sampled straight ahead
        at the far edge is now pulled in. At 0.315 it was left alone.
        """
        assert layout.clip_to_reach(layout.SCATTER_X[1], 0.0) == pytest.approx((0.280, 0.0))

    def test_clip_to_reach_pulls_the_bad_corner_in_along_its_radius(self):
        x, y = layout.clip_to_reach(layout.SCATTER_X[1], layout.SCATTER_Y[1])
        assert math.hypot(x, y) == pytest.approx(layout.SCATTER_MAX_RADIUS)
        # Radial: the azimuth is preserved, which is what the base yaw has to cover.
        assert math.atan2(y, x) == pytest.approx(
            math.atan2(layout.SCATTER_Y[1], layout.SCATTER_X[1])
        )
        # ...and it stays inside the scatter box, so a clipped brick is still on the desk.
        assert layout.SCATTER_X[0] <= x <= layout.SCATTER_X[1]
        assert layout.SCATTER_Y[0] <= y <= layout.SCATTER_Y[1]

    def test_clip_to_reach_is_idempotent(self):
        once = layout.clip_to_reach(layout.SCATTER_X[1], layout.SCATTER_Y[1])
        assert layout.clip_to_reach(*once) == pytest.approx(once)

    def test_clip_to_reach_handles_the_origin(self):
        assert layout.clip_to_reach(0.0, 0.0) == (0.0, 0.0)

    def test_every_corner_of_the_scatter_box_is_reachable_after_clipping(self):
        for x, y in itertools.product(layout.SCATTER_X, layout.SCATTER_Y):
            cx, cy = layout.clip_to_reach(x, y)
            assert math.hypot(cx, cy) <= layout.SCATTER_MAX_RADIUS + 1e-12


def test_spatial_scale_widens_the_camera_and_bin_x_draws_only():
    rng = random.Random(3)
    draws = [layout.sample_run_draw(rng, spatial_scale=3.0) for _ in range(300)]
    cam = [abs(v) for d in draws for v in d["overhead_cam_dpos"]]
    rot = [abs(v) for d in draws for v in d["overhead_cam_drot_deg"]]
    assert max(cam) > 0.01 and max(cam) <= 0.03 + 1e-9
    assert max(rot) > 2.0 and max(rot) <= 6.0 + 1e-9
    assert all(-0.03 - 1e-9 <= d["bin_x"] - layout._BIN_X <= 0.06 + 1e-9 for d in draws)
    assert any(d["bin_x"] - layout._BIN_X > 0.02 for d in draws)
    for d in draws:
        for y, nominal in zip(d["bin_ys"], layout._BIN_Y):
            assert abs(y - nominal) <= 0.02 + 1e-9
    assert layout.sample_run_draw(random.Random(3)) == layout.sample_run_draw(
        random.Random(3), spatial_scale=1.0
    )


def test_the_dish_fills_the_driven_reach_band():
    """Campaign d: the near edge is driven-bounded at 0.10 m, the far edge at 0.280 m."""
    near = layout.TRAY_CENTER[0] - layout.TRAY_RADIUS
    far = layout.TRAY_CENTER[0] + layout.TRAY_RADIUS
    assert near >= 0.10
    assert far <= layout.SCATTER_MAX_RADIUS
    assert far <= layout.top_down_x_limit(layout.DEFAULT_BIN_LAYOUT)
    # Carried forward from the rectangular tray's test: the far edge must also clear the
    # TIGHTEST bin layout the run draw can produce, not only the default one.
    tightest = layout.BinLayout(bin_x=0.43)  # the smallest bin_x the run draw produces
    assert far <= layout.top_down_x_limit(tightest)


def test_inside_tray_is_the_circle_and_the_bounds_are_its_box():
    cx, cy = layout.TRAY_CENTER
    assert layout.inside_tray(cx, cy)
    assert layout.inside_tray(cx + layout.TRAY_RADIUS - 0.001, cy)
    assert not layout.inside_tray(cx + layout.TRAY_RADIUS + 0.001, cy)
    assert layout.inside_tray(cx + layout.TRAY_RADIUS + 0.002, cy, margin=0.005)
    assert not layout.inside_tray(cx + layout.TRAY_RADIUS - 0.002, cy, margin=-0.005)
    (x0, x1), (y0, y1) = layout.tray_bounds()
    assert (x0, x1) == (cx - layout.TRAY_RADIUS, cx + layout.TRAY_RADIUS)
    assert (y0, y1) == (cy - layout.TRAY_RADIUS, cy + layout.TRAY_RADIUS)


def test_the_rim_segments_ring_the_dish_and_lean_outward():
    boxes = layout.tray_static_boxes()
    assert len(boxes) == layout.TRAY_SEGMENTS
    radii = []
    for name, center, size, quat in boxes:
        assert name.startswith("tray_rim_")
        assert len(center) == 3 and len(size) == 3 and len(quat) == 4
        radii.append(
            math.hypot(center[0] - layout.TRAY_CENTER[0], center[1] - layout.TRAY_CENTER[1])
        )
        assert center[2] > 0.0
    # Every segment sits at the same radius, and the ring closes: neighbouring centres are
    # no further apart than the segment's own length.
    assert max(radii) - min(radii) < 1e-9
    step = 2.0 * math.pi * radii[0] / layout.TRAY_SEGMENTS
    assert step <= boxes[0][2][1]


def test_a_part_at_the_rim_top_is_outside_the_interior_circle():
    """The rim leans out, so its top edge is TRAY_WALL_HEIGHT further out than its base."""
    top_radius = layout.TRAY_RADIUS + layout.TRAY_WALL_HEIGHT * math.tan(layout.TRAY_RIM_TILT_RAD)
    assert top_radius > layout.TRAY_RADIUS
    assert layout.TRAY_CENTER[0] + top_radius <= layout.SCATTER_MAX_RADIUS + layout.TRAY_WALL_HEIGHT


def test_the_heap_bin_holds_the_heap_and_keeps_the_reach_limit():
    solid_cm3 = 159.0  # a 50 part heldout heap, layout seed 20261010
    volume_cm3 = 1e6 * math.prod(layout.HEAP_BIN_INTERIOR)
    assert volume_cm3 / solid_cm3 > 5.0
    # The x half extent is what bin_front_x reads, and the dish's far edge depends on it.
    assert layout.HEAP_BIN_INTERIOR[0] == layout.BIN_INTERIOR[0]
    rim_z = layout._BIN_FLOOR_TOP_Z + layout.HEAP_BIN_INTERIOR[2]
    assert rim_z < oracle.RELEASE_Z - 0.05


def test_the_heap_bin_is_one_box_on_the_shelf_at_y_zero():
    boxes = layout.heap_bin_static_boxes(layout.DEFAULT_BIN_LAYOUT)
    assert len(boxes) == 5
    half_y = layout.HEAP_BIN_INTERIOR[1] / 2 + layout.BIN_WALL
    assert half_y <= layout.SHELF_SIZE[1] / 2
    for _name, center, _size in boxes:
        assert abs(center[1]) <= half_y + 1e-9
    (_x_range, y_range, _z) = layout.overhead_view_region()
    assert y_range[0] <= -half_y and y_range[1] >= half_y
