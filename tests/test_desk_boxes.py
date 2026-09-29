"""MS7a desk boxes: the side-band draw, the yawed-box grader and the overhead coverage."""

import argparse
import itertools
import math
import pathlib
import random
import subprocess
import sys
import types

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import (
    batch_loop,
    cameras,
    checker,
    desk_boxes,
    grasp,
    heap_oracle,
    layout,
    loop,
    oracle,
    parts,
    stowed,
)
from maker_arm_lab.core.grasp import polygon_distance
from maker_arm_lab.core.quat import compose_offset_rot, quat_mul
from maker_arm_lab.mj import overlay
from test_mj_desk_boxes import FIXED


def _tray_outer():
    (x0, x1), (y0, y1) = layout.tray_bounds()
    w = layout.TRAY_WALL
    return [(x0 - w, y0 - w), (x1 + w, y0 - w), (x1 + w, y1 + w), (x0 - w, y1 + w)]


@pytest.mark.parametrize("n", [1, 4])
def test_the_band_lies_beside_the_arm_never_behind_it_or_in_front_of_the_tray(n):
    (x0, x1), (y0, y1) = desk_boxes.band_x(n), desk_boxes.DESK_BOX_BAND_Y
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
            assert desk_boxes.in_band(pose, n), pose
            assert abs(pose[2]) <= math.radians(45.0) + 1e-12
            print_ = desk_boxes.corners(pose, desk_boxes.interior(n))
            assert polygon_distance(print_, corridor) >= desk_boxes.DESK_BOX_CLEARANCE
            assert polygon_distance(print_, _tray_outer()) >= desk_boxes.DESK_BOX_CLEARANCE
            # Never under the arm base either: every corner stays off the base disc.
            assert min(math.hypot(*c) for c in print_) > desk_boxes.ARM_BASE_RADIUS
        for a, b in itertools.combinations(poses, 2):
            size = desk_boxes.interior(n)
            gap = polygon_distance(desk_boxes.corners(a, size), desk_boxes.corners(b, size))
            assert gap >= desk_boxes.DESK_BOX_GAP
        if n == 4:
            # Two per side: boxes 0 and 2 on the left (+y), 1 and 3 on the right.
            assert [math.copysign(1.0, p[1]) for p in poses] == [1.0, -1.0, 1.0, -1.0]
        else:
            sides.add(math.copysign(1.0, poses[0][1]))
    if n == 1:
        assert sides == {1.0, -1.0}  # rung 1 draws the side per episode


@pytest.mark.parametrize("n", [4, 5, 6])
def test_the_box_draw_keeps_its_sizes_its_band_its_frame_and_its_gaps(n):
    corridor = desk_boxes.pick_corridor()
    (x0, x1), (y0, y1) = desk_boxes.B1_BAND_X, desk_boxes.B1_BAND_Y
    for seed in range(40):
        poses, sizes, extra = desk_boxes.draw_boxes(random.Random(seed), n)
        assert (poses, sizes, extra) == desk_boxes.draw_boxes(random.Random(seed), n)
        assert extra == 0
        assert abs(sum(math.copysign(1.0, p[1]) for p in poses)) == n % 2
        for pose, (ix, iy, iz) in zip(poses, sizes):
            assert x0 <= pose[0] <= x1 and y0 <= abs(pose[1]) <= y1, pose
            assert abs(pose[2]) <= desk_boxes.DESK_BOX_MAX_YAW + 1e-12
            assert desk_boxes.BOX_MIN <= min(ix, iy) and max(ix, iy) <= desk_boxes.BOX_MAX
            assert max(ix, iy) <= desk_boxes.BOX_MAX_ASPECT * min(ix, iy)
            assert desk_boxes.BOX_DEPTH[0] <= iz <= desk_boxes.BOX_DEPTH[1]
            print_ = desk_boxes.corners(pose, (ix, iy, iz))
            assert min(c[0] for c in print_) >= desk_boxes.FRAME_BACK_X - 1e-12
            assert polygon_distance(print_, corridor) >= desk_boxes.DESK_BOX_CLEARANCE
        for a, b in itertools.combinations(range(n), 2):
            gap = polygon_distance(
                desk_boxes.corners(poses[a], sizes[a]), desk_boxes.corners(poses[b], sizes[b])
            )
            assert gap >= desk_boxes.DESK_BOX_GAP


def test_a_side_the_band_cannot_hold_fails_its_draw():
    with pytest.raises(RuntimeError):
        desk_boxes.draw_boxes(random.Random(0), 7, rounds=2)
    with pytest.raises(RuntimeError):
        desk_boxes.draw_boxes(random.Random(0), 5, band_x=desk_boxes.DESK_BOX_BAND_X, rounds=2)


def test_a_side_that_exhausts_its_size_draws_draws_again_and_counts_the_rounds():
    """MS7e ledger D7: five boxes place on B1's band in every ceiling seed at 200 size draws,
    so 10 size draws a round exhaust some rounds; the count stays and the rounds show."""
    extras = []
    for seed in range(20290000, 20290040):
        poses, _, extra = desk_boxes.draw_boxes(random.Random(seed), 5, size_draws=10)
        assert len(poses) == 5
        extras.append(extra)
    assert max(extras) > 0 and min(extras) == 0


def test_a_seed_reproduces_the_draw():
    a = desk_boxes.sample_boxes(random.Random(7), 4)
    b = desk_boxes.sample_boxes(random.Random(7), 4)
    assert a == b and desk_boxes.sample_boxes(random.Random(8), 4) != a


def test_four_boxes_draw_a_class_permutation_and_one_box_takes_every_class():
    rng = random.Random(3)
    maps = {desk_boxes.class_map(rng, 4) for _ in range(200)}
    assert all(sorted(m) == [0, 1, 2, 3] for m in maps) and len(maps) == 24
    state = rng.getstate()
    assert desk_boxes.class_map(rng, 1) == (0, 0, 0, 0) and rng.getstate() == state
    heap = [parts.make_part("slope", 1, 2), parts.make_part("brick", 1, 2, 5)]
    heap += [parts.make_part("brick", 1, 6), parts.make_part("brick", 2, 3)]
    # BASE_CLASSES order: slope, tall, narrow, wide.
    assert desk_boxes.part_destinations(heap, (2, 0, 3, 1)) == [2, 0, 3, 1]
    assert desk_boxes.part_destinations(heap * 2, (0, 0, 0, 0)) == [0] * 8


def test_a_box_footprint_on_the_corridor_is_refused():
    x0, y0 = desk_boxes.DESK_BOX_BAND_X[0], desk_boxes.DESK_BOX_BAND_Y[0]
    one, many = desk_boxes.interior(1), desk_boxes.interior(4)
    assert not desk_boxes.clear([(layout.TRAY_CENTER[0], 0.12, 0.0)], [one])  # on the tray wall
    assert not desk_boxes.clear([(0.1, 0.3, 0.0), (0.1, 0.35, 0.0)], [many] * 2)  # overlap
    assert desk_boxes.clear([(x0 + 0.05, y0 + 0.03, 0.0)], [many])
    # The one box of rung 1 is 2 cm wider, so the same pose can clear as one of four and not
    # alone: a 45 degree yaw at the band's inner edge beside the tray.
    tight = (layout.TRAY_CENTER[0], 0.20, math.radians(45.0))
    assert desk_boxes.clear([tight], [many]) and not desk_boxes.clear([tight], [one])


def test_one_box_is_the_11_cm_box_and_four_keep_the_9_cm_box():
    assert desk_boxes.interior(1) == (0.11, 0.11, 0.06)
    assert desk_boxes.interior(4) == layout.BIN_INTERIOR == (0.09, 0.09, 0.04)
    assert desk_boxes.rim_z(desk_boxes.interior(1)) == pytest.approx(0.065)
    z, half = desk_boxes.stow_region(desk_boxes.interior(1))
    assert z - half[2] == pytest.approx(desk_boxes.DESK_BOX_WALL)  # the band starts at the floor
    assert half == pytest.approx((0.055, 0.055, 0.04))
    # The band stops under the release: a part held there over the box is not in it yet.
    assert z + half[2] < desk_boxes.DESK_BOX_RELEASE_Z - 0.01
    z4, half4 = desk_boxes.stow_region(desk_boxes.interior(4))
    assert z4 == pytest.approx(0.045) and half4 == pytest.approx((0.045, 0.045, 0.04))


# -- the grader ------------------------------------------------------------------------------

POSES = [(0.08, 0.22, math.radians(40.0)), (0.08, -0.22, math.radians(-30.0))]
Z, HALF = desk_boxes.stow_region(desk_boxes.interior(len(POSES)))


def _boxes():
    return torch.tensor([[(x, y, Z, yaw) for x, y, yaw in POSES]])


def _in_box(pose, u, v, z):
    """A world point at (u, v) in the box frame, height z."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * u - s * v, y + s * u + c * v, z)


def test_a_part_at_rest_in_its_own_yawed_box_counts_and_nowhere_else():
    floor = desk_boxes.DESK_BOX_WALL
    half_x = desk_boxes.interior(len(POSES))[0] / 2
    pos = torch.tensor(
        [
            [
                _in_box(POSES[0], 0.03, -0.03, floor + 0.005),  # own box (0), near a corner
                _in_box(POSES[0], 0.0, 0.0, floor + 0.005),  # at rest in box 0, not its own
                (0.2, 0.0, 0.005),  # on the desk, in no box
                # inside the box's axis-aligned bounding square, outside the yawed interior
                _in_box(POSES[1], half_x + 0.004, half_x + 0.004, floor + 0.005),
                _in_box(POSES[1], 0.0, 0.01, floor + 2.2 * desk_boxes.interior(2)[2]),
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


@pytest.mark.parametrize("n", [1, 4])
def test_every_box_corner_at_every_band_extreme_is_in_the_overhead_frame(n):
    (x0, x1), (y0, y1) = desk_boxes.band_x(n), desk_boxes.DESK_BOX_BAND_Y
    rim = desk_boxes.rim_z(desk_boxes.interior(n))
    centres = [(x, s * y) for x in (x0, x1) for y in (y0, y1) for s in (1.0, -1.0)]
    points = [
        (*c, z)
        for x, y in centres
        for yaw in (-desk_boxes.DESK_BOX_MAX_YAW, 0.0, desk_boxes.DESK_BOX_MAX_YAW)
        for c in desk_boxes.corners((x, y, yaw), desk_boxes.interior(n))
        for z in (0.0, rim)
    ]
    worst = math.inf
    for pos, mat in _camera_poses():
        for p in points:
            u, v = overlay.project_point(pos, mat, cameras.OVERHEAD_FOV_DEG, 224, 224, p)
            worst = min(worst, u, v, 224 - u, 224 - v)
    assert worst > 0.0, worst


# -- the teacher on stub snapshots -----------------------------------------------------------

BRICK = parts.make_part("brick", 2, 4)


def _snapshot(positions):
    pos = torch.tensor(positions, dtype=torch.float32)
    n = pos.shape[0]
    return oracle.SceneSnapshot(
        brick_pos=pos,
        brick_yaw=torch.zeros(n),
        brick_lin_vel=torch.zeros(n, 3),
        pick_point=torch.tensor([0.15, 0.0, 0.12]),
        brick_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone(),
    )


def _teacher(destination):
    return heap_oracle.HeapOracle(
        layout.DEFAULT_BIN_LAYOUT,
        [BRICK] * 3,
        destination,
        oracle.nominal_draw(),
        rng=random.Random(0),
        boxes=POSES,
    )


def test_the_teacher_releases_over_the_parts_own_box_at_the_pick_tilt_with_no_raise():
    orc = _teacher(torch.tensor([1, 0, 1]))
    assert orc.part_boxes == [1, 0, 1] and orc.destination is None
    for part, box in ((0, 1), (1, 0)):
        x, y, _ = POSES[box]
        target = orc._release_target(part)
        assert target.tolist() == pytest.approx([x, y, desk_boxes.DESK_BOX_RELEASE_Z])
    assert orc.release_tilt == oracle.PICK_TILT_RAD and not orc.raises
    assert orc.retreat_point == desk_boxes.DESK_BOX_RETREAT_POINT
    # A lifted part goes straight to EXTEND, whose goal is the part's own box.
    scene = _snapshot([(0.19, -0.04, 0.005), (0.19, 0.0, 0.005), (0.19, 0.04, 0.005)])
    orc.phase, orc.target_brick = oracle.Phase.LIFT, 1
    orc.report(oracle.PhaseResult(outcome="converged", lifted=True), scene)
    assert orc.phase == oracle.Phase.EXTEND
    goal = orc.next_goal(scene)
    assert goal.target_pos[:2].tolist() == pytest.approx(POSES[0][:2], abs=1e-6)
    assert goal.tilt_rad == oracle.PICK_TILT_RAD


def test_the_teacher_grades_its_own_box_and_never_targets_a_boxed_part():
    orc = _teacher(0)  # one index for every part
    assert orc.part_boxes == [0, 0, 0]
    floor = desk_boxes.DESK_BOX_WALL + 0.005
    scene = _snapshot(
        [_in_box(POSES[0], 0.0, 0.0, floor), _in_box(POSES[1], 0.0, 0.0, floor), (0.19, 0, 0.005)]
    )
    assert orc._stowed(scene) == {0}  # part 1 rests in box 1, not its own
    assert orc._candidates(scene).tolist() == [False, False, True]


def test_the_ms6_constructor_path_is_unchanged():
    orc = heap_oracle.HeapOracle(
        layout.DEFAULT_BIN_LAYOUT, [BRICK], torch.tensor(1), oracle.nominal_draw()
    )
    assert orc.destination == 1 and orc.part_boxes is None and orc.raises
    assert orc.release_tilt == oracle.RELEASE_TILT_RAD
    assert orc.retreat_point == oracle.RETREAT_POINT


def test_the_box_walls_stand_where_the_box_geoms_put_them_on_a_yawed_box():
    """MS7d L9: four walls a box, their outer corners the footprint's ``corners``, their inner
    faces the interior's, their tops the rim, ``outward`` away from the box centre, the
    heading along the wall and no lean."""
    n = len(FIXED)
    walls = desk_boxes.box_walls(FIXED, desk_boxes.interiors(n))
    assert len(walls) == 4 * n
    ix, _, _ = desk_boxes.interior(n)
    for k, pose in enumerate(FIXED):
        mine = walls[4 * k : 4 * k + 4]
        vertices = [p for w in mine for p in w.footprint]
        for corner in desk_boxes.corners(pose, desk_boxes.interior(n)):
            assert min(math.dist(corner, p) for p in vertices) < 1e-12
        for w in mine:
            assert w.box and w.lean == 0.0 and w.top == desk_boxes.rim_z(desk_boxes.interior(n))
            assert grasp.wall_distance(pose[:2], w) == pytest.approx(ix / 2)
            cx = sum(x for x, _ in w.footprint) / 4 - pose[0]
            cy = sum(y for _, y in w.footprint) / 4 - pose[1]
            ox, oy = w.outward
            assert (cx * ox + cy * oy, cx * oy - cy * ox) == pytest.approx(
                ((ix + desk_boxes.DESK_BOX_WALL) / 2, 0.0), abs=1e-12
            )
            assert math.cos(w.heading) * ox + math.sin(w.heading) * oy == pytest.approx(0.0)


def _beside_box_wall(box, gap, turn=0.0):
    """A 2x4 brick lying flat outside the wall of box ``box`` that faces the tray, its centre
    ``gap`` (m) off the wall's outer face at the wall's middle, its long axis turned ``turn``
    off the wall; and the wall's outward normal."""
    pose = FIXED[box]
    side = -1.0 if pose[1] > 0.0 else 1.0
    _, hy = desk_boxes.outer_half(desk_boxes.interior(len(FIXED)))
    x, y, _ = _in_box(pose, 0.0, side * (hy + gap), 0.0)
    yaw = pose[2] + turn
    scene = oracle.SceneSnapshot(
        brick_pos=torch.tensor([[x, y, BRICK.size[2] / 2]]),
        brick_yaw=torch.tensor([yaw]),
        brick_lin_vel=torch.zeros(1, 3),
        pick_point=torch.tensor([0.3, 0.0, 0.1]),
        brick_quat=torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]),
    )
    return scene, (-side * math.sin(pose[2]), side * math.cos(pose[2]))


def _box_wall_selector(on):
    """The heap loop's selector on the protocol's leaning tray, with the box walls or not."""
    sel = heap_oracle.HeapOracle(
        layout.DEFAULT_BIN_LAYOUT, [BRICK], 0, oracle.nominal_draw(), boxes=FIXED, tray=True
    )
    sel.walls = grasp.tray_walls(30.0) + (
        desk_boxes.box_walls(FIXED, desk_boxes.interiors(len(FIXED))) if on else ()
    )
    return sel


def test_a_box_wall_refuses_a_finger_across_it_and_keeps_the_closing_axis_along_it():
    """MS7d L9: a 2x4 brick 2 mm off the outside of a yawed box wall, its long axis along the
    wall. Across the wall a finger lands on it, so that grasp is refused; along the wall the
    stroke clears. The tray walls alone rank both, as with the switch off."""
    scene, _ = _beside_box_wall(0, 0.002 + BRICK.size[1] / 2)
    mask = torch.tensor([True])
    along = FIXED[0][2]
    off = [g.heading for g in _box_wall_selector(False)._ranked(scene, mask)]
    on = _box_wall_selector(True)
    assert sorted(off) == pytest.approx([along, along + math.pi / 2])
    assert [g.heading for g in on._ranked(scene, mask)] == pytest.approx([along])
    assert loop.box_wall_refusals(on, scene, 0) == 1
    assert loop.box_wall_refusals(_box_wall_selector(False), scene, 0) == 0


def test_the_rake_sweeps_a_part_off_a_box_wall_away_from_the_box():
    """MS7d L9: a brick 12 mm off a box wall at 50 degrees ranks no grasp once the box walls
    count, so the selector rakes it off the wall along the box's outward normal: the tips land
    clear of every box wall, and the hull at the end clears every wall, farther from the box.
    With the tray walls alone the brick is a forced pick."""
    scene, (ox, oy) = _beside_box_wall(3, 0.012, math.radians(50.0))
    assert _box_wall_selector(False).select_target(scene) == 0
    sel = _box_wall_selector(True)
    assert sel.select_target(scene) is None and sel.push is not None
    rake = sel.push
    assert rake.part == 0 and rake.box_wall
    assert math.cos(rake.heading) * ox + math.sin(rake.heading) * oy == pytest.approx(1.0)
    boxes = [w for w in sel.walls if w.box]
    tips = grasp.rake_footprint(rake.start)
    assert all(grasp.polygon_distance(tips, list(w.footprint)) > 0.0 for w in boxes)
    assert grasp._clear_of_walls(rake.end, sel.walls)
    box = FIXED[3][:2]
    assert math.dist(rake.end, box) > math.dist(scene.brick_pos[0, :2].tolist(), box)


def test_desk_boxes_is_a_batch_argument_recorded_in_run_meta():
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    args = parser.parse_args(["--out_dir", "/tmp/x"])
    assert args.desk_boxes == 0
    args = parser.parse_args(["--out_dir", "/tmp/x", "--desk_boxes", "4"])
    draw = layout.sample_run_draw(random.Random(0))
    meta = batch_loop.run_meta(
        args,
        draw,
        oracle.OracleNoise(),
        None,
        num_bricks=6,
        export_mode="EXPORT_ALL",
        backend="mujoco",
        task="t",
    )
    assert meta["desk_boxes"] == 4


@pytest.mark.parametrize("argv", [["--band_x", "nope"], ["--interior", "0.07,0.07"]])
def test_the_envelope_check_prints_its_fail_token_on_a_usage_error(argv):
    script = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_release_envelope_check.py"
    )
    out = subprocess.run(
        [sys.executable, str(script), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert out.stdout.strip().splitlines()[-1].startswith("ENVELOPE_FAIL ")


def test_the_heap_clients_take_the_desk_boxes_flag():
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    for name in ("mj_eval_heap.py", "mj_dagger_heap.py"):
        text = (root / name).read_text()
        assert '"--desk_boxes"' in text and "desk_boxes=args.desk_boxes" in text, name
        assert 'payload["desk_boxes"] = args.desk_boxes' in text, name
    for name in (
        "mj_eval_heap.py",
        "mj_dagger_heap.py",
        "mj_generate_batch.py",
        "mj_inspect_heap.py",
    ):
        text = (root / name).read_text()
        assert "boxes=env.desk_box_poses[i]" in text and "int(env.destination_bins[i])" not in text


def test_the_deep_floor_reaches_under_the_desk_and_keeps_its_top_and_walls():
    """MS7e flat track, lever 1: only the floor's collider grows, downward."""
    w = desk_boxes.DESK_BOX_WALL
    for n in (1, 4):
        shallow = desk_boxes.box_geoms(desk_boxes.interior(n))
        deep = desk_boxes.box_geoms(desk_boxes.interior(n), desk_boxes.DESK_BOX_FLOOR_DEPTH)
        assert deep[1:] == shallow[1:]
        (_, (_, _, cz), (sx, sy, sz)), (_, _, (sx0, sy0, sz0)) = deep[0], shallow[0]
        assert (sx, sy, sz0) == (sx0, sy0, w)
        assert cz + sz / 2 == pytest.approx(w) and cz - sz / 2 == pytest.approx(
            -desk_boxes.DESK_BOX_FLOOR_DEPTH
        )


def test_deep_box_floor_is_a_batch_argument_recorded_in_run_meta():
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    assert not parser.parse_args(["--out_dir", "/tmp/x"]).deep_box_floor
    args = parser.parse_args(["--out_dir", "/tmp/x", "--desk_boxes", "4", "--deep_box_floor"])
    meta = batch_loop.run_meta(
        args,
        layout.sample_run_draw(random.Random(0)),
        oracle.OracleNoise(),
        None,
        num_bricks=6,
        export_mode="EXPORT_ALL",
        backend="mujoco",
        task="t",
    )
    assert meta["deep_box_floor"] is True
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    assert "deep_box_floor=args.deep_box_floor" in (root / "mj_generate_batch.py").read_text()
    text = (root / "mj_eval_heap.py").read_text()
    assert '"--deep_box_floor"' in text and "deep_box_floor=args.deep_box_floor" in text
    assert 'payload["deep_box_floor"] = args.deep_box_floor' in text
