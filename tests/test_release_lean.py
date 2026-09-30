"""MS7e rung B2, the release lean (design sections 4.5 and 10.2): under ``--release_lean`` the
rig leans EXTEND toward its box and the overhead camera moves back; with it off, both are
MS7d's."""

import argparse
import itertools
import math
import pathlib
import random

import mujoco
import numpy as np
import pytest

from maker_arm_lab.core import batch_loop, cameras, desk_boxes, layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.quat import quat_mul
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj import overlay, scene
from maker_arm_lab.mj.env import HEAP_PARK_X, MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import LEAN_AZIMUTH, OracleRig
from test_mj_heap_oracle import _place

BRICK = parts.make_part("brick", 2, 4)
BEHIND = (-0.05, 0.19, 0.3)
"""A box behind the B1 band: with the lean pinned to +x, EXTEND has no plan there (the design
section 3.1 probe); leaning toward it, the part lands."""
PARKED = [(-0.6, 0.3 * j, 0.0) for j in (1, 2, 3)]


def _draw():
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


def _cycle(release_lean: bool) -> dict:
    """One lone-part cycle into the box at BEHIND, from SETTLE to the end of RETREAT.

    Records the phase and the commanded joint target of every step, and the phase and lean
    azimuth of every target the solver is asked to meet."""
    env = MjLegoSortEnv(
        1,
        _draw(),
        parts=[BRICK, BRICK],
        cameras=False,
        desk_boxes=4,
        gripper_mode="continuous",
        release_lean=release_lean,
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, [BEHIND, *PARKED])
    env.destination_bins[0] = 0
    _place(
        env,
        [(*layout.TRAY_CENTER, BRICK.size[2] / 2 + 0.001, 0.3), (HEAP_PARK_X, 0.0, 0.0, 0.0)],
    )
    rng = random.Random(1)

    def factory(i, bl, draw):
        return HeapOracle(
            bl, env.parts, env.destination_bins[i], draw, rng=rng, boxes=env.desk_box_poses[i]
        )

    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        rng,
        oracle_factory=factory,
        release_lean=release_lean,
    )
    leans = []
    solve, solve_path = rig.ik.solve, rig.ik.solve_path

    def record(targets):
        leans.extend((rig.goals[0].phase, t.lean_azimuth) for t in targets)

    rig.ik.solve = lambda target, **kw: (record([target]), solve(target, **kw))[1]
    rig.ik.solve_path = lambda targets, **kw: (record(targets), solve_path(targets, **kw))[1]
    steps = []
    for _ in range(3000):
        phase = rig.goals[0].phase
        out = rig.step()
        steps.append((phase, rig.q_cmd[0].clone()))
        goal = rig.goals[0]
        if phase == oracle.Phase.RETREAT and (goal is None or goal.phase != phase):
            break
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    result = {
        "steps": steps,
        "leans": leans,
        "stowed": bool(per_brick_stowed(env)[0, 0]),
        "stats": rig.episode_stats(0),
        "ik": rig.ik,
    }
    env.close()
    return result


@pytest.fixture(scope="module")
def cycles():
    return {lean: _cycle(lean) for lean in (False, True)}


def _last_q(steps, phase):
    return [q for p, q in steps if p == phase][-1]


def test_with_the_switch_off_every_target_leans_to_plus_x(cycles):
    """Off is MS7d exactly: every target the solver sees carries LEAN_AZIMUTH, the only input
    the switch changes, so every plan (and the leg resolution, whose lean change is 0.0) is
    main's. Behind the band that means no EXTEND plan, as the design probe measured."""
    off = cycles[False]
    assert {lean for _, lean in off["leans"]} == {LEAN_AZIMUTH}
    assert off["stats"]["unplanned_legs"] >= 1 and not off["stowed"]


def test_under_the_switch_extend_leans_toward_its_box(cycles):
    on = cycles[True]
    azimuth = math.atan2(BEHIND[1], BEHIND[0])
    extend = [lean for phase, lean in on["leans"] if phase == oracle.Phase.EXTEND]
    retreat = [lean for phase, lean in on["leans"] if phase == oracle.Phase.RETREAT]
    others = {
        lean
        for phase, lean in on["leans"]
        if phase not in (oracle.Phase.EXTEND, oracle.Phase.RETREAT)
    }
    # The release target is the box centre in float32 (HeapOracle), hence the tolerance.
    assert extend[0] == LEAN_AZIMUTH and extend[-1] == pytest.approx(azimuth, abs=1e-6)
    assert all(LEAN_AZIMUTH <= lean <= azimuth + 1e-6 for lean in extend)
    assert retreat[0] == extend[-1] and retreat[-1] == LEAN_AZIMUTH
    assert others == {LEAN_AZIMUTH}
    released = on["ik"].pose_of(_last_q(on["steps"], oracle.Phase.EXTEND))
    assert released.lean_azimuth == pytest.approx(azimuth, abs=1e-6)
    retreated = on["ik"].pose_of(_last_q(on["steps"], oracle.Phase.RETREAT))
    assert retreated.lean_azimuth == pytest.approx(LEAN_AZIMUTH, abs=1e-6)
    assert on["stowed"] and on["stats"]["unplanned_legs"] == 0, on["stats"]


def test_the_grasp_legs_do_not_change(cycles):
    """Every command before EXTEND, from SETTLE through LIFT, is the same with the switch on."""

    def grasp_legs(steps):
        out = []
        for phase, q in steps:
            if phase == oracle.Phase.EXTEND:
                return out
            out.append((phase, q.tolist()))
        raise AssertionError("no EXTEND")

    off, on = grasp_legs(cycles[False]["steps"]), grasp_legs(cycles[True]["steps"])
    assert {phase for phase, _ in on} >= {oracle.Phase.DESCEND, oracle.Phase.LIFT}
    assert off == on


# -- the overhead camera -------------------------------------------------------------------


def test_the_camera_moves_back_only_under_the_switch():
    draw = _draw()
    served = {}
    for lean in (False, True):
        env = MjLegoSortEnv(1, draw, parts=[BRICK], cameras=False, release_lean=lean)
        env.reset(seed=3)
        served[lean] = env.context_camera[0].copy()
        env.close()
    pos, rot = scene.overhead_camera_pose(draw)
    assert pos == [p + d for p, d in zip(cameras.OVERHEAD_CAM_POS, draw["overhead_cam_dpos"])]
    assert served[False][:3].tolist() == np.float32(pos).tolist()
    moved = served[True] - served[False]
    assert moved[0] == pytest.approx(-cameras.OVERHEAD_B2_BACK_M, abs=1e-6)
    assert np.all(moved[1:] == 0.0)
    for back in (0.0, cameras.OVERHEAD_B2_BACK_M):
        model, _ = scene.build_model(draw, 1, cameras_on=True, camera_back=back)
        c = model.camera("overhead").id
        assert model.cam_pos[c].tolist() == pytest.approx(scene.overhead_camera_pose(draw, back)[0])
        scene.apply_cameras(model, draw, back)
        assert model.cam_pos[c].tolist() == pytest.approx(scene.overhead_camera_pose(draw, back)[0])
    assert scene.overhead_camera_pose(draw, 0.0) == (pos, rot)


def _worst_margin(back: float, points) -> float:
    """Smallest pixel margin of ``points`` over the 64 corners of the episode's camera draw
    (+-0.01 m and +-2 degrees per axis) around the camera moved ``back`` m."""
    worst = math.inf
    for dpos in itertools.product((-0.01, 0.01), repeat=3):
        for drot in itertools.product((-2.0, 2.0), repeat=3):
            draw = {"overhead_cam_dpos": dpos, "overhead_cam_drot_deg": drot}
            pos, rot = scene.overhead_camera_pose(draw, back)
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.array(quat_mul(rot, cameras.ROS_TO_MUJOCO_CAM)))
            for p in points:
                u, v = overlay.project_point(pos, mat, cameras.OVERHEAD_FOV_DEG, 224, 224, p)
                worst = min(worst, u, v, 224 - u, 224 - v)
    return worst


def test_the_camera_shift_is_the_smallest_that_frames_every_b2_box_and_the_tray():
    """MS7e design section 4.4 item 3: every rim corner of a B2 box keeps 4 px inside the
    frame, and the leaned tray stays inside it, at the extremes of the camera draw."""
    (x0, x1), (y0, y1) = desk_boxes.B2_BAND_X, desk_boxes.B2_BAND_Y
    wall = desk_boxes.DESK_BOX_WALL
    hx = hy = desk_boxes.BOX_MAX / 2 + wall  # the largest drawn box, at the deepest rim
    rim = wall + desk_boxes.BOX_DEPTH[1]
    boxes = []
    for x, y, s, yaw in itertools.product((x0, x1), (y0, y1), (1.0, -1.0), (-45.0, 0.0, 45.0)):
        c, n = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for u, v in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
            boxes += [(x + c * u - n * v, s * y + n * u + c * v, z) for z in (0.0, rim)]
    (tx0, tx1), (ty0, ty1) = layout.tray_bounds()
    top = layout.TRAY_WALL_HEIGHT * math.tan(math.radians(30.0)) + layout.TRAY_WALL
    tray = [
        (x, y, layout.TRAY_WALL_HEIGHT)
        for x in (tx0 - top, tx1 + top)
        for y in (ty0 - top, ty1 + top)
    ]
    back = cameras.OVERHEAD_B2_BACK_M
    assert _worst_margin(back, boxes) >= 4.0
    assert _worst_margin(back - 0.001, boxes) < 4.0
    assert _worst_margin(back, tray) > 0.0
    assert _worst_margin(0.0, boxes) < 0.0  # the unshifted camera loses the back boxes


# -- the B2 box draw -----------------------------------------------------------------------


def test_under_the_switch_the_box_draw_takes_b2s_band_count_and_frame():
    env = MjLegoSortEnv(
        2,
        _draw(),
        parts=[BRICK] * 3,
        cameras=False,
        desk_boxes=4,
        tray=True,
        box_draw=8,
        release_lean=True,
        class_names=parts.PART_CLASSES,
        episode_draws=True,
    )
    (x0, x1), (y0, y1) = desk_boxes.B2_BAND_X, desk_boxes.B2_BAND_Y
    counts, behind_b1 = set(), False
    for seed in range(12):
        env.reset(seed=seed)
        for i in range(2):
            n = int(env.box_count[i])
            counts.add(n)
            poses, sizes = env.desk_box_poses[i, :n], env.desk_box_sizes[i, :n]
            assert desk_boxes.clear(poses, sizes)
            for pose, size in zip(poses, sizes):
                assert x0 <= pose[0] <= x1 and y0 <= abs(pose[1]) <= y1
                back = min(c[0] for c in desk_boxes.corners(pose, size))
                assert back >= desk_boxes.B2_FRAME_BACK_X - 1e-12
                behind_b1 |= pose[0] < desk_boxes.B1_BAND_X[0]
    env.close()
    assert counts == {4, 5, 6, 7, 8} and behind_b1


def test_the_unseen_rim_row_keeps_b2s_rim_frame():
    assert desk_boxes.B2_UNSEEN_DRAWS["size"] == desk_boxes.UNSEEN_DRAWS["size"]
    assert desk_boxes.B2_UNSEEN_DRAWS["rim"]["depth"] == desk_boxes.UNSEEN_DRAWS["rim"]["depth"]
    env = MjLegoSortEnv(
        1,
        _draw(),
        parts=[BRICK],
        cameras=False,
        desk_boxes=4,
        box_draw=8,
        box_unseen="rim",
        release_lean=True,
    )
    assert env._box_draw_args["back"] == desk_boxes.B2_UNSEEN_DRAWS["rim"]["back"]
    assert env._box_draw_args["band_x"] == desk_boxes.B2_BAND_X
    env.close()


def _checked(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--desk_boxes", type=int, default=0)
    batch_loop.add_ms7e_arguments(parser)
    args = parser.parse_args(argv)
    batch_loop.check_ms7e(parser, args)
    return args


def test_eight_boxes_need_the_switch_and_the_switch_is_built_in_every_client():
    assert "release_lean" not in batch_loop.UNBUILT_SWITCHES
    with pytest.raises(SystemExit):
        _checked(["--desk_boxes", "4", "--box_draw", "8"])
    assert _checked(["--desk_boxes", "4", "--box_draw", "8", "--release_lean"]).box_draw == 8
    with pytest.raises(SystemExit):
        _checked(["--desk_boxes", "4", "--box_draw", "9", "--release_lean"])
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    for name in ("mj_eval_heap.py", "mj_generate_batch.py"):
        # the env (the camera and the band) and the rig (the lean)
        assert (root / name).read_text().count("release_lean=args.release_lean") == 2
