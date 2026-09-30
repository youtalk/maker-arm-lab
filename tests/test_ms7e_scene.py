"""MS7e design section 10.1: the three families and every scene switch meet in one heap. Round
physics, the LDraw hulls, the flat parts on a deep box floor, the per-episode box draw and the
B2 lean build one env, and a 50-part heap settles in the leaned tray."""

import argparse
import random

import numpy as np

from maker_arm_lab.core import batch_loop, desk_boxes, parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv

WAVE = [
    "--tray",
    "--desk_boxes",
    "8",
    "--tray_lean",
    "30",
    "--box_draw",
    "8",
    "--release_lean",
    "--families",
    "round,ldraw,flat",
    "--deep_box_floor",
]


def test_every_family_and_switch_meet_in_one_settled_heap():
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    batch_loop.add_ms7e_arguments(parser)
    argv = ["--out_dir", "/tmp/x", "--heap_k", "50", "--parts_split", "train", "--layout_seed"]
    args = parser.parse_args([*argv, "8", *WAVE])
    batch_loop.check_ms7e(parser, args)
    heap = batch_loop.heap_parts(args)
    families = {p.family for p in heap}
    assert {"brick", "round", "ldraw"} <= families and families & set(parts.FLAT_FAMILIES)
    env = MjLegoSortEnv(
        1,
        sample_run_draw(random.Random(7)),
        parts=heap,
        cameras=False,
        gripper_mode="continuous",
        desk_boxes=args.desk_boxes,
        tray=True,
        tray_lean_deg=args.tray_lean,
        class_names=batch_loop.heap_classes(args),
        box_draw=args.box_draw,
        deep_box_floor=True,
        release_lean=True,
        round_physics=True,
    )
    env.reset(seed=7)
    m, d, h = env.models[0], env.datas[0], env.handles
    assert env.class_names == parts.PART_CLASSES
    assert desk_boxes.BOX_COUNT_MIN <= env.box_count[0] <= desk_boxes.B2_CEILING
    floor = m.geom("desk_box_0_floor").id
    assert m.geom_pos[floor][2] - m.geom_size[floor][2] < -0.5 * desk_boxes.DESK_BOX_FLOOR_DEPTH
    z = np.array([d.qpos[a + 2] for a in h.brick_qpos])
    speed = np.array([np.linalg.norm(d.qvel[a : a + 3]) for a in h.brick_dof])
    # Every part on the desk or in the tray: none fell through, none flew off. The settle ends
    # with a few parts still rocking in place (main's heaps too: up to 3.5 rad/s, 0.02 m/s).
    assert np.isfinite(z).all() and z.min() > 0.0 and z.max() < 0.1, z
    assert np.median(speed) < 0.005 and speed.max() < 0.05, speed
