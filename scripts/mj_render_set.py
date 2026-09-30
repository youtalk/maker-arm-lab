"""The MS7f render set (design section 6.4): the classifier's three views on teacher cycles.

One process is one run seed: ``--num_envs`` heaps of ``--heap_k`` parts, driven by the heap
teacher as ``scripts/mj_generate_batch.py`` builds it (``OracleRig`` with ``HeapOracle``, the same
flags), until each env's first episode ends or its teacher is DONE. Nothing is exported, and the
env renders no frame per step: ``env.cameras`` goes off after the build and each view is rendered
at its event from the state the heap loop's frame of that step shows.

* overhead: the step the teacher selects a target, as the heap loop under ``--select_at_release``
  does: its preselection when RELEASE ends, or a HOVER on a part it had not selected (the first
  cycle, a cycle with no preselection). One frame jitter (``--frame_jitter``), then the narrow
  view (``MjLegoSortEnv.narrow_overhead``) of the target and of up to ``--extra_overhead`` other
  parts in the tray, each aimed at the part's centre plus a perceived-like error
  (``AIM_SIGMA_M``). Only parts the perceived state would see (``perception._hidden``): the
  loop never classifies a hidden part. The arm may still stand between the camera and the part,
  as it may in the loop, so each overhead frame records how much of the part the camera sees.
* lift: the wrist frame of the step a ``hold.HoldDetector`` fed the loop's inputs (the measured
  jaw gap, the last command's gap, the FK pick point's height) reads the lift.
* carry: the wrist frame ``loop.CARRY_VIEW_STEPS`` later, if the detector still reads a lift.
* wrist (``--wrist_every S``, off by default): more wrist frames of the same lift for training,
  every S steps up to twice the carry view's offset while the detector reads a lift.

Labels read the simulator, a recording tool and not a decision (design section 6.4): the part's
category and name and whether its kind is held out; on a wrist frame the part both jaws press
(``checker.held_brick_from_forces``, as ``loop.wrist_class``; -1 and category "" with none) and
the passenger count, ``oracle.passengers`` against the heights at the HOVER start, and of those
the riders, which rose at least ``RIDE_SHARE`` of the jaws' own rise since the hold began, with
the largest share of a rider the wrist camera sees in its frame (``rider_seen``); on an overhead
frame the share of the part the camera sees (``part_seen``, ``_seen``); on every frame with a
part the angle of the part's own z axis, its studs' axis, from world up (``tilt``: near 0 upright,
near 180 upside down, a side between; the tray's lean leans only its walls). T2's rule reads
the LIFT gate 0.10 m up and the lift frame is 0.05 m up, so the riders guard the label against a
neighbour the grasp only nudged. Measured on the first four re-rendered C1 heaps (2026-09-30) they
equal the passengers on 27 of 28 wrist frames: the passengers ride, and most of 16 such lift
frames looked at show none of them, hidden under or behind the held part.

    MUJOCO_GL=egl PYTHONPATH=$PWD/source .venv/bin/python scripts/mj_render_set.py \\
        --out_dir /tmp/rs --num_envs 1 --heap_k 50 --cycle_budget 65 --parts_split train \\
        --run_seed 20302000 --layout_seed 20302000 --nearest_first_pick_order \\
        --hover_jitter_min 0.035 --hover_jitter_max 0.080 --episode_draws --frame_jitter \\
        --gripper continuous --episode_length_s 1500 --tray --desk_boxes 8 --tray_lean 30 \\
        --box_draw 8 --release_lean --carry_via --passenger_check \\
        --families round,ldraw,flat --deep_box_floor

Writes ``<out_dir>/render_<run_seed>.npz``: ``frames`` (N, 224, 224, 3) uint8 and one (N,) array
per ``LABELS`` name, plus ``args`` (the command line as JSON). Prints exactly one ``RENDER_OK`` or
``RENDER_FAIL`` line on every exit path, argparse errors included, and exits through
``os._exit``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from maker_arm_lab.core import layout, oracle, perception
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core.batch_loop import (
    add_batch_arguments,
    add_ms7e_arguments,
    build_run,
    check_ms7e,
    heap_classes,
    heap_parts,
    make_induction,
)
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.hold import HoldDetector, commanded_gap
from maker_arm_lab.core.loop import CARRY_VIEW_STEPS
from maker_arm_lab.mj import overlay
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.robots.frames import GRIPPER_JOINT_NAMES
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

AIM_SIGMA_M = tuple(
    math.hypot(b, n) for b, n in zip(perception.POS_BIAS_SIGMA_M, perception.POS_NOISE_SIGMA_M)
)
"""m per axis: the narrow view aims at the part's centre plus this Gaussian error, the perceived
centre's episode bias and per-look noise together, as the loop aims at a detection."""
RIDE_SHARE = 0.5  # INITIAL: the C1 read's passenger label
"""A passenger rides when it rose at least this share of the jaws' rise since the hold began."""
LABELS = (
    "view",
    "category",
    "part",
    "heldout",
    "passengers",
    "riders",
    "held",
    "part_seen",
    "rider_seen",
    "env",
    "step",
    "tilt",
)
"""The label arrays beside ``frames``: the view (overhead, lift, carry), the category and name of
the part the view is of, whether its kind is held out, the passenger and rider counts (-1 on an
overhead frame), the part index (-1 on a wrist frame with nothing between the jaws), the share of that
part the overhead camera sees (-1 on a wrist frame), the largest share of a rider the wrist
camera sees in its frame (-1 with no rider or on an overhead frame), the env, the control
step and the part's tilt in degrees (``_tilt``, -1 with no part)."""


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("RENDER_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    add_batch_arguments(parser)
    add_ms7e_arguments(parser)
    parser.add_argument(
        "--wrist_every",
        type=int,
        default=0,
        help="extra wrist frames (view 'wrist') every this many steps after a lift; 0 is off",
    )
    parser.add_argument(
        "--extra_overhead",
        type=int,
        default=3,
        help="narrow views of other seen parts in the tray per HOVER start, besides the target",
    )
    args = parser.parse_args(argv)
    check_ms7e(parser, args)
    if args.heap_k <= 0 or args.desk_boxes <= 0 or args.no_cameras:
        parser.error(
            "the render set records heap cycles into desk boxes: it needs --heap_k > 0, "
            "--desk_boxes > 0 and the cameras"
        )
    return args


def build(args):
    """The env and the teacher rig, as ``scripts/mj_generate_batch.py`` builds them, exporting
    nothing and rendering nothing per step."""
    draw, bl, noise = build_run(args)
    parts = heap_parts(args)
    env = MjLegoSortEnv(
        args.num_envs,
        draw,
        num_bricks=len(parts),
        parts=parts,
        cameras=True,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        episode_length_s=args.episode_length_s or 1500.0,
        episode_draws=args.episode_draws,
        frame_jitter=args.frame_jitter,
        gripper_mode=args.gripper,
        desk_boxes=args.desk_boxes,
        tray=args.tray,
        tray_lean_deg=args.tray_lean,
        class_names=heap_classes(args),
        box_draw=args.box_draw,
        box_unseen=args.box_unseen,
        deep_box_floor=args.deep_box_floor,
        release_lean=args.release_lean,
        round_physics="round" in args.families,
    )
    env.cameras = False  # every view is rendered on demand at its event
    env.reset(seed=args.run_seed)
    induction = make_induction(args)
    rig_rng = random.Random(args.run_seed)

    def factory(i, layout_i, episode_draw):
        return HeapOracle(
            layout_i,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            induction=induction,
            rng=rig_rng,
            cycle_budget=args.cycle_budget,
            boxes=env.desk_box_poses[i],
            box_sizes=env.desk_box_sizes[i],
            tray=env.tray,
            tray_lean_deg=env.tray_lean_deg,
            wall_band_cont=args.wall_band_cont,
        )

    rig = OracleRig(
        env,
        bl,
        noise,
        rig_rng,
        induction=induction,
        transport_speed=args.transport_speed,
        oracle_factory=factory,
        release_lean=args.release_lean,
        carry_via=args.carry_via,
        passenger_check=args.passenger_check,
    )
    return env, rig


def _positions(env, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Env ``i``'s part centres (K, 3) and orientations (K, 4) now."""
    data = [env.scene[f"brick_{b}"].data for b in range(env.cfg.num_bricks)]
    return torch.stack([d.root_pos_w[i] for d in data]), torch.stack(
        [d.root_quat_w[i] for d in data]
    )


def _tilt(quat) -> float:
    """Degrees between the z axis of a part at orientation ``quat`` (w, x, y, z) and world up."""
    _, x, y, _ = (float(v) for v in quat)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _held(env, i: int) -> int:
    """The part both jaws press, or -1: ``loop.wrist_class``'s read."""
    forces = [
        env.scene[f"finger_{s}_brick"].data.force_matrix_w[i : i + 1] for s in ("left", "right")
    ]
    left, right = (f.norm(dim=-1).squeeze(1) for f in forces)
    return int(held_brick_from_forces(left, right, min_force=HELD_MIN_FORCE)[0])


def _in_tray(xy) -> bool:
    (x0, x1), (y0, y1) = layout.tray_bounds()
    return x0 <= float(xy[0]) <= x1 and y0 <= float(xy[1]) <= y1


def _seen(env, i: int, b: int, cam: int | None = None) -> float:
    """The share of part ``b``'s nine sample points, its centre and its bounding box's corners
    pulled 40 % in, that a ray from the camera ``cam`` (the overhead one by default) meets on
    the part first, of those inside the camera's frame: 0 hidden, 1 in the open. (A
    segmentation render would count pixels, but its multisampled edges blend geom ids past the
    scene's, and MuJoCo 3.13's remap then raises.)"""
    m, d, h = env.models[i], env.datas[i], env.handles
    cam = env._cam_overhead if cam is None else cam
    eye = d.cam_xpos[cam].copy()
    body = h.brick_bodies[b]
    centre, rot = d.xpos[body], d.xmat[body].reshape(3, 3)
    half = 0.3 * np.asarray(env.parts[b].size)
    corners = [rot @ (half * np.array(s)) for s in itertools.product((-1.0, 1.0), repeat=3)]
    geom = np.zeros(1, dtype=np.int32)
    hits = 0
    fovy = float(m.cam_fovy[cam])
    for point in [centre] + [centre + c for c in corners]:
        uv = overlay.project_point(eye, d.cam_xmat[cam], fovy, 224, 224, point)
        if uv is None or not (0.0 <= uv[0] < 224.0 and 0.0 <= uv[1] < 224.0):
            continue
        mujoco.mj_ray(m, d, eye, point - eye, None, 1, -1, geom)
        hits += int(geom[0]) in h.body_geoms[b]
    return hits / 9.0


def _overhead_views(env, i, target, pick, aim_rng, extra, jitter) -> list[tuple]:
    """The narrow views of one selection: (frame, part, share seen, tilt) for the target and up to
    ``extra`` other parts in the tray, each only if the perceived state would see it."""
    pos, quat = _positions(env, i)
    sizes = torch.tensor([p.size for p in env.parts])
    hidden = perception._hidden(pos, quat, sizes, tuple(pick[:2].tolist()))
    seen = [b for b in range(len(env.parts)) if b not in hidden and _in_tray(pos[b])]
    others = [b for b in seen if b != target]
    aim_rng.shuffle(others)
    chosen = ([target] if target in seen else []) + others[:extra]
    if chosen and jitter:
        env.jitter_overhead(i)  # one frame: the real rig crops every part out of it
    views = []
    for b in chosen:
        aim = [float(v) + s * aim_rng.gauss(0.0, 1.0) for v, s in zip(pos[b], AIM_SIGMA_M)]
        views.append((env.narrow_overhead(i, aim), b, _seen(env, i, b), _tilt(quat[b])))
    return views


def record(env, rig, args, heldout: set[str], deadline: float) -> tuple[list, list]:
    """Drive the teacher and render the views; returns (frames, label rows)."""
    n = env.num_envs
    robot = env.scene["robot"]
    jaw_ids = robot.find_joints(list(GRIPPER_JOINT_NAMES))[0]
    detectors = [HoldDetector() for _ in range(n)]
    aims = [random.Random(f"render:{args.run_seed}:{i}") for i in range(n)]
    start_z: list[torch.Tensor | None] = [None] * n
    carry_at = [-1] * n
    lift_at = [-1] * n
    selected = [None] * n  # the part of the env's last selection
    preselected = [None] * n  # and the teacher's last preselection
    finished = [False] * n
    gap_cmd = robot.data.joint_pos[:, jaw_ids].sum(dim=-1).tolist()
    frames, rows = [], []

    def keep(frame, view, part, passengers, seen, i, step, tilt, riders=-1, rider_seen=-1.0):
        cat = _parts.part_category(env.parts[part]) if part >= 0 else ""
        name = env.parts[part].name if part >= 0 else ""
        frames.append(frame)
        row = (
            view,
            cat,
            name,
            name in heldout,
            passengers,
            riders,
            part,
            seen,
            rider_seen,
            i,
            step,
            tilt,
        )
        rows.append(row)

    step = 0
    while not all(finished):
        if time.time() > deadline:
            raise TimeoutError(f"wall clock at step {step}")
        gaps = robot.data.joint_pos[:, jaw_ids].sum(dim=-1)
        for i in range(n):
            if finished[i]:
                continue
            d = detectors[i]
            pick = rig.pick_point(i).to("cpu")
            d.update(float(gaps[i]), gap_cmd[i], float(pick[2]))
            goal, orc = rig.goals[i], rig.oracles[i]
            hover = goal is not None and goal.phase == oracle.Phase.HOVER
            hover = hover and rig.steps_in_phase[i] == 0
            target = None
            nxt = getattr(orc, "next_brick", None)
            if nxt is not None and nxt != preselected[i]:
                target = preselected[i] = nxt  # the preselection when RELEASE ends
            elif hover and orc.target_brick != selected[i]:
                target = orc.target_brick  # a selection with no preselection
            if hover:
                preselected[i] = None
                start_z[i] = _positions(env, i)[0][:, 2].clone()
            if target is not None:
                selected[i] = target
                views = _overhead_views(
                    env, i, target, pick, aims[i], args.extra_overhead, args.frame_jitter
                )
                for frame, b, seen, tilt in views:
                    keep(frame, "overhead", b, -1, seen, i, step, tilt)
            wrist = []
            if d.lift_started and start_z[i] is not None:
                wrist.append("lift")
                carry_at[i], lift_at[i] = step + CARRY_VIEW_STEPS, step
            if step == carry_at[i] and d.lifted:
                wrist.append("carry")
            since = step - lift_at[i]
            every = args.wrist_every
            extra = every and d.lifted and 0 < since <= 2 * CARRY_VIEW_STEPS and since % every == 0
            if extra and step != carry_at[i]:
                wrist.append("wrist")
            for view in wrist:
                held = _held(env, i)
                pos, quat = _positions(env, i)
                others = oracle.passengers(pos, start_z[i], pick, held)
                jaws = RIDE_SHARE * (float(pick[2]) - d.start_z)
                rode = [b for b in others if float(pos[b, 2] - start_z[i][b]) >= jaws]
                r = env.renderers[i]
                r.update_scene(env.datas[i], camera="wrist")
                shown = max((_seen(env, i, b, env._cam_wrist) for b in rode), default=-1.0)
                tilt = _tilt(quat[held]) if held >= 0 else -1.0
                frame = r.render().copy()
                keep(frame, view, held, len(others), -1, i, step, tilt, len(rode), shown)
            if orc is not None and orc.phase in (oracle.Phase.DONE, oracle.Phase.ABANDON):
                finished[i] = True
        action = rig.actions()
        gap_cmd = [commanded_gap(float(g), env.gripper_mode) for g in action[:, 6]]
        _, _, terminated, truncated, _ = env.step(action)
        out = rig.observe((terminated | truncated).cpu())
        for i in (out["dones"] | out["abandoned"]).nonzero().flatten().tolist():
            finished[i] = True
        step += 1
    return frames, rows


def run(args) -> dict:
    os.makedirs(args.out_dir, exist_ok=True)
    _, held = _parts.split(_parts.catalog(args.families), args.parts_seed)
    env, rig = build(args)
    try:
        frames, rows = record(
            env, rig, args, {p.name for p in held}, time.time() + 60.0 * args.max_wall_minutes
        )
    finally:
        env.close()
    if not frames:
        raise RuntimeError("no view was rendered")
    columns = {name: np.array(col) for name, col in zip(LABELS, zip(*rows))}
    out = Path(args.out_dir) / f"render_{args.run_seed}.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, frames=np.stack(frames), args=np.array(json.dumps(vars(args))), **columns
    )
    tmp.replace(out)
    views = columns["view"].tolist()
    return {
        "out": str(out),
        "frames": len(frames),
        "by_view": {v: views.count(v) for v in ("overhead", "lift", "carry", "wrist")},
        "passenger_frames": int((columns["passengers"] > 0).sum()),
        "rider_frames": int((columns["riders"] > 0).sum()),
        "empty_wrist": int(((columns["held"] < 0) & (columns["view"] != "overhead")).sum()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
        print("RENDER_OK " + json.dumps(summary), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("RENDER_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())
