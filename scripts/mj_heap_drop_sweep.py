"""Experiment 1 of the MS6 lever verification design: the drop, the settle and the tray.

No teacher, no rendering, no recorder. One sample is a tray interior, a tray wall height, a
drop band, a settle length and a layout seed. The script patches the matching
``core/layout.py`` constants, builds one heap ``MjLegoSortEnv`` exactly as
``scripts/mj_generate_batch.py`` does, resets it (which drops the parts and settles the
physics), and at step zero measures what the fixed teacher would see: how many parts sit
outside the tray, how many have every grasp blocked by a wall or by a neighbor, how many rest
on another part, and how many are outside the arm's reach. It writes one JSON object per
sample to ``<out_dir>/samples.jsonl`` and prints one ``SWEEP_OK {json}`` or ``SWEEP_FAIL
{json}`` line.

Runs under the project's own venv, with ``MUJOCO_GL=egl``:

    MUJOCO_GL=egl uv run python scripts/mj_heap_drop_sweep.py --out_dir /tmp/s0 --seeds 5 \\
        --heap_k 50 --trays 130x200 --trays 170x200@180 --drop_bands 30:1.2 --drop_bands 15:0.6 \\
        --settle_steps 480 --settle_steps 960 --wall_heights 20 --wall_heights 40

``--trays`` takes ``WxH`` or ``WxH@cx``, where ``cx`` is the tray center's x in millimeters.
A candidate tray from the reach map comes with its own center: a wider tray at the layout's
default center can push its far corners past the arm's reach, so the sweep must honor
whatever center the candidate names, not assume ``layout.TRAY_CENTER``.

``measure_heap`` (below) is the pure part of this: part sizes, poses and quaternions, the
tray bounds, the wall footprints and the reach bounds in, the counts out. It never touches an
env, so a test can call it on a hand-placed layout. The env building and the ``core.layout``
patching live only in ``run_sample``.
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from maker_arm_lab.core import batch_loop, grasp, heap_oracle, layout
from maker_arm_lab.mj.env import MjLegoSortEnv

CURRENT_TRAY_MM = f"{layout.TRAY_INTERIOR[0] * 1000.0:g}x{layout.TRAY_INTERIOR[1] * 1000.0:g}"
CURRENT_WALL_HEIGHT_MM = layout.TRAY_WALL_HEIGHT * 1000.0
CURRENT_DROP_BAND_MM = (
    f"{layout.HEAP_DROP_Z_MIN * 1000.0:g}:{layout.HEAP_DROP_Z_PER_PART * 1000.0:g}"
)
CURRENT_SETTLE_STEPS = layout.HEAP_SETTLE_STEPS


def _half_height(size: torch.Tensor, quat: torch.Tensor) -> float:
    """Half the part's size along its most-vertical body axis (``candidate_grasps``' own rule)."""
    vertical = grasp.body_axes_world(quat)[:, 2].abs()
    return 0.5 * float(size[int(vertical.argmax())])


def measure_heap(
    sizes: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    tray_bounds: tuple[tuple[float, float], tuple[float, float]],
    wall_footprints: list[list[tuple[float, float]]],
    reach_x_limit: float,
    reach_radius: float,
) -> dict[str, int]:
    """The design's per-sample counts at step zero. Pure: no env, only geometry.

    ``outside``, ``stacked`` and ``unreachable`` read every part's position and size directly.
    ``wall_blocked``, ``blocked`` and ``no_grasp`` restrict to the parts inside the tray, in
    reach and under ``heap_oracle.CANDIDATE_MAX_Z`` -- the same mask ``HeapOracle._candidates``
    hands ``rank_grasps`` -- so they answer "of what the teacher would actually rank, how many
    rank empty", with the same wall footprints and the same ranking the teacher scans with.
    ``wall_blocked`` and ``blocked`` count a candidate with no grasp at all as blocked too, so
    they are not the same population as ``no_grasp``: a reader wanting only "ranked but every
    grasp fails" reads ``wall_blocked - no_grasp`` / ``blocked - no_grasp``.
    """
    n = sizes.shape[0]
    (x0, x1), (y0, y1) = tray_bounds
    inside = torch.zeros(n, dtype=torch.bool)
    reachable = torch.zeros(n, dtype=torch.bool)
    below_candidate_z = torch.zeros(n, dtype=torch.bool)
    outside = stacked = unreachable = 0
    for b in range(n):
        x, y, z = (float(v) for v in pos[b])
        is_inside = x0 <= x <= x1 and y0 <= y <= y1
        is_reachable = x <= reach_x_limit and math.hypot(x, y) <= reach_radius
        inside[b] = is_inside
        reachable[b] = is_reachable
        below_candidate_z[b] = z <= heap_oracle.CANDIDATE_MAX_Z
        outside += int(not is_inside)
        unreachable += int(not is_reachable)
        if z > _half_height(sizes[b], quat[b]):
            stacked += 1
    candidates = inside & reachable & below_candidate_z
    ranked = grasp.rank_grasps(sizes, pos, quat, candidates, wall_footprints)
    by_part: dict[int, list[grasp.Grasp]] = {}
    for g in ranked:
        by_part.setdefault(g.part, []).append(g)
    wall_blocked = blocked = no_grasp = 0
    for b in torch.nonzero(candidates).flatten().tolist():
        grasps = by_part.get(b, [])
        if not grasps:
            no_grasp += 1
        if not grasps or all(g.wall < grasp.CLEARANCE_MIN for g in grasps):
            wall_blocked += 1
        if not grasps or all(g.clearance < grasp.CLEARANCE_MIN for g in grasps):
            blocked += 1
    return {
        "outside": outside,
        "wall_blocked": wall_blocked,
        "blocked": blocked,
        "no_grasp": no_grasp,
        "stacked": stacked,
        "unreachable": unreachable,
    }


def _parse_tray(spec: str) -> tuple[tuple[float, float], float]:
    """``WxH`` or ``WxH@cx``, all in millimeters. ``cx`` is the tray center's x; a candidate
    tray with no center keeps today's ``layout.TRAY_CENTER`` x, so the reach map's centered
    candidates and a plain size both mean "at the layout's own place." The y center is always
    0: the arm and every candidate tray are symmetric in y."""
    size, _, cx = spec.lower().partition("@")
    w, _, h = size.partition("x")
    if not h:
        raise ValueError(f"--trays wants WxH or WxH@cx in millimeters, got {spec!r}")
    cx_mm = float(cx) if cx else layout.TRAY_CENTER[0] * 1000.0
    return (float(w), float(h)), cx_mm


def _parse_drop_band(spec: str) -> tuple[float, float]:
    lo, _, per = spec.partition(":")
    if not per:
        raise ValueError(f"--drop_bands wants min:per_part in millimeters, got {spec!r}")
    return float(lo), float(per)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=5, help="layout seeds per grid cell")
    parser.add_argument("--heap_k", type=int, default=50)
    parser.add_argument("--parts_split", choices=batch_loop.PARTS_SPLITS, default="heldout")
    parser.add_argument("--parts_seed", type=int, default=20260915)
    parser.add_argument("--trays", action="append", default=None, help="WxH in mm, repeatable")
    parser.add_argument(
        "--wall_heights", action="append", type=float, default=None, help="mm, repeatable"
    )
    parser.add_argument(
        "--drop_bands", action="append", default=None, help="min:per_part in mm, repeatable"
    )
    parser.add_argument("--settle_steps", action="append", type=int, default=None)
    args = parser.parse_args(argv)
    args.trays = args.trays or [CURRENT_TRAY_MM]
    args.wall_heights = args.wall_heights or [CURRENT_WALL_HEIGHT_MM]
    args.drop_bands = args.drop_bands or [CURRENT_DROP_BAND_MM]
    args.settle_steps = args.settle_steps or [CURRENT_SETTLE_STEPS]
    return args


def _base_batch_args(
    out_dir: str, heap_k: int, parts_split: str, parts_seed: int, layout_seed: int
) -> argparse.Namespace:
    """A ``core.batch_loop``-style args namespace, at its own defaults except the four the
    sweep varies. ``build_run`` and ``heap_parts`` read the rest through this."""
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    base = parser.parse_args(["--out_dir", str(out_dir)])
    base.heap_k = heap_k
    base.parts_split = parts_split
    base.parts_seed = parts_seed
    base.layout_seed = layout_seed
    base.num_envs = 1
    return base


@contextlib.contextmanager
def _patched_layout(
    tray_interior_m, tray_center_m, wall_height_m: float, drop_band_m, settle_steps: int
):
    """Sets the sample's ``core.layout`` constants for the scope of one env build and
    measurement, then restores them. Every reader below (``_drop_heap``, ``tray_bounds``,
    ``tray_static_boxes``, ``tray_wall_obstacles``) reads these module attributes at call
    time, which is what makes patching them from outside trustworthy."""
    names = (
        "TRAY_INTERIOR",
        "TRAY_CENTER",
        "TRAY_WALL_HEIGHT",
        "HEAP_DROP_Z_MIN",
        "HEAP_DROP_Z_PER_PART",
        "HEAP_SETTLE_STEPS",
    )
    saved = {name: getattr(layout, name) for name in names}
    layout.TRAY_INTERIOR = tray_interior_m
    layout.TRAY_CENTER = tray_center_m
    layout.TRAY_WALL_HEIGHT = wall_height_m
    layout.HEAP_DROP_Z_MIN = drop_band_m[0]
    layout.HEAP_DROP_Z_PER_PART = drop_band_m[1]
    layout.HEAP_SETTLE_STEPS = settle_steps
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(layout, name, value)


def _assert_tray_matches(model) -> None:
    """The trust check the design calls for: the compiled scene's wall geoms must be the
    sample's own tray, not a stale one from a constant read at import time somewhere else.
    ``layout.tray_static_boxes()`` folds ``TRAY_CENTER`` into every wall's position, so this
    catches a wrong center the same way it catches a wrong size."""
    for name, center, size in layout.tray_static_boxes():
        geom = model.geom(name)
        half = [0.5 * s for s in size]
        if not np.allclose(geom.pos, center, atol=1e-9) or not np.allclose(
            geom.size, half, atol=1e-9
        ):
            raise AssertionError(
                f"{name}: scene has pos={geom.pos.tolist()} size={geom.size.tolist()}, "
                f"sample tray wants pos={center} size={half}"
            )


def _measure_sample(env: MjLegoSortEnv, parts, bl: layout.BinLayout) -> dict:
    h, d = env.handles, env.datas[0]
    n = len(parts)
    sizes = torch.tensor([p.size for p in parts], dtype=torch.float32)
    pos = torch.zeros((n, 3), dtype=torch.float32)
    quat = torch.zeros((n, 4), dtype=torch.float32)
    speed_max = 0.0
    for b in range(n):
        adr = h.brick_qpos[b]
        pos[b] = torch.tensor(d.qpos[adr : adr + 3], dtype=torch.float32)
        quat[b] = torch.tensor(d.qpos[adr + 3 : adr + 7], dtype=torch.float32)
        dof = h.brick_dof[b]
        speed_max = max(speed_max, float(np.linalg.norm(d.qvel[dof : dof + 3])))
    counts = measure_heap(
        sizes,
        pos,
        quat,
        layout.tray_bounds(),
        grasp.tray_wall_obstacles(),
        layout.top_down_x_limit(bl),
        layout.SCATTER_MAX_RADIUS,
    )
    counts["speed_max"] = speed_max
    return counts


def run_sample(
    tray_mm: tuple[float, float],
    tray_cx_mm: float,
    wall_height_mm: float,
    drop_band_mm: tuple[float, float],
    settle_steps: int,
    layout_seed: int,
    args: argparse.Namespace,
) -> dict:
    """One grid cell's one layout seed: patch, build, reset, check, measure, restore."""
    tray_interior_m = (tray_mm[0] / 1000.0, tray_mm[1] / 1000.0)
    tray_center_m = (tray_cx_mm / 1000.0, 0.0)
    wall_height_m = wall_height_mm / 1000.0
    drop_band_m = (drop_band_mm[0] / 1000.0, drop_band_mm[1] / 1000.0)
    with _patched_layout(tray_interior_m, tray_center_m, wall_height_m, drop_band_m, settle_steps):
        base = _base_batch_args(
            args.out_dir, args.heap_k, args.parts_split, args.parts_seed, layout_seed
        )
        draw, bl, _noise = batch_loop.build_run(base)
        parts = batch_loop.heap_parts(base)
        env = MjLegoSortEnv(1, draw, num_bricks=len(parts), parts=parts, cameras=False)
        try:
            _assert_tray_matches(env.models[0])
            env.reset(seed=layout_seed)
            sample = _measure_sample(env, parts, bl)
        finally:
            env.close()
    sample.update(
        tray_w_mm=tray_mm[0],
        tray_h_mm=tray_mm[1],
        tray_cx_mm=tray_cx_mm,
        wall_height_mm=wall_height_mm,
        drop_min_mm=drop_band_mm[0],
        drop_per_part_mm=drop_band_mm[1],
        settle_steps=settle_steps,
        layout_seed=layout_seed,
        heap_k=len(parts),
    )
    return sample


def run(args: argparse.Namespace) -> tuple[int, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "samples.jsonl")
    trays = [_parse_tray(t) for t in args.trays]
    drop_bands = [_parse_drop_band(d) for d in args.drop_bands]
    grid = itertools.product(
        trays, args.wall_heights, drop_bands, args.settle_steps, range(args.seeds)
    )
    count = 0
    with open(out_path, "w") as f:
        for (
            (tray_mm, tray_cx_mm),
            wall_height_mm,
            drop_band_mm,
            settle_steps,
            layout_seed,
        ) in grid:
            sample = run_sample(
                tray_mm, tray_cx_mm, wall_height_mm, drop_band_mm, settle_steps, layout_seed, args
            )
            f.write(json.dumps(sample) + "\n")
            count += 1
    return count, out_path


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        count, out_path = run(args)
    except SystemExit as e:  # argparse
        print("SWEEP_FAIL " + json.dumps({"error": f"argparse exit {e.code}"}), flush=True)
        return 2
    except Exception as e:  # noqa: BLE001 -- any failure must become SWEEP_FAIL, not a crash.
        traceback.print_exc()
        print("SWEEP_FAIL " + json.dumps({"error": repr(e)}), flush=True)
        return 1
    print("SWEEP_OK " + json.dumps({"samples": count, "out": out_path}), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
