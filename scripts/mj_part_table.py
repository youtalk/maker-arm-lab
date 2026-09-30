"""The lone-part table (MS7e design section 6.2): the envelope check's trial on a list of parts.

One trial drops ONE part inside the tray of the MS7d scene (four desk boxes, the tray walls
leaned 30 degrees) and lets it settle: a uniform point over the tray interior inset by the
part's half diagonal plus ``layout.TRAY_DROP_INSET``, a uniform random orientation, the lowest
corner ``DROP_HEIGHT_M`` over the floor, so it comes to rest on a random face. The heap teacher
(``HeapOracle`` with the desk boxes, the tray and its lean, noise and induction off) then runs
through the rig and the physics, from SETTLE until the oracle closes its first cycle: the end
of RETREAT, or the give-up after the grasp retries. This is the loop of
``scripts/mj_release_envelope_check.py``, with the part and the boxes drawn instead of fixed.

Seeds (ledger R1): part p's trials take their seeds from
``random.Random(base + parts.catalog_index(p))``, one ``randrange(2**31)`` per trial, with the
base 20289000 unless ``--seed_base`` moves it. A trial's seed resets the env (box poses, class
map, physics DR), seeds the rig and, through ``"pose:<seed>"``, the drop. So a trial reads the
same in any worker and any order.

The reads, all off the driven arm and the part:

* lifted: a LIFT that the rig's own gate called lifted (the part rose ``LIFT_MIN_RISE`` and
  stays inside ``LIFT_MAX_XY_ERROR`` of the pick point) and the oracle carried on.
* stowed: ``core.stowed``'s grader, the part at rest in its class's box after the cycle, with
  the envelope check's ``SETTLE_STEPS`` of hold.
* end_phase: ``<ended>><next>``, the phase that closed the cycle and the oracle's phase after
  it (``RETREAT>HOVER``, ``LIFT>REOPEN`` for the give-up after the grasp retries,
  ``PUSH_SWEEP>DONE`` for a teacher that found no grasp and stopped), ``TIMEOUT:<phase>``
  after ``MAX_STEPS``, or ``SETTLE_OUT`` for a part whose settle left the tray (a failure, no
  cycle runs).
  drop_phase: the phase in which a lifted part stopped riding in the jaws.
* at each LIFT: the jaw gap (m), the two jaw contact forces on the part (N, left and right)
  and the part's z, at the start (after the close) and at the gate, so a squeeze-out (the gap
  near closed, no force) and a one-sided slide (one force) show apart.

    MUJOCO_GL=egl PYTHONPATH=$PWD/source .venv/bin/python scripts/mj_part_table.py \\
        --families round --trials 12 --procs 6 --out /tmp/round.json

The scene follows the heap clients: the eight classes when a part lies beyond brick and slope
or the box draw is on (``batch_loop.heap_classes``), ``round_physics`` on a round part, and
``--deep_box_floor``, ``--box_draw`` and ``--release_lean`` as the recorder takes them.

Prints exactly one ``TABLE_OK {json}`` or ``TABLE_FAIL {json}`` line on every exit path,
argparse errors included, and exits through ``os._exit``. A trial that raised is recorded
with its traceback and makes the token ``TABLE_FAIL``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import torch

import maker_arm_lab
from maker_arm_lab.core import desk_boxes, layout, oracle
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.oracle import LIFT_MIN_RISE, Phase
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import HEAP_PARK_X, MjLegoSortEnv, _random_quat
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
    GRIPPER_OPEN_ACTION,
    LIFT_MAX_XY_ERROR,
    OracleRig,
)
from mj_release_envelope_check import EPISODE_LENGTH_S, MAX_STEPS, SETTLE_STEPS, _arm_box_contact

TABLE_SEED_BASE = 20289000
"""Ledger R1: part p's trial generator is seeded ``TABLE_SEED_BASE + parts.catalog_index(p)``.
``--seed_base`` moves the base: the MS7f ledger's R1 seeds its tables at 20303000."""
DESK_BOXES = 4
TRAY_LEAN_DEG = 30.0
"""The MS7d scene: ``--tray --desk_boxes 4 --tray_lean 30``."""
DROP_HEIGHT_M = 0.005
"""The drop's lowest possible corner over the tray floor: the centre starts one half diagonal
plus this high, so no orientation starts inside the floor."""


def part_order() -> list[str]:
    """Every part name the table knows, in the catalog order (``parts.catalog_index``)."""
    return [p.name for p in _parts.catalog(tuple(_parts.FAMILY_KEYS))]


def family(name: str) -> str:
    """The family a row pools into: the name's prefix, with plates and tiles pooled as flat."""
    prefix = name.split("_")[0]
    return "flat" if prefix in ("plate", "tile") else prefix


FAMILIES = ("brick", "slope", "round", "ldraw", "flat", "plate", "tile", "joint")


def family_parts(fam: str) -> list[str]:
    return [n for n in part_order() if family(n) == fam or n.split("_")[0] == fam]


def trial_seeds(name: str, trials: int, base: int = TABLE_SEED_BASE) -> list[int]:
    """Ledger R1: the seeds of part ``name``'s trials 0 .. trials-1."""
    rng = random.Random(base + _parts.catalog_index(name))
    return [rng.randrange(2**31) for _ in range(trials)]


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("TABLE_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--parts", type=str, help="comma-separated part names")
    which.add_argument(
        "--families", type=str, help=f"comma-separated families of {', '.join(FAMILIES)}"
    )
    parser.add_argument("--trials", type=int, default=12, help="trials per part")
    parser.add_argument("--procs", type=int, default=1, help="local worker processes")
    parser.add_argument("--out", type=str, default="", help="write the table JSON here")
    parser.add_argument(
        "--seed_base",
        type=int,
        default=TABLE_SEED_BASE,
        help="part p's trials and scene draw take base + parts.catalog_index(p)",
    )
    parser.add_argument(
        "--deep_box_floor",
        action="store_true",
        help="MS7e flat lever 1: each box floor's collider reaches under the desk top",
    )
    parser.add_argument(
        "--box_draw", type=int, default=0, help="MS7e: draw 4 to N desk boxes per trial"
    )
    parser.add_argument(
        "--release_lean", action="store_true", help="MS7e: the B2 rung, EXTEND leans to its box"
    )
    args = parser.parse_args(argv)
    if args.trials < 1 or args.procs < 1:
        parser.error("--trials and --procs must be positive")
    hi = desk_boxes.B2_CEILING if args.release_lean else desk_boxes.B1_CEILING
    if args.box_draw and not desk_boxes.BOX_COUNT_MIN <= args.box_draw <= hi:
        parser.error(f"--box_draw draws {desk_boxes.BOX_COUNT_MIN} to at most {hi} boxes")
    if args.families:
        fams = args.families.split(",")
        unknown = [f for f in fams if f not in FAMILIES]
        if unknown:
            parser.error(f"unknown families {unknown}; known: {list(FAMILIES)}")
        names = [n for f in fams for n in family_parts(f)]
    else:
        names = args.parts.split(",")
    args.names = list(dict.fromkeys(names))
    for name in args.names:
        try:
            _parts.catalog_index(name)
            _parts.by_name([name])
        except (KeyError, ValueError):
            parser.error(f"part {name!r} is not constructible or not in the MS7e catalog order")
    return args


def _grip(env, rig) -> dict:
    """The jaw gap (m), the two jaw forces on part 0 (N) and part 0's z (m), now."""
    pinch = rig.pinch_evidence(0, 0)
    return {
        "gap_m": round(float(rig.robot.data.joint_pos[0, rig.finger_ids].sum()), 5),
        "force_n": [round(pinch["left"], 3), round(pinch["right"], 3)],
        "z_m": round(float(env.scene["brick_0"].data.root_pos_w[0, 2]), 4),
    }


def _riding(env, rig) -> bool:
    """Part 0 up in the jaws, on the rig's own held-part numbers (``OracleRig._held_part``)."""
    pos = env.scene["brick_0"].data.root_pos_w[0]
    xy_error = float((pos[:2] - rig.pick_point(0)[:2]).norm())
    return float(pos[2]) > LIFT_MIN_RISE and xy_error < LIFT_MAX_XY_ERROR


def _drop_part(env, rng: random.Random) -> dict:
    """Part 0 dropped at a random tray pose and settled; its twin parked off the desk.

    The parked twin is never stowed, so the env's success never fires and never auto-resets
    the episode under the trial (the envelope check's ``_place_part``).
    """
    m, d, h = env.models[0], env.datas[0], env.handles
    part = env.parts[0]
    x, y = layout.sample_tray_point(rng, part.half_diagonal + layout.TRAY_DROP_INSET)
    drop = (x, y, part.half_diagonal + DROP_HEIGHT_M)
    for b, (xyz, quat) in enumerate(
        ((drop, _random_quat(rng)), ((HEAP_PARK_X, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    ):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 3] = xyz
        d.qpos[adr + 3 : adr + 7] = quat
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=layout.HEAP_SETTLE_STEPS)
    env._refresh([0], images=False)
    body = h.brick_bodies[0]
    pos = d.xpos[body]
    # Row 2 of the body rotation holds the world z of each body axis: the one near +-1 is up.
    up = d.xmat[body].reshape(3, 3)[2]
    axis = max(range(3), key=lambda k: abs(up[k]))
    (x0, x1), (y0, y1) = layout.tray_bounds()
    grow = layout.TRAY_WALL_HEIGHT * math.tan(math.radians(env.tray_lean_deg))  # the rim
    inside = x0 - grow <= pos[0] <= x1 + grow and y0 - grow <= pos[1] <= y1 + grow
    return {
        "drop": [round(v, 4) for v in drop],
        "rest": [round(float(v), 4) for v in pos],
        "face": ("+" if up[axis] > 0 else "-") + "xyz"[axis],
        "settle_out": not inside,
    }


def _part_end(env) -> str:
    """Where part 0 ended: ``own_box`` or ``other_box`` (its centre over that box's outer
    footprint; ``:under_floor`` when the centre lies under the box floor's top, a part that
    went through the floor), ``tray``, ``desk`` or ``off`` (below the desk top)."""
    pos = env.scene["brick_0"].data.root_pos_w[0]
    for k, pose in enumerate(env.desk_box_poses[0]):
        if desk_boxes.in_outer_footprint(pos[:2], [pose], [env.desk_box_sizes[0, k]], margin=0.0):
            where = "own_box" if k == int(env.destination_bins[0, 0]) else "other_box"
            return where + (":under_floor" if float(pos[2]) < desk_boxes.DESK_BOX_WALL else "")
    (x0, x1), (y0, y1) = layout.tray_bounds()
    if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1:
        return "tray"
    return "desk" if float(pos[2]) > -0.01 else "off"


def run_trial(env, seed: int) -> dict:
    start = time.monotonic()
    env.reset(seed=seed)
    record = _drop_part(env, random.Random(f"pose:{seed}"))
    record.update(
        box=int(env.destination_bins[0, 0]),
        lifts=[],
        lifted=False,
        stowed=False,
        end_phase="SETTLE_OUT",
        drop_phase=None,
        part_end=None,
        steps=0,
        arm_box_contact_steps=0,
    )
    if record["settle_out"]:
        record["part_end"] = _part_end(env)
        record["wall_s"] = round(time.monotonic() - start, 2)
        return record
    rng = random.Random(seed)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            boxes=env.desk_box_poses[i],
            box_sizes=env.desk_box_sizes[i],
            tray=env.tray,
            tray_lean_deg=env.tray_lean_deg,
        )

    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        rng,
        oracle_factory=factory,
        release_lean=env.release_lean,
    )
    orc = rig.oracles[0]
    m = env.models[0]
    box_geoms = {g for g in range(m.ngeom) if m.body_mocapid[m.geom_bodyid[g]] >= 0}
    lifts = record["lifts"]
    prev, touches, steps = rig.goals[0], 0, 0
    record["end_phase"] = None
    while steps < MAX_STEPS:
        out = rig.step()
        steps += 1
        touches += _arm_box_contact(env, box_geoms)
        goal = rig.goals[0]
        if goal is not prev:
            if prev.phase == Phase.LIFT:
                lifts[-1]["gate"] = _grip(env, rig)
                lifts[-1]["lifted"] = goal is not None and goal.phase in (Phase.RAISE, Phase.EXTEND)
                record["lifted"] |= lifts[-1]["lifted"]
            if goal is not None and goal.phase == Phase.LIFT:
                lifts.append(
                    {"width_m": round(float(orc.grasp_width), 5), "close": _grip(env, rig)}
                )
        now = goal.phase.name if goal is not None else prev.phase.name
        if record["lifted"] and record["drop_phase"] is None and not _riding(env, rig):
            record["drop_phase"] = now
        if orc.cycle_outcomes or goal is None or bool(out["dones"][0] | out["abandoned"][0]):
            ended = prev.phase.name if goal is not prev else now
            record["end_phase"] = f"{ended}>{orc.phase.name}"
            break
        prev = goal
    if record["end_phase"] is None:
        record["end_phase"] = f"TIMEOUT:{prev.phase.name}"
    hold = torch.zeros(1, 7)
    hold[0, :6] = rig.q_cmd[0].float()
    hold[0, 6] = GRIPPER_OPEN_ACTION
    for _ in range(SETTLE_STEPS):
        if bool(per_brick_stowed(env)[0, 0]):
            break
        env.step(hold)
    record.update(
        stowed=bool(per_brick_stowed(env)[0, 0]),
        part_end=_part_end(env),
        end_xyz=[round(float(v), 4) for v in env.scene["brick_0"].data.root_pos_w[0]],
        steps=steps,
        arm_box_contact_steps=touches,
        abandon=rig.stats[0]["abandon"],
        wall_s=round(time.monotonic() - start, 2),
    )
    return record


_ENVS: dict[tuple, MjLegoSortEnv] = {}


def scene_switches(args) -> dict:
    """The run's scene switches, one set for every part (the heap clients' rule)."""
    beyond = {p.family for p in _parts.by_name(args.names)} - {"brick", "slope"}
    return {
        "eight_classes": bool(beyond or args.box_draw),
        "deep_box_floor": args.deep_box_floor,
        "box_draw": args.box_draw,
        "release_lean": args.release_lean,
    }


def _env(name: str, switches: dict, base: int) -> MjLegoSortEnv:
    """One env per part per worker process, built once; every trial resets it."""
    key = (name, base, *switches.values())
    if key not in _ENVS:
        (part,) = _parts.by_name([name])
        _ENVS[key] = MjLegoSortEnv(
            1,
            layout.sample_run_draw(random.Random(base + _parts.catalog_index(name))),
            parts=[part, part],
            cameras=False,
            episode_length_s=EPISODE_LENGTH_S,
            gripper_mode="continuous",
            desk_boxes=DESK_BOXES,
            tray=True,
            tray_lean_deg=TRAY_LEAN_DEG,
            class_names=_parts.PART_CLASSES if switches["eight_classes"] else _parts.BASE_CLASSES,
            box_draw=switches["box_draw"],
            deep_box_floor=switches["deep_box_floor"],
            release_lean=switches["release_lean"],
            round_physics=part.family == "round",  # MS7e design section 6.4
        )
    return _ENVS[key]


def _task(job: tuple[str, int, int, dict, int]) -> dict:
    name, t, seed, switches, base = job
    torch.set_num_threads(1)  # one core per worker; small tensors never thread anyway
    try:
        record = run_trial(_env(name, switches, base), seed)
    except Exception as err:  # noqa: BLE001 -- the table records it and fails its token
        record = {
            "error": f"{err}\n{traceback.format_exc()}",
            "lifts": [],
            "lifted": False,
            "stowed": False,
            "settle_out": False,
            "end_phase": "ERROR",
            "drop_phase": None,
            "part_end": None,
            "steps": 0,
            "wall_s": 0.0,
        }
    return {"part": name, "trial": t, "seed": seed, **record}


def _median(values):
    return round(statistics.median(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    """One table row: counts, the failure census and the medians at the first LIFT's gate."""
    n = len(records)
    fails = [r for r in records if not r["stowed"]]
    gates = [r["lifts"][0]["gate"] for r in records if r["lifts"] and "gate" in r["lifts"][0]]
    stowed = sum(r["stowed"] for r in records)
    return {
        "trials": n,
        "lifted": sum(r["lifted"] for r in records),
        "stowed": stowed,
        "stow_rate": round(stowed / n, 4) if n else None,
        "settle_out": [f"{r['part']}#{r['trial']}" for r in records if r["settle_out"]],
        "errors": sum("error" in r for r in records),
        "end_phase": dict(Counter(r["end_phase"] for r in fails)),
        "drop_phase": dict(Counter(r["drop_phase"] for r in fails if r["drop_phase"])),
        "part_end": dict(Counter(r["part_end"] for r in fails)),
        "lift_gap_m": _median([g["gap_m"] for g in gates]),
        "lift_force_lo_n": _median([min(g["force_n"]) for g in gates]),
        "lift_force_hi_n": _median([max(g["force_n"]) for g in gates]),
        "steps": _median([r["steps"] for r in records]),
        "wall_s": _median([r["wall_s"] for r in records]),
    }


def _git_sha(path: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def run(args) -> dict:
    switches = scene_switches(args)
    jobs = [
        (n, t, s, switches, args.seed_base)
        for n in args.names
        for t, s in enumerate(trial_seeds(n, args.trials, args.seed_base))
    ]
    if args.procs == 1:
        records = [_task(job) for job in jobs]
    else:
        with multiprocessing.get_context("spawn").Pool(args.procs) as pool:
            records = pool.map(_task, jobs, chunksize=1)
    rows = {n: summarize([r for r in records if r["part"] == n]) for n in args.names}
    fams = {
        f: summarize([r for r in records if family(r["part"]) == f])
        for f in dict.fromkeys(family(n) for n in args.names)
    }
    return {
        "source_sha": _git_sha(Path(maker_arm_lab.__file__).parent),
        "script_sha": _git_sha(Path(__file__).parent),
        "scene": {
            "desk_boxes": DESK_BOXES,
            "tray": True,
            "tray_lean_deg": TRAY_LEAN_DEG,
            **switches,
        },
        "trials_per_part": args.trials,
        "seed_base": args.seed_base,
        "total": summarize(records),
        "families": fams,
        "parts": rows,
        "records": records,
    }


def _num(v, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def table_text(result: dict) -> str:
    """The per-part rows, then the pooled family rows in brackets. gap_mm and F_lo/F_hi (the
    weaker and the stronger jaw, N) are medians at the first LIFT's gate."""
    lines = [
        (
            f"{'row':<14} {'n':>3} {'lift':>4} {'stow':>4} {'rate':>6} {'gap_mm':>6} "
            f"{'F_lo':>5} {'F_hi':>5} {'steps':>6} {'wall_s':>6}  failures"
        )
    ]
    rows = [*result["parts"].items(), *((f"[{k}]", v) for k, v in result["families"].items())]
    for name, r in rows:
        gap = None if r["lift_gap_m"] is None else 1000 * r["lift_gap_m"]
        lines.append(
            f"{name:<14} {r['trials']:>3} {r['lifted']:>4} {r['stowed']:>4} "
            f"{r['stow_rate']:>6.3f} {_num(gap, '.1f'):>6} {_num(r['lift_force_lo_n'], '.2f'):>5} "
            f"{_num(r['lift_force_hi_n'], '.2f'):>5} {_num(r['steps'], '.0f'):>6} "
            f"{_num(r['wall_s'], '.1f'):>6}  end {r['end_phase']} drop {r['drop_phase']} "
            f"at {r['part_end']} settle_out {r['settle_out']}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2))
        print(table_text(result), flush=True)
        keep = ("trials", "lifted", "stowed", "stow_rate", "errors")
        summary = {
            "out": args.out,
            "source_sha": result["source_sha"],
            "total": {k: result["total"][k] for k in keep},
            "families": {f: {k: r[k] for k in keep} for f, r in result["families"].items()},
            "parts": {
                n: [r["trials"], r["lifted"], r["stowed"]] for n, r in result["parts"].items()
            },
        }
        ok = result["total"]["errors"] == 0
        print(("TABLE_OK " if ok else "TABLE_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("TABLE_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())
