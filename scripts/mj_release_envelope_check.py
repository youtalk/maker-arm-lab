"""Desk-box release envelope: the MuJoCo twin of ``scripts/release_envelope_check.py`` (MS7a).

MS7a design, desk boxes: this check certifies the side band that the box draw samples from
(``core.desk_boxes.band_x`` and ``DESK_BOX_BAND_Y``). One lone part lies at a fixed tray
point, the tray centre, so the pick leg is the same for every box and only the release leg
varies; the part's yaw steps through ``--part_yaws`` over the trials of a point, because the
grasp heading the carry holds decides whether EXTEND plans at all. One desk box stands at a band point: the band's four corners and its centre, on both
sides, at both yaw extremes. The heap teacher (``HeapOracle`` with desk boxes) then drives the
whole cycle for real, through the rig and the physics, from SETTLE to the end of RETREAT.

A trial passes on two reads, both off the driven arm and the part, never off a solver:

* EXTEND converged: the pick point ended inside EXTEND's own 8 mm stop of the release pose.
* The part landed: at rest inside its box's stow region (``core.stowed``'s grader) after
  RETREAT, with up to ``SETTLE_STEPS`` of hold for a part still moving.

It also reports what the band's owner needs to shrink it: the landing offset in the box frame,
the steps EXTEND and RETREAT took, RETREAT's residual, the tightest joint margin at the release
and the control steps on which an arm geom touched a box.

    MUJOCO_GL=egl PYTHONPATH=$PWD/source .venv/bin/python scripts/mj_release_envelope_check.py \\
        --trials 2 --out /tmp/envelope.json

Prints exactly one ``ENVELOPE_OK {json}`` or ``ENVELOPE_FAIL {json}`` line on every exit path,
argparse errors included, and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import torch

from maker_arm_lab.core import desk_boxes, layout, oracle
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import HEAP_PARK_X, MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
    GRIPPER_OPEN_ACTION,
    OracleRig,
)

PARK_BOX_X = -0.6
"""Where the boxes not under test wait: behind the arm and off the desk."""
EXTEND_STOP_M = 0.008
"""EXTEND's own stop tolerance (``EnvOracle.extend_goal``), read against the driven arm."""
SETTLE_STEPS = 60
"""Hold steps after RETREAT for a part still moving in the box."""
MAX_STEPS = 3000
EPISODE_LENGTH_S = 150.0


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("ENVELOPE_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def _pair(text: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in text.split(","))
    return lo, hi


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument(
        "--band_x", type=_pair, default=None, help="default: core.desk_boxes.band_x(--desk_boxes)"
    )
    parser.add_argument("--band_y", type=_pair, default=desk_boxes.DESK_BOX_BAND_Y)
    parser.add_argument("--trials", type=int, default=4, help="driven trials per band point")
    parser.add_argument(
        "--part_yaws",
        type=lambda text: [float(v) for v in text.split(",")],
        default=[0.0, 45.0, 90.0, -45.0],
        help="degrees; trial k of a point picks the part at yaw k modulo their number",
    )
    parser.add_argument("--seed", type=int, default=20267400)
    parser.add_argument("--part", type=str, default="brick_2x4", help="a catalog part name")
    parser.add_argument("--gripper", choices=("binary", "continuous"), default="continuous")
    parser.add_argument(
        "--desk_boxes",
        type=int,
        default=1,
        help="boxes in the scene, which sizes the box (core.desk_boxes.interior); box 0 "
        "stands at the band point and the others wait off the desk",
    )
    parser.add_argument("--out", type=str, default="", help="write the trial records here")
    args = parser.parse_args(argv)
    if args.band_x is None:
        args.band_x = desk_boxes.band_x(args.desk_boxes)
    return args


def band_points(band_x, band_y) -> list[tuple[float, float]]:
    """The band's four corners and its centre, as (x, |y|)."""
    (x0, x1), (y0, y1) = band_x, band_y
    return [(x0, y0), (x0, y1), (x1, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]


def _place_part(env, part, yaw: float) -> None:
    """Part 0 at the tray centre at ``yaw`` (rad), settled; part 1 parked off the desk.

    The parked twin is never stowed, so the env's success never fires and never auto-resets
    the episode under the read (``MjLegoSortEnv.step``).
    """
    m, d, h = env.models[0], env.datas[0], env.handles
    for b, (xyz, a) in enumerate(
        (((*layout.TRAY_CENTER, part.size[2] / 2 + 0.001), yaw), ((HEAP_PARK_X, 0.0, 0.0), 0.0))
    ):
        adr = h.brick_qpos[b]
        d.qpos[adr : adr + 3] = xyz
        d.qpos[adr + 3 : adr + 7] = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
        d.qvel[h.brick_dof[b] : h.brick_dof[b] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=120)
    env._refresh([0], images=False)


def _arm_box_contact(env, box_geoms: set[int]) -> bool:
    m, d = env.models[0], env.datas[0]
    for k in range(d.ncon):
        g1, g2 = int(d.contact[k].geom1), int(d.contact[k].geom2)
        for a, b in ((g1, g2), (g2, g1)):
            if a in box_geoms and m.geom_contype[b] == 1:  # robot geoms are contype 1
                return True
    return False


def _joint_margin(rig) -> float:
    q = rig.robot.data.joint_pos[0, rig.arm_ids]
    return float(torch.minimum(q - rig.limits[:, 0], rig.limits[:, 1] - q).min())


def run_trial(env, rig, part, pose, seed: int, part_yaw: float = 0.0) -> dict:
    env.reset(seed=seed)
    parked = [(PARK_BOX_X, 0.3 * j, 0.0) for j in range(1, env.desk_boxes)]
    env.place_desk_boxes(0, [pose, *parked])
    env.destination_bins[0] = 0  # the lone part goes to box 0 whatever the scene's map
    _place_part(env, part, part_yaw)
    rig.rearm([0])
    m = env.models[0]
    box_geoms = {g for g in range(m.ngeom) if m.body_mocapid[m.geom_bodyid[g]] >= 0}
    record = {"pose": [round(v, 4) for v in pose], "seed": seed, "part_yaw": part_yaw}
    prev, steps_in_phase, touches = rig.goals[0], 0, 0
    for _ in range(MAX_STEPS):
        out = rig.step()
        steps_in_phase += 1
        touches += _arm_box_contact(env, box_geoms)
        goal = rig.goals[0]
        if goal is not prev:
            if prev is not None and prev.target_pos is not None:
                err = float((prev.target_pos - rig.pick_point(0)).norm())
                name = prev.phase.name.lower()
                record[f"{name}_err_mm"] = round(1000.0 * err, 2)
                record[f"{name}_steps"] = steps_in_phase
                if prev.phase == oracle.Phase.EXTEND:
                    record["release_tilt_deg"] = round(math.degrees(rig.tool_tilt(0)), 1)
                    record["joint_margin_rad"] = round(_joint_margin(rig), 3)
            steps_in_phase = 0
            ended_retreat = prev is not None and prev.phase == oracle.Phase.RETREAT
            prev = goal
            if ended_retreat:
                break
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    hold = torch.zeros(1, 7)
    hold[0, :6] = rig.q_cmd[0].float()
    hold[0, 6] = GRIPPER_OPEN_ACTION
    for _ in range(SETTLE_STEPS):
        if bool(per_brick_stowed(env)[0, 0]):
            break
        env.step(hold)
    pos = env.scene["brick_0"].data.root_pos_w[0]
    x, y, yaw = pose
    dx, dy = float(pos[0]) - x, float(pos[1]) - y
    c, s = math.cos(yaw), math.sin(yaw)
    record["landing_mm"] = [
        round(1000.0 * (c * dx + s * dy), 1),
        round(1000.0 * (c * dy - s * dx), 1),
    ]
    record["arm_box_contact_steps"] = touches
    record["abandoned"] = bool(rig.abandoned[0])
    record["extend_ok"] = record.get("extend_err_mm", math.inf) < 1000.0 * EXTEND_STOP_M
    record["landed"] = bool(per_brick_stowed(env)[0, 0])
    record["ok"] = record["extend_ok"] and record["landed"]
    return record


def run(args) -> dict:
    (part,) = _parts.by_name([args.part])
    draw = layout.sample_run_draw(random.Random(args.seed))
    env = MjLegoSortEnv(
        1,
        draw,
        parts=[part, part],
        cameras=False,
        desk_color=False,
        physics_dr=False,
        start_pose_offset_rad=0.0,
        episode_length_s=EPISODE_LENGTH_S,
        gripper_mode=args.gripper,
        desk_boxes=args.desk_boxes,
    )
    env.reset(seed=args.seed)
    rng = random.Random(args.seed)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            boxes=env.desk_box_poses[i],
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    points = []
    trial_seed = args.seed
    try:
        for sign in (1.0, -1.0):
            for x, ay in band_points(args.band_x, args.band_y):
                for yaw_deg in (-45.0, 45.0):
                    pose = (x, sign * ay, math.radians(yaw_deg))
                    trials = []
                    for k in range(args.trials):
                        trial_seed += 1
                        part_yaw = math.radians(args.part_yaws[k % len(args.part_yaws)])
                        trials.append(run_trial(env, rig, part, pose, trial_seed, part_yaw))
                    point = {
                        "x": x,
                        "y": sign * ay,
                        "yaw_deg": yaw_deg,
                        "landed": sum(t["landed"] for t in trials),
                        "extend_ok": sum(t["extend_ok"] for t in trials),
                        "trials": trials,
                    }
                    points.append(point)
                    print(
                        f"POINT x={x:.3f} y={sign * ay:+.3f} yaw={yaw_deg:+.0f}: landed "
                        f"{point['landed']}/{args.trials} extend_ok {point['extend_ok']}/"
                        f"{args.trials} extend_err_mm="
                        f"{[t.get('extend_err_mm') for t in trials]} landing_mm="
                        f"{[t['landing_mm'] for t in trials]} retreat_err_mm="
                        f"{[t.get('retreat_err_mm') for t in trials]} arm_box_contact_steps="
                        f"{[t['arm_box_contact_steps'] for t in trials]}",
                        flush=True,
                    )
    finally:
        env.close()
    n = len(points) * args.trials
    return {
        "band_x": list(args.band_x),
        "band_y": list(args.band_y),
        "part": args.part,
        "part_yaws": args.part_yaws,
        "gripper": args.gripper,
        "desk_boxes": args.desk_boxes,
        "interior": list(desk_boxes.interior(args.desk_boxes)),
        "release_z": desk_boxes.DESK_BOX_RELEASE_Z,
        "trials": n,
        "landed": sum(p["landed"] for p in points),
        "extend_ok": sum(p["extend_ok"] for p in points),
        "points": points,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
        ok = result["landed"] == result["trials"] and result["extend_ok"] == result["trials"]
        summary = {k: v for k, v in result.items() if k != "points"}
        print(("ENVELOPE_OK " if ok else "ENVELOPE_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("ENVELOPE_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())
