"""Replay one heap episode with the teacher's decisions logged and a third-person video.

The phase A recorder kept only pick-cycle frames, so the drop, the settle, every push and the
idle tail after the give-up were never seen. This script builds one heap env exactly as
``scripts/mj_generate_batch.py`` does, drives it with ``OracleRig`` and a logging subclass of
``HeapOracle``, and writes:

* ``<out_dir>/events.jsonl``: one JSON object per line. ``scan`` on every selection, ``phase``
  on every phase report, ``exit`` / ``enter`` when a part crosses the reach band, ``end``
  once.
* ``<out_dir>/inspect.mp4``: a free camera over the heap, one frame per ``--video_stride``
  control steps, with the teacher's state stamped on it.

    MUJOCO_GL=egl uv run python scripts/mj_inspect_heap.py --out_dir /tmp/i0 --heap_k 12 \\
        --parts_split heldout --run_seed 20261000 --layout_seed 20261010

Prints exactly one ``INSPECT_OK {json}`` or ``INSPECT_FAIL {json}`` line and exits through
``os._exit``. Nothing under ``core/`` changes for the inspection (design D4).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback

# --viewer needs an on-screen GL context, and the video renderer shares the backend, so
# that run moves to glfw. Read off argv: the backend is fixed at the first mujoco import.
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")

import mujoco
import numpy as np
import torch

from maker_arm_lab.core import grasp, layout, oracle
from maker_arm_lab.core.batch_loop import (
    add_batch_arguments,
    build_run,
    heap_episode_length_s,
    heap_parts,
    make_induction,
)
from maker_arm_lab.core.checker import pinch_min_force, pinch_min_stall
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

VIDEO_SIZE = (640, 480)
CAMERA_DISTANCE = 0.55
CAMERA_ELEVATION_DEG = -55.0
CAMERA_AZIMUTH_DEG = 150.0
VIDEO_FPS = 30


AT_REST_STRIDE = 30
"""Control steps between reads of what the bin holds at rest, for the end event."""
EXIT_HYSTERESIS = 0.005
"""A part is logged out of reach once its center is this far (m) past the reach band's edge,
and back in once it is this far inside. A part resting at the edge jitters across it by a
millimeter for thousands of steps; without the band every jitter is an exit."""


class InspectHeapOracle(HeapOracle):
    """``HeapOracle`` that writes its decisions to ``self.events``; behavior unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[dict] = []
        self.step = 0
        self.lift_detail: dict = {}
        """The pinch factors behind the next LIFT report, filled by the rig wrapper in
        ``run()``. Telemetry only: the gate itself is untouched."""

    def _event(self, kind: str, phase: str | None = None, **fields) -> dict:
        event = {"step": self.step, "phase": phase or self.phase.name, "kind": kind, **fields}
        self.events.append(event)
        return event

    def _next_brick(self, scene: oracle.SceneSnapshot) -> int | None:
        done = self._stowed(scene) | self._sorted
        alive = torch.tensor([b not in done for b in range(self.num_bricks)])
        by_part: dict[int, list[grasp.Grasp]] = {}
        for g in self._ranked(scene, alive):
            by_part.setdefault(g.part, []).append(g)
        mask = self._candidates(scene)
        parts = []
        for b in torch.nonzero(alive).flatten().tolist():
            x, y, z = scene.brick_pos[b].tolist()
            grasps = by_part.get(b, [])
            parts.append(
                {
                    "id": b,
                    "family": self.parts[b].family,
                    "name": self.parts[b].name,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                    "reach": layout.in_reach(x, y, self.bl),
                    "cell_open": self.failures.get(self.cell_of((x, y)), 0) < 3,
                    "candidate": bool(mask[b]),
                    "axes": len(grasps),
                    "best_clearance": max((g.clearance for g in grasps), default=None),
                    "min_swept": min((getattr(g, "swept", 0) for g in grasps), default=None),
                }
            )
        pushes_before = self.pushes_in_a_row
        chosen = super()._next_brick(scene)
        g = self.grasp if chosen is not None else None
        self._event(
            "scan",
            candidates=int(mask.sum()),
            parts=parts,
            chosen=None
            if g is None
            else {
                "part": g.part,
                "family": self.parts[g.part].family,
                "heading": round(g.heading, 3),
                "width": round(g.width, 4),
                "half_height": round(g.half_height, 4),
                "clearance": None if math.isinf(g.clearance) else round(g.clearance, 4),
                "swept": getattr(g, "swept", None),
                "height": round(g.height, 4),
            },
            push=None
            if self.push is None or chosen is not None
            else {"start": self.push.start, "end": self.push.end},
            pushes_in_a_row=pushes_before,
            cycles=self.cycles,
            budget_left=self.cycle_budget - self.cycles,
            out_of_reach=self.out_of_reach,
        )
        return chosen

    def report(self, result: oracle.PhaseResult, scene: oracle.SceneSnapshot) -> None:
        target = self.target_brick
        phase = self.phase.name
        pose = None if target is None else [round(v, 4) for v in scene.brick_pos[target].tolist()]
        super().report(result, scene)
        self._event(
            "phase",
            phase=phase,
            outcome=result.outcome,
            pinch_ok=result.pinch_ok,
            lifted=result.lifted,
            target=target,
            family=None if target is None else self.parts[target].family,
            retries_used=self.retries_used,
            next_phase=self.phase.name,
            target_pose=pose,
            **self.lift_detail,
        )
        self.lift_detail = {}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_batch_arguments(parser)
    parser.add_argument("--video_stride", type=int, default=8, help="control steps per frame")
    parser.add_argument(
        "--video_tail_s", type=float, default=5.0, help="seconds to keep stepping after DONE"
    )
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the MuJoCo GUI on the episode (needs a display; closing it ends the run)",
    )
    args = parser.parse_args(argv)
    if args.heap_k <= 0:
        parser.error("--heap_k must be positive: this script inspects heap episodes only")
    if args.num_envs != 1:
        parser.error("--num_envs must be 1: one episode, one log, one video")
    return args


class Video:
    def __init__(self, env: MjLegoSortEnv, path: str):
        from PIL import Image, ImageDraw

        self._image, self._draw = Image, ImageDraw
        self.renderer = mujoco.Renderer(env.models[0], VIDEO_SIZE[1], VIDEO_SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (*layout.HEAP_CENTER, 0.0)
        self.camera.distance = CAMERA_DISTANCE
        self.camera.elevation = CAMERA_ELEVATION_DEG
        self.camera.azimuth = CAMERA_AZIMUTH_DEG
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
                "-r",
                str(VIDEO_FPS),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                path,
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def frame(self, data: mujoco.MjData, lines: list[str]) -> None:
        self.renderer.update_scene(data, self.camera)
        image = self._image.fromarray(self.renderer.render())
        draw = self._draw.Draw(image)
        draw.rectangle((0, 0, 260, 14 * len(lines) + 8), fill=(0, 0, 0))
        for k, line in enumerate(lines):
            draw.text((4, 4 + 14 * k), line, fill=(255, 255, 255))
        self.proc.stdin.write(np.asarray(image).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        self.renderer.close()


def _install_lift_wrapper(rig) -> None:
    """Record what the pinch gate saw on every LIFT, then let the untouched gate decide."""
    inner_lift = rig._lift_result

    def lift_result(i: int, outcome: str):
        """Record what the pinch gate saw, then let the untouched gate decide."""
        orc_i = rig.oracles[i]
        width = orc_i.grasp_width
        min_size = getattr(orc_i, "grasp_min_size", min(layout.BRICK_SIZE))
        floor_stall = pinch_min_stall(width, min_size)
        floor_force = pinch_min_force(width, min_size)
        ev = rig.pinch_evidence(i, orc_i.target_brick, floor_stall, floor_force)
        end = rig._brick_pos(i, orc_i.target_brick)
        orc_i.lift_detail = {
            "stall": round(ev["stall"], 5),
            "left": round(ev["left"], 3),
            "right": round(ev["right"], 3),
            "asymmetry": round(ev["asymmetry"], 3),
            "min_stall": round(floor_stall, 5),
            "min_force": round(floor_force, 3),
            "rise": round((end[2] - rig.brick_start[i][2]).item(), 4),
            "xy_error": round((end[:2] - rig.pick_point(i)[:2]).norm().item(), 4),
            "grasp_width": round(width, 4),
        }
        return inner_lift(i, outcome)

    rig._lift_result = lift_result


def run(args: argparse.Namespace) -> dict:
    os.makedirs(args.out_dir, exist_ok=True)
    draw, bl, noise = build_run(args)
    parts = heap_parts(args)
    episode_length_s = args.episode_length_s
    if episode_length_s is None:
        episode_length_s = heap_episode_length_s(len(parts), args.cycle_budget)
    env = MjLegoSortEnv(
        1,
        draw,
        num_bricks=len(parts),
        parts=parts,
        cameras=False,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        episode_length_s=episode_length_s,
        episode_draws=args.episode_draws,
        frame_jitter=args.frame_jitter,
        gripper_mode=args.gripper,
        desk_boxes=args.desk_boxes,
    )
    env.reset(seed=args.run_seed)
    induction = make_induction(args)
    rig_rng = random.Random(args.run_seed)
    holder: list[InspectHeapOracle] = []

    def factory(i, layout_i, episode_draw):
        orc = InspectHeapOracle(
            layout_i,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            induction=induction,
            rng=rig_rng,
            cycle_budget=args.cycle_budget,
            boxes=env.desk_box_poses[i],
        )
        holder.append(orc)
        return orc

    rig = OracleRig(
        env,
        bl,
        noise,
        rig_rng,
        induction=induction,
        transport_speed=args.transport_speed,
        oracle_factory=factory,
    )
    orc = holder[-1]
    _install_lift_wrapper(rig)
    video = None if args.no_video else Video(env, os.path.join(args.out_dir, "inspect.mp4"))
    gui = None
    if args.viewer:
        import mujoco.viewer

        gui = mujoco.viewer.launch_passive(env.models[0], env.datas[0])
    tail_steps = round(args.video_tail_s * 30)

    def poses() -> list[list[float]]:
        return [[round(v, 4) for v in rig._brick_pos(0, b).tolist()] for b in range(len(parts))]

    prev = poses()
    crossings: list[dict] = []
    outside = [not layout.in_reach(x, y, bl, -EXIT_HYSTERESIS) for x, y, _ in prev]
    for b, (x, y, _) in enumerate(prev):
        if outside[b]:
            crossings.append(
                {
                    "step": 0,
                    "phase": "SETTLE",
                    "kind": "exit",
                    "part": b,
                    "family": parts[b].family,
                    "from": None,
                    "to": prev[b],
                    "target": None,
                    "in_push": False,
                }
            )
    step = 0
    done_step = None
    reason = "timeout"
    at_rest = 0
    while True:
        orc.step = step
        if step % AT_REST_STRIDE == 0:
            # Read before the step: the step that ends the episode also resets the env, and a
            # snapshot taken after it sees the next heap.
            at_rest = len(orc._stowed(rig.snapshot(0)))
        out = rig.step()
        step += 1
        now = poses()
        finished = bool(out["dones"][0]) or bool(out["abandoned"][0])
        for b in range(len(parts) if not finished else 0):
            x, y = now[b][:2]
            left = not outside[b] and not layout.in_reach(x, y, bl, -EXIT_HYSTERESIS)
            back = outside[b] and layout.in_reach(x, y, bl, EXIT_HYSTERESIS)
            if left or back:
                outside[b] = left
                crossings.append(
                    {
                        "step": step,
                        "phase": orc.phase.name,
                        "kind": "exit" if left else "enter",
                        "part": b,
                        "family": parts[b].family,
                        "from": prev[b],
                        "to": now[b],
                        "target": orc.target_brick,
                        "in_push": orc.phase in oracle.PUSH_PHASES,
                    }
                )
        prev = now
        if gui is not None:
            if not gui.is_running():
                reason = "closed"
                break
            gui.sync()
        if video is not None and step % args.video_stride == 0:
            g = orc.grasp
            target_name = "" if orc.target_brick is None else parts[orc.target_brick].name
            video.frame(
                env.datas[0],
                [
                    f"step {step}  {orc.phase.name}",
                    f"target {orc.target_brick} {target_name}",
                    f"clearance {'' if g is None or math.isinf(g.clearance) else f'{g.clearance * 1000:.1f} mm'}",
                    f"pushes_in_a_row {orc.pushes_in_a_row}  cycles {orc.cycles}",
                    f"stowed {len(orc._sorted)}  out of reach {orc.out_of_reach}  retries {orc.retries_used}",
                ],
            )
        if bool(out["abandoned"][0]):
            reason = "abandon"
            break
        if bool(out["dones"][0]):
            reason = "done" if orc.phase == oracle.Phase.DONE else "timeout"
            break
        if orc.phase == oracle.Phase.DONE:
            if done_step is None:
                done_step = step
            elif step - done_step >= tail_steps:
                reason = "budget" if orc.cycles >= orc.cycle_budget else "done"
                break
    if video is not None:
        video.close()
    if gui is not None:
        gui.close()
    stats = orc.extra_stats()
    end = {
        "step": step,
        "phase": orc.phase.name,
        "kind": "end",
        "reason": reason,
        "done_step": done_step,
        "idle_steps": 0 if done_step is None else int(env.max_episode_length) - done_step,
        "stowed": len(orc._sorted),
        # What the bin held at rest within AT_REST_STRIDE steps of the end. `_sorted` counts
        # a part only if RETREAT saw it at rest; a part that settled later is here, not there.
        "stowed_at_rest": max(
            at_rest, len(orc._stowed(rig.snapshot(0))) if reason != "timeout" else 0
        ),
        "parts": len(parts),
        "extra_stats": stats,
    }
    events = sorted(orc.events + crossings, key=lambda e: e["step"]) + [end]
    log = os.path.join(args.out_dir, "events.jsonl")
    with open(log, "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    env.close()
    return {
        "events": log,
        "video": None if video is None else os.path.join(args.out_dir, "inspect.mp4"),
        "frames": 0 if video is None else video.frames,
        **{k: v for k, v in end.items() if k not in ("kind", "phase")},
    }


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        summary = run(args)
    except SystemExit as e:  # argparse
        print("INSPECT_FAIL " + json.dumps({"error": f"argparse exit {e.code}"}), flush=True)
        return 2
    except Exception as e:  # noqa: BLE001 -- any failure must become INSPECT_FAIL.
        traceback.print_exc()
        print("INSPECT_FAIL " + json.dumps({"error": repr(e)}), flush=True)
        return 1
    print("INSPECT_OK " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
