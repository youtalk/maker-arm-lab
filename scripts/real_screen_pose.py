"""The iPad's screen pose in the base frame, from the calibrated overhead camera (MS5 sensing
plan task 3).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_screen_pose.py \\
        --rig rig_cal.json --frames 10 \\
        --out screen.json

Run it with ROS 2 Jazzy sourced, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` and the
``ROS_DOMAIN_ID`` of the Conduit setup; keep the ROS Python path in ``PYTHONPATH``.

With ``board.png`` on the iPad, the script keeps ``--frames`` JPEGs, solves the screen pose in
each (``real.tags.screen_in_camera``) and composes it with the rig's overhead pose. The rig
must come from ``real_calibrate_overhead.py`` (it sets ``overhead_size``) and the stream must
have that size: a turned phone fails. ``--out`` (a new file) gets the mean pose (``pos``,
``quat`` w x y z) and the token's numbers: ``spread_mm`` (the RMS distance of the frames'
positions from the mean), ``offset_mm`` and ``yaw_deg`` (the pose against
``nominal_screen_pose``) and, with ``--compare PREV.json``, ``moved_mm`` and ``moved_deg``
(against an earlier screen pose; task 9 voids the reads after a move over 0.5 mm). Prints
exactly one ``SCREEN_OK {...}`` or ``SCREEN_FAIL {reason}`` line and leaves through
``os._exit``.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("SCREEN_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--timeout_s", type=float, default=30.0)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="an earlier screen pose JSON to measure the move against")
    return p


def _run(args) -> tuple[str, int]:
    from maker_arm_lab.real import ros
    from maker_arm_lab.real.camera_model import Pose
    from maker_arm_lab.real.layouts import nominal_screen_pose
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.tags import (
        IPAD_BOARD,
        FrameTap,
        detect_in_jpeg,
        mean_pose,
        screen_in_camera,
    )

    if args.frames < 1:
        return "SCREEN_FAIL " + json.dumps({"reason": "usage", "detail": "--frames >= 1"}), 2
    out = Path(args.out).expanduser()
    if out.exists() or not out.parent.is_dir():
        reason = f"{out} exists" if out.exists() else f"{out.parent} is not a folder"
        return "SCREEN_FAIL " + json.dumps({"reason": reason}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    if rig.overhead_size is None:
        reason = "the rig has no overhead_size: calibrate first (real_calibrate_overhead.py)"
        return "SCREEN_FAIL " + json.dumps({"reason": reason}), 1
    prev = None
    if args.compare:
        d = json.loads(Path(args.compare).expanduser().read_text())
        prev = Pose.from_pos_quat(d["pos"], d["quat"])
    cam_pose = Pose.from_pos_quat(rig.overhead_pos, rig.overhead_quat)
    tap = FrameTap()
    sub = ros.ConduitSubscriber(on_jpeg=tap.on_jpeg, on_info=tap.on_info).start()
    try:
        tap.start()
        end = time.monotonic() + args.timeout_s
        while len(tap.frames) < args.frames and time.monotonic() < end:
            time.sleep(0.02)
        frames = tap.stop()[: args.frames]
    finally:
        sub.close()
    if len(frames) < args.frames or tap.camera is None:
        reason = f"{len(frames)} of {args.frames} frames, camera info {tap.camera}"
        return "SCREEN_FAIL " + json.dumps({"reason": reason, "error": sub.error}), 1
    size = (tap.camera.width, tap.camera.height)
    if size != tuple(rig.overhead_size):
        reason = (
            f"the stream is {size[0]}x{size[1]}, the rig was calibrated at "
            f"{rig.overhead_size[0]}x{rig.overhead_size[1]}: did the phone turn?"
        )
        return "SCREEN_FAIL " + json.dumps({"reason": reason}), 1
    poses = []
    for f in frames:
        corners = detect_in_jpeg(tap.camera, f.data)
        if sum(i in IPAD_BOARD.tags for i in corners) < 4:
            continue
        p, _ = screen_in_camera(tap.camera, corners, IPAD_BOARD)
        poses.append(Pose(cam_pose.rot @ p.rot, cam_pose.rot @ p.pos + cam_pose.pos))
    if len(poses) * 2 <= len(frames):
        reason = f"the board showed 4 tags in {len(poses)} of {len(frames)} frames"
        return "SCREEN_FAIL " + json.dumps({"reason": reason}), 1
    mean = mean_pose(poses)
    spread = float(np.sqrt(np.mean([np.sum((p.pos - mean.pos) ** 2) for p in poses])) * 1000.0)
    nominal = nominal_screen_pose(rig)
    rel = mean.rot @ nominal.rot.T
    result = {
        "spread_mm": spread,
        "offset_mm": float(np.linalg.norm(mean.pos - nominal.pos) * 1000.0),
        "yaw_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
    }
    if prev is not None:
        cos = (np.trace(mean.rot @ prev.rot.T) - 1.0) / 2.0
        result["moved_mm"] = float(np.linalg.norm(mean.pos - prev.pos) * 1000.0)
        result["moved_deg"] = math.degrees(math.acos(min(1.0, max(-1.0, cos))))
    with open(out, "x") as f:
        f.write(json.dumps({"pos": mean.pos.tolist(), "quat": mean.quat()} | result, indent=2))
    return "SCREEN_OK " + json.dumps(result), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "SCREEN_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "SCREEN_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
