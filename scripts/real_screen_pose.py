"""The iPad's screen pose in the base frame, from the calibrated overhead camera (MS5 sensing
plan task 3).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_screen_pose.py \\
        --rig rig_cal.json --frames 10 \\
        --out screen.json

Run it with ROS 2 Jazzy sourced, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` and the
``ROS_DOMAIN_ID`` of the Conduit setup; keep the ROS Python path in ``PYTHONPATH``.

With ``board.png`` on the iPad, the script keeps ``--frames`` JPEGs, solves the screen pose in
each (``real.tags.screen_in_camera``) and composes it with the rig's overhead pose. ``--out``
gets the mean pose (``pos``, ``quat`` w x y z) and ``spread_mm`` (the RMS distance of the
frames' positions from the mean). Prints exactly one ``SCREEN_OK {spread_mm}`` or
``SCREEN_FAIL {reason}`` line and leaves through ``os._exit``.
"""

import argparse
import json
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
    return p


def _run(args) -> tuple[str, int]:
    from maker_arm_lab.real import ros
    from maker_arm_lab.real.camera_model import Pose
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
    if not out.parent.is_dir():
        return "SCREEN_FAIL " + json.dumps({"reason": f"{out.parent} is not a folder"}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
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
    out.write_text(
        json.dumps({"pos": mean.pos.tolist(), "quat": mean.quat(), "spread_mm": spread}, indent=2)
    )
    return "SCREEN_OK " + json.dumps({"spread_mm": spread}), 0


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
