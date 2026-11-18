"""Log one AprilTag's pose from the overhead camera, a line per frame (MS5 sensing plan task 3).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_tag_log.py \\
        --rig rig_cal.json --tag_id 6 \\
        --tag_size 0.03 --seconds 20 --out tag.jsonl

Run it with ROS 2 Jazzy sourced, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` and the
``ROS_DOMAIN_ID`` of the Conduit setup; keep the ROS Python path in ``PYTHONPATH``.

The tag is a 36h11 tag of side ``--tag_size`` m (the black border's outer edge). Each line of
``--out`` holds ``stamp_ns``, ``arrival_mono_ns``, ``rms_px``, ``cam_pos`` (the tag centre in
the camera frame) and ``pos`` and ``quat`` (w x y z), the tag in the base frame through the
rig's overhead pose; a frame without the tag has null for them. The tag frame is x right,
y up, z toward the viewer, as ``real.tags`` draws it. Prints exactly one ``TAGLOG_OK {frames,
seen_share, rate_hz, pos_std_mm}`` or ``TAGLOG_FAIL {reason}`` line and leaves through
``os._exit``. ``pos_std_mm`` is the RMS spread of ``pos`` over the frames that saw the tag.
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
        print("TAGLOG_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True)
    p.add_argument("--tag_id", type=int, required=True)
    p.add_argument("--tag_size", type=float, required=True, help="tag side, m")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--out", required=True)
    return p


def _run(args) -> tuple[str, int]:
    from maker_arm_lab.real import ros
    from maker_arm_lab.real.camera_model import Pose
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.tags import FrameTap, detect_in_jpeg, solve_pose

    if args.seconds <= 0 or args.tag_size <= 0:
        detail = "--seconds and --tag_size must be positive"
        return "TAGLOG_FAIL " + json.dumps({"reason": "usage", "detail": detail}), 2
    out = Path(args.out).expanduser()
    if not out.parent.is_dir():
        return "TAGLOG_FAIL " + json.dumps({"reason": f"{out.parent} is not a folder"}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    cam_pose = Pose.from_pos_quat(rig.overhead_pos, rig.overhead_quat)
    half = args.tag_size / 2
    tag_xyz = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    tap = FrameTap()
    sub = ros.ConduitSubscriber(on_jpeg=tap.on_jpeg, on_info=tap.on_info).start()
    try:
        tap.start()
        time.sleep(args.seconds)
        frames = tap.stop()
    finally:
        sub.close()
    if not frames or tap.camera is None:
        reason = f"{len(frames)} frames in {args.seconds:g} s, camera info {tap.camera}"
        return "TAGLOG_FAIL " + json.dumps({"reason": reason, "error": sub.error}), 1
    rows, seen = [], []
    for f in frames:
        row = {"stamp_ns": f.at.stamp_ns, "arrival_mono_ns": f.at.arrival_mono_ns, "rms_px": None,
               "cam_pos": None, "pos": None, "quat": None}  # fmt: skip
        corners = detect_in_jpeg(tap.camera, f.data).get(args.tag_id)
        if corners is not None:
            tag_in_cam, rms = solve_pose(tap.camera, corners, tag_xyz)  # camera in the tag frame
            r_ct = tag_in_cam.rot.T  # camera from tag
            t_ct = -r_ct @ tag_in_cam.pos
            world = Pose(cam_pose.rot @ r_ct, cam_pose.rot @ t_ct + cam_pose.pos)
            row.update(rms_px=rms, cam_pos=t_ct.tolist(), pos=world.pos.tolist(),
                       quat=world.quat())  # fmt: skip
            seen.append(world.pos)
        rows.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    span = (frames[-1].at.arrival_mono_ns - frames[0].at.arrival_mono_ns) / 1e9
    std = float(np.sqrt(np.var(seen, axis=0).sum()) * 1000.0) if len(seen) > 1 else None
    result = {
        "frames": len(rows),
        "seen_share": len(seen) / len(rows),
        "rate_hz": (len(rows) - 1) / span if span > 0 else None,
        "pos_std_mm": std,
    }
    return "TAGLOG_OK " + json.dumps(result), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "TAGLOG_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "TAGLOG_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
