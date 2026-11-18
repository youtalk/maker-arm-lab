"""Calibrate the overhead camera on the iPad tag board (MS5 sensing plan task 3).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_calibrate_overhead.py \\
        --rig rig.json --placements 3 \\
        --frames 10 --rig_out rig_cal.json --report cal_report.json

Run it with ROS 2 Jazzy sourced, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` and the
``ROS_DOMAIN_ID`` of the Conduit setup; keep the ROS Python path in ``PYTHONPATH``.

The iPad shows ``board.png`` (``scripts/real_layouts.py`` writes it) one to one. Placement 0 is
the iPad on its taped reference spot, at ``real.layouts.nominal_screen_pose(rig)``; the others
lie flat anywhere in the camera's view. Before each placement the script waits for Enter, keeps
``--frames`` JPEGs from the ``/conduit`` wide stream, and detects the tags (a tag counts when it
shows in more than half of the frames; its corners are the median). The camera K is the first
``camera_info``'s, read with ``Pinhole.from_ros_k``, and the image is used as published.
``real.tags.calibrate`` gives the pose. ``--rig_out`` is the input rig with ``overhead_pos``,
``overhead_quat`` and ``overhead_size`` replaced and ``overhead_k`` set to null, so the warp keeps
following ``camera_info``. The input rig is never written. Prints exactly one
``CALIB_OK {plane_rms_mm, check_rms_px, check_rms_mm, check_rms_view_px, coverage}`` or
``CALIB_FAIL {reason}`` line and leaves through ``os._exit``. ``check_rms_view_px`` is the check
error in pixels of the 224 target view (``sim_overhead(rig.overhead_back_m)``), the equivalence
check of MS5 revision section 7.3.
"""

import argparse
import dataclasses
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
        print("CALIB_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True)
    p.add_argument("--placements", type=int, default=3, help="iPad placements (>= 3)")
    p.add_argument("--frames", type=int, default=10, help="frames kept per placement")
    p.add_argument("--timeout_s", type=float, default=30.0, help="per placement")
    p.add_argument("--rig_out", required=True, help="the rig JSON to create")
    p.add_argument("--report", required=True, help="the report JSON to create")
    return p


def placement_corners(frames, camera, board, min_tags):
    """{id: median corners} of the board's tags that show in more than half of ``frames``;
    a tag of another board is not counted."""
    from maker_arm_lab.real.tags import detect_in_jpeg

    seen: dict[int, list] = {}
    for f in frames:
        for i, c in detect_in_jpeg(camera, f.data).items():
            if i in board.tags:
                seen.setdefault(i, []).append(c)
    corners = {i: np.median(cs, axis=0) for i, cs in seen.items() if len(cs) * 2 > len(frames)}
    if len(corners) < min_tags:
        raise ValueError(f"{len(corners)} tags in the frames of a placement, need {min_tags}")
    return corners


def _run(args) -> tuple[str, int]:
    from maker_arm_lab.real import ros
    from maker_arm_lab.real.camera_model import overhead_warp, project, sim_overhead
    from maker_arm_lab.real.layouts import nominal_screen_pose
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.tags import IPAD_BOARD, MIN_TAGS, FrameTap, calibrate

    if args.placements < 3 or args.frames < 1:
        detail = "--placements >= 3 and --frames >= 1"
        return "CALIB_FAIL " + json.dumps({"reason": "usage", "detail": detail}), 2
    out_path, report_path = Path(args.rig_out).expanduser(), Path(args.report).expanduser()
    for p in (out_path, report_path):
        if p.exists() or not p.parent.is_dir():
            detail = f"{p} exists" if p.exists() else f"{p.parent} is not a folder"
            return "CALIB_FAIL " + json.dumps({"reason": "output", "detail": detail}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    reference = nominal_screen_pose(rig)
    tap = FrameTap()
    sub = ros.ConduitSubscriber(on_jpeg=tap.on_jpeg, on_info=tap.on_info).start()
    placements = []
    try:
        for n in range(args.placements):
            where = "on its taped reference spot" if n == 0 else "flat anywhere in the view"
            print(
                f"placement {n + 1}/{args.placements}: iPad {where}; press Enter", file=sys.stderr
            )
            input()
            tap.start()
            end = time.monotonic() + args.timeout_s
            while len(tap.frames) < args.frames and time.monotonic() < end:
                time.sleep(0.02)
            frames = tap.stop()[: args.frames]
            if len(frames) < args.frames or tap.camera is None:
                reason = f"{len(frames)} of {args.frames} frames, camera info {tap.camera}"
                return "CALIB_FAIL " + json.dumps({"reason": reason, "error": sub.error}), 1
            placements.append(placement_corners(frames, tap.camera, IPAD_BOARD, MIN_TAGS))
    finally:
        sub.close()
    camera = tap.camera
    res = calibrate(camera, placements, IPAD_BOARD, reference)
    # The check rows in the target view: the true and the measured corners, through the
    # simulator's overhead camera.
    view_cam, view_pose = sim_overhead(rig.overhead_back_m)
    d = [
        np.subtract(project(view_cam, view_pose, t), project(view_cam, view_pose, m))
        for t, m in res.report["check_pairs"]
    ]
    view_rms = float(np.sqrt((np.array(d) ** 2).sum(axis=1).mean()))
    pos, quat = tuple(float(v) for v in res.pose.pos), res.pose.quat()
    new = dataclasses.replace(
        rig,
        overhead_pos=pos,
        overhead_quat=quat,
        overhead_k=None,
        overhead_size=(camera.width, camera.height),
    )
    coverage = overhead_warp(camera, res.pose, rig.desk_z, rig.overhead_back_m).coverage
    result = {
        "plane_rms_mm": res.plane_rms_mm,
        "check_rms_px": res.check_rms_px,
        "check_rms_mm": res.check_rms_mm,
        "check_rms_view_px": view_rms,
        "coverage": coverage,
    }
    report_path.write_text(json.dumps(result | res.report | {"k": camera.ros_k()}, indent=2))
    with open(out_path, "x") as f:
        f.write(new.to_json())
    return "CALIB_OK " + json.dumps(result), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "CALIB_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "CALIB_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
