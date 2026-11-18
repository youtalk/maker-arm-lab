"""Calibrate the wrist camera's intrinsics from its ROS topic (MS5 sensing plan task 11).

    source ~/opt/env.sh  # ROS and the Thor's venv
    .venv/bin/python scripts/real_wrist_intrinsics.py --rig ~/data/maker-arm-ms5/rig.json \
        --size 640x480 --board 9x6 --square 0.02001 --frames 25 --gap_s 1.0 \
        --rig_out ~/data/maker-arm-ms5/rig_wrist.json

The iPad shows ``real.layouts.checker_png()`` and the operator moves it in front of the
camera at 0.1 to 0.3 m and at several tilts. The script reads ``WRIST_TOPIC`` through
``real.ros.ConduitSubscriber``, keeps a frame where ``cv2.findChessboardCorners`` succeeds, at
most one per ``--gap_s`` seconds of arrival time, and calls ``cv2.calibrateCamera`` after
``--frames`` kept frames. ``--rig_out`` is the input rig with ``wrist_k`` (OpenCV's K, the ROS
convention), ``wrist_size`` and ``wrist_dist``; it must not exist, and the input rig is never
written. The script opens no device. It prints progress lines to stderr and exactly one
``WRIST_K_OK {rms_px, fov_deg: [h, v], coverage, frames}`` (``coverage`` is the share of the
simulated wrist view that the camera covers through the lab's wrist warp) or ``WRIST_K_FAIL {reason}`` line on every exit path and leaves through ``os._exit``.
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

DROP_FAIL = 10
"""Dropped frames, with none accepted, after which the wait ends."""
SILENT_FAIL_S = 10.0
"""Seconds with no frame at all after which the wait ends."""
PROGRESS_S = 5.0


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("WRIST_K_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def _pair(text: str, sep: str) -> tuple[int, int]:
    a, b = text.lower().split(sep)
    return int(a), int(b)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True, help="rig JSON (real.rig.RigConfig)")
    p.add_argument("--size", default="640x480", help="the wrist frame, WIDTHxHEIGHT")
    p.add_argument("--board", default="9x6", help="inner corners, COLSxROWS")
    p.add_argument("--square", type=float, default=0.02001, help="square side, m")
    p.add_argument("--frames", type=int, default=25, help="kept frames (>= 5)")
    p.add_argument("--gap_s", type=float, default=1.0, help="least arrival gap of kept frames")
    p.add_argument("--timeout_s", type=float, default=300.0)
    p.add_argument("--rig_out", required=True, help="the rig JSON to create")
    return p


def _progress(seen, kept, src, frames):
    print(
        f"frames seen {seen}, views kept {kept}/{frames}, dropped {src.dropped}, error {src.error}",
        file=sys.stderr,
        flush=True,
    )


def collect(src, subscriber, cv2, board, frames, gap_s, timeout_s, topic="the wrist topic"):
    """(gray frames with the board's corners, the reason the wait ended early or None)."""
    kept, seen, last_seen, last_kept = [], 0, None, None
    start = last_print = time.monotonic()
    end = start + timeout_s
    while len(kept) < frames and time.monotonic() < end:
        time.sleep(0.02)
        now = time.monotonic()
        if now - last_print >= PROGRESS_S:
            last_print = now
            _progress(seen, len(kept), src, frames)
        latest = src.frames.get()
        if latest is None:
            if src.dropped >= DROP_FAIL:
                return kept, f"{src.dropped} frames dropped and none accepted: {src.error}"
            if src.dropped == 0 and now - start >= SILENT_FAIL_S:
                reason = f"no frame arrived on {topic} in {SILENT_FAIL_S:g} s"
                return kept, "; ".join([reason, *([subscriber.error] if subscriber.error else [])])
            continue
        if latest.arrival_mono_ns == last_seen:
            continue
        last_seen = latest.arrival_mono_ns
        seen += 1
        if last_kept is not None and (last_seen - last_kept) / 1e9 < gap_s:
            continue
        gray = cv2.cvtColor(latest.raw, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )  # fmt: skip
            kept.append(corners)
            last_kept = last_seen
            _progress(seen, len(kept), src, frames)
    return kept, None


def wait_for_views(src, sub, cv2, board, args, topic="the wrist topic"):
    """(kept corner sets, None), or (what was kept, the reason it fell short). Closes ``sub``;
    the reason quotes the source's and the subscriber's ``error``."""
    try:
        kept, early = collect(src, sub, cv2, board, args.frames, args.gap_s, args.timeout_s, topic)
    finally:
        sub.close()
    if len(kept) >= args.frames:
        return kept, None
    reason = early or f"{len(kept)} of {args.frames} frames with the board before the timeout"
    causes = [e for e in (src.error, sub.error) if e and e not in reason]
    return kept, "; ".join([reason, *causes])


def calibrate_views(views, board, square_m, size):
    """(K row-major in the ROS and OpenCV convention, the five distortion coefficients, rms_px)
    from the board's corners in each view, ``board`` inner corners wide by high."""
    import cv2

    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_m
    rms, k, dist, _, _ = cv2.calibrateCamera([obj] * len(views), views, size, None, None)
    return tuple(float(v) for v in k.ravel()), tuple(float(v) for v in dist.ravel()[:5]), rms


def _run(args) -> tuple[str, int]:
    import cv2

    from maker_arm_lab.real import ros
    from maker_arm_lab.real.camera_model import Pinhole, quat_matrix, wrist_warp
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.sources import WristFrames

    w, h = _pair(args.size, "x")
    board = _pair(args.board, "x")
    if args.frames < 5:
        return "WRIST_K_FAIL " + json.dumps({"reason": "usage", "detail": "--frames >= 5"}), 2
    out_path = Path(args.rig_out).expanduser()
    if not out_path.parent.is_dir():
        detail = f"{out_path.parent} is not a folder"
        return "WRIST_K_FAIL " + json.dumps({"reason": "rig_out", "detail": detail}), 1
    if out_path.exists():
        detail = f"{out_path} exists"
        return "WRIST_K_FAIL " + json.dumps({"reason": "rig_out", "detail": detail}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    # The source reads only the size of this placeholder camera.
    src = WristFrames(
        Pinhole.from_ros_k(w, h, (1, 0, w / 2 - 0.5, 0, 1, h / 2 - 0.5, 0, 0, 1)), None
    )
    sub = ros.ConduitSubscriber(on_wrist=src.on_jpeg).start()
    kept, reason = wait_for_views(src, sub, cv2, board, args, ros.WRIST_TOPIC)
    if reason:
        return "WRIST_K_FAIL " + json.dumps({"reason": reason}), 1
    k, dist, rms = calibrate_views(kept, board, args.square, (w, h))
    cam = Pinhole.from_ros_k(w, h, k)
    rig = dataclasses.replace(rig, wrist_k=k, wrist_size=(w, h), wrist_dist=dist)
    warp = wrist_warp(cam, quat_matrix(rig.wrist_rot).T)
    result = {
        "rms_px": rms,
        "fov_deg": list(cam.fov_deg()),
        "coverage": warp.coverage,
        "frames": len(kept),
    }
    try:
        with open(out_path, "x") as f:
            f.write(rig.to_json())
    except OSError as err:
        detail = f"{err}; K {json.dumps(k)}, dist {json.dumps(dist)}, result {json.dumps(result)}"
        return "WRIST_K_FAIL " + json.dumps({"reason": "write_rig", "detail": detail}), 1
    return "WRIST_K_OK " + json.dumps(result), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "WRIST_K_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "WRIST_K_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
