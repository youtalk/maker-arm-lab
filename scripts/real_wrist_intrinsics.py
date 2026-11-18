"""Calibrate the wrist camera's intrinsics from its ROS topic (MS5 sensing plan task 11).

    source ~/opt/env.sh  # ROS
    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_wrist_intrinsics.py \
        --rig ~/data/maker-arm-ms5/rig.json \
        --size 640x480 --board 9x6 --square 0.02001 --frames 25 --gap_s 1.0 \
        --rig_out ~/data/maker-arm-ms5/rig_wrist.json

Keep the ROS Python path in ``PYTHONPATH``: replacing it drops rclpy. The iPad shows
``real.layouts.checker_png()`` and the operator holds it in front of the camera at 0.15 to
0.3 m (at 0.1 m the 200 x 140 mm board does not fit a camera of about 90 degrees) and at
several tilts. The gripper jaws can block image corners, so the operator covers the corners
the camera can see. The script reads ``WRIST_TOPIC`` through ``real.ros.ConduitSubscriber``.
A frame becomes a view when ``cv2.findChessboardCorners`` succeeds, the board was also found
in the last ``--still_n`` frames with the median corner moving no more than ``--still_px``, and at least
``--gap_s`` seconds of arrival time passed since the last view and its mean corner
displacement from every earlier view is at least ``--min_move_px``. After ``--frames`` views it
solves with ``cv2.calibrateCameraExtended``, drops the views whose own error is above
``OUTLIER_X`` times the median and 0.5 px (once, if ``--min_views`` remain) and solves again.
It saves each view as ``view_NN.png`` and ``views.json`` (corners, error, ``dropped``) in
``--save_dir`` (default: ``--rig_out`` without its suffix plus ``_views``); that folder must
not exist. ``--rig_out`` is the input rig with ``wrist_k`` (OpenCV's K, the ROS convention),
``wrist_size`` and ``wrist_dist``; it must not exist, and the input rig is never written. The
script opens no device. It prints progress lines to stderr and exactly one
``WRIST_K_OK {rms_px, fov_deg: [h, v], coverage, frames, dropped_views, std_px: {fx, fy, cx,
cy}, save_dir}`` (``coverage`` is the share of the simulated wrist view that the camera covers
through the lab's wrist warp; ``frames`` counts the views of the final solve) or
``WRIST_K_FAIL {reason}`` line on every exit path and leaves through ``os._exit``.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

DROP_FAIL = 10
"""Dropped frames, with none accepted, after which the wait ends."""
SILENT_FAIL_S = 10.0
"""Seconds with no frame at all after which the wait ends."""
PROGRESS_S = 5.0
OUTLIER_X = 3.0
"""A view whose rms is above this times the median view rms (and 0.5 px) is dropped."""
OUTLIER_MIN_PX = 0.5
MIN_MOVE_PX = 20.0
"""Least mean corner displacement of a new view from every kept view, px."""


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
    p.add_argument("--still_n", type=int, default=3, help="frames the board must hold still")
    p.add_argument("--still_px", type=float, default=0.5, help="largest corner move when still")
    p.add_argument("--min_move_px", type=float, default=MIN_MOVE_PX, help="least move per view")
    p.add_argument("--min_views", type=int, default=15, help="least views to allow an outlier drop")
    p.add_argument(
        "--save_dir", default=None, help="folder for the views (default next to rig_out)"
    )
    p.add_argument("--timeout_s", type=float, default=300.0)
    p.add_argument("--rig_out", required=True, help="the rig JSON to create")
    return p


def still(history: Sequence[np.ndarray], still_px: float, still_n: int = 3) -> bool:
    """True when the last ``still_n`` corner arrays (N, 1, 2) exist and the median corner
    moved no more than ``still_px`` between consecutive ones."""
    if len(history) < still_n:
        return False
    last = history[-still_n:]
    return all(
        float(np.median(np.linalg.norm(b.reshape(-1, 2) - a.reshape(-1, 2), axis=1))) <= still_px
        for a, b in pairwise(last)
    )


def _progress(seen, kept, src, frames, counts):
    print(
        f"frames seen {seen}, views kept {kept}/{frames}, moving {counts['moving']}, "
        f"same_pose {counts['same_pose']}, dropped {src.dropped}, error {src.error}",
        file=sys.stderr,
        flush=True,
    )


def _move(a, b):
    """Mean corner displacement between two corner arrays, px."""
    return float(np.linalg.norm(a.reshape(-1, 2) - b.reshape(-1, 2), axis=1).mean())


def collect(
    src, subscriber, cv2, board, frames, gap_s, timeout_s, topic="the wrist topic",
    still_n=3, still_px=0.5, min_move_px=MIN_MOVE_PX, counts=None,
):  # fmt: skip
    """([(RGB frame, corners)] of the kept views, the reason the wait ended early or None).
    ``counts``, if given, receives the frames rejected as ``moving`` and as ``same_pose``."""
    counts = {} if counts is None else counts
    counts.update(moving=0, same_pose=0)
    kept, seen, last_seen, last_kept, history = [], 0, None, None, []
    start = last_print = time.monotonic()
    end = start + timeout_s
    while len(kept) < frames and time.monotonic() < end:
        time.sleep(0.02)
        now = time.monotonic()
        if now - last_print >= PROGRESS_S:
            last_print = now
            _progress(seen, len(kept), src, frames, counts)
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
        gray = cv2.cvtColor(latest.raw, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board)
        if not found:
            history.clear()
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )  # fmt: skip
        history = [*history, corners][-still_n:]
        if len(history) >= still_n and not still(history, still_px, still_n):
            counts["moving"] += 1
            continue
        if last_kept is not None and (last_seen - last_kept) / 1e9 < gap_s:
            continue
        if len(history) < still_n:
            continue
        if any(_move(corners, c) < min_move_px for _, c in kept):
            counts["same_pose"] += 1
            continue
        kept.append((latest.raw.copy(), corners))
        last_kept = last_seen
        _progress(seen, len(kept), src, frames, counts)
    return kept, None


def wait_for_views(src, sub, cv2, board, args, topic="the wrist topic"):
    """(kept views, None), or (what was kept, the reason it fell short). Closes ``sub``;
    the reason quotes the source's and the subscriber's ``error``."""
    counts = {}
    try:
        kept, early = collect(
            src, sub, cv2, board, args.frames, args.gap_s, args.timeout_s, topic,
            args.still_n, args.still_px, args.min_move_px, counts,
        )  # fmt: skip
    finally:
        sub.close()
    if len(kept) >= args.frames:
        return kept, None
    reason = early or f"{len(kept)} of {args.frames} frames with the board before the timeout"
    causes = [e for e in (src.error, sub.error) if e and e not in reason]
    tally = f"rejected as moving {counts['moving']}, as same_pose {counts['same_pose']}"
    return kept, "; ".join([reason, tally, *causes])


@dataclasses.dataclass(frozen=True)
class Calib:
    k: tuple[float, ...]
    dist: tuple[float, ...]
    rms_px: float
    view_rms_px: tuple[float, ...]
    """Each view's rms in the first solve, the one that decided the drop."""
    std_px: dict[str, float]
    dropped: tuple[int, ...]
    kept_outliers: tuple[int, ...] = ()
    """Outliers left in because dropping them would leave fewer than ``min_views``."""


def _solve(cv2, obj, views, size):
    rms, k, dist, rvecs, tvecs, std, _, _ = cv2.calibrateCameraExtended(
        [obj] * len(views), views, size, None, None
    )
    per_view = []
    for v, r, t in zip(views, rvecs, tvecs):
        proj = cv2.projectPoints(obj, r, t, k, dist)[0].reshape(-1, 2)
        per_view.append(float(np.sqrt(((proj - v.reshape(-1, 2)) ** 2).sum(axis=1).mean())))
    return rms, k, dist, std.ravel(), per_view


def calibrate_views(views, board, square_m, size, min_views=15) -> Calib:
    """The camera from the board's corners in each view, ``board`` inner corners wide by high.
    Views far above the median error are dropped once, if ``min_views`` remain."""
    import cv2

    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2) * square_m
    rms, k, dist, std, per_view = _solve(cv2, obj, views, size)
    first = tuple(per_view)
    limit = max(OUTLIER_X * float(np.median(per_view)), OUTLIER_MIN_PX)
    bad = tuple(i for i, r in enumerate(per_view) if r > limit)
    kept_outliers = ()
    if bad and len(views) - len(bad) >= min_views:
        rest = [v for i, v in enumerate(views) if i not in bad]
        rms, k, dist, std, _ = _solve(cv2, obj, rest, size)
    elif bad:
        kept_outliers, bad = bad, ()
    names = ("fx", "fy", "cx", "cy")
    return Calib(
        tuple(float(v) for v in k.ravel()), tuple(float(v) for v in dist.ravel()[:5]), float(rms),
        first, {n: float(std[i]) for i, n in enumerate(names)}, bad, kept_outliers,
    )  # fmt: skip


def save_views(save_dir, rgbs, corners, calib=None):
    """Write ``view_NN.png`` for each view and ``views.json`` into the new folder ``save_dir``."""
    save_dir = Path(save_dir)
    save_dir.mkdir()
    entries = []
    for i, (rgb, c) in enumerate(zip(rgbs, corners)):
        name = f"view_{i:02d}.png"
        Image.fromarray(rgb).save(save_dir / name)
        entry = {"file": name, "corners": c.reshape(-1, 2).tolist()}
        entry["rms_px"] = calib.view_rms_px[i] if calib else None
        if calib and i in calib.dropped:
            entry["dropped"] = True
        entries.append(entry)
    (save_dir / "views.json").write_text(json.dumps(entries))


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
    if args.still_n < 2:
        return "WRIST_K_FAIL " + json.dumps({"reason": "usage", "detail": "--still_n >= 2"}), 2
    out_path = Path(args.rig_out).expanduser()
    if not out_path.parent.is_dir():
        detail = f"{out_path.parent} is not a folder"
        return "WRIST_K_FAIL " + json.dumps({"reason": "rig_out", "detail": detail}), 1
    if out_path.exists():
        detail = f"{out_path} exists"
        return "WRIST_K_FAIL " + json.dumps({"reason": "rig_out", "detail": detail}), 1
    save_dir = (
        Path(args.save_dir).expanduser()
        if args.save_dir
        else out_path.with_name(out_path.stem + "_views")
    )
    if save_dir.exists() or not save_dir.parent.is_dir():
        detail = f"{save_dir} exists" if save_dir.exists() else f"{save_dir.parent} is not a folder"
        return "WRIST_K_FAIL " + json.dumps({"reason": "save_dir", "detail": detail}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    # The source reads only the size of this placeholder camera.
    src = WristFrames(
        Pinhole.from_ros_k(w, h, (1, 0, w / 2 - 0.5, 0, 1, h / 2 - 0.5, 0, 0, 1)), None
    )
    sub = ros.ConduitSubscriber(on_wrist=src.on_jpeg).start()
    kept, reason = wait_for_views(src, sub, cv2, board, args, ros.WRIST_TOPIC)
    rgbs, corners = [v[0] for v in kept], [v[1] for v in kept]
    if reason:
        if kept:
            save_views(save_dir, rgbs, corners)
        return "WRIST_K_FAIL " + json.dumps({"reason": reason}), 1
    calib = calibrate_views(corners, board, args.square, (w, h), args.min_views)
    k, dist, rms = calib.k, calib.dist, calib.rms_px
    try:
        save_views(save_dir, rgbs, corners, calib)
    except OSError as err:
        detail = f"{err}; K {json.dumps(k)}, dist {json.dumps(dist)}, rms_px {rms}"
        return "WRIST_K_FAIL " + json.dumps({"reason": "save_dir", "detail": detail}), 1
    cam = Pinhole.from_ros_k(w, h, k)
    rig = dataclasses.replace(rig, wrist_k=k, wrist_size=(w, h), wrist_dist=dist)
    warp = wrist_warp(cam, quat_matrix(rig.wrist_rot).T)
    result = {
        "rms_px": rms,
        "fov_deg": list(cam.fov_deg()),
        "coverage": warp.coverage,
        "frames": len(kept) - len(calib.dropped),
        "dropped_views": list(calib.dropped),
        "std_px": calib.std_px,
        "save_dir": str(save_dir),
    }
    if calib.kept_outliers:
        result["outliers_kept"] = list(calib.kept_outliers)
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
