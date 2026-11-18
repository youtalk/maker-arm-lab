"""The overhead perception read, with no arm and no policy (MS5 sensing plan task 7).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_perception_read.py \\
        --rig rig_cal.json --layout ~/data/maker-arm-ms5/layouts/sparse0.json \\
        --screen screen.json --worker tcp://127.0.0.1:5600 --frames 20 \\
        --out reads/sparse0.json --save_dir reads/sparse0

Run it with ROS 2 Jazzy sourced, ``RMW_IMPLEMENTATION=rmw_zenoh_cpp`` and the
``ROS_DOMAIN_ID`` of the Conduit setup; keep the ROS Python path in ``PYTHONPATH``. The
perception worker (``perception_stack/worker.py``) must already run.

The script subscribes to the iPhone's Conduit stream, warps each new frame, asks the worker for
the boxes and the parts, and matches them against the truth layout (``real.knob_read``). The
rig must come from ``real_calibrate_overhead.py`` (it sets ``overhead_size``) and the stream
must have that size: a turned phone fails. With ``--screen`` (from ``real_screen_pose.py``)
the truth is the layout moved by the screen's measured pose. ``--out`` gets the per-frame
detections, the knob table, ``k_drift``, ``rebuilds``, ``k_first`` and ``k_last`` (the K, ROS
convention, that the warp used at the first and the last frame), the rig's ``tray_floor_z``
(the floor the worker should have), the warp's ``coverage`` and the p50 and p95 of arrival
minus capture stamp (these include the offset between the iPhone's clock and the host's). A
frame counts only when its bundle carries depth, which the worker needs for the parts. With
``--save_dir`` it also writes the warped 224 view (``view_NNN.png``) and the raw JPEG
(``raw_NNN.jpg``) of frame 0 and of every fifth frame. Prints exactly one
``PERCEPTION_READ_OK {table}`` or ``PERCEPTION_READ_FAIL {reason}`` line and leaves through
``os._exit``. Pool the reads with ``real_perception_verdict.py``.
"""

import argparse
import dataclasses
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

SAVE_EVERY = 5
DEPTH_TOPIC = "/conduit/camera/wide/depth/image_raw"  # real.ros.NAMESPACE + TOPICS["depth"]


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("PERCEPTION_READ_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--screen", default=None)
    p.add_argument("--worker", default="tcp://127.0.0.1:5600")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--timeout_s", type=float, default=60.0)
    p.add_argument("--out", required=True)
    p.add_argument("--save_dir", default=None)
    return p


def box_period(rig) -> float:
    """A quarter turn for a square box interior (sides within 5 mm, as a caliper reads a square
    box), a half turn otherwise."""
    x, y, _ = rig.box_interior
    return math.pi / 2 if abs(x - y) < 0.005 else math.pi


def truth_layout(rig, layout, screen_path):
    """``layout`` moved by the screen pose in ``screen_path`` (``pos``, ``quat``), or as it is."""
    if screen_path is None:
        return layout
    from maker_arm_lab.real.camera_model import Pose
    from maker_arm_lab.real.layouts import nominal_screen_pose, place_layout

    d = json.loads(Path(screen_path).expanduser().read_text())
    return place_layout(layout, nominal_screen_pose(rig), Pose.from_pos_quat(d["pos"], d["quat"]))


def _save(save_dir: Path, i: int, latest) -> None:
    from PIL import Image

    save_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(latest.view).save(save_dir / f"view_{i:03d}.png")
    (save_dir / f"raw_{i:03d}.jpg").write_bytes(bytes(latest.raw))


def run_read(
    overhead,
    client,
    layout,
    frames: int,
    save_dir: Path | None,
    timeout_s: float = 60.0,
    clock=time.monotonic,
    sleep=time.sleep,
    period: float = math.pi / 2,
    size: tuple[int, int] | None = None,
) -> dict:
    """Wait for ``frames`` JPEGs with distinct stamps, ask ``client`` for the boxes and the
    parts of each, and return the payload. Raises ``TimeoutError`` when they do not arrive and
    ``ValueError`` when the camera is not ``size`` (the calibration's)."""
    from maker_arm_lab.real.knob_read import FrameResult, knob_table

    end = clock() + timeout_s
    seen: set = set()
    results, rows, lags, ks = [], [], [], []
    while len(rows) < frames:
        latest = overhead.frames.get()
        key = latest and (latest.arrival_mono_ns if latest.stamp_ns is None else latest.stamp_ns)
        bundle = latest and key not in seen and overhead.bundle(latest)
        if not bundle or "depth" not in bundle.arrays:
            if clock() > end:
                raise TimeoutError(
                    f"{len(rows)} of {frames} frames with depth in {timeout_s} s; "
                    f"dropped {getattr(overhead, 'dropped', 0)}, "
                    f"camera info {'yes' if getattr(overhead, 'camera', 1) else 'no'}, "
                    f"depth {'yes' if getattr(overhead, 'depth_camera', 1) else 'no'} "
                    f"(depth comes on {DEPTH_TOPIC})"
                )
            sleep(0.01)
            continue
        camera = overhead.camera
        if size is not None and (camera.width, camera.height) != tuple(size):
            raise ValueError(
                f"the stream is {camera.width}x{camera.height}, the rig was calibrated at "
                f"{size[0]}x{size[1]}: did the phone turn?"
            )
        ks.append(list(camera.ros_k()))
        seen.add(key)
        if save_dir is not None and len(rows) % SAVE_EVERY == 0:
            _save(Path(save_dir), len(rows), latest)
        boxes = client.call("boxes", bundle.header, bundle.arrays)[0].get("poses", [])
        detections = client.call("parts", bundle.header, bundle.arrays)[0].get("detections", [])
        results.append(FrameResult(layout, detections, boxes))
        rows.append({"stamp_ns": latest.stamp_ns, "detections": detections, "boxes": boxes})
        if latest.stamp_ns is not None:  # arrival on the wall clock, from its monotonic stamp
            arrival_ns = time.time_ns() - (time.monotonic_ns() - latest.arrival_mono_ns)
            lags.append((arrival_ns - latest.stamp_ns) / 1e6)
    warp = getattr(overhead, "warp", None)
    p50, p95 = np.percentile(lags, [50, 95]) if lags else (math.nan, math.nan)
    return {
        "layout": dataclasses.asdict(layout),
        "frames": rows,
        "table": knob_table(results, period),
        "box_period": period,
        "k_drift": overhead.k_drift,
        "rebuilds": overhead.rebuilds,
        "k_first": ks[0],
        "k_last": ks[-1],
        "coverage": getattr(warp, "coverage", math.nan),
        "latency_ms": {"p50": float(p50), "p95": float(p95)},
    }


def _run(args) -> tuple[str, int]:
    if args.frames < 1:
        return "PERCEPTION_READ_FAIL " + json.dumps({"reason": "usage: --frames >= 1"}), 2
    from maker_arm_lab.real.camera_model import Pinhole, Pose
    from maker_arm_lab.real.layouts import Layout
    from maker_arm_lab.real.perceive import WorkerClient
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.sources import ConduitOverhead

    out = Path(args.out).expanduser()
    if not out.parent.is_dir():
        return "PERCEPTION_READ_FAIL " + json.dumps({"reason": f"{out.parent} is not a folder"}), 1
    rig = RigConfig.from_json(Path(args.rig).expanduser())
    layout = truth_layout(rig, Layout.from_json(Path(args.layout).expanduser()), args.screen)
    camera = None
    if rig.overhead_size is None:
        detail = "the rig has no overhead_size: calibrate first (real_calibrate_overhead.py)"
        return "PERCEPTION_READ_FAIL " + json.dumps({"reason": "rig", "detail": detail}), 1
    if rig.overhead_k is not None:
        camera = Pinhole.from_ros_k(*rig.overhead_size, rig.overhead_k)
    from maker_arm_lab.real import ros  # rclpy: only once the inputs are checked

    overhead = ConduitOverhead(
        Pose.from_pos_quat(rig.overhead_pos, rig.overhead_quat),
        camera,
        rig.desk_z,
        rig.overhead_back_m,
    )
    client = WorkerClient(args.worker)
    sub = ros.ConduitSubscriber(
        on_jpeg=overhead.on_jpeg,
        on_info=overhead.on_info,
        on_depth=overhead.on_depth,
        on_depth_info=overhead.on_depth_info,
        on_error=overhead.on_error,
    ).start()
    try:
        payload = run_read(
            overhead,
            client,
            layout,
            args.frames,
            Path(args.save_dir).expanduser() if args.save_dir else None,
            args.timeout_s,
            period=box_period(rig),
            size=rig.overhead_size,
        )
    except (TimeoutError, ValueError) as err:
        return "PERCEPTION_READ_FAIL " + json.dumps(
            {"reason": str(err), "error": overhead.error}
        ), 1
    finally:
        sub.close()
        client.close()
    payload["tray_floor_z"] = rig.tray_floor_z
    out.write_text(json.dumps(payload, indent=2))
    return "PERCEPTION_READ_OK " + json.dumps(payload["table"]), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "PERCEPTION_READ_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token = "PERCEPTION_READ_FAIL " + json.dumps({"reason": "exception", "detail": detail})
        code = 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
