"""MS5 design P7: the stamp and photon latency methods on the Conduit stream.

Runs under the lab's uv venv with ROS 2 Jazzy sourced and the ``real`` group installed:

    source /opt/ros/jazzy/setup.zsh
    uv run python scripts/conduit_latency.py stamp --seconds 300 \\
        --out ~/data/maker-arm-ms5/p7/stamp.json
    uv run python scripts/conduit_latency.py photon --on_cmd '<light on>' --off_cmd '<light off>' \\
        --roi 900,600,1020,720 --cycles 30 --out ~/data/maker-arm-ms5/p7/photon.json

``stamp`` logs arrival minus header stamp on every JPEG and depth frame for ``--seconds``.
``photon`` runs ``--on_cmd`` and ``--off_cmd`` in turn every ``--period`` seconds and finds
each switch in the mean brightness of ``--roi`` (x0,y0,x1,y1 in the published JPEG's pixels).
The switch time is taken when the command returns; its run time is reported as the switch
window. The stamp latency of the same frames gives the session's clock offset. Prints
``LATENCY_OK`` or ``LATENCY_FAIL`` with a JSON summary on every exit path and exits through
``os._exit``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from maker_arm_lab.real import latency


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("LATENCY_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def _roi(text: str) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"roi needs x0,y0,x1,y1, got {text!r}") from None
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError(f"empty roi {text!r}")
    return x0, y0, x1, y1


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True, parser_class=TokenArgumentParser)
    stamp = modes.add_parser("stamp")
    stamp.add_argument("--seconds", type=float, default=300.0)
    photon = modes.add_parser("photon")
    photon.add_argument("--on_cmd", required=True)
    photon.add_argument("--off_cmd", required=True)
    photon.add_argument("--roi", type=_roi, required=True)
    photon.add_argument("--cycles", type=int, default=30)
    photon.add_argument("--period", type=float, default=1.0)
    photon.add_argument("--settle", type=float, default=3.0)
    for mode in (stamp, photon):
        mode.add_argument("--namespace", default="/conduit")
        mode.add_argument("--out", default="")
    return parser


def roi_brightness(jpeg: bytes, roi: tuple[int, int, int, int]) -> float:
    """Mean luminance of ``roi`` in a JPEG, decoded at the coarsest DCT scale that keeps the
    roi at least 8 px wide."""
    from PIL import Image

    image = Image.open(io.BytesIO(jpeg))
    full_w, full_h = image.size
    x0, y0, x1, y1 = roi
    reduce = max(1, min(8, (x1 - x0) // 8))
    image.draft("L", (full_w // reduce, full_h // reduce))
    s = image.size[0] / full_w
    box = (
        int(x0 * s),
        int(y0 * s),
        max(int(x1 * s), int(x0 * s) + 1),
        max(int(y1 * s), int(y0 * s) + 1),
    )
    gray = np.asarray(image.convert("L").crop(box), dtype=np.float64)
    return float(gray.mean())


def run_stamp(args) -> dict:
    from maker_arm_lab.real.ros import ConduitSubscriber

    frames: dict[str, list] = {"jpeg": [], "depth": []}
    lock = threading.Lock()

    def keep(key):
        def on(frame):
            with lock:
                frames[key].append(frame.at)

        return on

    sub = ConduitSubscriber(args.namespace, on_jpeg=keep("jpeg"), on_depth=keep("depth")).start()
    try:
        time.sleep(args.seconds)
    finally:
        sub.close()
    with lock:
        got = {k: list(v) for k, v in frames.items()}
    out = {k: latency.stamp_summary(v) for k, v in got.items() if v}
    missing = [k for k, v in got.items() if not v]
    out["seconds"] = args.seconds
    out["pass"] = not missing
    if missing:
        out["reason"] = f"no frames on {missing}"
    return out


def run_photon(args) -> dict:
    from maker_arm_lab.real.ros import ConduitSubscriber

    samples = []  # (arrival_ns, brightness, Arrival)
    lock = threading.Lock()

    def on_jpeg(frame):
        value = roi_brightness(frame.data, args.roi)
        with lock:
            samples.append((frame.at.arrival_ns, value, frame.at))

    sub = ConduitSubscriber(args.namespace, on_jpeg=on_jpeg).start()
    switches, windows = [], []
    try:
        time.sleep(args.settle)
        for k in range(2 * args.cycles):
            on = k % 2 == 0
            before = time.time_ns()
            subprocess.run(shlex.split(args.on_cmd if on else args.off_cmd), check=True)
            after = time.time_ns()
            switches.append((after, on))
            windows.append((after - before) / 1e6)
            time.sleep(args.period)
    finally:
        sub.close()
    with lock:
        got = list(samples)
    if len(got) < 10:
        return {"pass": False, "reason": f"only {len(got)} frames arrived"}
    values = [v for _, v, _ in got]
    threshold = 0.5 * (latency.percentile(values, 10) + latency.percentile(values, 90))
    seen = latency.transitions([(t, v) for t, v, _ in got], threshold)
    delays = latency.pair_switches(switches, seen, int(args.period * 1e9))
    ordered = sorted(a.arrival_mono_ns for _, _, a in got)
    period_ms = latency.percentile([(b - a) / 1e6 for a, b in pairwise(ordered)], 50)
    out = latency.photon_summary(delays, period_ms)
    stamp = latency.summary(a.latency_ms for _, _, a in got)
    c2a = out["capture_to_arrival_ms"]
    out.update(
        switches=len(switches),
        paired=len(delays),
        switch_window_ms=latency.summary(windows),
        threshold=round(threshold, 1),
        stamp_latency_ms=stamp,
        clock_offset_ms=None if c2a is None else latency.clock_offset_ms(stamp["mean"], c2a),
    )
    # A light the camera misses more than one switch in five is not a read.
    out["pass"] = len(delays) >= 0.8 * len(switches)
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_stamp(args) if args.mode == "stamp" else run_photon(args)
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2))
        token = "LATENCY_OK " if summary["pass"] else "LATENCY_FAIL "
        print(token + json.dumps(summary), flush=True)
        return 0 if summary["pass"] else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("LATENCY_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())
