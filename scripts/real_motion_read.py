"""The motion reads of one session log (MS5 control plan task 9): jitter, sway, yield and flex,
the numbers the pre-registered adoption rules are judged on.

    .venv/bin/python scripts/real_motion_read.py --log ~/data/maker-arm-ms5/logs/s4.h5 \\
        --tag_log ~/data/maker-arm-ms5/tags/s4.jsonl --latency_ms 91 \\
        --rig ~/data/maker-arm-ms5/rig.json --pairs_out ~/data/maker-arm-ms5/s4_pairs.json \\
        --out ~/data/maker-arm-ms5/s4_motion.json

``--log`` is a session log of ``scripts/real_session.py``. ``--tag_log`` is the overhead
camera's gripper-tag log (``scripts/real_tag_log.py`` of the sensing plan) and
``--latency_ms`` (required with it) the sensing ledger's stamp p50: a frame's host time is its
``arrival_mono_ns`` less it. ``--rig`` is read as a plain JSON object, not through
``RigConfig``, for ``tag_offset`` (the tag centre in the ``grasp_center`` frame, m) and the
registered ``overhead_pos`` / ``overhead_quat`` (world from camera, (w, x, y, z)). With both
overhead keys the tag's world position is its ``cam_pos`` through that pose; otherwise the
logged ``pos``. The tip-side metrics (``tip_sway``, ``flex_share``) need a tag log and a rig
with ``tag_offset``; without one the read reports the motor side only and says why under
``tip_side``. ``--pairs_out`` writes the hold pairs (``real.motion.hold_pairs``) whenever a
tag log is given, ``{"schema", "log", "tag_log", "latency_ms", "pairs": [{"q", "tag_cam",
"start_t", "end_t", "label", "n_q", "n_tag"}]}``, the input of the sensing plan's
``register_base``. No output file is ever overwritten.

The output JSON (``--out``, schema ``maker-arm-lab/motion-read/1``):

- ``metrics``: name -> ``{"joints": [6] | null, "tip": float | null, "count": int}`` for
  ``tracking_rms``, ``stop_sway``, ``hold_jitter``, ``free_residual_rms``, ``yield_mm``,
  ``drag_resistance``, ``tip_sway`` and ``flex_share``. ``joints`` is one value per arm joint
  in URDF order (``link_002_joint`` .. ``link_007_joint``), ``tip`` the value pooled at the
  tip, ``count`` the segments behind it (the free rows, for ``free_residual_rms``). A null is
  a value with no data (an empty window, or a metric without that side).
- ``units``: name -> ``{"joints": unit | null, "tip": unit | null}`` (rad, mm, N m, fraction).
- ``pooling``: name -> how ``metrics`` pools the segments (rms over rows, mean, max).
- ``segments``: one entry per read window in time order, ``{"index", "kind", "label",
  "start_t", "end_t", "rows", "reads": {name: {"joints", "tip"}}}``; ``kind`` is ``traj``,
  ``stop``, ``hold`` or ``prompt``, ``index`` the row in the log's ``segments`` (a stop: its
  transport's).
- ``tip_side``: ``{"available", "reason", "tag_world"}``; ``tag_world`` is ``rig_overhead``
  or ``logged_pos``.
- ``pairs``: null without a tag log, else ``{"holds", "pairs", "out"}``.
- ``log``, ``tag_log``, ``rig``, ``latency_ms``, ``tag_frames``, ``joint_names``.

It prints exactly one ``MOTION_READ_OK {out, metrics: {name: {joints, tip}}, tip_side}`` or
``MOTION_READ_FAIL {reason, detail}`` line on every exit path and exits through
``os._exit``.
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


class Failure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _fail_usage(message: str) -> None:
    print("MOTION_READ_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
    os._exit(2)


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _fail_usage(message)


def _latency(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0.0):
        raise argparse.ArgumentTypeError("latency_ms is a finite number >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="the session log (HDF5)")
    parser.add_argument("--tag_log", help="the gripper-tag log (JSONL)")
    parser.add_argument("--latency_ms", type=_latency, help="the tag stamp latency, ms")
    parser.add_argument("--rig", help="the rig JSON, for tag_offset and the overhead pose")
    parser.add_argument("--pairs_out", help="the hold pairs JSON to create")
    parser.add_argument("--out", required=True, help="the read JSON to create")
    return parser


def _read_tag_log(path: Path) -> list[dict]:
    lines = []
    with open(path) as f:
        for k, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError as err:
                raise Failure("tag_log", f"{path} line {k}: {err}") from err
            if not isinstance(line, dict) or "arrival_mono_ns" not in line:
                raise Failure("tag_log", f"{path} line {k}: no arrival_mono_ns")
            lines.append(line)
    return lines


def _rig(path: Path | None):
    """``tag_offset`` and the overhead pose, or None, from the rig JSON as a plain dict (ruling
    R7: it must read the same before and after ``RigConfig`` gains ``tag_offset``)."""
    if path is None:
        return None, None
    if not path.is_file():
        raise Failure("rig", f"no such rig: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise Failure("rig", f"{path}: {err}") from err
    if not isinstance(data, dict):
        raise Failure("rig", f"{path} is not a JSON object")
    offset = data.get("tag_offset")
    if offset is not None:
        offset = [float(v) for v in offset]
        if len(offset) != 3 or not all(math.isfinite(v) for v in offset):
            raise Failure("rig", f"tag_offset must be three finite numbers, got {offset}")
    pose = None
    if data.get("overhead_pos") is not None and data.get("overhead_quat") is not None:
        from maker_arm_lab.real.camera_model import quat_matrix

        pos = [float(v) for v in data["overhead_pos"]]
        quat = [float(v) for v in data["overhead_quat"]]
        if len(pos) != 3 or len(quat) != 4:
            raise Failure("rig", "overhead_pos is 3 numbers and overhead_quat 4")
        pose = (pos, quat_matrix(quat))
    return offset, pose


def run(args) -> dict:
    log_path = Path(args.log).expanduser()
    out = Path(args.out).expanduser()
    pairs_out = Path(args.pairs_out).expanduser() if args.pairs_out else None
    taken = [str(p) for p in (out, pairs_out) if p is not None and p.exists()]
    if taken:
        raise Failure("exists", f"{taken} exist; a read is never overwritten")
    if not log_path.is_file():
        raise Failure("log", f"no such log: {log_path}")
    tag_path = Path(args.tag_log).expanduser() if args.tag_log else None
    if tag_path is not None and not tag_path.is_file():
        raise Failure("tag_log", f"no such tag log: {tag_path}")
    rig_path = Path(args.rig).expanduser() if args.rig else None
    tag_offset, pose = _rig(rig_path)

    from maker_arm_lab.assets.urdf_prep import ARM_JOINT_NAMES
    from maker_arm_lab.core.global_ik import GlobalIK
    from maker_arm_lab.real import motion

    try:
        log = motion.load_log(log_path)
    except (OSError, KeyError) as err:
        raise Failure("log", f"{log_path}: {err}") from err
    tag_lines = _read_tag_log(tag_path) if tag_path is not None else None
    body = motion.read(
        log,
        GlobalIK(),
        tag_lines=tag_lines,
        latency_ms=args.latency_ms,
        tag_offset=tag_offset,
        world_from_cam=pose,
    )
    pairs = body.pop("pairs_list")
    body["pairs"] = None
    if pairs is not None:
        holds = sum(1 for s in body["segments"] if s["kind"] == "hold")
        body["pairs"] = {"holds": holds, "pairs": len(pairs), "out": str(pairs_out or "")}
    body.update(
        log=str(log_path),
        tag_log=None if tag_path is None else str(tag_path),
        rig=None if rig_path is None else str(rig_path),
        latency_ms=args.latency_ms,
        joint_names=list(ARM_JOINT_NAMES),
    )
    if pairs is not None and pairs_out is not None:
        pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pairs_out, "x") as f:
            json.dump(
                {
                    "schema": "maker-arm-lab/hold-pairs/1",
                    "log": str(log_path),
                    "tag_log": str(tag_path),
                    "latency_ms": args.latency_ms,
                    "pairs": pairs,
                },
                f,
                indent=2,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(body, f, indent=2)
    return {
        "out": str(out),
        "metrics": {
            k: {"joints": v["joints"], "tip": v["tip"]} for k, v in body["metrics"].items()
        },
        "tip_side": body["tip_side"],
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.tag_log and args.latency_ms is None:
        _fail_usage("--tag_log needs --latency_ms (the sensing ledger's stamp p50)")
    try:
        token, code = "MOTION_READ_OK " + json.dumps(run(args)), 0
    except Failure as err:
        token = "MOTION_READ_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token, code = "MOTION_READ_FAIL " + json.dumps({"reason": "interrupted", "detail": ""}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "MOTION_READ_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
