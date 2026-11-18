"""Register the overhead camera to the arm's base from a gripper tag (MS5 sensing plan task 3).

    PYTHONPATH=$PWD/source:$PYTHONPATH .venv/bin/python scripts/real_register_base.py \
        --rig rig.json --pairs pairs.json --rig_out rig_base.json --report base_report.json

``pairs.json`` is ``[{"q": [6 joints], "tag_cam": [x, y, z]}, ...]``: the arm's joint vector
and the gripper tag's centre in the camera frame at that pose (``scripts/real_tag_log.py``'s
``cam_pos``), at least 8 pairs at varied poses. ``real.tags.register_base`` starts from the
rig's provisional overhead pose. ``--rig_out`` is the input rig with ``overhead_pos``,
``overhead_quat`` and ``tag_offset``; the input rig is never written. Prints exactly one
``REGISTER_OK {rms_mm, max_mm, shift_mm, yaw_deg, tilt_deg}`` or ``REGISTER_FAIL {reason}`` line
and leaves through ``os._exit``. The shift, yaw and tilt are the change against the
provisional pose.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

MIN_PAIRS = 8


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("REGISTER_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("--rig", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--rig_out", required=True, help="the rig JSON to create")
    p.add_argument("--report", required=True, help="the report JSON to create")
    return p


def _run(args) -> tuple[str, int]:
    pairs = json.loads(Path(args.pairs).expanduser().read_text())
    if len(pairs) < MIN_PAIRS:
        reason = f"{len(pairs)} pairs, need {MIN_PAIRS}"
        return "REGISTER_FAIL " + json.dumps({"reason": reason}), 1
    out_path, report_path = Path(args.rig_out).expanduser(), Path(args.report).expanduser()
    for p in (out_path, report_path):
        if p.exists() or not p.parent.is_dir():
            detail = f"{p} exists" if p.exists() else f"{p.parent} is not a folder"
            return "REGISTER_FAIL " + json.dumps({"reason": "output", "detail": detail}), 1
    from maker_arm_lab.core.global_ik import GlobalIK
    from maker_arm_lab.real.camera_model import Pose
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.real.tags import register_base

    rig = RigConfig.from_json(Path(args.rig).expanduser())
    guess = Pose.from_pos_quat(rig.overhead_pos, rig.overhead_quat)
    q = np.array([p["q"] for p in pairs], dtype=np.float64)
    tag_cam = np.array([p["tag_cam"] for p in pairs], dtype=np.float64)
    pose, off, rms, per = register_base(q, tag_cam, GlobalIK(), guess)
    rel = pose.rot @ guess.rot.T  # the turn in the base frame
    result = {
        "rms_mm": rms,
        "max_mm": float(per.max()),
        "shift_mm": float(np.linalg.norm(pose.pos - guess.pos) * 1000.0),
        "yaw_deg": math.degrees(math.atan2(rel[1, 0], rel[0, 0])),
        "tilt_deg": math.degrees(math.acos(float(np.clip(rel[2, 2], -1.0, 1.0)))),
    }
    new = dataclasses.replace(
        rig,
        overhead_pos=tuple(float(v) for v in pose.pos),
        overhead_quat=pose.quat(),
        tag_offset=tuple(float(v) for v in off),
    )
    report_path.write_text(json.dumps(result | {"residuals_mm": per.tolist()}, indent=2))
    with open(out_path, "x") as f:
        f.write(new.to_json())
    return "REGISTER_OK " + json.dumps(result), 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except KeyboardInterrupt:
        token, code = "REGISTER_FAIL " + json.dumps({"reason": "interrupted"}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "REGISTER_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
