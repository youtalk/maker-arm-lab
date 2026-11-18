"""The pass or fail of the overhead perception reads (MS5 sensing plan task 7).

    .venv/bin/python scripts/real_perception_verdict.py reads/sparse0.json reads/cluster0.json \\
        reads/heap0.json --check_rms_mm 1.5

Each READ.json is the ``--out`` of ``real_perception_read.py``. The frames of every sparse and
cluster read pool into one knob table (``real.knob_read``), which is judged against the
limits. ``--check_rms_mm`` is the calibration error of the overhead camera: each axis's
``pos_bias_mm`` grows by it before it meets its limit. A row with no sample fails. A heap read
has no truth positions: its detection and poured counts go under ``report_only``. Prints
exactly one ``PERCEPTION_PASS {table, rows}`` or ``PERCEPTION_FAIL {table, rows}`` line (a
usage or input error prints ``PERCEPTION_FAIL {reason}``) and leaves through ``os._exit``.
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


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("PERCEPTION_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = TokenArgumentParser(description=__doc__)
    p.add_argument("reads", nargs="+")
    p.add_argument("--check_rms_mm", type=float, required=True)
    return p


def _run(args) -> tuple[str, int]:
    from maker_arm_lab.real.knob_read import FrameResult, knob_table, verdict
    from maker_arm_lab.real.layouts import Layout, TruthBox, TruthPart

    frames, report_only, period = [], {}, math.pi / 2
    for path in args.reads:
        read = json.loads(Path(path).expanduser().read_text())
        d = read["layout"]
        lay = Layout(
            d["name"],
            d["kind"],
            [TruthPart(**p) for p in d["parts"]],
            [TruthBox(**b) for b in d["boxes"]],
            d.get("frame", "nominal"),
        )
        rows = [FrameResult(lay, f["detections"], f["boxes"]) for f in read["frames"]]
        if lay.kind == "heap":
            report_only[Path(path).stem] = knob_table(rows)["counts"]["heap"]
        else:
            frames += rows
            period = read["box_period"]
    table = knob_table(frames, period)
    table["pos_bias_mm"] = [abs(b) + args.check_rms_mm for b in table["pos_bias_mm"]]
    ok, over = verdict(table)
    payload = {"table": table, "rows": over, "report_only": report_only}
    return ("PERCEPTION_PASS " if ok else "PERCEPTION_FAIL ") + json.dumps(payload), 0 if ok else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = _run(args)
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token = "PERCEPTION_FAIL " + json.dumps({"reason": "exception", "detail": detail})
        code = 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
