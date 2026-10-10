"""The sheet of one real episode (MS5 design section 7.1): where the operator places the four
desk boxes and which class goes to which box, drawn from a pre-registered seed.

    uv run python scripts/real_episode_sheet.py --seed 20265000 --episode 0 \\
        --out ~/data/maker-arm-ms5/sheets/e000.json

Prints the placement table, then ``SHEET_OK`` with the sheet as JSON, or ``SHEET_FAIL``, on
every exit path, and exits through ``os._exit``. The loop reads only ``class_to_box`` from
the file; it perceives the boxes' poses.
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
        print("SHEET_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--out", default="")
    return parser


def table(sheet: dict) -> str:
    lines = ["box  side  x mm   y mm  yaw deg  classes"]
    for label, (x, y, yaw) in enumerate(sheet["box_poses"]):
        classes = [c for c, box in zip(sheet["classes"], sheet["class_to_box"]) if box == label]
        side = "+y" if y > 0 else "-y"
        lines.append(
            f"box {label} {side:>4} {1e3 * x:6.1f} {1e3 * y:6.1f} {math.degrees(yaw):8.1f}  "
            + ", ".join(classes)
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from maker_arm_lab.real.rig import episode_sheet

        sheet = episode_sheet(args.seed, args.episode)
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(sheet, indent=2))
        print(table(sheet), flush=True)
        print("SHEET_OK " + json.dumps(sheet), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("SHEET_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())
