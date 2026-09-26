"""The tracking bench client (tracking design section 8): Task 11 runs this for the official
gate.

Thin CLI over ``maker_arm_lab.mj.bench_gate.run_suite``: runs the pre-registered scenario
suite, or with ``--smoke`` the single short sag check, and prints ``BENCH_OK``/``BENCH_FAIL``
with a JSON summary (the gate table and the verdict, or the smoke numbers). ``--out`` also
writes the whole result -- every scenario, the calibration, the report and the gate table --
as JSON. A failed write adds ``out_error`` to the summary and leaves the verdict as it is.

Prints its token on every exit path, argparse errors and exceptions included, and exits
through ``os._exit``. ``bench_gate`` is imported inside ``main`` so a usage error never pays
for MuJoCo and torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("BENCH_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="write the full result JSON here")
    parser.add_argument("--smoke", action="store_true", help="mj.bench_gate's short smoke run")
    parser.add_argument("--seed", type=int, default=20260925)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from maker_arm_lab.mj import bench_gate

        result = bench_gate.run_suite(seed=args.seed, smoke=args.smoke)
        ok = bool(result["pass"])
        if args.smoke:
            summary = dict(result["smoke"])
        else:
            summary = {"gate": result["gate"], "pass": result["pass"]}
        if args.out:
            # A failed write is reported next to the verdict and never changes it.
            try:
                Path(args.out).write_text(json.dumps(result, indent=2, default=float))
            except Exception as exc:  # noqa: BLE001
                summary["out_error"] = f"{type(exc).__name__}: {exc}"
        token = "BENCH_OK " if ok else "BENCH_FAIL "
        print(token + json.dumps(summary, default=float), flush=True)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("BENCH_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())
