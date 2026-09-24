"""The MS7c hold agreement: the proprioceptive hold detector against the contact truth.

Runs under the repo venv (``.venv/bin/python scripts/hold_agreement.py``). Two modes:

* ``--payloads GLOB...``: the agreement over every cycle of the EVAL payloads
  ``scripts/mj_eval_heap.py --out`` writes, from their census rows (``core.hold.agreement``).
* ``--traces GLOB...``: replays ``core.hold.HoldDetector`` over the per-step traces
  ``scripts/mj_eval_heap.py --hold_trace`` writes, at the module thresholds or the overrides
  given, rebuilds each cycle's row (``core.hold.cycle_rows``) and reports their agreement. At
  the module thresholds a trace reproduces its own payload's agreement.

Prints exactly one terminal token line: ``HOLD_OK {json}`` when the pre-registered bar of MS7c
design section 5 passes (``core.hold.meets_bar``), ``HOLD_FAIL {json}`` when it does not, and
``HOLD_ERROR {json}`` on any error, argparse errors included. Exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import traceback
from pathlib import Path

from maker_arm_lab.core import hold

THRESHOLDS = ("min_excess", "min_gap", "stable", "stable_steps", "confirm_steps", "exit_steps")


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("HOLD_ERROR " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--payloads", nargs="+", help="EVAL payload JSON paths or globs")
    mode.add_argument("--traces", nargs="+", help="gzipped JSON-lines hold trace paths or globs")
    for name, kind in zip(THRESHOLDS, (float, float, float, int, int, int)):
        parser.add_argument(
            f"--{name}",
            type=kind,
            default=None,
            help=f"--traces only: override core.hold.{name.upper()}",
        )
    args = parser.parse_args(argv)
    if args.payloads and any(getattr(args, n) is not None for n in THRESHOLDS):
        parser.error("threshold overrides replay a trace: they need --traces")
    return args


def _paths(patterns: list[str]) -> list[Path]:
    paths = sorted(Path(p) for pattern in patterns for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"nothing matches {patterns}")
    return paths


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        rows = []
        if args.payloads:
            paths = _paths(args.payloads)
            for path in paths:
                for episode in json.loads(path.read_text())["episodes"]:
                    rows.extend(episode["cycle_census"])
            thresholds = {}
        else:
            paths = _paths(args.traces)
            thresholds = {n: getattr(args, n) for n in THRESHOLDS if getattr(args, n) is not None}
            for path in paths:
                with gzip.open(path, "rt") as f:
                    rows.extend(hold.cycle_rows((json.loads(line) for line in f), **thresholds))
        summary = hold.agreement(rows)
        summary["files"] = len(paths)
        summary["thresholds"] = thresholds
        ok = hold.meets_bar(summary)
        summary["pass"] = ok
    except Exception as err:  # noqa: BLE001 -- every failure must become a token
        detail = f"{type(err).__name__}: {err}\n{traceback.format_exc()}"
        print("HOLD_ERROR " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1
    print(("HOLD_OK " if ok else "HOLD_FAIL ") + json.dumps(summary), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    os._exit(main())
