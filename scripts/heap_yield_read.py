"""The MS6 phase A read: the heap teacher's yield over a set of ``*.meta.json`` sidecars.

Stdlib only, runs under system python3 like the other supervisors. Reads the per-episode
counters ``scripts/mj_generate_batch.py --heap_k`` writes, sums them, reports a row per part
family, and applies the pre-registered rule of the design (section 10): at least
``--min_stowed`` of the parts stowed and at most ``--max_cycles_per_stowed`` pick cycles per
stowed part. Prints exactly one of ``YIELD_OK {json}`` or ``YIELD_FAIL {json}`` on stdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path


def read_meta(paths: list[Path]) -> dict:
    """Sum the heap counters of every sidecar; a family row counts picks and stows per family."""
    summary = {
        "episodes": 0,
        "parts_total": 0,
        "parts_stowed": 0,
        "parts_outside_tray": 0,
        "cycles": 0,
        "pushes": 0,
        "runs": len(paths),
    }
    families: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "stowed": 0})
    for path in paths:
        meta = json.loads(Path(path).read_text())
        names = meta["run_meta"]["parts"]
        for episode in meta["episodes"]:
            summary["episodes"] += 1
            summary["parts_total"] += len(names)
            summary["parts_stowed"] += int(episode.get("bricks_delivered", 0))
            summary["parts_outside_tray"] += int(episode.get("parts_outside_tray", 0))
            summary["cycles"] += int(episode.get("cycles", 0))
            summary["pushes"] += int(episode.get("pushes", 0))
            outcomes = episode.get("cycle_outcomes", [])
            for k, part in enumerate(episode.get("cycle_parts", [])):
                family = names[part].split("_")[0]
                families[family]["attempts"] += 1
                if k < len(outcomes) and outcomes[k]:
                    families[family]["stowed"] += 1
    stowed = summary["parts_stowed"]
    summary["stowed_fraction"] = stowed / summary["parts_total"] if summary["parts_total"] else 0.0
    summary["cycles_per_stowed"] = summary["cycles"] / stowed if stowed else None
    summary["families"] = dict(sorted(families.items()))
    return summary


def verdict(summary: dict, min_stowed: float, max_cycles_per_stowed: float) -> bool:
    if summary.get("episodes", 0) == 0 or summary.get("cycles_per_stowed") is None:
        return False
    return (
        summary["stowed_fraction"] >= min_stowed
        and summary["cycles_per_stowed"] <= max_cycles_per_stowed
    )


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"YIELD_FAIL usage: {message}", flush=True)
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--meta", nargs="+", required=True, help="sidecar paths or globs")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here too")
    parser.add_argument("--min_stowed", type=float, default=0.90)
    parser.add_argument("--max_cycles_per_stowed", type=float, default=1.3)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = sorted(Path(p) for pattern in args.meta for p in glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no sidecar matches {args.meta}")
        summary = read_meta(paths)
        summary["rule"] = {
            "min_stowed": args.min_stowed,
            "max_cycles_per_stowed": args.max_cycles_per_stowed,
        }
        ok = verdict(summary, args.min_stowed, args.max_cycles_per_stowed)
        summary["pass"] = ok
        if args.out is not None:
            args.out.write_text(json.dumps(summary, indent=2))
    except Exception as exc:  # noqa: BLE001 -- every failure must become a token
        print("YIELD_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1
    print(("YIELD_OK " if ok else "YIELD_FAIL ") + json.dumps(summary), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
