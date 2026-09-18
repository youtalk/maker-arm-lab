"""Sum the ``events.jsonl`` logs of ``mj_inspect_heap.py`` runs into one table.

Stdlib only. Prints one ``INSPECT_READ {json}`` line and a table per episode, then the
totals the inspection design asks for: the ejection routes by phase bucket, the give-up
state, and the per-family cycle outcomes.

    python3 scripts/heap_inspect_read.py ~/data/maker-arm-ms6/b/inspect/*/events.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter

GRASP_PHASES = {"DESCEND", "CLOSE", "LIFT", "REOPEN"}
TRANSIT_PHASES = {"RAISE", "EXTEND", "RELEASE", "RETREAT"}


def exit_bucket(event: dict) -> str:
    if event["step"] == 0:
        return "settle"
    if event.get("in_push"):
        return "push"
    if event.get("target") is not None and event["part"] == event["target"]:
        return "carried"  # the target leaving the reach band in the jaws, not an ejection
    if event["phase"] in GRASP_PHASES:
        return "grasp"
    if event["phase"] in TRANSIT_PHASES:
        return "transit"
    return "other"


def read_episode(path: str) -> dict:
    with open(path) as f:
        events = [json.loads(line) for line in f]
    end = next(e for e in events if e["kind"] == "end")
    scans = [e for e in events if e["kind"] == "scan"]
    exits = Counter(exit_bucket(e) for e in events if e["kind"] == "exit")
    lifts = [e for e in events if e["kind"] == "phase" and e["phase"] == "LIFT"]
    outcomes = Counter()
    for e in lifts:
        # R1 (rung, MS6 rework section 4) moved the LIFT transition to `lifted` alone. The old
        # `pinch_ok and lifted` gate under-counts real lifts in any read taken after that rung.
        # The refused-pinch information is not lost: it is the "rescued" count below.
        ok = bool(e["lifted"])
        outcomes[(e["family"], "ok" if ok else "fail")] += 1
    rescued = sum(1 for e in lifts if e.get("lifted") and e.get("pinch_ok") is False)
    stats = end["extra_stats"]
    stowed_by_family = Counter()
    families = {}
    for s in scans:
        for p in s["parts"]:
            families[p["id"]] = p["family"]
    # The cycle log tells which parts stowed: a True outcome belongs to that cycle's part.
    for part, ok in zip(stats.get("cycle_parts", []), stats.get("cycle_outcomes", [])):
        if ok:
            stowed_by_family[families.get(part, "?")] += 1
    first = scans[0]["parts"] if scans else []
    last = scans[-1] if scans else None
    return {
        "name": os.path.basename(os.path.dirname(os.path.abspath(path))),
        "parts": end["parts"],
        "stowed": end["stowed"],
        "stowed_at_rest": end.get("stowed_at_rest", end["stowed"]),
        "cycles": stats.get("cycles", 0),
        "pushes": stats.get("pushes", 0),
        "forced_picks": stats.get("forced_picks", 0),
        "reason": end["reason"],
        "done_step": end["done_step"],
        "idle_steps": end["idle_steps"],
        "outside_at_start": sum(1 for p in first if not p["reach"]),
        "outside_last": stats.get("parts_out_of_reach", 0),
        "tips": stats.get("tips", 0),
        "tips_ok": stats.get("tips_ok", 0),
        "rescued": rescued,
        "exits": dict(exits),
        "last_scan": None
        if last is None
        else {
            "candidates": last["candidates"],
            "with_axes": sum(1 for p in last["parts"] if p["axes"]),
            "best_clearance": max(
                (p["best_clearance"] for p in last["parts"] if p["best_clearance"] is not None),
                default=None,
            ),
        },
        "lift_outcomes": {f"{fam}_{k}": v for (fam, k), v in outcomes.items()},
        "stowed_by_family": dict(stowed_by_family),
    }


def summarize(episodes: list[dict]) -> dict:
    exits = Counter()
    lifts = Counter()
    stowed = Counter()
    for e in episodes:
        exits.update(e["exits"])
        lifts.update(e["lift_outcomes"])
        stowed.update(e["stowed_by_family"])
    parts = sum(e["parts"] for e in episodes)
    done = sum(e["stowed"] for e in episodes)
    at_rest = sum(e["stowed_at_rest"] for e in episodes)
    return {
        "episodes": len(episodes),
        "parts": parts,
        "stowed": done,
        "stowed_fraction": done / parts if parts else 0.0,
        "stowed_at_rest": at_rest,
        "stowed_at_rest_fraction": at_rest / parts if parts else 0.0,
        "cycles_per_stowed": (sum(e["cycles"] for e in episodes) / done) if done else None,
        "pushes": sum(e["pushes"] for e in episodes),
        "forced_picks": sum(e["forced_picks"] for e in episodes),
        "outside_at_start_mean": sum(e["outside_at_start"] for e in episodes) / len(episodes),
        "outside_last": sum(e["outside_last"] for e in episodes),
        "tips": sum(e["tips"] for e in episodes),
        "tips_ok": sum(e["tips_ok"] for e in episodes),
        "rescued": sum(e["rescued"] for e in episodes),
        "exits": dict(exits),
        "lift_outcomes": dict(lifts),
        "stowed_by_family": dict(stowed),
        "idle_steps_median": sorted(e["idle_steps"] for e in episodes)[len(episodes) // 2],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="events.jsonl files")
    args = parser.parse_args(argv)
    episodes = [read_episode(p) for p in args.logs]
    print(
        f"{'episode':28s} {'stowed':>6s} {'cycles':>6s} {'pushes':>6s} {'forced':>6s} "
        f"{'out0':>4s} {'outL':>4s} {'tips':>4s} {'tipsOk':>6s} {'resc':>4s} "
        f"{'reason':>8s} {'idle':>6s}  exits"
    )
    for e in episodes:
        print(
            f"{e['name']:28s} {e['stowed']:3d}/{e['parts']:<3d} {e['cycles']:6d} "
            f"{e['pushes']:6d} {e['forced_picks']:6d} {e['outside_at_start']:4d} "
            f"{e['outside_last']:4d} {e['tips']:4d} "
            f"{e['tips_ok']:6d} {e['rescued']:4d} "
            f"{e['reason']:>8s} {e['idle_steps']:6d}  {e['exits']}"
        )
    summary = summarize(episodes)
    print("INSPECT_READ " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
