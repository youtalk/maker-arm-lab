"""MS7j readiness suite: the rows and the tolerance table (MS7j design sections 6.1 and 6.4).

Stdlib only, like run_gate_eval.py, and it imports no maker_arm_lab.

  rows --candidate act|pi05 [--expected]
        One JSON line per row of the candidate (with --expected, only the A1 levels of MS7j
        design section 6.2), after a first line for the baseline, named "base". A line holds
        the row's fields and "argv": the supervisor flags of the row's read, which the caller
        completes with --checkpoint, --seed, --ledger, --processes and the host flags. Then
        SUITE_OK {"rows": N}.
  table --base DIR --row NAME=DIR ... --out FILE.md
        Pairs the eval_NNN.json payloads of each row with the baseline's, writes the table to
        FILE.md and prints SUITE_OK {json}. A row is refused when its draws are not the
        baseline's (a missing or extra file, another seed, other parts or another num_envs),
        when its policy is not the baseline's, or when its payloads ran other knobs than its
        name says: SUITE_FAIL {json}, exit 1, and no table. The tool never pairs a part of the
        draws.

BASE_ARGS is policy S's supervisor argv (``reads/i_S_d2039/fleet.json``), each flag once at the
value that argparse kept, without --checkpoint, --seed, --ledger, --processes, --only_processes,
--out_dir and --client_flags. BASE_CLIENT is its --client_flags. The changes from S:

- --heap_k 10 --num_envs 5 (S: 150 and 1), and BASE_CLIENT gains --sparse_gap 0.035.
- --desk_boxes 4 (S: 8), without --box_draw, --families, --categories, --classifier, --cls_flip
  and --vote_weights. The parts are the default catalog, so the classes are the four base
  classes.
- --no-frame_jitter, without --tray_refill and --box_empty. --episode_draws stays. The
  supervisor's --frame_jitter is a BooleanOptionalAction, so --no-frame_jitter clears the
  --frame_jitter that fleet.py's PROTOCOL puts first (ruling R27). The overhead camera knobs
  refuse frame jitter.
- --max_cycles_per_stowed 1000 (S kept 1000 too) and --threshold 0 (S: 0.85), without
  --min_stows_per_hour and --heldout_threshold. The suite is no gate.

A row name is group/axis/level. A negative level has its sign, and a positive level has none, so
every name is safe in a read tag. The client knobs use the --knob=value form, because a bare
negative value after a flag parses as another flag.

The tolerance rule (MS7j design section 6.4). For draw k, c_k is the baseline's correct
placements minus the row's, and p_k the baseline's parts. Then drop = 100 * sum(c) / sum(p) and
se = 100 * sqrt(n) * stdev(c) / sum(p), and the row passes when drop + se <= 5 points.
"""

import argparse
import json
import math
import os
import shlex
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run_gate_eval, imported in calls

BASE_ARGS: tuple[str, ...] = (
    *("--backend", "mujoco", "--num_envs", "5", "--heap_k", "10", "--parts_seed", "20260915"),
    *("--parts_split", "all", "--episode_draws", "--context", "camera_box_target"),
    *("--gripper", "continuous", "--eval_timeout_s", "72000", "--max_cycles_per_stowed", "1000"),
    "--no-frame_jitter",
    *("--threshold", "0", "--n_action_steps", "50", "--action_warp", "1.5"),
    *("--tray", "--desk_boxes", "4", "--tray_lean", "30", "--deep_tray", "--deep_box_floor"),
    *("--scripted_rake", "--part_failures", "--pick_snap", "--dwell_close", "--adopt_held"),
    *("--park_done", "--rake_turn", "--perceived", "--box_filter_xy", "--release_gap"),
    *("--rearm_hold", "--hold_force", "--force_cap", "--marker_bias", "--wrist_fix"),
    *("--class_views", "--box_walls", "--release_lean", "--carry_via", "--passenger_check"),
    *("--thin_snap", "--wall_band_cont", "--snap_closed", "--missed_lift"),
    *("--cell_failure_cap", "5", "--carry_guard", "--overhang_guard"),
)
BASE_CLIENT = (
    "--select_at_release --replan_on_switch --squeeze_floor --square_wrist --max_wall_minutes 900"
    " --sparse_gap 0.035"
)
LIFTED = frozenset({"stowed", "carrying", "lifted_lost", "wrong_box"})
"""The cycle classes whose ``last_held`` part left the tray floor."""
DRAW_KEYS = ("run_seed", "layout_seed", "permutation", "parts", "num_envs")
"""What a row's payload must share with the baseline's of the same name to pair."""
TOLERANCE = 5.0  # points of placement rate


@dataclass(frozen=True)
class Row:
    name: str
    group: str
    axis: str
    level: str
    client: str  # knobs added to BASE_CLIENT
    server: str  # supervisor flags: the inference delay
    act: bool
    pi05: bool
    expected: bool  # the A1 level of MS7j design section 6.2


def _six(index: int, value: str) -> str:
    return ",".join(value if i == index else "0" for i in range(6))


def _off(camera: str, index: int, warp: bool = False):
    return lambda v: f"--{camera}_offset={_six(index, v)}" + " --warp_back" * warp


SIGNED_CM = "0.01 -0.01 0.05 -0.05"
# group, axis, levels, the expected (A1) levels, the knob of a level
AXES = (
    ("obs_delay", "steps", "3 6", "3", lambda v: f"--obs_delay={v}"),
    ("infer_delay", "steps", "1 6 12", "1", lambda v: f"--infer_delay {v}"),
    ("overhead_pose", "tilt", "1 -1 3 -3 6 -6", "1 -1", _off("overhead", 0)),
    ("overhead_pose", "height", SIGNED_CM, "0.01 -0.01", _off("overhead", 5)),
    ("overhead_pose", "forward", SIGNED_CM, "0.01 -0.01", _off("overhead", 3)),
    ("overhead_pose", "lateral", SIGNED_CM, "0.01 -0.01", _off("overhead", 4)),
    ("wrist_pose", "pitch", "2 5", "2", _off("wrist", 0)),
    ("wrist_pose", "yaw", "2 5", "2", _off("wrist", 1)),
    ("wrist_pose", "forward", "0.002 0.01", "0.002", _off("wrist", 5)),
    ("wrist_pose", "lateral", "0.002 0.01", "0.002", _off("wrist", 3)),
    ("mount", "height", "0.04 -0.04", "0.04 -0.04", _off("overhead", 5, True)),
    ("mount", "forward", "0.02 -0.02 0.05 -0.05", "0.02 -0.02", _off("overhead", 3, True)),
    ("mount", "lateral", "0.02 -0.02 0.05 -0.05", "0.02 -0.02", _off("overhead", 4, True)),
    ("mount", "tilt", "3 -3 6 -6", "3 -3", _off("overhead", 0, True)),
    ("mount", "wrist", "0.01", "", _off("wrist", 3, True)),
    ("fov", "overhead", "2 -2 10 -10", "2 -2", lambda v: f"--overhead_fov_delta={v} --warp_back"),
    ("fov", "wrist", "2 -2 10 -10", "2 -2", lambda v: f"--wrist_fov_delta={v} --warp_back"),
    ("image", "jpeg", "85", "85", lambda v: f"--jpeg_quality={v}"),
    ("image", "blur", "1.0", "1.0", lambda v: f"--blur_sigma={v}"),
    ("image", "exposure", "1 -1", "1 -1", lambda v: f"--exposure_stops={v}"),
    ("light", "hard", "on", "on", lambda v: "--hard_light"),
)
"""MS7j design section 6.1. Overhead shifts are in the world (dx forward, dy lateral, dz up),
wrist shifts in the camera's optical frame (dx lateral, dz forward along the view). rx tilts the
view up and down, ry turns it left and right. ACT reads the inference delay 1, pi0.5 6 and 12."""


def _rows() -> tuple[Row, ...]:
    rows = []
    for group, axis, levels, expected, knob in AXES:
        server = group == "infer_delay"
        for v in levels.split():
            act = not server or v == "1"
            flags = knob(v)
            rows.append(
                Row(
                    f"{group}/{axis}/{v}",
                    group,
                    axis,
                    v,
                    "" if server else flags,
                    flags if server else "",
                    act,
                    not server or not act,
                    v in expected.split(),
                )
            )
    return tuple(rows)


ROWS = _rows()
BY_NAME = {r.name: r for r in ROWS}
BASE = Row("base", "base", "", "", "", "", True, True, False)


def argv(row: Row) -> list[str]:
    """The supervisor flags of the row's read, without the checkpoint, seed and host flags."""
    client = f"{BASE_CLIENT} {row.client}".strip()
    return [*BASE_ARGS, *row.server.split(), "--client_flags", client]


def measures(payload: dict) -> dict:
    """Part counts and steps over the payload's episodes. A part is lifted when it is stowed or
    is the ``last_held`` part of a cycle whose class is in ``LIFTED``, once however many cycles
    held it. ``never`` is the parts never lifted."""
    out = dict.fromkeys(("parts", "lifted", "correct", "wrong", "never", "steps"), 0)
    for ep in payload["episodes"]:
        held = {
            row["last_held"]
            for row, cls in zip(ep["cycle_census"], ep["cycle_classes"], strict=True)
            if cls in LIFTED and row["last_held"] >= 0
        }
        out["parts"] += ep["num_bricks"]
        out["lifted"] += len(held | set(ep["stowed_parts"]))
        out["correct"] += ep["bricks_stowed"]
        out["wrong"] += ep["wrong_box_parts"]
        out["steps"] += ep["steps"]
    out["never"] = out["parts"] - out["lifted"]
    return out


def paired(base: dict[str, dict], row: dict[str, dict]) -> dict:
    """The row's drop against the baseline, paired by draw (``eval_NNN.json`` name to payload).
    Raises ``ValueError`` unless both hold the same draws, two or more, of one policy. The row's
    policy id may end in its server's ``+delay<D>``."""
    if set(base) != set(row):
        raise ValueError(f"draws differ: baseline {sorted(base)}, row {sorted(row)}")
    if len(base) < 2:
        raise ValueError(f"a paired SE needs two draws or more, got {len(base)}")
    names = sorted(base)
    for name in names:
        for key in DRAW_KEYS:
            if base[name].get(key) != row[name].get(key):
                raise ValueError(f"{name}: the row's {key} is not the baseline's")
        delay = (row[name].get("inference") or {}).get("infer_delay", 0)
        policy = row[name].get("policy_id", "").removesuffix(f"+delay{delay}")
        if policy != base[name].get("policy_id", ""):  # +delay<D> is the last part of an id
            raise ValueError(f"{name}: the row's policy {policy} is not the baseline's")
    b = [measures(base[n]) for n in names]
    c = [x["correct"] - measures(row[n])["correct"] for x, n in zip(b, names)]
    parts = sum(x["parts"] for x in b)
    drop = 100.0 * sum(c) / parts
    se = 100.0 * math.sqrt(len(c)) * statistics.stdev(c) / parts
    return {
        "n": len(c),
        "drop": drop,
        "se": se,
        "passes": drop + se <= TOLERANCE,
        "better": sum(x < 0 for x in c),
        "same": sum(x == 0 for x in c),
        "worse": sum(x > 0 for x in c),
    }


def _size(level: str) -> float:
    return 1.0 if level == "on" else abs(float(level))


def largest_passing(results: dict[str, dict]) -> dict[str, str | None]:
    """Per ``group/axis`` of the rows in ``results``, the largest level that passes at both signs
    and at every smaller level of ``results``, or None. A level whose mirror row is missing does
    not pass."""
    out = {}
    for key in dict.fromkeys(f"{r.group}/{r.axis}" for r in ROWS if r.name in results):
        axis = [r for r in ROWS if f"{r.group}/{r.axis}" == key]
        out[key] = None
        for size in sorted({_size(r.level) for r in axis if r.name in results}):
            same = [r for r in axis if _size(r.level) == size]
            if not all(r.name in results and results[r.name]["passes"] for r in same):
                break
            out[key] = same[0].level
    return out


def _load(path: str, row: Row) -> dict[str, dict]:
    """A read's payloads, each checked to have run the row's knobs."""
    import run_gate_eval as ge

    want = (ge.stress_block(shlex.split(argv(row)[-1])), int(row.level) if row.server else 0)
    payloads = {}
    for p in sorted(Path(path).expanduser().glob("eval_[0-9][0-9][0-9].json")):
        payload = json.loads(p.read_text())
        got = (payload.get("stress", {}), (payload.get("inference") or {}).get("infer_delay", 0))
        if got != want:
            raise ValueError(f"{row.name}: {p} ran the knobs {got}, the row needs {want}")
        payloads[p.name] = payload
    return payloads


def _totals(payloads: dict[str, dict]) -> dict:
    import run_gate_eval as ge

    total = dict.fromkeys(("parts", "lifted", "correct", "wrong", "never", "steps"), 0)
    for payload in payloads.values():
        for key, value in measures(payload).items():
            total[key] += value
    seconds = total["steps"] / ge.CONTROL_HZ
    total["s_per_correct"] = seconds / total["correct"] if total["correct"] else None
    return total


def _line(name: str, expected: str, n: int, t: dict, p: dict | None = None) -> str:
    rule = ["", "", "", "", ""]
    if p:
        rule = [f"{p['drop']:.2f}", f"{p['se']:.2f}", f"{p['drop'] + p['se']:.2f}"]
        rule += ["pass" if p["passes"] else "FAIL", f"{p['better']}/{p['same']}/{p['worse']}"]
    s = "-" if t["s_per_correct"] is None else f"{t['s_per_correct']:.2f}"
    cells = [name, expected, str(n), f"{t['correct']}/{t['parts']}", *rule]
    cells += [str(t["lifted"]), str(t["wrong"]), str(t["never"]), s]
    return "| " + " | ".join(cells) + " |"


def table(base_dir: str, specs: list[str], out: str) -> dict:
    base = _load(base_dir, BASE)
    results, totals = {}, {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if name not in BY_NAME or not path:
            raise ValueError(f"--row {spec!r}: needs NAME=DIR with a row NAME")
        if name in results:
            raise ValueError(f"--row {name} given twice")
        payloads = _load(path, BY_NAME[name])
        results[name] = paired(base, payloads)
        totals[name] = _totals(payloads)
    largest = largest_passing(results)
    head = "| Row | A1 | n | Correct | Drop | SE | Drop + SE | Rule | Better/same/worse"
    lines = [
        "# Readiness suite: tolerance table",
        "",
        (
            f"Baseline: {base_dir}. A row passes when drop + SE <= {TOLERANCE:g} points (MS7j"
            " design section 6.4). Drop and SE are in points of correct placements per part."
            " Seconds are sim seconds per correct placement."
        ),
        "",
        head + " | Lifted | Wrong box | Never lifted | s per correct |",
        "|---" * 13 + "|",
        _line("base", "", len(base), _totals(base)),
    ]
    for name, p in results.items():
        lines.append(_line(name, "yes" if BY_NAME[name].expected else "", p["n"], totals[name], p))
    lines += ["", "## Largest passing level per axis", "", "| Axis | Level |", "|---|---|"]
    lines += [f"| {k} | {'none' if v is None else v} |" for k, v in largest.items()]
    Path(out).expanduser().write_text("\n".join(lines) + "\n")
    failed = [n for n, p in results.items() if BY_NAME[n].expected and not p["passes"]]
    return {"out": str(out), "draws": len(base), "largest": largest, "failed_expected": failed}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(f"usage: {message}")


def _parse(args: list[str]) -> argparse.Namespace:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rows = sub.add_parser("rows")
    rows.add_argument("--candidate", choices=("act", "pi05"), required=True)
    rows.add_argument("--expected", action="store_true", help="only the A1 levels")
    tab = sub.add_parser("table")
    tab.add_argument("--base", required=True, help="the baseline read's directory")
    tab.add_argument("--row", action="append", required=True, help="NAME=DIR, once per row")
    tab.add_argument("--out", required=True, help="the table, Markdown")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    code = 1
    try:
        a = _parse(sys.argv[1:] if args is None else args)
        if a.command == "rows":
            chosen = [r for r in ROWS if getattr(r, a.candidate) and (r.expected or not a.expected)]
            for row in (BASE, *chosen):
                print(json.dumps({**asdict(row), "argv": argv(row)}))
            token = "SUITE_OK " + json.dumps({"rows": len(chosen)})
        else:
            token = "SUITE_OK " + json.dumps(table(a.base, a.row, a.out))
        code = 0
    except BaseException as exc:  # noqa: BLE001 -- the token holds on --help and a crash too
        token = "SUITE_FAIL " + json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    print(token, flush=True)
    os._exit(code)


if __name__ == "__main__":
    main()
