"""Gate-evaluation supervisor: one policy server + N eval processes + aggregation.

Stdlib-only, like generate_dataset.py, so it runs under the system Python while its
subprocesses run in their own venvs: lerobot_stack/policy_server.py under
lerobot_stack/.venv, scripts/eval_policy.py under isaaclab.sh -p (VIRTUAL_ENV bootstrapped
the way generate_dataset.py's _isaaclab_batch_env does it). Ported from openyam-lab's
run_gate_eval.py and cut down (MS2 spec section 5.2): no held-out permutation logic (the
MS2 campaign's schedule covers all 24 bin-color permutations), no replay injection, no
unstick plumbing, no parallel lanes.

Draws: --processes run-level draws, each a uniformly random bin-color permutation and a
(run_seed, layout_seed) pair derived from --seed with the campaign's exact formula
((seed * 1000 + i) % 2**32, + 500 for the layout), asserted disjoint from every seed in the
campaign ledger (--ledger). The MS2 seed sets are fixed in the plan: corpus 20260907
(ledger), selection 20260917, official 20260927; ten apart because the formula spreads a
campaign of R runs over seed*1000 + [0, R) and + [500, 500 + R).

Each process's EVAL_OK payload is written to <out_dir>/eval_NNN.json and REUSED on a later
invocation with the same out_dir and checkpoint, so an interrupted ladder rung or gate
resumes where it stopped instead of re-rolling finished draws (a per-unit completion file,
the harness-task-sweeps lesson). Delete the file to re-roll a draw.

Prints GATE_OK or GATE_FAIL with the aggregate against --threshold; gate_results.json
carries everything (per-brick rate with its Wilson 95% interval, the reported-not-gated
secondaries, the draws, the served inference config).
"""

import argparse
import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

ALL_PERMUTATIONS = sorted(itertools.permutations(range(4)))
_SEED_BOUND = 2**32


class EvalProcessError(RuntimeError):
    """One eval process failed (bad exit, no EVAL_OK token, or timeout)."""


class GateRunError(RuntimeError):
    """A process exhausted its retries; the gate run is invalid."""


@dataclass
class EvalDraw:
    process_index: int
    permutation: tuple
    run_seed: int
    layout_seed: int


@dataclass
class GateCfg:
    checkpoint: str
    out_dir: Path
    processes: int
    num_envs: int
    seed: int
    threshold: float
    ledger: Path
    isaaclab_path: Path = field(default_factory=lambda: Path.home() / "src/IsaacLab")
    port: int = 5555
    device: str = "cuda"
    retries: int = 2
    n_action_steps: int | None = None
    temporal_ensemble_coeff: float | None = None
    only_processes: tuple | None = None
    eval_timeout_s: float = 4 * 3600
    backend: str = "isaac"  # "mujoco" spawns scripts/mj_eval_policy.py under .venv/bin/python
    episode_draws: bool = False  # MuJoCo only
    frame_jitter: bool = False  # MuJoCo only
    spatial_scale: float = 1.0  # MuJoCo only: the generalization ladder of the MS4 roadmap
    context: str = "none"  # MuJoCo only: the context channel served to the policy
    # MS6 phase B (design 5.3): a heap gate spawns scripts/mj_eval_heap.py.
    heap_k: int = 0  # 0 is the six-brick task
    parts_seed: int = 20260915
    parts_split: str = "heldout"
    cycle_budget: int | None = None  # None: the client's default, ceil(1.3 * K)
    max_cycles_per_stowed: float | None = None  # the second gate rule, None: not applied
    # MS7f design section 4.2: the throughput bar and the held-out row's bar, None: not applied.
    min_stows_per_hour: float | None = None
    heldout_threshold: float | None = None
    gripper: str = "binary"  # MuJoCo only: the checkpoint's handshake (MS6 phase B design 5.4)
    desk_boxes: int = 0  # MuJoCo heap only: MS7a desk boxes; 0 is the MS6 shelf bin
    tray: bool = False  # MuJoCo heap only: the heap drops into the tray (MS7a)
    tray_lean: float = 0.0  # MuJoCo heap only: the tray walls lean outward, degrees
    deep_box_floor: bool = False  # MuJoCo heap only: MS7e flat lever 1, deeper box floors
    # MuJoCo only: extra flags handed to the eval client verbatim, e.g. the heap loop's
    # protocol switches. A cached payload must have run with the same switches.
    client_flags: tuple = ()
    # MuJoCo heap only: the MS7e switches (MS7e design section 10.1), forwarded to the heap
    # client and recorded in its payload's "ms7e" block; every one off is MS7d.
    families: tuple = ()
    box_draw: int = 0
    release_lean: bool = False
    carry_via: bool = False
    box_unseen: str | None = None
    heap_family: str | None = None
    passenger_check: bool = False
    wall_band_cont: bool = False
    thin_snap: bool = False
    snap_closed: bool = False
    # MS7e unseen-box ladder, the seen row (design section 8.3): (RUN0, LAYOUT0), draw i reads a
    # recorded run's own seeds RUN0 + i and LAYOUT0 + i (make_eval_draws); None is the formula.
    seen_seeds: tuple | None = None
    # MuJoCo heap only: the MS7f switches (MS7f design section 12.1) at their values, forwarded
    # to the heap client and recorded in its payload's "ms7f" block; MS7F_OFF is MS7e.
    ms7f: dict = field(default_factory=dict)


FAMILY_KEYS = ("round", "ldraw", "flat")
"""``core.parts.FAMILY_KEYS``'s keys, in order, copied because the supervisor imports no
maker_arm_lab; the test pins it."""
READY_FAMILIES: frozenset[str] = frozenset({"round", "ldraw", "flat"})
"""``core.parts.READY_FAMILIES``, copied and pinned the same way: ``--families`` with another
key is refused before a server starts (MS7e design section 10.2)."""
UNBUILT_SWITCHES: tuple[str, ...] = ()
"""``core.batch_loop.UNBUILT_SWITCHES``, copied and pinned the same way."""


def families_key(text: str) -> tuple:
    """``--families``: the canonical tuple the heap client records, the keys of ``FAMILY_KEYS``
    in its order, once each (``core.batch_loop.families_arg``). Raises ValueError on an
    unknown key."""
    keys = [k.strip() for k in text.split(",") if k.strip()]
    unknown = [k for k in keys if k not in FAMILY_KEYS]
    if unknown:
        raise ValueError(f"--families: unknown family keys {unknown}; keys: {FAMILY_KEYS}")
    return tuple(k for k in FAMILY_KEYS if k in keys)


MS7E_OFF = {
    "families": [],
    "box_draw": 0,
    "release_lean": False,
    "carry_via": False,
    "box_unseen": None,
    "heap_family": None,
    "passenger_check": False,
    "wall_band_cont": False,
    "thin_snap": False,
    "snap_closed": False,
}
"""The "ms7e" block of a read with every MS7e switch off, what an MS7d payload ran with."""


def ms7e_block(cfg: "GateCfg") -> dict:
    """The payload's "ms7e" block a read with ``cfg`` records (``core.batch_loop.ms7e_switches``);
    the cache matches on it."""
    return {
        "families": list(cfg.families),
        "box_draw": cfg.box_draw,
        "release_lean": cfg.release_lean,
        "carry_via": cfg.carry_via,
        "box_unseen": cfg.box_unseen,
        "heap_family": cfg.heap_family,
        "passenger_check": cfg.passenger_check,
        "wall_band_cont": cfg.wall_band_cont,
        "thin_snap": cfg.thin_snap,
        "snap_closed": cfg.snap_closed,
    }


MS7F_OFF = {"categories": False}
"""``core.batch_loop.MS7F_OFF``, copied because the supervisor imports no maker_arm_lab; the test
pins it. The "ms7f" block of a read with every MS7f switch off, what an MS7e payload ran with."""
MS7F_READY: frozenset[str] = frozenset({"categories"})
"""``core.batch_loop.MS7F_READY``, copied and pinned the same way: any other MS7f switch at a
value but its off value is refused before a server starts (MS7f design section 12.2)."""


def ms7f_block(cfg: "GateCfg") -> dict:
    """The payload's "ms7f" block a read with ``cfg`` records (``core.batch_loop.ms7f_switches``);
    the cache matches on it."""
    return MS7F_OFF | cfg.ms7f


def ms7f_flags(block: dict) -> list[str]:
    """The heap client's flags for an "ms7f" block: a switch at its off value sends nothing, an
    on switch its flag, and a valued switch its flag and value."""
    flags = []
    for name, value in block.items():
        if value == MS7F_OFF[name]:
            continue
        flags += [f"--{name}"] if value is True else [f"--{name}", str(value)]
    return flags


MS7D_LEVERS = {
    "box_filter_xy": "L1",
    "release_gap": "L2",
    "rearm_hold": "L8",
    "box_walls": "L9",
    "marker_bias": "L3",
    "wrist_fix": "L4",
    "class_views": "L5",
    "hold_force": "L6",
    "force_cap": "L7",
}
"""The MS7d switches and their levers (MS7d design section 4), each forwarded to the MuJoCo
heap client as its own flag. All but ``hold_force`` and ``force_cap`` need ``--perceived``."""
LOOP_SWITCHES = (
    "select_at_release",
    "replan_on_switch",
    "row_refresh",
    "squeeze_floor",
    "close_gate",
    "square_wrist",
    "scripted_place",
    "scripted_rake",
    "part_failures",
    "pick_snap",
    "dwell_close",
    "adopt_held",
    "rake_crowd",
    "park_done",
    "rake_turn",
    "perceived",
    "contact_hold",
    *MS7D_LEVERS,
)
"""The heap loop's protocol switches (core.loop.rollout); the client records them in the
payload's inference block under "loop", and the cache matches on them."""
LOOP_SETTINGS = ("rake_cap",)
"""The heap loop's integer settings, recorded and matched like the switches; 0 is off."""
PERCEPTION_KNOBS = (
    "pos",
    "heading",
    "size",
    "box",
    "miss",
    "class",
    "tool",
    "wrist",
    "wrist_bias",
    "wrist_miss",
    "rho",
    "torque",
    "torque_k",
)
"""``core.perception.KNOBS``, copied because the supervisor imports no torch; the test pins it."""
PERCEPTION_EXTRA_KNOBS = ("pos_z",)
"""``core.perception.EXTRA_KNOBS``: outside ``all``, written only when not 1."""


def noise_scales_key(text: str) -> str:
    """The canonical ``--noise_scales`` text the heap client records under ``--perceived``:
    ``core.perception.format_scales(parse_scales(text))``, a stdlib copy that the test pins
    against those two. Raises ValueError on an unknown knob or a scale that is not a finite
    number >= 0."""
    scales = dict.fromkeys(PERCEPTION_KNOBS + PERCEPTION_EXTRA_KNOBS, 1.0)
    for item in text.split(","):
        if not item.strip():
            continue
        name, _, value = (part.strip() for part in item.partition("="))
        if name != "all" and name not in scales:
            raise ValueError(f"--noise_scales: unknown perception knob {name!r}")
        try:
            scale = float(value)
        except ValueError:
            scale = math.nan
        if not (math.isfinite(scale) and scale >= 0.0):
            raise ValueError(f"--noise_scales: {name} needs a finite number >= 0, got {value!r}")
        for knob in PERCEPTION_KNOBS if name == "all" else (name,):
            scales[knob] = scale
    extra = tuple(k for k in PERCEPTION_EXTRA_KNOBS if scales[k] != 1.0)
    names = PERCEPTION_KNOBS + extra
    return ",".join(f"{k}={repr(float(scales[k])).removesuffix('.0')}" for k in names)


def loop_switches(client_flags) -> dict:
    """Which of ``LOOP_SWITCHES`` the client flags turn on, each of ``LOOP_SETTINGS``'s values
    (``--rake_cap 4`` reads 4), and the ``noise_scales`` the client records: the canonical
    scales under ``--perceived``, else empty."""
    flags = list(client_flags)
    found = {name: f"--{name}" in flags for name in LOOP_SWITCHES}
    for name in LOOP_SETTINGS:
        key = f"--{name}"
        found[name] = int(flags[flags.index(key) + 1]) if key in flags else 0
    text = flags[flags.index("--noise_scales") + 1] if "--noise_scales" in flags else ""
    found["noise_scales"] = noise_scales_key(text) if found["perceived"] else ""
    return found


def make_eval_draws(processes: int, seed: int, seen: tuple | None = None) -> list[EvalDraw]:
    """The draws of ``seed``, or with ``seen`` = (RUN0, LAYOUT0) draw i on the seeds RUN0 + i
    and LAYOUT0 + i of a recorded corpus run, with the recorder's default permutation."""
    rng = random.Random(f"gate:{seed}")
    run0, layout0 = seen or (seed * 1000, seed * 1000 + 500)
    return [
        EvalDraw(
            process_index=i,
            permutation=(0, 1, 2, 3) if seen else tuple(rng.choice(ALL_PERMUTATIONS)),
            run_seed=(run0 + i) % _SEED_BOUND,
            layout_seed=(layout0 + i) % _SEED_BOUND,
        )
        for i in range(processes)
    ]


def gate_draws(cfg: "GateCfg") -> list[EvalDraw]:
    """Every draw of the read, checked against the campaign ledger: a fresh read shares no seed
    with a recorded run, and a seen read (``cfg.seen_seeds``) reads only recorded runs' pairs."""
    runs = json.loads(cfg.ledger.read_text()).get("runs", [])
    draws = make_eval_draws(cfg.processes, cfg.seed, cfg.seen_seeds)
    if cfg.seen_seeds is None:
        assert_seeds_disjoint(draws, runs)
        return draws
    recorded = {(int(r["seeds"]["run_seed"]), int(r["seeds"]["layout_seed"])) for r in runs}
    for d in draws:
        if (d.run_seed, d.layout_seed) not in recorded:
            raise ValueError(f"--seen_seeds: draw {d.process_index} is no recorded run: {d}")
    return draws


def select_draws(draws: list[EvalDraw], only_processes: tuple | None) -> list[EvalDraw]:
    """Filter a FULL draw list; a subset re-run must revisit the same draws, so the list is
    always drawn in full and filtered afterwards."""
    if only_processes is None:
        return draws
    valid = {d.process_index for d in draws}
    unknown = sorted(set(only_processes) - valid)
    if unknown:
        raise ValueError(f"--only_processes {unknown} is outside the {len(draws)} drawn processes")
    wanted = set(only_processes)
    return [d for d in draws if d.process_index in wanted]


def ledger_seeds(ledger_runs: list[dict]) -> set[int]:
    recorded = set()
    for run in ledger_runs:
        recorded.add(int(run["seeds"]["run_seed"]))
        recorded.add(int(run["seeds"]["layout_seed"]))
    return recorded


def assert_seeds_disjoint(draws: list[EvalDraw], ledger_runs: list[dict]) -> None:
    recorded = ledger_seeds(ledger_runs)
    for d in draws:
        if d.run_seed in recorded or d.layout_seed in recorded:
            raise ValueError(f"eval draw {d.process_index} collides with a campaign seed: {d}")


def assert_seed_sets_disjoint(*seed_sets: tuple[str, list[EvalDraw]]) -> None:
    """Selection and official draws must not share a seed with each other either."""
    seen: dict[int, str] = {}
    for name, draws in seed_sets:
        for d in draws:
            for s in (d.run_seed, d.layout_seed):
                if s in seen and seen[s] != name:
                    raise ValueError(f"seed {s} appears in both {seen[s]} and {name}")
                seen[s] = name


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


CONTROL_HZ = 30.0
"""``core.loop.CONTROL_HZ``, copied because the supervisor imports no maker_arm_lab; pinned."""
INTERVAL_NAMES = ("select_to_close", "close_to_lift", "lift_to_release", "release_to_next")
"""``core.loop.INTERVAL_NAMES``, the S0 intervals (MS7f design section 8.1); pinned."""


def aggregate_gate(
    process_payloads: list[dict],
    threshold: float,
    max_cycles_per_stowed: float | None = None,
    min_stows_per_hour: float | None = None,
    heldout_threshold: float | None = None,
) -> dict:
    """Per-brick success rate = bricks delivered to their correct bin / (num_bricks x
    episodes), as RolloutTracker grades it, gated against `threshold`. Everything else is
    reported, not gated. The Wilson interval is over bricks and therefore optimistic (the
    six bricks of one episode are not independent); the report reads it as such.

    `max_cycles_per_stowed` (MS6 phase B design 5.3) is the heap gate's second rule: pick
    cycles per stowed part at most this. None (the six-brick task) reports cycles but does
    not gate on them.

    MS7f design section 4.2 adds two bars, each gated only when set: ``min_stows_per_hour``, the
    correct stows per simulated hour over every step of every episode (3600 x stowed / (steps /
    30)), and ``heldout_threshold``, the stowed share of the held-out row (each episode's
    ``stowed_heldout``: parts of kinds neither the policy nor the classifier saw), which fails
    when the read holds no held-out part. With both unset the verdict is MS7e's."""
    episodes = [ep for p in process_payloads for ep in p["episodes"]]
    if max_cycles_per_stowed is not None and any("cycles" not in ep for ep in episodes):
        # A six-brick payload has no cycles; counting it as 0 would pass the rule for free.
        raise ValueError("the cycles rule needs every episode's cycles; a payload lacks them")
    bricks_total = sum(ep["num_bricks"] for ep in episodes)
    bricks_stowed = sum(ep["bricks_stowed"] for ep in episodes)
    rate = bricks_stowed / bricks_total if bricks_total else 0.0
    cycles_total = sum(ep.get("cycles", 0) for ep in episodes)
    cycles_per_stowed = (cycles_total / bricks_stowed) if bricks_stowed else None
    cycles_ok = max_cycles_per_stowed is None or (
        cycles_per_stowed is not None and cycles_per_stowed <= max_cycles_per_stowed
    )
    steps_total = sum(ep["steps"] for ep in episodes)
    per_hour = 3600.0 * bricks_stowed * CONTROL_HZ / steps_total if steps_total else 0.0
    hour_ok = min_stows_per_hour is None or per_hour >= min_stows_per_hour
    held = {
        k: sum(ep.get("stowed_heldout", {}).get(k, 0) for ep in episodes)
        for k in ("stowed", "total")
    }
    held_rate = held["stowed"] / held["total"] if held["total"] else None
    held_ok = heldout_threshold is None or (
        held_rate is not None and held_rate >= heldout_threshold
    )
    grasp_attempts = sum(ep.get("grasp_attempts", 0) for ep in episodes)
    grasp_successes = sum(ep.get("grasp_successes", 0) for ep in episodes)
    grasp_successes_strict = sum(ep.get("grasp_successes_strict", 0) for ep in episodes)
    releases_queued = sum(p.get("releases_queued", 0) for p in process_payloads)
    wrong_bin = sum(ep.get("wrong_bin_deposits", 0) for ep in episodes)
    clamp_steps = sum(ep["clamp_steps"] for ep in episodes)
    # The close-event instrument: pairs (nearest brick at the close, held within the window),
    # paired by index so a window still open at the episode's end pairs with nothing.
    closes = [
        (cm, ok)
        for ep in episodes
        for cm, ok in zip(ep.get("closes_nearest_cm", []), ep.get("closes_held_window", []))
    ]
    near = [(cm, ok) for cm, ok in closes if 0.0 <= cm <= 2.0]
    confusion: dict[str, int] = {}
    for ep in episodes:
        for color, landed in ep.get("releases", []):
            if landed is not None and landed != color:
                key = f"{color}->{landed}"
                confusion[key] = confusion.get(key, 0) + 1
    lo, hi = wilson_interval(bricks_stowed, bricks_total)
    # MS7a desk boxes: bricks_stowed counts parts at rest in their own box only (the grader),
    # and these rows say where the rest went. Absent on every payload without desk boxes.
    # MS7e design section 8.1 adds the rows per family and per box count; an MS7d payload
    # has no family rows and four boxes in every episode.
    boxed = [ep for ep in episodes if "stowed_by_class" in ep]
    box_rows = {}
    if boxed:
        by_family: dict[str, dict] = {}
        by_count: dict[str, dict] = {}
        for ep in boxed:
            count = {"stowed": ep["bricks_stowed"], "total": ep["num_bricks"]}
            for table, rows in (
                (by_family, ep.get("stowed_by_family", {})),
                (by_count, {str(ep.get("box_count", len(ep["stowed_by_box"]))): count}),
            ):
                for name, row in rows.items():
                    total = table.setdefault(name, {"stowed": 0, "total": 0})
                    for k in ("stowed", "total"):
                        total[k] += row[k]
        box_rows = {
            "stowed_by_class": {
                c: {
                    k: sum(ep["stowed_by_class"][c][k] for ep in boxed) for k in ("stowed", "total")
                }
                for c in boxed[0]["stowed_by_class"]
            },
            "stowed_by_family": by_family,
            "stowed_by_box_count": dict(sorted(by_count.items(), key=lambda kv: int(kv[0]))),
            # Under the box draw an episode lists only its own boxes, so the lists differ in length.
            "stowed_by_box": [
                {
                    k: sum(
                        ep["stowed_by_box"][m][k] for ep in boxed if m < len(ep["stowed_by_box"])
                    )
                    for k in ("stowed", "total")
                }
                for m in range(max(len(ep["stowed_by_box"]) for ep in boxed))
            ],
            "wrong_box_parts": sum(ep["wrong_box_parts"] for ep in boxed),
            "wrong_box_cycles": sum(ep["cycle_classes"].count("wrong_box") for ep in boxed),
        }
    return {
        "gate": {
            "threshold": threshold,
            "episodes": len(episodes),
            "bricks_total": bricks_total,
            "bricks_stowed": bricks_stowed,
            "per_brick_rate": rate,
            "wilson95": [lo, hi],
            "cycles_total": cycles_total,
            "cycles_per_stowed": cycles_per_stowed,
            "max_cycles_per_stowed": max_cycles_per_stowed,
            "steps_total": steps_total,
            "stows_per_hour": per_hour,
            "min_stows_per_hour": min_stows_per_hour,
            "stows_per_hour_passed": hour_ok,
            "heldout": {
                **held,
                "rate": held_rate,
                "wilson95": list(wilson_interval(held["stowed"], held["total"])),
            },
            "heldout_threshold": heldout_threshold,
            "heldout_passed": held_ok,
            "passed": rate >= threshold and cycles_ok and hour_ok and held_ok,
        },
        "secondary": {
            "all_six_episode_rate": (
                sum(1 for ep in episodes if ep["success"]) / len(episodes) if episodes else 0.0
            ),
            "episodes_with_any_brick": sum(1 for ep in episodes if ep["bricks_stowed"] > 0),
            "bricks_stowed_histogram": {
                str(k): sum(1 for ep in episodes if ep["bricks_stowed"] == k)
                for k in range((episodes[0]["num_bricks"] if episodes else 0) + 1)
            },
            "grasp_attempts_graded_total": grasp_attempts,
            "grasp_successes_total": grasp_successes,
            "grasp_success_rate": (grasp_successes / grasp_attempts) if grasp_attempts else None,
            "grasp_success_rate_strict": (
                (grasp_successes_strict / grasp_attempts) if grasp_attempts else None
            ),
            "grasp_attempts_unresolved_total": sum(
                ep.get("grasp_attempts_unresolved", 0) for ep in episodes
            ),
            "releases_queued_total": releases_queued,
            "wrong_bin_deposits_total": wrong_bin,
            "wrong_bin_rate": (wrong_bin / releases_queued) if releases_queued else None,
            "never_closed_episodes": sum(
                1 for ep in episodes if ep.get("first_close_step", -1) == -1
            ),
            "wrong_bin_confusion": dict(sorted(confusion.items())),
            "closes_graded_window": len(closes),
            "closes_within_2cm": len(near),
            "hold_rate_window": (sum(ok for _, ok in closes) / len(closes)) if closes else None,
            "hold_rate_window_within_2cm": (sum(ok for _, ok in near) / len(near))
            if near
            else None,
            "clamp_steps_total": clamp_steps,
            "clamp_fraction": (clamp_steps / steps_total) if steps_total else 0.0,
            "clamp_steps_per_joint": [
                sum(ep.get("clamp_steps_per_joint", [0] * 6)[j] for ep in episodes)
                for j in range(6)
            ],
            "mean_steps": (steps_total / len(episodes)) if episodes else None,
            **box_rows,
            **_ms7f_rows(episodes),
        },
    }


def _ms7f_rows(episodes: list[dict]) -> dict:
    """The MS7f rows of a read (interface item 4): the events, the rows per category under
    --categories and the S0 intervals summed per cycle outcome (``core.loop.interval_sums``, a
    stdlib copy). Absent on a payload that predates them."""
    rows: dict = {}
    events = [ep["events"] for ep in episodes if "events" in ep]
    if events:
        rows["events"] = {n: sum(e.get(n, 0) for e in events) for n in sorted(set().union(*events))}
    by_category: dict[str, dict] = {}
    for ep in episodes:
        for name, row in ep.get("stowed_by_category", {}).items():
            total = by_category.setdefault(name, {"stowed": 0, "total": 0})
            for k in ("stowed", "total"):
                total[k] += row[k]
    if by_category:
        rows["stowed_by_category"] = by_category
    timed = [ep for ep in episodes if "cycle_intervals" in ep]
    if timed:
        by: dict[str, dict] = {}
        for ep in timed:
            for outcome, four in zip(ep["cycle_classes"], ep["cycle_intervals"]):
                row = by.setdefault(outcome, {"cycles": 0, "steps": [0, 0, 0, 0]})
                row["cycles"] += 1
                row["steps"] = [a + b for a, b in zip(row["steps"], four)]
        rows["cycle_intervals"] = {
            "names": list(INTERVAL_NAMES),
            "by_outcome": dict(sorted(by.items())),
        }
    return rows


def _extract_json_line(text: str, token: str) -> dict:
    """Last `token {json}` line wins, matching generate_dataset.parse_batch_output."""
    for line in reversed(text.splitlines()):
        if line.startswith(token + " "):
            return json.loads(line[len(token) + 1 :])
    return {}


def _server_command(cfg: GateCfg) -> list[str]:
    cmd = [
        "lerobot_stack/.venv/bin/python",
        "lerobot_stack/policy_server.py",
        "--checkpoint",
        cfg.checkpoint,
        "--port",
        str(cfg.port),
        "--device",
        cfg.device,
    ]
    if cfg.n_action_steps is not None:
        cmd += ["--n_action_steps", str(cfg.n_action_steps)]
    if cfg.temporal_ensemble_coeff is not None:
        cmd += ["--temporal_ensemble_coeff", str(cfg.temporal_ensemble_coeff)]
    return cmd


class _ServerHandle:
    def __init__(self, proc: subprocess.Popen, ready: dict):
        self._proc = proc
        self.ready = ready

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def _start_server(cfg: GateCfg) -> _ServerHandle:
    """Launch policy_server.py and block until it prints SERVER_READY (300 s budget)."""
    proc = subprocess.Popen(_server_command(cfg), stdout=subprocess.PIPE, text=True)
    ready = threading.Event()
    lines: list[str] = []

    def _reader() -> None:
        for line in proc.stdout:
            lines.append(line)
            if line.startswith("SERVER_READY "):
                ready.set()
                break

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout=300)
    if not ready.is_set():
        proc.kill()
        proc.wait()
        raise EvalProcessError(
            f"policy_server.py did not print SERVER_READY within 300s:\n{''.join(lines)}"
        )
    return _ServerHandle(proc, _extract_json_line("".join(lines), "SERVER_READY"))


MUJOCO_PYTHON = ".venv/bin/python"


def _isaaclab_eval_env(cfg: GateCfg) -> dict[str, str]:
    if cfg.backend == "mujoco":
        return {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    env.setdefault("VIRTUAL_ENV", str(cfg.isaaclab_path / ".venv"))
    return env


def _eval_command(cfg: GateCfg, draw: EvalDraw) -> list[str]:
    if cfg.backend == "mujoco":
        script = "scripts/mj_eval_heap.py" if cfg.heap_k > 0 else "scripts/mj_eval_policy.py"
        prefix = [MUJOCO_PYTHON, script]
        if cfg.episode_draws:
            prefix.append("--episode_draws")
        if cfg.frame_jitter:
            prefix.append("--frame_jitter")
        if cfg.spatial_scale != 1.0:
            prefix += ["--spatial_scale", str(cfg.spatial_scale)]
        if cfg.context != "none":
            prefix += ["--context", cfg.context]
        if cfg.heap_k > 0:
            prefix += [
                "--heap_k",
                str(cfg.heap_k),
                "--parts_seed",
                str(cfg.parts_seed),
                "--parts_split",
                cfg.parts_split,
            ]
            if cfg.cycle_budget is not None:
                prefix += ["--cycle_budget", str(cfg.cycle_budget)]
            prefix += ["--gripper", cfg.gripper]
            if cfg.desk_boxes:
                prefix += ["--desk_boxes", str(cfg.desk_boxes)]
            if cfg.tray_lean:
                prefix += ["--tray_lean", str(cfg.tray_lean)]
            if cfg.tray:
                prefix.append("--tray")
            if cfg.deep_box_floor:
                prefix.append("--deep_box_floor")
            if cfg.families:
                prefix += ["--families", ",".join(cfg.families)]
            if cfg.box_draw:
                prefix += ["--box_draw", str(cfg.box_draw)]
            if cfg.release_lean:
                prefix.append("--release_lean")
            if cfg.carry_via:
                prefix.append("--carry_via")
            if cfg.box_unseen:
                prefix += ["--box_unseen", cfg.box_unseen]
            if cfg.heap_family:
                prefix += ["--heap_family", cfg.heap_family]
            if cfg.passenger_check:
                prefix.append("--passenger_check")
            if cfg.wall_band_cont:
                prefix.append("--wall_band_cont")
            if cfg.thin_snap:
                prefix.append("--thin_snap")
            if cfg.snap_closed:
                prefix.append("--snap_closed")
            prefix += ms7f_flags(ms7f_block(cfg))
        prefix += list(cfg.client_flags)
    else:
        prefix = [
            f"{cfg.isaaclab_path}/isaaclab.sh",
            "-p",
            "scripts/eval_policy.py",
            "--headless",
            "--enable_cameras",
        ]
    return prefix + [
        "--num_envs",
        str(cfg.num_envs),
        "--run_seed",
        str(draw.run_seed),
        "--layout_seed",
        str(draw.layout_seed),
        "--permutation",
        ",".join(str(p) for p in draw.permutation),
        "--server",
        f"tcp://127.0.0.1:{cfg.port}",
        "--out",
        str(cfg.out_dir / f"eval_{draw.process_index:03d}.raw.json"),
    ]


def _run_eval_subprocess(cfg: GateCfg, draw: EvalDraw) -> dict:
    cmd = _eval_command(cfg, draw)
    try:
        result = subprocess.run(
            cmd,
            env=_isaaclab_eval_env(cfg),
            capture_output=True,
            text=True,
            check=False,
            timeout=cfg.eval_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvalProcessError(
            f"eval process {draw.process_index} exceeded {cfg.eval_timeout_s}s"
        ) from exc
    (cfg.out_dir / f"eval_{draw.process_index:03d}.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise EvalProcessError(
            f"eval process {draw.process_index} exited {result.returncode}:\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    payload = _extract_json_line(result.stdout, "EVAL_OK")
    if not payload:
        raise EvalProcessError(
            f"eval process {draw.process_index} printed no EVAL_OK token:\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    return payload


def _cached_payload(cfg: GateCfg, draw: EvalDraw) -> dict | None:
    """A finished draw's payload from a previous invocation, if it is for this checkpoint
    and this draw; anything else is ignored rather than trusted."""
    path = cfg.out_dir / f"eval_{draw.process_index:03d}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    same_draw = (
        payload.get("run_seed") == draw.run_seed
        and payload.get("layout_seed") == draw.layout_seed
        and list(payload.get("permutation", [])) == list(draw.permutation)
    )
    same_policy = str(payload.get("policy_id", "")).startswith(f"act:{cfg.checkpoint}")
    # A six-brick payload carries no heap keys, so it never matches a heap gate or back.
    same_task = payload.get("heap_k", 0) == cfg.heap_k
    if cfg.heap_k > 0:
        budget = cfg.cycle_budget or math.ceil(1.3 * cfg.heap_k)  # the client's default
        same_task = same_task and (
            payload.get("parts_split") == cfg.parts_split
            and payload.get("parts_seed") == cfg.parts_seed
            and payload.get("cycle_budget") == budget
            and payload.get("gripper", "binary") == cfg.gripper
            and payload.get("desk_boxes", 0) == cfg.desk_boxes
            and payload.get("tray", False) == cfg.tray
            and float(payload.get("tray_lean", 0.0)) == cfg.tray_lean
            # A payload that predates the deep floor ran without it.
            and payload.get("deep_box_floor", False) == cfg.deep_box_floor
            # A payload that predates the MS7e switches, or a later one, ran with them off.
            and MS7E_OFF | payload.get("ms7e", {}) == ms7e_block(cfg)
            # The same for the MS7f switches (MS7f design section 12.2).
            and MS7F_OFF | payload.get("ms7f", {}) == ms7f_block(cfg)
        )
    # The served inference configuration is part of what a draw measured. An open-loop read
    # (the checkpoint's own n_action_steps) must never stand in for a temporal-ensemble read.
    served = payload.get("inference") or {}
    same_inference = served.get("temporal_ensemble_coeff") == cfg.temporal_ensemble_coeff and (
        cfg.n_action_steps is None or served.get("n_action_steps") == cfg.n_action_steps
    )
    # The loop's protocol switches are part of what a draw measured; a payload that predates
    # them ran with every switch off.
    recorded = served.get("loop") or {}
    same_inference = same_inference and loop_switches(cfg.client_flags) == {
        **{name: bool(recorded.get(name, False)) for name in LOOP_SWITCHES},
        **{name: int(recorded.get(name, 0)) for name in LOOP_SETTINGS},
        "noise_scales": str(recorded.get("noise_scales", "")),
    }
    return payload if same_draw and same_policy and same_task and same_inference else None


def run_gate(cfg: GateCfg, run_eval, start_server) -> dict:
    """Run every selected draw against one server; aggregate; write gate_results.json.

    `run_eval(draw) -> payload` and `start_server() -> handle with .stop() and .ready` are
    injected so the orchestration has CPU tests. Finished draws are reused from
    eval_NNN.json; the server is started only if at least one draw still has to run."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_draws(gate_draws(cfg), cfg.only_processes)
    payloads: dict[int, dict] = {}
    todo = []
    for draw in selected:
        cached = _cached_payload(cfg, draw)
        if cached is not None:
            payloads[draw.process_index] = cached
            print(
                f"[gate] process {draw.process_index}: reusing eval_{draw.process_index:03d}.json"
            )
        else:
            todo.append(draw)
    server_ready: dict = {}
    if todo:
        server = start_server()
        server_ready = dict(getattr(server, "ready", {}) or {})
        try:
            for draw in todo:
                payload = None
                for attempt in range(cfg.retries + 1):
                    try:
                        payload = run_eval(draw)
                        break
                    except EvalProcessError as err:
                        print(
                            f"[gate] process {draw.process_index} attempt {attempt + 1} failed: {err}",
                            flush=True,
                        )
                if payload is None:
                    raise GateRunError(
                        f"process {draw.process_index} failed {cfg.retries + 1} times"
                    )
                (cfg.out_dir / f"eval_{draw.process_index:03d}.json").write_text(
                    json.dumps(payload, indent=2)
                )
                payloads[draw.process_index] = payload
                partial = aggregate_gate([payloads[k] for k in sorted(payloads)], cfg.threshold)
                print(
                    f"[gate] process {draw.process_index}: per-brick so far "
                    f"{partial['gate']['bricks_stowed']}/{partial['gate']['bricks_total']} = "
                    f"{partial['gate']['per_brick_rate']:.3f} over {partial['gate']['episodes']} episodes",
                    flush=True,
                )
        finally:
            server.stop()
    ordered = [payloads[k] for k in sorted(payloads)]
    results = aggregate_gate(
        ordered,
        cfg.threshold,
        cfg.max_cycles_per_stowed,
        cfg.min_stows_per_hour,
        cfg.heldout_threshold,
    )
    results["checkpoint"] = cfg.checkpoint
    results["seed"] = cfg.seed
    results["ledger"] = str(cfg.ledger)
    results["num_envs"] = cfg.num_envs
    # What the server SERVED with, from the handshake each eval process recorded (identical
    # across processes by construction), falling back to the SERVER_READY line. Never the
    # CLI request.
    results["inference"] = next(
        (p["inference"] for p in ordered if p.get("inference")), server_ready.get("inference", {})
    )
    results["policy_id"] = next((p.get("policy_id") for p in ordered), None)
    results["ms7e"] = ms7e_block(cfg)
    results["ms7f"] = ms7f_block(cfg)
    results["draws"] = [
        {
            "process_index": d.process_index,
            "permutation": list(d.permutation),
            "run_seed": d.run_seed,
            "layout_seed": d.layout_seed,
        }
        for d in selected
    ]
    results["wall_sec_per_episode"] = (
        sum(p.get("wall_sec", 0.0) for p in ordered) / results["gate"]["episodes"]
        if results["gate"]["episodes"]
        else None
    )
    # MS7a design, the fallback: a pass through the scripted place is its own outcome.
    scripted = "--scripted_place" in cfg.client_flags
    results["gate"]["scripted_place"] = scripted
    results["gate"]["outcome"] = (
        ("PASS (scripted place)" if scripted else "PASS") if results["gate"]["passed"] else "FAIL"
    )
    (cfg.out_dir / "gate_results.json").write_text(json.dumps(results, indent=2))
    token = "GATE_OK" if results["gate"]["passed"] else "GATE_FAIL"
    print(
        f"{token} "
        + json.dumps({"checkpoint": cfg.checkpoint, "seed": cfg.seed, **results["gate"]}),
        flush=True,
    )
    return results


def _seed_pair(raw: str) -> tuple:
    try:
        run0, layout0 = (int(part) for part in raw.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not RUN0,LAYOUT0: {raw!r}") from err
    return run0, layout0


def _process_list(raw: str) -> tuple:
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not a comma-separated integer list: {raw!r}") from err
    if not values:
        raise argparse.ArgumentTypeError("selects no process")
    return values


class _TokenEmittingParser(argparse.ArgumentParser):
    """A parser whose usage errors still honour the token contract.

    argparse's own ``error()`` exits 2 with a usage message on stderr and no token, which a
    supervisor grepping the log cannot tell from a hang."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print("GATE_ERROR", flush=True)
        raise SystemExit(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="a pretrained_model directory")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="the campaign ledger.json")
    parser.add_argument("--threshold", type=float, required=True, help="per-brick pass rate")
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--isaaclab_path", type=Path, default=Path.home() / "src/IsaacLab")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=None)
    parser.add_argument("--only_processes", type=_process_list, default=None)
    parser.add_argument("--eval_timeout_s", type=float, default=4 * 3600)
    parser.add_argument("--backend", choices=("isaac", "mujoco"), default="isaac")
    parser.add_argument(
        "--episode_draws",
        action="store_true",
        help="MuJoCo only: bins, lights and cameras are redrawn per episode (MS4 lever 1)",
    )
    parser.add_argument(
        "--frame_jitter",
        action="store_true",
        help="MuJoCo only: small overhead-camera jitter resampled per rendered frame",
    )
    parser.add_argument(
        "--spatial_scale",
        type=float,
        default=1.0,
        help="MuJoCo only: multiplier on the camera-pose and bin_x jitter of the run draw "
        "and of every per-episode draw (1.0 = training; the generalization ladder of the "
        "MS4 roadmap)",
    )
    parser.add_argument(
        "--context",
        choices=("none", "camera", "privileged", "camera_box", "camera_box_target"),
        default="none",
        help="MuJoCo only: which context channel the env serves to the policy "
        "(observation.environment_state); must match the checkpoint",
    )
    parser.add_argument(
        "--heap_k", type=int, default=0, help="MuJoCo only: a heap of K parts (MS6)"
    )
    parser.add_argument("--parts_seed", type=int, default=20260915)
    parser.add_argument("--parts_split", choices=("train", "heldout", "all"), default="heldout")
    parser.add_argument("--cycle_budget", type=int, default=None)
    parser.add_argument(
        "--max_cycles_per_stowed",
        type=float,
        default=None,
        help="the second gate rule of MS6 phase B: pick cycles per stowed part at most this "
        "(default 1.3 when --heap_k > 0, else not applied)",
    )
    parser.add_argument(
        "--min_stows_per_hour",
        type=float,
        default=None,
        help="MS7f design section 4.2: the throughput bar, correct stows per simulated hour over "
        "every step of every episode; not applied by default",
    )
    parser.add_argument(
        "--heldout_threshold",
        type=float,
        default=None,
        help="MS7f design section 4.2: the held-out row's bar, the stowed share of parts of "
        "held-out kinds (meaningful under --parts_split all); not applied by default",
    )
    parser.add_argument(
        "--gripper",
        choices=("binary", "continuous"),
        default="binary",
        help="MuJoCo only: forwarded to the heap client; must match the checkpoint's handshake",
    )
    parser.add_argument(
        "--desk_boxes",
        type=int,
        default=0,
        help="MuJoCo heap only: N desk boxes replace the shelf bin (MS7a); forwarded to the "
        "heap client",
    )
    parser.add_argument(
        "--tray",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client, which drops the heap into the "
        "tray (MS7a)",
    )
    parser.add_argument(
        "--tray_lean",
        type=float,
        default=0.0,
        help="MuJoCo heap only: forwarded to the heap client, which leans the tray walls outward "
        "by this many degrees (0 is vertical)",
    )
    parser.add_argument(
        "--scripted_place",
        action="store_true",
        help="MuJoCo heap only: the MS7a fallback read, forwarded to the heap client as "
        "--scripted_place; a pass is reported as 'PASS (scripted place)'",
    )
    parser.add_argument(
        "--scripted_rake",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --scripted_rake, which runs the "
        "teacher's rake off a tray wall where the policy's loop finds no target",
    )
    parser.add_argument(
        "--part_failures",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --part_failures, which books a "
        "failed cycle against its part instead of its 5 cm cell",
    )
    parser.add_argument(
        "--pick_snap",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --pick_snap, which moves the "
        "commanded pick point onto the served row on the final approach",
    )
    parser.add_argument(
        "--dwell_close",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --dwell_close, which closes the "
        "jaws for a policy that waits at grasp height",
    )
    parser.add_argument(
        "--adopt_held",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --adopt_held, which serves the "
        "box of a lifted part that is not the target",
    )
    parser.add_argument(
        "--rake_cap",
        type=int,
        default=0,
        help="MuJoCo heap only: forwarded to the heap client as --rake_cap N, which lets the "
        "loop rake one part up to N times; 0 keeps the teacher's cap",
    )
    parser.add_argument(
        "--rake_crowd",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --rake_crowd, which lets a rake "
        "start land the tips into a neighbour when no start is clear",
    )
    parser.add_argument(
        "--park_done",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --park_done, which holds an env "
        "whose episode has ended at its last command while the rest of the batch runs",
    )
    parser.add_argument(
        "--rake_turn",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --rake_turn, which retries a "
        "rake the rig declined at the teacher's next jaw headings",
    )
    parser.add_argument(
        "--perceived",
        action="store_true",
        help="MuJoCo heap only: forwarded to the heap client as --perceived (MS7c), whose "
        "decisions read only the perceived state and the hold detector",
    )
    parser.add_argument(
        "--noise_scales",
        type=str,
        default="",
        help="MuJoCo heap only, with --perceived: the perception knob scales, e.g. 'heading=2', "
        "forwarded in their canonical form; empty is every knob at 1, the nominal",
    )
    parser.add_argument(
        "--contact_hold",
        action="store_true",
        help="MuJoCo heap only, with --perceived: forwarded as --contact_hold, the fallback "
        "whose serving rules read the contacts for their stop flag",
    )
    for name, lever in MS7D_LEVERS.items():
        parser.add_argument(
            f"--{name}",
            action="store_true",
            help=f"MuJoCo heap only: forwarded to the heap client as --{name}, MS7d {lever}",
        )
    parser.add_argument(
        "--families",
        type=str,
        default="",
        help="MuJoCo heap only, MS7e: forwarded to the heap client as --families, the part "
        f"families the catalog adds (a comma list of {', '.join(FAMILY_KEYS)})",
    )
    parser.add_argument(
        "--box_draw",
        type=int,
        default=0,
        help="MuJoCo heap only, MS7e: forwarded as --box_draw N, the per-episode box count "
        "uniform from 4 to N; 0 is off",
    )
    parser.add_argument(
        "--release_lean",
        action="store_true",
        help="MuJoCo heap only, MS7e B2: forwarded as --release_lean",
    )
    parser.add_argument(
        "--carry_via",
        action="store_true",
        help="MuJoCo heap only, MS7e R12 T1 with --release_lean: forwarded as --carry_via",
    )
    parser.add_argument(
        "--box_unseen",
        choices=("size", "rim"),
        default=None,
        help="MuJoCo heap only, MS7e: forwarded as --box_unseen, an unseen-box ladder row",
    )
    parser.add_argument(
        "--heap_family",
        choices=FAMILY_KEYS,
        default=None,
        help="MuJoCo heap only, MS7e: forwarded as --heap_family, a family heap read; the key "
        "must be in --families",
    )
    parser.add_argument(
        "--passenger_check",
        action="store_true",
        help="MuJoCo heap only, MS7e teacher lever T2: forwarded as --passenger_check",
    )
    parser.add_argument(
        "--wall_band_cont",
        action="store_true",
        help="MuJoCo heap only, MS7e serving lever S3 with --gripper continuous: forwarded as "
        "--wall_band_cont",
    )
    parser.add_argument(
        "--thin_snap",
        action="store_true",
        help="MuJoCo heap only, MS7e serving lever S2 with --pick_snap: forwarded as --thin_snap",
    )
    parser.add_argument(
        "--snap_closed",
        action="store_true",
        help="MuJoCo heap only, the MS7e flat study's rule with --pick_snap: forwarded as "
        "--snap_closed",
    )
    parser.add_argument(
        "--deep_box_floor",
        action="store_true",
        help="MuJoCo heap only, MS7e flat lever 1: forwarded as --deep_box_floor, the desk box "
        "floors' colliders reach under the desk top",
    )
    parser.add_argument(
        "--categories",
        action="store_true",
        help="MuJoCo heap only, MS7f: forwarded as --categories, the six categories over the "
        "drawn boxes; needs --box_draw",
    )
    parser.add_argument(
        "--seen_seeds",
        type=_seed_pair,
        default=None,
        help="MS7e unseen-box ladder, the seen row: RUN0,LAYOUT0, draw i reads the seeds RUN0 + i "
        "and LAYOUT0 + i of a recorded run of the ledger (refused otherwise) instead of --seed's. "
        "With held-out parts only env 0 draws the recorded boxes, so it needs --num_envs 1",
    )
    parser.add_argument(
        "--client_flags",
        type=str,
        default="",
        help="MuJoCo only: extra flags handed to the eval client verbatim, e.g. "
        "'--select_at_release --replan_on_switch' (the heap loop's protocol switches)",
    )
    return parser.parse_args(argv)


def _build_cfg(args: argparse.Namespace) -> GateCfg:
    if args.gripper != "binary" and (args.heap_k <= 0 or args.backend != "mujoco"):
        # Only the MuJoCo heap client drives a continuous gripper (phase B design 5.4).
        raise ValueError("--gripper continuous needs --heap_k > 0 and --backend mujoco")
    if args.scripted_place and (args.heap_k <= 0 or args.backend != "mujoco"):
        raise ValueError("--scripted_place needs --heap_k > 0 and --backend mujoco")
    if args.scripted_rake and (args.heap_k <= 0 or args.backend != "mujoco"):
        raise ValueError("--scripted_rake needs --heap_k > 0 and --backend mujoco")
    if args.part_failures and (args.heap_k <= 0 or args.backend != "mujoco"):
        raise ValueError("--part_failures needs --heap_k > 0 and --backend mujoco")
    if args.pick_snap and (args.heap_k <= 0 or args.backend != "mujoco"):
        raise ValueError("--pick_snap needs --heap_k > 0 and --backend mujoco")
    if args.rake_cap < 0:
        raise ValueError(f"--rake_cap must be 0 (off) or a positive cap, got {args.rake_cap}")
    for name in (
        "dwell_close",
        "adopt_held",
        "rake_cap",
        "rake_crowd",
        "park_done",
        "rake_turn",
        "perceived",
        *MS7D_LEVERS,
    ):
        if getattr(args, name) and (args.heap_k <= 0 or args.backend != "mujoco"):
            raise ValueError(f"--{name} needs --heap_k > 0 and --backend mujoco")
    perceived_only = [n for n in MS7D_LEVERS if n not in ("hold_force", "force_cap")]
    for name in ("noise_scales", "contact_hold", *perceived_only):
        if getattr(args, name) and not args.perceived:
            raise ValueError(f"--{name} needs --perceived")
    if args.perceived and (not args.desk_boxes or args.gripper != "continuous"):
        # The client's own usage error, before a server starts and every draw loses to it.
        raise ValueError("--perceived needs --desk_boxes and --gripper continuous")
    scales = noise_scales_key(args.noise_scales) if args.noise_scales else ""
    if args.desk_boxes and (args.heap_k <= 0 or args.backend != "mujoco" or args.desk_boxes < 0):
        # Only the MuJoCo heap client builds desk boxes (MS7a design, desk boxes).
        raise ValueError("--desk_boxes needs a positive count, --heap_k > 0 and --backend mujoco")
    if args.tray_lean and not args.tray:
        raise ValueError("--tray_lean leans the tray's walls: it needs --tray")
    if args.tray and args.heap_k <= 0:
        raise ValueError("--tray needs --heap_k > 0: the tray holds a heap")
    if args.deep_box_floor and args.desk_boxes <= 0:
        raise ValueError("--deep_box_floor deepens the desk box floors: it needs --desk_boxes")
    if args.passenger_check and args.desk_boxes <= 0:
        raise ValueError(
            "--passenger_check keeps passengers out of other boxes: it needs --desk_boxes"
        )
    if args.wall_band_cont and args.gripper != "continuous":
        raise ValueError(
            "--wall_band_cont is the continuous jaws' band: it needs --gripper continuous"
        )
    if args.thin_snap and not args.pick_snap:
        raise ValueError("--thin_snap moves pick_snap's floor: it needs --pick_snap")
    if args.snap_closed and not args.pick_snap:
        raise ValueError("--snap_closed ends pick_snap's landing hold: it needs --pick_snap")
    families = families_key(args.families)
    if any(getattr(args, name) for name in MS7E_OFF) and (
        args.heap_k <= 0 or args.backend != "mujoco"
    ):
        raise ValueError("the MS7e switches need --heap_k > 0 and --backend mujoco")
    unbuilt = [k for k in families if k not in READY_FAMILIES]
    if unbuilt:
        # The client's own usage error (MS7e design section 10.2), before a server starts.
        raise ValueError(f"--families {','.join(unbuilt)} is not built on this commit")
    for name in UNBUILT_SWITCHES:
        if getattr(args, name):
            raise ValueError(f"--{name} is not built on this commit")
    if args.heap_family and args.heap_family not in families:
        raise ValueError("--heap_family must name one of --families")
    if args.carry_via and not args.release_lean:
        raise ValueError("--carry_via bends the B2 carry: it needs --release_lean")
    ms7f = {name: getattr(args, name) for name in MS7F_OFF}
    for name, value in ms7f.items():
        if value == MS7F_OFF[name]:
            continue
        if args.heap_k <= 0 or args.backend != "mujoco":
            raise ValueError(f"--{name} needs --heap_k > 0 and --backend mujoco")
        if name not in MS7F_READY:
            # The client's own usage error (MS7f design section 12.2), before a server starts.
            raise ValueError(f"--{name} is not built on this commit")
    if args.categories and not args.box_draw:
        raise ValueError("--categories maps the categories onto the drawn boxes: needs --box_draw")
    if args.seen_seeds and args.num_envs != 1:
        # The env's one generator draws env 0's boxes before any heap; each later env's boxes
        # follow the heap before it, which held-out parts change.
        raise ValueError("--seen_seeds reproduces a recorded run's env 0: it needs --num_envs 1")
    if args.heap_k > 0:
        if args.backend != "mujoco":
            raise ValueError("--heap_k needs --backend mujoco: the Isaac env has no heap")
        if args.context == "privileged":
            # The heap client serves none, camera and camera_box only. Without this guard the
            # gate starts a server and then loses every draw to the client's own usage error.
            raise ValueError("--context privileged has no heap channel: use camera_box")
        if args.max_cycles_per_stowed is None:
            args.max_cycles_per_stowed = 1.3  # the heap gate's rule (phase B design 5.3)
    return GateCfg(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        processes=args.processes,
        num_envs=args.num_envs,
        seed=args.seed,
        threshold=args.threshold,
        ledger=args.ledger,
        isaaclab_path=args.isaaclab_path,
        port=args.port,
        device=args.device,
        retries=args.retries,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        only_processes=args.only_processes,
        eval_timeout_s=args.eval_timeout_s,
        backend=args.backend,
        episode_draws=args.episode_draws,
        frame_jitter=args.frame_jitter,
        spatial_scale=args.spatial_scale,
        context=args.context,
        heap_k=args.heap_k,
        parts_seed=args.parts_seed,
        parts_split=args.parts_split,
        cycle_budget=args.cycle_budget,
        max_cycles_per_stowed=args.max_cycles_per_stowed,
        min_stows_per_hour=args.min_stows_per_hour,
        heldout_threshold=args.heldout_threshold,
        gripper=args.gripper,
        desk_boxes=args.desk_boxes,
        tray=args.tray,
        tray_lean=args.tray_lean,
        deep_box_floor=args.deep_box_floor,
        families=families,
        box_draw=args.box_draw,
        release_lean=args.release_lean,
        carry_via=args.carry_via,
        box_unseen=args.box_unseen,
        heap_family=args.heap_family,
        passenger_check=args.passenger_check,
        wall_band_cont=args.wall_band_cont,
        thin_snap=args.thin_snap,
        snap_closed=args.snap_closed,
        seen_seeds=args.seen_seeds,
        ms7f=ms7f,
        client_flags=tuple(shlex.split(args.client_flags))
        + (("--scripted_place",) if args.scripted_place else ())
        + (("--scripted_rake",) if args.scripted_rake else ())
        + (("--part_failures",) if args.part_failures else ())
        + (("--pick_snap",) if args.pick_snap else ())
        + (("--dwell_close",) if args.dwell_close else ())
        + (("--adopt_held",) if args.adopt_held else ())
        + (("--rake_cap", str(args.rake_cap)) if args.rake_cap else ())
        + (("--rake_crowd",) if args.rake_crowd else ())
        + (("--park_done",) if args.park_done else ())
        + (("--rake_turn",) if args.rake_turn else ())
        + (("--perceived",) if args.perceived else ())
        + (("--noise_scales", scales) if scales else ())
        + (("--contact_hold",) if args.contact_hold else ())
        + tuple(f"--{name}" for name in MS7D_LEVERS if getattr(args, name)),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = _build_cfg(args)
        results = run_gate(
            cfg,
            run_eval=lambda draw: _run_eval_subprocess(cfg, draw),
            start_server=lambda: _start_server(cfg),
        )
    except (GateRunError, EvalProcessError, ValueError, FileNotFoundError) as exc:
        print(f"GATE_ERROR: {exc}", file=sys.stderr, flush=True)
        print("GATE_ERROR", flush=True)
        return 2
    return 0 if results["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
