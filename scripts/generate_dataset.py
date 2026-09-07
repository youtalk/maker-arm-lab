"""Overnight campaign supervisor for Maker Arm MS1 dataset generation.

Drives many generate_batch.py + convert_to_lerobot.py runs back to back until
a target number of successful episodes has been exported, writing a
crash-safe JSON ledger after every run so an interrupted campaign resumes
exactly where it left off.

This module is deliberately stdlib-only (argparse, dataclasses, itertools,
json, os, random, subprocess, sys, pathlib) so it can run under the system
python3 -- no isaaclab, torch, or lerobot virtualenv required. The
subprocesses it launches (generate_batch.py under isaaclab.sh,
convert_to_lerobot.py under lerobot_stack/.venv/bin/python) carry those
dependencies themselves. isaaclab.sh's own bootstrap (before it even reaches
generate_batch.py) needs VIRTUAL_ENV or CONDA_PREFIX set to find its bundled
python -- a bare non-interactive `python3 scripts/generate_dataset.py` (how
an unattended overnight campaign is launched, e.g. via nohup) doesn't have
either, so _isaaclab_batch_env() sets VIRTUAL_ENV itself (derived from
--isaaclab_path) unless the caller's environment already provides one.

Each campaign run is one subprocess and one ledger entry: the supervisor launches
generate_batch.py, parses its BATCH_OK token, and appends that run's record to
ledger.json before the next run starts, having marked that run started on disk
beforehand, tracked its phase through generation and conversion, and cleared the
marker only once the append succeeds. Nothing about a
campaign's progress is
inferred from the final CAMPAIGN_OK line, which a supervisor killed mid-campaign
never prints -- on this host background jobs are liable to be swept, and a chain
that dies in the middle must not be indistinguishable from one that never started.
--resume rebuilds state from the ledger alone, so a campaign interrupted anywhere
loses at most the runs that were in flight. It regenerates a run interrupted
before its convert began, and refuses to start at all when one was interrupted
after it or when the ledger and the dataset disagree about the episode count. Every batch subprocess
carries a timeout derived from its own deadline, so a wedged child stops the
campaign instead of hanging it. Symmetrically, every exit path from
main() prints exactly one of CAMPAIGN_OK / CAMPAIGN_FAIL, so a tokenless log means
the process is still alive rather than that it failed quietly.

Run with:  python3 scripts/generate_dataset.py --out_dir <dir> --dataset_root <dir>
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


class BatchFailed(RuntimeError):
    """Raised when a subprocess's stdout lacks the expected contract token."""


class ResumeMismatch(RuntimeError):
    """Raised when the on-disk state of a campaign is not safe to resume from."""


# The wall-clock ceiling handed to every generate_batch.py run as --max_wall_minutes. Passed
# explicitly on every run rather than left to that script's own 90.0 default, which is under a
# 50-attempt run at 12 envs and would truncate the run the acceptance gate is read from.
#
# Sized from D6's measured throughput at --num_envs 12, cameras included. The worst measured
# wall_sec_per_episode with cameras was 138.3 s (D6 run 1; the four runs span 122.9-138.3 s,
# and the closest-matched pair reads 126.1 s with cameras against 126.8 s without, so at 12
# envs cameras cost about nothing -- D5's 1-env 3.9-vs-2.9 min/episode camera ratio does not
# reproduce here). A run overshoots --attempts by up to num_envs - 1, because the driver tests
# the attempt count once per control step and the last envs conclude together, so the worst
# case is 50 + 11 = 61 episodes: 61 x 138.3 s = 141 min. 180 leaves ~28 % headroom over that,
# and also covers the pessimistic cross-check in which D5's camera ratio did hold at 12 envs
# (61 x 122.9 s x 1.34 = 167 min). This is a backstop against a pathological run, not a
# throughput target: too tight silently truncates a good run, while too loose costs nothing
# that _batch_timeout_seconds below does not already bound.
BATCH_WALL_MINUTES = 180.0

# How far past its own deadline a generate_batch.py run may go before the supervisor stops
# waiting. The child's --max_wall_minutes bounds its episode loop only; Isaac Lab's Kit
# bootstrap (1-2 min) happens before that loop starts and the shard flush after it ends (D6
# wrote 1.61 GB for 7 demos, so a 50-attempt shard is several GB), and neither is inside the
# deadline the child enforces on itself.
BATCH_TIMEOUT_GRACE_MINUTES = 15.0


def permutation_schedule(seed: int) -> Iterator[tuple]:
    """Yield all 24 permutations of (0, 1, 2, 3), seeded-shuffled, without replacement
    within a cycle; reshuffle (with a derived seed) each time the pool empties.

    Infinite generator -- callers pull one permutation per campaign run via next().
    """
    cycle = 0
    while True:
        pool = list(itertools.permutations(range(4)))
        # Derive a fresh per-cycle seed from the top-level seed so cycles reshuffle
        # differently from each other while the whole schedule stays reproducible.
        # random.Random() only accepts None/int/float/str/bytes/bytearray, so the
        # (seed, cycle) pair is folded into a single string rather than passed as a
        # tuple.
        random.Random(f"{seed}:{cycle}").shuffle(pool)
        yield from pool
        cycle += 1


def _extract_json_line(text: str, token: str) -> dict:
    """Return the JSON payload following `token ` on its own line, or {} if absent."""
    for line in text.splitlines():
        if line.startswith(token + " "):
            return json.loads(line[len(token) + 1 :])
    return {}


def parse_batch_output(text: str) -> dict:
    """Parse generate_batch.py's stdout contract.

    Expects a `BATCH_OK exported=<N> attempts=<M>` token, plus optional
    `RUN_META {json}` / `RUN_STATS {json}` lines. Additive informational keys in
    those JSON lines are parsed tolerantly, never validated against a closed schema.
    Raises BatchFailed if the BATCH_OK token is missing (e.g. BATCH_FAIL, or a crash).
    """
    exported = attempts = None
    for line in text.splitlines():
        if line.startswith("BATCH_OK "):
            for tok in line.split():
                if tok.startswith("exported="):
                    exported = int(tok.split("=", 1)[1])
                elif tok.startswith("attempts="):
                    attempts = int(tok.split("=", 1)[1])
    if exported is None or attempts is None:
        raise BatchFailed(f"no BATCH_OK token found in batch output:\n{text}")
    return {
        "exported": exported,
        "attempts": attempts,
        "meta": _extract_json_line(text, "RUN_META"),
        "stats": _extract_json_line(text, "RUN_STATS"),
    }


def parse_convert_output(text: str) -> dict:
    """Parse convert_to_lerobot.py's stdout contract: `CONVERT_OK episodes=<n> total=<total>`."""
    for line in text.splitlines():
        if line.startswith("CONVERT_OK "):
            out = {}
            for tok in line.split()[1:]:
                key, _, val = tok.partition("=")
                out[key] = int(val)
            return out
    raise BatchFailed(f"no CONVERT_OK token found in convert output:\n{text}")


@dataclass
class CampaignCfg:
    out_dir: Path
    dataset_root: Path
    # MS1's campaign size, and the same number generate_batch.py extrapolates a --probe
    # run's wall clock to (its CAMPAIGN_TARGET_SUCCESSES). openyam-lab's 600 came across
    # with this file; test_campaign_target_default_matches_generate_batch_s_extrapolation_target
    # pins the two together by source text, since this module is stdlib-only and cannot
    # import generate_batch.py.
    target: int = 50
    attempts_per_run: int = 50
    num_envs: int = 12
    seed: int = 0
    isaaclab_path: Path = field(default_factory=lambda: Path("."))
    # Forwarded to generate_batch.py as --max_wall_minutes, and the basis for the supervisor's
    # own timeout over that subprocess (_batch_timeout_seconds).
    max_wall_minutes: float = BATCH_WALL_MINUTES
    # Demonstration-distribution width. Both default to None = generate_batch.py's
    # own defaults (noise_scale 1.0, start_pose_offset_rad 0.05). They are reachable
    # from the campaign because these two knobs are the only ones that widen the state
    # distribution the demonstrations cover; that mattered on openyam-lab's arm, whose
    # policy missed its gate on compounding error. Not re-measured on the Maker Arm --
    # MS1's own runs are what would say whether it needs them.
    noise_scale: float | None = None
    start_pose_offset_rad: float | None = None
    # Deliberate first-attempt grasp-failure injection (generate_batch.py's
    # oracle.FailureInduction). induce_prob 0.0 = off, identical to
    # pre-induction behaviour; the other four are only meaningful once it is
    # nonzero, so _batch_command forwards all five together, gated on it.
    induce_prob: float = 0.0
    induce_offset_weight: float = 0.7
    induce_offset_min: float = 0.015
    induce_offset_max: float = 0.025
    induce_noise_boost: float = 5.0
    # Split-at-source: forwarded to convert_to_lerobot.py. Default off = the
    # whole-attempt episode layout, byte-identical conversion path.
    split_cycles: bool = False
    sidecar_dir: Path | None = None


class Ledger:
    """JSON-file-backed record of every campaign run, crash-safe (written after each run)."""

    def __init__(self, path: Path):
        self.path = path
        self.runs: list[dict] = []

    @property
    def total_successes(self) -> int:
        return sum(r["exported"] for r in self.runs)

    @property
    def last_converted_total(self) -> int | None:
        """The dataset episode count the converter itself last reported, or None.

        None means the check in check_resume_consistency cannot run: either nothing has been
        converted yet, or the ledger predates this field. Runs that exported nothing are
        skipped -- they never called the converter, so they carry no total.
        """
        for record in reversed(self.runs):
            total = record.get("converted_total")
            if total is not None:
                return total
        return None

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        ledger = cls(path)
        if path.exists():
            ledger.runs = json.loads(path.read_text()).get("runs", [])
        return ledger

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"total_successes": self.total_successes, "runs": self.runs}, indent=2)
        )

    def append(self, record: dict) -> None:
        self.runs.append(record)
        self.save()


def _format_permutation(permutation: tuple) -> str:
    return "(" + ",".join(str(p) for p in permutation) + ")"


def _shard_hdf5_path(shard_path: Path) -> Path:
    """Resolve a run's extensionless shard base path to the actual file on disk.

    generate_batch.py's --shard_name is an extensionless base name (e.g. "shard_003");
    verified against real artifacts, the subprocess writes the dataset to
    "<out_dir>/<shard_name>.hdf5" -- plus a small "<shard_name>.meta.json" sidecar this module
    deliberately never touches (a few KB per run, kept on disk for post-hoc debugging, excluded
    from the disk-bound accounting and deletion below since it is negligible next to the
    multi-GB .hdf5 payload). Every consumer of "the shard" after generation --
    convert_to_lerobot.py's --shards argument, the exists()/unlink() disk-bound cleanup, and the
    ledger's "shard" field -- must resolve to this actual file, never the extensionless base
    path, which is only ever used to build --shard_name and is never itself created on disk.
    """
    return shard_path.with_name(shard_path.name + ".hdf5")


_START_MARKER_SUFFIX = ".started"

# The phases a run passes through, recorded in its start marker so that an interrupted run can
# be told apart by what it had already done. Only GENERATING is safe to resume through: the
# convert has not been submitted yet, so nothing of that run can be in the dataset.
#
# CONVERTING is a distinct phase rather than an implementation detail. A run's convert is
# submitted to the background executor and then runs concurrently with the NEXT run's batch --
# hours -- before finalize() collects its result. A marker that only turned unsafe once that
# result was collected would still read GENERATING throughout the entire period its convert was
# committing episodes, which is exactly the window this whole guard exists for.
MARKER_PHASE_GENERATING = "generating"
MARKER_PHASE_CONVERTING = "converting"
MARKER_PHASE_CONVERTED = "converted"


def _start_marker_path(cfg: CampaignCfg, run_index: int) -> Path:
    return cfg.out_dir / f"run_{run_index}{_START_MARKER_SUFFIX}"


def _write_start_marker(
    cfg: CampaignCfg, run_index: int, phase: str, permutation: tuple, shard_name: str
) -> None:
    """Record a run's phase, replacing any earlier marker for it atomically.

    Written via a temporary file and os.replace because the reader fails closed: a marker
    truncated by the very kill it is recording would be unreadable, and unreadable means
    "refuse". Making the write atomic keeps that from happening to a marker that would
    otherwise have been safe to clear. The temporary name ends in .tmp, so _started_run_indices'
    "run_*.started" glob cannot pick a half-written one up either.
    """
    run_seed, layout_seed = _derive_seeds(cfg, run_index)
    payload = {
        "run_index": run_index,
        "phase": phase,
        "seeds": {"run_seed": run_seed, "layout_seed": layout_seed},
        "permutation": list(permutation),
        "shard": shard_name,
    }
    path = _start_marker_path(cfg, run_index)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, path)


def _marker_phase(cfg: CampaignCfg, run_index: int) -> str | None:
    """A marker's recorded phase, or None if it cannot be read as one."""
    try:
        phase = json.loads(_start_marker_path(cfg, run_index).read_text()).get("phase")
    except (OSError, ValueError):
        return None
    return phase if isinstance(phase, str) else None


def _started_run_indices(cfg: CampaignCfg) -> list[int]:
    """Run indices with a start marker still on disk, i.e. started and not yet banked."""
    indices = []
    for path in cfg.out_dir.glob(f"run_*{_START_MARKER_SUFFIX}"):
        stem = path.name[len("run_") : -len(_START_MARKER_SUFFIX)]
        if stem.isdigit():
            indices.append(int(stem))
    return sorted(indices)


def check_no_interrupted_run(cfg: CampaignCfg, ledger: Ledger) -> None:
    """Refuse to start when a previous run marked itself started and never reached the ledger.

    check_resume_consistency below compares the converter's reported episode total against the
    dataset's own count, which needs a ledgered total to compare against -- so it cannot see an
    interruption during the FIRST run of a fresh --dataset_root. That run is not less exposed
    than the others; for a two-run campaign it is half of them. It is also worse than
    symmetric: if the kill lands before the first Ledger.append then ledger.json does not exist
    either, so main()'s "ledger already exists, pass --resume" refusal does not fire, and a
    bare re-invocation would regenerate run 0's identical seeds and shard name straight over
    whatever the killed run had already committed.

    The marker closes that without restructuring the ledger into a two-phase commit: it is
    written before the batch subprocess starts and removed only once that run's ledger record
    has been appended, so an orphan marker is exactly the window between the two.

    It carries the run's PHASE, and only an unsafe phase refuses. A run swept while still
    generating had not submitted its convert, so nothing of it can be in the dataset and the
    campaign regenerates it without asking. That is not a convenience: on this host a sweep
    during the multi-hour batch is the likeliest interruption by far, and a guard that fires
    routinely on a benign case teaches whoever meets it to clear markers unread, which is how
    guards stop working. Narrowing when it fires is what lets the refusal mean what it says.

    An unreadable marker is treated as unsafe. It says nothing about whether that run
    converted, and "unreadable" must not be the one input that unlocks the automatic path.

    Refuses rather than repairing, for the same reason check_resume_consistency does: whether
    the interrupted run committed episodes is a question about the dataset, and a converter
    killed mid-write can leave a dataset whose own metadata under-reports what is in it. The
    message says which comparison settles it, and what to do either way.
    """
    ledgered = {record["run_index"] for record in ledger.runs}
    orphans = [index for index in _started_run_indices(cfg) if index not in ledgered]
    if not orphans:
        return

    phases = {index: _marker_phase(cfg, index) for index in orphans}
    unsafe = [index for index, phase in phases.items() if phase != MARKER_PHASE_GENERATING]
    if not unsafe:
        for index in orphans:
            _start_marker_path(cfg, index).unlink(missing_ok=True)
            print(
                f"run {index}: interrupted before its shard was converted, so nothing of it "
                f"reached the dataset -- regenerating it",
                flush=True,
            )
        return

    listed = ", ".join(f"run {index} ({phases[index] or 'unreadable'})" for index in unsafe)
    raise ResumeMismatch(
        f"{listed} reached the converter and never reached {cfg.out_dir / 'ledger.json'}: "
        f"the campaign was interrupted after that run's convert began, so whether it committed "
        f"episodes to {cfg.dataset_root} is unknown. Compare that dataset's meta/info.json "
        f"total_episodes against the ledger's last converted_total: if they agree the run "
        f"committed nothing and the marker can be deleted, and if they do not the dataset has "
        f"to be restored to the ledger's total first. Resuming without settling that would "
        f"regenerate the run and convert it a second time."
    )


def _dataset_total_episodes(dataset_root: Path) -> int | None:
    """The dataset's own episode count, read from the one file of it this module needs.

    Returns None when the dataset does not exist yet -- which is the normal state before the
    first convert, and a mismatch only once the ledger says otherwise.
    """
    info = dataset_root / "meta" / "info.json"
    if not info.exists():
        return None
    return json.loads(info.read_text())["total_episodes"]


def check_resume_consistency(cfg: CampaignCfg, ledger: Ledger) -> None:
    """Refuse to resume a campaign whose ledger and dataset disagree about the episode count.

    The window this guards is real rather than theoretical, and on this host it is expected
    rather than unlikely: background jobs get swept, so the supervisor can be killed between
    convert_to_lerobot.py committing a run's episodes to the LeRobotDataset and Ledger.append
    recording that it did. Resume regenerates every un-ledgered run index (see run_campaign),
    so it would convert that run a second time and append a duplicate batch to the same
    dataset root -- and because a regenerated run replays the seeds but not the physics, the
    duplicates would not even be byte-identical to spot by eye.

    Comparing the converter's own reported total against the dataset's own count closes it.
    Both numbers are the same quantity from the same source (convert_to_lerobot.py prints
    `CONVERT_OK episodes=<n> total=<total>` straight from dataset.meta.total_episodes), which
    is why this is exact rather than approximate, and why it survives --split_cycles -- there
    one demo becomes several episodes, so summing the ledger's `exported` counts would not
    equal the dataset's episode count while this does.

    Refuses rather than repairs: rolling a LeRobotDataset back to a given episode count is not
    something this module can do safely, and the two honest recoveries -- restore the dataset,
    or start a fresh --dataset_root -- are choices for whoever knows which run was lost.
    """
    expected = ledger.last_converted_total
    if expected is None:
        return
    actual = _dataset_total_episodes(cfg.dataset_root)
    if actual == expected:
        return

    if actual is None:
        found = "it does not exist"
        cause = "the dataset was moved or deleted, or --dataset_root points somewhere else"
    elif actual > expected:
        found = f"it now holds {actual}"
        cause = (
            "a convert committed episodes that never reached the ledger -- the supervisor was "
            "killed between the two. Resuming would regenerate that run index and convert it "
            "again, duplicating them"
        )
    else:
        found = f"it now holds only {actual}"
        cause = "the dataset lost episodes the ledger recorded, so resuming would under-count"
    raise ResumeMismatch(
        f"{ledger.path} last recorded the converter reporting {expected} episodes in "
        f"{cfg.dataset_root}, but {found}: {cause}. Restore the dataset to {expected} "
        f"episodes, or start a fresh --dataset_root, before resuming."
    )


# numpy's np.random.seed (Isaac Lab's configure_seed) requires 0 <= seed < 2**32.
_SEED_BOUND = 2**32


def _derive_seeds(cfg: CampaignCfg, run_index: int) -> tuple[int, int]:
    """Per-run (run_seed, layout_seed), wrapped into numpy's accepted range.

    The naive `cfg.seed * 1000 + run_index` (`+ 500` for layout_seed) overflows 2**32 - 1 for
    any date-shaped campaign seed: MS1's own --seed 20260905 gives 20,260,905,000, versus
    2**32 - 1 =~ 4.29e9. Isaac Lab's configure_seed() calls np.random.seed(seed) directly from
    generate_batch.py's env.reset(seed=args.run_seed), which raises ValueError before any of
    generate_batch.py's own code runs -- this crashed run 0 of openyam-lab's campaign launch
    (--seed 20260807; its earlier mini run used --seed 42, far under the boundary, so it never
    exercised this path). `% 2**32` keeps every derived seed valid while staying
    collision-free across a single campaign's run_index range: 0..999 always differ from each
    other by at most 999 (<< 2**32), whatever cfg.seed is, so two distinct run_index values in
    that range can never wrap around to the same value (see
    test_derive_seeds_distinct_within_a_campaign_run_range).
    """
    run_seed = (cfg.seed * 1000 + run_index) % _SEED_BOUND
    layout_seed = (cfg.seed * 1000 + 500 + run_index) % _SEED_BOUND
    return run_seed, layout_seed


def run_campaign(
    cfg: CampaignCfg,
    run_batch: Callable[[int, tuple, Path], dict],
    run_convert: Callable[[Path], dict],
) -> int:
    """Drive runs until total_successes >= cfg.target; resumes from an existing ledger.

    Pipelined at depth 1: run_convert(shard k) -- CPU-only video encoding -- runs in a
    background thread while run_batch(k+1, ...) -- GPU generation -- runs on this thread, so
    the two fully overlap. Both stages stay internally serial: run_batch is only ever called
    from this thread (never concurrently with itself -- concurrent Isaac Lab subprocesses are
    VRAM-bound and rejected), and a fresh run_convert is only submitted after the *previous*
    one's result() has already been collected, so at most one is ever in flight (converts
    target a single LeRobotDataset root and can't accept concurrent appends). Steady-state
    disk footprint is therefore at most 2 shards: the one converting and the one generating;
    on a crash, up to one further un-ledgered shard may also be left behind (see below).

    Ledger append (and this function's one progress line per run) happens in
    convert-completion order, immediately after each run's convert result() is collected --
    exactly the same crash-safety contract as before: a crash mid-campaign loses at most the
    in-flight run(s).

    Stop condition: a local running total, seeded from the resumed ledger and incremented by
    each run's `exported` count the instant run_batch returns (not gated on that run's own
    convert having completed), decides whether to launch the next run. This keeps the
    original serial code's "overshoot by at most one run" invariant intact under pipelining:
    the one run that pushes the total to/past cfg.target is still fully generated, still
    handed to run_convert, and still counted in the ledger -- never discarded -- but no
    further run is launched once the running total has reached cfg.target.

    If run_convert raises, the exception propagates from the finalize() call that awaits it
    (either the unconditional one right after the next run_batch, or the one inside the
    run_batch except-handler below). The run whose convert raised is never appended to the
    ledger, nor is the run generated immediately after it (already running, concurrently, when
    the failure surfaced -- pipelining's unavoidable "wasted" run). Both runs' shards, if
    written, are left on disk, undeleted.

    If run_batch itself raises (including KeyboardInterrupt -- caught via `except
    BaseException`, not just Exception), `pending` -- the *previous* run's convert, already
    submitted to the background executor and quite possibly already finished -- is still
    finalized before the exception propagates: its episodes may already be committed to the
    LeRobotDataset, so dropping its ledger record here would silently lose track of it and
    resume would re-convert (and thus duplicate) it. If that pending convert had *also* failed,
    finalize() retrieves and raises its exception too (chained via __context__ onto the
    run_batch exception, rather than left un-retrieved for the executor's shutdown to discard).

    Either way, a resumed campaign regenerates every un-ledgered run index from scratch by
    calling run_batch again (matching the pre-existing, non-pipelined resume model) rather than
    trying to salvage a partially-generated or partially-converted shard. Regenerating is not
    reproducing: the derived seeds are replayed, but GPU physics does not repeat bit for bit
    across processes, so a regenerated run is a fresh draw that may export a different count.
    Nothing here depends on it repeating -- the ledger records what each run actually exported,
    and the target is a running total, not a per-run expectation. What resume does NOT tolerate
    is a run that converted without being ledgered, since regenerating that index would convert
    it twice; check_no_interrupted_run and check_resume_consistency refuse the resume outright
    in that case, from the start-marker side and the episode-count side respectively. A run
    interrupted BEFORE its convert was submitted committed nothing, and is regenerated without
    asking -- the marker's phase is what tells the two apart.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = cfg.out_dir / "ledger.json"
    ledger = Ledger.load(ledger_path)
    # Before the early return, not after: reporting a campaign complete over a dataset known to
    # disagree with its own ledger is the worst outcome available, because nothing downstream
    # would ever look again.
    check_no_interrupted_run(cfg, ledger)
    check_resume_consistency(cfg, ledger)

    if ledger.total_successes >= cfg.target:
        return ledger.total_successes

    schedule = permutation_schedule(cfg.seed)
    run_index = len(ledger.runs)
    for _ in range(run_index):
        next(schedule)  # keep the schedule aligned with already-completed runs on resume

    def finalize(pending: tuple[int, tuple, Path, dict, Future | None]) -> None:
        p_run_index, p_permutation, p_shard_path, p_batch_result, p_future = pending
        converted = False
        converted_total = None
        if p_future is not None:
            # Blocks until convert finishes; re-raises its exception here. Its reported total is
            # kept, not discarded: it is what check_resume_consistency compares against on the
            # next resume.
            converted_total = p_future.result().get("total")
            converted = True
            # After the convert returned and before the append: the phase now records that this
            # run's episodes are in the dataset but not yet in the ledger, which is the state a
            # resume must never regenerate through.
            _write_start_marker(
                cfg, p_run_index, MARKER_PHASE_CONVERTED, p_permutation, p_shard_path.name
            )
        deleted = False
        if p_shard_path.exists():
            p_shard_path.unlink()
            deleted = True
        p_run_seed, p_layout_seed = _derive_seeds(cfg, p_run_index)
        ledger.append(
            {
                "run_index": p_run_index,
                "seeds": {
                    "run_seed": p_run_seed,
                    "layout_seed": p_layout_seed,
                },
                "permutation": list(p_permutation),
                "exported": p_batch_result["exported"],
                "attempts": p_batch_result["attempts"],
                "shard": p_shard_path.name,
                "converted": converted,
                "converted_total": converted_total,
                "deleted": deleted,
            }
        )
        # Only now, with the run safely in the ledger, may its start marker go: the window this
        # guards is precisely the one between the convert above and this append.
        _start_marker_path(cfg, p_run_index).unlink(missing_ok=True)
        print(
            f"run {p_run_index}: perm {_format_permutation(p_permutation)} exported "
            f"{p_batch_result['exported']}/{p_batch_result['attempts']} "
            f"-> total {ledger.total_successes}/{cfg.target}",
            flush=True,
        )

    committed_total = ledger.total_successes
    with ThreadPoolExecutor(max_workers=1) as convert_executor:
        pending: tuple[int, tuple, Path, dict, Future | None] | None = None
        while committed_total < cfg.target:
            permutation = next(schedule)
            shard_path = cfg.out_dir / f"shard_{run_index:03d}"
            # Written before the subprocess, not after it returns: a marker written afterwards
            # would miss exactly the kill it exists to detect. See check_no_interrupted_run.
            _write_start_marker(
                cfg,
                run_index,
                MARKER_PHASE_GENERATING,
                permutation,
                _shard_hdf5_path(shard_path).name,
            )

            # Generation runs here, on this thread, concurrently with `pending`'s convert (if
            # any) running in the background executor thread -- this is the overlap.
            try:
                batch_result = run_batch(run_index, permutation, shard_path)
            except BaseException:
                # This run never reached run_convert, so nothing of it can be in the dataset and
                # its marker has nothing to protect. Dropping it keeps a routine BATCH_FAIL from
                # demanding manual intervention before the next resume -- which is how a guard
                # ends up deleted reflexively instead of read.
                _start_marker_path(cfg, run_index).unlink(missing_ok=True)
                # See the docstring's crash-safety paragraph: `pending` (the previous run's
                # already-submitted convert) must still be finalized -- not silently dropped --
                # before this exception propagates.
                if pending is not None:
                    finalize(pending)
                    pending = None
                raise
            committed_total += batch_result["exported"]

            if pending is not None:
                try:
                    # Waits on the (likely already-finished) background convert.
                    finalize(pending)
                except BaseException:
                    # This iteration's own run has not reached run_convert either -- the failure
                    # surfaced before its convert was submitted -- so its marker protects
                    # nothing and would only make the next resume demand a decision about a run
                    # that committed nothing. `pending`'s own marker stays put: its convert is
                    # exactly the one whose outcome is now unknown.
                    _start_marker_path(cfg, run_index).unlink(missing_ok=True)
                    raise
                pending = None

            shard_hdf5_path = _shard_hdf5_path(shard_path)
            future = None
            if batch_result["exported"] > 0:
                # Before the submit, not after: the executor thread may start committing
                # episodes the instant it is handed the job, and from here until finalize()
                # collects it this run is no longer safe to regenerate blindly.
                _write_start_marker(
                    cfg, run_index, MARKER_PHASE_CONVERTING, permutation, shard_hdf5_path.name
                )
                future = convert_executor.submit(run_convert, shard_hdf5_path)
            item = (run_index, permutation, shard_hdf5_path, batch_result, future)
            if future is None:
                finalize(item)  # nothing to convert, nothing to overlap with -- finalize now
            else:
                pending = item  # defer: overlaps with the next iteration's run_batch call

            run_index += 1

        if pending is not None:
            finalize(pending)

    return ledger.total_successes


def _batch_command(
    cfg: CampaignCfg, run_index: int, permutation: tuple, shard_path: Path
) -> list[str]:
    run_seed, layout_seed = _derive_seeds(cfg, run_index)
    cmd = [
        f"{cfg.isaaclab_path}/isaaclab.sh",
        "-p",
        "scripts/generate_batch.py",
        "--headless",
        "--enable_cameras",
        "--out_dir",
        str(cfg.out_dir),
        "--shard_name",
        shard_path.name,
        "--attempts",
        str(cfg.attempts_per_run),
        "--num_envs",
        str(cfg.num_envs),
        "--run_seed",
        str(run_seed),
        "--layout_seed",
        str(layout_seed),
        "--permutation",
        ",".join(str(p) for p in permutation),
        # Always explicit. generate_batch.py's own default is 90 minutes, under a 50-attempt
        # run at 12 envs; a truncated run still prints BATCH_OK, so leaving it to the default
        # would quietly shorten the very run the acceptance gate reads.
        "--max_wall_minutes",
        str(cfg.max_wall_minutes),
    ]
    if cfg.noise_scale is not None:
        cmd += ["--noise_scale", str(cfg.noise_scale)]
    if cfg.start_pose_offset_rad is not None:
        cmd += ["--start_pose_offset_rad", str(cfg.start_pose_offset_rad)]
    if cfg.induce_prob > 0:
        cmd += [
            "--induce_prob",
            str(cfg.induce_prob),
            "--induce_offset_weight",
            str(cfg.induce_offset_weight),
            "--induce_offset_min",
            str(cfg.induce_offset_min),
            "--induce_offset_max",
            str(cfg.induce_offset_max),
            "--induce_noise_boost",
            str(cfg.induce_noise_boost),
        ]
    return cmd


def _convert_command(cfg: CampaignCfg, shard_path: Path) -> list[str]:
    cmd = [
        "lerobot_stack/.venv/bin/python",
        "lerobot_stack/convert_to_lerobot.py",
        "--shards",
        str(shard_path),
        "--root",
        str(cfg.dataset_root),
    ]
    if cfg.split_cycles:
        cmd.append("--split_cycles")
        if cfg.sidecar_dir is not None:
            cmd += ["--sidecar_dir", str(cfg.sidecar_dir)]
    return cmd


def _batch_timeout_seconds(cfg: CampaignCfg) -> float:
    """How long to wait on a generate_batch.py subprocess before giving up on it.

    Derived from the deadline the child was actually handed rather than hard-coded, so the two
    cannot drift apart. A child that exits without its BATCH_OK token is loud -- the supervisor
    raises BatchFailed and the campaign stops with CAMPAIGN_FAIL -- but a WEDGED child produces
    no token and no log line at all, and would otherwise block the supervisor forever. That is
    a demonstrated failure mode here, not a hypothetical one: release_envelope_check.py's
    single-process layout loop wedges after its second gym.make, burning 100 % of a core with
    no GPU work and no I/O.
    """
    return (cfg.max_wall_minutes + BATCH_TIMEOUT_GRACE_MINUTES) * 60.0


def _isaaclab_batch_env(cfg: CampaignCfg) -> dict[str, str]:
    """Environment for the generate_batch.py subprocess (isaaclab.sh -p ...).

    isaaclab.sh's own bootstrap -- before it ever reaches generate_batch.py's --shard_name/
    --num_envs args -- needs VIRTUAL_ENV (or CONDA_PREFIX) set to find its bundled `python`.
    Interactively, that's normally inherited from an already-`source`d venv, but a bare
    non-interactive `python3 scripts/generate_dataset.py` -- exactly how an unattended
    overnight campaign is launched (e.g. via nohup) -- has neither, and isaaclab.sh fails with
    "python: command not found" before generate_batch.py itself ever runs (found running Task
    13's real-subprocess conversion step; that task's own invocation worked around it by
    pre-exporting VIRTUAL_ENV in its interactive shell, which this module can't rely on a
    caller having done for it).

    Derives VIRTUAL_ENV from cfg.isaaclab_path (isaaclab.sh's own venv lives at
    "<isaaclab_path>/.venv") but never overrides a VIRTUAL_ENV the calling environment already
    provides -- if the campaign is itself launched from inside an already-activated venv
    (isaaclab's or otherwise), that takes precedence.
    """
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    env.setdefault("VIRTUAL_ENV", str(cfg.isaaclab_path / ".venv"))
    return env


def _make_real_run_batch(cfg: CampaignCfg) -> Callable[[int, tuple, Path], dict]:
    def write_log(run_index: int, stdout: str | None, stderr: str | None) -> None:
        (cfg.out_dir / f"run_{run_index}.log").write_text((stdout or "") + (stderr or ""))

    def run_batch(run_index: int, permutation: tuple, shard_path: Path) -> dict:
        cmd = _batch_command(cfg, run_index, permutation, shard_path)
        try:
            result = subprocess.run(
                cmd,
                env=_isaaclab_batch_env(cfg),
                capture_output=True,
                text=True,
                check=False,  # a non-zero exit still yields stdout to parse for BATCH_FAIL
                timeout=_batch_timeout_seconds(cfg),
            )
        except subprocess.TimeoutExpired as exc:
            # The run whose log is most worth having is the one that hung, and a wedge leaves no
            # BATCH_FAIL token to read instead. subprocess.run buffers the child's output and
            # kills it before raising, so this is the only chance to keep what it did say.
            write_log(run_index, exc.stdout, exc.stderr)
            raise
        write_log(run_index, result.stdout, result.stderr)
        return parse_batch_output(result.stdout)

    return run_batch


def _make_real_run_convert(cfg: CampaignCfg) -> Callable[[Path], dict]:
    def run_convert(shard_path: Path) -> dict:
        result = subprocess.run(
            _convert_command(cfg, shard_path), capture_output=True, text=True, check=False
        )
        return parse_convert_output(result.stdout)

    return run_convert


def _build_cfg(args: argparse.Namespace) -> CampaignCfg:
    return CampaignCfg(
        out_dir=args.out_dir,
        dataset_root=args.dataset_root,
        target=args.target,
        attempts_per_run=args.attempts_per_run,
        num_envs=args.num_envs,
        seed=args.seed,
        isaaclab_path=args.isaaclab_path,
        max_wall_minutes=args.max_wall_minutes,
        noise_scale=args.noise_scale,
        start_pose_offset_rad=args.start_pose_offset_rad,
        induce_prob=args.induce_prob,
        induce_offset_weight=args.induce_offset_weight,
        induce_offset_min=args.induce_offset_min,
        induce_offset_max=args.induce_offset_max,
        induce_noise_boost=args.induce_noise_boost,
        split_cycles=args.split_cycles,
        sidecar_dir=args.sidecar_dir,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--attempts_per_run", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_wall_minutes",
        type=float,
        default=BATCH_WALL_MINUTES,
        help=(
            "wall-clock ceiling for each generate_batch.py run, forwarded as its own "
            "--max_wall_minutes; the supervisor's timeout over that subprocess is derived "
            "from it"
        ),
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="multiplier on every OracleNoise axis (generate_batch.py's default is 1.0)",
    )
    parser.add_argument(
        "--start_pose_offset_rad",
        type=float,
        default=None,
        help=(
            "half-width of the arm's start-pose randomisation at reset "
            "(LegoSortDatagenEnvCfg's default is 0.05; widen it to demonstrate recovery "
            "from the poses a trained policy's rollouts actually visit)"
        ),
    )
    parser.add_argument(
        "--induce_prob",
        type=float,
        default=0.0,
        help=(
            "per-pick probability of a deliberate first-attempt grasp failure, forwarded to "
            "generate_batch.py (0 = off, identical to pre-induction behaviour)"
        ),
    )
    parser.add_argument("--induce_offset_weight", type=float, default=0.7)
    parser.add_argument("--induce_offset_min", type=float, default=0.015)
    parser.add_argument("--induce_offset_max", type=float, default=0.025)
    parser.add_argument("--induce_noise_boost", type=float, default=5.0)
    parser.add_argument("--split_cycles", action="store_true")
    parser.add_argument("--sidecar_dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--isaaclab_path",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_PATH", str(Path.home() / "src/IsaacLab"))),
    )
    args = parser.parse_args(argv)
    if args.sidecar_dir is not None and not args.split_cycles:
        # Mirrors convert_to_lerobot.py's own guard (parser.error there, ValueError in
        # convert()) so the two entry points agree on which flag combinations are legal --
        # without this, a bare --sidecar_dir with no --split_cycles would run this whole
        # script to completion, exit 0, and produce no sidecars, silently defeating the
        # only reason --sidecar_dir was passed.
        parser.error("--sidecar_dir requires --split_cycles")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _build_cfg(args)
    ledger_path = cfg.out_dir / "ledger.json"

    if args.dry_run:
        # Always describe run 0's commands: the dry run's purpose is to let a human
        # sanity-check the command lines before an overnight campaign, not to reflect
        # mid-campaign state.
        permutation = next(permutation_schedule(cfg.seed))
        shard_path = cfg.out_dir / "shard_000"
        print("run 0:", " ".join(_batch_command(cfg, 0, permutation, shard_path)))
        print("run 0:", " ".join(_convert_command(cfg, _shard_hdf5_path(shard_path))))
        return 0

    if ledger_path.exists() and not args.resume:
        print(
            f"CAMPAIGN_FAIL: {ledger_path} already exists; pass --resume to continue it",
            file=sys.stderr,
        )
        print("CAMPAIGN_FAIL", flush=True)
        return 1

    run_batch = _make_real_run_batch(cfg)
    run_convert = _make_real_run_convert(cfg)

    try:
        total = run_campaign(cfg, run_batch, run_convert)
    except (BatchFailed, ResumeMismatch) as exc:
        print(f"CAMPAIGN_FAIL: {exc}", file=sys.stderr, flush=True)
        print("CAMPAIGN_FAIL", flush=True)
        return 1
    except BaseException:
        # Every exit path emits its token. A campaign that dies without printing CAMPAIGN_FAIL
        # is, to whoever is tailing the log or grepping for the token, indistinguishable from
        # one still running -- and this is launched detached on a host that sweeps background
        # jobs, so "still running" is a hypothesis someone will actually entertain. BaseException
        # rather than Exception because KeyboardInterrupt (Ctrl-C on an overnight run) and
        # SystemExit are not Exceptions. The exception itself is re-raised untouched: the
        # interpreter prints its traceback to stderr and sets the exit status, and this handler
        # only guarantees the token beside it.
        print("CAMPAIGN_FAIL", flush=True)
        raise

    runs = len(Ledger.load(ledger_path).runs)
    print(f"CAMPAIGN_OK total={total} runs={runs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
