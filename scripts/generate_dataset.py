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
ledger.json before the next run starts. Nothing about a campaign's progress is
inferred from the final CAMPAIGN_OK line, which a supervisor killed mid-campaign
never prints -- on this host background jobs are liable to be swept, and a chain
that dies in the middle must not be indistinguishable from one that never started.
--resume rebuilds state from the ledger alone, so a campaign interrupted anywhere
loses at most the runs that were in flight. Symmetrically, every exit path from
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
    and the target is a running total, not a per-run expectation.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = cfg.out_dir / "ledger.json"
    ledger = Ledger.load(ledger_path)

    if ledger.total_successes >= cfg.target:
        return ledger.total_successes

    schedule = permutation_schedule(cfg.seed)
    run_index = len(ledger.runs)
    for _ in range(run_index):
        next(schedule)  # keep the schedule aligned with already-completed runs on resume

    def finalize(pending: tuple[int, tuple, Path, dict, Future | None]) -> None:
        p_run_index, p_permutation, p_shard_path, p_batch_result, p_future = pending
        converted = False
        if p_future is not None:
            p_future.result()  # blocks until convert finishes; re-raises its exception here
            converted = True
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
                "deleted": deleted,
            }
        )
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

            # Generation runs here, on this thread, concurrently with `pending`'s convert (if
            # any) running in the background executor thread -- this is the overlap.
            try:
                batch_result = run_batch(run_index, permutation, shard_path)
            except BaseException:
                # See the docstring's crash-safety paragraph: `pending` (the previous run's
                # already-submitted convert) must still be finalized -- not silently dropped --
                # before this exception propagates.
                if pending is not None:
                    finalize(pending)
                    pending = None
                raise
            committed_total += batch_result["exported"]

            if pending is not None:
                finalize(pending)  # waits on the (likely already-finished) background convert
                pending = None

            shard_hdf5_path = _shard_hdf5_path(shard_path)
            future = (
                convert_executor.submit(run_convert, shard_hdf5_path)
                if batch_result["exported"] > 0
                else None
            )
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
    def run_batch(run_index: int, permutation: tuple, shard_path: Path) -> dict:
        cmd = _batch_command(cfg, run_index, permutation, shard_path)
        result = subprocess.run(
            cmd,
            env=_isaaclab_batch_env(cfg),
            capture_output=True,
            text=True,
            check=False,  # a non-zero exit still yields stdout to parse for BATCH_FAIL
        )
        (cfg.out_dir / f"run_{run_index}.log").write_text(result.stdout + result.stderr)
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
    except BatchFailed as exc:
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
