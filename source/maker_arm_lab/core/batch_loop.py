"""The generation-run loop and its records, shared by both simulator backends.

``scripts/generate_batch.py`` (Isaac Lab) and ``scripts/mj_generate_batch.py`` (MuJoCo) differ
only in how they build the environment. Everything that decides what a run RECORDS lives here,
so the two backends' ``RUN_META`` / ``RUN_STATS`` lines and ``<shard>.meta.json`` sidecars carry
the same fields with the same meanings and a campaign ledger cannot tell them apart except by
the ``backend`` field. The duck-typed env surface is the one ``OracleRig`` reads; see
``maker_arm_lab.mj.env`` for the list.

The campaign target the ``--probe`` extrapolation is quoted against stays the MS1 number (50
successes) for the same reason it did in generate_batch.py: it is a probe convention, not a
campaign size.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from maker_arm_lab.core import layout, oracle
from maker_arm_lab.core.layout import sample_run_draw

CAMPAIGN_TARGET_SUCCESSES = 50


def build_run(args) -> tuple[dict, layout.BinLayout, oracle.OracleNoise]:
    """The run-level draw (RNG'd from --layout_seed) with --permutation applied, plus the
    derived BinLayout and OracleNoise, from the parsed generate_batch-style ``args``."""
    perm = tuple(int(v) for v in args.permutation.split(","))
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"--permutation must be a permutation of 0,1,2,3, got {args.permutation!r}"
        )
    if not 0.0 <= args.induce_prob <= 1.0:
        raise ValueError(f"--induce_prob must be in [0, 1], got {args.induce_prob}")
    if args.induce_offset_min > args.induce_offset_max:
        raise ValueError(
            f"--induce_offset_min ({args.induce_offset_min}) must be <= "
            f"--induce_offset_max ({args.induce_offset_max})"
        )
    rng = random.Random(args.layout_seed)
    draw = sample_run_draw(rng)
    draw["slot_colors"] = perm
    bl = layout.BinLayout(
        bin_x=draw["bin_x"], bin_ys=tuple(draw["bin_ys"]), slot_colors=draw["slot_colors"]
    )
    noise = oracle.OracleNoise(
        hover_height=(0.12 - 0.02 * args.noise_scale, 0.12 + 0.02 * args.noise_scale),
        lateral_offset=0.003 * args.noise_scale,
        yaw_offset=0.0873 * args.noise_scale,
        gain_scale=(1.0 - 0.3 * args.noise_scale, 1.0),
        action_noise_std=0.05 * args.noise_scale,
        via_jitter=0.02 * args.noise_scale,
        # Outside --noise_scale on purpose: it removes a SHORTCUT (the fixed R,B,Y,G,R,B
        # color schedule) rather than widening a noise band, so no recipe string should
        # read as if it might have carried it.
        randomize_pick_order=args.randomize_pick_order,
    )
    return draw, bl, noise


def make_induction(args) -> oracle.FailureInduction | None:
    if args.induce_prob <= 0.0:
        return None
    return oracle.FailureInduction(
        prob=args.induce_prob,
        offset_weight=args.induce_offset_weight,
        offset_range=(args.induce_offset_min, args.induce_offset_max),
        noise_boost=args.induce_noise_boost,
    )


def run_meta(
    args,
    draw: dict,
    noise: oracle.OracleNoise,
    induction: oracle.FailureInduction | None,
    *,
    num_bricks: int,
    export_mode: str,
    backend: str,
    task: str,
) -> dict:
    return {
        "task": task,
        "backend": backend,
        "run_seed": args.run_seed,
        "layout_seed": args.layout_seed,
        "num_envs": args.num_envs,
        "num_bricks": num_bricks,
        "attempts": args.attempts,
        "probe": args.probe,
        "noise_scale": args.noise_scale,
        "start_pose_offset_rad": args.start_pose_offset_rad,
        # What was CONFIGURED; each episode's own "pick_order" records what was picked.
        "randomize_pick_order": noise.randomize_pick_order,
        "induction": None
        if induction is None
        else {
            "prob": induction.prob,
            "offset_weight": induction.offset_weight,
            "offset_range": list(induction.offset_range),
            "noise_boost": induction.noise_boost,
        },
        "cameras": not args.no_cameras,
        "permutation": draw["slot_colors"],
        "shard_name": args.shard_name,
        "out_dir": args.out_dir,
        "max_wall_minutes": args.max_wall_minutes,
        "dataset_export_mode": export_mode,
        "run_draw": draw,
        # Carried in the sidecar itself, not only in a docstring: a consumer reading
        # <shard>.meta.json alone has no other way to learn that "sort_attempts" is not a
        # success count.
        "outcome_fields_note": (
            "episodes[].sort_attempts is an ATTEMPT count, not a success count -- the oracle "
            "bumps it when a RETREAT phase ends, whether or not the brick it was carrying "
            "arrived, so it counts a brick dropped in transit and can read either side of "
            "the truth. Use episodes[].succeeded (the env's own 'success' termination, the "
            "tensor the recorder's export filter grades on) as the label, "
            "episodes[].bricks_delivered for partial credit, and run_stats.successes (the "
            "recorder's own exported_successful_episode_count) for the run total."
        ),
        "reproducibility_note": (
            "Episodes after the first per env conclude via the env's auto-reset, which takes "
            "no seed of its own -- per-episode replay comes from each demo's own recorded "
            "'initial_state' in the HDF5, not from a seed. This run is identified by run_seed "
            "(OracleRig's noise-draw RNG) and layout_seed (sample_run_draw's RNG), plus "
            "num_envs -- per-step action noise is drawn from Torch's process-global RNG, so "
            "which noise sample lands on which control step depends on how many envs are "
            "stepped together, not on the two seeds alone."
        ),
    }


def run_batch(env, rig, target_attempts: int, max_wall_minutes: float) -> dict:
    """Drive ``rig`` until ``target_attempts`` episodes concluded or the deadline passed.

    Returns the raw counters ``run_stats`` turns into RUN_STATS. Closes the env on every
    exit path (the recorder's export count is read before that), as generate_batch.py did.
    """
    episodes: list[dict] = []
    abandon_reasons: Counter = Counter()
    control_steps = 0
    clamped_steps = 0
    attempts = 0
    cycle_bricks: list[set[int]] = [set() for _ in range(rig.num_envs)]
    start = time.time()
    deadline = start + max_wall_minutes * 60
    try:
        while attempts < target_attempts and time.time() < deadline:
            out = rig.step()
            control_steps += 1
            # Per-env flag: on how many envs the arm's command had to be clamped this step.
            clamped_steps += int(env.action_manager.get_term("arm_action").clamped.sum().item())
            for i, b in enumerate(env.oracle_cycle_brick.tolist()):
                if b >= 0:
                    cycle_bricks[i].add(b)
            finished = out["dones"] | out["abandoned"]
            if bool(finished.any()):
                env_ids = finished.nonzero(as_tuple=False).squeeze(-1).tolist()
                done_ids = set(out["dones"].nonzero(as_tuple=False).squeeze(-1).tolist())
                for i in env_ids:
                    # episode_stats is frozen by the rig on the step the episode ended, so this
                    # read is safe for every env in the batch (see OracleRig.episode_stats).
                    stats = rig.episode_stats(i)
                    if i in done_ids:
                        outcome = "success" if stats["succeeded"] else "time_out"
                    else:
                        outcome = "abandoned"
                        abandon_reasons[stats["abandon"] or "unknown"] += 1
                    episodes.append(
                        {"env": i, "outcome": outcome, "cycles": len(cycle_bricks[i]), **stats}
                    )
                    cycle_bricks[i] = set()
                attempts += len(env_ids)
                abandoned_ids = out["abandoned"].nonzero(as_tuple=False).squeeze(-1).tolist()
                if abandoned_ids:
                    # Watchdog/oracle abandon: explicit env-scoped reset. The episode exports
                    # with success=False -- dropped in the default 'succeeded' mode.
                    env.reset(env_ids=torch.tensor(abandoned_ids, device=env.device))
                rig.rearm(env_ids)
    finally:
        wall_seconds = time.time() - start
        exported = env.recorder_manager.exported_successful_episode_count
        env.close()
    return {
        "episodes": episodes,
        "abandon_reasons": abandon_reasons,
        "control_steps": control_steps,
        "clamped_steps": clamped_steps,
        "attempts": attempts,
        "wall_seconds": wall_seconds,
        "exported": exported,
    }


def run_stats(counters: dict, *, target_attempts: int, num_envs: int, probe: bool) -> dict:
    episodes = counters["episodes"]
    abandon_reasons = counters["abandon_reasons"]
    control_steps = counters["control_steps"]
    clamped_steps = counters["clamped_steps"]
    attempts = counters["attempts"]
    wall_seconds = counters["wall_seconds"]
    exported = counters["exported"]
    dones_count = sum(1 for e in episodes if e["outcome"] in ("success", "time_out"))
    abandoned_count = sum(1 for e in episodes if e["outcome"] == "abandoned")
    stats = {
        "attempts": attempts,
        "attempts_target": target_attempts,
        # Authoritative success count (the recorder's own running total), NOT a sum over
        # episodes[]["outcome"].
        "successes": exported,
        "time_outs": dones_count - exported,
        "abandons": abandoned_count,
        "abandon_reasons": dict(abandon_reasons),
        "watchdog_trips": abandon_reasons.get("watchdog", 0),
        "control_steps": control_steps,
        # The denominator is env control steps, not control steps, because the numerator
        # counts one per env.
        "clamped_steps": clamped_steps,
        "clamp_fraction": clamped_steps / (control_steps * num_envs) if control_steps else 0.0,
        "wall_seconds": wall_seconds,
        "sim_steps_per_sec": control_steps / wall_seconds if wall_seconds > 0 else 0.0,
        "deadline_hit": attempts < target_attempts,
        "episodes_per_env": {
            str(i): c for i, c in sorted(Counter(e["env"] for e in episodes).items())
        },
        # The oracle's planning-lookahead counters, summed over the run's episodes.
        "raise_vetoed": sum(int(e.get("raise_vetoed", 0)) for e in episodes),
        "hover_lowered": sum(int(e.get("hover_lowered", 0)) for e in episodes),
        # The per-run audit of --randomize_pick_order: with six bricks the lowest-index
        # sequence occurs by chance once in 720 episodes.
        "off_schedule_episodes": sum(
            1 for e in episodes if list(e.get("pick_order", [])) != sorted(e.get("pick_order", []))
        ),
    }
    if probe:
        success_rate = exported / attempts if attempts else 0.0
        wall_sec_per_episode = wall_seconds / attempts if attempts else 0.0
        stats["success_rate"] = success_rate
        stats["wall_sec_per_episode"] = wall_sec_per_episode
        stats[f"extrapolated_{CAMPAIGN_TARGET_SUCCESSES}_success_wall_hours"] = (
            (CAMPAIGN_TARGET_SUCCESSES / success_rate) * wall_sec_per_episode / 3600.0
            if success_rate > 0
            else None
        )
    return stats


def write_meta(
    out_dir: str | Path, shard_name: str, meta: dict, stats: dict, episodes: list
) -> Path:
    path = Path(out_dir) / f"{shard_name}.meta.json"
    path.write_text(
        json.dumps({"run_meta": meta, "run_stats": stats, "episodes": episodes}, indent=2)
    )
    return path


def add_batch_arguments(parser) -> None:
    """The generate_batch.py flags both backends share (AppLauncher flags excluded)."""
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--shard_name", type=str, default="shard")
    parser.add_argument("--attempts", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--run_seed", type=int, default=0, help="Seeds OracleRig's noise-draw RNG.")
    parser.add_argument(
        "--layout_seed", type=int, default=0, help="Seeds sample_run_draw's run-level RNG."
    )
    parser.add_argument(
        "--permutation",
        type=str,
        default="0,1,2,3",
        help="Comma-separated color-slot permutation, overrides sample_run_draw's own shuffle "
        "-- the campaign supervisor schedules permutations without replacement.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Multiplier on every OracleNoise axis (0 = OracleNoise's own nominal minus the "
        "hover-height band collapsed to its center).",
    )
    parser.add_argument(
        "--randomize_pick_order",
        action="store_true",
        help="Pick the next brick uniformly at random among the ones still on the table instead "
        "of taking the lowest index (openyam-lab's largest gate lever). Off by default.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print extra throughput/extrapolation fields in RUN_STATS. Does not change "
        "--attempts.",
    )
    parser.add_argument(
        "--start_pose_offset_rad",
        type=float,
        default=None,
        help="half-width of the uniform per-joint offset applied to the arm's start pose at "
        "reset (the env's own default is 0.05).",
    )
    parser.add_argument(
        "--induce_prob",
        type=float,
        default=0.0,
        help="Per-pick probability of a deliberate first-attempt grasp failure (0 = off).",
    )
    parser.add_argument("--induce_offset_weight", type=float, default=0.7)
    parser.add_argument("--induce_offset_min", type=float, default=0.015)
    parser.add_argument("--induce_offset_max", type=float, default=0.025)
    parser.add_argument("--induce_noise_boost", type=float, default=5.0)
    parser.add_argument(
        "--export_mode",
        type=str,
        choices=("succeeded", "all"),
        default="succeeded",
        help="'succeeded' exports only episodes whose 'success' termination fired; 'all' "
        "exports every concluded episode (instrument-validation shards only).",
    )
    parser.add_argument("--no_cameras", action="store_true", help="cameras off in the env cfg.")
    parser.add_argument("--max_wall_minutes", type=float, default=90.0)
