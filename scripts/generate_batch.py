"""Batch generation: one process = one generation run = many oracle episodes.

The MS1 campaign's per-shard worker, carried over from openyam-lab's M2
campaign. ``LegoSortDatagenEnvCfg`` fixes one run-level draw (bin layout, color
permutation, lighting, floor color, camera jitter -- ``datagen.sample_run_draw``)
for the whole process; ``--num_envs`` parallel envs are then driven by one
``OracleRig`` through as many reset-level episodes as fit in ``--attempts``
(or ``--max_wall_minutes``, whichever comes first). Recorders default to
``EXPORT_SUCCEEDED_ONLY`` mode (``--export_mode succeeded``, unchanged from
before this flag existed), so success export is fully automatic: the
active ``"success"`` termination term firing inside ``env.step()`` drives
Isaac Lab's own auto-reset, which calls ``record_pre_reset`` -- that call
grades the episode against ``termination_manager``'s cached "success" tensor
and writes it if true -- with zero driver code here (recorder_manager.py
:391-419). A watchdog/oracle-exhausted "abandon" is not an Isaac Lab
termination at all (see ``oracle_runtime.OracleRig``): this driver ends that
episode itself with an explicit ``env.reset(env_ids=...)``, which exports it
too, gated ``success=False`` and so dropped in the default mode.
``--export_mode all`` switches to ``EXPORT_ALL``, exporting every concluded
episode (success, time_out, and abandoned) regardless of outcome. This is
not for training data -- it exists for instrument-validation shards (e.g. a
``--induce_prob 1.0`` run, where every pick is deliberately sabotaged and no
episode can complete all bricks in one time budget): a timed-out episode's
recorded closes against known induced offsets are exactly the ground truth
an offline check needs, and the succeeded-only filter would drop all of it.

``--permutation`` overrides ``sample_run_draw``'s uniform color-slot shuffle
explicitly: the campaign supervisor (``scripts/generate_dataset.py``) schedules
permutations without replacement across runs, which a per-run coin flip cannot
guarantee.

Reproducibility: episodes after the first per env end via Isaac Lab's
auto-reset, which takes no seed of its own -- per-episode replay comes from
each demo's own recorded ``initial_state`` in the HDF5, not from a seed. This
*run* is identified by ``run_seed`` (drives ``OracleRig``'s noise-draw RNG,
via ``core.oracle.sample_episode_draw``) and ``layout_seed`` (drives
``sample_run_draw``'s RNG), plus ``num_envs`` -- the per-step action noise in
``oracle_runtime.OracleRig.step`` is drawn from Torch's process-global RNG, so
which noise sample lands on which control step depends on how many envs are
stepped together each control step, not just on the two Python seeds above.
All of this -- plus the full run draw and per-episode retry/outcome counts --
is written to ``<shard_name>.meta.json`` alongside the shard, and printed as
one-line ``RUN_META``/``RUN_STATS`` JSON for the supervisor to parse from
stdout.

Three of ``OracleRig.episode_stats()``'s fields describe how an episode went
and they are not the same number, so this driver is deliberate about which one
it grades with. ``"sort_attempts"`` counts pick-and-place cycles the ORACLE
carried through RETREAT -- an ATTEMPT count, which counts a brick dropped in
transit and can disagree with the label in both directions; it is reported and
never graded on. ``"bricks_delivered"`` is partial credit read off the bricks.
``"succeeded"`` is the env's own ``success`` termination -- the exact tensor
Isaac Lab's recorder grades ``EXPORT_SUCCEEDED_ONLY`` against -- and it is the
LABEL, so each episode record's ``"outcome"`` reads it for envs that ended via
termination (``"success"`` or ``"time_out"``), and the driver's own abandon
reason for envs the watchdog/oracle gave up on (``"abandoned"``). RUN_STATS'
``successes`` count is the recorder's own
``exported_successful_episode_count``, which is incremented from that same
success term (recorder_manager.py:410-416, 514-517) and is authoritative -- not
a sum over per-episode outcomes.

Every one of those per-episode fields is read through ``episode_stats()``
between the episode ending and this driver's ``rearm()``, which is the window
``OracleRig`` guarantees them in: ``succeeded`` and ``bricks_delivered`` come
from env state the next ``env.step()`` overwrites for the WHOLE batch, so the
rig freezes them the step an episode ends. That matters here and nowhere else
so far -- with ``--num_envs`` > 1 the natural pattern is to finish env A, keep
stepping so env B can finish, and only then read A. Do not cache them here; the
freeze is the contract.

RUN_STATS also carries ``clamped_steps`` and ``clamp_fraction``: how often the
joint-position action term had to clamp the oracle's command to the arm's own
limits, over all control steps of all envs. Spec section 5 gates a generation
run at ``clamp_fraction <= 0.01`` -- a non-zero fraction means the oracle is
demanding targets the real arm would refuse (spec section 4.3), so it is a
property of the demonstrations rather than of the simulator.

Each episode's ``episode_stats()`` payload also always carries
``"induced_offset"``/``"induced_noise"`` counters (how many of that episode's
picks got a deliberate first-attempt grasp failure of each kind), propagated
into the sidecar the same way -- they read ``0`` on runs where ``--induce_prob``
is left at its default. It also always carries ``"pick_order"``, the brick indices
that episode targeted in order, which is what makes ``--randomize_pick_order``
auditable per episode rather than only in ``RUN_META``: with the flag off it reads
``[0, 1, ..., num_bricks - 1]``.

Run:
  $ISAACLAB_PATH/isaaclab.sh -p scripts/generate_batch.py --headless \\
      --out_dir /tmp/batch0 --no_cameras --attempts 6 --num_envs 3 --probe
  $ISAACLAB_PATH/isaaclab.sh -p scripts/generate_batch.py --headless --enable_cameras \\
      --out_dir /tmp/batch1 --attempts 10 --num_envs 12 --probe
"""

import argparse
import json
import os
import random
import time
import traceback
from collections import Counter

from isaaclab.app import AppLauncher

# The successful-episode target a --probe run extrapolates its wall clock to. This is MS1's
# own campaign size (scripts/generate_dataset.py --target 50, Task D7's acceptance run), not
# openyam-lab's 600 -- the number came across with the file and would have overstated the
# campaign by 12x. The RUN_STATS key is built from it so the two cannot drift apart.
CAMPAIGN_TARGET_SUCCESSES = 50

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Datagen-v0")
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
    help="Comma-separated color-slot permutation, overrides sample_run_draw's own shuffle -- "
    "the campaign supervisor schedules permutations without replacement.",
)
parser.add_argument(
    "--noise_scale",
    type=float,
    default=1.0,
    help="Multiplier on every OracleNoise axis (0 = OracleNoise's own nominal minus the "
    "hover-height band collapsed to its center; see the OracleNoise construction below).",
)
parser.add_argument(
    "--randomize_pick_order",
    action="store_true",
    help="Pick the next brick uniformly at random among the ones still on the table instead "
    "of taking the lowest index. Brick color is index %% 4, so the default order demonstrates "
    "the fixed color schedule R,B,Y,G,R,B in every episode, which openyam-lab measured its "
    "policy following instead of the brick it carries (8.3%% wrong-bin on-schedule vs 42.0%% "
    "off-schedule). Off by default, so every MS1 recipe stays byte-identical. Drawn from the "
    "--run_seed RNG, and each episode's record carries the pick_order it actually used.",
)
parser.add_argument(
    "--probe",
    action="store_true",
    help="Print extra throughput/extrapolation fields in RUN_STATS (sim_steps_per_sec, "
    "wall_sec_per_episode, the CAMPAIGN_TARGET_SUCCESSES wall-clock extrapolation). Does "
    "not change "
    "--attempts -- a probe run is just a normal run with --attempts set low (10 is the "
    "convention) plus this extra reporting.",
)
parser.add_argument(
    "--start_pose_offset_rad",
    type=float,
    default=None,
    help=(
        "half-width of the uniform per-joint offset applied to the arm's start pose at reset "
        "(LegoSortDatagenEnvCfg's own default is 0.05). openyam-lab's v1 campaign left it at "
        "0.05, so every "
        "demonstration begins within 0.05 rad of the home pose -- an order of magnitude narrower "
        "than the 1-sigma-plus region a trained policy's rollouts actually visit "
        "(measured on openyam-lab's arm, in its M3 replan; not re-measured here). Widen "
        "it to demonstrate recovery from those poses."
    ),
)
parser.add_argument(
    "--induce_prob",
    type=float,
    default=0.0,
    help="Per-pick probability of a deliberate first-attempt grasp failure "
    "(0 = off, identical to pre-induction behaviour)",
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
    help="Which concluded episodes reach the exported HDF5. 'succeeded' (default, unchanged "
    "behaviour) exports only episodes whose 'success' termination fired -- maps to "
    "DatasetExportMode.EXPORT_SUCCEEDED_ONLY. 'all' exports every concluded episode "
    "(success, time_out, and abandoned) -- maps to EXPORT_ALL. Use 'all' only for "
    "instrument-validation shards (e.g. --induce_prob 1.0 runs) where a timed-out "
    "episode's recorded closes against known induced offsets are the ground truth being "
    "checked, not for training data.",
)
parser.add_argument("--no_cameras", action="store_true", help="cameras=False in the env cfg.")
parser.add_argument("--max_wall_minutes", type=float, default=90.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# env.reset(seed=args.run_seed) below reaches Isaac Lab's configure_seed(), which calls
# np.random.seed(seed) directly -- numpy requires 0 <= seed < 2**32 and raises ValueError deep
# inside isaaclab's reset path otherwise, well after AppLauncher has already paid the ~1-2 minute
# Kit startup cost. A caller-derived seed outside that range (this crashed openyam-lab's M2
# campaign run 0 -- generate_dataset.py's --seed 20260807 naively multiplied by 1000 overflowed,
# and Task D7 ports that supervisor here too, so the same seed can still reach this) is checked
# here instead, before AppLauncher even launches, so a bad seed fails fast with a clear message.
for _seed_arg in ("run_seed", "layout_seed"):
    _seed_val = getattr(args, _seed_arg)
    if not (0 <= _seed_val < 2**32):
        parser.error(
            f"--{_seed_arg}={_seed_val} is out of numpy's np.random.seed range (0 <= seed < 2**32)"
        )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab (and friends) are only importable once AppLauncher has launched Kit;
# the deferral is deliberate, as in smoke_usd.py.
import gymnasium as gym
import torch
from isaaclab.managers import DatasetExportMode

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.core import layout, oracle
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import (
    LegoSortDatagenEnvCfg,
    sample_run_draw,
)
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

# --export_mode's string choices, resolved against the isaaclab enum here (only
# importable post-AppLauncher, see above) rather than at argparse-definition time.
EXPORT_MODES = {
    "succeeded": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    "all": DatasetExportMode.EXPORT_ALL,
}


def build_run(args: argparse.Namespace) -> tuple[dict, layout.BinLayout, oracle.OracleNoise]:
    """The run-level draw (RNG'd from --layout_seed) with --permutation applied,
    plus the derived BinLayout and OracleNoise -- split out from ``run()`` so
    RUN_META can be assembled before anything GPU-side is touched."""
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


def run() -> tuple[bool, int, int]:
    """Returns ``(ok, exported, attempts)``."""
    os.makedirs(args.out_dir, exist_ok=True)
    target_attempts = args.attempts
    draw, bl, noise = build_run(args)

    if args.start_pose_offset_rad is not None:
        draw = {**draw, "start_pose_offset_rad": args.start_pose_offset_rad}
    env_cfg = LegoSortDatagenEnvCfg(cameras=not args.no_cameras, **draw)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    # Read back by RecorderManager (getattr(env.cfg, "env_name", None)) and
    # stamped into the exported HDF5's metadata -- see recorder_manager.py:182.
    env_cfg.env_name = args.task
    env_cfg.recorders.dataset_export_dir_path = args.out_dir
    env_cfg.recorders.dataset_filename = args.shard_name
    export_mode = EXPORT_MODES[args.export_mode]
    env_cfg.recorders.dataset_export_mode = export_mode
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset(seed=args.run_seed)

    induction = None
    if args.induce_prob > 0.0:
        induction = oracle.FailureInduction(
            prob=args.induce_prob,
            offset_weight=args.induce_offset_weight,
            offset_range=(args.induce_offset_min, args.induce_offset_max),
            noise_boost=args.induce_noise_boost,
        )
    rig = OracleRig(env, bl, noise, random.Random(args.run_seed), induction=induction)

    run_meta = {
        "task": args.task,
        "run_seed": args.run_seed,
        "layout_seed": args.layout_seed,
        "num_envs": args.num_envs,
        "num_bricks": env_cfg.num_bricks,
        "attempts": target_attempts,
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
        "dataset_export_mode": export_mode.name,
        "run_draw": draw,
        # Carried in the sidecar itself, not only in this module's docstring: a
        # consumer reading <shard>.meta.json alone has no other way to learn
        # that "sort_attempts" is not a success count.
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
            "Episodes after the first per env conclude via Isaac Lab's auto-reset, which "
            "takes no seed of its own -- per-episode replay comes from each demo's own "
            "recorded 'initial_state' in the HDF5, not from a seed. This run is identified "
            "by run_seed (OracleRig's noise-draw RNG) and layout_seed (sample_run_draw's "
            "RNG), plus num_envs -- per-step action noise is drawn from Torch's "
            "process-global RNG, so which noise sample lands on which control step depends "
            "on how many envs are stepped together, not on the two seeds alone."
        ),
    }
    print("RUN_META " + json.dumps(run_meta), flush=True)

    episodes: list[dict] = []
    abandon_reasons: Counter = Counter()
    control_steps = 0
    clamped_steps = 0
    attempts = 0
    cycle_bricks: list[set[int]] = [set() for _ in range(rig.num_envs)]
    start = time.time()
    deadline = start + args.max_wall_minutes * 60
    try:
        while attempts < target_attempts and time.time() < deadline:
            out = rig.step()
            control_steps += 1
            # Per-env flag, OR-accumulated across the control step's physics substeps by
            # mdp.actions.ClampedJointPositionAction, so this sums to "env control steps on
            # which the arm's command had to be clamped" over the whole batch.
            clamped_steps += int(env.action_manager.get_term("arm_action").clamped.sum().item())
            for i, b in enumerate(env.oracle_cycle_brick.tolist()):
                if b >= 0:
                    cycle_bricks[i].add(b)
            finished = out["dones"] | out["abandoned"]
            if bool(finished.any()):
                env_ids = finished.nonzero(as_tuple=False).squeeze(-1).tolist()
                done_ids = set(out["dones"].nonzero(as_tuple=False).squeeze(-1).tolist())
                for i in env_ids:
                    # stats["succeeded"] is get_term("success"), the exact tensor
                    # RecorderManager grades EXPORT_SUCCEEDED_ONLY against
                    # (recorder_manager.py:410-416), frozen by the rig on the step this
                    # episode ended. Reading it from the rig rather than from the
                    # termination manager here is what makes the read safe for the whole
                    # of env_ids: with several envs finishing on the same step the loop is
                    # over a batch, and any later env.step() would have overwritten the
                    # live tensor for every one of them.
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
                    # Watchdog/oracle abandon: explicit env-scoped reset. The episode
                    # exports with success=False -- dropped in the default 'succeeded'
                    # mode (EXPORT_SUCCEEDED_ONLY), kept when --export_mode all is set.
                    env.reset(env_ids=torch.tensor(abandoned_ids, device=env.device))
                rig.rearm(env_ids)
    finally:
        wall_seconds = time.time() - start
        exported = env.recorder_manager.exported_successful_episode_count
        env.close()

    dones_count = sum(1 for e in episodes if e["outcome"] in ("success", "time_out"))
    abandoned_count = sum(1 for e in episodes if e["outcome"] == "abandoned")
    run_stats = {
        "attempts": attempts,
        "attempts_target": target_attempts,
        # Authoritative success count (RecorderManager's own running total),
        # NOT a sum over episodes[]["outcome"] -- see the module docstring.
        "successes": exported,
        "time_outs": dones_count - exported,
        "abandons": abandoned_count,
        "abandon_reasons": dict(abandon_reasons),
        "watchdog_trips": abandon_reasons.get("watchdog", 0),
        "control_steps": control_steps,
        # Spec section 5 gates a run at clamp_fraction <= 0.01. The denominator is env
        # control steps, not control steps, because the numerator counts one per env.
        "clamped_steps": clamped_steps,
        "clamp_fraction": clamped_steps / (control_steps * rig.num_envs) if control_steps else 0.0,
        "wall_seconds": wall_seconds,
        "sim_steps_per_sec": control_steps / wall_seconds if wall_seconds > 0 else 0.0,
        "deadline_hit": attempts < target_attempts,
        "episodes_per_env": {
            str(i): c for i, c in sorted(Counter(e["env"] for e in episodes).items())
        },
        # The oracle's planning-lookahead counters (MS1 report section 9.3), summed over
        # the run's episodes so the campaign ledger can carry them without re-reading the
        # sidecar. Both read 0 on a run where every RAISE and every drawn hover planned.
        "raise_vetoed": sum(int(e.get("raise_vetoed", 0)) for e in episodes),
        "hover_lowered": sum(int(e.get("hover_lowered", 0)) for e in episodes),
        # How many episodes actually targeted bricks out of lowest-index order. The
        # per-run audit of --randomize_pick_order: with the flag on and six bricks, the
        # lowest-index sequence occurs by chance once in 720 episodes, so a run where this
        # reads 0 shipped the fixed schedule whatever RUN_META claims.
        "off_schedule_episodes": sum(
            1 for e in episodes if list(e.get("pick_order", [])) != sorted(e.get("pick_order", []))
        ),
    }
    if args.probe:
        success_rate = exported / attempts if attempts else 0.0
        wall_sec_per_episode = wall_seconds / attempts if attempts else 0.0
        run_stats["success_rate"] = success_rate
        run_stats["wall_sec_per_episode"] = wall_sec_per_episode
        run_stats[f"extrapolated_{CAMPAIGN_TARGET_SUCCESSES}_success_wall_hours"] = (
            (CAMPAIGN_TARGET_SUCCESSES / success_rate) * wall_sec_per_episode / 3600.0
            if success_rate > 0
            else None
        )
    print("RUN_STATS " + json.dumps(run_stats), flush=True)

    meta_path = os.path.join(args.out_dir, f"{args.shard_name}.meta.json")
    with open(meta_path, "w") as f:
        json.dump({"run_meta": run_meta, "run_stats": run_stats, "episodes": episodes}, f, indent=2)

    ok = args.probe or exported > 0
    return ok, exported, attempts


try:
    ok, exported, attempts = run()
except Exception:  # noqa: BLE001 -- any failure must become BATCH_FAIL, not a crash.
    traceback.print_exc()
    ok, exported, attempts = False, 0, 0

print(f"BATCH_OK exported={exported} attempts={attempts}" if ok else "BATCH_FAIL", flush=True)
os._exit(0 if ok else 1)
