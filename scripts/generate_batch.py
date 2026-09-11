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
import traceback

from isaaclab.app import AppLauncher

from maker_arm_lab.core.batch_loop import (
    add_batch_arguments,
    build_run,
    make_induction,
    run_batch,
    run_meta,
    run_stats,
    write_meta,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Datagen-v0")
# Every other flag is shared with scripts/mj_generate_batch.py through core.batch_loop.
add_batch_arguments(parser)
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
from isaaclab.managers import DatasetExportMode

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import (
    LegoSortDatagenEnvCfg,
)
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

# --export_mode's string choices, resolved against the isaaclab enum here (only
# importable post-AppLauncher, see above) rather than at argparse-definition time.
EXPORT_MODES = {
    "succeeded": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    "all": DatasetExportMode.EXPORT_ALL,
}


def run() -> tuple[bool, int, int]:
    """Returns ``(ok, exported, attempts)``."""
    os.makedirs(args.out_dir, exist_ok=True)
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

    induction = make_induction(args)
    rig = OracleRig(env, bl, noise, random.Random(args.run_seed), induction=induction)
    meta = run_meta(
        args,
        draw,
        noise,
        induction,
        num_bricks=env_cfg.num_bricks,
        export_mode=export_mode.name,
        backend="isaac",
        task=args.task,
    )
    print("RUN_META " + json.dumps(meta), flush=True)
    counters = run_batch(env, rig, args.attempts, args.max_wall_minutes)
    stats = run_stats(
        counters, target_attempts=args.attempts, num_envs=args.num_envs, probe=args.probe
    )
    print("RUN_STATS " + json.dumps(stats), flush=True)
    write_meta(args.out_dir, args.shard_name, meta, stats, counters["episodes"])
    ok = args.probe or counters["exported"] > 0
    return ok, counters["exported"], counters["attempts"]


try:
    ok, exported, attempts = run()
except Exception:  # noqa: BLE001 -- any failure must become BATCH_FAIL, not a crash.
    traceback.print_exc()
    ok, exported, attempts = False, 0, 0

print(f"BATCH_OK exported={exported} attempts={attempts}" if ok else "BATCH_FAIL", flush=True)
os._exit(0 if ok else 1)
