"""DAgger rollouts on the MuJoCo backend: one process = one shard of ``--num_envs`` episodes.

The policy server drives every env; the oracle rig takes an env over on a trigger and runs
one brick cycle (``maker_arm_lab.core.dagger_loop``). Only takeover frames are recorded, to
``<out_dir>/<shard_name>.hdf5``, and the payload of the mixed rollout is printed as
``DAGGER_OK {json}``. Runs under the project's own venv:

    uv run python scripts/mj_dagger_batch.py --num_envs 12 --run_seed 1 --layout_seed 2 \\
        --permutation 0,1,2,3 --episode_draws --frame_jitter --context camera \\
        --server tcp://127.0.0.1:5700 --round 1 --out_dir /tmp/dag --shard_name proc_0
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_stack"))
import policy_client
from maker_arm_lab.core import oracle
from maker_arm_lab.core.dagger_loop import TAKEOVER_MEAN_INTERVAL, rollout
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

NUM_BRICKS = 6


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument(
        "--run_seed", type=int, default=0, help="env.reset seed and the trigger rng"
    )
    parser.add_argument("--layout_seed", type=int, default=0, help="seeds sample_run_draw")
    parser.add_argument("--permutation", type=str, default="0,1,2,3", help="bin-color permutation")
    parser.add_argument("--episode_draws", action="store_true", help="per-episode scene draws")
    parser.add_argument("--frame_jitter", action="store_true", help="per-frame camera jitter")
    parser.add_argument("--spatial_scale", type=float, default=1.0)
    parser.add_argument("--context", choices=("none", "camera", "privileged"), default="none")
    parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--round", type=int, default=1, help="DAgger round: the random interval doubles per round"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="where the shard is written")
    parser.add_argument("--shard_name", type=str, default="shard")
    parser.add_argument("--out", type=str, default="", help="write the DAGGER_OK payload here too")
    parser.add_argument("--max_wall_minutes", type=float, default=90.0)
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.round < 1:
        parser.error("--round starts at 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    perm = tuple(int(v) for v in args.permutation.split(","))
    if sorted(perm) != [0, 1, 2, 3]:
        print("DAGGER_FAIL " + json.dumps({"reason": "permutation", "detail": args.permutation}))
        return 1
    policy = None
    env = None
    try:
        draw = sample_run_draw(random.Random(args.layout_seed), args.spatial_scale)
        draw["slot_colors"] = perm
        env = MjLegoSortEnv(
            args.num_envs,
            draw,
            num_bricks=NUM_BRICKS,
            episode_draws=args.episode_draws,
            frame_jitter=args.frame_jitter,
            spatial_scale=args.spatial_scale,
            context=args.context,
            export_dir=args.out_dir,
            shard_name=args.shard_name,
            export_all=True,
        )
        env.reset(seed=args.run_seed)
        rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(args.run_seed))
        policy = policy_client.ServerPolicy(args.server)
        payload = rollout(
            env,
            rig,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=perm,
            random_interval=TAKEOVER_MEAN_INTERVAL * 2 ** (args.round - 1),
            max_wall_minutes=args.max_wall_minutes,
        )
        payload["round"] = args.round
        payload["shard"] = str(Path(args.out_dir) / f"{args.shard_name}.hdf5")
    except EvalFailure as err:
        print("DAGGER_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail}), flush=True)
        return 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        print(
            "DAGGER_FAIL "
            + json.dumps({"reason": "exception", "detail": f"{err}\n{traceback.format_exc()}"}),
            flush=True,
        )
        return 1
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
    print("DAGGER_OK " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
