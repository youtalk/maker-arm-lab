"""DAgger rollouts on a heap: one process = one shard of ``--num_envs`` episodes (phase B
design 5.5).

The heap twin of ``scripts/mj_dagger_batch.py``: the policy server drives every env inside
``maker_arm_lab.core.loop``, and the oracle rig takes an env over on a trigger and runs one
cycle on the loop's own target. Only takeover frames are recorded, to
``<out_dir>/<shard_name>.hdf5``, and the payload is printed as ``DAGGER_OK {json}``. Runs
under the project's own venv:

    uv run python scripts/mj_dagger_heap.py --num_envs 5 --heap_k 50 --parts_split train \\
        --run_seed 20261510 --layout_seed 20261610 --episode_draws --frame_jitter \\
        --context camera_box --gripper continuous --server tcp://127.0.0.1:5700 --round 1 \\
        --out_dir ~/data/maker-arm-ms6/l/dagger/r1 --shard_name proc_0
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
from maker_arm_lab.core.batch_loop import PARTS_SPLITS, heap_parts
from maker_arm_lab.core.dagger_loop import TAKEOVER_MEAN_INTERVAL
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.loop import (
    CYCLE_STEP_BUDGET,
    HELD_NO_RISE_STEPS,
    TakeoverConfig,
    cycle_budget,
    episode_length_s,
    rollout,
)
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


class _TokenEmittingParser(argparse.ArgumentParser):
    """A parser whose usage errors still honour the DAGGER_OK / DAGGER_FAIL contract."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print("DAGGER_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        raise SystemExit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument(
        "--run_seed", type=int, default=0, help="env.reset seed, the rig and trigger rng"
    )
    parser.add_argument("--layout_seed", type=int, default=0, help="seeds sample_run_draw")
    parser.add_argument("--permutation", type=str, default="0,1,2,3", help="bin-color permutation")
    parser.add_argument("--episode_draws", action="store_true", help="per-episode scene draws")
    parser.add_argument("--frame_jitter", action="store_true", help="per-frame camera jitter")
    parser.add_argument("--spatial_scale", type=float, default=1.0)
    parser.add_argument(
        "--context",
        choices=("none", "camera", "camera_box", "camera_box_target"),
        default="camera_box",
    )
    parser.add_argument(
        "--gripper",
        choices=("binary", "continuous"),
        default="binary",
        help="the env's gripper contract; must match the checkpoint's handshake",
    )
    parser.add_argument("--heap_k", type=int, default=50, help="parts in the heap (> 0)")
    parser.add_argument("--parts_seed", type=int, default=20260915)
    parser.add_argument("--parts_split", choices=PARTS_SPLITS, default="train")
    parser.add_argument(
        "--cycle_budget",
        type=int,
        default=None,
        help="pick cycles per episode; the default is ceil(1.3 * K) (phase B design 5.1)",
    )
    parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--round", type=int, default=1, help="DAgger round: the random interval doubles per round"
    )
    parser.add_argument(
        "--trigger",
        choices=("random", "failure"),
        default="random",
        help="random: the round's mean interval plus the trouble triggers (campaign l and m); "
        "failure: no random draw, and a takeover when a held part has not risen for "
        "loop.HELD_NO_RISE_STEPS steps (loop.HELD_NO_RISE) or at half the budget with no lift",
    )
    parser.add_argument(
        "--select_at_release",
        action="store_true",
        help="select the next target when the release opens the jaws (loop.release_ready)",
    )
    parser.add_argument(
        "--replan_on_switch",
        action="store_true",
        help="send the server a per-env replan flag on the step a target row is written",
    )
    parser.add_argument(
        "--squeeze_floor",
        action="store_true",
        help="cap the policy's closing gap command at the teacher's close set point "
        "(loop.floored_gap); the policy's frames are not recorded, so this only changes what "
        "the policy executes",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="where the shard is written")
    parser.add_argument("--shard_name", type=str, default="shard")
    parser.add_argument("--out", type=str, default="", help="write the DAGGER_OK payload here too")
    parser.add_argument(
        "--max_wall_minutes",
        type=float,
        default=None,
        help="the wall-clock cap; the default allows a generous 0.15 s per step of the episode "
        "(every cycle of the budget at its full step budget)",
    )
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.heap_k <= 0:
        parser.error("--heap_k must be positive")
    if args.round < 1:
        parser.error("--round starts at 1")
    if args.cycle_budget is None:
        args.cycle_budget = cycle_budget(args.heap_k)
    if args.max_wall_minutes is None:
        args.max_wall_minutes = args.cycle_budget * CYCLE_STEP_BUDGET * 0.15 / 60.0
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    policy = None
    env = None
    close_error = None
    try:
        perm = tuple(int(v) for v in args.permutation.split(","))
        if sorted(perm) != [0, 1, 2, 3]:
            raise EvalFailure("permutation", args.permutation)
        draw = sample_run_draw(random.Random(args.layout_seed), args.spatial_scale)
        draw["slot_colors"] = perm
        parts = heap_parts(args)
        env = MjLegoSortEnv(
            args.num_envs,
            draw,
            num_bricks=len(parts),
            parts=parts,
            episode_draws=args.episode_draws,
            frame_jitter=args.frame_jitter,
            spatial_scale=args.spatial_scale,
            context=args.context,
            episode_length_s=episode_length_s(args.cycle_budget),
            export_dir=args.out_dir,
            shard_name=args.shard_name,
            export_all=True,
            gripper_mode=args.gripper,
        )
        env.reset(seed=args.run_seed)
        rig_rng = random.Random(args.run_seed)

        def factory(i, bl, episode_draw):
            return HeapOracle(
                bl,
                env.parts,
                int(env.destination_bins[i]),
                episode_draw,
                rng=rig_rng,
                cycle_budget=args.cycle_budget,
            )

        rig = OracleRig(
            env, env.cfg.bin_layout, oracle.OracleNoise(), rig_rng, oracle_factory=factory
        )
        policy = policy_client.ServerPolicy(args.server)
        if args.trigger == "random":
            takeover = TakeoverConfig(
                random_interval=TAKEOVER_MEAN_INTERVAL * 2 ** (args.round - 1)
            )
        else:
            takeover = TakeoverConfig(random_interval=None, held_no_rise_steps=HELD_NO_RISE_STEPS)
        payload = rollout(
            env,
            rig,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=perm,
            cycle_budget=args.cycle_budget,
            max_wall_minutes=args.max_wall_minutes,
            takeover=takeover,
            select_at_release=args.select_at_release,
            replan_on_switch=args.replan_on_switch,
            squeeze_floor=args.squeeze_floor,
        )
        payload["round"] = args.round
        payload["trigger"] = args.trigger
        payload["parts_split"] = args.parts_split
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
        try:
            if policy is not None:
                policy.close()
            if env is not None:
                env.close()
        except Exception as err:  # noqa: BLE001 -- the token contract
            # The token is printed AFTER the try/except/finally, not here: a body that
            # already returned printed its own token and that return wins over this block,
            # so exactly one token is printed on either path (mj_dagger_batch.py's rule).
            close_error = f"{err}\n{traceback.format_exc()}"
    if close_error is not None:
        print("DAGGER_FAIL " + json.dumps({"reason": "close", "detail": close_error}), flush=True)
        return 1
    try:
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
    except Exception as err:  # noqa: BLE001 -- shard is already on disk; still print a token
        print("DAGGER_FAIL " + json.dumps({"reason": "write_out", "detail": str(err)}), flush=True)
        return 1
    print("DAGGER_OK " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
