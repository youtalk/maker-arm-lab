"""Gate-evaluation client: one process = one run-level draw = N lockstep episodes.

Runs MakerArm-LegoSort-Eval-v0 with --num_envs parallel envs, all reset together, one
episode per env (MS2 report section 5.1). Actions come from the policy server over ZMQ
(lerobot_stack/policy_client.py, the policy_protocol wire format) or, with
--random_actions, from a uniform random source (GPU harness smoke, no server).

The rollout itself -- the (7,) `observation.state`, the Rust clamp on the six arm targets,
the checker-direct grading through core.gate.RolloutTracker, the deferred wrong-bin and grasp
instruments and the payload -- is `maker_arm_lab.core.eval_loop.rollout`, shared with the
MuJoCo client `scripts/mj_eval_policy.py`, so the two backends measure by one set of rules.
This file only builds the Isaac environment and honours the token contract.

Deliberately NOT here (MS2 spec section 1.2 item 8): the unstick escape, replay bundles,
preview video, the wrist fold, and the close-event diagnostics. The optional --trace_dir
dumps the (7,) state and raw/clamped action per step for distribution comparisons.

Usage (from the repo root; VIRTUAL_ENV bootstrap as in generate_dataset.py):

  $ISAACLAB_PATH/isaaclab.sh -p scripts/eval_policy.py --headless --enable_cameras \\
      --num_envs 12 --run_seed 1 --layout_seed 2 --permutation 0,1,2,3 \\
      --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json

Prints exactly one terminal token line: EVAL_SMOKE_OK / EVAL_OK / EVAL_FAIL, and exits
through os._exit after simulation_app.close() (the Kit teardown otherwise hangs).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_stack"))
import policy_client
from maker_arm_lab.core.eval_loop import EvalFailure, out_stem, rollout

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Eval-v0")
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--run_seed", type=int, default=0, help="env.reset seed for the whole process")
parser.add_argument("--layout_seed", type=int, default=0, help="seeds sample_run_draw")
parser.add_argument("--permutation", type=str, default="0,1,2,3", help="bin-color permutation")
parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
parser.add_argument("--out", type=str, default="", help="write the EVAL_OK payload JSON here too")
parser.add_argument(
    "--random_actions", action="store_true", help="GPU smoke: no server, uniform random actions"
)
parser.add_argument("--smoke_steps", type=int, default=200, help="steps in --random_actions mode")
parser.add_argument("--max_wall_minutes", type=float, default=60.0)
parser.add_argument(
    "--trace_dir",
    type=str,
    default="",
    help="dump one npz per env with the per-step (7,) state and raw/clamped action",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

for _seed_arg in ("run_seed", "layout_seed"):
    if not (0 <= getattr(args, _seed_arg) < 2**32):
        parser.error(f"--{_seed_arg} is outside numpy's seed range (0 <= seed < 2**32)")
if args.random_actions and args.trace_dir:
    parser.error("--trace_dir records a policy rollout; it has no meaning with --random_actions")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import sample_run_draw
from maker_arm_lab.tasks.manager_based.lego_sort.eval_cfg import LegoSortEvalEnvCfg


def fail(reason: str, detail: str) -> None:
    print("EVAL_FAIL " + json.dumps({"reason": reason, "detail": detail}), flush=True)
    simulation_app.close()
    os._exit(1)


def build_cfg() -> LegoSortEvalEnvCfg:
    perm = tuple(int(v) for v in args.permutation.split(","))
    if sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(
            f"--permutation must be a permutation of 0,1,2,3, got {args.permutation!r}"
        )
    draw = sample_run_draw(random.Random(args.layout_seed))
    draw["slot_colors"] = perm
    cfg = LegoSortEvalEnvCfg(cameras=True, **draw)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    return cfg


def run() -> int:
    cfg = build_cfg()
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset(seed=args.run_seed)
    policy = None if args.random_actions else policy_client.ServerPolicy(args.server)
    try:
        payload = rollout(
            env,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=tuple(int(v) for v in args.permutation.split(",")),
            max_wall_minutes=args.max_wall_minutes,
            random_actions=args.random_actions,
            smoke_steps=args.smoke_steps,
            trace_dir=args.trace_dir,
            out_stem=out_stem(args.out, args.run_seed),
        )
    except EvalFailure as err:
        fail(err.reason, err.detail)
    if args.random_actions:
        print("EVAL_SMOKE_OK " + json.dumps(payload), flush=True)
    else:
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
        print("EVAL_OK " + json.dumps(payload), flush=True)
    if policy is not None:
        policy.close()
    env.close()
    return 0


try:
    code = run()
except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
    import traceback

    fail("exception", f"{err}\n{traceback.format_exc()}")
simulation_app.close()
os._exit(code)
