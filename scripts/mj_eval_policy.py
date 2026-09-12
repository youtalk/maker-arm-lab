"""Gate-evaluation client on the MuJoCo backend: one process = one run-level draw.

The MuJoCo twin of ``scripts/eval_policy.py``: the same flags (minus Isaac's AppLauncher
ones), the same policy server over ZMQ, the same grading (``maker_arm_lab.core.eval_loop``)
and the same ``EVAL_OK`` / ``EVAL_SMOKE_OK`` / ``EVAL_FAIL`` token contract, so
``scripts/run_gate_eval.py --backend mujoco`` drives it exactly as it drives the Isaac client.
The scene is the datagen scene (cameras, run-level draw, reset-level randomization), as the
Isaac ``MakerArm-LegoSort-Eval-v0`` cfg inherits it. Runs under the project's own venv:

    uv run python scripts/mj_eval_policy.py --num_envs 12 --run_seed 1 --layout_seed 2 \\
        --permutation 0,1,2,3 --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json

``--viewer`` opens the MuJoCo GUI on env 0 and grades the rollout as usual, for watching a
trained policy instead of reading its payload. The window and the offscreen camera renderer
share one GL backend, so that run goes through glfw and needs a display:

    MUJOCO_GL=glfw uv run python scripts/mj_eval_policy.py --viewer --num_envs 1 \\
        --run_seed 1 --layout_seed 2 --permutation 0,1,2,3 --server tcp://127.0.0.1:5555
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

# --viewer needs an on-screen GL context; the offscreen camera renderer shares the backend,
# so the whole process moves to glfw for that run. Read off argv because the backend is fixed
# at the first mujoco import, which happens below.
os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_stack"))
import policy_client
from maker_arm_lab.core.eval_loop import EvalFailure, out_stem, rollout
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv

NUM_BRICKS = 6


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Eval-mujoco")
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument(
        "--run_seed", type=int, default=0, help="env.reset seed for the whole process"
    )
    parser.add_argument("--layout_seed", type=int, default=0, help="seeds sample_run_draw")
    parser.add_argument("--permutation", type=str, default="0,1,2,3", help="bin-color permutation")
    parser.add_argument("--episode_draws", action="store_true", help="per-episode scene draws")
    parser.add_argument("--frame_jitter", action="store_true", help="per-frame camera jitter")
    parser.add_argument(
        "--spatial_scale",
        type=float,
        default=1.0,
        help="multiplier on the camera-pose and bin_x jitter of every draw (1.0 = training) "
        "(the run draw only; per-episode draws stay at 1.0)",
    )
    parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument(
        "--out", type=str, default="", help="write the EVAL_OK payload JSON here too"
    )
    parser.add_argument(
        "--random_actions", action="store_true", help="smoke: no server, uniform random actions"
    )
    parser.add_argument(
        "--smoke_steps", type=int, default=200, help="steps in --random_actions mode"
    )
    parser.add_argument("--max_wall_minutes", type=float, default=60.0)
    parser.add_argument(
        "--trace_dir",
        type=str,
        default="",
        help="dump one npz per env with the per-step (7,) state and raw/clamped action",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the MuJoCo GUI on env 0 (use --num_envs 1; slows the rollout to render rate)",
    )
    parser.add_argument("--headless", action="store_true", help="accepted for Isaac-command parity")
    parser.add_argument("--enable_cameras", action="store_true", help="accepted for parity")
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.random_actions and args.trace_dir:
        parser.error(
            "--trace_dir records a policy rollout; it has no meaning with --random_actions"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    perm = tuple(int(v) for v in args.permutation.split(","))
    if sorted(perm) != [0, 1, 2, 3]:
        print("EVAL_FAIL " + json.dumps({"reason": "permutation", "detail": args.permutation}))
        return 1
    policy = None
    env = None
    viewer = None
    try:
        draw = sample_run_draw(random.Random(args.layout_seed), args.spatial_scale)
        draw["slot_colors"] = perm
        env = MjLegoSortEnv(
            args.num_envs,
            draw,
            num_bricks=NUM_BRICKS,
            cameras=not args.random_actions,
            episode_draws=args.episode_draws,
            frame_jitter=args.frame_jitter,
        )
        env.reset(seed=args.run_seed)
        if args.viewer:
            import mujoco.viewer

            # env 0 only: the batch is a Python loop over one model/data pair per env, and the
            # viewer holds one pair. The auto-reset inside step() writes the same MjData, so
            # the window follows the whole rollout, episode after episode.
            viewer = mujoco.viewer.launch_passive(env.models[0], env.datas[0])
            stepped = env.step

            def env_step(action):
                out = stepped(action)
                viewer.sync()
                return out

            env.step = env_step
        policy = None if args.random_actions else policy_client.ServerPolicy(args.server)
        payload = rollout(
            env,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=perm,
            max_wall_minutes=args.max_wall_minutes,
            random_actions=args.random_actions,
            smoke_steps=args.smoke_steps,
            trace_dir=args.trace_dir,
            out_stem=out_stem(args.out, args.run_seed),
        )
    except EvalFailure as err:
        print("EVAL_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail}), flush=True)
        return 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        print(
            "EVAL_FAIL "
            + json.dumps({"reason": "exception", "detail": f"{err}\n{traceback.format_exc()}"}),
            flush=True,
        )
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
    if args.random_actions:
        print("EVAL_SMOKE_OK " + json.dumps(payload), flush=True)
    else:
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
        print("EVAL_OK " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
