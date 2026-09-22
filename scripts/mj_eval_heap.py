"""Heap gate client on the MuJoCo backend: one process = one run-level draw (phase B design 5.3).

The heap twin of ``scripts/mj_eval_policy.py``: the same policy server over ZMQ and the same
``EVAL_OK`` / ``EVAL_FAIL`` token contract, with ``maker_arm_lab.core.loop`` as the rollout,
so ``scripts/run_gate_eval.py --backend mujoco --heap_k K`` drives it. Runs under the
project's own venv:

    uv run python scripts/mj_eval_heap.py --num_envs 5 --heap_k 50 --parts_split heldout \\
        --run_seed 20261400 --layout_seed 20261410 --permutation 0,1,2,3 --episode_draws \\
        --frame_jitter --context camera_box --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json

``--oracle_driver`` replaces the policy with the oracle rig on every cycle: the preflight's
driver and the loop's end-to-end check, no server and no cameras needed.
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
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.loop import CYCLE_STEP_BUDGET, cycle_budget, episode_length_s, rollout
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


class _TokenEmittingParser(argparse.ArgumentParser):
    """A parser whose usage errors still honour the EVAL_OK / EVAL_FAIL contract."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print("EVAL_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        raise SystemExit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--run_seed", type=int, default=0, help="env.reset seed and the rig rng")
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
    parser.add_argument("--heap_k", type=int, default=50, help="parts in the heap (> 0)")
    parser.add_argument("--parts_seed", type=int, default=20260915)
    parser.add_argument("--parts_split", choices=PARTS_SPLITS, default="heldout")
    parser.add_argument(
        "--cycle_budget",
        type=int,
        default=None,
        help="pick cycles per episode; the default is ceil(1.3 * K) (phase B design 5.1)",
    )
    parser.add_argument(
        "--gripper",
        choices=("binary", "continuous"),
        default="binary",
        help="the env's gripper contract; must match the checkpoint's handshake",
    )
    parser.add_argument("--server", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--out", type=str, default="", help="write the EVAL_OK payload here too")
    parser.add_argument(
        "--oracle_driver", action="store_true", help="the rig drives every cycle; no server"
    )
    parser.add_argument(
        "--max_wall_minutes",
        type=float,
        default=None,
        help="the wall-clock cap; the default allows a generous 0.15 s per step of the episode "
        "(every cycle of the budget at its full step budget)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="",
        help="dump env 0's wrist and overhead policy frames as PNGs here, every "
        "loop.FRAME_EVERY steps of each cycle (forensics; needs the policy server)",
    )
    args = parser.parse_args(argv)
    for name in ("run_seed", "layout_seed"):
        if not (0 <= getattr(args, name) < 2**32):
            parser.error(f"--{name} is outside numpy's seed range (0 <= seed < 2**32)")
    if args.heap_k <= 0:
        parser.error("--heap_k must be positive: this client evaluates heap episodes only")
    if args.cycle_budget is None:
        args.cycle_budget = cycle_budget(args.heap_k)
    if args.max_wall_minutes is None:
        args.max_wall_minutes = args.cycle_budget * CYCLE_STEP_BUDGET * 0.15 / 60.0
    return args


def frame_writer(frames_dir: str):
    """``loop.rollout``'s ``on_frame``: ``c<ordinal>_s<step>_<camera>.png`` under ``frames_dir``."""
    from PIL import Image

    root = Path(frames_dir)
    root.mkdir(parents=True, exist_ok=True)

    def write(ordinal: int, step: int, wrist, overhead) -> None:
        for name, frame in (("wrist", wrist), ("overhead", overhead)):
            Image.fromarray(frame).save(root / f"c{ordinal:03d}_s{step:04d}_{name}.png")

    return write


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
            cameras=not args.oracle_driver,
            episode_draws=args.episode_draws,
            frame_jitter=args.frame_jitter,
            spatial_scale=args.spatial_scale,
            context=args.context,
            episode_length_s=episode_length_s(args.cycle_budget),
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
        policy = None if args.oracle_driver else policy_client.ServerPolicy(args.server)
        payload = rollout(
            env,
            rig,
            policy,
            run_seed=args.run_seed,
            layout_seed=args.layout_seed,
            permutation=perm,
            cycle_budget=args.cycle_budget,
            max_wall_minutes=args.max_wall_minutes,
            on_frame=frame_writer(args.frames_dir) if args.frames_dir else None,
        )
        payload["parts_split"] = args.parts_split
        payload["parts_seed"] = args.parts_seed
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
        print("EVAL_FAIL " + json.dumps({"reason": "close", "detail": close_error}), flush=True)
        return 1
    try:
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
    except Exception as err:  # noqa: BLE001 -- the rollout already ran; still print a token
        print("EVAL_FAIL " + json.dumps({"reason": "write_out", "detail": str(err)}), flush=True)
        return 1
    print("EVAL_OK " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
