"""Gate-evaluation client: one process = one run-level draw = N lockstep episodes.

Runs MakerArm-LegoSort-Eval-v0 with --num_envs parallel envs, all reset together, one
episode per env (MS2 spec section 5.1). Actions come from the policy server over ZMQ
(lerobot_stack/policy_client.py, the policy_protocol wire format) or, with
--random_actions, from a uniform random source (GPU harness smoke, no server).

Per control step: build the (7,) `observation.state` the policy was trained on
(core.gate.policy_state: six URDF arm joints + the left jaw's opening fraction), send it
with the two 224x224 camera frames, receive the (7,) action, pass the six arm targets
through core.gate.clamp_arm_actions against the articulation's limits, hand the gripper
sign to the env's binary gripper term, and step. Grading is checker-direct:
mdp.terminations.per_brick_stowed feeds a core.gate.RolloutTracker every step and the
success flag is the env's own `success` termination, so driver bookkeeping is never the
metric. Wrong-bin deposits and grasp attempts are the same deferred, lower-bound
instruments openyam-lab's harness carried (their read rules are in the MS2 report).

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
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lerobot_stack"))
import policy_client

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
import torch

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.core.checker import containing_bin, held_brick_from_forces
from maker_arm_lab.core.gate import (
    RolloutTracker,
    clamp_arm_actions,
    grade_grasp_attempts,
    policy_state,
)
from maker_arm_lab.core.trace import RolloutTrace
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import sample_run_draw
from maker_arm_lab.tasks.manager_based.lego_sort.eval_cfg import LegoSortEvalEnvCfg
from maker_arm_lab.tasks.manager_based.lego_sort.mdp.observations import (
    brick_color_ids,
)
from maker_arm_lab.tasks.manager_based.lego_sort.mdp.terminations import (
    _bin_centers_cached,
    _bin_half_extents_cached,
    _brick_states,
    _env_bin_layout,
    per_brick_stowed,
)

WRONG_BIN_SETTLE_STEPS = 30
"""Control steps (1 s at 30 Hz) to wait after a release before reading the bin. A released
brick is still falling the instant the jaws open, and `containing_bin` only names a bin
once the brick is inside it and slow. Inherited from openyam-lab unchanged."""

GRASP_SETTLE_STEPS = 20
"""Control steps to wait after a close command before asking whether a brick is held.
Under the oracle's CLOSE_STEPS = 45 jaw-stroke budget (core/oracle.py); inherited from
openyam-lab and not re-measured on the Maker Arm's gripper, so the grasp rate it grades is a
relative instrument (MS2 report read rules)."""

HELD_MIN_FORCE = 1.0
"""Newtons on the weaker jaw for `held_brick_from_forces` to call a brick held: the loose
ATTRIBUTION threshold ("which brick was between the jaws"), deliberately under the quality
gate core.checker.PINCH_MIN_FORCE = 2.0 that grade_grasp_attempts scores strictly."""


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
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    # policy_state() and convert_to_lerobot.py both take columns 0..5 as the arm and 6 as the
    # left jaw; if the articulation ever orders its joints differently, fail here rather
    # than evaluate the policy on a permuted state.
    if list(arm_ids) != list(range(6)):
        fail("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    limits = robot.data.joint_pos_limits[:, arm_ids, :].to("cpu")  # (N,6,2)
    tracker = RolloutTracker(args.num_envs, cfg.num_bricks)
    policy = None if args.random_actions else policy_client.ServerPolicy(args.server)
    trace = RolloutTrace(args.num_envs) if args.trace_dir else None
    device = str(env.device)
    bin_centers = _bin_centers_cached(_env_bin_layout(env), device)
    bin_half_extents = _bin_half_extents_cached(device)
    target_bins = brick_color_ids(cfg.num_bricks)  # (B,) cpu long
    pending_releases: list[tuple[int, int, int]] = []  # (env, brick, due_step)
    pending_grasps: list[tuple[int, int]] = []  # (env, due_step)
    releases_queued = 0
    grip_open_prev = None

    started = time.monotonic()
    max_steps = args.smoke_steps if args.random_actions else int(env.max_episode_length)
    step_index = 0
    while step_index < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > args.max_wall_minutes:
            fail("wall_clock", f"exceeded {args.max_wall_minutes} min at step {step_index}")
        per_brick = per_brick_stowed(env)
        tracker.latch_pre_step(per_brick)
        state = policy_state(robot.data.joint_pos).to("cpu").numpy()
        if args.random_actions:
            span = limits[..., 1] - limits[..., 0]
            arm = limits[..., 0] + torch.rand(args.num_envs, 6) * span
            grip = torch.where(torch.rand(args.num_envs, 1) < 0.5, -1.0, 1.0)
            action_cpu = torch.cat([arm, grip], dim=-1)
        else:
            wrist = env.obs_buf["images"]["wrist_rgb"].to("cpu").numpy()
            overhead = env.obs_buf["images"]["overhead_rgb"].to("cpu").numpy()
            action_cpu = torch.from_numpy(policy.step(state, wrist, overhead))
        tracker.record_gripper(action_cpu[:, 6])
        grip_open = action_cpu[:, 6] >= 0.0
        # One contact read per step, pre-step, shared by the release detection and the
        # grasp grading below.
        left_mag = env.scene["finger_left_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        right_mag = env.scene["finger_right_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        held = held_brick_from_forces(left_mag, right_mag, min_force=HELD_MIN_FORCE).to("cpu")
        if grip_open_prev is not None:
            # Releases: a brick held on an open-after-close is queued and read back a second
            # later. A (env, brick) already awaiting its check is not re-queued, so gripper
            # chatter counts one physical deposit once. Counted for LIVE episodes only.
            released = grip_open & ~grip_open_prev & (held >= 0)
            queued = {(e, b) for e, b, _ in pending_releases}
            for i in torch.nonzero(released).flatten().tolist():
                key = (i, int(held[i]))
                if key not in queued:
                    pending_releases.append((*key, step_index + WRONG_BIN_SETTLE_STEPS))
                    if not bool(tracker.done[i]):
                        releases_queued += 1
            # Grasp attempts: one close-after-open per env, graded after the settle window.
            # An env with an attempt still pending does not open a second one (one approach
            # to one pose is the unit; chatter inside the window is the same attempt).
            awaiting = {e for e, _ in pending_grasps}
            opened = [
                i
                for i in torch.nonzero(~grip_open & grip_open_prev).flatten().tolist()
                if i not in awaiting
            ]
            if opened:
                pending_grasps.extend((i, step_index + GRASP_SETTLE_STEPS) for i in opened)
                tracker.record_grasp_attempt(torch.tensor(opened, dtype=torch.long))
        grip_open_prev = grip_open
        due = [entry for entry in pending_releases if entry[2] <= step_index]
        if due:
            pending_releases = [entry for entry in pending_releases if entry[2] > step_index]
            env_ids = torch.tensor([e for e, _, _ in due], dtype=torch.long)
            brick_ids = torch.tensor([b for _, b, _ in due], dtype=torch.long)
            pos, vel = _brick_states(env)
            landed = containing_bin(pos, vel, bin_centers, bin_half_extents).to("cpu")
            settled = landed[env_ids, brick_ids]
            # -1 is a brick that reached no bin at all: a failed cycle, not a mis-sort.
            wrong = (settled >= 0) & (settled != target_bins[brick_ids])
            if wrong.any():
                tracker.record_wrong_bin(env_ids[wrong])
        due_grasps = [entry for entry in pending_grasps if entry[1] <= step_index]
        if due_grasps:
            pending_grasps = [entry for entry in pending_grasps if entry[1] > step_index]
            graded = torch.tensor([e for e, _ in due_grasps], dtype=torch.long)
            loose, strict = grade_grasp_attempts(left_mag, right_mag, graded, HELD_MIN_FORCE)
            tracker.record_grasp_outcome(graded, loose, strict)
        arm_clamped, overshoot = clamp_arm_actions(action_cpu[:, :6], limits)
        tracker.record_clamps(overshoot)
        clamped_action = torch.cat([arm_clamped, action_cpu[:, 6:7]], dim=-1)
        if trace is not None:
            trace.append(state, action_cpu.numpy(), clamped_action.numpy(), (~tracker.done).numpy())
        _, _, terminated, truncated, _ = env.step(clamped_action.to(env.device))
        success = env.termination_manager.get_term("success")
        tracker.record_step(step_index, terminated, truncated, success)
        step_index += 1
        if step_index % 500 == 0:
            print(
                f"[eval] step {step_index}/{max_steps} done={int(tracker.done.sum())}/"
                f"{args.num_envs} stowed_latched={int(tracker._latched.sum())}",
                flush=True,
            )

    wall = time.monotonic() - started
    if args.random_actions:
        payload = {"steps_run": step_index, "num_envs": args.num_envs, "wall_sec": round(wall, 1)}
        print("EVAL_SMOKE_OK " + json.dumps(payload), flush=True)
    else:
        if not tracker.all_done:
            fail("incomplete", f"{int((~tracker.done).sum())} envs still running at {max_steps}")
        episodes = [r.to_json() for r in tracker.results]
        steps_total = sum(e["steps"] for e in episodes)
        payload = {
            "run_seed": args.run_seed,
            "layout_seed": args.layout_seed,
            "permutation": [int(v) for v in args.permutation.split(",")],
            "num_envs": args.num_envs,
            "num_bricks": cfg.num_bricks,
            "policy_id": policy.policy_id,
            "n_action_steps": policy.n_action_steps,
            # Echoed off the handshake: what the server served with, not what was asked.
            "inference": policy.inference,
            "episodes": episodes,
            "steps_run": step_index,
            "clamp_fraction": (
                sum(e["clamp_steps"] for e in episodes) / steps_total if steps_total else 0.0
            ),
            # Run-level: releases detected while the releasing episode was live, the
            # denominator wrong_bin_deposits is a numerator of.
            "releases_queued": releases_queued,
            "wall_sec": round(wall, 1),
            "wall_sec_per_episode": round(wall / args.num_envs, 1),
        }
        if trace is not None:
            stem = Path(args.out).stem if args.out else f"eval_seed{args.run_seed}"
            payload["traces"] = [str(p) for p in trace.save(args.trace_dir, stem)]
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
