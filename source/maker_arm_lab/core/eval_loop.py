"""The gate-evaluation rollout loop, shared by both simulator backends.

``scripts/eval_policy.py`` (Isaac Lab) and ``scripts/mj_eval_policy.py`` (MuJoCo) build an
environment and hand it here with a policy. Everything that decides what a rollout MEASURES
-- the deferred grasp and wrong-bin instruments, the tracker, the payload fields -- lives in
this one place, so a per-brick number from either backend is read by the same rules (the
MS2 report's read rules apply to both). The env surface is the duck-typed one described in
``maker_arm_lab.mj.env``.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from maker_arm_lab.core.checker import containing_bin, held_brick_from_forces
from maker_arm_lab.core.gate import (
    RolloutTracker,
    clamp_arm_actions,
    grade_grasp_attempts,
    policy_state,
)
from maker_arm_lab.core.stowed import (
    _bin_centers_cached,
    _bin_half_extents_cached,
    _brick_states,
    _env_bin_layout,
    brick_color_ids,
    per_brick_stowed,
)
from maker_arm_lab.core.trace import RolloutTrace
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX

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


class EvalFailure(Exception):
    """A rollout that cannot produce a number; the scripts print it as EVAL_FAIL."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def rollout(
    env,
    policy,
    *,
    run_seed: int,
    layout_seed: int,
    permutation: tuple[int, ...],
    max_wall_minutes: float,
    random_actions: bool = False,
    smoke_steps: int = 200,
    trace_dir: str = "",
    out_stem: str = "",
    log=print,
) -> dict:
    """One lockstep rollout of ``env.num_envs`` episodes; returns the EVAL payload.

    With ``random_actions`` the policy is ignored, uniform random actions run for
    ``smoke_steps`` and the smaller EVAL_SMOKE payload comes back. Raises ``EvalFailure``
    on the joint-order check, the wall clock and an incomplete batch, as the Isaac script's
    ``fail`` did.
    """
    num_envs = env.num_envs
    num_bricks = env.cfg.num_bricks
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    # policy_state() and convert_to_lerobot.py both take columns 0..5 as the arm and 6 as the
    # left jaw; if the articulation ever orders its joints differently, fail here rather
    # than evaluate the policy on a permuted state.
    if list(arm_ids) != list(range(6)):
        raise EvalFailure("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    limits = robot.data.joint_pos_limits[:, arm_ids, :].to("cpu")  # (N,6,2)
    tracker = RolloutTracker(num_envs, num_bricks)
    trace = RolloutTrace(num_envs) if trace_dir else None
    device = str(env.device)
    bin_centers = _bin_centers_cached(_env_bin_layout(env), device)
    bin_half_extents = _bin_half_extents_cached(device)
    target_bins = brick_color_ids(num_bricks)  # (B,) cpu long
    pending_releases: list[tuple[int, int, int]] = []  # (env, brick, due_step)
    pending_grasps: list[tuple[int, int]] = []  # (env, due_step)
    releases_queued = 0
    grip_open_prev = None

    started = time.monotonic()
    max_steps = smoke_steps if random_actions else int(env.max_episode_length)
    step_index = 0
    while step_index < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > max_wall_minutes:
            raise EvalFailure("wall_clock", f"exceeded {max_wall_minutes} min at step {step_index}")
        per_brick = per_brick_stowed(env)
        tracker.latch_pre_step(per_brick)
        state = policy_state(robot.data.joint_pos).to("cpu").numpy()
        if random_actions:
            span = limits[..., 1] - limits[..., 0]
            arm = limits[..., 0] + torch.rand(num_envs, 6) * span
            grip = torch.where(torch.rand(num_envs, 1) < 0.5, -1.0, 1.0)
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
            log(
                f"[eval] step {step_index}/{max_steps} done={int(tracker.done.sum())}/"
                f"{num_envs} stowed_latched={int(tracker._latched.sum())}",
                flush=True,
            )

    wall = time.monotonic() - started
    if random_actions:
        return {"steps_run": step_index, "num_envs": num_envs, "wall_sec": round(wall, 1)}
    if not tracker.all_done:
        raise EvalFailure(
            "incomplete", f"{int((~tracker.done).sum())} envs still running at {max_steps}"
        )
    episodes = [r.to_json() for r in tracker.results]
    steps_total = sum(e["steps"] for e in episodes)
    payload = {
        "run_seed": run_seed,
        "layout_seed": layout_seed,
        "permutation": [int(v) for v in permutation],
        "num_envs": num_envs,
        "num_bricks": num_bricks,
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
        "wall_sec_per_episode": round(wall / num_envs, 1),
    }
    if trace is not None:
        stem = out_stem or f"eval_seed{run_seed}"
        payload["traces"] = [str(p) for p in trace.save(trace_dir, stem)]
    return payload


def out_stem(out: str, run_seed: int) -> str:
    return Path(out).stem if out else f"eval_seed{run_seed}"
