"""DAgger rollouts with the oracle as expert (MS4 c2 design 6.3).

A policy drives every env. On a trigger the oracle rig takes the env over, runs one brick
cycle from the state the policy reached, and hands it back. The env records only the takeover
frames (``env.oracle_actor``), so the converter's ``--dagger`` keeps cycles the policy never
drove and every kept frame has a full chunk horizon of oracle motion. Like the eval loop this
runs one episode per env in lockstep and reads the env through the duck-typed surface of
``maker_arm_lab.mj.env``.
"""

from __future__ import annotations

import random
import time
from collections import Counter

import torch

from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.eval_loop import EvalFailure, _nearest_brick_cm
from maker_arm_lab.core.gate import RolloutTracker, clamp_arm_actions, policy_state
from maker_arm_lab.core.stowed import (
    _bin_half_extents_cached,
    _brick_states,
    env_bin_centers,
    per_brick_stowed,
)
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX, TOOL_BODY

TAKEOVER_MEAN_INTERVAL = 300
"""Mean control steps between random takeovers in round 1; doubled per round (design 6.3)."""
NO_CLOSE_STEPS = 400
"""Trouble: the jaws have been open this long without a close."""
CLOSE_FAR_CM = 2.0
"""Trouble: a close with no brick within this distance of the tool."""
CLOSED_NO_PINCH_STEPS = 60
"""Trouble: the jaws have been closed this long with nothing pinched."""
WORKSPACE = ((0.0, 0.50), (-0.35, 0.35), (-0.03, 0.45))
"""Trouble: the grasp center outside this base-frame box (x, y, z ranges in m)."""
MAX_TAKEOVERS = 12
"""Takeovers per episode at most: two per brick."""
DECLINE_COOLDOWN = 60
"""Control steps before a declined env is offered to the oracle again.

A decline is a NORMAL outcome, not an error: the oracle answers DONE or ABANDON at entry once
the desk is clear, and ruling P32 declines a brick the policy is carrying above LIFT_HEIGHT.
Three of the five triggers do not read the counters below -- ``workspace`` in particular has
no counter and no open-to-closed edge to wait on, so it fires on every idle step the pick
point sits outside the box -- so without a cooldown a declined env rebuilds an ``EnvOracle``,
snapshots the scene, recomputes the stowed set and wipes ``rig.stats[i]`` on every remaining
step of the episode."""


def inside_workspace(point) -> bool:
    return all(lo <= float(v) <= hi for v, (lo, hi) in zip(point, WORKSPACE))


def trigger_reason(
    *,
    grip_open: bool,
    closed_now: bool,
    held: int,
    nearest_cm: float,
    pick_point,
    since_open: int,
    closed_no_pinch: int,
    random_draw: float,
    random_p: float,
) -> str | None:
    """Why the oracle takes this env over now, or None. Pure, so every row has a test."""
    if random_draw < random_p:
        return "random"
    if grip_open and since_open >= NO_CLOSE_STEPS:
        return "no_close"
    if closed_now and held < 0 and not 0.0 <= nearest_cm <= CLOSE_FAR_CM:
        return "close_far"
    if not grip_open and held < 0 and closed_no_pinch >= CLOSED_NO_PINCH_STEPS:
        return "closed_no_pinch"
    if not inside_workspace(pick_point):
        return "workspace"
    return None


def rollout(
    env,
    rig,
    policy,
    *,
    run_seed: int,
    layout_seed: int,
    permutation: tuple[int, ...],
    random_interval: float | None,
    max_wall_minutes: float,
    log=print,
) -> dict:
    """One lockstep DAgger rollout of ``env.num_envs`` episodes; returns the DAGGER payload."""
    num_envs = env.num_envs
    num_bricks = env.cfg.num_bricks
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    if list(arm_ids) != list(range(6)):
        raise EvalFailure("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    context_mode = getattr(env, "context_mode", "none")
    served_width = int(env.obs_buf["context"].shape[1]) if context_mode != "none" else 0
    wants = int(getattr(policy, "context_dim", 0))
    if wants != served_width:
        raise EvalFailure(
            "context",
            f"policy takes a {wants}-wide context, the env serves {served_width} ({context_mode})",
        )
    limits = robot.data.joint_pos_limits[:, arm_ids, :].to("cpu")
    tracker = RolloutTracker(num_envs, num_bricks)
    device = str(env.device)
    bin_half_extents = _bin_half_extents_cached(device)
    tool = robot.find_bodies(TOOL_BODY)[0][0]
    rng = random.Random(run_seed)
    random_p = 0.0 if random_interval is None else 1.0 / random_interval
    recorder = getattr(env, "recorder", None)
    rig.rearm(list(range(num_envs)), idle=True)
    env.oracle_actor = torch.zeros(num_envs, dtype=torch.int16)
    since_open = [0] * num_envs
    closed_no_pinch = [0] * num_envs
    ordinal = [0] * num_envs
    takeovers = [0] * num_envs
    cooldown = [0] * num_envs
    declined = 0
    triggers: Counter = Counter()
    stowed_takeovers = 0
    failed_takeovers = 0
    takeovers_failed_unnegated = 0
    grip_open_prev = None

    started = time.monotonic()
    max_steps = int(env.max_episode_length)
    step_index = 0
    while step_index < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > max_wall_minutes:
            raise EvalFailure("wall_clock", f"exceeded {max_wall_minutes} min at step {step_index}")
        tracker.latch_pre_step(per_brick_stowed(env))
        bin_centers = env_bin_centers(env, device)
        state = policy_state(robot.data.joint_pos).to("cpu").numpy()
        wrist = env.obs_buf["images"]["wrist_rgb"].to("cpu").numpy()
        overhead = env.obs_buf["images"]["overhead_rgb"].to("cpu").numpy()
        context = env.obs_buf["context"].to("cpu").numpy() if served_width else None
        # Every env asks the policy every step, so its temporal ensemble stays warm through a
        # takeover and the hand-back starts from a full window.
        policy_action = torch.from_numpy(policy.step(state, wrist, overhead, context=context))
        left_mag = env.scene["finger_left_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        right_mag = env.scene["finger_right_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        held = held_brick_from_forces(left_mag, right_mag, min_force=HELD_MIN_FORCE).to("cpu")
        pos, _ = _brick_states(env)
        nearest_cm = _nearest_brick_cm(
            pos.to("cpu"),
            robot.data.body_pos_w[:, tool, :2].to("cpu"),
            held,
            bin_centers.to("cpu"),
            bin_half_extents.to("cpu"),
        )
        grip_open = policy_action[:, 6] >= 0.0
        for i in range(num_envs):
            if bool(tracker.done[i]) or not rig.idle[i]:
                continue
            open_i = bool(grip_open[i])
            closed_now = grip_open_prev is not None and bool(grip_open_prev[i]) and not open_i
            since_open[i] = since_open[i] + 1 if open_i else 0
            no_pinch = not open_i and int(held[i]) < 0
            closed_no_pinch[i] = closed_no_pinch[i] + 1 if no_pinch else 0
            if cooldown[i] > 0:
                # The counters above still run through a cooldown: they describe what the
                # policy is doing, not what the oracle answered.
                cooldown[i] -= 1
                continue
            reason = trigger_reason(
                grip_open=open_i,
                closed_now=closed_now,
                held=int(held[i]),
                nearest_cm=nearest_cm[i],
                pick_point=rig.pick_point(i).tolist(),
                since_open=since_open[i],
                closed_no_pinch=closed_no_pinch[i],
                random_draw=rng.random(),
                random_p=random_p,
            )
            if reason is None or takeovers[i] >= MAX_TAKEOVERS:
                continue
            # Reset on BOTH branches: a trigger that has been answered -- taken or declined --
            # must not fire again off the same stale count on the very next step.
            since_open[i] = 0
            closed_no_pinch[i] = 0
            if rig.resume(i):
                takeovers[i] += 1
                ordinal[i] += 1
                env.oracle_actor[i] = ordinal[i]
                triggers[reason] += 1
            else:
                declined += 1
                cooldown[i] = DECLINE_COOLDOWN
        grip_open_prev = grip_open
        rig_action = rig.actions().to("cpu")
        idle = torch.tensor(rig.idle)
        action_cpu = rig_action.clone()
        action_cpu[idle] = policy_action[idle]
        tracker.record_gripper(action_cpu[:, 6])
        arm_clamped, overshoot = clamp_arm_actions(action_cpu[:, :6], limits)
        tracker.record_clamps(overshoot)
        clamped_action = torch.cat([arm_clamped, action_cpu[:, 6:7]], dim=-1)
        if step_index == max_steps - 1 and recorder is not None:
            # The time-out cuts every running takeover short of its RETREAT: those cycles
            # did not stow their brick, and the frames must be negated before the env's step
            # exports the episode. The env's own step call below still stashes a frame for
            # THIS step under the (unchanged) positive ordinal unless the actor is also
            # zeroed here -- without that, one positive-actor frame of a timed-out takeover
            # survives into the corpus as a one-frame "kept" cycle (ruling P7).
            for i in range(num_envs):
                if int(env.oracle_actor[i]) > 0:
                    recorder.negate_actor(i, ordinal[i])
                    env.oracle_actor[i] = 0
        _, _, terminated, truncated, _ = env.step(clamped_action.to(env.device))
        success = env.termination_manager.get_term("success")
        tracker.record_step(step_index, terminated, truncated, success)
        dones = (terminated | truncated).cpu()
        rig.observe(dones)
        for i in range(num_envs):
            result = rig.takeover_result[i]
            if result is None:
                continue
            if result:
                stowed_takeovers += 1
            else:
                failed_takeovers += 1
                if recorder is not None:
                    recorder.negate_actor(i, ordinal[i])
            rig.takeover_result[i] = None
            env.oracle_actor[i] = 0
        done_ids = torch.nonzero(dones).flatten().tolist()
        if done_ids:
            for i in done_ids:
                if int(env.oracle_actor[i]) > 0:
                    # A takeover that ended with the episode never reached RETREAT. Success
                    # means its brick is in; the time-out case was negated before the step.
                    if bool(success[i]):
                        stowed_takeovers += 1
                    else:
                        failed_takeovers += 1
                        # This branch cannot negate the frames it counts: env.step() above
                        # already called recorder.end(i, ...) and flushed this env's buffer, so
                        # the positive-actor frames of this cycle are already on disk (ruling
                        # P7's defect class, one step too late to fix here). The invariant that
                        # is meant to keep this branch unreachable: a slot's FIRST episode can
                        # only truncate on the global last step (episode_length_buf starts at 0
                        # on env.reset() and max_steps is env.max_episode_length itself), which
                        # the time-out block above already negates and zeros before env.step()
                        # runs. A LATER episode in the same slot resets episode_length_buf to 0
                        # again, so its own truncation lands max_episode_length steps after that
                        # reset -- past max_steps, outside this rollout's window. Success is
                        # handled above. If this still fires, something broke that invariant, so
                        # it is counted separately instead of raised: a mid-rollout crash costs
                        # hours of GPU time, and the count surfaces the corpus contamination in
                        # the campaign ledger instead of hiding it.
                        takeovers_failed_unnegated += 1
            # The env auto-reset these; the rig must not carry a takeover into the next
            # episode, and nothing of it is recorded (one episode per env, as in the eval loop).
            rig.rearm(done_ids, idle=True)
            for i in done_ids:
                env.oracle_actor[i] = 0
        step_index += 1
        if step_index % 500 == 0:
            log(
                f"[dagger] step {step_index}/{max_steps} done={int(tracker.done.sum())}/"
                f"{num_envs} takeovers={sum(takeovers)}",
                flush=True,
            )

    wall = time.monotonic() - started
    if not tracker.all_done:
        raise EvalFailure(
            "incomplete", f"{int((~tracker.done).sum())} envs still running at {max_steps}"
        )
    episodes = [r.to_json() for r in tracker.results]
    steps_total = sum(e["steps"] for e in episodes)
    return {
        "run_seed": run_seed,
        "layout_seed": layout_seed,
        "permutation": [int(v) for v in permutation],
        "num_envs": num_envs,
        "num_bricks": num_bricks,
        "policy_id": policy.policy_id,
        "n_action_steps": policy.n_action_steps,
        "inference": policy.inference,
        "context": context_mode,
        "episode_draws": bool(getattr(env.cfg, "episode_draws", False)),
        "frame_jitter": bool(getattr(env.cfg, "frame_jitter", False)),
        "spatial_scale": float(getattr(env.cfg, "spatial_scale", 1.0)),
        "random_interval": random_interval,
        "episodes": episodes,
        "steps_run": step_index,
        "takeovers": sum(takeovers),
        "takeovers_per_env": takeovers,
        "triggers": dict(triggers),
        "takeovers_declined": declined,
        "takeovers_stowed": stowed_takeovers,
        "takeovers_failed": failed_takeovers,
        "takeovers_failed_unnegated": takeovers_failed_unnegated,
        "clamp_fraction": (
            sum(e["clamp_steps"] for e in episodes) / steps_total if steps_total else 0.0
        ),
        "wall_sec": round(wall, 1),
        "wall_sec_per_episode": round(wall / num_envs, 1),
    }
