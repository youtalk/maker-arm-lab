"""The heap outer loop: a policy drives pick cycles that a scripted selector sets up.

Phase B design 5.2. One cycle: the selector (a ``HeapOracle`` per env, the teacher's own
rule) names the target, the loop writes it into ``env.target_part`` so the overhead marker
moves, the policy drives, and the privileged grader ends the cycle. Three ends: a part that
was not stowed when the cycle began is at rest in the bin (the target, or any part the jaws
carried, rung b5's rule), the cycle spent ``CYCLE_STEP_BUDGET`` control steps, or the target
came to rest on the desk outside ``layout.in_reach`` and outside the bin, out of the jaws
(knocked out of reach; a carry crosses out-of-reach space in flight and may pause there). A
failed cycle marks its cell in the selector's failure memory. A part an earlier cycle stowed
never ends a later one, so a stowed part that settles late is not credited twice.
There is no reset between cycles. The episode ends when the selector returns None (no
candidate, or the cycle budget is spent) or every part is stowed.

With ``policy=None`` the oracle rig drives every cycle through the same takeover entry the
DAgger mode uses (``OracleRig.resume(i, target=...)``): the preflight's driver and the
loop's own end-to-end check. A takeover holds the env until the rig hands it back at
RETREAT, DONE or ABANDON, and the loop grades only while the rig is idle, so one takeover is
one loop cycle.

Like the eval loop this runs one episode per env in lockstep and reads the env through the
duck-typed surface of ``maker_arm_lab.mj.env``. Torch and stdlib only.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import torch

from maker_arm_lab.core import grasp, layout, oracle
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.dagger_loop import DECLINE_COOLDOWN, trigger_reason
from maker_arm_lab.core.eval_loop import EvalFailure, _nearest_brick_cm
from maker_arm_lab.core.gate import RolloutTracker, clamp_arm_actions, policy_state
from maker_arm_lab.core.heap_oracle import CANDIDATE_MAX_Z
from maker_arm_lab.core.stowed import _brick_states, env_heap_bin_centers, per_brick_stowed
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX, TOOL_BODY

CYCLE_STEP_BUDGET = 800
"""Control steps one policy cycle may take. The phase B preflight (plan Task 15) reads the k
corpus's 1974 teacher cycles (obs/actor over ~/data/maker-arm-ms6/k/corpus): a median of
378.5 steps, p90 572, p95 732, max 1407. 800 is kept because the median is within 10% of 400
and 800 is above the p95."""
CYCLE_BUDGET_FACTOR = 1.3
"""Pick cycles per part an episode may spend: the gate's own cycles-per-stowed rule."""
AT_REST_SPEED = 0.02
"""m/s: the grader's at-rest speed, ``checker.bricks_at_rest_in_bins``'s ``max_speed``."""
IDLE_CLOSE_GAP = grasp.OPEN_HALF_GAP
"""m: the continuous close threshold of an env with no open cycle, half the full jaw gap.

Under the continuous gripper a commanded gap below the grasp width of the cycle's target
reads as a close (``cycle_close_below``). The teacher lands at the matched gap
``2 * (landing_half_gap(w) - JAW_HOLD_BAND)`` = w + 8 mm and closes to
``2 * max(0, w / 2 - CLOSE_SQUEEZE_M)`` = w - 6 mm, so w splits every grasp's two set
points, while one fixed threshold cannot: across the catalog the landings span 15.8 to
47.8 mm and the closes 1.8 to 33.8 mm. An env with no cycle has no w. Half the full gap
keeps a fully open coast reading open and a zero command reading closed. Triggers and
duty-cycle stats read the threshold; the metric never does."""
HALF_BUDGET_NO_HOLD = "half_budget_no_hold"
"""The heap trigger of phase B design 5.5: the cycle is at half its step budget and the jaws
hold nothing."""


@dataclass(frozen=True)
class TakeoverConfig:
    """The DAgger mode (phase B design 5.5). ``random_interval`` is the mean control steps
    between random takeovers, None for trouble triggers only. An episode may spend
    ``max_per_episode_factor`` takeovers per cycle of its budget (the design's "two per cycle
    of the budget"); a loop cycle holds at most one takeover, since a takeover's end ends it."""

    random_interval: float | None
    max_per_episode_factor: float = 2.0


def cycle_budget(k: int) -> int:
    """The pick cycles an episode of ``k`` parts may spend (phase B design 5.1: 65 at K=50)."""
    return math.ceil(CYCLE_BUDGET_FACTOR * k)


def episode_length_s(budget: int, control_hz: float = 30.0) -> float:
    """The env's episode length: every cycle of the budget at its full step budget."""
    return budget * CYCLE_STEP_BUDGET / control_hz


def grip_open_flags(
    gripper: torch.Tensor, gripper_mode: str, close_below: torch.Tensor | None = None
) -> torch.Tensor:
    """(N,) bool: which raw gripper commands read as an open jaw.

    Binary: at or above zero, the env's contract; ``close_below`` is ignored. Continuous: a
    commanded gap at or above ``close_below``, the (N,) per-env threshold in meters
    (``IDLE_CLOSE_GAP`` for every env when None); under it the jaws are closing onto the part.
    """
    return open_signal(gripper, gripper_mode, close_below) >= 0.0


def open_signal(
    gripper: torch.Tensor, gripper_mode: str, close_below: torch.Tensor | None = None
) -> torch.Tensor:
    """The gripper column shifted so that ``>= 0`` means open in either mode, which is the
    sign convention ``RolloutTracker.record_gripper`` counts duty cycles and flips by.
    ``close_below`` as in ``grip_open_flags``."""
    g = gripper.detach().to(device="cpu", dtype=torch.float32)
    if gripper_mode != "continuous":
        return g
    if close_below is None:
        return g - IDLE_CLOSE_GAP
    return g - close_below.to(device="cpu", dtype=torch.float32)


def cycle_close_below(selector, scene, target: int) -> float:
    """m: the continuous close threshold of a cycle on ``target``, read once when it opens.

    The grasp width the teacher would rank first on the target, the way
    ``HeapOracle.aim_hover_at`` ranks it for a takeover. When no grasp ranks (no closing
    axis near the jaw plane), the part's smallest footprint side: the width any grasp on a
    part lying flat would close across.
    """
    mask = torch.zeros(selector.num_bricks, dtype=torch.bool)
    mask[target] = True
    ranked = selector._ranked(scene, mask)
    if ranked:
        return float(ranked[0].width)
    return float(min(selector.parts[target].size[:2]))


@dataclass
class Cycle:
    ordinal: int  # 1-based within the episode
    target: int
    start_step: int
    stowed_before: set[int]
    close_below: float = IDLE_CLOSE_GAP  # m, the continuous close threshold (cycle_close_below)
    takeover: int = 0  # the rig's takeover ordinal while the rig holds this cycle, else 0
    # "takeover_failed" or "takeover_declined": the rig's answer ends the cycle as a failure
    # (phase B design 5.5), after the stowed rule had its chance. A decline while the jaws
    # hold a part is not an answer: the cycle runs on (``_take_over``).
    takeover_end: str | None = None
    taken_over: bool = False  # one takeover per loop cycle (phase B design 5.5)


@dataclass
class EnvState:
    selector: object  # the HeapOracle that selects for this env
    cycle: Cycle | None = None
    outcomes: list[bool] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    stowed_parts: list[int] = field(default_factory=list)
    takeovers: int = 0
    takeovers_stowed: int = 0
    takeovers_failed: int = 0
    done_reason: str | None = None


def _layout_of(rig, i: int):
    layouts = getattr(rig.env, "bin_layouts", None)
    return rig.bl if layouts is None else layouts[i]


def _selector(rig, i: int):
    """A fresh HeapOracle for env ``i``, built the way ``OracleRig.resume`` builds its own."""
    return rig.oracle_factory(i, _layout_of(rig, i), oracle.nominal_draw())


def cycle_end(
    cycle: Cycle,
    stowed_now: set[int],
    resting_out_of_reach: bool,
    step: int,
    credited: set[int] = frozenset(),
) -> tuple[str, list[int]] | None:
    """Why the cycle ends now and the parts it stowed, or None while it runs. Pure.

    ``resting_out_of_reach`` is ``resting_out_of_reach()`` on the target: a carry passes
    through out-of-reach space on its way to the bin, so only a part knocked out and at rest
    ends the cycle. ``credited`` holds the parts earlier cycles of the episode stowed: a part
    still moving in the bin when this cycle opened is missing from ``stowed_before``, and it
    must not end this cycle when it settles.
    """
    new = sorted(stowed_now - cycle.stowed_before - credited)
    if new:
        return "stowed", new
    if cycle.takeover_end is not None:
        return cycle.takeover_end, []
    if step - cycle.start_step >= CYCLE_STEP_BUDGET:
        return "budget", []
    if resting_out_of_reach:
        return "out_of_reach", []
    return None


def resting_out_of_reach(pos, speed: float, held: bool, bl, stow_half_xy) -> bool:
    """A part knocked out of reach: at rest (the grader's speed rule), not between the jaws,
    down on the desk or heap (the selector's ``CANDIDATE_MAX_Z``), outside ``layout.in_reach``
    and outside the destination bin's footprint. A slow or paused carry is none of these."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if held or speed > AT_REST_SPEED or z > CANDIDATE_MAX_Z:
        return False
    cx, cy, _ = layout.heap_bin_center(bl)
    in_bin = abs(x - cx) <= float(stow_half_xy[0]) and abs(y - cy) <= float(stow_half_xy[1])
    return not in_bin and not layout.in_reach(x, y, bl)


def _close_cycle(st: EnvState, reason: str, stowed: list[int], step: int, scene) -> None:
    cycle = st.cycle
    ok = reason == "stowed"
    st.outcomes.append(ok)
    st.reasons.append(reason)
    st.steps.append(step - cycle.start_step)
    st.targets.append(cycle.target)
    st.stowed_parts.extend(stowed)
    if not ok and scene is not None:
        st.selector._mark_failure(scene, cycle.target)
    st.cycle = None


def rollout(
    env,
    rig,
    policy,
    *,
    run_seed: int,
    layout_seed: int,
    permutation: tuple[int, ...],
    cycle_budget: int,
    max_wall_minutes: float,
    takeover=None,
    log=print,
) -> dict:
    """One lockstep rollout of ``env.num_envs`` heap episodes; returns the EVAL payload.

    ``policy`` is a ``policy_client.ServerPolicy`` or None for the oracle driver.
    ``takeover`` is a ``TakeoverConfig`` (the DAgger mode, phase B design 5.5) or None.
    Raises ``EvalFailure`` on a scene, context, action-space or gripper mismatch, the wall
    clock and an incomplete batch, as the eval loop does.
    """
    num_envs = env.num_envs
    k = env.cfg.num_bricks
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    if list(arm_ids) != list(range(6)):
        raise EvalFailure("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    if getattr(env, "parts", None) is None:
        raise EvalFailure("scene", "the heap loop needs a heap env (parts=...), not six bricks")
    context_mode = getattr(env, "context_mode", "none")
    served_width = int(env.obs_buf["context"].shape[1]) if context_mode != "none" else 0
    gripper_mode = getattr(env, "gripper_mode", "binary")
    if policy is not None:
        wants = int(getattr(policy, "context_dim", 0))
        if wants != served_width:
            raise EvalFailure(
                "context",
                f"policy takes a {wants}-wide context, the env serves {served_width} ({context_mode})",
            )
        if getattr(policy, "action_space", "absolute") != "absolute":
            raise EvalFailure("action_space", "the heap loop drives absolute joint targets only")
        served_gripper = getattr(policy, "gripper", "binary")
        if served_gripper != gripper_mode:
            raise EvalFailure(
                "gripper", f"the checkpoint is {served_gripper}, the env serves {gripper_mode}"
            )
    if takeover is not None and policy is None:
        raise EvalFailure("takeover", "the takeover triggers need a policy to take over from")
    limits = robot.data.joint_pos_limits[:, arm_ids, :].to("cpu")
    tracker = RolloutTracker(num_envs, k)
    recorder = getattr(env, "recorder", None)
    rig.rearm(list(range(num_envs)), idle=True)
    env.oracle_actor = torch.zeros(num_envs, dtype=torch.int16)
    env.target_part[:] = -1
    # A policy reads the marker on the loop's target, so a takeover's frames must show that
    # marker too, not the rig's cycle part (they differ when the rig adopts a held part).
    env.marker_from_rig = policy is None
    states = [EnvState(selector=_selector(rig, i)) for i in range(num_envs)]
    for st in states:
        if st.selector.cycle_budget != cycle_budget:
            raise EvalFailure(
                "cycle_budget",
                f"the selector spends {st.selector.cycle_budget} cycles, the loop {cycle_budget}",
            )
    ordinal = [0] * num_envs  # the recorder's takeover ordinal per env
    loop_done = torch.zeros(num_envs, dtype=torch.bool)
    triggers = (
        _TriggerState(num_envs, takeover, run_seed, cycle_budget) if takeover is not None else None
    )

    started = time.monotonic()
    max_steps = int(env.max_episode_length)
    step = 0
    while step < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > max_wall_minutes:
            raise EvalFailure("wall_clock", f"exceeded {max_wall_minutes} min at step {step}")
        per_brick = per_brick_stowed(env).to("cpu")
        tracker.latch_pre_step(per_brick)
        pos, vel = _brick_states(env)
        pos, speed = pos.to("cpu"), vel.norm(dim=-1).to("cpu")
        left_mag = env.scene["finger_left_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        right_mag = env.scene["finger_right_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        held = held_brick_from_forces(left_mag, right_mag, min_force=HELD_MIN_FORCE).to("cpu")
        # 1. Grade and select on the envs the loop holds: episode live, rig idle.
        for i in range(num_envs):
            if bool(tracker.done[i]) or not rig.idle[i]:
                continue
            st = states[i]
            stowed_now = {b for b in range(k) if bool(per_brick[i, b])}
            if st.cycle is not None:
                b = st.cycle.target
                out = resting_out_of_reach(
                    pos[i, b],
                    float(speed[i, b]),
                    int(held[i]) == b,
                    _layout_of(rig, i),
                    env.stow_half_extents,
                )
                end = cycle_end(st.cycle, stowed_now, out, step, set(st.stowed_parts))
                if end is not None:
                    _close_cycle(st, end[0], end[1], step, rig.snapshot(i))
            if st.cycle is not None:
                continue
            if len(stowed_now) == k:
                loop_done[i] = True
                st.done_reason = "all_stowed"
                continue
            scene = rig.snapshot(i)
            target = st.selector.select_target(scene)
            if target is None:
                loop_done[i] = True
                st.done_reason = (
                    "cycle_budget" if st.selector.cycles >= cycle_budget else "no_candidate"
                )
                continue
            st.cycle = Cycle(
                ordinal=len(st.outcomes) + 1,
                target=target,
                start_step=step,
                stowed_before=stowed_now,
                close_below=cycle_close_below(st.selector, scene, target),
            )
            env.target_part[i] = target
            if policy is None:
                _take_over(rig, env, st, i, ordinal, reason="driver")
        close_below = torch.tensor(
            [IDLE_CLOSE_GAP if st.cycle is None else st.cycle.close_below for st in states]
        )
        # 2. The policy's action for the whole batch (its temporal ensemble stays warm).
        policy_action = None
        if policy is not None:
            # The frame the policy reads was rendered unmarked at the end of the last step,
            # and pass 1 may have moved the target since: stamp it now (phase B design 5.2).
            stamp = getattr(env, "stamp_markers", None)
            if stamp is not None:
                stamp()
            state = policy_state(robot.data.joint_pos).to("cpu").numpy()
            wrist = env.obs_buf["images"]["wrist_rgb"].to("cpu").numpy()
            overhead = env.obs_buf["images"]["overhead_rgb"].to("cpu").numpy()
            context = env.obs_buf["context"].to("cpu").numpy() if served_width else None
            policy_action = torch.from_numpy(policy.step(state, wrist, overhead, context=context))
            if triggers is not None:
                grip_open = grip_open_flags(policy_action[:, 6], gripper_mode, close_below)
                triggers.offer(env, rig, states, tracker, grip_open, held, pos, ordinal, step)
        # 3. Compose: the rig's row where it drives, the policy's where the rig is idle.
        rig_action = rig.actions().to("cpu")
        action_cpu = rig_action.clone()
        if policy_action is not None:
            idle = torch.tensor(rig.idle)
            action_cpu[idle] = policy_action[idle]
        tracker.record_gripper(open_signal(action_cpu[:, 6], gripper_mode, close_below))
        arm_clamped, overshoot = clamp_arm_actions(action_cpu[:, :6], limits)
        tracker.record_clamps(overshoot)
        clamped = torch.cat([arm_clamped, action_cpu[:, 6:7]], dim=-1)
        if step == max_steps - 1 and recorder is not None:
            # The time-out cuts a running takeover short of RETREAT: negate before the env's
            # step exports the episode (the six-brick DAgger loop's ruling P7).
            for i in range(num_envs):
                if int(env.oracle_actor[i]) > 0:
                    recorder.negate_actor(i, ordinal[i])
                    states[i].takeovers_failed += 1
                    env.oracle_actor[i] = 0
        _, _, terminated, truncated, _ = env.step(clamped.to(env.device))
        dones = (terminated | truncated).cpu()
        rig.observe(dones)
        # 4. Takeovers the rig handed back this step.
        for i in range(num_envs):
            result = rig.takeover_result[i]
            if result is None:
                continue
            st = states[i]
            if result:
                st.takeovers_stowed += 1
            else:
                st.takeovers_failed += 1
                if recorder is not None:
                    recorder.negate_actor(i, ordinal[i])
                if st.cycle is not None:
                    # Phase B design 5.5: the loop grades the takeover's cycle like any other,
                    # so a failed hand-back ends it (next pass 1, after the stowed rule).
                    st.cycle.takeover_end = "takeover_failed"
            if st.cycle is not None:
                st.cycle.takeover = 0
            rig.takeover_result[i] = None
            env.oracle_actor[i] = 0
        # 5. Episode ends: the env's own (auto-reset) and the loop's.
        success = env.termination_manager.get_term("success").to("cpu")
        newly_env = dones & ~tracker.done
        tracker.record_step(step, dones | loop_done, truncated.cpu(), success)
        for i in torch.nonzero(newly_env).flatten().tolist():
            st = states[i]
            if int(env.oracle_actor[i]) > 0:
                # A takeover the episode's end cut short: its part is in on success (the
                # six-brick DAgger loop's rule), and its frames are already exported.
                if bool(success[i]):
                    st.takeovers_stowed += 1
                else:
                    st.takeovers_failed += 1
            if st.cycle is not None:
                if bool(success[i]):
                    # Every part is in: the cycle stowed all the parts not credited before.
                    new = set(range(k)) - st.cycle.stowed_before - set(st.stowed_parts)
                    _close_cycle(st, "stowed", sorted(new), step + 1, None)
                else:
                    _close_cycle_unmarked(st, step + 1)
            st.done_reason = st.done_reason or ("all_stowed" if bool(success[i]) else "time_out")
            rig.rearm([i], idle=True)
            env.oracle_actor[i] = 0
        if recorder is not None:
            # An episode the loop ends (selector dry or cycle budget spent) never reaches the
            # env's own end: export its takeover frames here. No takeover runs on a loop-done
            # env (pass 1 sees only idle rigs), and ``end`` is a no-op on an env already ended.
            for i in torch.nonzero(loop_done).flatten().tolist():
                recorder.end(i, states[i].done_reason == "all_stowed")
        loop_done.zero_()
        step += 1
        if step % 500 == 0:
            log(
                f"[loop] step {step}/{max_steps} done={int(tracker.done.sum())}/{num_envs} "
                f"cycles={sum(len(s.outcomes) for s in states)}",
                flush=True,
            )

    wall = time.monotonic() - started
    if not tracker.all_done:
        raise EvalFailure(
            "incomplete", f"{int((~tracker.done).sum())} envs still running at {max_steps}"
        )
    episodes = []
    for i, result in enumerate(tracker.results):
        st = states[i]
        final = tracker.stowed_mask(i)
        episodes.append(
            {
                **result.to_json(),
                "cycles": len(st.outcomes),
                "cycle_outcomes": list(st.outcomes),
                "cycle_reasons": list(st.reasons),
                "cycle_steps": list(st.steps),
                "cycle_targets": list(st.targets),
                "stowed_parts": list(st.stowed_parts),
                "stowed_families": sorted(env.parts[p].family for p in range(k) if final[p]),
                "done_reason": st.done_reason,
                "takeovers": st.takeovers,
                "takeovers_stowed": st.takeovers_stowed,
                "takeovers_failed": st.takeovers_failed,
            }
        )
    steps_total = sum(e["steps"] for e in episodes)
    cycles_total = sum(e["cycles"] for e in episodes)
    stowed_total = sum(e["bricks_stowed"] for e in episodes)
    payload = {
        "run_seed": run_seed,
        "layout_seed": layout_seed,
        "permutation": [int(v) for v in permutation],
        "num_envs": num_envs,
        "num_bricks": k,
        "heap_k": k,
        "parts": [p.name for p in env.parts],
        "cycle_budget": cycle_budget,
        "cycle_step_budget": CYCLE_STEP_BUDGET,
        "policy_id": "oracle" if policy is None else policy.policy_id,
        "n_action_steps": None if policy is None else policy.n_action_steps,
        "inference": {} if policy is None else policy.inference,
        "context": context_mode,
        "gripper": gripper_mode,
        "episode_draws": bool(getattr(env.cfg, "episode_draws", False)),
        "frame_jitter": bool(getattr(env.cfg, "frame_jitter", False)),
        "spatial_scale": float(getattr(env.cfg, "spatial_scale", 1.0)),
        "episodes": episodes,
        "steps_run": step,
        "cycles_total": cycles_total,
        "stowed_total": stowed_total,
        "cycles_per_stowed": (cycles_total / stowed_total) if stowed_total else None,
        "takeovers": sum(e["takeovers"] for e in episodes),
        "takeovers_stowed": sum(e["takeovers_stowed"] for e in episodes),
        "takeovers_failed": sum(e["takeovers_failed"] for e in episodes),
        "triggers": {} if triggers is None else dict(triggers.counts),
        "takeovers_declined": 0 if triggers is None else triggers.declined,
        "clamp_fraction": (
            sum(e["clamp_steps"] for e in episodes) / steps_total if steps_total else 0.0
        ),
        "wall_sec": round(wall, 1),
        "wall_sec_per_episode": round(wall / num_envs, 1),
    }
    return payload


def _close_cycle_unmarked(st: EnvState, step: int) -> None:
    """An episode the env ended (time-out) closes its open cycle without a scene to mark."""
    cycle = st.cycle
    st.outcomes.append(False)
    st.reasons.append("episode_end")
    st.steps.append(step - cycle.start_step)
    st.targets.append(cycle.target)
    st.cycle = None


def _take_over(
    rig, env, st: EnvState, i: int, ordinal: list[int], *, reason: str, holding: bool = False
) -> bool:
    """Hand env ``i`` to the rig on the loop's target; False when the rig declines.

    A decline with nothing held ends the cycle as a failure (phase B design 5.5: "the cycle
    ends as a failure, and the loop moves on"); ``_close_cycle`` marks the selector's failure
    memory. A decline while the jaws hold a part (``holding``, the loop's held read) keeps the
    six-brick behavior: the cycle runs on and the caller's cooldown spaces the next trigger.
    ``OracleRig.resume`` declines most of a correct carry (RULING P32: a part above
    ``LIFT_HEIGHT - LIFT_MIN_RISE``), and ending the cycle there would spend it and mark the
    failure memory at a point in the air."""
    if rig.resume(i, target=st.cycle.target):
        st.cycle.taken_over = True
        ordinal[i] += 1
        st.takeovers += 1
        st.cycle.takeover = ordinal[i]
        env.oracle_actor[i] = ordinal[i]
        return True
    if not holding:
        st.cycle.taken_over = True
        st.cycle.takeover_end = "takeover_declined"
    return False


class _TriggerState:
    """Per-env trigger counters and the offer of a takeover: the six-brick DAgger loop's logic
    on the loop's cycles (the MS4 four triggers, the random draw, the decline cooldown) plus
    the half-budget trigger (phase B design 5.5). ``offer`` runs once per step before the
    actions are composed, on the envs the policy holds. Caps (design 5.5): one takeover per
    loop cycle, ``max_per_episode_factor * cycle_budget`` per episode."""

    def __init__(self, num_envs: int, config: TakeoverConfig, run_seed: int, cycle_budget: int):
        self.config = config
        self.max_per_episode = math.floor(config.max_per_episode_factor * cycle_budget)
        self.rng = random.Random(run_seed)
        self.random_p = 0.0 if config.random_interval is None else 1.0 / config.random_interval
        self.since_open = [0] * num_envs
        self.closed_no_pinch = [0] * num_envs
        self.cooldown = [0] * num_envs
        self.grip_open_prev: torch.Tensor | None = None
        self.counts: Counter = Counter()
        self.declined = 0

    def offer(self, env, rig, states, tracker, grip_open, held, pos, ordinal, step) -> None:
        """``grip_open`` is the policy's (N,) open read on the loop's per-env close threshold
        and ``held`` the loop's per-step held read, so the triggers and the grader agree."""
        robot = env.scene["robot"]
        tool = robot.find_bodies(TOOL_BODY)[0][0]
        nearest_cm = _nearest_brick_cm(
            pos,
            robot.data.body_pos_w[:, tool, :2].to("cpu"),
            held,
            env_heap_bin_centers(env, "cpu").to("cpu"),
            torch.as_tensor(env.stow_half_extents).to("cpu"),
        )
        for i in range(env.num_envs):
            st = states[i]
            if bool(tracker.done[i]) or not rig.idle[i] or st.cycle is None:
                continue
            open_i = bool(grip_open[i])
            closed_now = (
                self.grip_open_prev is not None and bool(self.grip_open_prev[i]) and not open_i
            )
            self.since_open[i] = self.since_open[i] + 1 if open_i else 0
            no_pinch = not open_i and int(held[i]) < 0
            self.closed_no_pinch[i] = self.closed_no_pinch[i] + 1 if no_pinch else 0
            if self.cooldown[i] > 0:
                self.cooldown[i] -= 1
                continue
            reason = trigger_reason(
                grip_open=open_i,
                closed_now=closed_now,
                held=int(held[i]),
                nearest_cm=nearest_cm[i],
                pick_point=rig.pick_point(i).tolist(),
                since_open=self.since_open[i],
                closed_no_pinch=self.closed_no_pinch[i],
                random_draw=self.rng.random(),
                random_p=self.random_p,
            )
            if (
                reason is None
                and int(held[i]) < 0
                and step - st.cycle.start_step >= CYCLE_STEP_BUDGET // 2
            ):
                reason = HALF_BUDGET_NO_HOLD
            if reason is None or st.cycle.taken_over or st.takeovers >= self.max_per_episode:
                continue
            # Reset on both branches: an answered trigger must not fire again off the same count.
            self.since_open[i] = 0
            self.closed_no_pinch[i] = 0
            if _take_over(rig, env, st, i, ordinal, reason=reason, holding=int(held[i]) >= 0):
                self.counts[reason] += 1
            else:
                # The decline cooldown stays (phase B design 5.5). A decline on a held part
                # leaves the cycle running and the cooldown spaces the next trigger on it; a
                # decline with nothing held ends the cycle, and the cooldown spaces the next
                # cycle's first trigger from a scene the rig just refused.
                self.declined += 1
                self.cooldown[i] = DECLINE_COOLDOWN
        self.grip_open_prev = grip_open
