"""The heap outer loop: a policy drives pick cycles that a scripted selector sets up.

Phase B design 5.2. One cycle: the selector (a ``HeapOracle`` per env, the teacher's own
rule) names the target, the loop writes it into ``env.target_part`` so the overhead marker
moves, the policy drives, and the privileged grader ends the cycle. Three ends: a part that
was not stowed when the cycle began is at rest in the bin (the target, or any part the jaws
carried, rung b5's rule), the cycle spent ``CYCLE_STEP_BUDGET`` control steps, or the target
came to rest on the desk outside ``layout.in_reach`` and outside the bin, out of the jaws
(knocked out of reach; a carry crosses out-of-reach space in flight and may pause there). A failed cycle marks its cell in the selector's failure
memory.
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
import time
from dataclasses import dataclass, field

import torch

from maker_arm_lab.core import layout, oracle
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.gate import RolloutTracker, clamp_arm_actions, policy_state
from maker_arm_lab.core.heap_oracle import CANDIDATE_MAX_Z
from maker_arm_lab.core.stowed import _brick_states, per_brick_stowed
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX

CYCLE_STEP_BUDGET = 800
"""INITIAL: control steps one policy cycle may take, twice the b4 teacher's median cycle of
about 400 steps (17094 to 20677 steps over about 50 cycles, docs/ms6-bare-desk-heap.md
section 3). The phase B preflight (plan Task 15) fixes it from the corpus shards."""
CYCLE_BUDGET_FACTOR = 1.3
"""Pick cycles per part an episode may spend: the gate's own cycles-per-stowed rule."""
AT_REST_SPEED = 0.02
"""m/s: the grader's at-rest speed, ``checker.bricks_at_rest_in_bins``'s ``max_speed``."""
CONTINUOUS_CLOSE_GAP = 0.030
"""m: under the continuous gripper a commanded jaw gap below this reads as a close. The
smallest matched landing gap in the catalog is 31.8 mm (a 7.8 mm slope) and the widest close
command, on a 31.8 mm brick, about 30 mm. Triggers and duty-cycle stats read it; the metric
never does."""


def cycle_budget(k: int) -> int:
    """The pick cycles an episode of ``k`` parts may spend (phase B design 5.1: 65 at K=50)."""
    return math.ceil(CYCLE_BUDGET_FACTOR * k)


def episode_length_s(budget: int, control_hz: float = 30.0) -> float:
    """The env's episode length: every cycle of the budget at its full step budget."""
    return budget * CYCLE_STEP_BUDGET / control_hz


def grip_open_flags(gripper: torch.Tensor, gripper_mode: str) -> torch.Tensor:
    """(N,) bool: which raw gripper commands read as an open jaw.

    Binary: at or above zero, the env's contract. Continuous: a commanded gap at or above
    ``CONTINUOUS_CLOSE_GAP``; under it the jaws are closing onto something.
    """
    g = gripper.detach().to(device="cpu", dtype=torch.float32)
    if gripper_mode == "continuous":
        return g >= CONTINUOUS_CLOSE_GAP
    return g >= 0.0


def open_signal(gripper: torch.Tensor, gripper_mode: str) -> torch.Tensor:
    """The gripper column shifted so that ``>= 0`` means open in either mode, which is the
    sign convention ``RolloutTracker.record_gripper`` counts duty cycles and flips by."""
    g = gripper.detach().to(device="cpu", dtype=torch.float32)
    return g - CONTINUOUS_CLOSE_GAP if gripper_mode == "continuous" else g


@dataclass
class Cycle:
    ordinal: int  # 1-based within the episode
    target: int
    start_step: int
    stowed_before: set[int]
    takeover: int = 0  # the rig's takeover ordinal while the rig holds this cycle, else 0
    takeover_failed: bool = False


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
    cycle: Cycle, stowed_now: set[int], resting_out_of_reach: bool, step: int
) -> tuple[str, list[int]] | None:
    """Why the cycle ends now and the parts it stowed, or None while it runs. Pure.

    ``resting_out_of_reach`` is ``resting_out_of_reach()`` on the target: a carry passes
    through out-of-reach space on its way to the bin, so only a part knocked out and at rest
    ends the cycle.
    """
    new = sorted(stowed_now - cycle.stowed_before)
    if new:
        return "stowed", new
    if cycle.takeover_failed:
        return "takeover_failed", []
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
    states = [EnvState(selector=_selector(rig, i)) for i in range(num_envs)]
    for st in states:
        if st.selector.cycle_budget != cycle_budget:
            raise EvalFailure(
                "cycle_budget",
                f"the selector spends {st.selector.cycle_budget} cycles, the loop {cycle_budget}",
            )
    ordinal = [0] * num_envs  # the recorder's takeover ordinal per env
    loop_done = torch.zeros(num_envs, dtype=torch.bool)
    triggers = _TriggerState(num_envs, takeover, run_seed) if takeover is not None else None

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
                end = cycle_end(st.cycle, stowed_now, out, step)
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
            )
            env.target_part[i] = target
            if policy is None:
                _take_over(rig, env, st, i, ordinal, reason="driver")
        # 2. The policy's action for the whole batch (its temporal ensemble stays warm).
        policy_action = None
        if policy is not None:
            state = policy_state(robot.data.joint_pos).to("cpu").numpy()
            wrist = env.obs_buf["images"]["wrist_rgb"].to("cpu").numpy()
            overhead = env.obs_buf["images"]["overhead_rgb"].to("cpu").numpy()
            context = env.obs_buf["context"].to("cpu").numpy() if served_width else None
            policy_action = torch.from_numpy(policy.step(state, wrist, overhead, context=context))
            if triggers is not None:
                triggers.offer(
                    env, rig, states, tracker, policy_action, pos, ordinal, step, gripper_mode
                )
        # 3. Compose: the rig's row where it drives, the policy's where the rig is idle.
        rig_action = rig.actions().to("cpu")
        action_cpu = rig_action.clone()
        if policy_action is not None:
            idle = torch.tensor(rig.idle)
            action_cpu[idle] = policy_action[idle]
        tracker.record_gripper(open_signal(action_cpu[:, 6], gripper_mode))
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
                if st.cycle is not None and policy is None:
                    st.cycle.takeover_failed = True
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
                    _close_cycle(st, "stowed", [st.cycle.target], step + 1, None)
                else:
                    _close_cycle_unmarked(st, step + 1)
            st.done_reason = st.done_reason or ("all_stowed" if bool(success[i]) else "time_out")
            rig.rearm([i], idle=True)
            env.oracle_actor[i] = 0
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
        episodes.append(
            {
                **result.to_json(),
                "cycles": len(st.outcomes),
                "cycle_outcomes": list(st.outcomes),
                "cycle_reasons": list(st.reasons),
                "cycle_steps": list(st.steps),
                "cycle_targets": list(st.targets),
                "stowed_parts": list(st.stowed_parts),
                "stowed_families": sorted(env.parts[p].family for p in st.stowed_parts),
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


def _take_over(rig, env, st: EnvState, i: int, ordinal: list[int], *, reason: str) -> bool:
    """Hand env ``i`` to the rig on the loop's target; False when the rig declines."""
    if rig.resume(i, target=st.cycle.target):
        ordinal[i] += 1
        st.takeovers += 1
        st.cycle.takeover = ordinal[i]
        env.oracle_actor[i] = ordinal[i]
        return True
    st.cycle.takeover_failed = True
    return False


class _TriggerState:
    """Placeholder until plan Task 16 adds the takeover triggers of phase B design 5.5."""

    def __init__(self, num_envs: int, config, run_seed: int):
        raise EvalFailure("takeover", "takeover triggers arrive with the DAgger branch")
