"""Gate-evaluation rollout bookkeeping (design spec section 2.3, Eval-v0).

Torch-only, like core.checker: an eval script delegates here so the gate
arithmetic has CPU unit tests. One RolloutTracker tracks one lockstep
rollout: N parallel envs, one episode each (an eval process is 5 envs x 1
episode; envs that finish keep stepping harmlessly until all are done).

ManagerBasedRLEnv.step() auto-resets a terminated env before returning, so
the brick state that earned the termination is unreadable afterwards. The
tracker latches per-brick stowed state at the top of every control step; on
done, a successful env is finalized at num_bricks by definition and a failed
one from the latch (at most one 30 Hz step stale -- a brick both arriving
and settling in the final step is the only unmeasured case). The rule this
structure encodes, carried over from openyam-lab: core.checker is the
metric, driver bookkeeping never is.
"""

from dataclasses import asdict, dataclass, field

import torch

from .checker import PINCH_MIN_FORCE, held_brick_from_forces


@dataclass
class EpisodeResult:
    env_index: int
    success: bool
    bricks_stowed: int
    num_bricks: int
    steps: int
    clamp_steps: int
    gripper_open_steps: int = 0
    gripper_transitions: int = 0
    gripper_band_steps: int = 0
    first_close_step: int = -1
    wrong_bin_deposits: int = 0
    grasp_attempts: int = 0
    grasp_successes: int = 0
    grasp_successes_strict: int = 0
    grasp_attempts_unresolved: int = 0
    clamp_steps_per_joint: list = field(default_factory=lambda: [0] * 6)
    clamp_overshoot_max_per_joint: list = field(default_factory=lambda: [0.0] * 6)

    def to_json(self) -> dict:
        return asdict(self)


class RolloutTracker:
    def __init__(self, num_envs: int, num_bricks: int):
        self.num_envs = num_envs
        self.num_bricks = num_bricks
        self.done = torch.zeros(num_envs, dtype=torch.bool)
        self.results: list[EpisodeResult | None] = [None] * num_envs
        self._latched = torch.zeros(num_envs, num_bricks, dtype=torch.bool)
        self._clamp_steps = torch.zeros(num_envs, dtype=torch.long)
        self._clamp_steps_joint = torch.zeros(num_envs, 6, dtype=torch.long)
        self._clamp_overshoot_max = torch.zeros(num_envs, 6)
        self._gripper_open = torch.zeros(num_envs, dtype=torch.long)
        self._gripper_band = torch.zeros(num_envs, dtype=torch.long)
        self._gripper_flips = torch.zeros(num_envs, dtype=torch.long)
        self._gripper_prev: torch.Tensor | None = None
        self._gripper_step = 0
        self._first_close = torch.full((num_envs,), -1, dtype=torch.long)
        self._wrong_bin = torch.zeros(num_envs, dtype=torch.long)
        self._grasp_attempts = torch.zeros(num_envs, dtype=torch.long)
        self._grasp_successes = torch.zeros(num_envs, dtype=torch.long)
        self._grasp_successes_strict = torch.zeros(num_envs, dtype=torch.long)
        self._grasp_pending = torch.zeros(num_envs, dtype=torch.long)

    def latch_pre_step(self, per_brick: torch.Tensor) -> None:
        active = ~self.done
        self._latched[active] = per_brick.detach().to(device="cpu", dtype=torch.bool)[active]

    def record_clamps(self, overshoot: torch.Tensor) -> None:
        """Accumulate clamp statistics from the (N,6) signed overshoot.

        Takes the overshoot rather than a boolean so the two can never
        disagree, and so `clamp_steps` can be attributed to a joint. Attribution
        is the whole point: a pick-and-place workspace sits comfortably inside
        the limits on joints 1/2/3/5, while the v1 dataset itself reaches the
        stops on joint4 and joint6 -- so "clamping" means something completely
        different depending on which joint is doing it, and the old any-joint
        boolean could not tell those apart.
        """
        over = overshoot.detach().to(device="cpu", dtype=torch.float32)
        active = ~self.done
        clamped = over != 0.0
        self._clamp_steps[active] += clamped.any(dim=-1).to(torch.long)[active]
        self._clamp_steps_joint[active] += clamped.to(torch.long)[active]
        self._clamp_overshoot_max[active] = torch.maximum(
            self._clamp_overshoot_max[active], over.abs()[active]
        )

    def record_gripper(self, gripper: torch.Tensor) -> None:
        """Accumulate duty-cycle stats from the raw gripper column, shape (N,).

        Convention matches clamp_arm_actions and the env's binary gripper
        term: g >= 0 commands open. The column is the raw server output --
        the gripper is never clamped. A transition is an open/closed flip
        between consecutive recorded steps; the first step has no
        predecessor and cannot flip.

        `first_close_step` is the call index of the first `g < 0` this env
        sees while active -- a policy that never closes the gripper never
        even attempted a grasp, which the stow-count alone cannot tell apart
        from a grasp that closed on the wrong brick. Latched: only the FIRST
        close is recorded, so a later close/reopen/close cannot overwrite it.
        """
        g = gripper.detach().to(device="cpu", dtype=torch.float32)
        open_now = g >= 0.0
        active = ~self.done
        self._gripper_open += (open_now & active).to(torch.long)
        self._gripper_band += ((g.abs() < 0.5) & active).to(torch.long)
        if self._gripper_prev is not None:
            self._gripper_flips += ((open_now != self._gripper_prev) & active).to(torch.long)
        self._gripper_prev = open_now
        first_closing_now = ~open_now & active & (self._first_close < 0)
        self._first_close[first_closing_now] = self._gripper_step
        self._gripper_step += 1

    def record_wrong_bin(self, env_ids: torch.Tensor) -> None:
        """Increment the per-env wrong-bin-deposit counter for `env_ids`.

        `env_ids` may repeat within one call (two bricks landing in the wrong
        bin on the same step) -- index_add_ accumulates duplicates correctly,
        unlike a plain fancy-index += which only keeps the last write.

        The resulting count is a LOWER BOUND on an episode's wrong-bin deposits,
        by two mechanisms on the caller's side (the eval script):

        - A release has to settle before the bin it reached can be named, so a
          release inside the final WRONG_BIN_SETTLE_STEPS of a rollout never
          reaches its check and is dropped. Under-counting is the safe direction
          here: the wrong-bin ratio is read as evidence of weak color
          conditioning, and a metric that could only over-report would invert
          that reading.
        - A queued release whose env has already terminated still fires, and is
          graded against the AUTO-RESET second episode's brick positions, so it
          can increment a counter it did not earn. That is harmless only because
          record_step() below builds each EpisodeResult exactly once, on the
          terminating step, and never rebuilds it -- the increment lands in
          `self._wrong_bin` after that env's result has already been frozen, so
          nothing reads it. Anything that later re-finalizes a result, or keeps
          reading `_wrong_bin` after done, would turn this into an over-count.
        """
        ids = env_ids.detach().to(device="cpu", dtype=torch.long)
        self._wrong_bin.index_add_(0, ids, torch.ones_like(ids))

    def record_grasp_attempt(self, env_ids: torch.Tensor) -> None:
        """Open one PENDING grasp attempt per entry of `env_ids`.

        A grasp attempt is one open -> close gripper transition; whether it
        gripped anything is unknowable at that instant, so the attempt is parked
        here and graded later by record_grasp_outcome(). `env_ids` may repeat
        within a call, so index_add_ rather than a fancy-index += (same reason as
        record_wrong_bin).

        The three counters this pair maintains partition every attempt:
        `grasp_attempts` counts the attempts that were GRADED, `grasp_successes`
        the graded ones that held a brick, and `grasp_attempts_unresolved` the
        ones still pending when the episode was finalized -- an attempt opened
        too near the end of an episode for its settle window to close. Keeping
        the third apart is the point: it is neither a success nor a failure, and
        folding it into either silently biases `grasp_successes / grasp_attempts`
        -- the hole that openyam-lab found in its own wrong-bin counter, where
        14 of 90 deposits were structurally ungraded and were nonetheless left
        in the denominator. That is an OPENYAM measurement, carried over as the
        design rationale for keeping the third counter; nothing here has been
        measured on the Maker Arm.

        Like record_wrong_bin this is deliberately NOT gated on ~done, so an
        attempt opened on an env's terminating step still reaches that env's
        EpisodeResult (the caller must run before record_step, which is what
        makes that true). Increments for an env that finished on an EARLIER step
        are inert for the same reason as there: record_step builds each result
        exactly once and never rebuilds it.
        """
        ids = env_ids.detach().to(device="cpu", dtype=torch.long)
        self._grasp_pending.index_add_(0, ids, torch.ones_like(ids))

    def record_grasp_outcome(
        self, env_ids: torch.Tensor, held: torch.Tensor, held_strict: torch.Tensor
    ) -> None:
        """Grade one pending attempt per entry of `env_ids`; the two masks say how.

        `held[k]` is True when the jaws of env `env_ids[k]` were holding a brick
        when the attempt's settle window came due. Each entry moves one attempt
        out of the pending pool and into the graded pool, so a caller must never
        grade more attempts than it opened -- doing so drives
        `grasp_attempts_unresolved` negative rather than failing loudly, which is
        why the pairing lives in one place in the eval loop.

        `held_strict[k]` is the same question at the codebase's QUALITY pinch
        threshold rather than its attribution one (core.checker's
        PINCH_MIN_FORCE vs the eval script's looser attribution threshold). Both
        are recorded because the two differ only on marginal, brushing contacts
        -- exactly the "gripped nothing" cycles this metric exists to count --
        so the pair is a threshold-sensitivity read on the headline rate that
        would otherwise cost a second GPU rollout to obtain. A strict success
        is a success by construction (the strict threshold is the higher one), so
        `grasp_successes_strict <= grasp_successes` always.
        """
        ids = env_ids.detach().to(device="cpu", dtype=torch.long)
        ok = held.detach().to(device="cpu", dtype=torch.bool).to(torch.long)
        ok_strict = held_strict.detach().to(device="cpu", dtype=torch.bool).to(torch.long)
        if ok.shape != ids.shape or ok_strict.shape != ids.shape:
            raise ValueError(
                f"env_ids {tuple(ids.shape)}, held {tuple(ok.shape)} and held_strict "
                f"{tuple(ok_strict.shape)} must match"
            )
        ones = torch.ones_like(ids)
        self._grasp_pending.index_add_(0, ids, -ones)
        self._grasp_attempts.index_add_(0, ids, ones)
        self._grasp_successes.index_add_(0, ids, ok)
        self._grasp_successes_strict.index_add_(0, ids, ok_strict)

    def record_step(self, step_index: int, terminated, truncated, success) -> None:
        term = terminated.detach().to(device="cpu", dtype=torch.bool)
        trunc = truncated.detach().to(device="cpu", dtype=torch.bool)
        succ = success.detach().to(device="cpu", dtype=torch.bool)
        newly = (term | trunc) & ~self.done
        for i in torch.nonzero(newly).flatten().tolist():
            ok = bool(succ[i])
            self.results[i] = EpisodeResult(
                env_index=i,
                success=ok,
                bricks_stowed=self.num_bricks if ok else int(self._latched[i].sum()),
                num_bricks=self.num_bricks,
                steps=step_index + 1,
                clamp_steps=int(self._clamp_steps[i]),
                gripper_open_steps=int(self._gripper_open[i]),
                gripper_transitions=int(self._gripper_flips[i]),
                gripper_band_steps=int(self._gripper_band[i]),
                first_close_step=int(self._first_close[i]),
                wrong_bin_deposits=int(self._wrong_bin[i]),
                grasp_attempts=int(self._grasp_attempts[i]),
                grasp_successes=int(self._grasp_successes[i]),
                grasp_successes_strict=int(self._grasp_successes_strict[i]),
                # Snapshot, not a running total: whatever is still awaiting its
                # settle window at THIS instant is what this episode leaves
                # ungraded, and the result is never rebuilt afterwards.
                grasp_attempts_unresolved=int(self._grasp_pending[i]),
                clamp_steps_per_joint=[int(v) for v in self._clamp_steps_joint[i]],
                clamp_overshoot_max_per_joint=[float(v) for v in self._clamp_overshoot_max[i]],
            )
        self.done |= newly

    @property
    def all_done(self) -> bool:
        return bool(self.done.all())


def grade_grasp_attempts(
    left_mag: torch.Tensor,
    right_mag: torch.Tensor,
    env_ids: torch.Tensor,
    min_force: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grade grasp attempts from one contact read: (loose, strict) masks over `env_ids`.

    Both masks answer "were the jaws of this env holding a brick", the loose one
    at the caller's attribution threshold and the strict one at the codebase's
    quality gate, PINCH_MIN_FORCE. They are computed from ONE pair of force
    magnitudes so the two gradings are the same attempt scored twice rather than
    two readings taken at different instants.

    This lives here, and not inline in the eval script, for the reason stated
    at the top of this module: that script cannot be imported without
    Isaac, so anything left inside its rollout loop has no CPU test. The decision
    of WHAT a graded attempt means is the part worth pinning; the surrounding
    pending-list bookkeeping stays in the loop.

    The strict threshold is deliberately NOT a parameter. The whole value of the
    pair is that one side is the codebase's quality gate and the other is not, so
    a caller free to choose both could silently collapse them into the same
    number and make `grasp_successes_strict` a copy of `grasp_successes` with
    nothing to notice. For the same reason a `min_force` at or above
    PINCH_MIN_FORCE is refused rather than accepted.

    `held_brick_from_forces` takes its per-brick max BEFORE thresholding, so
    raising the threshold can only turn a named brick into -1, never select a
    different one: strict is a subset of loose by construction, and
    `grasp_successes_strict <= grasp_successes` follows.

    Args:
        left_mag: (N,B) left-jaw contact-force magnitude, per brick.
        right_mag: (N,B) right-jaw contact-force magnitude, per brick.
        env_ids: (k,) envs whose attempts came due; may repeat and need not be
            sorted -- the masks come back aligned with it, one entry per attempt.
        min_force: the loose (attribution) threshold, in newtons.
    """
    if min_force >= PINCH_MIN_FORCE:
        raise ValueError(
            f"min_force={min_force} must be below the quality threshold "
            f"PINCH_MIN_FORCE={PINCH_MIN_FORCE}; equal thresholds make the strict "
            "grading a copy of the loose one"
        )
    ids = env_ids.detach().to(device=left_mag.device, dtype=torch.long)
    loose = held_brick_from_forces(left_mag, right_mag, min_force=min_force)
    strict = held_brick_from_forces(left_mag, right_mag, min_force=PINCH_MIN_FORCE)
    return loose[ids] >= 0, strict[ids] >= 0


def clamp_arm_actions(
    actions: torch.Tensor, limits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp (N,6) arm joint targets into (N,6,2) [lower, upper] limits.

    Returns (clamped, overshoot): overshoot is (N,6) float, the SIGNED distance
    each dim was commanded outside its limit -- positive past the upper bound,
    negative past the lower, exactly zero when inside. Callers that only want
    the old any-joint flag take `(overshoot != 0).any(dim=-1)`.

    Signed and per-joint because magnitude and identity both matter: grazing a
    stop by 0.01 rad on a joint the training data also reaches is a different
    event from commanding 1 rad past a limit the task never approaches, and the
    old boolean reported them identically.

    The gripper column is never clamped -- its sign convention (>= 0 opens) is
    a threshold, not a position.
    """
    if actions.dim() != 2 or actions.shape[1] != 6 or limits.shape != (*actions.shape, 2):
        raise ValueError(
            f"expected actions (N,6) and limits (N,6,2), got {tuple(actions.shape)} / {tuple(limits.shape)}"
        )
    clamped = actions.clamp(min=limits[..., 0], max=limits[..., 1])
    return clamped, actions - clamped
