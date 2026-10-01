"""MS7f stretch runs (MS7f design 10.4, ledger R13): one env, a pool of --heap_k parts served
through the tray refill, full boxes emptied, the category map fixed for the run, a policy server
or the teacher (--oracle_driver).

The heap client (``scripts/mj_eval_heap.py``, with every one of its flags) runs with its
rollout replaced by a chain of rollouts on one env that is never reset, so the pool, the boxes
and the map carry over. The env never times out. A person acts on the three intervention rules:

1. The loop ends its episode (no candidate, or its cycle budget) while parts remain in the
   tray: the person removes them, and they stay unsorted. 1 intervention.
2. A part at rest outside the tray and every box goes back into the tray: 1 per part (the
   loop's ``stretch`` switch, between cycles).
3. ``loop.STRETCH_MISS_STREAK`` cycles in a row close with no new correct stow and end the
   episode: the person puts the parts between the jaws and over the tray's rim back into the
   tray, and the loop starts again with fresh memory. 1 intervention.

An episode that stows every part ends the run (the env's success resets the env). A new
episode of the chain draws its perception and sensor streams from run seed +
EPISODE_SEED_STEP x its index. The run ends when no part waits and the tray is empty, or after
MAX_EPISODES. It reports the parts served (the consecutive parts of the run), the correct share
(parts graded correct at an emptying, or at rest in a box of their own at the end, over the
parts served), correct stows per simulated hour over every control step, and the
interventions per 1000 parts served. Tokens STRETCH_OK / STRETCH_FAIL. Teacher usage:

    python scripts/mj_stretch_run.py --oracle_driver --num_envs 1 --heap_k 1000 --tray \\
        --tray_lean 30 --tray_refill --box_empty --desk_boxes 8 --box_draw 8 --categories \\
        ... --run_seed 20307400 --layout_seed 20307450 --out /tmp/stretch_0.json
"""

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mj_eval_heap
from maker_arm_lab.core import loop
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.stowed import parts_resting_box, per_brick_stowed

STRETCH_EPISODE_S = 1.0e7
"""s: the env's episode length, long enough that the env never times out: the runner ends it."""
MAX_EPISODES = 200
"""The most rollouts one run chains, a cap no run that makes progress reaches."""
EPISODE_SEED_STEP = 100_000
"""Run seed step between two episodes of a chain, clear of every MS7f seed range (R1)."""
INTERVENTIONS = ("tray_left", "strays", "miss_streak")
"""The three rules' counts: rule 1, rule 2 (per part), rule 3."""


class _Parser(mj_eval_heap._TokenEmittingParser):
    token = "STRETCH"
    max_heap_k = 1000  # the design's 1000-part run (MS7f design 10.4), served through the refill


def after_episode(env, ep: dict, counts: Counter) -> None:
    """A person's acts after one episode of the chain (the module docstring's rules 1 and 3),
    booked in ``counts``. Rule 2 ran inside the episode; its count is in ``ep``."""
    counts["strays"] += ep["interventions"]["strays"]
    if ep["done_reason"] == "miss_streak":
        counts["miss_streak"] += 1
        env.return_to_tray(0, sorted(set(env.jaw_parts(0)) | set(env.tray_top_parts(0))))
        return
    left = env.tray_parts(0)
    if left:
        counts["tray_left"] += 1
        env.remove_parts(0, left)


def stretch(env, rig, policy, *, run_seed: int, **kwargs) -> dict:
    """``loop.rollout``'s stand-in for ``mj_eval_heap.main``: the chain of episodes and the
    stretch report, as one payload."""
    if env.num_envs != 1 or not getattr(env, "tray_refill", False) or not kwargs.get("box_empty"):
        raise EvalFailure("stretch", "a stretch run is one env with --tray_refill and --box_empty")
    counts = Counter(dict.fromkeys(INTERVENTIONS, 0))
    events = Counter()
    emptied: dict[int, bool] = {}  # part -> graded correct at its emptying
    chain = []
    for n in range(MAX_EPISODES):
        payload = loop.rollout(
            env, rig, policy, run_seed=run_seed + EPISODE_SEED_STEP * n, stretch=True, **kwargs
        )
        ep = payload["episodes"][0]
        events.update(ep["events"])
        emptied.update({b: ok for b, _, ok in ep["emptied_parts"]})
        chain.append({key: ep[key] for key in ("done_reason", "steps", "cycles", "events")})
        if ep["done_reason"] == "all_stowed":
            break  # every part stowed or emptied correct: the env's success reset it
        after_episode(env, ep, counts)
        if not env.pool[0] and not env.tray_parts(0):
            break
    k = env.cfg.num_bricks
    if chain[-1]["done_reason"] == "all_stowed":
        served, correct, wrong, complete = k, k, 0, True
    else:
        served = k - len(env.pool[0])
        at_rest = per_brick_stowed(env)[0]
        correct = sum(emptied.values()) + int(at_rest.sum())
        # A part emptied from a box not its own, or at rest in one at the end.
        wrong = sum(not ok for ok in emptied.values())
        wrong += int(((parts_resting_box(env)[0] >= 0) & ~at_rest).sum())
        complete = not env.pool[0] and not env.tray_parts(0)
    steps = sum(e["steps"] for e in chain)
    total = sum(counts.values())
    return {
        "run_seed": run_seed,
        "policy_id": "oracle" if policy is None else policy.policy_id,
        "parts": k,
        "parts_served": served,
        "consecutive_parts": served,
        "complete": complete,
        "correct": correct,
        "correct_share": correct / served if served else None,
        "wrong_box": wrong,
        "steps_total": steps,
        "stows_per_hour": 3600.0 * correct / (steps / 30.0) if steps else None,
        "interventions": dict(counts),
        "interventions_total": total,
        "interventions_per_1000": 1000.0 * total / served if served else None,
        "events": {name: events[name] for name in ("tray_refill", "box_empty")},
        "episodes": chain,
    }


def main(argv=None) -> int:
    return mj_eval_heap.main(argv, parser_cls=_Parser, run=stretch, episode_s=STRETCH_EPISODE_S)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
