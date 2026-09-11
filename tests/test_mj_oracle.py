"""The GRASP and ORACLE parity checks of the MS3 design (section 3), as one CPU test.

The scripted oracle -- the same ``OracleRig`` that generated the MS1 and MS2 corpora on Isaac
-- runs a two-brick episode on the MuJoCo env with cameras off and must stow both bricks: a
real pinch at the measured stall (PINCH_MIN_STALL, PINCH_MIN_FORCE), a lift, a carry, a
release into the right bin, twice. About 15 s on one core. The six-brick episode is the
same code path six times over and is left to ``scripts/mj_generate_batch.py --probe``.
"""

import random

from maker_arm_lab.core import oracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


def test_oracle_sorts_two_bricks_on_mujoco():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    pinches = []
    steps = 0
    while steps < 4000:
        out = rig.step()
        steps += 1
        goal = rig.goals[0]
        if goal is not None and goal.phase is oracle.Phase.LIFT and rig.steps_in_phase[0] == 60:
            pinches.append(rig.pinch_evidence(0, rig.oracles[0].target_brick))
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    stats = rig.episode_stats(0)
    env.close()
    assert stats["succeeded"] is True, stats
    assert stats["bricks_delivered"] == 2 and stats["retries"] == 0
    assert stats["pick_order"] == [0, 1]
    assert len(pinches) == 2 and all(p["ok"] for p in pinches), pinches
    # The grip Isaac measured was 7.4-7.7 mm of stall at 7.6 N; MuJoCo's contact model lands
    # a little softer (design section 3 records the tuning). Anything under the floors would
    # have failed `ok` above; this pins the magnitude so a silent contact regression shows.
    assert all(0.0055 <= p["stall"] <= 0.0085 for p in pinches), pinches
    assert all(min(p["left"], p["right"]) >= 4.0 for p in pinches), pinches
    assert steps < 2000, steps  # two bricks in well under half the six-brick budget
