# MS6 phase A: the heap teacher read

Milestone: MS6, phase A. This phase builds the heap teacher. The heap teacher is a part
catalog, grasp synthesis on part geometry, and an `EnvOracle` subclass with a failure memory
and a push. This phase then runs the pre-registered teacher read on it. The design is a
document outside this repository. The campaign ran on 2026-09-16, from maker-arm-lab `main`
at `2e4b07d` (merge of PR #50). Every number below comes from a file under
`~/data/maker-arm-ms6/` on that host. The running record is `~/data/maker-arm-ms6/README.md`,
and this report comes from it.

## 1. Summary

**The teacher read fails, on both pre-registered conditions.** The stowed fraction reads
34 of 6000 parts, 0.0057. The Wilson 95 percent interval is [0.0041, 0.0079]. The rule needs
at least 0.90. Cycles per stowed part read 7.24. The rule allows at most 1.3. Both conditions
were pre-registered before the campaign ran. Both fail by a wide margin.

Section 4 names three candidate mechanisms from the per-cycle outcomes. They are candidates.
The data does not prove that any one of them is the cause. Flat parts, plates and tiles, are
never stowed. The teacher
attempts few pick cycles per episode against its budget. About one part in five leaves the
tray by a route the read does not identify. The drop and a push are two candidates, not the
complete set (section 4.1). The read failed, so the plan stops before phase B. The
throughput rows did not run (section 5). No constant changed before or during the read. This
report reflects the code as merged in PR #50. Section 8 records the repeated read on the
fixed teacher. It also fails.

## 2. What was built

Four pull requests landed the heap teacher before the campaign ran.

- PR #47: the part catalog and the tray.
- PR #48: grasp synthesis and the heap teacher.
- PR #49: heap mode in the environment, with the target marker, the depth model, and the box
  context.
- PR #50: the heap teacher on the rig, the batch loop, the scripts, and the docs.

Two design decisions from that work matter for this report.

Design H1 corrects the tray size. The roadmap assumed a 25 by 25 cm tray. A square that size
does not fit the arm's measured reach. Wherever it sits on the desk, its corners fall past
`SCATTER_MAX_RADIUS` (0.280 m). That is the largest radius every measured azimuth tracked to
(`source/maker_arm_lab/core/layout.py`). The built tray has a 13 by 20 cm interior, at x from
0.13 to 0.26 m and y from -0.10 to 0.10 m. Its near edge sits at x = 0.13 m, the start of the
measured scatter band (`SCATTER_X[0]`). Its far corners sit at radius 0.279 m, just under the
reach bound. The roadmap's "thousands" throughput arithmetic comes from a different passage
than the 25 by 25 cm tray size. It assumed a 30 by 30 cm tray. That tray size is a question for
the real arm on the MA track, not a question this simulation can settle.

Design H4 fixes the part set and the part count K per generation run, not per episode. Poses,
colors, and the destination bin still draw per episode. A part set that changes every episode
needs a new model compile and a new renderer for each one. The campaign instead draws
diversity across runs, through many short runs.

## 3. The campaign

The read ran from the detached worktree `~/src/maker-arm-lab-ms6`. That worktree stays at
`2e4b07d` for the record. Ten processes ran the heap branch of
`scripts/mj_generate_batch.py`, 12 environments each, one episode per environment, on the
held-out part split. The parts seed was `20260915`. Process `p` used run seed `20260950 + p`
and layout seed `20260900 + p`. The heap size was K = 50, the cycle budget was 50, and the run
used `--nearest_first_pick_order` with cameras off. A preflight run at
`--attempts 2 --num_envs 2 --heap_k 8` checked both drivers on real flags first
(`~/data/maker-arm-ms6/a/preflight/`).

Every process printed `BATCH_OK`. Wall clock per process ran from 64.1 to 96.0 minutes, mean
80.8 minutes, over 10 processes. No process reached the 240-minute cap. Control rate ran from
5.4 to 8.0 steps per second per process, taken from each process's `sim_steps_per_sec` field.
Five of the ten processes logged a watchdog trip. That count matches the five episode
abandons in section 4.

One episode has a video, under the read's own seeds: held-out split, K = 50, run seed
20260950, layout seed 20260900, cameras on (`~/data/maker-arm-ms6/a/video/episode.mp4`). It
runs 87 seconds at 30 Hz, with overhead RGB, wrist RGB, and overhead depth side by side. A
heap run exports only pick-cycle frames. The clip holds 2617 of the episode's 30900 control
steps. In that episode the teacher stowed 1 of 50 parts, in 4 cycles. It attempted part 44
three times without stowing it. It pushed 3 times. Its last candidate scan counted 5 parts
outside the tray, a snapshot at that scan (Candidate 3 in section 4.1 defines the count). That
count is not the 7 parts sitting outside the tray at step zero, reported later in section 4.1.
The two readings come from different points in the episode, and the last scan already excludes
the 1 part the teacher had stowed by then. This file lives outside the repository, and this
report does not commit it. It illustrates the
campaign. It is not itself a source of the campaign's numbers.

## 4. The read

| quantity | measured | rule | verdict |
| --- | --- | --- | --- |
| stowed fraction | 34 / 6000 = 0.0057, Wilson 95 percent [0.0041, 0.0079] | at least 0.90 | FAIL |
| cycles per stowed part | 7.24 | at most 1.3 | FAIL |
| parts outside the tray, last scan per episode (Candidate 3) | 1107 of 6000 = 18.5 percent | not gated | reported |
| pushes | 367 = 3.06 per episode, capped at 3 in 115 of 120 episodes (Candidate 2) | not gated | reported |
| pick cycles | 246 over 120 episodes = 2.05 per episode | budget is 50 | reported |
| episodes at the 3-push cap | 115 of 120 = 95.8 percent | not gated | reported |
| episodes with zero pick cycles | 50 of 120 = 41.7 percent | not gated | reported |

Over the 120 episodes, 115 timed out and 5 were abandoned. No episode succeeded. The run
logged 423 retries in total. The failure memory excluded 46 cells. An excluded cell is a 5 cm
cell where the teacher failed three times and then gave up on it. Parts outside the tray per
episode ranged from 0 to 23, with a median of 8.5.

Per family, over the held-out split:

| family | cycles attempted | stowed |
| --- | --- | --- |
| brick | 96 | 27 |
| slope | 45 | 7 |
| plate | 62 | 0 |
| tile | 43 | 0 |

Plates and tiles took 105 of the 246 cycles and stowed none of them. Bricks and slopes took
the other 141 cycles. They stowed 34 parts, all of the parts the read counted as stowed.

### 4.1 Candidate mechanisms

When the read fails, the design requires the report to name the mechanism from the per-cycle
outcomes. The design also names the likely fixes: the clearance margin, the drop band, and
the push length, one constant at a time. Three candidates are visible in this data. Each is
presented with its evidence. The data does not prove that any one of them is the cause.

**Candidate 1: flat parts are never stowed.** Plates and tiles took 105 of the 246 cycles and
stowed 0 of them. Bricks and slopes stowed 34 of their 141 cycles. `CLEARANCE_MIN` in
`source/maker_arm_lab/core/grasp.py` is 0.003 m. A grasp needs at least that jaw-box
clearance to be selected. `tests/test_mj_heap_oracle.py` stows parts in two hand-placed
layouts: three spread bricks, and a two-part stack with a third loose part. Both assert 3
parts delivered. The spread-brick test hand-places its bricks at x = 0.195, with 45 mm of y
separation between them. That test's own comment records a check at 40 mm of separation. At
that spacing the best reachable grasp on the outer two parts reads 0.00287 m of clearance,
under `CLEARANCE_MIN`. No test in that file stows a plate or a tile. No test stows a dropped
heap. This is an inference. A flat part has a
smaller candidate grasp set. On the 13 by 20 cm tray interior (design H1), that smaller set
sits closer to the tray walls and to its neighbors. More of a flat part's grasps can fall
under `CLEARANCE_MIN` than a brick's grasps do. The data does not separate tray width from
part geometry as the cause.

**Candidate 2: the teacher gives up early, then idles.** Pushes per episode read
{0: 5, 3: 102, 4: 6, 5: 5, 6: 2}. 102 of 120 episodes ended with exactly 3 pushes, and 115 of
120 ended with 3 or more. `MAX_PUSHES_IN_A_ROW = 3`, in
`source/maker_arm_lab/core/heap_oracle.py`, caps consecutive pushes at 3. The 3.06 pushes per
episode in the table above is this cap, not a free-running push rate.

When no grasp clears and `pushes_in_a_row` reaches `MAX_PUSHES_IN_A_ROW`, `_next_brick` returns
`None`. It does not schedule another push. The prior push already cleared
`self.push` to `None` in `report`, so `_no_target_phase` then finds `self.push is None` and
returns `Phase.DONE`. The teacher stops picking and stops pushing for the rest of the episode.
`tests/test_heap_oracle.py`'s `test_three_pushes_in_a_row_end_the_episode_done` checks this
path directly: after 3 pushes in a row with no clear grasp, `orc.phase is oracle.Phase.DONE`
and `orc.next_goal(scene)` returns `None`.

Cycles per episode read
{0: 50, 1: 18, 2: 10, 3: 13, 4: 8, 5: 10, 6: 4, 7: 2, 8: 2, 9: 1, 12: 2}. 50 of 120 episodes
attempted zero pick cycles. `core/batch_loop.py`'s `cycle_actor` does not record a frame while
the oracle's phase is `SETTLE`, `DONE`, or `ABANDON`. Once a give-up episode reaches `DONE`, it
records nothing further. Recorded in-cycle frames per episode, over the 115 written episodes,
range from 480 to a median of 1268 and a mean of 1912. Each episode allows 30900 control steps.

This is an inference. The teacher does not run out of clock on a steady rate of work. It
reaches the push cap, goes idle, and the rest of the episode records nothing. `CLEARANCE_MIN`
and `MAX_PUSHES_IN_A_ROW` are where phase B's fix belongs, not the cycle budget.

**Candidate 3: parts leave the tray.** `self.outside_tray`, in `_candidates`
(`source/maker_arm_lab/core/heap_oracle.py`), resets to zero on every call. When a non-stowed
part at or below `CANDIDATE_MAX_Z` sits outside the tray bounds at that scan, it counts that
part. The reported total, 1107 of 6000 parts, 18.5 percent, is this count at each
episode's last candidate scan. It is not a running tally of every part that ever left the
tray. `source/maker_arm_lab/core/layout.py` sets the drop band. Part k lands at a height
uniform in `[HEAP_DROP_Z_MIN, HEAP_DROP_Z_MIN + HEAP_DROP_Z_PER_PART * K]`. The physics then
settles for `HEAP_SETTLE_STEPS` steps. The constants read `HEAP_DROP_Z_MIN = 0.03` and
`HEAP_DROP_Z_PER_PART = 0.0012`. At K = 50 the top of the drop band is
0.03 + 0.0012 * 50 = 0.09 m. `TRAY_WALL_HEIGHT`, in the same file, is 0.02 m. A push leg,
`PUSH_HALF_LENGTH_M` in `source/maker_arm_lab/core/grasp.py`, has a half length of 0.04 m.
The teacher allows up to `MAX_PUSHES_IN_A_ROW = 3` pushes before it must try a grasp again.
This is an inference. A part can drop from up to 0.09 m onto a pile inside a 0.02 m wall. On
that drop, or on the settle after it, the part can bounce or roll clear of the tray. A push
leg reaches 0.04 m from its start. Over up to three pushes in a row, that reach can carry a
part the rest of the way past a 0.02 m wall. The drop and a push are the two routes this read
can name, not the complete set. Other routes stay unruled out, such as a failed grasp or a
part dropped in transit across the 423 logged retries. The jaws brushing a neighbor during
DESCEND is a third. The data does not separate ejections caused by the drop and settle from
ejections caused by a push.

**A look at one episode's frames.** The illustration episode from section 3 sits in
`~/data/maker-arm-ms6/a/video/vid.hdf5`. Reading `initial_state/rigid_object/brick_<k>/root_pose`
for its 50 parts against the tray interior (design H1) finds 7 already outside the tray, before
step one. The episode's `obs/target_part` channel names only two targets, over its whole 2617
recorded frames. Part 19, a slope, sits at x = 0.220 m and y = 0.153 m. Part 44, a tile, sits at
x = -0.023 m and y = 0.122 m. Both sit outside the tray. Part 44 sits about 0.15 m past the tray's
near x edge, well clear of it. The teacher spent the episode on these two parts and stowed one.

`_candidates` in `source/maker_arm_lab/core/heap_oracle.py` does not drop a part from the
candidate set for sitting outside the tray. It skips a part already stowed or above
`CANDIDATE_MAX_Z` before the tally. If that part sits outside the tray bounds, it counts the
part in `self.outside_tray`. A candidate still needs the arm's reach: `x` against
`layout.top_down_x_limit`, and the radius `math.hypot(x, y)` against `SCATTER_MAX_RADIUS`. A
candidate also needs an open cell, its failure count under `CELL_FAILURE_CAP`.

It never checks tray membership as a reason to exclude a candidate. `rank_grasps` and
`select_grasp` in `core/grasp.py` then rank a candidate's grasps by height and by clearance
against the other parts and the tray walls. Neither function checks tray membership on its
own. A part outside the tray but inside reach and an open cell is a candidate on the same
terms as a part inside it. This episode's two targets show that path in practice.

This episode adds one more measurement to candidate 3. Seven of its 50 parts sit outside the
tray before the first step, right after the drop and the settle. This is an inference. It points
at a second route out of the tray, earlier than a push: the drop and the settle can eject a part
on their own, apart from any push that follows later in the episode.

The recorded phase channel, over the clip's 2617 frames, splits this way: DESCEND 24.5 percent,
the gripper-close phase 17.2 percent, HOVER 15.4 percent, and the three push phases 31.4 percent
combined. Every other phase holds under 5 percent. `obs/joint_pos_abs` holds pre-step joint
positions in URDF order, 6 arm joints and 2 jaws. Summing the absolute step-to-step change
across its 8 columns averages about 0.017 rad per step during a push, against 0.04 rad per
step elsewhere. `obs/clamped` reads false on every one of the 2617
frames. The arm slows during a push. It does not stall, and it never reaches the command clamp.

These readings come from one episode, not from the 120-episode read behind section 4's rates.
They point at a mechanism. They do not change the stowed fraction, the cycles per stowed part,
or any other rate in this report.

### 4.2 A test suite gap

The full suite has 705 passing tests. None of them caught this. `tests/test_mj_heap_oracle.py`
stows parts only in hand-placed layouts: three spread bricks, or a two-part stack. It never
stows a plate, a tile, or a dropped heap at K = 50. Section 6 lists this gap as an open
question.

## 5. The throughput rows

The throughput rows did not run. When the teacher read fails, the pre-registered plan says to
record which condition failed and stop before phase B. Both of its conditions failed
(section 4). No timing data exists for K = 10, 50, 100, or 200 in this campaign.

## 6. Open questions

- Which of the three candidate mechanisms in section 4.1 to fix first, and whether a fix to
  one moves the others. The pre-registered design names the clearance margin, the drop band,
  and the push length as the three constants in scope, one at a time.
- Whether the flat-part failure (candidate 1) comes from clearance, from tray width, or from
  both. No test isolates part geometry from tray width for a plate or a tile.
- Whether the drop band or the push length (candidate 3) accounts for more of the 18.5 percent
  ejected. The read does not separate drop-time ejections from push-time ejections.
- Why a push does not clear a grasp within the 3-push cap (candidate 2), and whether
  `CLEARANCE_MIN` or the push length or placement is the lever. `MAX_PUSHES_IN_A_ROW = 3` ends
  an episode's picking once it is hit, so the low pick-cycle count follows from that cap, not
  from a separate push trigger question.
- The test suite gap (section 4.2). Before phase B changes any of the three constants, add a
  test that stows a plate, a tile, and a dropped heap at a representative K. That test lets a
  fix get checked against a real regression, not only against this campaign's read.
- Whether the 13 by 20 cm tray interior (design H1) adds its own constraint on candidates 1
  and 3, apart from the drop band and the push length, and whether the MA track's real tray
  removes or changes that constraint.
- Whether `_candidates` in `core/heap_oracle.py` needs a rule to drop a part from the candidate
  set for sitting outside the tray, instead of only counting it. Today a part outside the tray
  but inside reach is a candidate on the same terms as a part inside it.
- `obs/context_camera` holds exactly one distinct row across the illustration episode's 2617
  frames. The overhead frames jitter by a few pixels every step, from the `--frame_jitter`
  augmentation. Whether the context channel needs that per-frame jitter, or holding it fixed
  is the intended behavior, is open.
- The readings in section 4.1 about the illustration episode's frames come from one episode,
  not from the 120-episode read. They point at a mechanism. Section 4's rates stay the numbers
  a fix must move.
- `tests/test_heap_oracle.py`'s `test_candidates_exclude_out_of_reach_lifted_and_stowed_parts`
  is the only unit test of the `outside_tray` counter. Its one out-of-tray part sits at
  x = 0.35 m, past `SCATTER_MAX_RADIUS` as well as outside the tray. It never exercises a part
  that sits outside the tray but still inside reach. No test in `tests/test_mj_heap_oracle.py`
  places such a part and checks that the teacher skips it, picks it, or handles it some other
  way. Section 4.2 lists the flat-part and dropped-heap gaps. This is a third gap.

## 7. Reproduction

```
# Code: maker-arm-lab pull requests #47-#50, main at 2e4b07d.
# Drivers and ledger under ~/data/maker-arm-ms6/.
git -C ~/src/maker-arm-lab worktree add --detach ~/src/maker-arm-lab-ms6 2e4b07d
ln -s ~/src/maker-arm-lab/.venv ~/src/maker-arm-lab-ms6/.venv

# The teacher read: 10 processes, held-out parts, K = 50 (design section 10).
bash ~/data/maker-arm-ms6/a/run_read.sh

# The throughput rows. This campaign did not run them, because the read above failed.
bash ~/data/maker-arm-ms6/a/run_rows.sh
```

## 8. The repeated read: the fixed teacher (campaign c)

**The teacher read fails again, on both pre-registered conditions.** The stowed fraction
reads 1619 of 6050 parts, 0.2676. The Wilson 95 percent interval is [0.2566, 0.2789], against
a rule of at least 0.90. Cycles per stowed part read 2.176, against a rule of at most 1.3. This is the
same pre-registered 120-episode teacher read as section 4, repeated on fresh seeds after the
teacher-logic fixes below. It still fails.

### 8.1 What changed since the failed read

Between the phase A read and this one, an inspection found four teacher-logic defects and
fixed all four. `docs/ms6-teacher-inspection.md` names the mechanisms and the fixes in full,
and this section does not restate them. PR #53 landed four fixes:

- A landing-spot clearance model with the tray walls as hard stops.
- A push that moves a wall-blocked part toward the tray center.
- A pressed-tip descend for flat parts.
- A part-aware LIFT gate.

PR #54 landed the inspection write-up. It reports three held-out K = 50 reads of the fixed
teacher: 15, 11 and 17 of 50 parts stowed. This campaign repeats the pre-registered
120-episode read on top of both.

### 8.2 The campaign

The campaign ran from the detached worktree `~/src/maker-arm-lab-ms6`, at commit `5773614`,
the merge of PR #54 into `main`. Ten processes ran the held-out part split, parts seed
20260915, at K = 50 and a cycle budget of 50. Process `p` used run seed `20261150 + p` and
layout seed `20261100 + p`, for `p` from 0 to 9. The flags matched the pre-registered design.
They set `--nearest_first_pick_order`, hover jitter from 0.035 to 0.080, `--episode_draws`,
`--frame_jitter`, and cameras off. `--max_wall_minutes 600` replaced the design's 240,
because the fixed teacher no longer gives up early and runs its full episode. Every process
printed `BATCH_OK`. Wall time per process ran 54.6 to 74.5 minutes, mean 62.4 minutes. The
ten processes ran together in about 75 minutes. Control rate ran 6.9 to 9.4 steps per second
per process, mean 8.3. No process reached the cap. One env auto-reset mid-batch and finished
a second episode, so the run recorded 121 episodes and 6050 parts, not 120 and 6000. The
throughput rows did not run this round. The inspection design holds them until the read
passes.

### 8.3 The read

| quantity | measured | rule | verdict |
| --- | --- | --- | --- |
| stowed fraction | 1619 / 6050 = 0.2676, Wilson 95 percent [0.2566, 0.2789] | at least 0.90 | FAIL |
| cycles per stowed part | 2.176 | at most 1.3 | FAIL |
| parts outside the tray, last scan | 2066 of 6050 = 0.342 | not gated | reported |
| pushes | 4151 = 34.3 per episode | not gated | reported |
| forced picks | 549 = 4.5 per episode | not gated | reported |
| pick cycles | 3523 over 121 episodes = 29.1 per episode | budget is 50 | reported |

Per family, over the held-out split:

| family | attempts | stowed |
| --- | --- | --- |
| brick | 1408 | 805 |
| plate | 763 | 246 |
| slope | 606 | 204 |
| tile | 746 | 174 |

The family rows sum to 1429 stowed parts. The run total, the sum of each episode's
`bricks_delivered`, reads 1619. Section 8.5 lists the 190 part gap as an open question.

Episode outcomes split 114 time outs and 7 abandons, out of 121 episodes. No episode
succeeded. Every abandon was a watchdog trip. Phase A also had watchdog trips, 5 of them
over 120 episodes. This read has 7. The seven trips here stowed 1 to 15 parts each before
they stopped. Stowed per episode ranged from 1 to 27, mean 13.38. Cycles per episode ranged
from 4 to 39, mean 29.12.

### 8.4 What this read adds beyond the inspection's three heaps

The inspection report's three held-out K = 50 heaps stowed 43 of 150 parts, 0.29. This read
stows 1619 of 6050 parts, 0.2676. This is an inference. The two numbers sit in the same
band, on a sample about 40 times larger. The three heaps only estimate a ceiling for the
fixed teacher. This read
measures it, across 121 episodes and ten seeds instead of three.

### 8.5 Open questions

- The outside-tray fraction rose to 0.342 (2066 of 6050), against 0.185 (1107 of 6000) in
  phase A, with about eleven times the pushes (4151 against 367). This read does not
  attribute the rise. It is consistent with the added pushes, and it is not proven to be
  caused by them.
- The 190 part gap between `bricks_delivered` (1619) and the parts credited to a family row
  (1429). A family row counts a pick cycle whose outcome was true. Some parts reach a bin by
  a path this read does not credit. This read does not explain the gap.
- This read logged 7 watchdog trips over 121 episodes, against 5 over 120 in phase A. Both
  reads have watchdog trips. Whether the small rise means anything is open.
- The inspection report's three owner decisions stay open. Whether to reopen the tray
  footprint or the pick primitive. Whether to reopen the drop band. Whether phase B waits on
  either answer, or starts from this read's ceiling.

### 8.6 Reproduction

```
# Code: maker-arm-lab pull requests #53-#54, main at 5773614.
# Drivers and ledger under ~/data/maker-arm-ms6/c/.
bash ~/data/maker-arm-ms6/c/run_read.sh

# The driver call that run_read.sh makes per process, p = 0 to 9:
.venv/bin/python scripts/mj_generate_batch.py --out_dir ~/data/maker-arm-ms6/c/read \
  --shard_name proc_0 --attempts 12 --num_envs 12 --heap_k 50 --cycle_budget 50 \
  --parts_split heldout --parts_seed 20260915 --run_seed 20261150 --layout_seed 20261100 \
  --nearest_first_pick_order --hover_jitter_min 0.035 --hover_jitter_max 0.080 \
  --episode_draws --frame_jitter --no_cameras --probe --max_wall_minutes 600
```
