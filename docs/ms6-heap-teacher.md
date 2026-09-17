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
tray during the drop or a push. The read failed, so the plan stops before phase B. The
throughput rows did not run (section 5). No constant changed before or during the read. This
report reflects the code as merged in PR #50.

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
reach bound. The roadmap's "thousands" throughput arithmetic assumed a 30 by 30 cm tray. That
tray size is a question for the real arm on the MA track, not a question this simulation can
settle.

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
three times without stowing it. It pushed 3 times, and it put 5 parts outside the tray. This
file lives outside the repository, and this report does not commit it. It illustrates the
campaign. It is not itself a source of the campaign's numbers.

## 4. The read

| quantity | measured | rule | verdict |
| --- | --- | --- | --- |
| stowed fraction | 34 / 6000 = 0.0057, Wilson 95 percent [0.0041, 0.0079] | at least 0.90 | FAIL |
| cycles per stowed part | 7.24 | at most 1.3 | FAIL |
| parts outside the tray | 1107 of 6000 = 18.5 percent | not gated | reported |
| pushes | 367 = 3.06 per episode | not gated | reported |
| pick cycles | 246 over 120 episodes = 2.05 per episode | budget is 50 | reported |

Over the 120 episodes, 115 timed out and 5 were abandoned. No episode succeeded. The run
logged 423 retries in total. The failure memory excluded 46 cells. An excluded cell is a 5 cm
cell where the teacher failed three times and then gave up on it. Parts outside the tray per
episode ranged from 0 to 23, with a median of 9.

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
clearance to be selected. `tests/test_mj_heap_oracle.py` stows parts in one layout only: three
bricks hand-placed at x = 0.195, with 45 mm of y separation between them. That test's own
comment records a check at 40 mm of separation. At that spacing the best reachable grasp on
the outer two parts reads 0.00287 m of clearance, under `CLEARANCE_MIN`. No test in that file
stows a plate or a tile. No test stows a dropped heap. This is an inference. A flat part has a
smaller candidate grasp set. On the 13 by 20 cm tray interior (design H1), that smaller set
sits closer to the tray walls and to its neighbors. More of a flat part's grasps can fall
under `CLEARANCE_MIN` than a brick's grasps do. The data does not separate tray width from
part geometry as the cause.

**Candidate 2: the teacher rarely attempts a cycle.** The budget is 50 cycles per episode. The
teacher used 2.05 of them on average. It pushed 3.06 times per episode, close to one push for
every pick cycle. 115 of 120 episodes hit the time out, not the cycle budget. This is an
inference. The teacher spends its episode time on pushes and on failed grasp searches, not on
picks. If that reading holds, raising the cycle budget alone will not raise the stowed count.
The evidence points at the push length or at the push trigger condition, not at the budget.

**Candidate 3: parts leave the tray.** 1107 of 6000 parts, 18.5 percent, ended up outside the
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
part the rest of the way past a 0.02 m wall. The data does not separate ejections caused by
the drop and settle from ejections caused by a push.

### 4.2 A test suite gap

The full suite has 705 passing tests. None of them caught this. `tests/test_mj_heap_oracle.py`
stows parts only in hand-placed layouts: three spread bricks, or a two-part stack. It never
stows a plate, a tile, or a dropped heap at K = 50. Section 6 lists this gap as an open
question.

## 5. The throughput rows

The throughput rows did not run. When the teacher read fails, the pre-registered plan says to record which condition failed
and stop before phase B. Both of its conditions failed (section 4). No timing data exists for K = 10, 50, 100, or 200 in this campaign.

## 6. Open questions

- Which of the three candidate mechanisms in section 4.1 to fix first, and whether a fix to
  one moves the others. The pre-registered design names the clearance margin, the drop band,
  and the push length as the three constants in scope, one at a time.
- Whether the flat-part failure (candidate 1) comes from clearance, from tray width, or from
  both. No test isolates part geometry from tray width for a plate or a tile.
- Whether the drop band or the push length (candidate 3) accounts for more of the 18.5 percent
  ejected. The read does not separate drop-time ejections from push-time ejections.
- Whether the push trigger or the push length drives the low pick-cycle count (candidate 2).
  An alternative is that the teacher spends its budget on grasp searches that find nothing
  above `CLEARANCE_MIN`.
- The test suite gap (section 4.2). Before phase B changes any of the three constants, add a
  test that stows a plate, a tile, and a dropped heap at a representative K. That test lets a
  fix get checked against a real regression, not only against this campaign's read.
- Whether the 13 by 20 cm tray interior (design H1) adds its own constraint on candidates 1
  and 3, apart from the drop band and the push length, and whether the MA track's real tray
  removes or changes that constraint.

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
