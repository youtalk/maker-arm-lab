# MS6 phase B, the second lever: a target channel and the retreat wait

Date: 2026-09-22. The owner set a goal on 2026-09-21 at 21:00: the learned heap policy stows at
least 0.80 of the parts, within half a day, on the task as it stands. The task is the one of
`docs/ms6-heap-policy.md`: the bare-desk heap of the held-out split at `--parts_seed 20260915`,
K of 50, a cycle budget of 65, the continuous gripper, open loop. The campaign ledger is
`~/data/maker-arm-ms6/m/README.md`. Every rule below went into that ledger before its number.

## 0. Read this first

The goal was not met. The best checked read of this campaign is 0.6320, 1896 of 3000 parts
stowed, at 2.00 cycles per stowed part, against a goal of 0.80. That read is the 400k
checkpoint of run m, served at 50 actions per forward under the loop of commit adedbfc. The
same checkpoint read 0.6250 before the episode allowance of that commit, inside the same
interval. The previous campaign's checked best was 0.4297 on the same 3000 parts, so the
gain is 0.20 with the two intervals disjoint. This is the first heap read above the 0.50
floor of R-GATE.

Three levers moved the number, and two protocol defects were found and fixed on the way.

The target channel hands the policy the part it is sent for: a 7-wide row with the part's
centre and the teacher's grasp, in place of a 5 px marker seen through a jittering camera. On
the selection draws it took the 50k checkpoint past the previous campaign's best, at one
eighth of the training. It took the 200k checkpoint to 0.510.

The serving horizon is a lever of its own. Executing 50 of the 100 predicted actions and
re-observing lifted 200k from 0.512 to 0.597 on the selection draws. 25 loses 0.29.

The cycle boundary of the heap loop did not match the corpus. The teacher's corpus switches
targets at the end of RETREAT. The loop switched the instant the part rested in the bin, up
to hundreds of steps later. The policy meanwhile picked on a stale target. The loop now
ends the cycle on bin entry and waits for the retreat. The wait still caps after most cycles,
and that stale-row time is the largest known cost left.

Section 5.2 holds the checked reads, 200k at 0.5197 and 400k at 0.6250 and 0.6320. Section 6 lists
what is open, and the next lever it names is the row switch that mirrors the teacher exactly.

## 1. Where the previous campaign stopped

The best read of `docs/ms6-heap-policy.md` is DAgger round 1 at 700k steps: 1289 of 3000 parts,
0.4297, at 3.0 cycles per stowed part. The census of that read names two failure classes:
`held_no_lift` at 0.34 of all cycles and `close_miss` at 0.20. DAgger moved `close_miss` down
and `held_no_lift` not at all, and the pre-registered rule closed the rounds at one.

Two measurements on that same read decided the levers of this campaign.

The first is the geometry of the failures. The loop records the xy distance from the pick point
to the target at each cycle's first closed step. The classes separate on it:

| class | cycles | first close, cm, p10 / p50 / p90 | steps, p50 |
| --- | --- | --- | --- |
| held_no_lift | 1324 | 0.5 / 13.4 / 28.5 | 800 |
| stowed | 1298 | 0.3 / 4.8 / 26.5 | 470 |
| close_miss | 771 | 0.5 / 1.6 / 19.6 | 800 |
| carrying | 158 | 0.3 / 3.1 / 26.5 | 800 |
| lifted_lost | 115 | 0.4 / 3.3 / 26.6 | 800 |

A `close_miss` closes 1.6 cm from the target: the jaws reach the part and land beside it. A
`held_no_lift` closes 13 cm from the target, at the distance of the bin, and 646 of the 1324
follow a stowed cycle.

The second is the cycle boundary. In the recorded corpus, the target marker and the cycle part switch at the end of RETREAT.
The arm is then at `RETREAT_POINT` with the jaws at the full gap, and the next HOVER starts
from there. The heap loop switched the target the instant the stowed part came to rest in
the bin. At that instant the policy was still at the release pose over the bin, with a chunk
in flight. No demonstration pairs that state with a fresh target.

## 2. The two levers

### 2.1 The target channel

The policy locates its target through a 5 px marker in the overhead frame. The overhead camera
jitters 5 mm and 1 degree on every frame. The context reports the per-episode pose and not the
jitter, so the marker fixes the target to about 10 mm. The teacher lands the jaws at the
part width plus 8 mm, which leaves 4 mm of error each side. The 1.6 cm of `close_miss` is that
localization error.

The lever hands the policy the quantity the marker stands for. A 7-wide target row joins the
9-wide camera and box context as `camera_box_target`, 16 wide:

| name | value |
| --- | --- |
| target_x, target_y, target_z | the target part's centre in the base frame at the cycle's first step |
| target_cos, target_sin | the closing-axis heading of the grasp the teacher ranks on the part |
| target_width | the part size along that axis, m |
| target_half_height | half the part size along its most vertical axis, m |

`core.heap_oracle.target_context` builds the row on both sides. When the teacher records, the
rig stashes the row next to the cycle part. The env copies it with the marker, and the
recorder writes `obs/context_target`. When the policy drives, the heap loop writes the row at
selection from the selector's grasp and the part's pose. The marker stays in the frame, and
the images stay in the policy. A perception module on the real arm reports the same row: the
loop already selects the target from privileged state, and the row replaces a pixel with the
number that pixel encodes.

The MS4 privileged-state ceiling read 0.022 with no images and was never explained. This
channel keeps the images, holds one part and not six, and one function builds it on both sides.

### 2.2 The retreat wait

After a cycle ends, the loop now waits before it selects the next target. It selects once the
pick point is within 6 cm of `RETREAT_POINT` with the jaws at least 0.8 open. A cap of 300
steps, at the 800-step cycle budget, ends the wait on a policy that never retreats. That is the state the corpus pairs with a new target. The wait lies between
cycles, so the cycle budget and the cycles-per-stowed rule count as before. Each episode
reports its wait as `select_wait_steps`.

The episode length follows the wait since commit adedbfc. It used to hold every cycle at its
full budget and nothing more. A wait that hit its cap therefore ate the cycles at the end
of the episode. On the 400k check, 41 of 60 episodes ended on the 52000-step cap at a median of 60
of 65 cycles. A quarter of their steps went to the wait. The episode now holds every cycle
at its full budget plus the wait at its cap before each, 65 x 1100 steps. The
pre-registered cycle budget is then what binds. Section 5.3 holds the read under it.

## 3. The code

Branch `feat/ms6-target-context`, two commits. f4cac44 adds the channel to the context names,
the heap oracle, the rig, the env, the recorder, the loop, the converter and the four scripts.
08f519a adds the retreat wait to the loop.
The suite reads 866 passed and ruff is clean. The campaign ran from the detached worktree
`~/src/maker-arm-lab-m` at 08f519a.

Three smokes ran before the campaign. A two-heap recording carried the row on every cycle
frame, with the marker on the cycle part and a unit-norm heading. Its conversion produced a
16-wide `observation.environment_state` with the target names after the box. A 100-step
checkpoint served through the policy server reported `+ctx16+gap`, and the loop's wait capped
at 300 steps per cycle on that untrained policy.

## 4. Corpus m

The corpus is 20 processes of 3 train-split heaps each, K of 50, with seeds 20264000 + p and
20264100 + p. The teacher and the flags are corpus_g's, cameras on, every episode exported.
The recording took 55 minutes on 20 cores with the GPU rendering, and the shards hold 64 GB.
`YIELD_OK`: 2771 of 3000 parts stowed, 0.9237, 136 out of reach, 2954 cycles, 1.066 cycles
per stowed part. Bricks stowed 2290 of 2450 and slopes 485 of 504. corpus_g read 0.9325 at
1.063 on 2000 parts, so the teacher is unchanged. Ten conversion lanes produced 2833 stowed
cycles and 1299506 frames, with a 16-wide `observation.environment_state`. The decode check
reads `CHECK_OK videos=40`. Training ran 400k steps at batch 8 with a checkpoint every 50k.
The rate was 24 to 50 steps per second, depending on the eval load beside it.

The retreat wait of section 2.2 changed twice during the campaign, each time on a measurement.

The first reads (50k, 100k, 200k) ran with the cycle end at rest in the bin. Every draw
spent about 63000 steps in the wait, which means the wait hit its 300-step cap after almost
every cycle. A diagnostic on the 50k policy (one heap of 12 parts, 13 waits) showed why. The
cycle ended once the part came to rest in the bin, hundreds of steps after the release. In
that gap the policy saw the stale target row and started a pick of its own. Commit 2feed28
ends the cycle on the part's entry into the bin, moving or not. The metric never reads the
cycle end. It is the at-rest grader at the episode's end.

The same diagnostic showed five retreats that triggered the wait. They did so at 5.4 to
6.0 cm from `RETREAT_POINT`, on the edge of the 6 cm radius. Eight waits never came within
14 cm.
In six of those eight the pick point sat over the bin at z 0.20 with the jaws closed: the
policy had gone back into the bin after the part it had just released, on the stale row.
Commit 3dbedc4 widens the radius to 10 cm, lowers the jaw opening to 0.7 and halves the cap to
150 steps.

## 5. The selection ladder and the check

Every read below is open loop with no temporal ensembling, on the held-out split, K of 50, a
cycle budget of 65. The draws are 0 to 3 of the selection seed 20261300 (1000 parts) unless
the row says otherwise. The previous campaign's best reads on those same draws are 0.3960 (run g
at 400k) and 0.4310 (round 1 at 700k).

### 5.1 The rungs

| checkpoint | loop | actions per forward | stowed of 1000 | fraction | Wilson 95 % |
| --- | --- | --- | --- | --- | --- |
| 50k | 08f519a, at-rest end | 100 | 449 | 0.4490 | 0.4184-0.4800 |
| 100k | 08f519a, at-rest end | 100 | 273 | 0.2730 | 0.2463-0.3014 |
| 200k | 08f519a, at-rest end | 100 | 510 | 0.5100 | 0.4790-0.5409 |
| 200k | 2feed28, bin entry, 6 cm | 100 | 512 | 0.5120 | 0.4810-0.5429 |
| 200k | 2feed28, bin entry, 6 cm | 50 | 597 | 0.5970 | 0.5663-0.6270 |
| 200k | 3dbedc4, bin entry, 10 cm | 25 | 305 | 0.3050 | 0.2773-0.3342 |
| 200k | 3dbedc4, bin entry, 10 cm | 50 | 538 | 0.5380 | 0.5070-0.5687 |
| 300k | 3dbedc4, bin entry, 10 cm | 50 | 406 | 0.4060 | 0.3760-0.4369 |
| 400k | 3dbedc4, bin entry, 10 cm | 50 | 585 | 0.5850 | 0.5542-0.6152 |

Three things the rungs say.

The target channel alone moves the read. The 50k checkpoint is one eighth of the training and
still under the at-rest cycle end. It reads 0.449 against the previous campaign's 0.431 on
the same draws. 200k reads 0.510, the first heap read above the 0.50 floor of R-GATE. The
100k rung reads 0.273 with `close_miss` at 0.40 of its cycles. The training loss falls
smoothly through it: 0.146 at 40k, 0.120 at 100k, 0.107 at 222k. The checkpoints of this run
vary far more than their loss does.

The serving horizon is a lever of its own. On the same draws and the same loop, executing 50
of the 100 predicted actions and re-observing lifts 200k from 0.512 to 0.597. The two
intervals barely touch. At 25 the read falls to 0.305 with `close_miss` at 0.35 of cycles.
A replan every 25 steps starts each chunk from a state that lags the last chunk's target.
That is the mechanism `docs/ms6-heap-policy.md` section 6 found at one action per forward. 50 is the
horizon of this campaign, and nothing between 25 and 100 was swept.

The bin-entry cycle end did not move the number by itself (0.512 against 0.510), because the
wait still hit its cap after almost every cycle. The wider wait of 3dbedc4 reads 0.538 where the 6 cm wait reads
0.597, at 200k and 50 actions on the same draws. 300k under it reads 0.406. The switch fired before the state the corpus
pairs with a new target. Commit
f719716 puts the constants back. The 300k and 400k rungs therefore have no read under the
final configuration that ranks them against 200k. Section 5.3 records the 400k read that
followed the check.

### 5.2 The check

The check reads draws 0 to 11 of the selection seed, 60 episodes and 3000 parts, under
f719716 at 50 actions per forward. Draws 0 to 3 are the ladder's cached payloads.

| checkpoint | stowed of 3000 | fraction | Wilson 95 % | cycles per stowed |
| --- | --- | --- | --- | --- |
| 200k | 1559 | 0.5197 | 0.5018-0.5375 | 2.271 |
| 400k | 1875 | 0.6250 | 0.6075-0.6422 | 1.927 |
| 400k, episode allowance (adedbfc) | 1896 | 0.6320 | 0.6146-0.6491 | 1.997 |

Draws 4 to 11 of the 200k read 0.46 where draws 0 to 3 read 0.60. The four selection draws
chose both the checkpoint and the horizon, so they carried selection luck. The checked read
is the number of this campaign.

The 400k check was run beside the 200k one and not after it. Under the widened wait of
3dbedc4, 400k read 0.585 where 200k read 0.538 on the selection draws. 400k therefore went
to the check under the final loop as well. Its selection draws read 680 of 1000, 0.680, and its
draws 4 to 11 read 1195 of 2000, 0.5975. The higher checked read, 400k, is the campaign's
number. The 300k and 350k checkpoints have no read under the final configuration.

The census of the two checks, as a share of all cycles:

| checkpoint | cycles | stowed | held_no_lift | close_miss | no_close | carrying | lifted_lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200k | 3540 | 0.365 | 0.260 | 0.232 | 0.088 | 0.032 | 0.021 |
| 400k | 3614 | 0.423 | 0.270 | 0.218 | 0.026 | 0.038 | 0.022 |

At 400k, cycles whose first close lies under 1 cm from the target stow 0.51 of the time.
Cycles that first close 3 cm or more away stow 0.45. Stows of a part other than the target
are 178 of 891, 20 percent, against 23 percent in round 1. Stowed cycles take a median of
409 steps at 400k and 382 at 200k, and the teacher's take 378.5. A cycle that works runs at
the teacher's pace. The failed cycles still run to the 800-step budget, 1955 of them at
400k.

### 5.3 The read under the episode allowance

The 400k checkpoint read draws 0 to 11 again under adedbfc, at 50 actions per forward, with
the episode at 65 x 1100 steps. No episode ended on the step cap. Every one ended on the
cycle budget or with no candidate left, where 41 of 60 ended on the cap before. The read
is 1896 of 3000, 0.6320, at 2.00 cycles per stowed part. That is 0.007 above the read
without the allowance, inside both intervals. The extra cycles at the end of an episode
went to the parts the policy already failed on. The allowance made the cycle budget bind
as designed and moved the number very little. By the ledger's rule it is the
campaign's number.

### 5.4 What the checks say together

`held_no_lift` at 0.26 to 0.27 is the class no lever of this campaign moved, and it is the
class the stale target row feeds. The wait after a stow hits its cap after most cycles. In
that time the policy closes on the part it just released or on a neighbour in the bin.
`close_miss` fell from 0.40 at the worst rung to 0.22, and the 50-action horizon is what
moved it. `no_close` fell from 0.088 to 0.026 between 200k and 400k, so the later checkpoint
commits to a grasp where the earlier one hovered.

## 6. What is still open

- The goal. 0.6320 against 0.80, a gap of 0.168. At 65 cycles for 50 parts the read needs a
  per-cycle stow rate of about 0.62, and the 400k policy stows 0.42 of its cycles.
- The remaining failure classes. At 200k and 50 actions per forward, `held_no_lift` is still
  about 0.28 of all cycles and `close_miss` about 0.20. The wait still hits its 300-step cap
  after most cycles, about 42000 to 63000 wait steps per 250-part draw. The policy therefore
  still spends time on a stale target row after a stow. The row switch that mirrors the teacher
  exactly, at the end of the policy's own retreat, is not built. A read that switches the row
  the instant the part enters the bin, with no wait, was not run.
- The horizon. 50 actions per forward beats 100 by 0.085 and 25 loses 0.29. Nothing between 25
  and 100 was swept, and the mechanism (a replan from a lagging state) says the best value
  sits somewhere above 25.
- The rungs. Only 200k and 400k carry checked reads under the final loop constants. 300k and 350k do
  not. The checkpoints of this run vary far more than their loss does, so a rung between them
  can read higher than either.
- The official seed 20261400 stays unburned. Every number here is on the selection seed.
- A DAgger round with the target channel was not run. The takeover triggers of `core/loop.py`
  work under the channel, and `~/data/maker-arm-ms6/m/round_m.sh` is written and untested.
- The real arm. The target row is what a perception module reports, and the sim reports it
  exactly. A noise model on the row, in training and at serving, is the sim-to-real work this
  report does not do.
- The wait constants. 6 cm, 0.8 open and 300 steps come from one diagnostic on the 50k policy
  and one A/B at 200k. They were not swept.
