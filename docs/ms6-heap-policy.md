# MS6 phase B: the heap policy and the open-loop protocol

Date: 2026-09-21. Phase B trains an ACT policy on the bare-desk heap of
`docs/ms6-bare-desk-heap.md` and puts it through a pre-registered gate. The gate is 0.85 of
the dropped parts stowed and at most 1.3 pick cycles per stowed part, on the held-out split
of the 29-part catalog at `--parts_seed 20260915`, K of 50, a cycle budget of 65 and 120
episodes. The design is `2026-09-18-ms6-phase-b-roadmap-design.md` section 5. The campaign
ledger with every number is `~/data/maker-arm-ms6/l/README.md`. Every rule below was written
into that ledger before its number arrived.

## 0. Read this first

The gate did not pass. The best read of the campaign is 0.4198 against a gate of 0.85, which is
0.4302 short of it.

Three results stand. The binary gripper label is unlearnable, and the continuous contract
that replaces it needed a new teacher release before the corpus reached the recording bar.
The heap policy serves open loop: the MS3 inference protocol, which every earlier milestone
used, reads 0.1868 against the open loop's 0.3230 on the same checkpoints. Design 5.1 carries
that addendum. Run g peaks at 400k steps and reads 0.4198 on the selection seed. DAgger round
1 reads 0.3463 at 600k on 3000 parts, below run g, and rises with the step count where run g
fell. The owner read that as an undertrained run and resumed it to 1000k steps. That ladder is
running and section 8 holds its empty table.

The official seed 20261400 is unburned. The selection numbers sit far under the 0.50 R-GATE
floor, so the official read would record a number and buy no direction.

## 1. The code that landed

Three pull requests, one stack (#79), merged into `main` as `0263d28`.

- #77 `feat/ms6-heap-loop`: `core/loop.py`, the heap outer loop. It selects the target through
  `HeapOracle.select_target`, stamps the marker, runs the policy for one cycle, grades the
  cycle with `core/stowed.py` and books the outcome in the cell failure memory. There is no
  reset between cycles. `scripts/mj_eval_heap.py` is its thin client, and
  `scripts/run_gate_eval.py` gained the `--heap_k` mode and the cycles-per-stowed rule.
- #78 `feat/ms6-continuous-gripper`: `gripper_mode` on the MuJoCo env. In continuous mode
  `action[6]` is the target jaw gap in meters, and the teacher emits its matched jaw opening
  directly. The contract travels through the converter, `train_act.sh`, the policy server
  handshake and every client. The Isaac backend stays binary.
- #80 `feat/ms6-heap-dagger`: the takeover triggers of `core/loop.py` and
  `scripts/mj_dagger_heap.py`. The rig takes the env over on the loop's current target, runs
  one cycle and hands it back.

The suite reads 855 passed at the merged head and ruff is clean. Two later drafts serve the
campaign and not the contract: #81 makes the supervisor's draw cache compare the served
inference configuration, and #82 adds the per-cycle census and its reader.

## 2. The binary base is unlearnable

`base_b` trained on the k corpus, binary gripper, 600k steps. It stows 0 of 32 diagnostic
cycles, on train seeds and on held-out seeds alike. Every cycle ends on the 800-step budget.
The policy toggles the gripper 62 times in 3200 steps and never starts a grasp.

The eval pipeline is not the cause. The step-0 state and context match the corpus frame 0 on
the same seeds. The cause is the label. The binary teacher holds a matched jaw opening by
flipping the command between plus one and minus one around a set point the student cannot
observe, because the flip rule reads `grasp_width`. The recorded gripper column therefore
flips a median of 69 times per cycle and is open 46 percent of the approach. ACT averages that
to close, and the jaws shut on empty air. An open-loop replay agrees with the recorded gripper
only 0.53 to 0.81 of the time, at an arm error near 0.03 rad. Design 5.10 names this risk.

No serving change repairs the checkpoint. Ruling of 2026-09-19: `base_b` is recorded as not
gated, with the reason "unlearnable gripper label, 0 of 32 diagnostic cycles". The continuous
run becomes the adopted mode by default and its official read carries R-GATE. The A/B of
design 5.4 therefore has no binary control. A relabelled binary run is still available. It
costs one conversion and four hours of training.

## 3. The continuous teacher needed a new release

The first continuous corpus read 1566 of 2000 stowed, 0.783, with 331 parts out of reach
against the k corpus's 51. The plan's own rule holds a conversion under 0.90, so the corpus
was traced as a defect instead of converted.

The grip was not the cause. The release was. At the matched gap the jaws open smoothly, and
the part stays on the lower jaw of the tilted jaw plane. The binary teacher frees the part
only by the jolt of its own flipping. Stuck parts read 20 to 26 per heap against 1 to 2.

The ladder on the four corpus heaps p0 to p3, K of 50, stowed of 200:

| release variant | stowed of 200 | fraction |
| --- | --- | --- |
| binary, the recorded k contract | 188 | 0.940 |
| continuous, steep jaw plane, matched gap | 158 | 0.790 |
| continuous, steep jaw plane, full-gap release | 174 | 0.870 |
| full gap plus a finger-contact wait | 175 | 0.875 |
| level jaw plane at release, first cut | 180 | 0.900 |
| level jaw plane with the paced wrist roll, adopted | 183 | 0.915 |

A debounced version of the finger-contact wait read 172, and a vertical lift bought nothing. A
flat jaw plane at release was tried inside the same trace and read 136 of 148 cycles, but it
raised EXTEND overruns, which the pre-registered bar forbids, so it was reversed. The adopted
fix is a level jaw plane too. The paced wrist roll is what made a level plane reach the release
without the overruns.

The pre-registered bar was 0.90 with no rise in EXTEND overruns or abandons. No variant of the
steep jaw plane reached it, and the campaign stopped for an owner decision on 2026-09-19. The
owner ruled a redesign of the release. EXTEND now rolls the jaw plane level with a wrist roll
that keeps 10 degrees off the wrist stops, and RETREAT unrolls before it plans. The first cut
(`1bdff46`) met the bar exactly at 180 of 200 with 0 overruns and 0 abandons. A review then
found that the RETREAT unroll started one roll step off the commanded wrist and jumped the
wrist up to 6 degrees every cycle. The re-record was stopped and deleted. `d31dc12` made the
unroll continuous and bounded the wrist pace under the URDF's 5 rad per second. It reads 183
of 200, 0.915, with 0 overruns and 0 abandons. Binary replays byte-identical through both
commits.

## 4. corpus_g, the dataset and the training run

`corpus_g` re-recorded at `d31dc12` with the same launcher and the same seeds as the k corpus:
8 processes of 5 train-split heaps of 50 parts, cameras on.

- `YIELD_OK` 1865 of 2000 stowed, 0.9325, 91 parts out of reach, 1982 cycles, 1.063 cycles per
  stowed part. The k corpus reads 0.9525 at 1.036.
- By family: bricks 1613 of 1723, slopes 243 of 259. The k corpus reads 1642 of 1717 and 246
  of 257.
- `dataset_g`: `CONVERT_OK episodes=1893 skipped=0` against the k corpus's 1921, 883621 frames,
  marker `{"action_space": "absolute", "gripper": "continuous"}`, the seventh action feature
  named `gripper_gap_m`. The video decode check reads `CHECK_OK`.
- `train_g`: `TRAIN_OK` at 600k steps, checkpoints every 50k.

The level release costs two points of corpus yield against the binary teacher.

## 5. The selection ladder of run g

The reads below served each checkpoint in its own configuration, which is 100 action steps per
forward and no temporal ensembling. Section 6 explains why that turned out to be the right
protocol and the wrong way to arrive at it. The seed is 20261300, held-out split, K of 50,
cycle budget 65, four parallel lanes per read.

| checkpoint | draws | stowed | fraction | Wilson 95 % | cycles per stowed |
| --- | --- | --- | --- | --- | --- |
| 200k | 24 | 1563 / 6000 | 0.2605 | 0.2495-0.2718 | 4.923 |
| 300k | 24 | 1971 / 6000 | 0.3285 | 0.3167-0.3405 | 3.913 |
| 400k | 24 | 2519 / 6000 | 0.4198 | 0.4074-0.4324 | 3.066 |
| 500k | 8 | 724 / 2000 | 0.3620 | 0.3412-0.3833 | 3.557 |
| 600k | 8 | 665 / 2000 | 0.3325 | 0.3122-0.3535 | 3.848 |

The 500k and 600k rungs ran 8 draws and not 24. A full 24-draw selection read costs about 2.5
hours and only ranks checkpoints, and the curve already sat far from the gate. The rule stayed
the highest stowed fraction, ties to the later checkpoint. The winner is 400k. The 500k and
600k intervals lie under the 400k interval with no overlap, so the run peaks near 400k and
then falls.

0.4198 against a gate of 0.85 leaves the policy 0.4302 short. It also sits under the 0.50 R-GATE
floor.

## 6. The inference protocol defect and the re-read

`gate.sh` and `round.sh` pass no inference flag. Every run g read therefore served the
checkpoint's own configuration: `n_action_steps` 100 and no temporal ensembling. Design 5.1
keeps the MS3 protocol, and every MS3 and MS4 read used `--n_action_steps 1
--temporal_ensemble_coeff 0.01`. The ladder of section 5 is therefore a set of diagnostic rows
and not a set of gate reads.

Three more defects came out of the same trace. `round.sh` served eight clients from one policy
server, and the ensemble state is per server. The gate worktree predated the level release. The
supervisor matched a cached draw on a `policy_id` prefix alone. The fixes are one server per
client slot, a new worktree at the level release, and a cache key that also compares the served
inference configuration. The re-read writes to `l/gate_te/` and never into `l/gate/`.

Rules written before any re-read number arrived: every policy server of a gate read and of a
DAgger rollout runs the MS3 flags with one client per server, every ledger row names the
`policy_id` suffix of its payloads, and the ensemble read is compared against the open-loop
read of the same checkpoint on draws 0 to 3. If the ensemble reads lower by more than 0.03 on
the pooled draws, the campaign stops before the official read and the owner decides.

The paired comparison, draws 0 to 3 of seed 20261300, 1000 parts per cell:

| rung | ensemble | open loop | difference |
| --- | --- | --- | --- |
| 200k | 168 / 1000 = 0.1680 | 226 / 1000 = 0.2260 | -0.0580 |
| 300k | 150 / 1000 = 0.1500 | 313 / 1000 = 0.3130 | -0.1630 |
| 400k | 261 / 1000 = 0.2610 | 396 / 1000 = 0.3960 | -0.1350 |
| 500k | 155 / 1000 = 0.1550 | 362 / 1000 = 0.3620 | -0.2070 |
| 600k | 200 / 1000 = 0.2000 | 318 / 1000 = 0.3180 | -0.1180 |
| pooled | 934 / 5000 = 0.1868 | 1615 / 5000 = 0.3230 | -0.1362 |

The pooled difference is -0.1362 against the -0.03 rule, so the campaign stopped before the
official read. The MS3 protocol reads lower on every one of the five rungs. The top ensemble
rung is 400k, the same rung the open loop picks.

The owner then asked for a coefficient sweep before any ruling. LeRobot's
`ACTTemporalEnsembler` weights the queued actions as `w_i = exp(-coeff * i)` with `w_0` the
oldest, so a larger positive coefficient leans harder on the older actions. The sweep ran on
the 400k checkpoint, draws 0 to 3, 1000 parts per cell:

| served protocol | stowed of 1000 | fraction |
| --- | --- | --- |
| open loop, `n_action_steps` 100 | 396 | 0.3960 |
| `n_action_steps` 1, coefficient 0.01 | 261 | 0.2610 |
| `n_action_steps` 1, coefficient 0.05 | 129 | 0.1290 |
| `n_action_steps` 1, coefficient 0.1 | 26 | 0.0260 |
| `n_action_steps` 1, coefficient 0.3 | 4 | 0.0040 |
| `n_action_steps` 1, no averaging at all | 3 | 0.0030 |

The fraction falls with the coefficient over the whole sweep, so 0.01 is the best ensemble
setting and not a mistuned one. No positive coefficient recovers the open loop. The
no-averaging cell names the mechanism: 63.7 percent of its cycles never opened the jaws, and
only 0.41 percent stowed. The heap gripper command is close to a step function. Averaging it
blurs the close. That is the mechanism of section 2 again, this time at serving time and not
at training time. The policy depends on executing whole 100-step chunks.

The cycle census of the re-read, as a share of all cycles, records where the cycles go:

| rung | held_no_lift | close_miss | stowed | no_close | lifted_lost | carrying | no_open |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200k | 0.3315 | 0.4390 | 0.1356 | 0.0401 | 0.0257 | 0.0177 | 0.0104 |
| 300k | 0.2810 | 0.3591 | 0.1264 | 0.1912 | 0.0224 | 0.0150 | 0.0050 |
| 400k | 0.3826 | 0.2712 | 0.2131 | 0.0646 | 0.0307 | 0.0363 | 0.0016 |
| 500k | 0.2719 | 0.4018 | 0.1275 | 0.1075 | 0.0449 | 0.0281 | 0.0184 |
| 600k | 0.3159 | 0.3587 | 0.1584 | 0.0396 | 0.0926 | 0.0301 | 0.0048 |

Cycles that stowed a part other than the target: 52 of 169 at 200k, 36 of 152 at 300k, 82 of
264 at 400k, 59 of 159 at 500k and 59 of 200 at 600k. The census never gates and it picks no
lever by itself.

The owner's ruling of 2026-09-20:

1. The heap task serves the policy open loop, for gate reads and for DAgger rollouts alike.
   Design 5.1 carries the addendum. The MS3 protocol stays the rule for the six-brick task,
   which has its own measurements.
2. The official seed 20261400 of run g stays unburned.
3. DAgger round 1 runs from the 400k checkpoint of run g, open loop.

The realizability probe of design 5.5 passed before the round: `PROBE_OK accuracy=0.9257
held=1758 free=4242` against the 0.90 rule.

## 7. DAgger round 1

Rollouts: 24 processes of 5 envs on train-split heaps, the 400k checkpoint of run g driving
open loop, one policy server per client slot, seeds `20261500 + 100 + p` and
`20261600 + 100 + p`. The seed stride is 100 per round and not the design's 10, because 24
processes overlap otherwise.

Over 120 episodes the mixed run stowed 5587 of 6000 parts, 0.931. There were 5515 takeovers.
The rig stowed 5248 and handed 267 back failed. It declined 347 more triggers.

| trigger | takeovers |
| --- | --- |
| random | 3539 |
| closed_no_pinch | 912 |
| close_far | 432 |
| half_budget_no_lift | 378 |
| no_close | 254 |

| census class | cycles |
| --- | --- |
| stowed | 5615 |
| lifted_lost | 315 |
| close_miss | 67 |
| no_close | 58 |
| held_no_lift | 45 |
| no_open | 2 |

The conversion ran in eight parallel lanes through `convert_parallel.sh`, which is new, because
the stock converter uses one core of twenty four. It took 17 minutes instead of about two
hours. It produced 5317 cycles and 2058690 frames, and the aggregate with `dataset_g` holds
7210 episodes and 2942311 frames. The decode check reads `CHECK_OK videos=76`. The raw shards
were 98 GB and were deleted after that check.

Training ran 600k steps at batch 8 and printed `TRAIN_OK` in 3 hours 26 minutes, at 48.6 steps
per second.

The selection ladder, open loop, draws 0 to 3 of seed 20261300, 1000 parts per cell:

| rung | round 1 | run g on the same draws |
| --- | --- | --- |
| 200k | 224 / 1000 = 0.2240 | 0.2260 |
| 300k | 299 / 1000 = 0.2990 | 0.3130 |
| 400k | 301 / 1000 = 0.3010 | 0.3960 |
| 500k | 325 / 1000 = 0.3250 | 0.3620 |
| 600k | 348 / 1000 = 0.3480 | 0.3180 |

The check step ran the top two rungs on draws 0 to 11, 3000 parts per cell. 600k reads 1039 of
3000, 0.3463, Wilson 0.3295 to 0.3635, at 3.671 cycles per stowed part. 500k reads 938 of
3000, 0.3127. The selected checkpoint of round 1 is 600k.

0.3463 is below the run g selection read of 0.4198, so round 1 misses its adoption rule of the
previous read plus 0.03.

## 8. The resume to 1000k steps

Round 1 rises with the step count over the whole ladder, where run g peaked at 400k and fell.
Its corpus is 3.3 times larger than run g's: 2942311 frames against 883621. The owner read the
run as undertrained rather than as a failed round, and resumed it to 1000k steps with
`resume_train.sh`, which is new because `train_act.sh` never resumes. The resume printed
`TRAIN_OK` at 1000k steps on 2026-09-21.

The resume ran 400k steps in 3 hours 9 minutes at 42.9 steps per second. The resumed
checkpoints carry seven-digit names, from `0650000` to `1000000`, because LeRobot pads to the
width of the new total.

The rules below went into the ledger before any number of this ladder arrived.

The selection protocol: open loop, the checkpoint's own `n_action_steps` of 100, no temporal
ensembling and no `+te` suffix in any payload's `policy_id`, on draws 0 to 3 of seed 20261300,
20 episodes and 1000 parts per cell. These are the same draws as the round 1 ladder and the run
g re-read. The launcher `ladder_r1_resume.sh` is new and runs all four rungs at once, four lanes
each, with port bases 100 apart because `gate_te.sh` puts its aggregating call on `PORT_BASE`
plus 90. Sixteen lanes put the load average at 33 and the GPU at 47 to 61 percent, above the
twelve-lane ceiling this host measured, so every lane runs slower and the whole ladder still
finishes in one wave. The heap read is bound by the MuJoCo step and not by inference, so a lane
count above the core count buys nothing but context switches.

The check and the adoption: the top rung of the whole round 1 ladder, the four new rungs against
the old 600k, then runs on draws 0 to 11, 3000 parts. The bar is the run g selection read of
0.4198 plus 0.03, which is 0.4498. Round 1 is adopted only if that checked read reaches 0.4498.
A miss ends the rounds at one. 600k already has its checked read at 0.3463.

The ladder is running. Its numbers do not exist yet.

| rung | stowed of 1000 | fraction | cycles per stowed |
| --- | --- | --- | --- |
| 700k | pending | pending | pending |
| 800k | pending | pending | pending |
| 900k | pending | pending | pending |
| 1000k | pending | pending | pending |

## 9. Two launcher defects of the 2026-09-20 session

Both were fixed and both cost machine time only.

1. `gate_te.sh TE=none` dropped the ensembling flag but kept `--n_action_steps 1`. That is the
   no-averaging cell of section 6, which reads 0.0030. Three round 1 rungs read 0 of 1000
   before the payload check caught it. `TE=none` now passes neither flag. The rollouts were
   never affected, because `round_te.sh` drops both.
2. `train_act.sh` writes `action_space.json` at the run root only when the run exits. A read
   against a mid-training checkpoint therefore finds no marker, and the policy server falls
   back to the binary contract. The marker was written by hand with the values the trainer
   writes at exit. The resumed run exited normally, so its marker is the trainer's own.

## 10. What is still open

- The resumed round 1 ladder of section 8, and the adoption call at 0.4498 on its checked read.
  Round 1 has already missed that bar at 600k with 0.3463. Design 5.5 and the ledger stop the
  rounds at the first miss and at three, so the resumed ladder is the only thing keeping the
  rounds open. A miss there ends them at one.
- The gate. The best read of the campaign is run g at 400k, 0.4198 on 6000 parts. Round 1's
  checked read is 0.3463 on 3000 parts. Both sit against a gate of 0.85, and nothing has
  cleared the 0.50 R-GATE floor.
- The official read on seed 20261400. The MS3 selection rule L3, which the ledger keeps in its
  gate protocol section, still owes the selected checkpoint one read. The owner held it back
  because the selection numbers already answer the R-GATE question.
- The binary control. `base_b` is not gated, so the continuous gripper has no A/B partner. A
  relabelled binary run would restore it at the cost of one conversion and four hours.
- Why the cycles fail. The measurement lives in the campaign ledger under "The failed-cycle
  measurement of run g at 400k (2026-09-20)", which records its provenance. The commanded gap
  per cycle has a median of 10 to 32 mm, and the
  teacher's matched landing gap for a 7.8 mm part is 15.8 mm, so the opening scale is right.
  Failed cycles close the jaws to 1 to 5 mm, which is the teacher's own close command, at a
  place where the part is not. The stowed fraction is flat across part widths: 7.8 mm parts
  read 0.19 to 0.28 and 15.8 mm parts read 0.255 to 0.258. The open question is whether a
  15.8 mm landing gap leaves any room for a student's approach error. A teacher variant that
  widens the landing opening for narrow parts is the framed alternative to a further round. It
  partly reverses the matched opening of the MS6 phase A rework and costs about six hours of
  re-recording.
- The open-loop ruling is a serving decision and not an explanation. A policy that must execute
  whole 100-step chunks cannot correct inside a chunk, and the real arm will need a protocol
  that can. Design 5.1 records the measurement, and MS5b needs the reason.
- The deferred minors of the three pull requests are listed in the execution ledger
  `.superpowers/sdd/2026-09-18-ms6-phase-b-plan/progress.md`. None of them blocks the campaign.
