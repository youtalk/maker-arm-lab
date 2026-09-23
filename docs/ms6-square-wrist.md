# MS6 phase B, campaign n: the release rule, the squeeze floor and the wrist squaring

Date: 2026-09-23. The owner set a goal on 2026-09-22 at 10:45 PDT: a learned heap-cleanup
policy that stows at least 0.80 of the parts. The deadline was 2026-09-24 at 10:45 PDT. The
task stayed as it was: the bare-desk heap of `docs/ms6-heap-policy.md`, the held-out split, K
of 50 and one bin. The approach was free. The campaign ledger is
`~/data/maker-arm-ms6/n/README.md`. Every rule below went into that ledger before its number.

## 0. Read this first

The goal is met. The checked read stows 2711 of 3000 parts, 0.9037, at 1.231 cycles per stowed
part. Its Wilson 95 % interval is 0.8926 to 0.9137. The official seed 20261400 confirms the
pass at 2759 of 3000, 0.9197 (0.9094 to 0.9289), with 1.198 cycles per stowed part. The
previous campaign ended at 0.6320 on the same checked read (`docs/ms6-target-context.md`).

The passing number is a learned policy plus two serving rules. The policy is T0 at 400k steps.
T0 starts from campaign m's 400k checkpoint and trains 400k more steps on a corpus recorded
under a new row switch. It runs open loop at 50 actions per forward. The two serving rules edit
the policy's command before the env receives it. The squeeze floor caps a closing gap command
at the teacher's close gap, the grasp width less 6 mm. The wrist squaring turns the wrist roll
toward the heading of the served target row on the final approach. It stops at the cycle's
first hold or lift.

Both rules take their targets only from the served context row. The floor reads the row's
width, and the wrist rule reads the row's center, heading and top. The task, the parts, the
split and the grader are unchanged. The one input that the rules take from simulator state is
the wrist rule's stop flag (section 5).

The wrist squaring moved the number most. On the four selection draws, the same checkpoint
reads 706 of 1000 without it and 902 of 1000 with it. A depth forensic on T0 at 250k found this
lever. The policy closed with its jaw a median 24 degrees off the part's nearest axis. The
teacher closes 2 to 4 degrees off. In 55 % of the failed pinches, a finger pad then rested on
top of the part. The pinch held the part too high, and the part stayed on the desk when the arm
rose.

The third change is the release rule. It is part of the protocol and does not edit the command.
The teacher and the eval loop now both switch to the next target when the jaws release a part
(section 2).

## 1. The rules and the starting point

The ledger fixed these rules before the first read:

| rule | value |
| --- | --- |
| metric | the checked read of campaign m: seed 20261300, draws 0 to 11, 3000 parts, `--parts_split heldout --parts_seed 20260915 --heap_k 50`, at-rest grader at episode end |
| pass | at least 2400 of 3000 |
| selection | draws 0 to 3 of the same seed (1000 parts) pick the checkpoint and the protocol |
| confirmation | the official seed 20261400 confirms a pass if time allows and never takes part in selection |
| record | every read states the loop commit, the `policy_id` and the split |
| cache | a read whose payload came from the supervisor cache is not a read |

A census class names the last stage that a cycle reached. `held_no_lift` means that the jaws
held a part and the part never rose 5 cm. `close_miss` means that the jaws closed and held
nothing. `no_close` means that the command never closed. `carrying` and `lifted_lost` are
lifted parts that the cycle did not stow.

The campaign starts from the last number of campaign m. Its 400k checkpoint, served open loop
at 50 actions per forward, read 1896 of 3000, 0.6320 (Wilson 0.615 to 0.649). The census of
that read over 3788 cycles:

| stowed | held_no_lift | close_miss | carrying | lifted_lost | no_close |
| --- | --- | --- | --- | --- | --- |
| 0.412 | 0.278 | 0.236 | 0.032 | 0.029 | 0.011 |

Two thirds of the retreat waits hit their 300-step cap, and the waits took 26 % of all steps.
22.6 % of the parts were never targeted. On the selection draws, the same checkpoint reads 642
of 1000 (189, 188, 130 and 135) under campaign m's loop. That is the baseline of the early
reads below.

The first lever of the ledger was more training. Campaign m's run resumed from 400k, and its
500k checkpoint reads 687 of 1000 on the selection draws under the old loop. That is 0.045 over
400k. The resume stopped at 515k to free the GPU and the CPUs for T0 and T1.

## 2. The release rule, on both sides

The previous report named the row switch as the next lever. The teacher's corpus switched the
target row at the end of RETREAT. The eval loop waited for a retreat that the policy often did
not make. This campaign moves the switch to the release on both sides.

On the teacher side, `HeapOracle` pre-selects its next target when RELEASE ends. It uses the
same ranking, failure memory and clearance model as HOVER. The rig serves the new row and
marker from the first RETREAT frame. HOVER keeps the choice while the part is still a
candidate, and ranks it again if the part moved more than 1 cm. A smoke of the rule stowed 143
of 150 parts, 0.9533, at 1.049 cycles per stowed part. All 143 row switches fell on the RELEASE
to RETREAT frame.

The eval loop mirrors the rule with these switches:

- `--select_at_release` selects the next target when the measured jaw opening reads 0.8 open
  after a stow. There is no retreat radius. After any other cycle end, it selects at once.
- Under `--select_at_release`, a release after a lift also ends the cycle. The teacher
  pre-selects at every RELEASE end, including a release outside the bin.
- `--replan_on_switch` sends a per-env replan flag after the loop writes a new row. The policy
  server keeps one action queue per env and refills a flagged queue at once.
- A cycle that reaches its step budget with a part in the jaws stays open for up to 400 more
  steps. Every teacher cycle ends with a RELEASE, so no demonstration shows a new row during a
  carry.

Before the per-env queues, the whole batch replanned in lockstep. A new row then reached the
policy 0 to 49 steps late. The sub-batch forward reads 0.0 max abs difference against the old
path on the m 400k checkpoint.

The first read of the rule lost. On campaign m's 400k checkpoint it read 595 of 1000 (140, 163,
154 and 138) against 642. On draw 0, the wait fell from 12000 to 2200 steps per episode. That
policy learned the old switch, so the ledger read the loss as a mismatch. Every policy trained
on the new corpora reads under the new rule. On the checked read, the mean wait is 52.8 steps
per cycle.

## 3. The corpora and the training runs

The new corpora use the release rule. The first three hold 60 heaps each, K of 50. The corpus
v3 waves ran on Machine C.

| corpus | recipe | stowed | fraction | cycles per stowed | out of reach | dataset |
| --- | --- | --- | --- | --- | --- | --- |
| corpus_v2 | plain | 2666 of 3000 | 0.8887 | 1.110 | 231 | dataset_v2: 2743 segments, 1,266,323 frames |
| corpus_v2i | induce 0.12, noise 2.0 | 2453 of 3000 | 0.8177 | 1.092 | 125 | dataset_v2i: 2534 segments, 1,494,238 frames |
| corpus_v2r (remote) | induce 0.12, noise 2.0 | 2461 of 3000 | 0.8203 | 1.097 | 138 | dataset_v2r: 2529 segments, 1,525,131 frames |
| corpus v3, wave 1 | plain | 5551 of 6000 | 0.925 | 1.055 | | dataset_v3a: 5626 segments |

Campaign m's corpus read 0.9237 with 136 parts out of reach. corpus_v2 reads 0.035 under it,
and wave 1 of corpus v3 reads level with it. The converter keeps only successful cycles either
way. `--induce_prob 0.12` makes a deliberate first-attempt miss that the REOPEN retry recovers.
`--noise_scale 2.0` is the MS3 noise recipe. The ledger added both because the policy's
failures repeat and the corpus holds almost no recovery.

Five training runs followed. All use LeRobot ACT with the 16-wide `camera_box_target` context
and the continuous gripper.

| run | start | data | steps | host |
| --- | --- | --- | --- | --- |
| T0 | campaign m's 400k | dataset_v2 | 400k, batch 8 | local RTX 5090 |
| T1 | scratch | v2 + v2i + v2r: 7806 segments, 4,285,692 frames | paused after its 200k read | local |
| T2 | campaign m's 400k | dataset_v2r | paused after 200k | remote RTX 5080 |
| T3 | T0 at 100k | v2 + dagger_r1: 5575 segments, 2,344,559 frames | 300k | remote |
| T4 | scratch | dataset_v4 = v2 + v3a + dagger_r1: 11201 segments, 4,950,094 frames | 800k | local |

T0 stopped at 100k to free the host for T1 and the reads, then resumed from 100k to 400k.
Campaign m's dataset is never mixed into this training. Its retreat frames carry the old row
switch.

## 4. The squeeze floor

Two forensics on campaign m's 400k checkpoint looked at `held_no_lift` first. The first traced
draw 0, which stowed 194 of 250. There, 58 of 80 `held_no_lift` cycles closed near the desk.
All 58 raised the pick point at least 5 cm, and the part rose a median 0.3 mm. The pick point
sat more than 3 mm above the part's center in 41 of those 58 closes. Only 39 of 127 stows
closed that high.

The second forensic read 305 `held_no_lift` cycles under the release rule. The grasp was
centered: 1.1 mm along the closing axis and 0.2 mm across it. The arm rose 10.8 cm, and the
part rose 0 mm. The gap command at the close sat a median 1.3 mm under the grasp width. The
teacher commands 6 mm under the width at every close. The jaw drive's force is its stall
distance times its stiffness. The policy therefore squeezed at about a fifth of the teacher's
force.

`--squeeze_floor` caps a continuous gap command under the row's width at
`grasp.close_gap(width)`, the width less 6 mm. It leaves a command at or above the width alone,
and also a command with no live row. The teacher commands this exact gap, so the rule changes
no recorded frame.

| checkpoint | loop | floor | stowed of 1000 | held_no_lift | close_miss |
| --- | --- | --- | --- | --- | --- |
| m 400k | old | off | 642 | 321 | 282 |
| m 400k | old | on | 704 | 331 | 238 |
| m 500k | old | off | 687 | 327 | 277 |
| m 500k | old | on | 671 | 343 | 270 |

The floor acted on 1160 of 1236 cycles of the 400k read, at a median squeeze margin of -6.0 mm.
It gained 62 parts on 400k and lost 16 on 500k. It fixed marginal closes and left
`held_no_lift` where it was. The ledger adopted it for every later read. A weak squeeze was
only part of the shallow pinch.

## 5. The wrist squaring

### 5.1 The depth forensic

The depth forensic traced every step of T0 at 250k on draws 0 and 1 of seed 20261300. Each draw
ran 3 envs with the floor and the release rule on. It read the pick point's height at the close
that made the first hold. That close is the last close edge before both fingers touch the
target at more than 0.5 N each. For a `close_miss` it is the first close. All heights are in mm
above the part's center, and all angles are between the jaw axis and the part's nearest
horizontal axis.

| class | closes | achieved height, median | commanded height, median | jaw angle, median |
| --- | --- | --- | --- | --- |
| stowed | 159 | +0.4 | -0.1 | 15 deg |
| held_no_lift | 116 | +4.5 | +1.4 | 24 deg |
| held_no_lift, flat parts | 41 | +4.8 | +0.5 | 30 deg |
| close_miss | 71 | +7.2 | +7.5 | 22 deg |
| teacher, corpus_v2 | 683 | 0.0 | 0.02 under achieved | 2 to 4 deg |

The teacher closes at the part's center, and its command meets the achieved height. Only 5 % of
its closes sit more than 3 mm high.

The forensic split the 116 `held_no_lift` closes by where the extra height came from:

| verdict | closes | share | evidence |
| --- | --- | --- | --- |
| blocked | 64 | 0.55 | the arm is at rest, the command is +0.7 mm and the pick point is +4.6 mm. 63 of 64 touch the target, with 0.9 of the contact force vertical |
| chosen high | 50 | 0.43 | the command itself is +3.4 mm |
| lag | 2 | 0.02 | the arm is still coming down at the close |

In the blocked closes, a finger pad rests on the target's top. The flat parts are almost all
blocked: brick_2x7 lying flat is blocked in 16 of 17 `held_no_lift` closes, at a command of
+0.5 mm. `close_miss` is chosen high in 85 % of its closes, with a command of +7.5 mm.

The jaw angle separates the classes. Over the stowed and `held_no_lift` closes:

| jaw angle | closes | held_no_lift share |
| --- | --- | --- |
| under 10 deg | 85 | 0.29 |
| 10 to 20 deg | 66 | 0.33 |
| 20 to 30 deg | 48 | 0.54 |
| 30 deg or more | 76 | 0.57 |

A pad lands on a part's top when the jaw crosses the part at an angle. The target row already
carries the heading that the teacher closes across. At the close, the policy's jaw sits a
median 53 to 57 degrees off that heading on `held_no_lift` and 32 to 47 on stows. The policy
does not use the row's heading.

The close gate of this campaign never acted in this band (section 7). It holds the jaws open
only while the pick point is more than 3 mm above the row's top. The shallow pinches close 3 to
6 mm above the part's center, below that line.

### 5.2 The rule

`--square_wrist` edits the wrist roll command, `link_007_joint`. The jaw follows that joint one
for one. The rule acts on a step when all of these hold:

- The cycle has a live row (width above 0).
- The pick point is within 40 mm of the row's center in xy.
- The pick point is below the row's top plus 40 mm.
- The cycle has not yet held a part or lifted one 5 cm.

On such a step, FK of the policy's command gives the jaw heading and the sign of the roll's
effect on it. The error to the row's heading folds into a half turn, because the parallel jaw
is symmetric under a half turn. If the target roll leaves the joint's limits, the rule uses the
other half-turn branch. If both leave them, it leaves the command alone. The served roll moves
at most 3 degrees per step toward the target. At the first hold or lift, the policy's own roll
returns at once.

The 40 mm radius comes from the same trace. The final descent starts a median 17 to 21 mm from
the row's center, p75 29 to 37 mm. A 40 mm radius starts the correction no later than the
descent in about 80 % of cycles.

The rule reads the served row, the policy's command and the arm's measured tool pose. Its stop
flag reads the loop's cycle state: a held part from the simulator's contacts, or a 5 cm rise of
the held part. That flag is the one input from simulator state. The real arm needs its own hold
signal there.

### 5.3 The selection reads

Both reads use T0 at 400k, 50 actions per forward, `--select_at_release --replan_on_switch
--squeeze_floor`, draws 0 to 3 of seed 20261300.

| serving | stowed of 1000 | fraction | Wilson 95 % | cycles per stowed | per draw |
| --- | --- | --- | --- | --- | --- |
| floor | 706 | 0.7060 | 0.6770-0.7334 | 1.806 | 176, 181, 190, 159 |
| floor + wrist | 902 | 0.9020 | 0.8820-0.9189 | 1.253 | 229, 236, 222, 215 |

The census of the same two reads, as counts and as a share of all cycles:

| serving | cycles | stowed | held_no_lift | close_miss | lifted_lost | no_close | carrying |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor | 1275 | 695 (0.545) | 335 (0.263) | 176 (0.138) | 43 (0.034) | 16 (0.013) | 10 (0.008) |
| floor + wrist | 1130 | 871 (0.771) | 90 (0.080) | 123 (0.109) | 30 (0.027) | 12 (0.011) | 4 (0.004) |

The rule acted on 98 % of cycles, a mean of 124 steps per cycle. Stowed cycles used a median of
90 squared steps and `close_miss` cycles 277. `held_no_lift` fell from 335 to 90 cycles. The
cycles that remain still close high, a median 5.8 mm above the center. The rule does not change
the depth that the policy chooses. Without the rule, 13 of 20 episodes ended on the 65-cycle
budget. With it, 2 of 20 did.

## 6. The checked read and the official read

The checked read ran T0 at 400k, 50 actions per forward, with `--select_at_release
--replan_on_switch --squeeze_floor --square_wrist`. The loop is commit 8ea9789 on branch
`ms6-v2-square`, and its tree is the same as 22fc4e5 of this stack. The `policy_id` is
`act:~/data/maker-arm-ms6/n/train_t0/checkpoints/400000/pretrained_model+nas50+ctx16+gap`,
and the split is the held-out one.

The first launch at 00:28 served draws 0 to 3 from the selection read's cache. The ledger says
a cached payload is not a read. That launch stopped at 00:36 and restarted fresh as
`gate/sel/t0_400000_sqw_chk`. All 12 draws ran on 12 lanes from 00:36 to 02:03. Each lane log
holds its own `GATE_OK`. The "reusing" lines in `final.log` are the supervisor's last pass,
which sums the payloads the lanes wrote. The official read ran the same way from 02:05, as
`gate/off/t0_400000_sqw_off`.

| read | seed | draws | stowed | fraction | Wilson 95 % | cycles per stowed |
| --- | --- | --- | --- | --- | --- | --- |
| checked | 20261300 | 0 to 11 | 2711 of 3000 | 0.9037 | 0.8926-0.9137 | 1.231 |
| checked, draws 0 to 3 | 20261300 | 0 to 3 | 919 of 1000 | 0.9190 | 0.9004-0.9344 | 1.205 |
| checked, draws 4 to 11 | 20261300 | 4 to 11 | 1792 of 2000 | 0.8960 | 0.8819-0.9086 | 1.245 |
| official | 20261400 | 0 to 11 | 2759 of 3000 | 0.9197 | 0.9094-0.9289 | 1.198 |

Per draw, 250 parts each:

| read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checked | 240 | 237 | 223 | 219 | 229 | 231 | 217 | 232 | 225 | 221 | 216 | 221 |
| official | 228 | 239 | 235 | 231 | 229 | 234 | 226 | 232 | 230 | 214 | 229 | 232 |

Draws 4 to 11 read 0.896, so the selection draws carried little luck this time. In the previous
campaign, the 200k checkpoint read 0.60 on draws 0 to 3 and 0.46 on draws 4 to 11. The same
code on the same four draws read 902 and 919. The rollout does not repeat exactly from one read
to the next.

The phase B gate of `docs/ms6-heap-policy.md` is 0.85 stowed and at most 1.3 cycles per stowed
part, over 120 episodes. The supervisor ran both reads with those two thresholds, and both end
on `GATE_OK`. Each read holds 60 episodes, so neither is that gate's read.

The census of the two reads:

| read | cycles | stowed | held_no_lift | close_miss | lifted_lost | no_close | carrying | no_open |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checked | 3338 | 2634 (0.789) | 280 (0.084) | 301 (0.090) | 72 (0.022) | 40 (0.012) | 10 (0.003) | 1 |
| official | 3306 | 2661 (0.805) | 277 (0.084) | 259 (0.078) | 78 (0.024) | 20 (0.006) | 11 (0.003) | 0 |

The census counts cycles. The grader counts parts at rest in the bin at the episode's end. The
two stowed counts therefore differ. On the checked read, 55 of 60 episodes ended with no
candidate left for the selector. 4 stowed all 50 parts, and 1 ended on the cycle budget. The
official read has 54, 5 and 1. The `held_no_lift` cycles of the checked read still close a
median 5.3 mm above the center.

## 7. Negative results

### 7.1 The induction and noise corpora

T2 is the same warm start as T0 on the induction corpus alone. T1 trains from scratch on the
plain and induction corpora together, about one part plain to two parts induction.

| run | data | 50k | 100k | 200k | no_close cycles |
| --- | --- | --- | --- | --- | --- |
| T0 | dataset_v2 | 547 | 597 | 636 | 50, 10, 17 |
| T2 | dataset_v2r | 297 | 292 | | 469 and 489 of 1300 |
| T1 | v2 + v2i + v2r | | | 352 | 255 of 1282 |

On the noise-scaled corpora, the policy often does not commit to a close. `no_close` takes 0.36
to 0.38 of T2's cycles and 0.20 of T1's. The ledger read this as a poisoned continuous gripper
and dropped both corpora. The read does not separate the induction from the noise.

### 7.2 DAgger round 1 teaches hesitation

Round 1 ran campaign m's 500k checkpoint under the release rule on 16 clients with `--trigger
failure` and no floor. 15 of 16 clients ended `DAGGER_OK`. The round holds 75 episodes, 3510 of
3750 parts stowed by the mix of policy and teacher, 0.936. The teacher took over 2942 times,
stowed 2797 of them (0.951) and declined 141. The triggers were `closed_no_pinch` 1990,
`held_no_rise` 359, `half_budget_no_lift` 296, `close_far` 220 and `no_close` 77. The converted
round, dataset_dagger_r1, holds 2832 segments and 1,078,236 frames.

T3 starts from T0 at 100k and trains on v2 plus the round:

| run | step | stowed of 1000 | held_no_lift | close_miss | no_close |
| --- | --- | --- | --- | --- | --- |
| T3 | 100k | 587 | 299 | 206 | 178 |
| T3 | 200k | 500 | | | |
| T3 | 300k | 528 | 309 | 282 | 124 |
| T0 | 200k | 636 | 406 | 194 | 17 |
| T0 | 300k | 594 | 359 | 295 | 6 |

The recovery frames trade `held_no_lift` for `no_close`. T0's `no_close` stays at 6 to 50 over
its whole ladder. T4 holds the round at 22 % of its frames. Its rungs read 464, 452 and 546 at
200k, 400k and 600k, with `no_close` at 80, 199 and 210. The ledger dropped T3, and T4 ran to
its end only as a hedge.

### 7.3 A shorter horizon

T0 at 250k reads 536 of 1000 at 25 actions per forward, against 631 at 50. The previous
campaign found the same loss under the old loop. The rung at 100 actions was never launched,
because the agent that owned it stopped.

### 7.4 The close gate

`--close_gate` holds the jaws at the teacher's landing gap, the width plus 8 mm. It acts while
the pick point is still coming down and is more than 3 mm above the row's top. It targeted the
mid-air closes: a quarter of the shallow pinches closed first more than 3 cm above the part.

The first version fired during the carry. The row keeps the target's desk position, so a closed
command over the bin met every condition. On m 500k with the floor, under the old loop, the
gate read 461 of 1000 against 671. `lifted_lost` rose from 27 to 187. The fix stops the gate at
the cycle's first hold or lift.

On T0 200k with the floor, the fixed gate reads 631 against 636. `held_no_lift` fell from 406
to 362, `close_miss` rose from 194 to 211 and `no_close` from 17 to 37. The gate acted on 435
of 1281 cycles. The `held_no_lift` closes still sat a median 5.3 mm above the center, against
5.2 mm without the gate. The gate stays off. The depth forensic later showed why it did not
help: half of those closes were blocked by a pad, not chosen high.

### 7.5 The row refresh

`--row_refresh` rewrote the row from the live scene at an open edge that follows a close
without a lift. That is the teacher's REOPEN refresh. Jaw chatter fired it 21.5 times per
cycle. Draws 0 and 1 read 50 and 85 of 250, against 140 and 163 without it. The read stopped
after two draws, and the switch is dropped.

### 7.6 The T0 ladder is noisy

Every rung reads under the release rule and the floor, without the wrist rule:

| step | stowed of 1000 | cycles | held_no_lift | close_miss |
| --- | --- | --- | --- | --- |
| 50k | 547 | 1290 | 392 | 252 |
| 100k | 597 | 1289 | 366 | 261 |
| 100k, re-read | 606 | 1295 | 370 | 278 |
| 150k | 422 | 1263 | 358 | 444 |
| 200k | 636 | 1290 | 406 | 194 |
| 250k | 631 | 1282 | 347 | 255 |
| 300k | 594 | 1286 | 359 | 295 |
| 350k | 594 | 1283 | 446 | 212 |
| 400k | 706 | 1275 | 335 | 176 |

The first 100k read ran before the carry grace. Every later rung ran with it. Neighboring rungs
differ by up to 214 parts (150k and 200k). The 400k rung reads 112 parts above 350k. The ledger
picked 400k as the best rung, and the wrist rule was read on 400k only.

## 8. The code

The campaign's code is a stack of six branches on main:

| branch | commits | what it adds |
| --- | --- | --- |
| `feat/ms6-release-select` | 2d2e316, f46034d, ba83acd, f6befd6, bba466f | the per-cycle forensic export, the replan flag and per-env action queues, `--select_at_release`, `--replan_on_switch`, `--row_refresh`, and the gate supervisor's `--client_flags` with a cache that matches on the loop switches |
| `feat/ms6-teacher-preselect` | 688e006, 29b8283, 57cdce5 | the teacher's pre-selection at RELEASE end, the pre-selected row and marker through RETREAT, and the loop's cycle end at a release after a lift |
| `feat/ms6-squeeze-floor` | f3df0e0 | `--squeeze_floor` and `grasp.close_gap` |
| `feat/ms6-heap-dagger-trigger` | 749faf3, b6ed900, 12c7571, 236802d | the `held_no_rise` trigger, the loop switches on the heap DAgger client, takeover frames that follow the rig's row, and the resumed-LIFT row fix |
| `feat/ms6-close-gate` | 2318ce5, 216180f, 76a3478 | `--close_gate`, the carry grace and the gate's stop at the first hold or lift |
| `feat/ms6-square-wrist` | 22fc4e5 | `--square_wrist` (`loop.squared_roll`) |

Every switch is off by default, so the old protocol stays reproducible. The payload's
`inference.loop` block records every switch, and the census exports `floored_steps`,
`gated_steps`, `squared_steps` and `grace_steps` per cycle. The reads ran from the worktree
`~/src/maker-arm-lab-n5` at 8ea9789, where the CPU suite read 892 passed and ruff was clean.

The campaign ran on three hosts: Machine A (the local RTX 5090), Machine B (a remote RTX 5080)
and Machine C (a 128-core aarch64 machine with an A2000).

## 9. What is still open

- A learned fix for the heading. The policy ignores the row's heading, and the wrist rule
  stands in for it. A policy that squares its own jaw is the next research question.
- The wrist rule on other checkpoints. It was read on T0 400k only.
- The stop flag. The wrist rule stops at the first hold or lift, which the loop reads from the
  simulator. The real arm needs its own hold signal for it.
- The chosen-high closes. At 250k they were 43 % of `held_no_lift`, and the wrist rule does not
  touch them. On the checked read, `held_no_lift` is 0.084 of cycles and `close_miss` 0.090.
- The hedges. T4 trained to 800k, and its 800k rung has no result yet. dataset_v3b (2853
  segments) sits on Machine C, and wave 3 records into dataset_v3c there. T5, a warm
  start from T0 400k on v2 plus v3a, is written and never started. The goal did not need them.
- The phase B gate. Its 120-episode read at 0.85 did not run under the new serving rules.
- The horizon. 100 actions per forward was not read under the release rule.
- The real arm. The simulator reports the target row exactly. A noise model on the row, in
  training and at serving, is still to do.
