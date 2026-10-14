# MS7i: local rounds, round 1

Date: 2026-10-14. Status: draft. This draft uses the ledger through its entry of 06:18 on
2026-10-14. Round 1 runs from 10-13 to 10-19. The collection, the labels, the audit, the S
reference reads and Track ACT are final for round 1. Track V and the choice of the round 2
levers are pending. Each pending item has a line that starts with "Pending:".

The design of MS7i is revision 3 of the design document of 2026-10-13. The owner started the run on
it with /goal at about 11:35 on 2026-10-13. This report cites the design as "spec section N" and
the round 1 plan of the same day as "plan section N". It restates every rule that it uses. It cites
the MS7h report, `docs/ms7h-beyond-the-oracle.md`, as "MS7h report section N". The campaign ledger
is `~/data/maker-arm-ms7/i/README.md`, and `STATE.md` beside it is the resume pointer. Every rule
went into the ledger before its number. Each read has a TAG, its name. The payloads of a read are
under `~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7i TAG starts with `i_`.

The names. The ledger names its rules IR1 to IR10. Names such as HD60 are deviations of the MS7h
ledger. The policies and drivers:

- S is the final MS7g stack: F5 200k with `--carry_guard --overhang_guard`, the classifier v10e3
  with `--cls_flip` and the final-only vote (MS7h report section 1). Every read pairs with S.
- F9n is the MS7g recipe from random init with the context fix. B is F9n 800k, served with the
  switches and the classifier of S.
- V is the fresh 30k save of π0.5, `h/vla/fresh/30000_n25`, served at a0j1slR25g20 (MS7h report
  section 7.15).
- ACT-C is ACT trained from scratch on the F9n recipe, on the corpus plus the own frames of B that
  succeeded (section 5).
- Run A and Run B are the two π0.5 fine-tunes of Track V (section 6).

Own frames are the frames that a driver policy served itself. Takeovers are the frames of the
scripted teacher when it takes over a cycle.

The measures. A correct stow, the rate and the held-out row keep their MS7f definitions (MS7f
report section 1.3):

- A correct stow is a part at rest in a box of its category.
- The rate is the correct stows per simulated hour.
- The held-out row counts the parts of the kinds that no policy and no classifier saw in training.

D is the sum over the heaps of the candidate's correct stows minus those of S, on the same seed and
draws. The paired SE is the standard error of D, from the differences heap by heap. A standard
error is the spread of a measured value. `i/bin/s1_score.py` computes both.

The times. All times are PDT. Each time is the stamp of the ledger entry that records the event. A
time without a date from 11:41 to 23:59 is on 2026-10-13. A time without a date from 00:00 to 06:18
is on 2026-10-14.

The numbers. Every number comes from the ledger or from a file that this report names. A few
totals are sums of those numbers.

## 1. Summary

MS7i runs in weekly rounds until the real arm arrives (spec section 4.1). Each round ends with a
report on a Monday, and the owner then chooses the levers of the next round. Round 1 has two
tracks. They share one collection and one set of label rules:

- Track V fine-tunes π0.5 twice on the second Colab account, as Run A and Run B (section 6).
- Track ACT trains ACT-C on the RTX 5090 of Machine A (section 5).

The collection ran on 10-13 from 14:44 to about 20:08, a day earlier than the plan (section 3). V
drove 50 heaps on Machine A and Machine B, and B drove 50 heaps on Machine C. Every driver served at
warp 1.0 and recorded every frame. The shards of 64 heaps form the training roots. The conversion
of the other 36 heaps used label rules that the audit later changed. Their raw shards are gone, so
they stay out of training.

The labels follow IR5 (section 4). The Claude audit ran four rounds and forced five rule changes.
The final rules read 3 disagreements in 100 sheets, 0.03, against a bar of 0.10.

Track ACT ended its round 1 path at the re-read (section 5). The three screens of ACT-C stowed
2512, 2502 and 2498 of 3000 parts correctly, against 2533 for S. The best screen, 400k, then read
draws 20 to 39. It stowed 2464 there, against 2526 for S, so D = -62 at a paired SE of 33.7. IR2
stops a candidate that falls under S on these draws. So ACT-C reads no checked read in round 1.

The screen gap of ACT-C 400k is noise: D = -21 is -0.60 paired SE (section 5.5). Its extra losses
are opens over the right box at the end of long carries. The parts land on the desk past the far
wall. One heap, draw 16, holds 12 of the 21 lost parts.

The reads of round 1 so far:

| read | policy | seed, draws | correct of 3000 | rate | held-out row | D against S, paired SE |
| --- | --- | --- | --- | --- | --- | --- |
| h_S_scr | S | 20307000, 0-19 | 2533 | 369.46 | 581 of 744, 0.7809 | |
| i_C400_scr | ACT-C 400k | 20307000, 0-19 | 2512 | 398.17 | 594 of 744, 0.7984 | -21, 34.8 |
| i_C600_scr | ACT-C 600k | 20307000, 0-19 | 2502 | 364.37 | 582 of 744, 0.7823 | -31, 44.0 |
| i_C800_scr | ACT-C 800k | 20307000, 0-19 | 2498 | 381.47 | 604 of 744, 0.8118 | -35, 33.7 |
| i_S_d2039 | S | 20307000, 20-39 | 2526 | 360.92 | 642 of 801, 0.8015 | |
| i_C400_d2039 | ACT-C 400k | 20307000, 20-39 | 2464 | 373.63 | 613 of 801, 0.7653 | -62, 33.7 |
| i_S_chk | S | 20307600, 0-19 | 2535 | 366.93 | 597 of 763, 0.7824 | |

Track V waits for the owner (section 6). Its inputs reached the Shared drive at 00:03 on 10-14. A
probe of the carry-guard margin runs now (section 5.6). Section 7 lists the round 2 levers for the
owner.

## 2. The bars and the candidate path

Spec section 4.2 sets the path of a candidate, and IR2 restates it. The same path holds for both
tracks. Revision 3 made the bars harder to pass by luck. Spec section 3 gives the reason. S read
2548, 2545, 2533, 2525 and 2507 on five reads. Only one of them had a held-out row of 0.80 or
more. At zero true gain, a candidate at the level of S passes the old checked bars 10 to 19 % of
the time.

### 2.1 The seeds

IR1 names every seed before any read uses it:

| use | seed | draws or drivers |
| --- | --- | --- |
| collection | run 20380000 + p, layout 20380200 + p, train split | Machine A V p 0-24, Machine B V p 50-74, Machine C B p 100-149 |
| smokes | run 20380900 + i, layout 20380950 + i | i 0-9 |
| screen | 20307000 | draws 0-19, 3000 parts |
| re-read | 20307000 | draws 20-39, 3000 parts |
| round 1 checked read | 20307600: run 20307600000 + i, layout + 500 | draws 0-19, 3000 parts |
| official read | 20307200 | draws 0-39, 6000 parts, one shot |

The collection layouts deviate from the spec. Spec section 5.2 set layout 20380100 + p, which
equals the runs of p 100 to 149. At 11:59 no earlier ledger or payload held a seed in 20380000 to
20380249 or a read of 20307600. HD73 of MS7h reserved 20307600, and no read used it.

### 2.2 The path

1. Screen. The candidate reads the screen draws. Its reference is h_S_scr, 2533 correct stows.
2. Re-read. The best screen of each track reads draws 20 to 39. HD32 (4) picks the best screen:
   the correct total first, then the rate. S reads the same draws once, as i_S_d2039. The
   candidate goes on only if its total there is not under the total of S.
3. Checked read. The candidate reads the checked seed, beside i_S_chk on the same draws. Each
   track reads at most one checked read a round. The candidate must meet all four bars of the
   table below.
4. Official read. It runs only after a checked pass and the approval of the owner. It reads the
   official seed, one shot. The candidate must reach at least 5100 of 6000, 200 an hour, a
   held-out row of 0.80 and D > 0 against S.

| bar of the checked read | value |
| --- | --- |
| correct in a box of the category | at least 2550 of 3000 |
| correct stows an hour | at least 200 |
| held-out row | at least 0.80 |
| D against i_S_chk | more than 0 |

A pass of the official read is a gate pass. The claim "beats S" needs a D of at least 2 paired SE,
pooled over the checked and the official reads (60 heaps).

IR3 sets the reads. Deciding reads run only on x86 lanes, on Machine A and Machine B, through
`i/bin/read_in.sh`, at warp 1.5. ACT candidates serve with the switches and the classifier of S.

### 2.3 The S reference reads

S read the re-read draws and the checked seed on 10-13, at 7i-r0 e7dc7ae, on 10 lanes each
(21:56). The screen reference h_S_scr is the MS7h read at c6d4981 (MS7h report section 11.2).

| read | seed, draws | host | correct of 3000 | rate | cycles a stowed part | held-out row |
| --- | --- | --- | --- | --- | --- | --- |
| h_S_scr | 20307000, 0-19 | MS7h | 2533, 0.8443 | 369.46 | | 581 of 744, 0.7809 |
| i_S_d2039 | 20307000, 20-39 | Machine A | 2526, 0.842 | 360.92 | 1.286 | 642 of 801, 0.8015 |
| i_S_chk | 20307600, 0-19 | Machine B | 2535, 0.845 | 366.93 | 1.268 | 597 of 763, 0.7824 |

S itself does not pass the checked bars on seed 20307600. 2535 is under 2550, and 0.7824 is under
0.80.

## 3. The shared collection

### 3.1 The recorder and the warp

The recorder switch `--record_all` records every frame of an episode (spec section 5.1). Each frame
carries its cycle number `obs/loop_cycle`, its actor `obs/actor` and its subtask `obs/subtask`. It
also carries `obs/policy_action`, the reply of the server before the serving rules. The plan found
that the census had no takeover step, served box, release step or last-held step (plan section 2,
fact F8). So the census rows of an all-frames heap also hold `box`, `last_held_at`, `release_at`
and `final_box`. The switch changes no served step.

The review of the earlier revisions found that both trained on own frames with actions after the
1.5 time warp. The served action after `warp_chunk` is the action that `mj/env.py` records.
Training on it and serving at 1.5 again gives about 2.25 times the speed (spec section 3). So IR4
makes every driver serve at `--action_warp 1.0` during the collection. The recorded actions keep
the pace of the policy. The reads still serve at warp 1.5 (IR3).

### 3.2 The drivers and the hosts

IR4 sets the drivers:

| host | driver | lanes | p | heaps |
| --- | --- | --- | --- | --- |
| Machine A, RTX 5090 at 400 W | V, one π0.5 server | 10 | 0-24 | 25 |
| Machine B, RTX 5080 | V, one π0.5 server | 10 | 50-74 | 25 |
| Machine C, RTX A2000, aarch64 | B, one CUDA ACT server a lane | 18 | 100-149 | 50 |

- Both drivers run in DAgger mode. The teacher takes over a cycle when one of the six MS7f wave 3
  triggers fires. The teacher uses the trims of HD60 (1).
- Machine C ran 18 lanes, not the 20 of IR4. An ACT server holds 390 MiB of the A2000, and an EGL
  client holds 186 MiB. So 18 lanes need about 10.4 GB (14:44).
- Every frame from Machine C comes from aarch64 physics. A lane plays V at 21 to 26 steps a second
  and B at 11 to 12 (`i/tables/infra/smoke.md` section 1).
- The takeover roots drop the leading frames of each takeover in which the teacher opens jaws that
  the policy left closed. The converter does this with `--drop_reopen_lead 0.002`, the rule of
  HD60 (2).
- `dag_mask.py` never runs on all-frames shards. There it turns teacher frames into false policy
  frames (plan section 2, fact F7).
- The raw shards of Machine A go to `/var/tmp` on the root partition. Machine A's `/home` had
  176 GB free, and a V heap is 3.1 to 4.2 GB raw (plan section 2, fact F5).

### 3.3 The smokes

Rule (plan section 4): in every smoke heap, the frames must equal the steps. `obs/loop_cycle` must
run contiguous and increasing, and every cycle must join its census row. Each host ran 2 lanes and
2 heaps at warp 1.0 on the smoke seeds, at 54d1bb6. All six heaps passed. Every conversion printed
CONVERT_OK, and every video decoded (14:44).

The smokes also give the first failure shares at warp 1.0. Spec section 10 asks for this
comparison. HD60 is the MS7h DAgger collection of V at warp 1.5: 13 Machine A heaps and 1,863 cycles
(`smoke.md` section 4).

| share a cycle | HD60, V at warp 1.5 | V at 1.0, Machine B (278 cycles) | V at 1.0, Machine A (293 cycles) | B at 1.0, Machine C |
| --- | --- | --- | --- | --- |
| takeovers | 0.322 | 0.399 | 0.406 | 0.221 |
| first close off the target | 0.128 | 0.108 | 0.109 | 0.046 |
| re-open within 10 steps of the first close | 0.164 | 0.072 | 0.102 | 0.025 |

- At warp 1.0 the takeovers of V rise. The rise is all in the trigger closed_no_pinch, 0.173 a
  cycle on Machine B against 0.095 for HD60. That trigger fires on a squeeze under the 2 mm of the hold
  detector.
- The re-opens of V fall to half or less, and its off-target first closes fall a little.
- With 4 V heaps these are directions, not results. No collection of B at warp 1.5 exists.

### 3.4 The run

The plan set the collection from the evening of 10-14 to 08:00 on 10-15. At 13:30 the controller
moved the start to 10-13, as soon as the smokes passed, for a day of slack. The collection started
at 14:44 at ada4e5f, with a stop time of 08:00 on 10-14. Every host ended long before the stop:

| host | end | heaps |
| --- | --- | --- |
| Machine A | COL_OK at 19:31 | 25 of 25 |
| Machine C | COL_OK at 19:38 | 50 of 50 |
| Machine B | COL_OK at about 20:08 | 25 of 25 |

The x86 hosts used about 6 of 24 cores, so the conversion ran during the collection (16:16). IR4
deletes a raw shard only after its conversion passes the join check and the video decode check.

### 3.5 The shards that left training

An old-rule shard is a shard whose conversion used a label rule that the audit later changed
(section 4.4). At first the conversion loops deleted a raw shard once its conversion passed. So an
old-rule shard lost its raw data and cannot take the new labels. It stays out of every training
root:

| audit step | heaps out | Machine A | Machine B | Machine C | folder |
| --- | --- | --- | --- | --- | --- |
| rule changes (1) to (3), 17:06 | 21 | p1, p3, p5, p6, p7, p8, p9 | p50, p51 | 12 heaps of the first pass (16:41) | `conv/<host>/r1old` |
| rule change (4), 17:43 | 15 | p0, p2, p4 | p52, p53, p55 to p59 | p101, p103, p107, p112, p114 | `conv/<host>/r2old` |

From 17:43 the loops ran with KEEP_RAW=1, so no raw shard went until the audit passed. Rule change
(5) cost no heap. Its round 3 shards kept their raw files and took the new rule (18:22).

One more incident cost no shard. At 16:58 a remote stop matched its own ssh shell with `pgrep -f`.
It sent SIGTERM to the process group of tailscaled on Machine B and Machine C. tailscaled lived, and the
collections did not change. The ssh shells and an rsync of the roots of B died. The new rule: a
remote pgrep uses a bracket pattern, and a stop kills only a pgid from a job file.

### 3.6 The final roots

The final merges ran under the label rule of record, f20db17 (20:28 and 20:51). Each host gives
three roots. The labeled root holds every own frame with its label. The positive root holds the
positive own frames only. The takeover root holds the teacher frames.

| host | driver | shards | labeled: episodes, frames | positive: episodes, frames | takeovers: episodes, frames | positive, negative, untagged |
| --- | --- | --- | --- | --- | --- | --- |
| Machine A | V | 15 | 2,183, 552,002 | 1,264, 213,396 | 861, 247,551 | 0.3866, 0.0713, 0.5421 |
| Machine B | V | 16 | 2,386, 589,080 | 1,314, 218,394 | 1,075, 425,783 | 0.3707, 0.0814, 0.5479 |
| Machine C | B | 33 | 4,779, 1,116,765 | 3,264, 651,374 | 1,277, 366,718 | 0.5833, 0.0453, 0.3714 |

The label shares of each driver (20:51):

- V: positive 431,790, negative 87,266 and untagged 622,026 of 1,141,082 own frames. The shares
  are 0.378, 0.076 and 0.545.
- B: 0.583, 0.045 and 0.371.

The roots of each track:

- Track V (21:51): own_a holds the labeled frames of V, 4,569 episodes and 1,141,082 frames. Its
  subtask shares are reach 0.4805, grasp 0.1762, carry 0.2251 and release 0.1182.
- own_b_positive holds the positive frames of V, 2,578 episodes and 431,790 frames.
- The takeover root of Track V holds the MS7i takeovers of V and the v1 takeovers of HD60. It has
  3,296 episodes and 1,189,019 frames, with teacher labels positive 0.4427 and untagged 0.5573
  (`i/vla/assemble.log`).
- All three roots and the corpus prompt codes reached the Shared drive by 00:02. `xfer_drive.sh`
  hashed each file again on the drive copy (00:03).
- Track ACT (21:44): the positive root of Machine C (651,374 frames) and its takeover root (366,718
  frames). Section 5.1 gives their share of the mix.

IR5 also asks for the share of own frames whose served action a serving rule changed. It comes
from `obs/policy_action`. A frame counts as changed if a joint target moved more than 1e-6 rad
from the policy's action, or if the gripper column differs. The labels behind the table come from
`own_labels()` at f20db17, and they match the conversion on 64 of 64 shards
(`i/tables/served_share/served_share.md`).

| frames | V | B | pooled |
| --- | --- | --- | --- |
| all own frames | 0.380 (433,900 of 1,141,082) | 0.412 (460,577 of 1,116,765) | 0.396 (894,477 of 2,257,847) |
| positive | 0.405 | 0.382 | 0.391 |
| negative | 0.443 | 0.584 | 0.494 |
| untagged | 0.354 | 0.439 | 0.388 |

Pooled, 0.159 of the frames change only in the joints, 0.153 only in the gripper, and 0.085 in
both. The recorder writes no rule tag on a frame, so the split by rule comes from the sign of each
change and the census counters. The arm changes come from the IK re-solve of `--pick_snap`, with
`--square_wrist` on the same steps (about 0.24 of the frames each). A narrower gap comes from
`--squeeze_floor` (0.216) and the carry guard (0.0075). A wider gap comes from `--snap_hold`
(0.0153). The time warp of 1.0 changed no frame. So the policies of both tracks learn about 40 %
of their own frames from the serving rules, not from the driver's own output.

## 4. The labels and the Claude audit

### 4.1 The subtask

A cycle has four subtasks: reach, grasp, carry and release (`SUBTASKS` in `core/hold.py`). The
labels split a cycle at its first carry frame. A frame reads carry after a lift. A lift is a closed
jaw command, a measured gap over the command and a rise of 5 cm since the close.

The first form of the rule failed on the corpus (13:03). 5,938 of 23,957 corpus episodes, 24.8 %,
never read carry. 495 of them never read a close. In the rest, the measured gap sat a median 1.3 mm
over the command. That is under the 2 mm margin of the hold detector, so the lift test failed. In a
sample of 300 such episodes, 88 % rose 5 cm. The diagnosis at 13:30 found three causes:

- The left jaw alone cannot see a hold that rests on the right jaw.
- A regrasp on a wider axis closes above the width of the served row.
- The row switch at a release read a one-frame carry.

The fix at ada4e5f gives the subtask its own margins. The hold detector keeps its 2 mm:

- Closed: the command is under the width of the previous frame plus 4 mm (SUBTASK_CLOSED_MARGIN).
- Gripping: the gap is at least 0.5 mm over the command (SUBTASK_MIN_EXCESS).
- Rise: 5 cm.

With the fix, 83.3 % of the corpus episodes read a carry, against 75.2 % before. The strict order
reach, grasp, carry, release rose from 47.7 % to 63.0 %. The 16.7 % with no carry are mostly
regrasps on a wider axis. Round 1 uses this rule for training and serving. Section 7 lists the
rest as a lever.

### 4.2 The labels of IR5

The labels use the truth of the census, only in training (spec section 5.3). A policy frame gets
one of three labels:

| label | frames |
| --- | --- |
| negative | the 30 frames before a failing event |
| positive | a grasp segment with a hold and a lift after the first close, and no re-open. Also a carry segment that the policy ends itself, with the part at rest in the served box. |
| untagged | every other policy frame, with the carries that a handoff or a takeover finished |

The failing events:

- a first close off the target
- a re-open within 10 steps
- the last close of a cycle that ends with no lift
- the start of a takeover
- the loss of a lifted part
- the release of a part that comes to rest on a rim, the desk or the tray

IR5 fixes the open choices:

- Off target: off_z, or an xy distance over 20 mm at the first close.
- A re-open: a closed-to-open subtask edge within 10 steps of its close.
- No lift: a truth max_rise under 0.05 m.
- The labels drop the rig frames of a handoff and the rakes.
- A wrong box of the policy is untagged. That is the right served box, with the part at rest
  elsewhere. A wrong box of the classifier counts as a positive carry.
- At rest: the `final_box` at the end of the episode.
- A negative window wins over a positive segment.
- The trained action is the served action. A frame that a serving rule changed keeps its label.

The spec labels every teacher frame positive, in the corpus and in the takeovers. Rule change (3)
narrows that (section 4.4).

### 4.3 The audit

Rule (IR6): each round samples 40 cycles a driver, 10 for each pair of segment and label, and 20
corpus episodes. A sampler renders each as a sheet of 6 frames, and two judge subagents read the
sheets. If more than 10 % disagree with their labels, the rule changes. The audit then runs again
before any upload or training. The judges also confirm that each negative window ends at its event.

| round | time | sample seed | shards | code | disagree | result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16:58 | 20380990 | 7 Machine A V shards and the corpus | the first label rule | V 4 of 40, corpus 6 of 20, together 10 of 60, 0.167 | over the bar |
| 2 | 17:43 | 20380991 | Machine A p0, p2, p4 and the corpus | 6f70f55 | V 6 of 40, corpus 0 of 20, together 6 of 60, 0.100 | at the bar |
| 3 | 18:22 | 20380992 | Machine A p10, p13, p16, p17, p19 | 177d3e1 | V 2 of 40, 0.050 | pass for V |
| 4 | 19:19 | 20380993 | Machine C p116, p121, p126, p131, p135 | f20db17 | B 1 of 40, 0.025 | pass for B |

- Round 1 disagreed on 10 sheets. So rule changes (1) to (3) followed.
- Round 2 read exactly 0.100, which is not over the bar. But four of its six V disagreements were
  re-open windows on clean cycles. So rule change (4) followed, and round 3 audited V again.
- Round 2 also had two judge errors, V05 and V07. The truth says that the jaws held the target
  part 127 and 108 steps. It rose 0.10 m and landed in the right box. The part hid behind the
  jaws.
- At 17:07 the judge prompt gained a note. The cause of a takeover can be invisible, so "no failure
  seen" alone is no disagreement. In the 7 round 1 V shards, 257 of 383 takeovers came after the
  carry start. Their triggers were mostly closed_no_pinch and half_budget_no_lift, which no frame
  shows. Takeovers stay negative.

### 4.4 The five rule changes

| change | time, code | rule | defect that it fixed |
| --- | --- | --- | --- |
| (1) | 17:06, 6f70f55 | The loss event needs a release that puts the part in no box. | In 7 V shards, 174 of 210 loss events sat in stowed cycles, 148 of them 0 to 4 frames before the release. The opening jaws let the part go while the command still read closed. |
| (2) | 17:06, 6f70f55 | A positive carry needs its first open at most RELEASE_LEAD = 10 steps before the census release. | The census release lags the first open by 1 to 4 steps. 15 % of the positive carries opened earlier, over the tray or in a tremble, and closed again. |
| (3) | 17:06, `feat/ms7i-prompt` 8dcc189, now 538ad33 | A teacher episode is positive only if it reaches a carry and then a release with no close after it. Otherwise it is untagged. | The corpus half of round 1 disagreed on 6 of 20. Teacher episodes with no carry, opens over the tray after a lift and an episode that starts held all read positive. |
| (4) | 17:43, 177d3e1 | A re-open needs a close of REOPEN_MIN_CLOSED = 3 steps or more that came before the first carry frame. | On 3 shards, 311 of 536 re-open events came after a 1-frame close and 99 after a 2-frame close. The command flickered around the width plus 4 mm. 221 came after the carry start. |
| (5) | 18:22, f20db17 | The grasp and carry segments are positive only if the census hold (`first_held_at`) began by the first carry frame. | In the 5 round 3 shards, 78 of 507 cycles with positives read the carry before the hold. Round 3's V00 is one: the cycle starts closed, reads carry at step 30 and holds at step 159. |

- Rule change (3) splits the corpus episodes. 69.0 % are clean, with 46.6 % of the frames. 16.7 %
  have no carry, 12.9 % regrasp after the release and 1.4 % have no release.
- Rule change (5) only removes positives, so the round 3 result of V stands.

### 4.5 The result

The final rules read V 2 of 40 (round 3), the corpus 0 of 20 (round 2) and B 1 of 40 (round 4).
That is 3 of 100, 0.03, against the bar of 0.10 (19:19). The corpus keeps its round 2 result,
because rule changes (4) and (5) do not change its codes. The label rule of record is
`feat/ms7i-own-frames` f20db17 for the converter and `feat/ms7i-prompt` 538ad33 for the teacher
codes.

The tallies are `i/audit/r2/tally.txt`, `r3/tally.txt` and `r4/tally.txt`. Round 1 has no tally
file. Its count is in the ledger at 16:58.

## 5. Track ACT

### 5.1 The recipe and the data

ACT-C is the F9n recipe (IR8, spec section 7.1): random init, the context fix, batch 8, a constant
learning rate of 1e-5 and 800k steps. It saves at 400k, 600k and 800k. ACT-C is filtered behavior
cloning: it learns only from frames that succeeded, with no new input. The spec dropped the
advantage column, because the review found it close to inert in lerobot 0.5.1 (spec section 7.1).

The data is the corpus of F9n, `g/w1234567/dataset`, plus the positive own frames and the
takeovers of B. `train_masked.py --extra_root` appends the extra roots to the corpus without a
merged copy (section 8). The features and normalizer statistics of the corpus serve every root.
Rule (IR8): the frames of B are at most 30 % of the total. A fixed random subset of the episodes
of B enforces the cap.

The mix of the run (`i/act/train_C.out`, the ACT_MIX line):

| source | frames | episodes |
| --- | --- | --- |
| the corpus | 9,676,460 | 23,957 |
| positive own frames of B, `conv/c/positive` | 651,374 | 3,264 |
| takeovers of B, `conv/c/takeovers` | 366,718 | 1,277 |
| B together | 1,018,092, share 0.0952 | 4,541 |
| the whole dataset | 10,694,552 | 28,498 |

- The frames of B are 9.5 % of the mix, under the 30 % cap. So the mix kept every episode of B.
- All frames of B come from aarch64 physics on Machine C. So 9.5 % of the frames of ACT-C come from
  aarch64 physics (spec section 5.2).
- Training ran on the 5090 at 400 W from 21:45 on 10-13 to 03:17 on 10-14 (TRAIN_OK).
- The spec expected 8.2 h at 27 steps a second. The run read 44 steps a second at 682k (02:32).

### 5.2 The screens

Rule (IR8 and IR2): the 400k, 600k and 800k saves read the screen with the switches and the
classifier of S. Each pairs with h_S_scr. HD32 (4) picks the best: the correct total first, then
the rate.

| read | correct of 3000 | rate | held-out row | cycles a stowed part | D against h_S_scr, paired SE | rate gap, SE |
| --- | --- | --- | --- | --- | --- | --- |
| i_C400_scr | 2512, 0.837 | 398.17 | 594 of 744, 0.7984 | 1.279 | -21, 34.8 | +28.70, 11.51 |
| i_C600_scr | 2502, 0.834 | 364.37 | 582 of 744, 0.7823 | 1.274 | -31, 44.0 | -5.09, 14.93 |
| i_C800_scr | 2498, 0.833 | 381.47 | 604 of 744, 0.8118 | 1.264 | -35, 33.7 | +12.01, 9.76 |
| h_S_scr | 2533, 0.8443 | 369.46 | 581 of 744, 0.7809 | | | |

A lifted_lost cycle held a part for at least 6 steps, lifted it at least 5 cm and lost it. The
census gave the cycle no stow credit. The jaw table of `s1_score.py` for the same reads:

| measure | ACT-C 400k | ACT-C 600k | ACT-C 800k | S |
| --- | --- | --- | --- | --- |
| re-open within 10 steps after a precise first close | 0.011 | 0.005 | 0.008 | 0.022 |
| first close off the target, stowed cycles | 0.041 | 0.040 | 0.044 | 0.068 |
| lifted_lost cycles | 202 | 204 | 182 | 183 |

- ACT-C closes more precisely than S at every save.
- 400k is faster than S by 28.70 correct stows an hour, 2.49 SE.
- No save stows more parts than S. D is -0.60, -0.70 and -1.04 paired SE.
- The best screen by HD32 (4) is 400k: 2512, then 2502 and 2498 (03:53).

### 5.3 The F9n and B screens on the same draws

F9n is the ACT-C recipe without the frames of B, and B is F9n 800k. Their screens on the same draws
show what the frames of B add at each save (04:34). The reads ran at different code revisions: F9n
400k and 600k at 273fabf, B at c6d4981 and ACT-C at e7dc7ae (`lifted_lost.md`).

| save | F9n: correct | F9n: rate | ACT-C: correct | ACT-C: rate | ACT-C minus F9n: correct | ACT-C minus F9n: rate |
| --- | --- | --- | --- | --- | --- | --- |
| 400k | 2474 | 296.9 | 2512 | 398.2 | +38 | +101 |
| 600k | 2513 | 325.8 | 2502 | 364.4 | -11 | +39 |
| 800k (B) | 2480 | 331.0 | 2498 | 381.5 | +18 | +50 |

The frames of B, 9.5 % of the mix, add correct stows at two of three saves and speed at all three.
S stays ahead of all three saves. These reads do not pair ACT-C with F9n, so the ledger gives no SE
for these differences.

### 5.4 The re-read of 400k

Rule (IR2 step 2): the candidate goes on only if its total on draws 20 to 39 is not under that of
i_S_d2039, 2526. `reread_best.sh` started i_C400_d2039 at 03:52, on 10 lanes of Machine B and 10 of
Machine A.

| read | correct of 3000 | rate | held-out row | cycles a stowed part | lifted_lost cycles | first close off the target |
| --- | --- | --- | --- | --- | --- | --- |
| i_C400_d2039 | 2464, 0.821 | 373.63 | 613 of 801, 0.7653 | 1.320 | 241 | 0.054 |
| i_S_d2039 | 2526, 0.842 | 360.92 | 642 of 801, 0.8015 | 1.286 | 194 | 0.057 |

- D = -62 at a paired SE of 33.7, -1.84 SE. The rate gap is +12.71 an hour, 1.15 SE.
- 2464 is under 2526, so ACT-C stops on the round 1 path. `chk_after.sh` started no checked read
  (04:31).
- Over draws 0 to 39, ACT-C 400k stows 4976 and S stows 5059, D = -83.
- On these draws the off-target shares of the first close are close: 0.054 against 0.057.

### 5.5 Where ACT-C 400k loses parts

A read-only subagent broke down the screen gap of 400k (`i/act/analysis/lifted_lost.md`, 02:53).
It reads six screens with the same switches, seed, draws and parts lists. Every count is per 3000
parts. Its definitions come from `core/loop.py`:

- The cycle end reason `placed` means that the jaws let go over a perceived box.
- The carry guard serves an open command as closed until the pick point is 10 mm inside the served
  box (`GUARD_BOX_MARGIN_M`). A guarded cycle has at least one guard step.
- The side of a part on the desk is its rest position minus the box center, along the carry
  direction. A positive value is the far side.

The findings:

1. The gap is not significant. D = -21 is -0.60 paired SE.
2. It is a desk gap. ACT-C 400k leaves 112 lost parts on the desk, against 88 for S. The other
   places sum to -3.
3. One heap holds most of it. Draw 16 holds 12 of the 21 parts and all of the lifted_lost excess.
   Without draw 16 the gap is -9 correct stows and -2 lifted_lost cycles.
4. The losses are not grip failures in the carry. The median `max_rise` of the lifted_lost cycles
   is 0.104 m for both. The loss rate of guarded cycles is 21.9 % for ACT-C and 22.8 % for S.
5. The losses come at the box, at the end of the carry. After a release over the right box, the
   loss comes a median 33 steps after the lift (S 37.5). A full carry is 41 steps (S 44).
6. The parts land past the far wall. ACT-C has 51 desk parts on the far side, against 30 for S.
   All of the excess comes from boxes 0.20 m or more from the tray.
7. Bricks carry the gap: 153 lost bricks against 120.

The core count is the number of desk parts lost in a no-guard, far-side, placed cycle. In each,
the policy itself opened the jaws over the right box, with the pick point 10 mm or more inside the
box walls. The part then fell past the far wall. The counts come from `lifted_lost.md` and, for
800k, from `ll800.out.txt` (03:53):

| read | no-guard, far-side, placed losses | top heap |
| --- | --- | --- |
| ACT-C 400k | 42 | 10 at draw 16 |
| ACT-C 600k | 45 | 16 at draw 19 |
| ACT-C 800k | 29 | 9 at draw 19 |
| S | 21 | 4 at draw 9 |
| F9n 400k | 41 | 6 at draw 8 |
| F9n 600k | 30 | 8 at draw 0 |
| B, F9n 800k | 25 | 7 at draw 1 |

The losses come in bursts at one box of one heap. 7 of the 10 in draw 16 are at one brick box,
0.33 m from the tray.

### 5.6 Two hypotheses

The analysis names two levers. Both are hypotheses.

H1, a serving rule. Serve ACT-C with `--guard_margin 0.020` (HD63, MS7h report section 7.15). A
20 mm margin keeps the jaws closed until the pick point is 20 mm inside every wall. After a stall
of 15 steps the loop hands the carry to the rig. On the 30k save of V over 7 heaps, 0.020 read +31
against its 0.010 twin, at a paired SE of 12.9. The core count is 42 for ACT-C 400k and 21 for S.
So the gain on this screen is at most about 21 parts, if ACT-C falls to the count of S. If the arm
still moves outward at the open, a wider margin only moves the release point. Then the probe will
show no gain. A margin change alters the serving of S, which decision I8 keeps fixed. So it needs
the owner.

The H1 probe (04:32) is evidence for round 2, not a candidate read. Rule, entered before its
number: S and ACT-C 400k each read the screen with `--guard_margin 0.020`. Each scores against its
own 0.010 screen, h_S_scr 2533 and i_C400_scr 2512. A gain of 2 SE_D or more for both makes the
margin a round 2 option. i_S_gm20_scr runs on Machine A and i_C400_gm20_scr on Machine B.

The H1 probe fails its rule. The margin hurts ACT-C, and S gains less than 2 SE_D:

| read | correct | rate | held-out | D against its 0.010 read | lifted_lost | budget |
| --- | --- | --- | --- | --- | --- | --- |
| i_C400_gm20_scr | 2468 | 346.9 | 0.811 | -44 (SE_D 37.7) | 250 vs 202 | 207 vs 147 |
| i_S_gm20_scr | 2570 | 359.7 | 0.809 | +37 (SE_D 25.4) | 188 vs 183 | 161 vs 143 |
| i_S_gm20_d2039 | 2531 | 371.6 | 0.779 | +5 (SE_D 24.5) | 192 vs 194 | 113 vs 152 |

S with the margin read 2570 on the screen, above the 2550 bar. The margin was one setting fixed
before the read, so the screen did not pick it among many. Fresh draws decided whether the gain
was real. Rule, entered before its number (05:42): S with the margin reads draws 20 to 39 against
i_S_d2039 2526, and D pooled over the 40 heaps must reach 2 paired SE. It read 2531, D +5. Pooled
over the 40 heaps, D is +42 at a paired SE of 35.2, which is 1.19 SE. So the margin is not a lever
on this evidence. On the screen, one heap (draw 12, +19) gave half of the gain.

With ACT-C, the margin holds more carries until the budget ends (207 cycles against 147), and
more parts end lost (250 against 202). So ACT-C does not carry the part deeper into the box when
the guard holds the jaws.

H2, the training length. The far-side trait belongs to the F9n recipe at 400k to 600k, not to the
frames of B. For F9n the core count falls with training: 41 at 400k, 30 at 600k and 25 at 800k.
The prediction, entered before the 800k screen (02:53): ACT-C 800k reads 25 to 30. A count of 40 or
more points at the own frames of B.

The check (03:53): ACT-C 800k reads 29, with 9 at draw 19. That is inside the predicted 25 to 30.
So the trait fades with training as for F9n, and the own frames of B do not add it. The correct
total does not follow the trait: 2512 at 400k, 2502 at 600k and 2498 at 800k.

The analysis also asks for an observation-only census field: the release xy in the box frame and
the tool speed at the open. It tells the two causes apart: a release near the far wall, or a
release while the arm moves.

## 6. Track V

IR7 sets the two runs. Both start from the fresh 30k weights on the second account, 8,000 steps
at batch 32. Their inputs reached the Shared drive at 00:03 on 10-14 (section 3.6). The owner
starts each runtime in the browser.

### 6.1 Run A

Pending: Run A, with the subtask and advantage prompt and all three labels, and its 4k and 8k
saves.

### 6.2 Run B

Pending: Run B, with the original prompt and the positive own frames only, and its 4k and 8k
saves.

### 6.3 The stage-1 reads

Pending: the stage-1 reads of each save on seed 20307000, draws 0 to 6, against the twin
h_Vf30k_a0j1slR25gm20_a (885). Then the 20-heap screen of the better save of each run.

### 6.4 Run A against Run B

Pending: D between the best saves of Run A and Run B on the 20-heap screen, and on draws 20 to 39
if both re-read.

## 7. Round 2 levers

The owner chooses the levers of round 2 after this report (spec section 4.1). This draft list gives
each lever with its evidence so far.

1. The subtask rule. Under the round 1 rule, 16.7 % of the corpus episodes (4,003) never read a
   carry. A study measured why (`i/tables/subtask_r2/subtask_r2.md`). Its truth proxy is the
   teacher's release over the box. In 2,411 of these episodes (10.1 % of the corpus) no carry is
   the right reading: 1,882 failed lifts, 526 with no lift and 3 with no close. Only 1,592
   (6.6 %) are real misses: a thin grip under the 3 mm gap floor (719), a close wider than the
   served row (405, of which 63 % on the first close), a cycle split after the lift (404) and a
   soft grip (64). So the ledger's first reading, "mostly regrasps on a wider axis", was wrong.
   Two variants stay on the perceived boundary. V1 lets a gap under 3 mm grip when it is at least
   1.0 mm over the command. It raises the corpus carry share from 83.3 % to 86.4 % and the
   positive share from 69.0 % to 71.8 %, and it changes 1.18 % of the own-frame labels. V2 adds a
   close threshold from the close's own command and a rise from the desk for a held start. It
   reads 89.6 % carry and 74.4 % positive, and it changes 3.00 % of the own-frame labels. Neither
   reads a carry in a corpus episode with no lift. The raw shards behind the final roots stay on
   disk (`STATE.md`), so a new rule can relabel round 1 without a new collection.
2. The carry-guard margin, H1. The core count is 42 for ACT-C 400k and 21 for S. The margin 0.020
   gained +31 on the 30k save of V, at a paired SE of 12.9. A change alters the serving of S, so it
   needs the owner. The census field of section 5.6 tells this cause apart from a release while
   the arm moves. The H1 probe of section 5.6 does not support it: ACT-C loses 44 parts with the
   margin, and S gains +42 over 40 heaps at 1.19 SE. It is not offered.

3. The ACT training length, H2. The core count falls with training: F9n 41, 30 and 25, and ACT-C
   42, 45 and 29, at 400k, 600k and 800k. The correct total of ACT-C does not rise with it: 2512,
   2502 and 2498. No ACT-C save beats S on the screen.
4. Pending: Track V levers, after Run A, Run B and their reads.

## 8. The code

The code of record is on four pushed branches. None of them is in main. `feat/ms7i-vla-dagger`
stacks on 4f0d095, which is in main ce97325. The first three are draft PRs #194, #195 and #196 in
stack #198. ACT-C trained from `feat/ms7i-act-mix` 87b320a. Its commit, cherry-picked onto main
with the same diff as b4e8ac5, is draft PR #197. The launchers, the conversion pipeline and the audit
tools are outside the repository, in `~/data/maker-arm-ms7/i/bin/`.

| branch | tip | base | what it adds |
| --- | --- | --- | --- |
| `feat/ms7i-vla-dagger` (#194) | 61763a8 | 4f0d095 | `dagger_vla.py` converts DAgger takeover shards into the π0.5 dataset format with the corpus statistics. `train_pi05.py --dagger_root` and `colab_dagger.sh` fine-tune π0.5 on a corpus and DAgger mix, and the converter gains `--drop_reopen_lead`. |
| `feat/ms7i-own-frames` (#195) | f20db17 | 61763a8 | `--record_all` records every frame with its cycle, its subtask and the census extras, and `core/hold.py` gains the subtask function. `convert_to_lerobot.py --own_frames` joins each cycle to its census row and writes the IR5 labels with rule changes (1), (2), (4) and (5). |
| `feat/ms7i-prompt` (#196) | 538ad33 | f20db17 | The optional prompt id on the wire, the prompt of each request in `pi05_server.py` with no forced replan, and `--pi05_prompt`. `train_pi05.py` trains on an N-source mix with a prompt code for each frame and the 20 % tag drop, and `prompt_codes.py` writes the teacher tags of rule change (3). |
| `feat/ms7i-act-mix` (#197 as b4e8ac5) | 87b320a | 61763a8 | `train_masked.py --extra_root` appends extra roots to the corpus, capped by `--extra_max_share` (0.3). Without the switch it reproduces the F9n recipe byte for byte. |
