# MS7g: close the MS7f gate

Date: 2026-10-05. Status: final. This report uses the ledger through its entry of 05:59 on
2026-10-05, deviation D22. MS7g closes as a near miss. The checked read of S misses the staged share
bar by 5 correct stows. It passes the held-out bar and the rate bar. The official read did not run.

The design of MS7g is the document that the owner reviewed and approved in chat on 2026-10-03. This
report cites it as "design section N" and restates every rule that it uses. The MS7f report,
`docs/ms7f-real-scale-sorting.md`, is cited as "MS7f report section N". The campaign ledger is
`~/data/maker-arm-ms7/g/README.md`, and `STATE.md` beside it is the resume pointer. Every rule went
into the ledger before its number. The one exception is deviation D8, which the ledger marks as
post-hoc. The mechanism tables are `~/data/maker-arm-ms7/g/tables/<TAG>.mech.txt`. The track tables
are under `~/data/maker-arm-ms7/g/tables/<track>/`. Each read has a TAG, its name. The payloads of a
read are under `~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7g TAG starts with `g_`.

The names. The design names its decisions G1 to G6. The ledger names its rules G1 to G12 and its
deviations D1 to D22. This report writes "decision G1" for the design and "rule G1" for the ledger.
A deviation of MS7f is written "MS7f D19". Candidate 6 is the final MS7f candidate: the checkpoint
F5 200k, the classifier v9e and the flags of MS7f report section 9.22. The warm MS7g retrains are
F7, F7n, F8 and F8n. The runs from random init are F9n and its control F9. The low-rate finishes
are A5 and A9. A checkpoint is written as "F7 100k". S0 and S are the two stacks of loop switches
that section 4 builds. Se is S with the policy ensemble of D20.

The times. All times are PDT. Each time is the stamp of the ledger entry that records the event. A
time without a date from 09:29 to 23:59 is on 2026-10-03. A time without a date from 00:44 to 06:40
is on 2026-10-04. Every later time carries its date, as "10-04 16:12" or "10-05 05:59". Three
ledger entries stand out of time order: 11:50, 14:18 and 04:21. This report keeps each stamp as the
ledger writes it.

The numbers. Every number comes from the ledger or from a table file that this report names. This
report computes no new number.

## 1. Summary

MS7f closed as failed (MS7f D25). Its final result is candidate 6's checked read f_F6R_chk. It stows
2522 of 3000 parts in a box of their category, 0.841, at 308.0 correct stows an hour. Its held-out
row reads 620 of 785, 0.790 (MS7f report section 0). It misses the staged share bar by 28 correct
stows and the held-out bar by 8 parts.

MS7g keeps the scene, the catalog, the part split and the staged bars of MS7f D12. Its goal is a
checked read that passes the staged bars. A pass of the one-shot official read on the unburned seed
20307200 must follow (decision G1). The owner started the run with /goal at 09:29 on 2026-10-03. The
time box ends at 23:59 on 2026-10-06, with no extension (rule G11).

The gate (rule G2). A correct stow, the rate and the held-out row keep their MS7f definitions (MS7f
report section 1.3). A correct stow is a part at rest in a box of its category. The rate is the
correct stows per simulated hour. The held-out row counts the parts of the kinds that neither the
policy nor the classifier saw in training.

| row | correct in a box of the category | correct stows an hour | held-out row |
| --- | --- | --- | --- |
| staged bars, the gate | at least 0.85 (2550 of 3000, 5100 of 6000) | at least 200 | at least 0.80 |
| stretch row, report-only | at least 0.92 | at least 300 | at least 0.85 |

The final candidate. The stack S is F5 200k with `--carry_guard --overhang_guard`, the classifier
v10e3 with `--cls_flip` and the final-only vote `g/tables/vote/vote_v10e3_flip_final.json`, served
at bfdffb2 (22:39). Its screen read g_Sf_scr stows 2548 of 3000, 0.8493 [0.836, 0.862], at 372.5
correct stows an hour. Its held-out row reads 594 of 744, 0.798. A report-only read of S on 20 fresh
heaps, g_Sf2_scr, stows 2507 of 3000. On the 40 screen heaps, S stows 5055 of 6000, 0.8425, with a
held-out row of 1215 of 1545, 0.786. Both rows are under the staged bars, and the rate is far over
its bar. The ledger reads the pass chance of an official read at this level as a few percent
(05:31).

The checked read (section 8.1). g_S_chk reads S on seed 20307100, draws 0 to 19, from 05:54 to
06:29. It stows 2545 of 3000, 0.8483 [0.835, 0.861], at 378.8 correct stows an hour. Its held-out
row reads 629 of 785, 0.801 [0.772, 0.828].

| staged bar | bar | g_S_chk | verdict |
| --- | --- | --- | --- |
| correct in a box of the category | at least 2550 of 3000 | 2545 | FAIL by 5 correct stows |
| correct stows an hour | at least 200 | 378.8 | PASS |
| held-out row | at least 0.80 (628 of 785) | 629 of 785, 0.801 | PASS |

Against candidate 6's checked read f_F6R_chk (2522, held-out 620 of 785, 308.0 an hour), S reads 23
more correct stows. It also reads 9 more held-out parts and 70.8 more correct stows an hour. Over 60
heaps (the 40 screen heaps and the checked read), S stows 7600 of 9000, 0.8444.

The official read (section 8.2). It did not run, because rule G9 starts it only on a checked pass.
Seed 20307200 stays unburned.

After the checked miss, the owner asked for a policy trained from random init with the context fix
(D19). F9n peaked at 2554 on the screen seed at 800k and fell after it. Its paired control F9
without the fix read 2430 at 800k, so the fix is worth 124 correct stows from scratch. A policy
ensemble (D20, 2545) and a finish at a low learning rate (D21, 2544 and 2519) followed. No policy
reached the adoption bar of 2578 (section 9).

The verdict of MS7g (D22, 10-05 05:59, an owner decision). MS7g closes as a near miss. The final
candidate is S, with the checked read g_S_chk. It is 5 correct stows short on the share bar, and it
passes the held-out bar and the rate bar. The emptying capacity of the boxes stays as it is.

What moved:

- The carry guard ended the hovers over the tray. The steps of air stalls with a held part fell from
  57,995 on R0 to 723 and 1109 on the two guard reads. The rate rose from 296.5 to 364.9 and 356.5
  correct stows an hour. The share did not move: +6 and -15 correct stows against R0. The owner
  adopted the guard for its visual effect (D5).
- The vote-window fix of the guard (7a2562a) took the late box switch, M_late, from 26 cycles to 10.
- The classifier lever moved the share a little. v10e3 with the mirrored frame and the final-only
  vote read 16 more correct stows than S0 on draws 0 to 19 and 37 more on draws 20 to 39, +53 of
  6000 pooled. In the loop, the taken class is right on 0.9432 of the lifts of g_Sf_scr, against
  0.9309 on g_S0_scr.

What did not move:

- No loop lever moved the share beyond the noise of a 20-heap pair, 30 correct stows (design section
  3.1). R0 reads 2501, and S0 reads 2532.
- `--clear_first` failed twice (g_cf_scr and g_Sfc_scr). `--action_warp 1.0` read 32 correct stows
  under the warp of 1.5.
- The nine-value fused vote raised the right class on the frame bank by 0.0133. In the loop it lost
  46 correct stows (g_Sv2_scr).
- Every warm retrain read under F5 200k: F7, with the REOPEN mask, by 103 and 116, F7n by 130 and
  95, F8 by 56 and F8n by 39. On these warm starts the context fix of D12 showed no effect beyond
  the noise.
- No policy from random init reached its bar of 2578. F9n read 2473 to 2554 from 200k to 1.2M steps.
  The ensemble of F5 200k and F9n 800k read 2545, and the low-rate finishes read 2544 and 2519.
- Classifier v12 and its ensembles read under v10e3 on the frame bank, and they stay out.
- The release track found no robot-side fix for the parts that bounce out of a full box (D16). The
  passenger track found no passenger detector (D17). The carry squeeze against grip slips failed its
  smoke gate (D18).

What the run found:

- From random init the context fix is worth 124 correct stows at 800k steps (F9n 2554 against F9
  2430, about 4 pair SEs), 97 more correct stows an hour and 0.039 more on the held-out row.
- Five reads of different policies and servings lie from 2544 to 2554 (section 9.5). S, Se, A9, A5
  and F9n 800k read 2519 to 2554 on the screen seed. The share has a plateau at this level.

Where the misses of S go (section 7). The two S reads put 298 and 316 parts in a wrong box. About
148 parts a read end outside every box. About 50 cycles a read release a part over the right box and
lose it outside every box (M_bounce 50 and 51). About 60 passengers a read ride into the box of the
held part.

## 2. The decisions and the deviations

### 2.1 The decisions of the design

The owner took these decisions on 2026-10-03, in chat (design section 2):

| id | question | decision |
| --- | --- | --- |
| G1 | the goal | The staged bars of MS7f D12 (0.85, 200 an hour, held-out 0.80) on a checked read, then the one-shot official read on seed 20307200. The stretch row stays report-only. |
| G2 | the scene | The MS7f D19 gate scene, unchanged. The simulation gets no harder and no easier. |
| G3 | the budget | Three days. The serving and perception levers come first, and a classifier retrain is allowed. If the stacked levers miss the checked read, one policy retrain runs as a pre-registered fallback. |
| G4 | a lifted part over the tray | A guard keeps the jaws closed on a lifted part unless the pick point is over a box of its current class. A still carry goes to the scripted rig, which reads only the perceived state. |
| G5 | the classifier | Both levers: a fused vote with fitted view weights, and a retrain with more classifier-only kinds by the rule of MS7f D4. An offline bank of in-loop frames judges both. |
| G6 | the method | Each lever is judged by its mechanism number and by a non-inferiority bound on the total. The adopted levers are read as one stack, then the checked read and the official read follow. |

### 2.2 The decisions of the owner during the run

| time | id | decision | reason |
| --- | --- | --- | --- |
| 12:40 | D5 | `--carry_guard` joins S without the mechanism bars of rule G6. The version is 7a2562a, with the vote-window fix. g_cg2_scr still reads as the report row. If it reads under 2471 (R0's correct stows minus 30), S takes b8c35d6, which passed the non-inferiority bound. | The owner wants the visual effect beyond the score: no hovers over the tray. |
| 20:36 | D12 | A small A/B test of the context findings of another session (section 6.3), at once. F7n is F7's recipe plus two input fixes. The owner accepts a restart of training from scratch. D12 replaces the single-retrain limit of D11. | The peer note of 20:33: the context of the policy enters unnormalized, and the camera roll flips its sign. |
| 06:35 | D19 | A policy retrain from random init with the context fix, F9n, after the checked miss (section 9). | The owner chose it after a detailed review of the levers in chat. |
| 10-05 05:59 | D22 | MS7g closes as a near miss. The final candidate is S with its checked read g_S_chk. The emptying capacity stays as it is (rule G2). | Every pre-registered lever was read, and none reached its bar (10-04 23:04). |

At 05:33 the controller asked the owner about the box emptying rule (section 7.6). The owner
answered with D22 (10-05 05:59): no change. The emptying capacity stays as it is, under rule G2.

### 2.3 The rules of the ledger

The ledger wrote rules G1 to G12 at 09:50, before any MS7g number:

| rule | subject | section |
| --- | --- | --- |
| G1 | the seeds | 3.2 |
| G2 | the gate | 1 |
| G3 | the read protocol | 3.1 |
| G4 | R0, candidate 6 with every new switch off | 4.1 |
| G5 | the mechanism numbers | 4.1 |
| G6 | the adoption rules of `--carry_guard`, `--clear_first` and `--overhang_guard` | 4.2 to 4.4 |
| G7 | the stack S, and the cut-off of 2026-10-05 14:00 for v12 | 4.6, 5.7 |
| G8 | the classifier levers, judged on the frame bank | 5 |
| G9 | the checked read and the official read | 8 |
| G10 | the fallback: wave 7, the REOPEN mask and retrain F7 | 6 |
| G11 | the time box, 2026-10-06 23:59 | 1 |
| G12 | the hosts and the code | 3.1, 12 |

The MS7f ledger and its rules R1 to R15 stay valid where the MS7g ledger does not replace them.

### 2.4 The deviations

| time | id | what changed | why |
| --- | --- | --- | --- |
| 09:50 | D1 | The run starts on 2026-10-03 09:29, one day before design day 1. Rule G11 stays the upper bound. The v12 cut-off of rule G7 stays at 2026-10-05 14:00. | A phase starts as soon as its inputs exist. |
| 09:50 | D2 | The bank recorder also captures the carry frame on cycles whose vote agreed. This is observation only, and the served run and its payload stay the same. | The bank needs the carry view on every lifted cycle, but the loop reads it only on a split vote. |
| 11:07 | D3 | The mechanism bar of `--clear_first` becomes "M_swept at most half of R0's M_swept", in place of 0.08. Written before any R0 or g_cf_scr number. | The design set 0.08 against about 0.19, a ratio of 0.42. The absolute bar fails on forced picks that no ranking can avoid. |
| 12:32 | D4 | The late-switch condition of the guard becomes M_late_wb at most a quarter of R0's, so at most 4. Written before any number of the fixed guard. | M_late counts cycles that no guard can reach: 9 of R0's 26 opened over a right-class box, the tray or the desk. |
| 12:40 | D5 | Owner decision: `--carry_guard` at 7a2562a joins S without the G6 mechanism bars (section 2.2). | The visual effect: no hovers over the tray. |
| 13:23 | D6 | Two pre-registered reads at fc2cccf: g_S0_scr (S0 = `--carry_guard --overhang_guard`) and g_S0w10_scr (the same at `--action_warp 1.0`). The warp of 1.0 joins S with at least 30 more correct stows than g_S0_scr and at least 200 an hour. | The x86 hosts idle until the classifier verdicts. The rate is far over its bar. MS7f D7 adopted the warp of 1.5 on 4-heap screens of candidate 1. |
| 13:50 | D7 | Two richer vote forms, V2 (per-class view weights) and V3 (a full stacker), with the bars of rule G8. v10e3 joins the classifier candidates. | The nine-value vote trades special for brick through one bias per class. |
| 14:18 | D8 | Post-hoc, written after the D7 numbers. The per-category bound of the vote is dropped for the nine-value vote. The loop read g_Sv_scr decides: the vote stays in S with at least g_S0_scr's correct stows and a held-out row of at least g_S0_scr's minus 0.01. The ledger flags this deviation for the report. | Every form, from 9 to 114 values, makes the same trade. Both share rows of the gate rise on the bank. The nine-value form is the one that the loop serves. |
| 14:36 | D9 | Wave 7 does not record the handoffs of `--carry_guard`. It records the trigger takeovers only, like waves 4 to 6. | A handoff resumes the rig without an actor ordinal, so `mj/env.py` skips those frames. A handoff frame needs a target row that the rig's carry entry does not have. |
| 20:23 | D10 | Pre-registered, replaces the v12 bars of rule G8 and keeps D8. The bank chooses the classifier among v9e, v10e3, v12, v12e, E4 and E5. Each reads with and without a mirrored frame, under the served rule and under its own nine-value vote. The winner joins S with at least 0.005 of the whole row over v9e with the nine-value vote (5327 of 5700). One loop read must then show at least 2532 correct stows. | No loop lever moved the share beyond the noise. The wrong class is the largest loss left. The bank predicts the in-loop class of the same cycles exactly. |
| 20:31 | D11 | Pre-registered: F7 trains now, and its two checkpoints read tonight. The better one replaces F5 200k in S with at least 30 more correct stows than F5 200k with the same switches. | The x86 hosts idle otherwise. A later miss of the checked read then has no fallback left. |
| 20:36 | D12 | Owner decision: the A/B of F7n (section 2.2). The policy with the most correct stows among F5 200k, F7 and F7n replaces F5 200k with at least 30 more. | The peer note of 20:33. |
| 21:13 | D13 | Pre-registered: the final-only fused vote. The box moves at selection and at the lift as under the served rule. At the carry view, the fused sum decides the final box once. g_Sf_scr is adopted with at least 2532 correct stows and a held-out row of at least 0.787. The queued g_Sc_scr is cancelled before its start. A second rule names g_Sc0_scr as the fallback. | The partial-sum vote of D8 lost in the loop because its bias moves the box at selection. |
| 23:14 | D14 | Pre-registered: two unmasked runs, F8 (no fix) and F8n (the context fix), on the same corpus with a paired batch order. D12's rule extends to them. | F7 100k's -103 points at the REOPEN mask, which confounds D12's context test. |
| 01:54 | D15 | Pre-registered: g_Sfc_scr = S + `--clear_first`. It joins S with at least 30 more correct stows than g_Sf_scr (2578) and a held-out row of at least 0.788. | The x86 slot after F8n is free while S waits for v12. Its own screen read +17 against R0, with 30 fewer wrong-box parts. |
| 04:26 | D16 | Pre-registered: a release fix for the bounce-outs behind its own switch, read as g_Sb_scr. Adopted with M_bounce at most 25 and at least 2518 correct stows. | M_bounce reads 50 and 51 under S against 31 under S0 and R0, the largest single loss after the wrong class. |
| 04:27 | D17 | Pre-registered: a passenger detector from the measured jaw gap at the lift. Only a detector with a precision of at least 0.9 and a recall of at least 0.5 gets a switch and a read g_Sp_scr. | About 60 passengers a read ride with the held part into its box under the guard. |
| 05:43 | D18 | Pre-registered: a carry squeeze against grip slips. Under `--carry_guard`, the loop tightens a closing gripper command to the measured gap minus `CARRY_SQUEEZE_M` (INITIAL). The squeeze lasts from the lift to the open over the served box. A smoke gate comes first. The read g_Sq_scr decides: the switch is adopted with at least a quarter fewer `lifted_lost` cycles than g_Sf_scr and at least 2518 correct stows. | About half of the bounce-outs are grip slips of flat and odd parts (D16 forensics). |
| 06:35 | D19 | Owner decision, pre-registered before any number: F9n is ACT trained from random init on waves 1 to 7 (no mask) with the context fix from step 0, about 400k steps. Its screens use S's serving and classifier. The best screen replaces F5 200k in S at 2578 or more (g_Sf_scr + 30). A second checked read follows, and the official read only on a pass. | The checked read missed by 5. The warm starts did not show the context fix. |
| 06:35 | D19a | Pre-registered: a paired control F9, F9n's recipe without the context fix, with the same random init and the same batch order. F9 400k gets one screen. D19's rule takes the best of F9n and F9. | The from-scratch run answers the owner's context question. |
| 06:38 | D19b | Amended before any F9 number: both runs continue to 800k at the same constant learning rate. The decision screens are F9n 400k, 600k, 800k and F9 800k. F9n 200k reads report-only. | The project's runs from scratch went to 600k to 800k, and the time box allows it. |
| 10-04 16:13 | D19c | Pre-registered before any number past 800k: both runs continue to 1.2M. The screens are F9n 1.0M and 1.2M. F9 1.2M also reads on one condition: F9 800k reads within 30 of F9n 800k. D19's rule stays. | The F9n curve still rose at 800k. |
| 10-04 19:49 | D20 | Pre-registered before the F9n 1.2M number: the policy server serves the mean of two policies' action chunks, F5 200k and the best F9n of 600k to 1.2M. When no F9 screen reaches 2578, g_Se_scr reads. It replaces S's policy at 2578 or more. | F5 200k and F9n read near 2550 each with S's serving (2548 and 2554 at 800k). |
| 10-04 20:49 | D21 | Pre-registered: two warm runs of 100k steps at a constant learning rate of 2e-6, A5 from F5 200k and A9 from F9n 800k. The better one replaces F5 200k in S at 2578 or more. | Every policy so far trained at a constant 1e-5, and F9n fell after 800k. |
| 10-05 05:59 | D22 | Owner decision: MS7g closes as a near miss (section 2.2). | Every pre-registered lever was read. |

### 2.5 The protocol changes without a D number

- 13:20: `feat/ms7g-selector` was force-pushed onto `feat/ms7g-carry-guard` 7a2562a. The new tip is
  fc2cccf, and 1643 tests pass.
- 20:21: g_Sv_scr ended READ_FAIL (section 5.4). From then on, every file that a switch names (a
  vote JSON or a classifier) is copied to Machine B before a read. Every waiter ends on `SYNC_READ_DONE`,
  which `sync_read.sh` prints on success and on failure.
- 03:06 and 03:41: two report-only reads on the screen seed, draws 20 to 39: g_Sf2_scr (S) and
  g_S02_scr (S0). No decision rests on them. They give a second sample of S and of the classifier
  lever.
- 05:31: the checked read waits. The levers of D16 and D17 read first. The checked read runs once S
  has them, or at 2026-10-05 18:00 at the latest. That leaves a day for the official read and the
  report inside rule G11. The reason: a pass of the checked read starts the official read at once
  (rule G9). The 40-heap estimate of S gives that read a pass chance of a few percent.
- 05:54: the lever list closed after D18, and the checked read of S started (section 8.1). The
  06:02 entry states that D16 to D18 close the lever list.
- 06:35: the MS7g branches became the GitHub stack #182 (section 12.1). Tag `ms7g-read-F9` = 273fabf
  holds S's code and the roll transform, and it serves the reads of D19.
- 10-04 17:07: F9 800k read more than 30 under F9n 800k, so F9 1.2M is not read (D19c). Its run
  stopped to give F9n the GPU.
- 10-04 20:14: tag `ms7g-read-Se` = 769c2eb serves the ensemble read of D20.

## 3. The protocol and the noise

### 3.1 The reads

Rule G3. Every read runs `g/bin/read_g.sh`. That script runs MS7f's `read_f2.sh` with candidate 6's
flags. The checkpoint is `f/train_F5/checkpoints/200000/pretrained_model`. The classifier is
`f/tables/cls/model_v9e/part_classifier.pt` unless a rule adopts another. EXTRA is `--action_warp
1.5 --missed_lift --cell_failure_cap 5 --tray_refill` plus the switches under test. MS7f report
section 1.4 lists the other flags.

Every read that decides runs on x86 only: 10 Machine A lanes (draws 0 to 9) and 10 Machine B lanes (draws
10 to 19), one env a lane. Both hosts run from a worktree at the SHA of the read. Machine C
records and renders only, because its physics differ (design section 4.3). A READ_FAIL reads once
more in a fresh TAG. A payload from the supervisor cache is not a read (MS7f R14).

| read | policy | seed and size | bars |
| --- | --- | --- | --- |
| screen | R0, each lever, S0, S and the retrains | 20307000, draws 0 to 19, 20 heaps of 150, 3000 parts | the adoption rule of each lever |
| report-only screen | S and S0 on fresh heaps | 20307000, draws 20 to 39, 3000 parts | none |
| checked | S, or a fallback candidate | 20307100, draws 0 to 19, 3000 parts | the staged bars |
| official | the first candidate whose checked read passes | 20307200, draws 0 to 39, 6000 parts, one shot, 20 Machine A and 15 Machine B lanes | the staged bars |

`g/bin/sync_read.sh` wraps one read. It puts the read worktree `~/src/maker-arm-lab-7g-r<N>` at the
SHA on Machine A and Machine B with `scripts/fleet.py sync`. Then it runs `read_g.sh`, writes the
mechanism table with `g/bin/mech.py` and prints `SYNC_READ_DONE`. Section 12 gives the commands.

### 3.2 The seeds

Rule G1 fixes every seed before its use. No MS7g seed lies in an older range. The part split keeps
`parts_seed` 20260915.

| use | seed |
| --- | --- |
| the frame bank | run 20320000 + p, layout 20320100 + p, p = 0 to 89, `--parts_split all`. Even p is the fit part, odd p the evaluation part. |
| the classifier-only kind renders (v12) | run 20321000 + i, i = 0 to 999 |
| wave 7 | run 20322000 + p, layout 20322100 + p, p = 0 to 89, train split |
| screens | 20307000, draws 0 to 19, and draws 20 to 39 for the report-only reads |
| checked reads | 20307100, draws 0 to 19 |
| the official read | 20307200, draws 0 to 39, one shot |
| smokes and development runs, never a verdict row | run 20323000 + i, layout 20323100 + i |

### 3.3 The noise of a read

Design section 3.1 measured the noise on the 20 heaps of f_F6R_chk: a mean of 126.10 correct stows
of 150 and a standard deviation of 4.81. The standard error (SE, the spread of a measured value) of
the difference between two independent reads of N heaps:

| N heaps | per 600 parts | per 3000 parts |
| --- | --- | --- |
| 4 | 13.6 | 68 |
| 8 | 9.6 | 48 |
| 12 | 7.9 | 39 |
| 20 | 6.1 | 30 |

Pairs on the same seeds do not help. Two reads on the same seeds have a per-heap correlation of
0.09, and a rerun diverges from the read after 2 to 6 cycles. So a 20-heap pair resolves about 60
correct stows at two standard errors. The 40-heap official read has an SE of about 30 on 6000 parts.

The MS7g reads show the same spread:

- One configuration on two sets of heaps: S reads 2548 on draws 0 to 19 and 2507 on draws 20 to 39.
  S0 reads 2532 and 2470.
- The three reads with the guard on read 2507, 2486 and 2532 (14:06).
- The in-loop taken class under the served rule is right on 0.9157 of the lifts of R0, 0.9309 of S0
  and 0.9237 of g_cg2_scr, a read-to-read spread of about 0.015 (21:12).
- M_bounce spans 31 to 59 over 13 reads on the same draws (`rel/forensics.txt`).
- The F9n curve reads 2473 to 2554 from 200k to 1.0M steps. Its spread lies within about 2.5 read
  SEs (10-04 18:47).
- Five reads of different policies and servings lie from 2544 to 2554. On the screen seed, S reads
  2548, Se 2545, A9 2544 and F9n 800k 2554. S reads 2545 on the checked seed (10-04 22:29).

So rules G6, D6, D11, D12 and D15 to D21 use a step of 30 correct stows, one SE of a 20-heap
pair. The step is an adoption step or a non-inferiority bound.

## 4. The loop levers

### 4.1 The census, L0, and R0

The census lands first (design section 5.1). `feat/ms7g-census` ab1deeb (10:34) adds these fields to
every cycle row of the payload's census:

- `release_place`: where the jaws of a lifted cycle first opened after the lift. The values are
  "tray", "right_box" (a box of the held part's true category), "wrong_box", "desk" and "rim". A
  lifted cycle is a cycle in which the detector read a lift or `--missed_lift` read a carry. The
  field is a truth read, for the census only.
- `held_class`: the true category of the part between the jaws at the lift.
- `swept`: the swept count of the grasp that the selector chose for the target.
- `air_stall_steps`, `air_stall_empty_steps` and `air_stalls`: the air stalls of the cycle. An air
  stall is 60 steps in a row with the FK pick point over the tray and at least 0.06 m above the
  desk. The FK pick point is the pick point from forward kinematics. Its 30-step mean speed is under
  0.3 mm a step (design section 5.2).
- `read_logp`, `lift_logp` and `carry_logp`: the six-class log-probabilities of each view.

The carry guard adds `guard_steps`, `guard_rim_steps`, `guard_vote_steps`, `handoff` and
`stall_open`. 1608 tests pass. The pin of rule G4 passes on two CPU scenes: with every new switch
off, a pinned draw gives the same payload as main 56c42b6, except for the new census fields.

Rule G5 defines the mechanism numbers. `g/bin/mech.py` computes them from the payloads of a read:

- M_tray_open: the lifted cycles whose jaws first opened over the tray.
- M_late: the `wrong_box` cycles whose taken class equals the true category of the part that entered
  the wrong box. The class was right, and the box was wrong.
- M_late_wb (D4): the M_late cycles whose jaws first opened over a wrong box.
- M_stall: the summed steps of air stalls while the hold detector holds a part.
- M_swept: the share of selections whose chosen grasp sweeps at least one neighbour.
- Reported with no bar: the handoffs and their stowed share, the riders, the passengers,
  `close_miss`, the guard holds over a box rim and the `wrong_box` cycles by vote case.

A rider is a part in a wrong box without a `wrong_box` cycle of its own. A passenger is a part in a
wrong box that was never a target and never held.

R0 (rule G4) is candidate 6 at ab1deeb with every new switch off, on the screen seed (10:34 to
11:19). It stows 2501 of 3000, 0.8337 [0.820, 0.847], in 3507 cycles and 911,040 steps, at 296.5
correct stows an hour. Its held-out row reads 586 of 744, 0.788. Its mechanism numbers
(`g_R0_scr.mech.txt`):

- M_tray_open 266, M_late 26 and M_swept 0.1882.
- M_stall 57,995 steps, 6.4 % of all steps, in 113 air stalls with a part.
- 124 riders, 55 passengers and 510 `close_miss` cycles.
- 234 `wrong_box` cycles. The cycles with a wrong class split by vote case: 67 agreed, 134 split and
  voted, 4 split with the carry view lost and 2 with no lift read. 358 parts end in a wrong box.
- The release places: the tray 266, the right box 2571, a wrong box 293, the desk 59 and a rim 32.
  68 lifted cycles never opened.

### 4.2 The carry guard

The switch `--carry_guard` joins three fixes: the late box switch, the hold to the box (M1 of the
video diagnosis) and the in-air stall watchdog (F1). Design section 5.2 describes it:

- The guard. While the hold detector holds a part and the cycle reads a lift, the served gripper
  command stays closed. The jaws open only with the FK pick point over the interior of a box of the
  current class, at least `GUARD_BOX_MARGIN_M` inside its walls. The margin is 0.010 m, marked
  INITIAL.
- The stall. An air stall, or 15 steps in a row in which the guard holds an open command closed,
  starts a handoff. If the detector holds a part, the loop hands the env to the scripted rig. The
  rig carries the part to the served box, releases it there and hands the env back. The rig reads
  only the hold detector, the FK pick point and the perceived box poses. If the detector holds
  nothing, the loop opens the jaws and fails the cycle (`stall_open`).
- The switch refuses `--passenger_drop`, which opens over the tray on purpose.

The build (10:58). `feat/ms7g-carry-guard` b8c35d6 holds 3064851, a perceived carry entry for the
rig, and b8c35d6, the guard with the stalls and the handoff. 1623 tests pass. The ledger accepts
four design deviations as built. The guard reads the lift of the current hold. An open is a gripper
command that ends the hold. The rig releases only over the served box. A cycle gets at most one
handoff.

A K = 50 smoke is a run on heaps of 50 parts, never a verdict row. This one ran on Machine A with
run 20323000 + i and layout 20323100 + i, i = 0 to 3. It read 4 heaps with the guard and 4 without
it. A GPU run is not a replay.

| row | guard | plain |
| --- | --- | --- |
| correct | 163 of 200 | 163 of 200 |
| M_tray_open | 5 | 19 |
| M_late | 0 | 1 |
| M_stall | 8 | 763 |
| riders | 9 | 3 |
| passengers | 7 | 1 |

The guard smoke had 6 handoffs, all from guard stalls and all placed, 5 of them stowed. It also had
5 `stall_open` cycles and 9 rim steps of 223 guard steps.

The read g_cg_scr (11:19 to 11:54) at b8c35d6 stows 2507 of 3000, 0.8357 [0.822, 0.848], in 3255
cycles and 742,084 steps, at 364.9 an hour. Its held-out row reads 580 of 744, 0.780. Rule G6
against R0:

| condition | bar | g_cg_scr | verdict |
| --- | --- | --- | --- |
| correct stows | at least 2471 | 2507, +6 | PASS |
| M_tray_open | at most 133 | 111 against 266 | PASS |
| M_late | at most 6.5 | 26 against 26 | MISS |
| M_stall | at most 28,998 | 723 against 57,995 | PASS |

The verdict: not adopted as built, on M_late. The guard removes time, not share, as design section
3.5 predicts for a fix that only removes time. The rows with no bar:

- 200 handoffs: 186 placed, 13 `lifted_lost`, 1 declined and 137 stowed.
- 4127 guard steps, 106 of them over a rim, and 47 `stall_open` cycles.
- 139 riders against 124, 83 passengers against 55 and 470 `close_miss` cycles against 510.

The forensics (12:30, `cg/late_forensics.txt`). Of the 26 M_late cycles of g_cg_scr, 16 opened over
a wrong box inside the vote window (R0 17). The box that the guard trusted was still the box of the
overhead read. The vote moves the box `CARRY_VIEW_STEPS` (15) steps after the lift. 15 of the 16
were `--missed_lift` carries, read late at the tray edge. 9 opened over a right-class box and ended
in a wrong box later (R0 4). 1 opened over the tray (R0 4) and 0 over the desk (R0 1). No guard
reaches these cycles. The 28 extra passengers are carries that the policy used to drop over the
tray: 109 lifts with the passenger head firing opened over a box, against 66 on R0. That is not a
gap of the guard.

The fix, 7a2562a. While a vote or the lift view is pending, no box counts as served. So every open
of a held, lifted part waits for the vote. The census adds `guard_vote_steps`. A K = 50 smoke on 4
heaps (seeds 20323000 + i) reads 170 correct stows, against 163 for b8c35d6 and 163 for plain. Its
M_late reads 0.

D4 (12:32) moves the late-switch condition to M_late_wb, at most a quarter of R0's 17, so at most 4.
D5 (12:40) then adopts the guard at 7a2562a by the owner's decision, without the mechanism bars.

The read g_cg2_scr (12:36 to 13:21) at 7a2562a stows 2486 of 3000, 0.8287 [0.815, 0.842], in 3270
cycles and 753,074 steps, at 356.5 an hour. Its held-out row reads 584 of 744, 0.785. Against R0:

- Correct stows -15. That passes the non-inferiority bar of 2471, so S keeps 7a2562a under D5.
- M_late 10 against 26, and M_late_wb 5 against 17. The D4 bar of 4 misses by one. The miss has no
  effect under D5.
- M_tray_open 108 and M_stall 1109.
- 247 handoffs: 234 placed, 13 `lifted_lost`, 176 stowed. 200 guard steps waited for a vote. 70
  `stall_open` cycles.
- 143 riders against 124, 74 passengers against 55, 539 `close_miss` cycles against 510 and 211
  `wrong_box` cycles against 234.

The reading of the ledger. The vote-window fix took the late switch from 26 to 10. The guard is
share-neutral, +6 and -15 on two reads. It buys 20 to 23 % of rate and ends the hovers. The share
has to come from the classifier: about 200 wrong-class cycles a read, about 130 from split votes and
about 67 agreed.

The guard margin. Design section 5.2 asks this report to state whether the margin of 0.010 m holds.
The census counts the guard steps over a box rim: 106 of 4127 on g_cg_scr, 75 of 4217 on g_cg2_scr,
94 of 4744 on g_S0_scr and 54 of 6186 on g_Sf_scr. The release forensics found no release within 2
cm of a wall and none near the margin (`rel/forensics.txt`). The verdict (06:02): the margin of
0.010 m (INITIAL) holds. On the guard reads, 75 to 106 of 4127 to 4217 guard steps lie over a rim.
The release geometry does not separate the bounce-outs from the stows. The checked read g_S_chk puts
95 of 5794 guard steps over a rim (`g_S_chk.mech.txt`).

### 4.3 `--clear_first`

The switch changes the ranking key of the grasps (design section 5.3):

```python
(g.swept > 0, -math.floor(g.height / HEIGHT_BAND), g.swept, -g.clearance, g.width)
```

A grasp that sweeps no standing neighbour comes first, and the height band second. The ranking also
picks the target part, so the switch changes the part that the policy is sent to.

The build (11:05). `feat/ms7g-selector` 81c1341 holds 0f57898 `--clear_first`, fe9722e
`--overhang_guard` and 81c1341, which keeps a part on the leaned wall's inner face in the tray. The
K = 50 single-heap smokes ran on Machine A with run 20323000 and layout 20323100. With the switch
off they read M_swept 0.215, 0.148 and 0.259. With it on they read 0.079, 0.161 and 0.094. A
diagnostic client logged the ranked grasps at each choice. In the heap with the switch on, 0 of 5
swept choices had a grasp that sweeps nothing and clears its landing floor. 3 were forced picks with
nothing clear, and 2 had only unswept grasps that land on a neighbour. So the residual sweeps are a
floor that the key cannot avoid, and D3 changes the bar to half of R0's M_swept.

The read g_cf_scr (11:55 to 12:36) at 81c1341 stows 2518 of 3000, 0.8393 [0.826, 0.852], in 3461
cycles and 854,589 steps, at 318.2 an hour. Its held-out row reads 584 of 744, 0.785. Rule G6 with
D3 against R0: correct stows +17 against a bar of 2471, PASS. M_swept 0.1393 against 0.1882, with a
bar of at most 0.0941, MISS. The verdict: not adopted. The rows with no bar, R0 in brackets: 207
`wrong_box` cycles (234), 328 wrong-box parts (358), 121 riders (124), 55 passengers (55), 505
`close_miss` cycles (510), M_tray_open 300 (266) and M_late_wb 12 (17). The reading of the ledger:
on full 150-part heaps the key moved only a quarter of the swept choices, because a dense tray often
offers no clear grasp. The gain of 17 lies inside the SE of a pair, 30.

D15 tried the switch once more, on top of S. g_Sfc_scr (02:28 to 03:06) at bfdffb2 stows 2517 of
3000, 0.8390 [0.825, 0.852], at 364.9 an hour. Its held-out row reads 573 of 744, 0.770. M_swept
falls to 0.1181 against 0.1862 on g_Sf_scr. The correct stows read 31 under g_Sf_scr, against a bar
of +30. The verdict: not adopted.

### 4.4 `--overhang_guard`

The switch has two parts (design section 5.4):

- The selector refuses a part outside the tray interior that lies within the leaned wall's overhang.
  The band is the wall's footprint at its top plus the landing half gap and the tip thickness.
- `CARRY_GRACE_STEPS` applies only to a cycle that reads a lift. A clamp on a wall cannot extend a
  cycle any more.

Rule G6 gives the switch no read of its own. Its unit tests pin it, and it joins the stack. After
the 13:20 restack, the selector commits are 5cbe184 (`--clear_first`), 1adfe2c (`--overhang_guard`)
and f654d4f (the wall's inner face). The tip fc2cccf adds a test that the three switches combine. A
K = 50 smoke of `--carry_guard --overhang_guard` at fc2cccf (Machine A, run 20323000, layout 20323100)
ends EVAL_OK with 43 of 50. It has 3 handoffs, all placed and stowed, and 0 overhang refusals.

S0 is candidate 6 with `--carry_guard --overhang_guard`. Its read g_S0_scr ran from 13:23 to 14:10,
with the ledger stamp 14:06. At fc2cccf it stows 2532 of 3000, 0.8440 [0.831, 0.857], in 3271 cycles
and 750,340 steps, at 364.4 an hour. Its held-out row reads 593 of 744, 0.797. Against R0:

- Correct stows +31.
- M_tray_open 117, M_late 20 (M_late_wb 7) and M_stall 903.
- 159 riders against 124, and 91 passengers against 55. The guard carries the passengers that the
  policy used to drop over the tray into the box of the held part.

MS7f's passenger head found 0.16 to 0.44 of the passengers, and `--passenger_drop` cost 24 correct
stows (MS7f D9 and D20). So the ledger tries no rider exception. S0 is the base of every later
screen.

### 4.5 The warp: D6

MS7f D7 adopted `--action_warp 1.5` for speed on 4-heap screens of candidate 1 (MS7f report section
9.3). D6 reads the warp of 1.0 on S0. g_S0w10_scr (14:06 to 14:51) at fc2cccf stows 2500 of 3000,
0.8333 [0.820, 0.846], in 3142 cycles and 920,721 steps, at 293.2 an hour. Its held-out row reads
579 of 744, 0.778. Against g_S0_scr it reads 32 fewer correct stows, against a bar of +30. The
verdict: the warp stays at 1.5.

### 4.6 The mechanism numbers of the loop reads

From the mechanism tables (`<TAG>.mech.txt`). The M_late_wb of g_cg_scr comes from
`cg/late_forensics.txt`, and the M_late_wb of R0 from the ledger entry of D4.

| row | g_R0_scr | g_cg_scr | g_cf_scr | g_cg2_scr | g_S0_scr | g_S0w10_scr | g_Sf_scr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 2501 | 2507 | 2518 | 2486 | 2532 | 2500 | 2548 |
| correct stows an hour | 296.5 | 364.9 | 318.2 | 356.5 | 364.4 | 293.2 | 372.5 |
| M_tray_open | 266 | 111 | 300 | 108 | 117 | 76 | 104 |
| M_late | 26 | 26 | 24 | 10 | 20 | 9 | 26 |
| M_late_wb | 17 | 16 | 12 | 5 | 7 | 4 | 9 |
| M_stall | 57,995 | 723 | 47,719 | 1109 | 903 | 404 | 857 |
| M_swept | 0.1882 | 0.1883 | 0.1393 | 0.1838 | 0.1843 | 0.1910 | 0.1862 |
| riders | 124 | 139 | 121 | 143 | 159 | 141 | 124 |
| passengers | 55 | 83 | 55 | 74 | 91 | 79 | 61 |
| `close_miss` cycles | 510 | 470 | 505 | 539 | 498 | 351 | 456 |
| `wrong_box` cycles | 234 | 212 | 207 | 211 | 196 | 217 | 174 |
| wrong-box parts | 358 | 351 | 328 | 354 | 355 | 358 | 298 |
| handoffs, placed, stowed | | 200, 186, 137 | | 247, 234, 176 | 242, 217, 172 | 169, 151, 115 | 243, 223, 175 |
| guard steps, over a rim | | 4127, 106 | | 4217, 75 | 4744, 94 | 3459, 51 | 6186, 54 |
| guard steps waiting for a vote | | | | 200 | 234 | 81 | 901 |
| `stall_open` cycles | | 47 | | 70 | 82 | 29 | 74 |
| release over the tray | 266 | 111 | 300 | 108 | 117 | 76 | 104 |
| release over the right box | 2571 | 2631 | 2605 | 2587 | 2605 | 2570 | 2641 |
| release over a wrong box | 293 | 247 | 266 | 266 | 251 | 265 | 220 |
| release over the desk | 59 | 45 | 47 | 46 | 39 | 42 | 51 |
| release over a rim | 32 | 24 | 19 | 18 | 13 | 6 | 16 |
| lifted, never opened | 68 | 22 | 44 | 19 | 32 | 21 | 25 |

## 5. The classifier levers

The classifier sets the box from three views (MS7f report section 5.1): the overhead view at
selection, the wrist view at the lift and the wrist view of the carry. The served rule is the rule
of candidate 6. On an agreement with the overhead read, the lift read sets the class. A split goes
to the summed log-probabilities of the views (design section 6.3).

### 5.1 The frame bank

The bank holds the frames that the served loop gives the classifier, with the labels of the
simulator (design section 6.1). The D24 recorder of MS7f records them, driven by candidate 6 on the
bank seeds. D2 adds the carry frame on cycles whose vote agreed. Even p is the fit part, and odd p
is the evaluation part. The fit uses only the frames of train kinds. The bank is never training data
for a classifier.

The bank (13:46). 90 of 90 heaps recorded with 0 EVAL_FAIL, at 21c77d6 (5d18874 before its rebase,
with `--bank_views`). Machine B recorded p = 60 to 89 on 5 GPU lanes. Machine C recorded p = 0
to 59 on 30 lanes. It used CPU policy servers after the A2000 ran out of memory with GPU servers.
The bank holds 11,401 lift events with all three views and a truth label, against a bar of 3000
(rule G8). The served rule replayed on the bank matches the census on every cycle of all 141 files
(90 bank files and 51 D24 files): 22,012 of 22,012 taken classes. So a rule read on the bank
predicts the in-loop class of the same cycle.

### 5.2 The nine-value vote: rule G8

The fused vote always sums `w_o * read_logp + w_l * lift_logp + w_c * carry_logp + b`: three view
weights and six category biases, nine values. A missing view drops out of the sum. The values come
from a maximum-likelihood fit of a multinomial logit on the fit part (design section 6.3).

The verdict of 13:46 (`vote/vote_v9e.json`). The fit uses even p of the train kinds plus the D24
files, 10,881 events. The evaluation uses odd p, 5700 events with three views.

| item | value |
| --- | --- |
| view weights | overhead 0.833, lift 1.039, carry 0.299 |
| category biases | brick +1.760, special +0.222, flat -0.157, round -0.827, joint -0.949, minifig -0.048 |
| whole row, bar +0.01 | 0.9212 to 0.9346, +0.0133, PASS |
| held-out row, bar a fall of at most 0.005 | 0.8542 to 0.8681, PASS |
| brick | 0.8869 to 0.9505 |
| special, bar a fall of at most 0.02 | 0.8662 to 0.8246, -0.0416, FAIL |
| joint | 0.9807 to 0.9631 |
| flat, round, minifig | within 0.01 |

The verdict: not adopted. A diagnostic with no bar fixed the biases at 0 and refit the weights: the
whole row gains 0.0018.

### 5.3 Two richer vote forms: D7

D7 fits two richer forms on the same fit part, with the same bars (14:15, `vote/v9e_d7/`):

- V2, per-class view weights: z_k = sum_v w_{v,k} logp_v[k] + b_k. On v9e the whole row reads 0.9344
  (+0.0132), the held-out row 1249 of 1440, and special 0.8221 (-0.0441).
- V3, a full stacker: z_k = sum_v sum_j W_{v,k,j} logp_v[j] + b_k, with L2 10. The whole row reads
  0.9353 (+0.0140), the held-out row 1248, and special 0.8155 (-0.0507).

Both are not adopted, on special. Every form buys brick with special: brick rises from 0.887 to
between 0.949 and 0.955. v10e3 (MS7f D22, the ensemble of v6a, v8 and v10) gives a row with no bar.
Under the served rule its whole row reads 0.9275 and its held-out row 1255, against v9e's 0.9212
and 1230. With V3 against v9e with V3, it gains 0.0040 of the whole row and 0.0056 of the held-out
row, against a bar of +0.03. Not adopted.

### 5.4 The nine-value vote in the loop: D8

D8 is post-hoc. The controller wrote it at 14:18, after the D7 numbers, and the ledger flags it for
this report. It drops the per-category bound for the nine-value vote, on three reasons. Every form,
from 9 to 114 values, makes the same trade. So the trade is the class prior of the gate's fixed
catalog mix, not a fit artifact. Both share rows of the gate rise on the evaluation part: the whole
row by 0.0133 and the held-out row by 0.0139. The nine-value form is the one that the loop already
serves. As a guard against a bank-only effect, the vote must also hold in the loop.

The vote branch was restacked onto the selector tip (14:40). `feat/ms7g-vote-weights` 69e7126 holds
the five vote commits, 2d8bf2a and 69e7126. Under `--vote_weights`, every lift keeps the vote
pending until the carry view, so the guard's pending test covers it. 1659 tests pass. A K = 50 smoke
(run 20323000) ends EVAL_OK with 41 of 50 and 6 handoffs, all placed and stowed.

The first read, g_Sv_scr, ended READ_FAIL. The 10 Machine B lanes died at argument parsing, because
`fleet.py read` ships only the checkpoint and the ledger, and `vote_v9e.json` existed only on
Machine A. The waiter watched only for `gate_results.json`, so about 5 hours of x86 time were lost
(found at 20:21). The 10 Machine A payloads are not used (rule G3).

The read g_Sv2_scr (20:21 to 21:11) at 69e7126 stows 2486 of 3000, 0.8287 [0.815, 0.842], in 3298
cycles and 763,222 steps, at 351.8 an hour. Its held-out row reads 563 of 744, 0.757. Against
g_S0_scr it reads 46 fewer correct stows and a held-out row 0.040 lower. The D8 test fails, and the
vote leaves S.

The vote forensics (21:12, from the census log-probabilities of the two reads):

- The loop's taken class equals the offline fused replay on 2520 of 2520 lifts. The code serves the
  fitted rule.
- On the lifts of g_Sv2_scr, the fused rule is right on 0.9274, and the served rule replayed on the
  same frames is right on 0.9119. The bank's +0.013 holds on the same frames.
- The loss is behavior. The bias moves the box at selection: the overhead box is wrong on 685 of
  3298 selections against 499 of 3271 on g_S0_scr. `lifted_lost` rose from 161 to 200, and the opens
  over the tray from 117 to 176.

### 5.5 The bank chooses the classifier: D10

D10 (20:23) replaces the v12 bars of rule G8. Each candidate reads the evaluation part (odd p, 5700
three-view events) under the served rule and under its own refit nine-value vote. Each also reads
with a horizontal mirror of each view, which averages the log-probabilities of the frame and of its
mirror image (`--cls_flip`). The winner has the highest whole row among the candidates whose
held-out row is at least v9e's under the served rule, 1230 of 1440. To join S, it must beat v9e with
the nine-value vote (5327) by at least 0.005 of the whole row. So the join bar is 5355.5.

The table of 04:24 (`vote/d10/table_v12.txt`, correct finals of the evaluation part):

| candidate | rule | whole | held-out | brick | special | flat | round | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (of) | | 5700 | 1440 | 2281 | 1203 | 716 | 369 | 569 | 562 |
| v9e | served | 5251 | 1230 | 2023 | 1042 | 709 | 367 | 558 | 552 |
| v9e | nine | 5327 | 1250 | 2168 | 992 | 702 | 365 | 548 | 552 |
| v9e, mirrored | served | 5257 | 1231 | 2026 | 1042 | 709 | 367 | 560 | 553 |
| v9e, mirrored | nine | 5330 | 1251 | 2169 | 992 | 701 | 366 | 550 | 552 |
| v10e3 | served | 5287 | 1255 | 2039 | 1055 | 710 | 367 | 560 | 556 |
| v10e3 | nine | 5356 | 1263 | 2176 | 1004 | 704 | 366 | 553 | 553 |
| v10e3, mirrored | served | 5288 | 1253 | 2039 | 1055 | 710 | 367 | 562 | 555 |
| v10e3, mirrored | nine | 5367 | 1266 | 2181 | 1004 | 704 | 367 | 557 | 554 |
| E4 (v6a + v8 + v10 + v11) | served | 5278 | 1250 | 2029 | 1051 | 712 | 367 | 563 | 556 |
| E4 | nine | 5359 | 1267 | 2178 | 1005 | 706 | 366 | 550 | 554 |
| E4, mirrored | served | 5277 | 1247 | 2031 | 1052 | 711 | 367 | 562 | 554 |
| E4, mirrored | nine | 5364 | 1269 | 2180 | 1003 | 707 | 366 | 553 | 555 |
| E5 (E4 + v12) | served | 5260 | 1250 | 2007 | 1060 | 711 | 367 | 562 | 553 |
| E5 | nine | 5344 | 1258 | 2171 | 1000 | 706 | 366 | 547 | 554 |
| E5, mirrored | served | 5258 | 1245 | 2010 | 1054 | 712 | 367 | 562 | 553 |
| E5, mirrored | nine | 5352 | 1260 | 2174 | 998 | 707 | 366 | 552 | 555 |
| v12 | served | 5025 | 1193 | 1822 | 1039 | 699 | 365 | 556 | 544 |
| v12 | nine | 5183 | 1204 | 2110 | 925 | 702 | 363 | 533 | 550 |
| v12, mirrored | served | 5044 | 1194 | 1828 | 1046 | 701 | 365 | 558 | 546 |
| v12, mirrored | nine | 5198 | 1202 | 2109 | 934 | 701 | 364 | 539 | 551 |
| v12e (v6a + v8 + v12) | served | 5223 | 1232 | 1989 | 1048 | 710 | 367 | 558 | 551 |
| v12e | nine | 5322 | 1244 | 2164 | 989 | 702 | 366 | 549 | 552 |
| v12e, mirrored | served | 5232 | 1231 | 1994 | 1051 | 710 | 367 | 559 | 551 |
| v12e, mirrored | nine | 5329 | 1246 | 2168 | 987 | 703 | 366 | 552 | 553 |

The winner is v10e3, mirrored, with its nine-value vote
(`vote/d10/v10e3_flip/votes/vote_nine.json`): 5367 against the join bar of 5355.5. The first D10
table (21:06, `vote/d10/table.txt`) held the same rows without v12, v12e and E5, with the same
winner. The mirrored frame adds 3 to 11 of 5700. The branch gained 3717704 (`vote_bank.py --members
--flip`) and ecb14fd (`--cls_flip` in the loop), with 1662 tests.

### 5.6 The final-only vote: D13

The loss of g_Sv2_scr came from the box moves at selection, not from the class. D13 (21:13) keeps
the box at selection and at the lift exactly as under the served rule: the raw overhead read, and a
split waits. At the carry view, read on every lift, the fitted fused sum of all three views plus the
bias decides the final box once. The queued g_Sc_scr, a partial-sum vote, was cancelled before its
start. g_Sf_scr reads the S0 switches, v10e3, `--cls_flip` and the final-only form of its nine-value
vote. It is adopted with at least 2532 correct stows and a held-out row of at least 0.787. A second
rule names the fallback. On a miss of g_Sf_scr, S takes v10e3 with `--cls_flip` under the served
rule, on two conditions: g_Sc0_scr reads at least 2532, and its held-out row reads at least 0.787.
Else v9e stays.

The build (21:43). `feat/ms7g-vote-weights` bfdffb2 holds 897cd22 (a vote JSON with "stage":
"final" is the final-only vote) and bfdffb2 (`vote_bank.py --stage final`), with 1665 tests. The
JSON is `g/tables/vote/vote_v10e3_flip_final.json`, copied to Machine B. On the bank's evaluation part,
on the three-view lifts, the final-only vote equals the nine-value vote exactly: 5367 of 5700,
held-out 1266. Over all 6038 lifts it reads 5649, against 5570 for the served rule and 5680 for the
nine-value vote. 338 lifts lost before the carry view keep the served box. A K = 50 smoke ends
EVAL_OK with 41 of 50.

The two reads:

- g_Sc0_scr (21:14 to 21:59) at ecb14fd, v10e3 with `--cls_flip` under the served rule, stows 2526
  of 3000, 0.8420 [0.829, 0.855], in 3227 cycles and 740,312 steps, at 368.5 an hour. Its held-out
  row reads 601 of 744, 0.808. Against g_S0_scr: -6, a held-out row +0.011, 332 wrong-box parts
  against 355, 129 riders against 159 and 438 `close_miss` cycles against 498. Its fallback rule
  misses by 6.
- g_Sf_scr (22:01 to 22:39) at bfdffb2, v10e3 with `--cls_flip` and the final-only vote, stows 2548
  of 3000, 0.8493 [0.836, 0.862], in 3243 cycles and 738,784 steps, at 372.5 an hour. Its held-out
  row reads 594 of 744, 0.798. Against g_S0_scr: +16, a held-out row +0.001. The in-loop taken class
  is right on 2359 of 2501 lifts, 0.9432, against 0.9309 on g_S0_scr, 0.9271 on g_Sc0_scr and 0.9274
  on g_Sv2_scr. 298 wrong-box parts against 355, 174 `wrong_box` cycles against 196, 61 passengers
  against 91 and 124 riders against 159. The overhead class is wrong on 482 of 3243 selections,
  against 499 of 3271 on g_S0_scr.

The D13 bar (at least 2532, held-out row at least 0.787) is met. S is now F5 200k with
`--carry_guard --overhang_guard`, model v10e3, `--cls_flip` and the final-only vote, at bfdffb2.
g_Sf_scr is its screen read.

### 5.7 Classifier v12

The ledger logged the kinds list at 13:10 from `kinds/kinds_v12.txt`, a file written at 10:05 before
any v12 render. `feat/ms7g-extra-kinds` 8802ece adds 39 classifier-only kinds,
`parts.CLASSIFIER_EXTRA_V12`, by the rule of MS7f D4. A twin rule keeps them away from the held-out
kinds. It allows no shared LDraw number and no held-out family with the same footprint. It also
allows no 1x2 or 1x3 slope of any angle and no 1x2 side-stud brick, among other terms. With MS7f's
45, the extras number 84. A test keeps them off the catalog and off the held-out twins, and 1600
tests pass (05:14).

The renders: 113 files on Machine C, seeds 20321000 + i, from 10:23 to 02:54. They hold
73,839 frames, 26,100 after `--cap 80`. v12 uses v8's recipe and data plus the renders, with repeat
2 by v8's own rule. Its early stop kept epoch 7, at a val_lift of 0.8667. v12 trained at 04:20 with
179,606 unique training frames. v12e is the ensemble of v6a, v8 and v12.

The bank verdict (04:24, D10 final). v12 alone reads 5025 under the served rule and 5183 with its
nine-value vote. Its brick row under the served rule reads 1822 against v9e's 2023. v12e reads 5322
with its nine-value vote and 5329 mirrored. E5 reads 5344 and 5352. All of them read under v10e3
mirrored with its nine-value vote, 5367. v12 stays out (rule G7, D10), and the classifier of S is
final.

C1 is MS7f's read of a classifier on its render sets (MS7f report section 5.2). The C1 rows have no
bar (`loop_sum` over all parts / held-out):

| set | v9e | v12 | v12e | v10e3 |
| --- | --- | --- | --- | --- |
| old set | 0.9389 / 0.8795 | 0.9089 / 0.8524 | 0.9378 / 0.8765 | 0.9439 / 0.8810 |
| deep set | 0.9235 / 0.8357 | 0.8675 / 0.7923 | 0.9224 / 0.8502 | 0.9321 / 0.8744 |

v12 shifts the line between brick and special toward special. The ledger gives the deep-set brick
row as 0.921 to 0.813. The C1 rows agree with the bank.

## 6. The policy levers

### 6.1 Wave 7

Rule G10 records wave 7 whether or not the fallback runs. It is the MS7f wave 4 recorder in the
refill scene on the train split, with the seeds of rule G1. The wave started at 14:20 with S0 as the
driver: candidate 6 with `--carry_guard --overhang_guard` at fc2cccf. S waited for the v12 verdict,
and the corpus keeps only the teacher's takeover frames. Machine C ran 30 lanes and Machine B 5
lanes. D9 (14:36) records the trigger takeovers only. The smoke (Machine A, K = 50, 3 heaps) ended
EVAL_OK three times with 13, 9 and 8 takeovers and 4, 1 and 3 handoffs. The conversion ended
CHECK_OK, and the mask MASK_OK at 7.38 %.

The wave (20:25, conversion and masks at 8d5293d):

| item | value |
| --- | --- |
| heaps | 89 of 90. p = 36 stopped at 20:25, at about 10 minutes a cycle, with no meta, and is in no dataset. |
| takeovers | 2,313, 2,122 of them stowed |
| handoffs placed, not recorded (D9) | 711 |
| episodes | 2,761 |
| frames | 1,075,963 |
| masked frames | 63,380, 5.89 % |
| roots | `g/w7/dataset_{b,ca,cb}` with their masks, all CHECK_OK on Machine A |

The recorder kept the teacher pace flags of waves 4 to 6.

### 6.2 The REOPEN mask and F7

Design section 3.6 found that the corpus teaches the policy to open at lift height. About 55 % of
the teacher's risen lifts end in a REOPEN. A risen lift is a LIFT in which the pick point rose at
least 5 cm. Rule G10 drops the 20 REOPEN frames that follow a risen lift in waves 1 to 7. A REOPEN
at grasp height stays.

The mask track (11:50). `feat/ms7g-reopen-mask` 8d5293d holds 800a3a4 (`reopen_mask.py` and
`pick_points.py`) and 8d5293d (`train_masked.py`, which is `lerobot-train` plus `--frame_mask`).
1609 repo tests and 151 lerobot_stack tests pass. The method is a sample filter in the trainer, with
no new conversion. A masked frame is never a sample, and it pads the action chunks that reach it,
like a segment cut at the REOPEN. The truth rule reads the raw shards. A proxy covers the converted
Machine B halves of waves 1 and 3. Against the truth on Machine A halves, the proxy has a precision of
1.0000 and a recall of 0.9928 to 1.0000. The launcher is `g/bin/retrain_F7.sh`.

The mask on waves 1 to 6 (`mask/mask_tables.md`):

| wave | episodes | frames | masked frames | masked share | risen lifts ending in REOPEN, truth part |
| --- | --- | --- | --- | --- | --- |
| w1 | 5204 | 1790156 | 28560 | 1.60 % | 21.58 % |
| w2 | 5180 | 1700159 | 29300 | 1.72 % | 22.49 % |
| w3 | 3281 | 1620457 | 116040 | 7.16 % | 72.30 % |
| w4 | 3897 | 1692319 | 119400 | 7.06 % | 67.85 % |
| w5 | 1886 | 911919 | 60600 | 6.65 % | 68.99 % |
| w6 | 1748 | 885487 | 58520 | 6.61 % | 71.98 % |
| all | 21196 | 8600497 | 412420 | 4.80 % | 55.06 % |

F7 (D11, 20:31). F7 starts warm from F5 200k and trains 100k steps on waves 1 to 7 with the mask,
OUT `g/train_F7`. The merged corpus `g/w1234567/dataset` holds 23957 episodes and 9,676,460 frames,
and the mask drops 475,800 of them (`retrain_F7.run.log`). The two screens use S0's switches,
because g_Sv2_scr lost under D8.

| read | correct of 3000 | against F5 200k (2532) | `close_miss` | M_tray_open | wrong-box parts |
| --- | --- | --- | --- | --- | --- |
| g_F7_100000_scr | 2429 | -103 | 841 | 55 | 385 |
| g_F7_050000_scr | 2416 | -116 | 922 | 67 | 379 |
| g_S0_scr, F5 200k | 2532 | | 498 | 117 | 355 |

The reading of the ledger: the masked retrain removes the reopen at lift height but loses grasps.
Both F7 checkpoints are not adopted.

### 6.3 The context fix: F7n

The peer note (20:33). Another session found two faults in the policy input:

1. The 16-wide context, `observation.environment_state`, enters ACT unnormalized. LeRobot 0.5.1 ACT
   maps only VISUAL, STATE and ACTION. ENV is IDENTITY at `normalize_processor.py:305`. The context
   has small dimensions: the camera xyz has a std of about 6 mm, target_z 12 mm and the widths 3 to
   9 mm. So they barely move the projection from 16 to 512.
2. The roll of the overhead camera sits near pi and wraps at +-pi (`core/quat.py` atan2). In
   `f/w12345`, 56 % read +3.107 to +3.141 and 44 % read -3.139 to -3.13, a std of 3.12 rad. The
   largest context value flips its sign between episodes.

Either fix changes the meaning of the input, so the projection must learn again. The ledger first
put it out of MS7g's single warm retrain, as an open item. D12 (20:36), an owner decision, tests it
at once.

The context track (21:46) found both faults as the note describes them. The preprocessor of F5 200k
passes the context raw (`ctx/prep_F5_200k.txt`). The roll satisfies |r| >= 3.1068 always, and 52.8 %
of `f/w12345` reads positive, with a std of 3.12 rad. A roll of +3.13 or of `3.13 - 2 pi` is the
same angle. The two values change the served action of F5 200k by up to 0.0168
(`ctx/serve_check.txt`).

The fix. `feat/ms7g-context-norm` holds e0a16d9 and ba61811. With e0a16d9, `policy_server` serves
the roll as `wrap(roll - pi)` for a checkpoint whose marker says "context": "roll_wrapped_norm". It
also adds `+ctxnorm` to the policy id. ba61811 adds `train_masked.py --context_norm --context_proj`,
which normalizes the context MEAN_STD with statistics over the fixed corpus. The serve branch
`feat/ms7g-context-norm-serve` 84ecb70 is 69e7126 plus e0a16d9. Old checkpoints serve
byte-identically. The projection starts as the fold (W diag(std), b + W mean). A paired check of
2000 steps chose it (`ctx/ab_table.txt`, mean loss of steps 1100 to 2000):

| projection start | mean loss |
| --- | --- |
| fold | 0.1145, equal to F7's |
| keep the old weights | 0.1317 |
| a new random start | 0.1597 |

F7n trains from 20:51 with F7's recipe and the fix, OUT `g/train_F7n`. Its training loss equals F7's
within 0.0002 per 10k-step block (`ctx/f7_vs_f7n_blocks.txt`). A 50k smoke at 84ecb70 ends EVAL_OK
with 40 of 50. Every F7n payload carries `+ctxnorm` in its policy id.

| read | correct of 3000 | against F7 at the same step | `close_miss` |
| --- | --- | --- | --- |
| g_F7n_100000_scr | 2402 | -27 | 850 |
| g_F7n_050000_scr | 2437 | +21 | 880 |

The reading of the ledger: the masked pair shows no context effect beyond the noise. All four masked
checkpoints read 95 to 130 under F5 200k. Not adopted.

### 6.4 F8 and F8n: D14

F7 100k's -103 points at the mask, which confounds the context test. D14 (23:14) trains two unmasked
runs on the same corpus, warm from F5 200k, 100k steps, with a paired batch order: F8 with no fix
and F8n with the context fix and the fold projection. Both ended TRAIN_OK.

| read | correct of 3000 | against F5 200k (2532) | held-out row | `close_miss` | M_tray_open |
| --- | --- | --- | --- | --- | --- |
| g_F8_100000_scr | 2476 | -56 | 590 of 744, 0.793 | 559 | 72 |
| g_F8n_100000_scr | 2493 | -39 | 590 of 744, 0.793 | 471 | 83 |

The mask caused most of F7's loss of grasps: F8 reads 559 `close_miss` cycles against F7 100k's 841.
The retrain on waves 1 to 7 itself does not gain, which is MS7f's plateau again.

### 6.5 The verdict on the policy

The D12 and D14 verdict (02:27): no retrain reaches F5 200k + 30. The best, F8n, reads 2493, 39
under F5 200k. So F5 200k stays in S.

The context verdict for the owner. F8n - F8 = +17. F7n - F7 = -27 at 100k and +21 at 50k. All three
lie inside the SE of a pair, 30. The only consistent sign is `close_miss` in the unmasked pair, 471
against 559. The masked pairs do not show it (850 against 841, 880 against 922). At 100k warm steps
the fix moves no share that a read can measure. It stays a recommended input fix for the next
training from scratch, where it costs nothing. MS7g does not adopt it for F5 200k. Section 9.2 gives
the answer from scratch: +124 correct stows.

## 7. The loss decomposition of S

### 7.1 The 40-heap estimate

The two report-only reads give each stack a second set of 20 heaps (draws 20 to 39):

| item | S0 | S |
| --- | --- | --- |
| draws 0 to 19 | g_S0_scr 2532 | g_Sf_scr 2548 |
| draws 20 to 39 | g_S02_scr 2470 | g_Sf2_scr 2507 |
| the 40 heaps | 5002 of 6000, 0.8337 | 5055 of 6000, 0.8425 |
| held-out row, draws 20 to 39 | 626 of 801, 0.782 | 621 of 801, 0.775 |
| held-out row, the 40 heaps | not pooled in the ledger | 1215 of 1545, 0.786 |

g_Sf2_scr (03:06 to 03:40) at bfdffb2 reads 2507 of 3000, 0.8357 [0.822, 0.848], in 3230 cycles and
746,663 steps, at 362.6 an hour. g_S02_scr (03:41 to 04:21) at fc2cccf reads 2470 of 3000, 0.8233
[0.809, 0.837], in 3166 cycles and 733,725 steps, at 363.6 an hour.

The classifier lever, S - S0: +16 on draws 0 to 19 and +37 on draws 20 to 39, +53 of 6000 pooled,
about +26 a 3000-part read. On 40 heaps S is under both staged bars: 0.8425 against 0.85, and 0.786
against 0.80. So a checked read now is likely to miss (03:40).

### 7.2 The rows by category

S on the 40 heaps of g_Sf_scr and g_Sf2_scr (05:43):

| category | correct | share |
| --- | --- | --- |
| brick | 1896 of 2144 | 0.884 |
| special | 1006 of 1283 | 0.784 |
| flat | 685 of 861 | 0.796 |
| joint | 590 of 726 | 0.813 |
| minifig | 543 of 614 | 0.884 |
| round | 335 of 372 | 0.901 |

By family, the lowest rows are ldraw 0.774, tile 0.776, slope 0.797 and plate 0.808.

### 7.3 The wrong class

The `wrong_box` cycles of S by vote case, from the mechanism tables (cycles / class wrong):

| vote case | g_S0_scr | g_Sf_scr | g_Sf2_scr | g_S02_scr |
| --- | --- | --- | --- | --- |
| no lift read | 2 / 1 | 3 / 2 | 1 / 1 | 1 / 1 |
| agreed | 67 / 57 | 75 / 62 | 68 / 59 | 86 / 75 |
| split, voted | 125 / 115 | 93 / 80 | 92 / 82 | 123 / 116 |
| split, carry view lost | 2 / 2 | 3 / 3 | 3 / 3 | 1 / 1 |
| all `wrong_box` cycles | 196 | 174 | 164 | 211 |
| wrong-box parts | 355 | 298 | 316 | 358 |

The final-only vote cuts the class-wrong split votes. On draws 0 to 19 they fall from 115 to 80, and
on draws 20 to 39 from 116 to 82. In the agreed cycles, the overhead view and the wrist view agree
on a wrong class. They stay at 57 to 75. Only a better classifier reaches them, and none after v10e3
read better on the bank (section 5.5).

The in-loop taken class is right on this share of the lifts (21:12, 22:39):

| read | R0 | S0 | g_cg2_scr | g_Sv2_scr | g_Sc0_scr | g_Sf_scr |
| --- | --- | --- | --- | --- | --- | --- |
| taken class right | 0.9157 | 0.9309 | 0.9237 | 0.9274 | 0.9271 | 0.9432 |

### 7.4 The riders and the passengers: D17

Under the guard, the passengers ride with the held part into its box. R0 reads 55 passengers, S0 91,
g_Sf_scr 61 and g_Sf2_scr 90. The riders read 124, 159, 124 and 152.

D17 (04:27) asks for a passenger detector from the measured jaw gap at the lift. The passenger track
traced S on 8 heaps on Machine A, from a worktree at bfdffb2 (`pas/detector.txt`). The heaps use
run 20323010 + i and layout 20323110 + i, i = 0 to 7, and hold 1202 lifts. 80 lifts carry a
passenger, and 34 carry one into a wrong box. The truth labels split the passenger lifts by contact:

| contact at the lift | passenger lifts | with a wrong-box passenger |
| --- | --- | --- |
| pinch2, each finger on a different part | 19 | 6 |
| side, a passenger touches a finger beside the main part | 34 | 18 |
| free, a passenger touches no finger | 27 | 10 |

Three of the detectors (precision / recall):

| detector | any passenger | a wrong-box passenger |
| --- | --- | --- |
| the jaw gap minus the served width, at the lift, at least -2 mm | 0.109 / 0.562 | 0.046 / 0.559 |
| the wrist head's `rider_read` alone | 0.795 / 0.388 | 0.385 / 0.441 |
| the rider logit, at least -3.0 (any) or -2.0 (wrong) | 0.427 / 0.512 | 0.354 / 0.500 |

The D17 verdict (05:41): no detector reaches a precision of 0.9 with a recall of 0.5, so no switch
is built. Only 19 of the 80 passenger lifts are two-part pinches. The other 61 ride beside or on the
held part with the jaws on one part, so the gap looks clean.

### 7.5 The unboxed parts and the bounce-outs: D16

The unboxed census (03:41, the left parts of g_Sf_scr and g_Sf2_scr). About 148 parts a read end
outside every box. 245 of 335 were never in a failed cycle. The places on the 40 heaps: the desk
within 0.30 m and low 147, the tray 80, the desk farther 48, and raised on a rim 20. Of the 195 desk
parts, only 33 are in `layout.in_reach`, and 25 of them are also in the overhead view. So a desk
pickup is not a lever: the desk parts come from drops and knocks out of reach. The ledger addition
of 06:02 gives the rest. 40 of the 335 parts lie parked off the scene (x > 2 m). The four places and
the 40 parked parts make up 335. The loop never perceives a parked part, and the grade counts it as
unboxed.

The bounce census (04:25, `bounce_census.txt`). M_bounce counts the `lifted_lost` cycles whose jaws
opened over the right box and whose part ends unboxed. g_Sf_scr reads 50, g_Sf2_scr 51, g_S0_scr 31
and g_R0_scr 31. These parts end on the desk or a rim, out of reach. That is the largest single loss
after the wrong class. D16 (04:26) asks for a release fix behind its own switch.

The release forensics (05:30, `rel/forensics.txt`):

- The payloads of 13 reads on draws 0 to 19 read M_bounce from 31 to 59: R0 31, S0 31, g_Sc0_scr 34,
  g_cf_scr 37, F8 39, F8n 41, F7 100k 46, g_S0w10_scr 47, g_Sv2_scr 48, g_cg_scr 49, g_Sf_scr 50,
  g_Sfc_scr 56 and g_cg2_scr 59. On draws 20 to 39, S0 reads 50 and S 51. S - S0 is +19 on draws 0
  to 19 and +1 on draws 20 to 39, so the extra under S does not repeat.
- The track traced S on 6 smoke heaps (run 20323000 + i, layout 20323100 + i, i = 0 to 5). They read
  776 of 900 correct stows in 969 cycles and 221,184 steps, with 6 box emptyings. Of 718 first opens
  over a right box, 15 are bounce-outs and 703 stowed.

| mechanism of the 15 bounce-outs | count |
| --- | --- |
| a slip in the carry: the part left the jaws 8 to 24 steps before the open and landed on the desk short of the box | 7 |
| an overfull box: the part was released at the box centre onto a pile 2.2 to 5.6 cm over the rim. The box load was 0.79 to 0.99 of the MS7f emptying capacity. 5 slid off within 12 to 304 steps, 1 still slid at the end of the watch, and 1 rested and was knocked off later. | 7 |
| bounced off a pile 0.3 cm over the rim | 1 |

The release geometry does not differ between the bounce-outs and the stows. Every release is central
and slow. No release lies within 2 cm of a wall or near the guard margin. Neither the box switch,
nor the guard margin, nor the release height reads differently.

The D16 verdict (05:32): no fix. The track built `--empty_at_rim`, a person who empties a box filled
to its rim. That changes the scene and makes it easier, so rule G2 rejects it, and its commit is on
no branch. The robot cannot choose a second box either. 0 of the 7 traced overfull bounce-outs had a
second box of their category. 130 releases went onto a pile at least 2 cm over the rim. The 24 of
them with a second box found it just as full. `lighter_box` balances the two boxes by booked volume,
so both overfill together. 29 of the 101 bounce-outs of g_Sf_scr and g_Sf2_scr had a category with
two boxes.

### 7.6 The emptying capacity

The capacity of a box in MS7f H2 is its interior volume divided by
`desk_boxes.BOX_VOLUME_PER_PART_VOLUME` = 2.0 (MS7f report section 3.3). At that load a person
empties the box. The constant is marked INITIAL. It was measured on the 50-part heaps of MS7e for
the box draw. In the refill scene the pile passes the rim long before that (`rel/forensics.txt`,
section 3):

| box load, a share of the capacity | the pile top over the rim |
| --- | --- |
| 0.3 to 0.4 | median +0.3 cm, 57 % over the rim |
| 0.5 to 0.6 | median +1.5 cm, 40 % over +2 cm |
| 0.7 to 0.8 | median +3.1 cm, 87 % |
| 0.8 to 1.0 | +3.7 to +3.8 cm, 100 % over +2 cm |

Releases with the part still in the jaws, by the pile top over the rim at the open:

| pile top over the rim | releases not in the box |
| --- | --- |
| empty box | 1 of 33 |
| under -1 cm | 6 of 329 |
| -1 to 0 cm | 0 of 114 |
| 0 to 2 cm | 1 of 140 |
| 2 cm or more | 21 of 130 |

So a box holds a pile 2 to 4 cm over its rim for a long stretch of each heap. About 16 % of the
releases onto such piles slide off, about 25 parts a read (05:33). A new calibration of the event to
"pile at the rim" is a scene change under rule G2. The decision belongs to the owner. The owner kept
the scene as it is (D22, section 2.2).

### 7.7 The grip slips: D18

About half of the bounce-outs are grip slips in the carry. The release track counted 12 slips among
689 policy releases on its 6 smoke heaps (`rel/forensics.txt`, section 4). The slips are flat and
odd shapes: tile 3, ldraw 2, slope 2, plate 2, and one each of joint, brick and minifig. The carries
that slip are not faster. Their peak carry speed has a median of 0.44 m/s, and their peak
acceleration 2.4 m/s2. The parts that rested in the box read 0.49 m/s and 2.6 m/s2. So a slip is a
grip problem, not a release problem.

D18 (05:43) builds a carry squeeze (section 2.4). Its smoke gate: on the 6 traced smoke heaps of the
release track, the slips fall to at most half of the 12, with no new failure mode. Then g_Sq_scr
reads S plus the switch.

The D18 verdict (05:53): the carry squeeze failed its smoke gate, and g_Sq_scr did not run. The
switch `--carry_squeeze` used `CARRY_SQUEEZE_M` 0.010. That is about 5 N a jaw, against the policy's
usual 2 N. The smoke compared it with S on the 6 smoke heaps, over the first 12,000 steps:

| row | the squeeze | S |
| --- | --- | --- |
| slips among the releases over the right box | 7 of 187 | 7 of 260 |
| parts out of the jaws within 30 steps of the lift | 52 | 5 |
| dropped cycles | 73 | 22 |

A command that follows the measured gap keeps closing on a part held at an edge until the part
squirts out. Nothing was pushed. The patch is in `g/tables/rel/carry_squeeze.patch`.

## 8. The checked read and the official read

### 8.1 The checked read

Rule G9. The checked read of S runs on seed 20307100, draws 0 to 19, with 10 Machine A and 10 Machine B
lanes. `g/bin/chk_official.sh` runs it and, on a pass, the official read. Every file that a switch
names is on Machine B first. The md5 checks of v10e3 and of the vote JSON on Machine B ran at 21:05 and 22:12.

| item | value |
| --- | --- |
| TAG | g_S_chk |
| configuration | S at bfdffb2: F5 200k, `--carry_guard --overhang_guard`, v10e3, `--cls_flip` and the final-only vote. D18 failed its smoke gate, so no other switch joins S. |
| hosts | worktree 7g-r1, 10 lanes on `a_g1l` and 10 lanes on `b_g1`, PORT_BASE 26600, unit ms7g-chk |
| start and end | 05:54 to 06:29 on 2026-10-04 |
| correct of 3000, share, Wilson 95 % interval | 2545, 0.8483, [0.835, 0.861] |
| cycles, steps | 3219, 725,567 |
| correct stows an hour | 378.8 |
| held-out row | 629 of 785, 0.801 [0.772, 0.828] |
| staged bars (2550, 200, 0.80) | The share FAILS by 5 correct stows. The rate PASSES. The held-out row PASSES (the bar is 628 of 785). The supervisor token is GATE_FAIL. |
| stretch row (2760, 300, 0.85) | Report-only. The share and the held-out row are under it. The rate is over it. |

The mechanism numbers against g_Sf_scr, S on the screen seed (`g_S_chk.mech.txt`):

| row | g_S_chk | g_Sf_scr |
| --- | --- | --- |
| M_tray_open | 111 | 104 |
| M_late | 22 | 26 |
| M_late_wb | 12 | 9 |
| M_stall | 625 | 857 |
| M_swept | 0.1976 | 0.1862 |
| riders | 148 | 124 |
| passengers | 83 | 61 |
| `close_miss` cycles | 419 | 456 |
| `wrong_box` cycles | 170 | 174 |
| wrong-box parts | 318 | 298 |
| `wrong_box` cycles, agreed (cycles / class wrong) | 57 / 50 | 75 / 62 |
| `wrong_box` cycles, split and voted (cycles / class wrong) | 107 / 92 | 93 / 80 |
| handoffs, placed, stowed | 207, 193, 147 | 243, 223, 175 |
| guard steps, over a rim | 5794, 95 | 6186, 54 |
| `stall_open` cycles | 44 | 74 |
| release over the right box | 2673 | 2641 |
| release over a wrong box | 220 | 220 |
| release over the desk | 35 | 51 |
| release over a rim | 6 | 16 |
| lifted, never opened | 14 | 25 |

Against candidate 6's checked read f_F6R_chk (2522, held-out 620 of 785, 308.0 an hour), S reads 23
more correct stows. It also reads 9 more held-out parts and 70.8 more correct stows an hour. Over 60
heaps (the screen draws 0 to 39 and the checked read), S stows 7600 of 9000, 0.8444. Design section
4.2 excludes a second read of the checked seed with the same S. The ledger named three next steps
for the owner: close, a new calibration of the emptying event (the question of 05:33), or a retrain
from scratch with the context fix. The owner chose the retrain (D19, section 9).

### 8.2 The official read

A pass of the checked read on the staged bars is the only start of the official read. The official
read is TAG `g_official`, seed 20307200, draws 0 to 39, one shot, 20 Machine A and 15 Machine B lanes. A
chain that finds `reads/g_official` already started does not start another. If the official read
fails, MS7g fails.

The official read did not run. The checked read g_S_chk missed the share bar, and rule G9 starts the
official read only on a pass. Seed 20307200 stays unburned (D22). The chains of D19 to D21 held a
second checked read each: g_S9_chk, g_Se_chk and g_SA_chk. None of them ran, because no screen
reached its bar of 2578. Section 9.6 gives the verdict of MS7g.

### 8.3 All reads in time order

The ledger time is the stamp of the entry that records the result. The Wilson 95 % intervals and the
rates are the ledger's.

| time | TAG | configuration | correct of 3000 | share | Wilson 95 % | correct stows an hour | held-out row | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11:19 | g_R0_scr | candidate 6, every new switch off | 2501 | 0.8337 | [0.820, 0.847] | 296.5 | 586 of 744, 0.788 | the base of rule G6 |
| 11:54 | g_cg_scr | + `--carry_guard` | 2507 | 0.8357 | [0.822, 0.848] | 364.9 | 580 of 744, 0.780 | not adopted as built (M_late) |
| 12:36 | g_cf_scr | + `--clear_first` | 2518 | 0.8393 | [0.826, 0.852] | 318.2 | 584 of 744, 0.785 | not adopted (M_swept) |
| 13:21 | g_cg2_scr | + `--carry_guard` with the vote-window fix | 2486 | 0.8287 | [0.815, 0.842] | 356.5 | 584 of 744, 0.785 | report row, in S by D5 |
| 14:06 | g_S0_scr | S0: + `--carry_guard --overhang_guard` | 2532 | 0.8440 | [0.831, 0.857] | 364.4 | 593 of 744, 0.797 | the base of later screens |
| 14:51 | g_S0w10_scr | S0 at `--action_warp 1.0` | 2500 | 0.8333 | [0.820, 0.846] | 293.2 | 579 of 744, 0.778 | not adopted (D6) |
| 20:21 | g_Sv_scr | S0 + the nine-value vote of v9e | READ_FAIL | | | | | the vote JSON was missing on Machine B |
| 21:11 | g_Sv2_scr | S0 + the nine-value vote of v9e | 2486 | 0.8287 | [0.815, 0.842] | 351.8 | 563 of 744, 0.757 | not adopted (D8) |
| 21:59 | g_Sc0_scr | S0 + v10e3 + `--cls_flip`, served rule | 2526 | 0.8420 | [0.829, 0.855] | 368.5 | 601 of 744, 0.808 | its fallback rule misses by 6 |
| 22:39 | g_Sf_scr | S0 + v10e3 + `--cls_flip` + the final-only vote = S | 2548 | 0.8493 | [0.836, 0.862] | 372.5 | 594 of 744, 0.798 | adopted (D13), S |
| 23:13 | g_F7_100000_scr | F7 100k + S0's switches | 2429 | 0.8097 | [0.795, 0.823] | 357.2 | 575 of 744, 0.773 | not adopted (D11) |
| 23:59 | g_F7_050000_scr | F7 50k + S0's switches | 2416 | 0.8053 | [0.791, 0.819] | 327.9 | 556 of 744, 0.747 | not adopted (D11) |
| 00:44 | g_F7n_100000_scr | F7n 100k + S0's switches | 2402 | 0.8007 | [0.786, 0.815] | 340.5 | 561 of 744, 0.754 | not adopted (D12) |
| 01:19 | g_F7n_050000_scr | F7n 50k + S0's switches | 2437 | 0.8123 | [0.798, 0.826] | 351.5 | 556 of 744, 0.747 | not adopted (D12) |
| 01:53 | g_F8_100000_scr | F8 100k + S0's switches | 2476 | 0.8253 | [0.811, 0.839] | 368.0 | 590 of 744, 0.793 | not adopted (D14) |
| 02:27 | g_F8n_100000_scr | F8n 100k + S0's switches | 2493 | 0.8310 | [0.817, 0.844] | 380.0 | 590 of 744, 0.793 | not adopted (D14) |
| 03:06 | g_Sfc_scr | S + `--clear_first` | 2517 | 0.8390 | [0.825, 0.852] | 364.9 | 573 of 744, 0.770 | not adopted (D15) |
| 03:40 | g_Sf2_scr | S, draws 20 to 39 | 2507 | 0.8357 | [0.822, 0.848] | 362.6 | 621 of 801, 0.775 | report-only |
| 04:21 | g_S02_scr | S0, draws 20 to 39 | 2470 | 0.8233 | [0.809, 0.837] | 363.6 | 626 of 801, 0.782 | report-only |
| 06:29 | g_S_chk | S, the checked read, seed 20307100 | 2545 | 0.8483 | [0.835, 0.861] | 378.8 | 629 of 785, 0.801 | FAILS the share bar by 5, passes the held-out and rate bars |
| 10-04 09:39 | g_F9n_200000_scr | F9n 200k + S's serving and classifier | 2473 | 0.8243 | [0.810, 0.838] | 273.8 | 588 of 744, 0.790 | report-only (D19b) |
| 10-04 11:59 | g_F9n_400000_scr | F9n 400k + S's serving and classifier | 2474 | 0.8247 | [0.811, 0.838] | 296.9 | 589 of 744, 0.792 | under 2578 (D19) |
| 10-04 14:05 | g_F9n_600000_scr | F9n 600k + S's serving and classifier | 2513 | 0.8377 | [0.824, 0.850] | 325.8 | 578 of 744, 0.777 | under 2578 (D19) |
| 10-04 16:12 | g_F9n_800000_scr | F9n 800k + S's serving and classifier | 2554 | 0.8513 | [0.838, 0.864] | 357.3 | 609 of 744, 0.8185 | under 2578 (D19) |
| 10-04 17:07 | g_F9_800000_scr | F9 800k, the control without the context fix | 2430 | 0.8100 | [0.796, 0.824] | 260.1 | 580 of 744, 0.780 | the control of D19a, under 2578 |
| 10-04 18:47 | g_F9n_1000000_scr | F9n 1.0M + S's serving and classifier | 2490 | 0.8300 | [0.816, 0.843] | 363.7 | 579 of 744, 0.778 | under 2578 (D19c) |
| 10-04 20:15 | g_F9n_1200000_scr | F9n 1.2M + S's serving and classifier | 2444 | 0.8147 | [0.800, 0.828] | 371.6 | 573 of 744, 0.770 | under 2578 (D19c) |
| 10-04 20:48 | g_Se_scr | S with the chunk mean of F5 200k and F9n 800k | 2545 | 0.8483 | [0.835, 0.861] | 377.3 | 600 of 744, 0.8065 | not adopted (D20) |
| 10-04 22:29 | g_A9_100000_scr | A9 100k + S's serving and classifier | 2544 | 0.8480 | [0.835, 0.860] | 363.9 | 608 of 744, 0.817 | under 2578 (D21) |
| 10-04 23:04 | g_A5_100000_scr | A5 100k + S's serving and classifier | 2519 | 0.8397 | [0.826, 0.852] | 354.9 | 591 of 744, 0.794 | not adopted (D21) |
| | g_official, 6000 parts | did not run | | | | | | seed 20307200 unburned (rule G9, D22) |

g_Sq_scr (D18) did not run, because the carry squeeze failed its smoke gate (section 7.7). A screen
has no gate verdict of its own, and the verdict column gives the adoption rule. Every screen but one
misses the staged share bar or the held-out bar. g_F9n_800000_scr reads 2554 and 0.8185, over both
staged bars, but its adoption bar was 2578. Every read passes the staged rate bar.

## 9. After the checked read: the policy levers

### 9.1 The policy from random init: D19 to D19c

The owner decided D19 at 06:35, after the checked miss and a detailed review of the levers in chat.
The rule went into the ledger before any number. F9n is ACT trained from random init with the
context fix. Its screens use S's serving and classifier. The best screen replaces F5 200k in S at
2578 or more, g_Sf_scr + 30. A second checked read on seed 20307100 follows, and the official read
only on a pass. D19a adds a paired control, F9, without the fix. It has the same random init and the
same batch order, so the run also answers the owner's context question.

The recipe (06:38, `g/tables/ctx/retrain_F9.sh`) is the project's recipe for ACT from scratch. The
MS6 runs m/train (400k), l/train_g (600k) and n/train_t1/t4 (800k) used it.

| item | value |
| --- | --- |
| policy | `--policy.type=act`, resnet18 backbone from ImageNet, chunk 100 |
| optimizer | constant learning rate 1e-5, with no scheduler |
| batch, seed | 8, 1000 |
| corpus | `g/w1234567/dataset`, waves 1 to 7, no mask |
| F9n | `train_masked.py --context_norm=roll_wrapped_norm` from step 0: the roll is unwrapped, and `observation.environment_state` is normalized MEAN_STD. The context projection starts fresh, with no fold. |
| F9, the control (D19a) | the same recipe without the fix, with the same init and batch order |
| code, units | 273fabf, units ms7g-train-f9n and ms7g-train-f9 from 06:37, 27.1 steps a second each |

D19b (06:38) took both runs to 800k before any F9 number, because the project's runs from scratch
went to 600k to 800k. Each run resumed at 400k with `lerobot-train --resume`. The resume keeps the
run's own training file, the constant rate of 1e-5 and the optimizer and RNG state. F9n adds
`--context_norm` again. A scratch run from 20 to 40 steps tested the resume first (06:40). D19c
(10-04 16:13) took both runs to 1.2M, because the F9n curve still rose at 800k. Unit ms7g-f9-chain
read each checkpoint as it landed, from worktree 7g-r2 at 273fabf (PORT_BASE 28000). Every F9n
payload carries `+ctxnorm` in its policy id.

The screens on seed 20307000, draws 0 to 19 (the ledger and the mechanism tables):

| read | correct of 3000 | share | Wilson 95 % | correct stows an hour | held-out row | `close_miss` | M_tray_open | wrong-box parts | passengers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F9n 200k | 2473 | 0.8243 | [0.810, 0.838] | 273.8 | 588 of 744, 0.790 | 583 | 187 | 307 | 56 |
| F9n 400k | 2474 | 0.8247 | [0.811, 0.838] | 296.9 | 589 of 744, 0.792 | 451 | 127 | 302 | 71 |
| F9n 600k | 2513 | 0.8377 | [0.824, 0.850] | 325.8 | 578 of 744, 0.777 | 453 | 105 | 294 | 53 |
| F9n 800k | 2554 | 0.8513 | [0.838, 0.864] | 357.3 | 609 of 744, 0.8185 | 441 | 108 | 305 | 91 |
| F9n 1.0M | 2490 | 0.8300 | [0.816, 0.843] | 363.7 | 579 of 744, 0.778 | 494 | 74 | 338 | 76 |
| F9n 1.2M | 2444 | 0.8147 | [0.800, 0.828] | 371.6 | 573 of 744, 0.770 | 418 | 92 | 362 | 101 |
| F9 800k, no fix | 2430 | 0.8100 | [0.796, 0.824] | 260.1 | 580 of 744, 0.780 | 526 | 186 | 373 | 108 |
| S, g_Sf_scr | 2548 | 0.8493 | [0.836, 0.862] | 372.5 | 594 of 744, 0.798 | 456 | 104 | 298 | 61 |

The reading of the ledger. From random init at 200k steps, F9n already reads within 75 of S. That is
the level of the warm retrains (F8 2476, F8n 2493). The curve rises to 2554 at 800k and falls after
it, to 2490 and 2444. At 800k it has 146 `wrong_box` cycles against S's 174, and M_late reads 17.
Both F9n 400k and F9n 1.2M read M_stall 0. The D19 and D19c verdict (10-04 20:15): no F9 screen
reaches 2578, so F5 200k stays in S.

### 9.2 The context verdict from scratch: D19a

F9 800k trained on the same corpus with the same init and batch order as F9n 800k, without the fix.
Its screen ended at 10-04 17:07:

| row | F9n 800k, with the fix | F9 800k, no fix |
| --- | --- | --- |
| correct of 3000 | 2554 | 2430 |
| correct stows an hour | 357.3 | 260.1 |
| held-out row | 609 of 744, 0.8185 | 580 of 744, 0.780 |
| steps | 772,008 | 1,008,998 |
| `close_miss` cycles | 441 | 526 |
| M_tray_open | 108 | 186 |
| riders | 159 | 190 |
| passengers | 91 | 108 |
| wrong-box parts | 305 | 373 |

The CONTEXT VERDICT for the owner (D12, D19a). From scratch the fix is worth 124 correct stows at
800k, about 4 pair SEs. It also gives 97 more correct stows an hour and 0.039 more on the held-out
row. The warm starts F7n and F8n did not show it (section 6.5). F9 1.2M is not read, because the gap
is over 30. Its run stopped to give F9n the GPU.

### 9.3 The policy ensemble: D20

D20 (10-04 19:49) went into the ledger before the F9n 1.2M number. The policy server runs two
checkpoints on each request. Each member keeps its own pre- and postprocessor, so F9n applies its
roll transform by its marker. The server serves the elementwise mean of the two unnormalized action
chunks. The members are F5 200k and the F9n checkpoint with the most correct stows of its 600k to
1.2M screens. When no F9 screen reaches 2578, g_Se_scr reads S with the ensemble on the screen seed.
It replaces S's policy at 2578 or more.

The build (10-04 20:14). `feat/ms7g-policy-ensemble` 769c2eb holds two commits on 273fabf:

- 5e220d7 `policy_server --ensemble_with PATH`: each member keeps its own pre- and postprocessor,
  and the served chunk is the mean of the unnormalized chunks.
- 769c2eb `run_gate_eval` forwards the flag, and the cache key reads `inference.ensemble_with`.

lerobot_stack 167 tests and the repo's 1679 tests pass. With the flag off, the server reproduces the
payload blocks of g_Sf_scr. A K = 50 smoke of F5 200k with F9n 800k ends EVAL_OK at 39 of 50 on one
heap. Without the ensemble, the smoke reads 44 of 50. F9n 800k and 1.2M were copied to Machine B with an
md5 check. Tag `ms7g-read-Se` marks the read SHA. Unit ms7g-d20-chain (`g/bin/d20_chain.sh`) applied
D19, D19c and D20 as written.

The read g_Se_scr ran from 10-04 20:15 to 20:48 at 769c2eb (worktree 7g-r3, PORT_BASE 29100). Its
member is F9n 800k. It stows 2545 of 3000, 0.8483 [0.835, 0.861], in 3119 cycles and 728,585 steps,
at 377.3 an hour. Its held-out row reads 600 of 744, 0.8065. It has 364 `close_miss` cycles against
S's 456, 308 wrong-box parts and 78 passengers. The D20 verdict: under 2578, not adopted. The branch
is draft PR #183 in the stack, off by default.

### 9.4 The low-rate finish: D21

D21 (10-04 20:49) went into the ledger before any number. Every policy so far trained at a constant
learning rate of 1e-5 with no scheduler, and the F9n screens fell after 800k (2554, 2490, 2444). D21
trains two warm runs side by side on Machine A GPU (`g/tables/ctx/retrain_A.sh`, code 273fabf):

| item | A5 | A9 |
| --- | --- | --- |
| start | F5 200k, no context fix | F9n 800k, the context fix kept with the same corpus stats |
| steps, learning rate | 100k at a constant 2e-6 (backbone 2e-6) | the same |
| corpus | `g/w1234567/dataset`, no mask | the same |
| batch, seed, batch order | 8, 1000, paired | the same |
| first logged loss | 0.1533 | 0.1464, against F9n's last 0.1498 |

Every logged step reads a learning rate of 2.0e-06. The first logged losses come from
`ctx/a9_loss_check.txt`. Each run made 27.1 steps a second. Unit ms7g-d21-chain read A9 and then A5
from worktree 7g-r2 at 273fabf (PORT_BASE 29700 and 29800). The better run replaces F5 200k in S at
2578 or more.

| read | correct of 3000 | share | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | `close_miss` | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_A9_100000_scr | 2544 | 0.8480 | [0.835, 0.860] | 3200 | 755,024 | 363.9 | 608 of 744, 0.817 | 444 | 303 |
| g_A5_100000_scr | 2519 | 0.8397 | [0.826, 0.852] | 3264 | 766,613 | 354.9 | 591 of 744, 0.794 | 533 | 334 |

The D21 verdict (10-04 23:04): the best, A9, reads 2544, under 2578. Not adopted.

### 9.5 The plateau on the screen seed

After D21 every pre-registered lever is spent (10-04 23:04). The policies and servings of the last
day read close together:

| read | policy and serving | correct of 3000 |
| --- | --- | --- |
| g_F9n_800000_scr | F9n 800k, from random init with the context fix | 2554 |
| g_Sf_scr | S, F5 200k | 2548 |
| g_Se_scr | the chunk mean of F5 200k and F9n 800k | 2545 |
| g_A9_100000_scr | F9n 800k + 100k steps at 2e-6 | 2544 |
| g_A5_100000_scr | F5 200k + 100k steps at 2e-6 | 2519 |
| g_S_chk | S on the checked seed 20307100 | 2545 |

Five reads of different policies and servings lie from 2544 to 2554 (10-04 22:29). With A5, the
range on the screen seed is 2519 to 2554. The ledger names one quantified lever left: the emptying
capacity, a scene change under rule G2 and the owner's call.

### 9.6 The close: D22

The owner decided D22 at 05:59 on 2026-10-05: MS7g closes as a near miss.

| item | value |
| --- | --- |
| the final candidate | S: F5 200k, `--carry_guard --overhang_guard`, v10e3, `--cls_flip` and the final-only vote, at bfdffb2 |
| the checked read g_S_chk | 2545 of 3000, 0.8483. The bar is 2550, so it misses by 5. |
| held-out row | 629 of 785, 0.801. It passes. |
| correct stows an hour | 378.8. It passes. |
| the official read | did not run. Seed 20307200 stays unburned. |
| against MS7f's final, candidate 6 (f_F6R_chk) | +23 correct stows, +9 held-out parts, +70.8 correct stows an hour |
| the emptying capacity | stays as it is (rule G2) |

The time box of rule G11 ends on 2026-10-06 23:59, and MS7g closes before it. The ledger left three
tasks at the close: the two-tray video of S, this report and the ensemble branch in the stack.
Section 12.1 gives the stack, and section 13 gives the video.

## 10. Errata to the MS7f report

Design section 3.8 carries three errata. They come from a second look at the payloads of f_F6R_chk
and at the debug traces of the video diagnosis.

1. MS7f report sections 10.1 and 10.6 say that most riders are the cycle's own part in a missed
   lift. That holds for candidate 1's screen, not for candidate 6. On f_F6R_chk, 101 of 122 riders
   were never the held part of any cycle. The payload count is 119 riders, or 122 by a per-part
   count that adds 3 parts not at rest. Of the 122, 56 were never a target and never held. 45 were a
   cycle's target that was never held. 19 were the held part of a `held_no_lift`, `carrying` or
   `close_miss` cycle, and 2 were other. The four debug traces hold 32 riders. 14 are passengers
   that entered the box of the held part, and 8 are held parts that were not the target. 7 are
   targets carried while another part was held, and 3 were knocked in or fell in. So passengers are
   the largest group (design section 3.3).
2. MS7f report section 10.1 says that an unboxed part ends outside every box. 9 of the 136 unboxed
   parts of f_F6R_chk lie inside a box region but not at rest, 6 of them in their own box (design
   section 3.4).
3. MS7f report section 10.1 reads candidate 6 as 55 correct stows better than candidate 4. That
   difference is 1.9 standard errors, and the read pair cannot resolve it (design section 3.1).

The MS7g ledger supports two additions:

4. MS7f report section 3.3 describes the box emptying event with no word on the pile height. In the
   refill scene the pile passes the rim at 0.3 to 0.4 of the capacity. It stands 2 to 4 cm over the
   rim before the event. Releases onto such piles slide off about 16 % of the time (section 7.6).
5. MS7f report section 5.1 says that the overhead read sets the box in the policy context. It does
   not say that every MS7f policy receives that 16-wide context unnormalized, or that the camera
   roll in it flips its sign between episodes (section 6.3). The served action of F5 200k changes by
   up to 0.0168 with the sign of the roll.

## 11. Open items

1. The context fix for every future run from scratch. F5 200k and every warm retrain receive an
   unnormalized context with a camera roll that flips its sign. From scratch the fix is worth 124
   correct stows at 800k (F9n 2554 against F9 2430, section 9.2). The warm starts did not show it
   (section 6.5). Every future training from random init takes the fix:
   `train_masked.py --context_norm=roll_wrapped_norm` on `feat/ms7g-context-norm` for training, and
   the marker gate of `policy_server.py` for serving.
2. The emptying capacity. `BOX_VOLUME_PER_PART_VOLUME` = 2.0 is INITIAL, and the pile passes the rim
   at 0.3 to 0.4 of the capacity in the refill scene (section 7.6). About 25 parts a read slide off
   an overfull pile. The owner kept the scene (D22), so the capacity stays as it is. It is the one
   quantified lever left (10-04 23:04). A change is a scene change under rule G2.
3. The passengers. No detector from the jaw gap reaches the D17 bar. 61 of 80 passenger lifts leave
   the jaws on one part. The wrist head's `rider_read` is the better signal at 0.795 / 0.388
   (section 7.4). Under the guard, the passengers go into the box of the held part instead of back
   to the tray. S reads 61 passengers on the screen seed and 83 on the checked read.
4. The grip slips. About half of the bounce-outs are slips of flat and odd parts in the carry
   (section 7.7). The carry squeeze of D18 made the drops worse: 52 parts out of the jaws within 30
   steps of the lift, against 5. A grip fix needs another mechanism than a tighter gap command.
5. The plateau on the screen seed. S, Se, A9, A5 and F9n 800k read 2519 to 2554 on the screen seed
   (section 9.5). The F9n curve peaks at 800k and falls after it at a constant rate of 1e-5. A
   finish at 2e-6 did not lift it. A lever that reaches the staged bars has to move the share
   beyond this band.
6. The held-out row. S reads 0.786 on its 40 screen heaps and 0.801 on the checked read. The weakest
   families are ldraw 0.774, tile 0.776, slope 0.797 and plate 0.808 (section 7.2). The classifier
   v12 aimed at the held-out row with 39 more classifier-only kinds and read worse on the bank and
   on C1 (section 5.7).
7. The wrong class on agreed views. 57 to 75 `wrong_box` cycles a read come from an overhead read
   and a wrist read that agree on the wrong class (section 7.3). No vote rule reaches them.
8. The vote and the loop. A rule can raise the right class on the bank and still lose correct stows
   in the loop. g_Sv2_scr lost them because its bias moves the box at selection (section 5.4). A
   vote rule needs its in-loop read before it serves.
9. The corpus. The REOPEN mask lost grasps (`close_miss` 841 and 922 against 498), and the warm
   retrain on waves 1 to 7 did not gain (section 6.4). The DAgger lever stays flat, as in MS7f.

## 12. Reproduction

### 12.1 The branches, the stack and the tags

The MS7g branches form the GitHub stack #184 on main 5beb717, as draft PRs. The stack #182 of 06:35
held #174 to #181. On 2026-10-05 the ensemble PR #183 went in between #180 and #181, and #184
replaced #182. Each stack commit is patch-identical to its original commit (06:35). At 273fabf,
1678 repo tests and 162 lerobot_stack tests pass. The reads ran at the original commits, and the
tags `ms7g-read-*` keep those commits.

| PR | branch | stack tip | stack commits | original commits |
| --- | --- | --- | --- | --- |
| #174 | `feat/ms7g-census` | 9611585 | 9611585 the L0 census fields and the air-stall detector | ab1deeb |
| #175 | `feat/ms7g-carry-guard` | 127b5aa | 68ac0b4 the rig's perceived carry entry, b1c0e27 `--carry_guard` with the stalls and the handoff, 127b5aa the vote-window fix | 3064851, b8c35d6, 7a2562a |
| #176 | `feat/ms7g-selector` | 190af5c | a7208e6 `--clear_first`, f6751c6 `--overhang_guard`, 6432384 the leaned wall's inner face, 190af5c a test that the switches combine | 5cbe184, 1adfe2c, f654d4f, fc2cccf. Before the 13:20 restack the tip was 81c1341. |
| #177 | `feat/ms7g-vote-weights` | 325ccda | 952cd41 `--bank_views`. 40e503a `--vote_weights`. ed67b2e `vote_bank.py`. a2f995a the D7 forms. 801fbfd a refusal of a form that the loop does not serve. 2263021 a test of the pending fused vote. 888187b a test that the switches combine. 8ea2459 ensembles and the mirror on the bank. 289ea19 `--cls_flip`. f7a5005 the final-only vote. 325ccda `vote_bank.py --stage final`. | f5a1794, 9a75edb, c1467eb, 22ce29e, 0520e01, 2d8bf2a, 69e7126, 3717704, ecb14fd, 897cd22, bfdffb2 |
| #178 | `feat/ms7g-extra-kinds` | de964dc | de964dc the 39 classifier-only kinds of v12 | 8802ece |
| #179 | `feat/ms7g-reopen-mask` | 8c8d2df | 9428524 the REOPEN mask, 8c8d2df `train_masked.py --frame_mask` | 800a3a4, 8d5293d |
| #180 | `feat/ms7g-context-norm` | 273fabf | ff8082d the roll transform of `policy_server.py`, 273fabf `train_masked.py --context_norm --context_proj` | e0a16d9, ba61811 |
| #183 | `feat/ms7g-policy-ensemble` | 769c2eb | 5e220d7 `policy_server.py --ensemble_with`, 769c2eb `run_gate_eval.py --ensemble_with` | the same, built on 273fabf |
| #181 | `docs/ms7g-report` | this report | this report | |

`feat/ms7g-context-norm-serve` 84ecb70 is 69e7126 plus e0a16d9, the serving side of the context fix
for the F7n and F8n reads. It is not in the stack. 81c1341 is on no pushed branch after the restack.
21c77d6, the bank's recording SHA, is the vote branch before its rebase.

The read tags:

| tag | SHA | reads |
| --- | --- | --- |
| `ms7g-read-R0` | ab1deeb | g_R0_scr |
| `ms7g-read-cg` | b8c35d6 | g_cg_scr |
| `ms7g-read-cf` | 81c1341 | g_cf_scr |
| `ms7g-read-cg2` | 7a2562a | g_cg2_scr |
| `ms7g-read-S0` | fc2cccf | g_S0_scr, g_S0w10_scr, g_S02_scr |
| `ms7g-read-Sv` | 69e7126 | g_Sv_scr, g_Sv2_scr, g_F7_100000_scr, g_F7_050000_scr, g_F8_100000_scr |
| `ms7g-read-Sc` | ecb14fd | g_Sc0_scr |
| `ms7g-read-Sf` | bfdffb2 | g_Sf_scr, g_Sfc_scr, g_Sf2_scr, g_S_chk |
| `ms7g-read-F7n` | 84ecb70 | g_F7n_100000_scr, g_F7n_050000_scr, g_F8n_100000_scr |
| `ms7g-read-F9` | 273fabf | the six F9n screens, g_F9_800000_scr, g_A9_100000_scr, g_A5_100000_scr |
| `ms7g-read-Se` | 769c2eb | g_Se_scr |

The test counts in the ledger:

| commit | repo tests | lerobot_stack tests |
| --- | --- | --- |
| ab1deeb, census | 1608 | |
| b8c35d6, carry guard | 1623 | |
| fc2cccf, selector after the restack | 1643 | |
| 41c2ab2, vote before its restack | 1620 | |
| 69e7126, vote after its restack | 1659 | |
| ecb14fd, `--cls_flip` | 1662 | |
| bfdffb2, the final-only vote | 1665 | |
| 8d5293d, reopen mask | 1609 | 151 |
| 8802ece, extra kinds | 1600 | |
| 273fabf, the context fix in the stack | 1678 | 162 |
| 769c2eb, the policy ensemble | 1679 | 167 |

### 12.2 The reads

Every read uses `g/bin/read_g.sh` with the checkpoint F5 200k and the classifier v9e unless the
table says otherwise. "S0 switches" is `--carry_guard --overhang_guard`. The vote JSONs are under
`~/data/maker-arm-ms7/g/tables/vote/`. v10e3 is `f/tables/cls/model_v10e3/part_classifier.pt`.

| TAG | SHA | checkpoint | classifier | ADD | draws | cycles | steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_R0_scr | ab1deeb | F5 200k | v9e | none | 0 to 19 | 3507 | 911,040 |
| g_cg_scr | b8c35d6 | F5 200k | v9e | `--carry_guard` | 0 to 19 | 3255 | 742,084 |
| g_cf_scr | 81c1341 | F5 200k | v9e | `--clear_first` | 0 to 19 | 3461 | 854,589 |
| g_cg2_scr | 7a2562a | F5 200k | v9e | `--carry_guard` | 0 to 19 | 3270 | 753,074 |
| g_S0_scr | fc2cccf | F5 200k | v9e | S0 switches | 0 to 19 | 3271 | 750,340 |
| g_S0w10_scr | fc2cccf | F5 200k | v9e | S0 switches, WARP=1.0 | 0 to 19 | 3142 | 920,721 |
| g_Sv_scr | 69e7126 | F5 200k | v9e | S0 switches `--vote_weights g/tables/vote/vote_v9e.json` | 0 to 19 | READ_FAIL | |
| g_Sv2_scr | 69e7126 | F5 200k | v9e | S0 switches `--vote_weights g/tables/vote/vote_v9e.json` | 0 to 19 | 3298 | 763,222 |
| g_Sc0_scr | ecb14fd | F5 200k | v10e3 | S0 switches `--cls_flip` | 0 to 19 | 3227 | 740,312 |
| g_Sf_scr | bfdffb2 | F5 200k | v10e3 | S0 switches `--cls_flip --vote_weights g/tables/vote/vote_v10e3_flip_final.json` | 0 to 19 | 3243 | 738,784 |
| g_F7_100000_scr | 69e7126 | `g/train_F7/checkpoints/100000` | v9e | S0 switches | 0 to 19 | 3588 | 734,410 |
| g_F7_050000_scr | 69e7126 | `g/train_F7/checkpoints/050000` | v9e | S0 switches | 0 to 19 | 3685 | 795,791 |
| g_F7n_100000_scr | 84ecb70 | `g/train_F7n/checkpoints/100000` | v9e | S0 switches | 0 to 19 | 3553 | 761,943 |
| g_F7n_050000_scr | 84ecb70 | `g/train_F7n/checkpoints/050000` | v9e | S0 switches | 0 to 19 | 3586 | 748,841 |
| g_F8_100000_scr | 69e7126 | `g/train_F8/checkpoints/100000` | v9e | S0 switches | 0 to 19 | 3203 | 726,651 |
| g_F8n_100000_scr | 84ecb70 | `g/train_F8n/checkpoints/100000` | v9e | S0 switches | 0 to 19 | 3204 | 708,451 |
| g_Sfc_scr | bfdffb2 | F5 200k | v10e3 | g_Sf_scr's ADD + `--clear_first` | 0 to 19 | 3244 | 744,919 |
| g_Sf2_scr | bfdffb2 | F5 200k | v10e3 | g_Sf_scr's ADD | 20 to 39 | 3230 | 746,663 |
| g_S02_scr | fc2cccf | F5 200k | v9e | S0 switches | 20 to 39 | 3166 | 733,725 |
| g_S_chk | bfdffb2 | F5 200k | v10e3 | g_Sf_scr's ADD | 0 to 19, seed 20307100 | 3219 | 725,567 |
| g_F9n_200000_scr | 273fabf | `g/train_F9n/checkpoints/200000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3595 | 975,343 |
| g_F9n_400000_scr | 273fabf | `g/train_F9n/checkpoints/400000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3391 | 899,921 |
| g_F9n_600000_scr | 273fabf | `g/train_F9n/checkpoints/600000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3292 | 833,000 |
| g_F9n_800000_scr | 273fabf | `g/train_F9n/checkpoints/800000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3166 | 772,008 |
| g_F9_800000_scr | 273fabf | `g/train_F9/checkpoints/800000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3323 | 1,008,998 |
| g_F9n_1000000_scr | 273fabf | `g/train_F9n/checkpoints/1000000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3195 | 739,374 |
| g_F9n_1200000_scr | 273fabf | `g/train_F9n/checkpoints/1200000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3140 | 710,242 |
| g_Se_scr | 769c2eb | F5 200k | v10e3 | g_Sf_scr's ADD + `--ensemble_with g/train_F9n/checkpoints/800000/pretrained_model` | 0 to 19 | 3119 | 728,585 |
| g_A9_100000_scr | 273fabf | `g/train_A9/checkpoints/100000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3200 | 755,024 |
| g_A5_100000_scr | 273fabf | `g/train_A5/checkpoints/100000` | v10e3 | g_Sf_scr's ADD | 0 to 19 | 3264 | 766,613 |

Each checkpoint path ends in `/pretrained_model`. All screens use seed 20307000, and g_S_chk uses
seed 20307100. Each read's
`PORT_BASE` and worktree are in its ledger entry or in `reads/<TAG>.log`.

### 12.3 The commands

One screen read of S, with the read worktree 7g-r1, as the ledger ran g_Sf_scr:

```bash
G=$HOME/data/maker-arm-ms7/g
VOTE=$G/tables/vote/vote_v10e3_flip_final.json
N=1 SHA=bfdffb29c8ea59a96c42aa73eed2919e034afe89 TAG=g_Sf_scr SEED=20307000 DRAWS=0-19 \
  PORT_BASE=24900 CLS=$HOME/data/maker-arm-ms7/f/tables/cls/model_v10e3/part_classifier.pt \
  ADD="--carry_guard --overhang_guard --cls_flip --vote_weights $VOTE" \
  BASE=g_S0_scr bash $G/bin/sync_read.sh
```

`sync_read.sh` syncs `~/src/maker-arm-lab-7g-r$N` to the SHA on `a_g${N}l` and `b_g$N`. It
runs `read_g.sh` with 10 lanes on each host. Then it writes `g/tables/$TAG.mech.txt` with
`g/bin/mech.py` against `BASE` and prints `SYNC_READ_DONE $TAG`. `read_g.sh` alone needs `TAG`,
`SEED`, `DRAWS`, `HOSTS`, `PORT_BASE` and `WT`. It also takes `ADD`, `CKPT`, `CLS` and `WARP`.

Before a read, copy every file that a switch names (a vote JSON or a classifier) to Machine B. The
mechanism table of any read:

```bash
python3 $HOME/data/maker-arm-ms7/g/bin/mech.py $HOME/data/maker-arm-ms7/reads/<TAG> \
  [$HOME/data/maker-arm-ms7/reads/<BASE_TAG>]
```

The checked read and, on a pass, the official read:

```bash
N=<n> SHA=<40-hex SHA> CLS=<classifier .pt> ADD='<switches>' TAG=<checked TAG> \
  bash $HOME/data/maker-arm-ms7/g/bin/chk_official.sh
```

The controller ran g_S_chk as the user unit ms7g-chk with these values:

```bash
M7=$HOME/data/maker-arm-ms7
N=1 SHA=bfdffb29c8ea59a96c42aa73eed2919e034afe89 \
  CLS=$M7/f/tables/cls/model_v10e3/part_classifier.pt \
  ADD="--carry_guard --overhang_guard --cls_flip \
       --vote_weights $M7/g/tables/vote/vote_v10e3_flip_final.json" \
  TAG=g_S_chk PORT_BASE=26600 OFF_PORT=27000 \
  bash $M7/g/bin/chk_official.sh > $M7/g/tables/g_S_chk.chain.log 2>&1
```

The log ends with `CHK_MISS g_S_chk`, so the official read on OFF_PORT did not start.

The retrains: `g/bin/retrain_F7.sh` (F7), `g/tables/ctx/retrain_F7n.sh`, `retrain_F8.sh` and
`retrain_F8n.sh`. The bank tables: `lerobot_stack/vote_bank.py` on `feat/ms7g-vote-weights`, and
`g/tables/vote/bin/d10_read.sh` for the D10 table.

The runs from random init and the low-rate finishes, from the worktree 7g-ctx at 273fabf:

```bash
C=$HOME/data/maker-arm-ms7/g/tables/ctx; G=$HOME/data/maker-arm-ms7/g
CTX=1 OUT=$G/train_F9n bash $C/retrain_F9.sh            # F9n to 400k, the context fix from step 0
CTX=0 OUT=$G/train_F9 bash $C/retrain_F9.sh             # F9, the paired control
CTX=1 OUT=$G/train_F9n OUTFILE=$C/retrain_F9n.out WAIT_UNIT=none STEPS=800000 \
  bash $C/resume_F9.sh                                  # D19b, and STEPS=1200000 for D19c
INIT=$G/train_F9n/checkpoints/800000/pretrained_model CTX=1 OUT=$G/train_A9 bash $C/retrain_A.sh
INIT=$HOME/data/maker-arm-ms7/f/train_F5/checkpoints/200000/pretrained_model CTX=0 \
  OUT=$G/train_A5 bash $C/retrain_A.sh                  # D21, LR=2e-6 by default
```

Each launcher writes the marker `action_space.json` into each checkpoint as it lands. A context-fix
checkpoint carries `"context": "roll_wrapped_norm"`, so the policy server applies the roll transform
and adds `+ctxnorm` to the policy id. The chains `g/bin/f9_chain.sh`, `f9b_chain.sh`,
`d20_chain.sh` and `d21_chain.sh` ran the screens and applied the rules of D19 to D21. Their logs
are `g/tables/f9_chain.log`, `f9b_chain.log`, `d20_chain.log` and `d21_chain.log`.

## 13. Video

The two-tray video of S in the refill scene, from the owner's camera (design section 11), on draw
14 of the checked seed:

`~/data/maker-arm-ms7/g/video/ms7g_S_refill_d14_two_trays.mp4`

The video is H.264, 1280x720, 60 fps, at 4x speed, 161.8 s long. A full decode reads no errors.
The heap is the one of MS7f's two-tray video: run seed 3127230830, layout seed 3127231330,
permutation 2,0,3,1. The camera stands across the desk at a distance of 0.9 m. The run stops at
the second refill, after the robot takes 80 parts from the tray. The kit is `cmds_S.py`,
`cmds_S_d14.json` and `run_video_S.sh` in the same folder. It runs from a worktree at bfdffb2.

| Run | Sim time at the cut | Correct | Wrong box | Cycles | Correct an hour |
|---|---|---|---|---|---|
| S (this video) | 10:19 | 73 | 5 | 83 | 424 |
| MS7f candidate 6 | 13:45 | 68 | 10 | 99 | 297 |

A gate episode does not replay exactly. A first render with the same commands reached the cut at
11:39 with 69 correct, 10 wrong, 88 cycles and 355 an hour. It stays beside the final file as
`ms7g_S_refill_d14_two_trays_run1.mp4`. One heap is a picture of the behavior, not a read.
