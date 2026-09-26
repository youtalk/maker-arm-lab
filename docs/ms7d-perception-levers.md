# MS7d: the perception levers

Date: 2026-09-25. Status: final. The official read passes both bars, and MS7d passes its gate
under the perceived state at nominal noise.

The design is the MS7d design. The owner approved it section by section in chat. It follows the MS7c
diagnosis, which read the payloads of the MS7c reads (`docs/ms7c-perceived-state.md`) and the traces
of the QDD contact spike. The campaign ledger is `~/data/maker-arm-ms7/d/README.md`, and `STATE.md`
beside it is the resume pointer. Every rule below went into that ledger before its number. The read
payloads are under `~/data/maker-arm-ms7/reads/<TAG>/`, and every MS7d TAG starts with `d_`. The
counts that the ledger does not give come from those payloads, through
`~/data/maker-arm-ms7/d/bin/row_report.py`. The requirement table v2 comes from
`~/data/maker-arm-ms7/d/bin/req_table_v2.py`, which reads the sweep payloads.

The ledger names its rules R1 to R9, and the isolation rows are also named R0 to R9. This report
writes "rule R5" for a rule and "row R5" for an isolation row.

All times are PDT. Each time is the time of the ledger entry that records the event. All times are
on 2026-09-25.

## 0. Summary

MS7c put one boundary between the simulator and the serving loop. Under `--perceived`, the loop
reads only a perceived state and a hold signal from the jaw gap. MS7c failed its gate on the cycles
bar. Its best checked read, candidate 3, stowed 2646 of 3000 parts at 1.329 cycles per correct stow,
against a bar of 1.3.

The MS7c diagnosis then traced the cycles of candidate 3. Four costs were each about as large as the
miss: the in-box re-picks, the per-episode lateral bias of the camera, the class errors and the
false holds. Three more costs were smaller: a slow release, a stop flag that one false hold latched
off for the rest of a cycle, and failed re-picks beside the desk boxes.

MS7d keeps the MS7c protocol and the MS7c gate. It adds nine levers. Each lever has its own switch,
and every switch is off by default:

| lever | switch | kind | what it removes |
| --- | --- | --- | --- |
| L1 | `--box_filter_xy` | loop fix | the in-box re-picks |
| L2 | `--release_gap` | loop fix | the dead steps of a slow release |
| L8 | `--rearm_hold` | loop change | the dead steps after a false or failed hold |
| L9 | `--box_walls` | loop change | the failed re-picks beside a desk box |
| L3 | `--marker_bias` | new real input, a marker on the gripper | the per-episode lateral bias |
| L4 | `--wrist_fix` | new real input, a wrist camera look | the bias and most of the noise at the grasp |
| L5 | `--class_views` | new real input and model, a third class view | the wrong-box parts |
| L6 | `--hold_force` | new real input, the torques of the arm's own motors | the false holds |
| L7 | `--force_cap` | safety | finger presses of up to 45 N |

The new inputs are stand-ins made of the truth plus noise, as in MS7c. Each one has knobs, and MS5a
measures their values on the real rig.

The pass rule is the MS7c rule. At nominal noise, a checked read must stow at least 2550 of 3000
held-out parts in their own class's box. It must use at most 1.3 cycles per correct stow. The better
passing checked read then gets one official read on the unused seed 20262100, with 6000 parts and
the bars 5100 and 1.3.

The gate outcome:

| read | checkpoint | switches | commit | correct stows | cycles per correct stow | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| candidate A, checked | train_P 200k | all nine | cdcfb65 | 2759 of 3000, 0.920 | 1.123 | both bars met |
| candidate B, checked | r2 A2 100k | all nine | cdcfb65 | 2698 of 3000, 0.899 | 1.211 | both bars met |
| official, candidate A | train_P 200k | all nine | cdcfb65 | 5521 of 6000, 0.920 | 1.123 | both bars met, PASS |

Both checked reads meet both bars. A has the lower cycles per correct stow, so A takes the official
read. The official read of A stows 5521 of 6000 at 1.123 cycles per correct stow, so it passes both
bars. MS7d passes its gate under the perceived state, with every stand-in knob at its nominal. The
stand-in values are optimistic until MS5a measures them, and the requirement table v2 gives the
margins.

The full-stack row and both checked reads ran before the verdicts on two levers, to save time while
slow CPU lanes read. Each of them counts on one condition: the adoption rule adopts every lever. The
rule adopts all nine levers (21:09). So the full-stack row d_R9_all counts as row R9, and both
checked reads stand.

Each checked read pairs with an MS7c candidate on the same heaps and the same noise draws. A gains
113 correct stows over candidate 3 and uses 0.206 fewer cycles per correct stow. B gains 95 over
candidate 1b and uses 0.204 fewer.

The other results:

- The hold calibration with the force gate passes at 5 N. It reads the same agreement as the MS7c
  calibration. The median first-hold offset meets its bar of 5 steps exactly, as in MS7c.
- L1 and L2 fix defects, and the rule adopts them. On the isolation seed, L1 takes the in-box
  re-picks from 67 to 1. The two together cut the cycles that end on the step budget from 885
  to 729.
- The no-harm rule also adopts the other seven levers. The wrist look, L4, is the largest lever: 69
  more correct stows and 0.103 fewer cycles per correct stow than the base. The marker bias
  estimate, L3, is next: 42 more and 0.072 fewer. No lever reads worse than the base.
- The full stack reads 2753 of 3000 at 1.129 cycles per correct stow on the isolation seed. The
  retrain trigger does not fire, so no retrain runs and no candidate C exists.
- On A, the census counts 605 fewer failed cycles than on candidate 3. `close_miss` falls by 353,
  `held_no_lift` by 109, `lifted_lost` by 88 and `wrong_box` by 41. False holds fall from 232 to 74,
  and wrong-box parts from 189 to 139.
- The requirement table v2 marks no knob tight. `pos` keeps the stow bar to at least scale 4 and the
  cycles bar to scale 3.26, a bias sigma of 6.5 mm per axis and a noise sigma of 4.9 mm in xy. Every
  other knob, and a class error that is fully persistent, keeps both bars at its top measured scale.
  Against the MS7c table, every tolerance is wider.
- Machine C ran 40 EGL sim processes on its one GPU. Its CPU lanes then took 4 to 6 hours a
  draw, against about 40 minutes on a GPU lane and about 90 minutes on the same server at 6 lanes.
  Two rows wrote no payload there in 7 hours. They stopped and read again, one of them on GPU lanes.

## 1. The goal, the gate and the rules

The ledger fixed rules R1 to R9 at 11:27, before any MS7d number. The design gives MS7d three days,
from 2026-09-26 to 2026-09-28, and at most one retrain, only on a pre-registered trigger. There is
no extension. The campaign ran on 2026-09-25 (section 2).

| read | seed and draws | pass |
| --- | --- | --- |
| checked, per candidate | 20261700, draws 0 to 11, 3000 parts, a fresh TAG | at least 2550 correct stows, at most 1.3 cycles per correct stow |
| official, only the better passing checked read | 20262100, draws 0 to 23, 6000 parts, 120 episodes, one shot | at least 5100 correct stows, at most 1.3 cycles per correct stow |

A correct stow is a part at rest in its own class's box. The design names three candidates, and each
one runs the full stack:

- A: train_P 200k. Its MS7c pair is candidate 3, TAG c_P_200000_chk.
- B: r2 A2 100k. Its MS7c pair is candidate 1b, TAG c_r2A2_100000_chk_b.
- C: the best screen of the retrain. C exists only after a retrain.

The checked seed and draws are those of MS7c. So each checked read pairs with its MS7c candidate on
the same heaps and the same noise draws (MS7d design section 7.5). The better passing checked read
is the one with the lower cycles per correct stow. The official read is the milestone's number. If
no checked read passes, the milestone fails, and the report gives the decomposition.

The read protocol (rule R3) is the MS7c R4 protocol. Every read passes `--tray --desk_boxes 4
--tray_lean 30`. It serves the nine MS7a switches `--select_at_release --replan_on_switch
--squeeze_floor --square_wrist --scripted_rake --part_failures --pick_snap --dwell_close
--adopt_held`, the tail fixes `--park_done --rake_turn`, and `--perceived`. It uses held-out parts
(`--parts_split heldout --parts_seed 20260915`), heaps of K = 50 and 50 actions per forward. The
knob scales are 1 unless a row says otherwise. Each read adds the MS7d switches of its row.
`~/data/maker-arm-ms7/d/bin/read_d.sh` builds every read from these flags.

The isolation rows and the adoption rule (rule R5). Every row reads train_P 200k on seed 20269900,
draws 0 to 11, 3000 parts. No gate read uses that seed. MS7c read its draws 0 to 3 in the snap
clearance tuning. Row R0 has no MS7d switch. Row R1 is the base, L1 and L2. Rows R2 to R8 each add
one lever to the base, and row R9 adds every adopted lever. The rule:

- L1 and L2 fix defects. They are adopted, unless row R1 reads worse than row R0 by 0.05 cycles per
  correct stow or more, or by 60 correct stows or more.
- Every other lever, L7 included, is adopted unless its row reads worse than row R1 by the same
  margins.

The margins are about 1.5 standard deviations of the difference of two reads of 3000 parts
(MS7d design section 7.3). A bootstrap over the 60 episodes of candidate 3 gives a standard
deviation of 0.023 cycles per correct stow at 3000 parts. Two runs of one control on the same seeds
differed by 26 parts in 1000 in the QDD spike. So the rollouts are not bit-identical, and every row
reads 3000 parts. Each row also reports its in-box re-picks, its false holds, its parts over the
failure cap, its hold agreement and its census. `row_report.py` prints these, and `row_report.py
--compare BASE TAG` prints the no-harm verdict. It uses the MS7c diagnosis definitions:

- A false hold is a cycle where the detector reads a hold and the contacts never hold a part
  (`hold_detected` with `held_steps` 0).
- An in-box re-pick is a cycle whose target was last placed and graded stowed.
- A part over the failure cap is a part with more than 3 failed cycles.

At 11:55, `row_report.py` reproduced the MS7c numbers exactly on the MS7c payloads. For candidate 3,
it gave 2646 and 1.329, with 92 in-box re-picks and 232 false holds, 170 to the budget. It also gave
19 parts over the cap with 96 failed cycles.

The retrain trigger (rule R6), set before row R9 read. If row R9 reads above 1.25 cycles per correct
stow, or under 2610 correct stows of 3000, the one retrain runs. The two values keep about two
standard deviations of margin to the bars at 3000 parts. The retrain records 120 train-split heaps
under `--perceived` with the adopted levers that the recorder reads. It warm-starts from train_P
200k on Machine B and trains 200k steps with the MS7a arm A recipe. Its screens read each 50k checkpoint
with the full stack on seed 20269900, draws 0 and 1.

The requirement table v2 (rule R7). The table reads the official winner with the full stack. With no
winner, it reads the best checked read. Every row reads seed 20261700, draws 0 and 1, 500 parts. One
row, structure only, sets every knob at 0. Then each of 12 knobs reads at 2 and at 4, with the other
knobs at 1. `rho` reads at 0 and at 2. The nominal row is draws 0 and 1 of the winner's checked
read.

The seeds (rule R1):

| use | seed | draws |
| --- | --- | --- |
| isolation rows, retrain screens | 20269900 | draws 0 to 11 (3000 parts), screens on draws 0 and 1 |
| checked | 20261700 | draws 0 to 11 (3000 parts), a fresh TAG |
| requirement table v2 | 20261700 | draws 0 and 1 (500 parts) |
| official | 20262100 | draws 0 to 23 (6000 parts, 120 episodes), unused before MS7d |
| hold calibration, process i | run seed 20267700 + i, layout seed 20268700 + i, i = 0 to 2 | one heap per process |
| retrain corpus, process p | run seed 20274000 + p, layout seed 20275000 + p, p = 0 to 39 | never a read seed. Not used, since no retrain runs. |

The rules that carry over (rule R8). A rule enters the ledger before its number. A payload from the
supervisor cache is not a read. Every read records the SHA, the `policy_id`, the split, the serving
switches, the MS7d switches and the knob scales. A mid-training checkpoint gets its
`action_space.json` marker before a read. A READ_FAIL, a read with no gate token, runs once more in
a fresh TAG with the same settings. The MS6 negatives stay closed: no noise-scaled corpus, no
round-1 DAgger frames, no 25-action horizon and no close gate. The generic abort on contact of the
QDD spike joins them.

The hosts (rule R9) keep their MS7c roles, `scripts/fleet.py` and the host table
`~/data/maker-arm-ms7/hosts.json`. One checkpoint ships to a host once, before the reads that need
it.

## 2. The decisions and the deviations

### 2.1 The decisions of the design

The owner took these decisions before the design (MS7d design section 2):

| id | decision |
| --- | --- |
| D1 | Pass the MS7c gate under the perceived state with a margin. Use loop fixes and inputs that a real rig can give. Two such inputs are a wrist-camera fix at the final approach and a second class view with a correlated error model. |
| D2 | The MS7c bars. The checked read uses seed 20261700, draws 0 to 11, so it pairs with the MS7c candidates. The official read uses the unused seed 20262100. |
| D3 | Three days, 2026-09-26 to 2026-09-28. At most one retrain, only on a pre-registered trigger. No extension. |
| D4 | Fold in the QDD spike: a hold gate on the force at the pick point (L6), a force cap (L7) and a census field at selection. The generic abort on contact stays closed. |
| D5 | A truth-plus-noise stand-in for every new input, each behind its own switch, measured by isolation rows on a common base. No learned or image-based detector. |
| D6 | Parallel worktrees build the lever branches on a common base. Each isolation row reads as soon as its branch is ready. |
| D7 | The paused MS6 runs T1 and T2 were killed on 2026-09-25, so both GPUs are free. |
| D8 | The latched stop flag gets L8, and the re-picks against the desk box walls get L9. The failure-count leak gets no lever, because L1 removes 14 of its 19 parts. |

The design puts these items out of scope: a generic abort on contact, a response by surface kind, a
guarded descent and a learned detector. Real-rig work is out too, and so are vertical walls, more
boxes, more classes and DAgger. The spike's generic abort on contact, `--touch_block`, lost 60 to 86
parts in 1000.

### 2.2 The deviations, the lane changes and the record gap

Every entry below is in the ledger at the time given. The lane changes keep the settings of their
reads. Where the ledger gives no reason, the table says so.

| time | ledger word | what happened | why |
| --- | --- | --- | --- |
| 11:27 | | The campaign runs on 2026-09-25. The design schedules it from 2026-09-26 to 2026-09-28. | The ledger gives no reason. |
| 11:57 | DEVIATION | With `--release_gap` on, a perceived `placed` or `dropped` end selects the next target at once. It does not wait in `release_ready` for the opening of 0.8. With the switch off, nothing changes. | Under MS7c the release itself ended at 0.8, so that wait was always empty. With L2 on, a wait there adds back the steps that L2 removes. |
| 11:57 | | Every host synced to 7e4173c with `--force`, after a check that no process had its cwd in any campaign worktree. `hosts.json` gained worktree aliases, `<host>_b` and `<host>_c` first and up to `<host>_h` by 12:27. The design names one campaign worktree per host. | The stopped lanes of the MS7c sweep read RUNNING forever. The aliases let reads at different SHAs share a host (section 11). |
| 12:18 | DEVIATIONS | `--rearm_hold` refuses `--contact_hold` with `EvalFailure("rearm_hold")`. A payload with the switch off gains `rearms: 0` fields. | The contact fallback replaces the detector's flag, so there the switch does nothing. The decisions and the draws with the switch off do not change. |
| 12:25 | | A payload with `--class_views` off gains three zero counts: `third_views`, `votes_changed` and `wrong_voted_class`. The vote compares the wrist class with the target's overhead class. It does so also on a lift of a part that is not the target. | The ledger records the counts without the word deviation. It calls the vote rule "as designed". |
| 12:58 | | The stack was restacked before rows R2 to R8 read. Those rows read at the pre-restack lever tips. The restack added three integration commits on top (section 3). | The restack keeps the lever code unchanged. Only the list of unbuilt switches, the payload count lists and the tests change. The design restacks after rows R2 to R8. |
| 13:40 | LANE CHANGE | Row R4 moves from a chain on `c_e:12` to `a_e:8` and `b_e:4`. The chain stopped before it launched. | The aarch64 lanes were slow (section 11). |
| 14:27 | LANE CHANGES | The chains of rows R7 and R3 on Machine C stop before they launch. Row R7 runs on `a_g:8` and `b_g:4`, and row R3 follows on `a_e:8` and `b_e:4`. | The aarch64 lanes ran at load 63 with 40 lanes. Rows R1, R5, R2 and R6 had run there for 2 to 2.5 h, while a GPU read takes about 40 min. |
| 15:31 | | No new read goes to Machine C until its rows end. | Its lanes had written no payload after 3 to 3.5 h at load 70. |
| 15:32 | PRE-REGISTERED, PROVISIONAL | The full stack reads as d_R9_all with all nine switches, before the verdicts of every row. It counts as row R9 on one condition: rule R5 adopts every lever. If a lever is dropped, row R9 reads again with the adopted set in a fresh TAG, and d_R9_all is descriptive only. | "A deviation to save time while the aarch64 rows run." |
| 16:39 | PRE-REGISTERED, PROVISIONAL | The checked reads of A and B, as d_A_chk and d_B_chk, with all nine switches, on the same condition as d_R9_all. If a lever is dropped, both read again with the adopted set in fresh TAGs. | The same reason. |
| 17:32 | FOUND, NOT IN THIS LEDGER BEFORE | d_R6_d11 is a fleet read of row R6's draw 11 alone, at 0fdc1cc with the row R6 settings, on `b_f:1`. It started about 17:10 with no ledger entry. The entry names it a record gap of the controller session. Its rule came before its number: the verdict of row R6 reads draws 0 to 10 from d_R6 and draw 11 from d_R6_d11, merged as d_R6_merged. If d_R6's own lane ends draw 11 first, the verdict uses that draw and d_R6_d11 is descriptive. | The controller session launched the read from its own background task and wrote no entry. Its purpose, read off its command: R6's aarch64 lane 0 held draws 5 and 11 in sequence, and a CPU draw took about 5 h. |
| 17:35 | | d_R6_d11 ends GATE_OK with 234 of 250, and it joins d_R6_merged. | The rule of 17:32. |
| 18:01 | PRE-REGISTERED, PROVISIONAL | The requirement table v2 reads candidate A with all nine switches, on the same condition. The official read waits for the last verdicts. | The official read is one shot. |
| 19:33 | LANE CHANGE | d_R2 on 12 `c_c` lanes and the 6 `c_f` lanes of d_R6 (draws 5 to 11) stop and end READ_FAIL. The ledger saw no payload of either. Row R2 reads again as d_R2g at c090ecc on `a_c:8` and `b_c:4`. Row R6's draws 5 to 10 read again as d_R6_d5_10 at 0fdc1cc on `c_f:6` alone. The verdict of row R6 reads d_R6_merged: draws 0 to 4 from d_R6's Machine B lane dir, draws 5 to 10 from d_R6_d5_10 and draw 11 from d_R6_d11. | Both reads had run 7 h with no payload, and their clients were due to hit the 480-minute wall cap near 20:20. The cause is the density of 40 EGL processes on Machine C's one A2000. |
| 19:53 | CORRECTIONS | The check of the payloads for this report corrected seven numbers of earlier entries: B against candidate 1b is -0.204 cycles, not -0.203. The Wilson intervals of rows R5 and R8 are 0.890 to 0.911 and 0.882 to 0.905. The aarch64 draws of rows R1 and R5 took 4.0 to 6.0 h. Row R4's approach force per draw is p50 0.34 to 0.37 N and p90 0.85 to 3.07 N. The 17:35 entry means the 17:32 rule. Row R6's aarch64 count was 7 draws until d_R6_d11 took draw 11, then 6. | The entries above stay as written. This report uses the corrected values. |
| 20:54 | | The stopped fleet reads d_R2 and d_R6 never wrote a token, and neither TAG has an aggregate. | Their aarch64 lanes were killed at 19:33. |
| 21:05 | two notes | The 21:05 entry on the requirement table was written before the full table was read on screen. A first note says every number was checked afterwards. A second note corrects the first: that "checked afterwards" was also written before a readable view of the table was on screen. The check then ran by reading `d/req_table_v2.md` in full, and every number in the entry and in the note matches the table. | The ledger records the order of events plainly. No number changed. |
| 21:09 | PRE-REGISTERED | The official read, d_A_official, after the last verdict (section 8). | Every lever is adopted, so the provisional reads stand. |

The fleet read of d_R6 ends before it pulls its lane dirs, so its Machine B lane dir,
`reads/d_R6/hosts/b_f`, was pulled by hand. The ledger took a descriptive look at those five Machine B
draws at 15:32: draws 0 to 4 stowed 1132 of 1250 at 1.187 cycles per correct stow, against 1117 at
1.310 for row R0 on the same draws. The entry says it is not a verdict.

## 3. The code

### 3.1 The stack

The code is one stack of one-concern branches on main cc5f009. Every new switch is off by default,
so the MS7c protocol stays reproducible. Every script keeps the token contract. The stack, bottom to
top:

| order | branch | tip | what it adds | PR | the lever row read at |
| --- | --- | --- | --- | --- | --- |
| 1 | `feat/ms7d-loop-fixes` | 7e4173c | The switch and knob plumbing (2b4c3b1). L1 (c45a669). L2 and the census field `target_in_box` (7e4173c). | #120 | 7e4173c, rows R0 and R1 |
| 2 | `feat/ms7d-rearm-hold` | c090ecc | L8 | #121 | c090ecc, row R2 |
| 3 | `feat/ms7d-box-walls` | b1acb8b | L9 | #122 | 9b896b5, row R8 |
| 4 | `feat/ms7d-marker-bias` | 0ffac99 | `MarkerBias` (bccda1d), the loop call of L3 (0ffac99) | #123 | de83232, row R5 |
| 5 | `feat/ms7d-wrist-fix` | 4837884 | `WristCamera` (cd90756), the serving row of L4 (4837884) | #124 | 0fdc1cc, row R6 |
| 6 | `feat/ms7d-class-views` | aac0b95 | `ClassViews` (7211f6e), the vote of L5 (aac0b95) | #125 | f011e8d, row R7 |
| 7 | `feat/ms7d-torque` | cdcfb65 | `core/torque.py` and `env.arm_torques` (72727d6), the loop wiring of L6 and L7 (3a6be66), and three integration commits: 6099f4f, 810fa9a, cdcfb65 | #126 | 7f8985e, rows R3 and R4. cdcfb65, row R9 and the checked reads |
| 8 | `docs/ms7d-report` | | this report | #128 | |

PRs #120 to #126 form GitHub stack #127. The six lever branches were built at the same time by six
subagents in parallel worktrees from 7e4173c (D6). Each lever keeps its logic in its own functions,
so the branches meet only in the plumbing of branch 1 and at their call sites.

At 12:58, the branches were restacked in the design order, 16 commits over main. The lever rows read
at the pre-restack tips in the last column. A `git range-diff` of each lever branch against its
pre-restack tip shows the same lever code. Only the list of unbuilt switches, the payload count
lists, the docstrings and the tests differ. Three integration commits sit on the torque branch:

- 6099f4f: the force window reads the serving row of the wrist look, as `--pick_snap` does, and
  `look_due` calls the same `snap_window`. It changes only a run with both `--wrist_fix` and
  `--hold_force` or `--force_cap`. No isolation row has both, so only row R9 and the checked reads
  read it.
- 810fa9a: `NOT_BUILT` is empty and goes away, since every lever is built.
- cdcfb65: a test rollout turns all nine switches on together on the perceived teacher scene.

The CPU suite passes at each tip. It runs 1179 tests at 7e4173c, 1181 at c090ecc, and 1183 at
9b896b5 and at de83232. It runs 1185 at 0fdc1cc, 1186 at f011e8d, 1191 at 7f8985e and 1216 at the
top, cdcfb65. The tests run under `MUJOCO_GL=disable`, as in CI.

### 3.2 The plumbing and the census field

2b4c3b1 adds the nine switches to `loop.rollout`, to the heap client `scripts/mj_eval_heap.py` and
to the gate supervisor `scripts/run_gate_eval.py`. Each payload records them in its inference block.
All switches except `--hold_force` and `--force_cap` need `--perceived`. Without it, such a switch
is a usage error of the client, with the token line. The two force switches also run on the teacher,
for the hold calibration. The supervisor forwards each switch only to the MuJoCo heap client, and
its cache keys on the switches and the knobs. The cache refuses the MS7c payloads on the MS7d code,
because their `noise_scales` list 6 knobs and the MS7d code lists 13.

`perception.KNOBS` gains seven new knobs: `tool`, `wrist`, `wrist_bias`, `wrist_miss`, `rho`,
`torque` and `torque_k`. So `all=V`, the payload and the cache key carry them. Until a lever branch
built its behavior, `loop.NOT_BUILT` listed its switch, and the rollout failed on it with
`EvalFailure("not_built")` before the first step. So a switch never ran as a no-op
(MS7d design section 6).

The census field `target_in_box` (MS7d design section 4.3) comes in 7e4173c. At each selection, it
records whether the target's true centre lies in a true box region, any box, by the grader's test.
It is a metric read, and no decision reads it. It measures L1 in every read, and it is the check of
the cause that the QDD spike asked for.

### 3.3 The loop changes

L1, `--box_filter_xy` (c45a669). Under `--perceived`, the MS7c candidate filter tested the noisy 3D
centre of a detection against the grader's box region. The lower face of that region is the box
floor. A z error of a few mm under a part at rest in its box put the part back among the candidates.
Candidate 3 spent 92 cycles on such parts (the MS7c diagnosis). With the switch, the filter tests
only the x and y of a detection against each perceived box's outer footprint. The footprint is the
interior plus the wall, grown by `desk_boxes.BOX_FILTER_MARGIN_M` = 5 mm on every side. The test
runs in the box's own frame (`desk_boxes.in_outer_footprint`). The margin carries an `INITIAL:`
marker. The loop sets `HeapOracle.box_filter_xy` on every selector, so the rake reads the same
candidate mask. The true-input loop keeps the grader's region. Its counts are the in-box re-picks
and `target_in_box`.

L2, `--release_gap` (7e4173c). A perceived release completed only at a measured opening of 0.8 of
the full gap. On candidate 3, 107 cycles ran to the step budget in that wait, 101 of them with the
part already in a box. With the switch, a release also completes at a measured jaw gap 10 mm wider
than the gap at the hold's end (`Cycle.release_gap0`). The 10 mm is `RELEASE_OPEN_M`. The constant
carries an `INITIAL:` marker. A new detected hold still cancels the release, and the MS7c rule still
completes it. Its counts are the budget ends and the `lifted_lost` cycles.

L8, `--rearm_hold` (c090ecc). Under `--perceived`, the stop flag of the serving rules was the
detector's first hold or lift in the cycle, latched. One false or failed hold turned
`--square_wrist`, `--pick_snap` and its landing hold, `--close_gate` and `--dwell_close` off for the
rest of the cycle. The policy's own wrist roll was then served at once near the ground. On candidate
3, the flag was on with no lift in 503 of 3517 cycles, a median 522 steps before the cycle ended.
With the switch, the serving rules read `Cycle.stop`, which `rearm_stop` updates after them on each
step. A detected hold or lift turns it on. A hold that ends with no lift turns it off, and a spent
`--dwell_close` is armed again. A lifted hold that ends keeps the flag. The census fields
`hold_detected`, `lift_detected` and `first_hold_at` do not change, so the hold agreement reads as
in MS7c. The lever makes no random draws. Its count is `rearms`, in the census row and in the
perception block.

L9, `--box_walls` (b1acb8b). The grasp model and the rake knew only the tray walls. On candidate 3,
33 of 68 re-picks of a part beside a desk box failed, against 2 of 22 on true inputs.
`desk_boxes.box_walls` builds the four walls of each perceived desk box from the perceived box
poses: each wall's rectangle turned by the box yaw, its top at the rim, facing out and with no lean.
The selectors add them to the tray walls. `grasp.wall_allows` then refuses a grasp whose finger
lands on a box wall. The rake sweeps a part within `grasp.RAKE_BAND_M` of a box wall off that wall,
away from the box. The rake hull clears every box wall. A box wall never adds the wide grasp along a
wall (`hugs_wall` skips it), because `WALL_MAX_GRASP_WIDTH` was driven on the tray walls only. The
teacher's oracles keep the tray walls only. Its counts are `box_wall_rakes` and `box_wall_refusals`,
the grasps on each selected target that a box wall alone refused.

### 3.4 The new inputs

L3, `--marker_bias` (bccda1d, 0ffac99). The per-episode camera bias goes into every grasp of an
episode. On candidate 3, the episodes with a lateral bias of 3.5 mm or more read 1.426 cycles per
correct stow. Those under 2.5 mm read about 1.265 (the MS7c diagnosis). `MarkerBias` models one
fixed point on the gripper, a fiducial of known geometry, that the overhead camera reads at every
perceive call. Its world point is the FK pick point. A look is that point plus the `Perceiver`'s
episode bias plus the tool noise. The tool noise is `TOOL_SIGMA_M` = 1 mm on each axis (knob `tool`,
`INITIAL:`). The estimate is the mean of the look minus the FK point over the episode's looks so
far. The loop subtracts it from the x, y and z of every detection. This happens before the selector,
the failure and rake memories, the row and the frozen marker read them. The box poses stay, because
MS7c models their bias apart from the camera's. The model has one limit: a real camera error is not
a pure translation, so a real estimate from the gripper removes it only near the gripper. Its counts
are `marker_looks`, one per perceive call, and `marker_error_mm`. The second is the sum over the
selections of the estimate's xy distance to the true bias, a metric read.

L4, `--wrist_fix` (cd90756, 4837884). MS7c found that the snap drives the tool onto the noisy
perceived centre. No MS7c serving change replaced a position fix at the final approach.
`WristCamera` is one env's wrist look at the target. It returns the target's true x, y and the
heading of its body x axis. It adds a hand-eye bias per episode and a noise per look. The nominals
carry `INITIAL:` markers. The look noise is `WRIST_SIGMA_M` = 0.5 mm and `WRIST_YAW_SIGMA_RAD` = 2
degrees (knob `wrist`). The hand-eye bias is `WRIST_BIAS_SIGMA_M` = 0.5 mm (knob `wrist_bias`), and
the miss rate is `WRIST_MISS_PROBABILITY` = 0.05 (knob `wrist_miss`). A look finds nothing at the
miss rate. It also finds nothing on a target whose centre another part covers from above. The loop
takes one look each time the pick point enters the snap window (`look_due`): within `SNAP_RADIUS` =
40 mm of the serving row's centre in xy and under its top plus `SQUARE_ABOVE` = 40 mm. A look that
finds the target sets `Cycle.serving_row` (`wrist_row`): the look's centre x and y, and the
closing-axis heading turned by the look's yaw minus the detection's yaw at selection, folded into
[-pi/4, pi/4). The z, the width and the half height stay. Only the serving rules read the serving
row: `--pick_snap` and its landing hold, `--square_wrist`, `--dwell_close` and `--close_gate`. The
policy's row and the frozen marker stay as selected, so the existing checkpoints run with no
retrain. The one truth read is `wrist_look_truth`, a decision accessor like `perception_truth`. Its
counts are `wrist_looks`, `wrist_misses` and `wrist_covered`. Two more are the sums of the serving
row's xy move (`wrist_shift_mm`) and heading turn (`wrist_turn_deg`) against the row at selection.

L5, `--class_views` (7211f6e, aac0b95). Each wrong wrist class put one part in a wrong box, 69 on
candidate 3. The MS7c model draws every class view apart at p. That is optimistic for a vote,
because a real classifier errs alike on the views of one part. `ClassViews` gives each part, once
per episode, a persistent wrong class at the chance share times p. The share is
`CLASS_PERSISTENT_SHARE` = 0.5 (`INITIAL:`) times the `rho` knob. p is `CLASS_ERROR_PROBABILITY` =
0.03 times the `class` knob. Both are capped at 1. Every view of a persistent part returns that
class. Any other part is wrong at q = p (1 - share) / (1 - share p), so a single view still errs at
p. At the detector's lift, a wrist class equal to the selection's overhead class is taken. Any other
draws a third view at once, a second wrist look during the carry. The loop takes the class that two
of the three views share, or the third view on a three-way split (`voted_class`). At a share of 0, a
view is the MS7c classification draw for draw. Its counts are `third_views`, `votes_changed` and
`wrong_voted_class`, a metric read.

### 3.5 The force: L6 and L7

The QDD spike found that the arm's joint torques detect a finger push of 5 N or more at any
plausible noise. On candidate 3, the detector read 232 false holds, 170 of them to the step budget.
In the spike traces, 82 % of the false holds had a finger pressed on the desk, the tray or a box.
Their median true force at the pick point was 12.0 N, against 0.6 N on true holds.

72727d6 adds `core/torque.py`, torch and stdlib only, and the MuJoCo member `env.arm_torques(i)`.
The member reads arrays only, with no contact loop. It gives the constraint torque on the six arm
joints (`qfrc_constraint`), which equals the contact torques J^T f. It also gives the gravity
torque, the arm actuator torques and the (3, 6) Jacobian of the FK pick point. It stands for the
torque reports of the motors. `ForceSensor` forms the residual: the constraint torque, plus k (a G +
b tau_dyn) with a and b per joint per episode, plus white noise. k is `TORQUE_MODEL_K` = 0.05 times
the `torque_k` knob, and the noise is `TORQUE_NOISE_SIGMA_NM` = 0.05 Nm times the `torque` knob.
Both carry `INITIAL:` markers. The force is the norm of the least-squares image of the tared
residual through the Jacobian (`pick_force`). Inside the snap window it reads against the median of
the 10 steps before the last entry, a tare taken in free motion. Elsewhere it reads against the
spike's rolling tare, steps t-15 to t-6.

3a6be66 wires the two switches. Under either one, the loop reads `env.arm_torques` at the top of
every live step, before the hold detector. The member and the sensor cost about 0.2 ms per env step
on the three-part test scene.

- L6, `--hold_force`. The detector enters a hold only on a step whose force is under
  `HOLD_FORCE_MAX_N` = 5 N (`INITIAL:`). A refused entry does not reset the count of steps that meet
  the jaw-gap rule. So the hold enters on the first step under the bar. The force never ends a hold.
  A true hold that starts with a finger on the floor enters at the lift, which takes the finger off.
  The census reads the gated detector. The trace rows carry the force as a ninth column.
  `scripts/hold_agreement.py --force_max` replays a trace at 8 N or 12 N without a new rollout.
- L7, `--force_cap`. On a step whose force is over `FORCE_CAP_N` = 15 N (`INITIAL:`), the six arm
  columns of the command, after every serving lever, the clamp and the park, become the measured arm
  joint positions. The gripper column passes. The cap applies to the policy and to the rig.

The payload gains a torque block only under a switch. It holds the steps capped, the hold entries
refused and the cycles with one, the knob scales and the bars. It also holds the force of the
approach steps, from a cycle's open to its first close, as p50, p90 and max.

### 3.6 Random streams, pairing and the no-truth test

Each new model draws from its own generator, seeded like the `Perceiver`:
`random.Random("<name>:<run_seed>:<env_index>")` with the names `marker`, `wrist`, `class_views` and
`torque`. Each draws the same count at every scale. The marker draws 3 values per look, and the
wrist 2 bias values and then 4 per look. The class views draw 2 values per part, and the torque 12
at the reset and 6 per step. The `Perceiver`'s own stream keeps its draw count on every call. A test
pins its draws with every MS7d switch off to a sequence recorded at cc5f009. So with every MS7d
switch off, the perception draws equal the MS7c draws, and every new knob multiplies paired draws.

Row R0 checks this. Its draws 0 to 3 stow 893 of 1000. MS7c's c_tune_k0 read 892 of 1000 on the same
draws at 8458f66. So the MS7d code with its switches off repeats MS7c.

The MS7c test that no decision reads the truth covers every new switch. It runs a perceived rollout
twice, and the second time every old truth accessor raises or returns noise. At cdcfb65 it runs with
all nine switches on. The wrist look reads the truth through one named accessor, the marker reads
FK, and the torque reads the env member. These stand for real sensors.

## 4. The hold calibration with the force gate

The bar (rule R4) came before the calibration. It is the MS7c R5 bar. On the teacher's cycles of the
three MS7c calibration heaps, precision and recall must be at least 0.95 for the events held and
lifted. The median offset of the first hold must be at most 5 steps. At a failure at 5 N, the rule
tries 8 N and then 12 N, and the first passing value freezes. With no passing value, L6 leaves the
milestone and L7 stays.

The ledger pre-registered the calibration at 12:25. It ran at 7f8985e on Machine A CPU in
`~/src/maker-arm-lab-ms7e`, from 12:27. The teacher drove with `--oracle_driver --hold_force` on the
rung 2 lean scene, one heap per process, with `--hold_trace`. It passed at 12:31. The results, over
142 cycles (47, 48 and 47):

| event | true positives | false positives | false negatives | precision | recall |
| --- | --- | --- | --- | --- | --- |
| held | 140 | 1 | 0 | 0.993 | 1.000 |
| lifted | 138 | 0 | 2 | 1.000 | 0.986 |

The teacher stowed 47, 47 and 47 of 50, as in MS7c. The median first-hold offset is 5.0 steps
against the bar of 5, met exactly, as in MS7c. The agreement equals the MS7c calibration count for
count. The trace replay at `--force_max 5` gives the same, HOLD_OK. The gate refused 1, 1 and 5 hold
entries on 1, 1 and 2 cycles. The approach force read p50 0.31 to 0.34 N, p90 0.62 to 0.87 N and max
8.1 to 15.9 N per heap. `HOLD_FORCE_MAX_N` freezes at 5 N for every read under `--hold_force`, and
the 8 N and 12 N replays are not needed.

Every read payload records the same agreement on the policy's cycles, as a diagnostic. The bar
applies to the teacher's cycles only, so these numbers change no rule. Over the 12 draws of each
read, with the first-hold offset pooled over the cycles:

| read | hold precision | hold recall | lift precision | lift recall | median first-hold offset |
| --- | --- | --- | --- | --- | --- |
| row R0, no MS7d switch | 0.909 | 0.989 | 0.965 | 0.984 | 6 steps |
| row R1, the base | 0.937 | 0.986 | 0.965 | 0.986 | 6 steps |
| row R2, L8 | 0.937 | 0.989 | 0.970 | 0.996 | 6 steps |
| row R3, L6 | 0.957 | 0.983 | 0.974 | 0.984 | 6 steps |
| row R4, L7 | 0.946 | 0.988 | 0.971 | 0.987 | 6 steps |
| row R5, L3 | 0.947 | 0.993 | 0.976 | 0.987 | 6 steps |
| row R6, L4 | 0.941 | 0.994 | 0.973 | 0.991 | 6 steps |
| row R7, L5 | 0.936 | 0.990 | 0.972 | 0.990 | 6 steps |
| row R8, L9 | 0.950 | 0.987 | 0.979 | 0.984 | 6 steps |
| d_R9_all, the full stack | 0.968 | 0.991 | 0.978 | 0.997 | 6 steps |
| candidate 3 (MS7c) | 0.910 | 0.988 | 0.956 | 0.985 | 6 steps |
| candidate A | 0.968 | 0.990 | 0.980 | 0.999 | 6 steps |
| candidate 1b (MS7c) | 0.909 | 0.985 | 0.953 | 0.983 | 6 steps |
| candidate B | 0.967 | 0.990 | 0.975 | 0.995 | 6 steps |

The hold precision rises from 0.909 to 0.968 on the full stack. The median first-hold offset on the
policy's cycles stays at 6 steps.

## 5. The isolation rows

### 5.1 The reads

Every row reads train_P 200k, policy id
`act:~/data/maker-arm-ms7/c/train_P/checkpoints/200000/pretrained_model+nas50+ctx16+gap`,
on seed 20269900, draws 0 to 11. Each row was pre-registered with its TAG, SHA and lanes before it
launched. The start time comes from the read's `fleet.json`, and the result entry is the ledger
entry that gives its number.

| row | lever | TAG | SHA | pre-registered | lanes | started, result entry |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | none | d_R0 | 7e4173c | 11:57 | Machine A 8, Machine B 4 | 11:58, 12:52 |
| R1 | L1 and L2, the base | d_R1 | 7e4173c | 11:57 | Machine B 2, aarch64 10 | 11:58, 17:44 |
| R2 | L8 | d_R2, then d_R2g | c090ecc | 12:18, re-read 19:33 | aarch64 12 (stopped), then Machine A 8, Machine B 4 | d_R2g 19:33, 20:54 |
| R3 | L6 | d_R3 | 7f8985e | 12:31 | Machine A 8, Machine B 4 | 15:13, 16:00 |
| R4 | L7 | d_R4 | 7f8985e | 12:27, moved 13:40 | Machine A 8, Machine B 4 | 13:41, 14:27 |
| R5 | L3 | d_R5 | de83232 | 12:08 | aarch64 12 | 12:09, 18:11 |
| R6 | L4 | d_R6_merged | 0fdc1cc | 12:27, 17:32, 19:33 | Machine B 4 (draws 0 to 4), Machine B 1 (draw 11), aarch64 6 (draws 5 to 10) | d_R6 from 12:27, d_R6_d11 17:10, d_R6_d5_10 19:34, 21:09 |
| R7 | L5 | d_R7 | f011e8d | 12:27, moved 14:27 | Machine A 8, Machine B 4 | 14:27, 15:31 |
| R8 | L9 | d_R8 | 9b896b5 | 12:20 | Machine A 8, Machine B 4 | 12:52, 13:40 |

### 5.2 The verdicts

Rule R5 compares row R1 with row R0, and every other lever row with row R1. A lever is dropped on
one condition: its row reads worse by 0.05 cycles per correct stow or more, or by 60 correct stows
or more.

| row | switches on the MS7c protocol | SHA | correct stows of 3000 | cycles per correct stow | in-box re-picks | false holds, to the budget | parts over the cap, their failed cycles | wrong-box parts | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | none | 7e4173c | 2665, 0.888 | 1.321 | 67 | 243, 188 | 21, 94 | 191 | the control |
| R1 | `--box_filter_xy --release_gap` | 7e4173c | 2662, 0.887 | 1.288 | 1 | 155, 98 | 11, 45 | 182 | L1 and L2 adopted: -3 stows, -0.033 cycles against R0 |
| R2 | the base, `--rearm_hold` | c090ecc | 2683, 0.894 | 1.268 | 2 | 147, 96 | 11, 48 | 155 | L8 adopted: +21, -0.021 |
| R3 | the base, `--hold_force` | 7f8985e | 2665, 0.888 | 1.276 | 1 | 98, 55 | 11, 46 | 190 | L6 adopted: +3, -0.012 |
| R4 | the base, `--force_cap` | 7f8985e | 2685, 0.895 | 1.272 | 0 | 139, 97 | 10, 45 | 191 | L7 adopted: +23, -0.016 |
| R5 | the base, `--marker_bias` | de83232 | 2704, 0.901 | 1.216 | 0 | 133, 92 | 5, 24 | 168 | L3 adopted: +42, -0.072 |
| R6 | the base, `--wrist_fix` | 0fdc1cc | 2731, 0.910 | 1.185 | 0 | 152, 103 | 10, 44 | 171 | L4 adopted: +69, -0.103 |
| R7 | the base, `--class_views` | f011e8d | 2683, 0.894 | 1.283 | 0 | 151, 111 | 12, 52 | 148 | L5 adopted: +21, -0.005 |
| R8 | the base, `--box_walls` | 9b896b5 | 2682, 0.894 | 1.274 | 0 | 114, 85 | 8, 34 | 167 | L9 adopted: +20, -0.014 |
| R9 | all nine | cdcfb65 | 2753, 0.918 | 1.129 | 0 | 67, 30 | 2, 8 | 130 | row R9, since every lever is adopted: +91, -0.160 |

Row R1 ended at 17:44, and the verdicts of rows R1, R3, R4, R7 and R8 came then. Row R5 came at
18:11, row R2 at 20:54 and row R6 at 21:09. No row reads worse than its base, so rule R5 adopts all
nine levers.

The Wilson 95 % intervals and the other counts per row:

| row | Wilson 95 % | per draw | placed, budget, dropped ends | episodes on the cycle allowance | rakes |
| --- | --- | --- | --- | --- | --- |
| R0 | 0.877 to 0.899 | 222 220 225 226 224 226 223 224 212 218 221 224 | 2562, 885, 73 | 12 of 60 | 514 |
| R1 | 0.876 to 0.898 | 222 219 226 206 219 228 222 222 230 231 222 215 | 2611, 729, 89 | 6 of 60 | 458 |
| R2 | 0.883 to 0.905 | 223 235 218 226 227 220 215 224 226 226 216 227 | 2622, 687, 92 | 8 of 60 | 473 |
| R3 | 0.877 to 0.899 | 223 218 230 220 223 230 218 221 223 226 218 215 | 2610, 711, 78 | 8 of 60 | 477 |
| R4 | 0.884 to 0.905 | 228 224 229 216 237 222 216 227 219 224 220 223 | 2633, 684, 98 | 7 of 60 | 506 |
| R5 | 0.890 to 0.911 | 223 228 222 228 231 231 223 226 219 219 230 224 | 2642, 562, 83 | 2 of 60 | 507 |
| R6 | 0.900 to 0.920 | 223 231 228 225 225 228 219 233 221 231 233 234 | 2678, 488, 68 | 1 of 60 | 488 |
| R7 | 0.883 to 0.905 | 220 223 230 200 225 229 236 214 226 226 227 227 | 2592, 750, 101 | 10 of 60 | 513 |
| R8 | 0.882 to 0.905 | 230 224 230 224 227 227 213 220 217 230 209 231 | 2592, 740, 84 | 5 of 60 | 457 |
| R9 | 0.907 to 0.927 | 225 229 228 219 234 229 229 240 235 232 231 222 | 2675, 355, 76 | 0 of 60 | 492 |

A few cycles end `stowed`: 1 in row R0, 2 in rows R3, R6 and R8, 1 in row R5 and 1 in row R9. These
are episodes that the env's own success ended with every part in its box.

### 5.3 The census of each row

The census class of every cycle:

| row | cycles | stowed | close_miss | held_no_lift | lifted_lost | wrong_box | carrying | no_close | no_open |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 3521 | 2427 | 515 | 235 | 197 | 82 | 44 | 20 | 1 |
| R1 | 3429 | 2469 | 427 | 242 | 155 | 90 | 32 | 13 | 1 |
| R2 | 3401 | 2499 | 406 | 222 | 135 | 74 | 42 | 23 | 0 |
| R3 | 3401 | 2466 | 386 | 243 | 159 | 93 | 36 | 18 | 0 |
| R4 | 3415 | 2494 | 358 | 252 | 161 | 98 | 24 | 28 | 0 |
| R5 | 3288 | 2522 | 308 | 179 | 151 | 89 | 15 | 24 | 0 |
| R6 | 3236 | 2555 | 276 | 156 | 117 | 84 | 31 | 15 | 2 |
| R7 | 3443 | 2494 | 431 | 252 | 158 | 51 | 33 | 24 | 0 |
| R8 | 3418 | 2475 | 361 | 258 | 171 | 94 | 38 | 21 | 0 |
| R9 | 3107 | 2580 | 189 | 104 | 121 | 55 | 43 | 15 | 0 |

The census counts cycles, and the grader counts parts at rest at the episode end, so the two stowed
counts differ.

### 5.4 Each lever in its row

- L1 and L2, row R1. The in-box re-picks fall from 67 to 1, and `target_in_box` from 107 to 0. The
  budget ends fall from 885 to 729, and the `lifted_lost` cycles that end on the budget from 133
  to 86. The env steps of the read fall from 437076 to 399913. False holds fall from 243 to 155, and
  the parts over the failure cap from 21 to 11. The stows stay level, -3, and the cycles per correct
  stow fall by 0.033.
- L8, row R2. The stop flag cleared 966 times. The count of false holds barely moves, 155 to 147,
  because L8 does not stop a false hold. It ends the dead steps after one: `held_no_lift` falls from
  242 to 222, `close_miss` from 427 to 406, and the budget ends from 729 to 687. The row gains 21
  stows and uses 0.021 fewer cycles per correct stow.
- L6, row R3. The gate refused 726 hold entries on 343 cycles. False holds fall from 155 to 98, the
  fewest of the single-lever rows, and those to the budget from 98 to 55. The `held_no_lift` class
  stays at 243. The hold precision rises from 0.937 to 0.957.
- L7, row R4. The cap held the arm on 6265 steps, 160 to 997 per draw. The approach force read p50
  0.34 to 0.37 N, p90 0.85 to 3.07 N and max 19.8 to 92.6 N per draw. `close_miss` falls from 427
  to 358. The row gains 23 stows, so the safety rule costs nothing.
- L3, row R5. The estimate's mean lateral error is 0.31 mm per selection, over 5273 marker looks,
  against a bias sigma of 2 mm per axis. `close_miss` falls from 427 to 308, and `held_no_lift` from
  242 to 179. The budget ends fall from 729 to 562, and the parts over the cap from 11 to 5. This is
  the second largest lever, after L4.
- L4, row R6. The loop took 4356 wrist looks. 216 missed, 0.050 of the looks, and 7 found the target
  covered. Over the 4133 looks that found the target, the serving row moved a mean 7.0 mm in xy and
  turned a mean 5.8 degrees. `close_miss` falls from 427 to 276, the fewest of the single-lever
  rows, and `held_no_lift` from 242 to 156. The budget ends fall from 729 to 488, and the episodes
  on the cycle allowance from 6 to 1. The row gains 69 stows and uses 0.103 fewer cycles per correct
  stow. This is the largest single lever. It reads with no retrain, because the policy's row and the
  marker stay as selected.
- L5, row R7. The loop drew 145 third views, and the vote changed the class 40 times. The wrong
  wrist class was drawn 83 times, and the class taken after the vote was wrong 45 times. Wrong-box
  parts fall from 182 to 148, and the `wrong_box` class from 90 to 51. The cycles per correct stow
  barely move, -0.005.
- L9, row R8. The rake swept a part off a box wall 7 times, and a box wall alone refused 4 grasps,
  so L9 itself acts rarely. False holds fall from 155 to 114, and the parts over the cap from 11
  to 8.

The seven lever changes against row R1 sum to +199 correct stows and -0.243 cycles per correct stow.
The full stack reads +91 and -0.160 against row R1 (section 6). So the effects do not add, as the
MS7c isolation rows also found. Several levers act on the same failed cycles. L3 and L4 both act on
the grasp point, and L6, L8 and L9 all act on the cycles after a false hold.

## 6. The full stack and the retrain trigger

The ledger pre-registered the full stack at 15:32 as d_R9_all, at cdcfb65 with all nine switches, on
`a_h:8` and `b_h:4` (section 2). It read from 16:00 to 16:39.

| item | d_R9_all | row R1 | row R0 |
| --- | --- | --- | --- |
| correct stows of 3000 | 2753, 0.918 | 2662, 0.887 | 2665, 0.888 |
| Wilson 95 % | 0.907 to 0.927 | 0.876 to 0.898 | 0.877 to 0.899 |
| cycles | 3107 | 3429 | 3521 |
| cycles per correct stow | 1.129 | 1.288 | 1.321 |
| in-box re-picks | 0 | 1 | 67 |
| false holds, to the budget | 67, 30 | 155, 98 | 243, 188 |
| parts over the cap, their failed cycles | 2, 8 | 11, 45 | 21, 94 |
| placed, budget, dropped ends | 2675, 355, 76 | 2611, 729, 89 | 2562, 885, 73 |
| `close_miss`, `lifted_lost`, `held_no_lift` cycles | 189, 121, 104 | 427, 155, 242 | 515, 197, 235 |
| wrong-box parts | 130 | 182 | 191 |
| episodes on the cycle allowance | 0 of 60 | 6 of 60 | 12 of 60 |

Against row R1, the full stack reads 91 more correct stows and 0.160 fewer cycles per correct stow.
Against row R0, it reads 88 more and 0.193 fewer. Its lever counts follow. The stop flag cleared 462
times. The loop took 4097 wrist looks, with 197 misses and 9 covered targets, and 5005 marker looks.
It drew 146 third views. The cap held the arm on 4449 steps, and the gate refused 418 hold entries
on 266 cycles. The rake swept 5 parts off a box wall.

The 15:32 rule counts d_R9_all as row R9 on one condition: rule R5 adopts every lever. It does
(21:09), so d_R9_all is row R9, and no row R9 reads again.

The retrain trigger (rule R6) reads row R9. 1.129 is not above 1.25, and 2753 is not under 2610. So
the trigger does not fire. No retrain runs, the retrain corpus seeds stay unused, and no candidate C
exists. The one retrain that the design allows is not spent.

## 7. The checked reads

### 7.1 The reads

The ledger pre-registered both checked reads at 16:39, provisional on the same condition as
d_R9_all. The condition held at 21:09, so both reads stand. Each reads at cdcfb65 with all nine
switches, nominal knobs and the rule R3 protocol. Each reads seed 20261700, draws 0 to 11, on
`a_h:8` and `b_h:4`. A ran first, from 16:39 to 17:18, and B after it, from 17:18 to 18:01.
The policy id of r2 A2 100k is
`act:~/data/maker-arm-ms7/r2/train_A2/checkpoints/100000/pretrained_model+nas50+ctx16+gap`.

| item | candidate 3 (MS7c) | candidate A | candidate 1b (MS7c) | candidate B |
| --- | --- | --- | --- | --- |
| checkpoint | train_P 200k | train_P 200k | r2 A2 100k | r2 A2 100k |
| TAG | c_P_200000_chk | d_A_chk | c_r2A2_100000_chk_b | d_B_chk |
| commit | a5482d6 | cdcfb65 | eae617a | cdcfb65 |
| MS7d switches | none | all nine | none | all nine |
| stowed of 3000 | 2646, 0.882 | 2759, 0.920 | 2603, 0.868 | 2698, 0.899 |
| Wilson 95 % | 0.870 to 0.893 | 0.909 to 0.929 | 0.855 to 0.879 | 0.888 to 0.910 |
| cycles | 3517 | 3098 | 3681 | 3266 |
| cycles per correct stow | 1.329 | 1.123 | 1.414 | 1.211 |
| stows over the bar of 2550 | 96 | 209 | 53 | 148 |
| cycles under the bar at this stow count | -78 | 488 | -298 | 241 |
| env steps | 442059 | 330413 | 474942 | 376289 |
| rakes | 479 | 485 | 463 | 507 |
| wrong-box parts | 189 | 139 | 147 | 122 |
| episodes on the cycle allowance | 15 of 60 | 0 of 60 | 27 of 60 | 1 of 60 |
| outcome | the cycles bar missed | both bars met | the cycles bar missed | both bars met |

A negative count of cycles under the bar is the count over it. A gains 113 correct stows over
candidate 3 and uses 0.206 fewer cycles per correct stow. B gains 95 over candidate 1b and uses
0.204 fewer. Both gains are well past the no-harm margins. The design gives a standard deviation of
0.023 cycles per correct stow for 3000 parts (MS7d design section 3.3). A sits 0.177 under the bar,
more than 7 standard deviations. B sits 0.089 under it, near 4.

A has the lower cycles per correct stow, so A is the better passing checked read and takes the
official read (18:01).

### 7.2 Per draw and per class

Per draw, 250 parts each:

| read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate 3 | 229 | 221 | 224 | 229 | 210 | 204 | 217 | 220 | 227 | 213 | 226 | 226 |
| candidate A | 230 | 229 | 227 | 223 | 233 | 231 | 233 | 233 | 227 | 228 | 232 | 233 |
| candidate 1b | 229 | 220 | 220 | 227 | 198 | 204 | 217 | 218 | 218 | 202 | 225 | 225 |
| candidate B | 226 | 224 | 227 | 217 | 232 | 219 | 223 | 222 | 222 | 235 | 225 | 226 |

A stows at least 223 on every draw. Candidate 3 read 204 and 210 on draws 5 and 4.

Per class, as stowed parts and fractions:

| class | parts | candidate 3 | candidate A | candidate 1b | candidate B |
| --- | --- | --- | --- | --- | --- |
| slope | 910 | 733 (0.805) | 791 (0.869) | 718 (0.789) | 768 (0.844) |
| tall | 900 | 815 (0.906) | 842 (0.936) | 817 (0.908) | 840 (0.933) |
| narrow | 415 | 380 (0.916) | 386 (0.930) | 360 (0.867) | 366 (0.882) |
| wide | 775 | 718 (0.926) | 740 (0.955) | 708 (0.914) | 724 (0.934) |

Every class gains on both checkpoints. The slope class gains the most, 58 parts on A and 50 on B.

### 7.3 Where the cycles went

The census class of every cycle, against the MS7c pair:

| census class | candidate 3 | candidate A | change | candidate 1b | candidate B | change |
| --- | --- | --- | --- | --- | --- | --- |
| all cycles | 3517 | 3098 | -419 | 3681 | 3266 | -415 |
| stowed | 2416 | 2602 | +186 | 2244 | 2438 | +194 |
| close_miss | 524 | 171 | -353 | 525 | 181 | -344 |
| held_no_lift | 253 | 144 | -109 | 366 | 171 | -195 |
| lifted_lost | 178 | 90 | -88 | 405 | 362 | -43 |
| wrong_box | 87 | 46 | -41 | 77 | 51 | -26 |
| carrying | 40 | 34 | -6 | 45 | 51 | +6 |
| no_close | 19 | 11 | -8 | 17 | 12 | -5 |
| no_open | 0 | 0 | 0 | 2 | 0 | -2 |
| failed cycles, all but stowed | 1101 | 496 | -605 | 1437 | 828 | -609 |

The census of the cycles that end on the step budget:

| census class | candidate 3 | candidate A | candidate 1b | candidate B |
| --- | --- | --- | --- | --- |
| budget cycles | 874 | 359 | 1141 | 533 |
| close_miss | 460 | 140 | 458 | 143 |
| held_no_lift | 233 | 128 | 336 | 144 |
| lifted_lost | 107 | 47 | 272 | 178 |
| carrying | 40 | 34 | 45 | 51 |
| no_close | 19 | 9 | 16 | 12 |
| wrong_box | 15 | 1 | 12 | 5 |
| no_open | 0 | 0 | 2 | 0 |

The cycle ends:

| read | placed | budget | dropped | stowed |
| --- | --- | --- | --- | --- |
| candidate 3 | 2570 | 874 | 72 | 1 |
| candidate A | 2683 | 359 | 55 | 1 |
| candidate 1b | 2356 | 1141 | 184 | 0 |
| candidate B | 2543 | 533 | 190 | 0 |

The costs that the MS7c diagnosis named, on each pair:

| cost | candidate 3 | candidate A | candidate 1b | candidate B | the levers that act on it |
| --- | --- | --- | --- | --- | --- |
| in-box re-picks | 92 | 0 | 106 | 2 | L1 |
| false holds, to the budget | 232, 170 | 74, 43 | 262, 198 | 81, 47 | L6 most, then L9 |
| parts over the failure cap, their failed cycles | 19, 96 | 1, 4 | 46, 211 | 12, 50 | L1, L3 |
| wrong-box parts | 189 | 139 | 147 | 122 | L5 |
| `lifted_lost` cycles to the budget | 107 | 47 | 272 | 178 | L1 and L2 |
| episodes on the cycle allowance | 15 | 0 | 27 | 1 | all |

On A, 605 fewer cycles fail than on candidate 3. `close_miss` makes 353 of them, 58 %. The isolation
rows show which levers act on each class. Against row R1 on the isolation seed:

- `close_miss`, 427 on row R1: row R6 (L4) reads 276, row R5 (L3) 308, row R4 (L7) 358, row R8 (L9)
  361, row R3 (L6) 386 and row R2 (L8) 406. The full stack reads 189.
- `held_no_lift`, 242 on row R1: row R6 (L4) reads 156, row R5 (L3) 179 and row R2 (L8) 222. Rows
  R3, R4, R7 and R8 read 243 to 258. The full stack reads 104.
- `lifted_lost`, 197 on row R0: row R1 (L1 and L2) reads 155. The full stack reads 121.
- `wrong_box`, 90 on row R1: row R7 (L5) reads 51. The full stack reads 55.

So the lateral position error at the grasp, which the MS7c report named as its first cost, is the
largest cost that MS7d removes. L3 removes the bias, and L4 fixes the position at the final
approach. The class errors are the second MS7c cost. L5 takes the `wrong_box` class from 87 to 46 on
A, but A still puts 139 parts in a wrong box. MS7c's paired read on true inputs put 75 there.

B keeps more `lifted_lost` cycles than A: 362 against 90, and 178 of them end on the budget. B also
ends 190 cycles `dropped`, against 55 on A. In MS7c, most of candidate 1b's `lifted_lost` budget
cycles were slow releases. In those, the r2 A2 policy opened the jaws less than 0.8 of the full gap.
The ledger gives no reason for B's count under L2.

### 7.4 The lever counts

| count | candidate 3 | candidate A | candidate 1b | candidate B |
| --- | --- | --- | --- | --- |
| selections | 3517 | 3098 | 3681 | 3266 |
| wrong overhead class | 134 (0.038) | 111 (0.036) | 143 (0.039) | 112 (0.034) |
| lifts classified | 2650 | 2751 | 2717 | 2881 |
| wrong wrist class drawn | 69 (0.026) | 77 (0.028) | 64 (0.024) | 85 (0.030) |
| third views, votes that changed the class, wrong class taken after the vote | | 153, 41, 38 | | 160, 50, 41 |
| empty lifts | 163 | 72 | 165 | 93 |
| look-again waits, steps, waits that found a target or a rake | 50, 959, 45 | 74, 2937, 68 | 60, 1166, 54 | 71, 6021, 57 |
| stop flag clears (rearms) | | 511 | | 830 |
| rakes off a box wall, grasps refused by a box wall | | 9, 2 | | 31, 29 |
| marker looks, mean estimate error per selection | | 6579, 0.33 mm | | 9854, 0.32 mm |
| wrist looks, misses, covered | | 4045, 196, 8 | | 4398, 204, 6 |
| mean serving-row move and turn per look that found the target | | 5.4 mm, 6.1 degrees | | 11.5 mm, 6.2 degrees |
| steps capped | | 3573 | | 5871 |
| hold entries refused, cycles with one | | 412, 233 | | 596, 350 |
| approach force per draw: p50, p90, max | | 0.33 to 0.36 N, 0.80 to 1.83 N, 19.2 to 54.8 N | | 0.34 to 0.36 N, 0.93 to 2.31 N, 32.2 to 98.9 N |

The wrist look misses on 0.048 of the looks on A, near the nominal 0.05. The marker estimate carries
a mean lateral error of 0.33 mm into each selection, against a bias sigma of 2 mm per axis. The vote
leaves 38 wrong classes of 2751 lifts on A, 0.014.

## 8. The official read

The 16:39 entry pre-registered the official read with the checked reads. The better passing checked
read, the one with the lower cycles per correct stow, takes it. A reads 1.123 and B 1.211, so A
takes it (18:01). The official read is one shot. The 18:01 entry makes it wait for the last
verdicts, rows R2 and R6. If every lever is adopted, it reads A with all nine switches. If not, A
and B read again with the adopted set first.

Both verdicts adopt their levers, L8 at 20:54 and L4 at 21:09. So A reads with all nine switches,
and no checked read runs again. The 21:09 entry pre-registered the official read with its TAG, SHA
and lanes, and it launched at 21:10. It passes at 5100 correct stows or more and at most 1.3 cycles
per correct stow. A READ_FAIL runs once more in a fresh TAG with the same settings.

| item | value |
| --- | --- |
| TAG | d_A_official |
| checkpoint | train_P 200k |
| commit | cdcfb65, worktree `maker-arm-lab-ms7h` |
| switches | all nine MS7d switches, nominal knobs, the rule R3 protocol |
| seed and draws | 20262100, draws 0 to 23, 6000 parts, 120 episodes |
| lanes | `a_h:12` and `b_h:12`, `PORT_BASE` 9900 |
| read at | launched 21:10, ended 21:54 |
| correct stows of 6000 | 5521, 0.920 |
| Wilson 95 % | 0.913 to 0.927 |
| cycles | 6199 |
| cycles per correct stow | 1.123 |
| in-box re-picks, false holds | 1 (`target_in_box` 0), 135 (63 to the budget) |
| wrong-box parts | 283 |
| episodes on the cycle allowance | 0 of 120 |
| outcome | GATE_OK, both bars met, PASS |

Per draw, 250 parts each:

| draw | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stowed | 234 | 226 | 230 | 214 | 231 | 236 | 232 | 227 | 229 | 239 | 237 | 224 |
| draw | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| stowed | 231 | 238 | 232 | 232 | 233 | 227 | 226 | 226 | 227 | 228 | 233 | 229 |

The official read passes both bars with a wide margin. It stows 421 parts over the bar of 5100. It
uses 978 fewer cycles than the bar allows at its stow count. The design gives a standard deviation
of 0.016 cycles per correct stow for 6000 parts (MS7d design section 3.3). So the read sits about 11
standard deviations under the bar. The lowest draw, draw 3, stows 214 of 250.

The official read repeats the checked read of A on a fresh seed:

| item | checked read of A | official read |
| --- | --- | --- |
| seed and draws | 20261700, draws 0 to 11 | 20262100, draws 0 to 23 |
| correct stows | 2759 of 3000, 0.920 | 5521 of 6000, 0.920 |
| cycles per correct stow | 1.123 | 1.123 |
| in-box re-picks | 0 | 1 |
| false holds, to the budget | 74, 43 | 135, 63 |
| parts over the cap, their failed cycles | 1, 4 | 4, 17 |
| wrong-box parts | 139 | 283 |
| placed, budget, dropped, stowed ends | 2683, 359, 55, 1 | 5366, 709, 121, 3 |
| `close_miss`, `held_no_lift`, `lifted_lost`, `wrong_box`, `carrying` cycles | 171, 144, 90, 46, 34 | 365, 217, 230, 106, 76 |
| hold precision and recall, lift precision and recall | 0.968, 0.990, 0.980, 0.999 | 0.968, 0.993, 0.980, 0.998 |
| episodes on the cycle allowance | 0 of 60 | 0 of 120 |

Per class, the official read stows 1608 of 1820 slope parts (0.884) and 1593 of 1705 tall parts
(0.934). It stows 751 of 820 narrow parts (0.916) and 1569 of 1655 wide parts (0.948). Its lever
counts: the stop flag cleared 972 times, and the loop took 8079 wrist looks with 380 misses and 18
covered targets. It took 11482 marker looks and drew 298 third views, and the vote changed the class
92 times. The cap held the arm on 7905 steps, and the gate refused 789 hold entries on 489 cycles.
The rake swept 16 parts off a box wall.

## 9. The requirement table v2

For each knob, the table gives the largest scale at which the stowed fraction stays at 0.85 or more
(rule R7). It also gives the largest scale at which the cycles stay at 1.3 per correct stow or
fewer. Both interpolate linearly between the measured scales. A knob whose scale-2 read misses a bar
is tight. A 500-part read has a Wilson 95 % interval of about plus or minus 0.03.

The table reads candidate A with all nine switches at cdcfb65, on seed 20261700, draws 0 and 1, 500
parts. Its TAGs start with `d_rt_`, and `d/bin/sweep_d.sh` reads each row on 2 lanes. The 18:01
entry gives Machine A the first 18 rows, 4 at a time, and Machine B the last 9, 2 at a time. The 19:33
entry keeps the rows on the GPU hosts. The rows read from 18:01 to 21:04, and the 21:05 entry
records the table (`~/data/maker-arm-ms7/d/req_table_v2.md`). The table stands on candidate A either
way: A is the official candidate, and A is also the best checked read (21:09).

| row | TAG | stowed of 500 [Wilson 95 %], cycles per correct stow |
| --- | --- | --- |
| nominal, every knob at 1 | d_A_chk, draws 0 and 1 | 459/500 = 0.918 [0.891, 0.939], 1.096 |
| structure only, every knob at 0 | d_rt_all0 | 471/500 = 0.942 [0.918, 0.959], 1.074 |

The scale-1 cell of every knob is the nominal row. The MS7c table read r2 A2 100k. Its nominal row
on the same draws read 449/500 = 0.898 at 1.292, only 0.008 under the cycles bar. Candidate 3, the
MS7c pair of A, read 450/500 at 1.280 on these draws. With every knob at 0, the full stack stows 12
more parts than at nominal and uses 0.022 fewer cycles per correct stow.

Each cell gives k/n, the fraction, the Wilson 95 % interval and the cycles per correct stow. A limit
of "4 or more" means that the bar still holds at scale 4, the top measured scale.

| knob | nominal | scale 2 | scale 4 | largest scale at 0.85 stowed, in physical units | largest scale at 1.3 cycles, in physical units | tight |
| --- | --- | --- | --- | --- | --- | --- |
| `pos` | bias sigma 2 mm per axis, noise sigma 1.5 mm in xy and 2 mm in z | 469/500 = 0.938 [0.913, 0.956], 1.143 | 439/500 = 0.878 [0.846, 0.904], 1.392 | 4 or more: bias sigma 8 mm per axis, noise sigma 6 mm in xy and 8 mm in z | 3.26: bias sigma 6.5 mm per axis, noise sigma 4.9 mm in xy and 6.5 mm in z | no |
| `heading` | sigma 4 degrees per detection | 454/500 = 0.908 [0.879, 0.930], 1.110 | 468/500 = 0.936 [0.911, 0.954], 1.139 | 4 or more: sigma 16 degrees | 4 or more: sigma 16 degrees | no |
| `size` | sigma 1 mm per detection | 465/500 = 0.930 [0.904, 0.949], 1.101 | 452/500 = 0.904 [0.875, 0.927], 1.175 | 4 or more: sigma 4 mm | 4 or more: sigma 4 mm | no |
| `box` | sigma 3 mm and 2 degrees per box | 457/500 = 0.914 [0.886, 0.936], 1.092 | 462/500 = 0.924 [0.897, 0.944], 1.115 | 4 or more: sigma 12 mm and 8 degrees | 4 or more: sigma 12 mm and 8 degrees | no |
| `miss` | probability 0.05 per perceive call | 456/500 = 0.912 [0.884, 0.934], 1.107 | 453/500 = 0.906 [0.877, 0.929], 1.128 | 4 or more: probability 0.20 | 4 or more: probability 0.20 | no |
| `class` | probability 0.03 per classification | 448/500 = 0.896 [0.866, 0.920], 1.163 | 444/500 = 0.888 [0.857, 0.913], 1.209 | 4 or more: probability 0.12 | 4 or more: probability 0.12 | no |
| `tool` | sigma 1 mm per axis per marker look | 466/500 = 0.932 [0.906, 0.951], 1.086 | 446/500 = 0.892 [0.862, 0.916], 1.155 | 4 or more: sigma 4 mm per axis | 4 or more: sigma 4 mm per axis | no |
| `wrist` | sigma 0.5 mm in xy and 2 degrees in yaw per wrist look | 456/500 = 0.912 [0.884, 0.934], 1.136 | 455/500 = 0.910 [0.882, 0.932], 1.174 | 4 or more: sigma 2 mm in xy and 8 degrees in yaw | 4 or more: sigma 2 mm in xy and 8 degrees in yaw | no |
| `wrist_bias` | sigma 0.5 mm per axis per episode | 465/500 = 0.930 [0.904, 0.949], 1.144 | 459/500 = 0.918 [0.891, 0.939], 1.153 | 4 or more: sigma 2 mm per axis | 4 or more: sigma 2 mm per axis | no |
| `wrist_miss` | probability 0.05 per wrist look | 446/500 = 0.892 [0.862, 0.916], 1.128 | 460/500 = 0.920 [0.893, 0.941], 1.104 | 4 or more: probability 0.20 | 4 or more: probability 0.20 | no |
| `torque` | sigma 0.05 Nm per joint per step | 466/500 = 0.932 [0.906, 0.951], 1.124 | 458/500 = 0.916 [0.888, 0.937], 1.109 | 4 or more: sigma 0.2 Nm | 4 or more: sigma 0.2 Nm | no |
| `torque_k` | k 0.05, the relative model error per joint per episode | 461/500 = 0.922 [0.895, 0.942], 1.174 | 457/500 = 0.914 [0.886, 0.936], 1.112 | 4 or more: k 0.2 | 4 or more: k 0.2 | no |

The `rho` knob is the persistent share of the class error, 0.5 times the scale, capped at 1. Its
rows are scales 0, 1 and 2. Scale 2 is a class error that is fully persistent.

| knob | scale 0, share 0 | scale 1, share 0.5 (nominal) | scale 2, share 1 | largest scale at 0.85 stowed | largest scale at 1.3 cycles | tight |
| --- | --- | --- | --- | --- | --- | --- |
| `rho` | 464/500 = 0.928 [0.902, 0.948], 1.082 | 459/500 = 0.918 [0.891, 0.939], 1.096 | 455/500 = 0.910 [0.882, 0.932], 1.105 | 2 or more: a persistent share of 1 | 2 or more: a persistent share of 1 | no |

Notes on the table:

- No knob is tight. Every scale-2 read meets both bars. `pos` is the only knob that misses a bar at
  scale 4. There it still stows 0.878, but it uses 1.392 cycles per correct stow, so its cycles
  limit interpolates to 3.26. Every other knob keeps both bars at scale 4, and `rho` keeps both at a
  share of 1. So even with a class error that repeats in every view of a part, the vote of L5 keeps
  the bars.
- MS7c marked `pos` and `class` tight on r2 A2 100k. `pos` kept the stow bar to 1.25 and the cycles
  bar to 1.02, and `class` to 1.83 and 1.09. Every MS7c knob except `miss` reached 1.3 cycles at a
  scale of 1.02 to 1.11, because the nominal row sat only 0.008 under the bar. The MS7d nominal row
  sits 0.204 under it. So every tolerance is wider: `pos` goes to 4 or more on stows and to 3.26 on
  cycles, and `class` to 4 or more on both. The two tables read different checkpoints, so the
  comparison holds the levers and the checkpoint together. Candidate 3, the same checkpoint as A
  with no MS7d lever, read 450/500 at 1.280 on these draws, near the MS7c nominal row.
- A 500-part row carries about 0.04 cycles per correct stow of read noise (21:05). `pos` at 2 reads
  469, 10 more than the nominal 459. Some knobs read better at scale 4 than at scale 2: `heading`
  468 against 454, `wrist_miss` 460 against 446, and `torque_k` 1.112 against 1.174 cycles. Each
  such pair lies inside that noise.
- The table moves one knob at a time, with the others at 1. MS7c found that the knobs share one
  cycles budget, so two knobs near their limits at once can miss a bar that each one alone keeps.
  The `pos` knob scales the bias and the per-detection noise together. Under the full stack, L3
  estimates the bias and L4 fixes the position at the final approach. The table does not split the
  two.

## 10. The hand-off to MS5a

The MS7d reads rest on inputs that MS7c did not have. MS7d passes its gate with every stand-in knob
at its nominal, and the stand-in values are optimistic until MS5a measures them. Each input is a
stand-in with an `INITIAL:` nominal. The first MS5a tasks measure them on the real rig, against the
limits of the table v2. Each limit holds with the other knobs at nominal, and a limit of "or more"
means the bar still held at the top measured scale:

- The gripper marker (L3). A fiducial on the gripper that the iPhone frame locates, knob `tool`.
  Both bars hold at a noise sigma of 4 mm or more per axis per look. The real camera error is not a
  pure translation, so MS5a also measures how far from the gripper the estimate holds.
- The wrist camera (L4). Its position error, hand-eye bias and miss rate on real parts, knobs
  `wrist`, `wrist_bias` and `wrist_miss`. Both bars hold at a look noise of 2 mm or more in xy and 8
  degrees in yaw. They also hold at a hand-eye bias of 2 mm or more per axis, and at a miss rate of
  0.20 or more.
- The class error on real parts and its persistence across views, knobs `class` and `rho`, with the
  MS7b classifier. Both bars hold at a class error of 0.12 or more per classification. They also
  hold at a class error that repeats in every view of a part.
- The torque noise and the model error of the real arm, knobs `torque` and `torque_k`, from the
  spike's free-motion logs and pushes against a scale. Both bars hold at a noise sigma of 0.2 Nm or
  more per joint per step. They also hold at a model error k of 0.2 or more.
- The force bars. `HOLD_FORCE_MAX_N` = 5 N passed the calibration in simulation. `FORCE_CAP_N` = 15
  N costs nothing in simulation. MS5a sets the cap on the real arm.
- The overhead camera knobs of MS7c: `pos`, `heading`, `size`, `box` and `miss`. `pos` is the
  closest to a bar. The stow bar holds at a bias sigma of 8 mm or more per axis with a noise sigma
  of 6 mm in xy. The cycles bar holds up to a bias sigma of 6.5 mm per axis with a noise sigma of
  4.9 mm in xy. The nominal is a bias sigma of 2 mm per axis and a noise sigma of 1.5 mm in xy.
- The gripper transmission from motor angle to jaw gap. The hold detector and L2 read the jaw gap,
  and the transmission still has no measured value.

The two loop constants of the base, `BOX_FILTER_MARGIN_M` = 5 mm and `RELEASE_OPEN_M` = 10 mm, read
only as a pair in row R1. No read varies either one.

## 11. The hosts and the host lessons

### 11.1 The hosts

The hosts keep their MS7c roles. `scripts/fleet.py` and the host table stay.

| host | MS7d work |
| --- | --- |
| Machine A, RTX 5090 | The controller session and the ledger. The hold calibration on CPU. 8 GPU lanes of rows R0, R8, R4, R7, R3, d_R9_all, d_A_chk, d_B_chk and d_R2g. The first 18 rows of the requirement table. 12 lanes of the official read. |
| Machine B, RTX 5080 | 4 GPU lanes of the same reads, 2 lanes of row R1, 4 lanes of row R6 and the single lane of d_R6_d11. The last 9 rows of the requirement table. 12 lanes of the official read. |
| Machine C | CPU lanes: 10 of row R1, 12 of row R5, 12 of d_R2 and 6 of d_R6, then 6 of d_R6_d5_10. |

No retrain ran, so Machine B trained nothing, and Machine C recorded no corpus.

### 11.2 The worktree aliases

The design runs each host from one campaign worktree, `~/src/maker-arm-lab-ms7`. MS7d read at eight
SHAs on the same three hosts. So from 11:57 to 12:27, `hosts.json` gained aliases `<host>_b` to
`<host>_h`. Each alias names the same host and its own worktree, `~/src/maker-arm-lab-ms7<x>`,
synced to one SHA. The MS7c table stays as `hosts.json.bak-ms7c`. `read_d.sh` takes the worktree
name, so the `fleet.py` of that worktree parses the flags of its own SHA.

| worktree | SHA | reads |
| --- | --- | --- |
| `maker-arm-lab-ms7` | 7e4173c | rows R0 and R1 |
| `maker-arm-lab-ms7b` | de83232 | row R5 |
| `maker-arm-lab-ms7c` | c090ecc | d_R2, d_R2g |
| `maker-arm-lab-ms7d` | 9b896b5 | row R8 |
| `maker-arm-lab-ms7e` | 7f8985e | the hold calibration, rows R3 and R4 |
| `maker-arm-lab-ms7f` | 0fdc1cc | d_R6, d_R6_d11, d_R6_d5_10 |
| `maker-arm-lab-ms7g` | f011e8d | row R7 |
| `maker-arm-lab-ms7h` | cdcfb65 | d_R9_all, d_A_chk, d_B_chk, the requirement table rows, d_A_official |

Each read also has its own `PORT_BASE`, from 8000 for row R0 upward, so each lane has its own policy
server.

### 11.3 The lessons

The density of Machine C was the largest host lesson. At 14:27, rows R1, R5, R2 and R6 ran
40 lanes there, 40 EGL sim processes on its one A2000, at load 63 to 70. MS7c ran 12 lanes there.
The payloads give a measured contrast, as minutes of wall time a draw:

| host and lanes | minutes a draw | reads |
| --- | --- | --- |
| Machine C, 40 CPU lanes at once | 240 to 360 (4.0 to 6.0 h). d_R2 and d_R6 wrote no payload in 7 h. | rows R1 and R5 |
| Machine C, 6 CPU lanes alone | 76 to 93, about 90 | d_R6_d5_10 |
| Machine A, 16 GPU lanes (8 of d_R2g beside 8 of the sweep rows) | 46 to 62 | d_R2g |
| Machine A 8 or Machine B 4 GPU lanes | 25 to 54, about 40 | rows R0, R3, R4, R7, R8 and R9, the checked reads |
| Machine A 12 and Machine B 12 GPU lanes | 25 to 44 | the official read, 24 draws from 21:10 to 21:54 |

The lane changes of 13:40 and 14:27 moved every later row to the GPU hosts. The entry of 15:31 sent
no new read to Machine C. At 40 lanes, a CPU lane there is too slow for a read that the
gate waits on. At 6 lanes, it read draws 5 to 10 of row R6 in about 90 minutes a draw.

The other lessons:

- The worktree aliases worked. Reads at eight SHAs ran on three hosts, and no read waited for a
  re-sync of a host.
- The record gap. The controller session launched d_R6_d11 from its own background task and wrote no
  entry. The ledger found it at 17:32 and wrote its rule before its number. Every launch needs its
  ledger entry first, a helper read of one draw too.
- A check goes into the ledger only after it runs. The 21:05 entry on the requirement table was
  written before the full table was read on screen. A first note then said that its numbers were
  checked, but that note too came before a readable view of the table. A second note says so. The
  check then ran on the full table, and every number matched. No number was wrong, but two entries
  named a check before it ran.
- A fleet read that ends before its lanes do does not pull their lane dirs. The Machine B lane dir of
  d_R6 was pulled by hand.
- A host sync with `--force` came only after a check that no process had its cwd in any campaign
  worktree. The stopped lanes of the MS7c sweep read RUNNING forever, so a plain check of the lane
  status was not enough.
- The supervisor cache refuses a payload with other knobs. It refused the MS7c payloads on the MS7d
  code, so no stale payload passed as a read.

## 12. What is still open

- The report PR #128, its review, and the merge of stack #127.
- A lane density rule for Machine C. At 40 EGL processes on its one GPU, a draw took 4 to 6
  hours, and at 6 lanes about 90 minutes. Each worktree alias in `hosts.json` has its own lane
  count, so the host table does not cap the lanes of the host as a whole.
- `qfrc_constraint` also carries the forces of the joint limits (12:25). A joint at its stop reads
  as a force at the pick point. No read measured how often that happens.
- With the cap on, the approach force still peaks at 19.2 to 54.8 N per draw on the checked read of
  A, and at 18.7 to 121.9 N on the official read. The cap holds the arm command still. It does not
  bound the peak force.
- The hold offset has no margin. It reads 5.0 steps on the teacher against a bar of 5, and 6 steps
  on the policy's cycles.
- A still puts 139 of 3000 parts in a wrong box, and 283 of 6000 on the official read. MS7c's paired
  read on true inputs put 75 of 3000 there.
- B keeps 362 `lifted_lost` cycles and 190 `dropped` ends under L2. The ledger gives no reason.
- `BOX_FILTER_MARGIN_M` and `RELEASE_OPEN_M` read only as a pair in row R1.
- Every new input is a truth-plus-noise stand-in, and its nominal is an `INITIAL:` estimate. The
  stand-ins can be optimistic, so a pass in simulation does not promise a pass on the real rig. MS5a
  measures every knob on the real rig, and the requirement table v2 gives the margins to the bars.

## 13. Conclusion

MS7d passes its gate under the perceived state at nominal noise. Both checked reads meet both bars,
A at 2759 of 3000 and 1.123 cycles per correct stow, B at 2698 and 1.211. The official read of A on
the fresh seed 20262100 stows 5521 of 6000, 0.920, at 1.123. The bars are 5100 and 1.3. Every
stand-in knob is at its nominal, and those values are optimistic until MS5a measures them.

MS7d delivers these results:

- Nine levers, each behind its own switch and off by default. A CPU test shows that no decision
  reads the truth with all nine on.
- Two loop fixes, L1 and L2. L1 removes the in-box re-picks, 67 to 1 on the isolation seed and 92 to
  0 on the checked seed.
- A force gate on the hold that meets the calibration bar at 5 N, and a force cap that costs
  nothing.
- Isolation rows that measure each lever on 3000 parts. No lever reads worse than the base, so all
  nine are adopted. The wrist look, L4, is the largest single lever, and the marker bias estimate,
  L3, the second. L8, L7, L5 and L9 each gain 20 to 23 stows, and L6 cuts the false holds most.
- Two checked reads that pass both bars with a margin: A at 2759 and 1.123, B at 2698 and 1.211.
  Against the same heaps and noise draws in MS7c, each gains about 100 correct stows and uses about
  0.2 fewer cycles per correct stow.
- The requirement table v2. No knob is tight, and every tolerance is wider than in MS7c. `pos` is
  the closest to a bar: it keeps the cycles bar up to scale 3.26.

The reads give these findings for MS5a:

- The lateral position error at the grasp was the largest MS7c cost. A gripper marker and a wrist
  look remove most of it in simulation. MS5a builds and measures these two inputs first.
- The class error is the largest cost that remains. A still puts 139 parts in a wrong box, against
  75 on true inputs in MS7c. The table holds both bars at a class error of 0.12 per classification.
  It also holds them at a class error that repeats in every view of a part.
- The arm's own torques give a usable force at the pick point in simulation. MS5a measures the
  torque noise and the model error on the real arm, and sets the force cap there.
