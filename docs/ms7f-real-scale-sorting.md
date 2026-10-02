# MS7f: real-scale sorting in simulation

Date: 2026-10-02. Status: draft. The campaign still runs. This draft uses the ledger up to its entry
of 10:00 on 2026-10-02. A line that starts with `TBD:` waits for a read or a decision that had no
number at that time. The controller fills those lines.

The design of MS7f is the document that the owner reviewed and approved in chat on 2026-09-30. This
report cites it as "MS7f design section N" and restates every rule that it uses. The campaign ledger
is `~/data/maker-arm-ms7/f/README.md`, and `STATE.md` beside it is the resume pointer. Every rule
below went into that ledger before its number. The measurement tables are under
`~/data/maker-arm-ms7/f/tables/`. Each read has a TAG, its name. The read payloads are under
`~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7f TAG starts with `f_`.

The ledger names its rules R1 to R15 and its deviations D1 to D20. The design names its decisions F1
to F10. The tracks are H (the heap), C (the classifier), P (the part families) and S (the speed).
The retrains are F1, F2, F3 and the pre-registered F4, and their checkpoints are written as "F1
100k". The ledger calls retrain F3 "retrain 3" and retrain F4 "retrain 4". This report writes "rule
R4", "decision F6" and "retrain F1" to keep them apart.

All times are PDT. Each time is the time of the ledger entry that records the event. Times from
10:13 to 23:47 are on 2026-09-30. Times after the first midnight, up to the entry of 23:26, are on
2026-10-01. The entries after the second midnight, from 00:52 to 10:00, are on 2026-10-02.

Numbers come from the ledger or from the files that it names. This report also computes some values
from the ledger's counts: fractions, rates an hour and Wilson intervals that the ledger does not
give. The text marks each of them.

## 0. Summary

MS7e met its gate. Its official read stowed 5299 of 6000 held-out parts in their own class's box,
0.883, at 1.271 cycles per correct stow. A correct stow took 21.5 simulated seconds, 167 an hour.
MS7f makes the task real-scale along three axes (MS7f design section 1):

1. Continuous operation and speed. The design's bar is at least 300 correct stows per simulated
   hour.
2. A dense large heap. An episode holds 150 parts, against 50 in MS7e.
3. A large catalog sorted by category. Two new families, joint and minifig, join the catalog. The
   parts go to six categories, and a classifier trained on rendered images replaces the class
   stand-in.

The gate has two rows since deviation D12 (10:38). The owner decided on staged bars at 10:35, after
the screens and with 2 of the 20 heaps of candidate 1's checked read visible. The design's bars stay
as a stretch row in every verdict:

| row | correct in a box of the category | correct stows an hour | held-out row |
| --- | --- | --- | --- |
| staged bars, the gate since D12 | at least 0.85 (2550 of 3000, 5100 of 6000) | at least 200 | at least 0.80 |
| stretch row, the design's bars | at least 0.92 (2760 of 3000, 5520 of 6000) | at least 300 | at least 0.85 |

The results so far:

- The base branch adds the six categories, the category map onto the boxes, the throughput, the
  held-out row, the four cycle intervals of S0 and heaps of up to 150 parts.
- H0 passes on the additive model. The official read on the deep tray projects to 5.17 h against
  the 8 h bar.
- The deep tray (H1) holds both bars of rule R4 against the refill (H1'), and decision F6 adopts
  it. H1 stows 842 of 900 at 18.42 s per correct stow, and H1' stows 857 of 900 at 19.10 s.
- Track P adopts both new families. Joint keeps 8 kinds. Minifig keeps 7 after deviation D3 drops
  the cowboy hat. The catalog holds 66 kinds.
- Track C misses its C1 bars. The serving model v5 reads 0.896 at the lift against 0.97, special
  0.819 against 0.90, and finds 0.44 of the passengers against 0.8. Its three-view serving rule
  reads 0.928. Model v6a, a ResNet50, passes the pre-registered v6 bar at 0.939 and replaces v5
  from candidate 1's checked read on.
- Track S adopts no pace rung. Three teacher trims pass on both architectures. The wave recipe M5
  cuts the recorder from 16.68 s to 11.24 s per delivered part on 50-part heaps, at a cost of
  0.012 of its share.
- The teacher reference read misses both bars of rule R9. It stows 641 of 750 at 11.27 s per
  correct stow on five of its six heaps. Its census traces most of the miss to `--book_miss`, and
  that switch leaves every teacher read.
- Wave 1 stopped once on two blockers. It then recorded 40 heaps of 150 at 4464b79 and delivered
  5216 of 6000, 0.869. Wave 2 delivered 5155 of 6000, 0.859.
- Retrain F1 trained 200k steps from train_E2 150k on wave 1. Its four screens read 353 to 425 of
  600 correct stows and 96.2 to 142.5 correct stows an hour. Candidate 1 is F1 100k.
- The serving warp `--action_warp 1.5` (D7) raises candidate 1's screen to 145.5 correct stows an
  hour at 428 of 600. X = 1.25 and X = 2.0 are out. `--passenger_drop` (D9) is not adopted. Model
  v6a reads 420 of 600 in the loop, against v5's 425.
- Candidate 1's checked read stows 2043 of 3000, 0.681, at 140.9 correct stows an hour, with a
  held-out row of 0.600. It fails all three staged bars and the stretch row. Rule R11's trigger
  fires, so candidate 2 gets a checked read.
- Retrain F2 trained 200k steps from F1 100k on waves 1 and 2. On paired x86 screens, F2 100k reads
  435 of 600 against F1 100k's 420. The gain is under the 18 stows that D11 asks, so no third wave
  of the same recipe records.
- The control read, E2 with no retrain, stows 768 of 1050, 0.731, at 151.3 correct stows an hour.
  That is the level of candidate 1's screen, so retrain F1 bought no visible gain.
- D13 picks candidate 2 on four x86 screens: F2 150k, with 438 of 600 against F2 100k's 435.
- Two loop levers pass their D14 screens on candidate 1. `--missed_lift` reads 418 of 600 at 210.5
  an hour against the base's 409 at 137.0. `--lift_view_steps 6` passes alone at 431 of 600. The
  pair reads 203.5 an hour, under 0.97 times 210.5, so only `--missed_lift` serves.
- Candidate 2's checked read, F2 150k with `--action_warp 1.5 --missed_lift`, stows 2065 of 3000,
  0.688, at 189.1 correct stows an hour, with a held-out row of 0.647. It fails all three staged
  bars and the stretch row. Against candidate 1's checked read it gains 22 correct stows and 48 an
  hour.
- D16 screens two more switches on candidate 2. `--cell_failure_cap 5` is adopted at 435 of 600,
  and `--carry_votes 2` is not, at 415.
- Classifier v7, v6a's recipe plus deep-tray renders, misses every check of its bar, and v6a stays.
  A new deep-tray C1 set shows the scene gap: v6a reads 0.897 there against 0.939 in the normal
  tray, and special 0.691 against 0.883.
- Candidate 3 is candidate 2 with `--cell_failure_cap 5`. Its checked read stows 2181 of 3000,
  0.727, at 189.5 correct stows an hour, with a held-out row of 0.666. It fails all three staged
  bars and the stretch row. It is the best checked read so far: 116 more correct stows than
  candidate 2 at the same rate.
- Deviation D17 (21:34), an owner decision, runs two levers in parallel through 10-03: classifier v8
  on the gate scene, and wave 3, on-policy data for the missed grasps, with retrain F3 on it.
- Deviation D18 (21:46) tests the PGS solver of PR #153 as a wall-time lever. Its paired screen
  reads 449 of 600 at 206.3 an hour against Newton's 435 at 191.9, outside the parity tolerance. So
  PGS is not adopted for reads, and it serves only the v8 renders.
- Classifier v8 misses its bar, like v7. It reads 0.907 on the deep-tray C1 set against v6a's 0.906
  and a bar of 0.936, and v6a stays.
- Wave 3 records 43 deep-tray heaps with 2447 teacher takeovers. The merged corpus f/w123/dataset
  holds 13,665 episodes and 5,110,772 frames, and retrain F3 runs on it.
- The stretch run r = 0 (D15), report-only, serves 1000 of 1000 parts with candidate 2's flags and
  the tray refill. It stows 801 correctly at 259.9 an hour, with 85 interventions per 1000 parts.
- Deviation D19 (06:40), an owner decision, moves the gate scene to the refilled deep tray,
  `--deep_tray --tray_refill`. The bars stay D12's.
- Candidate 3's checked read in the refill scene stows 2297 of 3000, 0.766, at 204.7 correct stows
  an hour, with a held-out row of 0.735. It passes the staged rate bar and fails the share and
  held-out bars.
- Classifier v9e, the mean of the logits of v6a and v8 with nothing trained, passes its
  pre-registered C1 bar at 0.9235 on the deep-tray set. Its paired screen in the refill scene reads
  474 of 600, at the adoption bar exactly, so v9e serves every later read.
- Retrain F3 ends at 200k steps. Its four screens in the refill scene read 477 to 498 of 600 against
  the pair's 468. Candidate 4 is F3 150k, served with v9e.
- Candidate 4's checked read in the refill scene stows 2467 of 3000, 0.822, at 300.0 correct stows
  an hour, with a held-out row of 0.786. It passes the staged rate bar and fails the share bar by 83
  correct stows and the held-out bar by 11 parts. It is the best checked read so far.
- Wave 4, the DAgger lever in the refill scene with candidate 4 driving, records for retrain F4.
  Deviation D20 screens three loop switches on candidate 4, because candidate 4 has rate to spare.
  Neither has a number yet.
- The code branches are draft PRs #154 to #163, #165 and #166 in GitHub stack #164, 12 PRs, and CI
  passes on all of them.

The decomposition of section 10 answers where the parts and the hours go:

- The parts. On candidate 4's checked read, 2467 of 3000 parts stow correctly. 376 end in a wrong
  box, 194 of them from the classifier's `wrong_box` cycles, and 157 end outside every box. The
  staged bar needs 83 more correct stows, and the wrong boxes now hold most of the misses.
- The hours. The stowed cycles hold 0.492 of candidate 4's steps and `close_miss` 0.254, still the
  largest sink. A stowed cycle takes 196 steps, 6.5 s, inside the 12 s of the stretch rate.
- The pace. Candidate 4 spends 888,218 steps, 35 % under the 1,377,000 that 2550 correct stows at
  200 an hour allow. That is also 11 % under the 993,600 of the stretch row. So the share bar binds
  on both rows: 2550 correct stows at the same steps read 310.1 an hour.
- The teacher. The wave teacher delivers a part every 13.91 to 16.23 s, 222 to 259 an hour, over the
  staged 200 and under the stretch 300.

The gate outcome. A screen and the control read have no bar, so their rows show the bars for
comparison only:

| read | checkpoint | commit, classifier, warp | correct stows | correct stows an hour | held-out row | staged bars (0.85, 200, 0.80) | stretch row (0.92, 300, 0.85) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate 1, screen | F1 100k | d6f893d, v5, X = 1.0 | 425 of 600, 0.708 | 131.4 | 99 of 151, 0.656 | no bar, under all three | no bar, under all three |
| candidate 1, warp screen | F1 100k | 03a5cd8, v5, X = 1.5 | 428 of 600, 0.713 | 145.5 | 91 of 151, 0.603 | no bar, under all three | no bar, under all three |
| candidate 1, checked | F1 100k | 03a5cd8, v6a, X = 1.5 | 2043 of 3000, 0.681 | 140.9 | 471 of 785, 0.600 | FAIL on all three | FAIL on all three |
| candidate 2, checked | F2 150k | 496f857, v6a, X = 1.5, `--missed_lift` | 2065 of 3000, 0.688 | 189.1 | 508 of 785, 0.647 | FAIL on all three | FAIL on all three |
| candidate 3, checked | F2 150k | bcbbe02, v6a, X = 1.5, `--missed_lift --cell_failure_cap 5` | 2181 of 3000, 0.727 | 189.5 | 523 of 785, 0.666 | FAIL on all three | FAIL on all three |
| candidate 3, checked, refill scene (D19) | F2 150k | bcbbe02, v6a, X = 1.5, `--missed_lift --cell_failure_cap 5 --tray_refill` | 2297 of 3000, 0.766 | 204.7 | 577 of 785, 0.735 | FAIL on the share and the held-out row, PASS on the rate | FAIL on all three |
| candidate 4, checked, refill scene | F3 150k | bcbbe02, v9e, X = 1.5, `--missed_lift --cell_failure_cap 5 --tray_refill` | 2467 of 3000, 0.822 | 300.0 | 617 of 785, 0.786 | FAIL on the share and the held-out row, PASS on the rate | FAIL on the share and the held-out row. The ledger reads the rate as met, and the payload gives 299.97. |
| candidate 4' or candidate 5, checked, refill scene | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| official | | | | | | runs only on the first checked read in the refill scene that passes the staged bars (D12, D19) | |
| control | train_E2 150k | d6f893d, v5, X = 1.0, aarch64 | 768 of 1050, 0.731 | 151.3 | 180 of 263, 0.684 | no bar, under all three | no bar, under all three |

The outcome on the full pile. No checked read on the full 150-part deep pile passes the staged bars
of D12. So by D12's rule no official read runs in that scene. Candidate 1 stows 2043 of 3000 at
140.9 an hour, candidate 2 2065 at 189.1, and candidate 3 2181 at 189.5, against 2550 and 200. The
held-out rows read 0.600, 0.647 and 0.666 against 0.80. The task itself works at real scale in
simulation: 150-part deep-tray heaps, six categories over 66 kinds, a render-trained classifier and
box emptying all run in every read. The share falls short on three sinks: the classifier, the missed
lifts and the parts left outside every box. The rate falls short because failed cycles hold most of
the steps.

Since deviation D19 (06:40), the gate scene is the refilled deep tray, and the full-pile results
stay in this report as the harder row. The official read runs on the first checked read in the
refill scene that passes the staged bars (section 9.18).

The outcome in the refill scene so far. Candidate 3 stows 2297 of 3000 at 204.7 an hour, and
candidate 4 2467 at 300.0, against 2550 and 200. Their held-out rows read 0.735 and 0.786 against
0.80. Both pass the staged rate bar and fail the share and held-out bars, so no official read runs
yet. Candidate 4 is 83 correct stows and 11 held-out parts short. Two levers are still open: the D20
switches and retrain F4 on wave 4 (sections 9.19 and 9.20).

TBD: the outcome in the refill scene after D20 and wave 4, and the official read's outcome on a pass
of a checked read there.

## 1. The goal, the decisions and the rules

### 1.1 The starting point

The design read the project at main dbc6a8c on 2026-09-30 (MS7f design section 3):

| item | value |
| --- | --- |
| MS7e official read | 5299 of 6000, 0.883, at 1.271 cycles per correct stow. 414 parts (0.069) in a wrong box, 287 (0.048) not stowed. |
| MS7e official pace | 3,417,355 control steps at 30 Hz over 6737 cycles: 16.9 s a cycle, 21.5 s per correct stow, 167 correct stows an hour |
| MS7e teacher pace | e_w1_ref2 on 150 parts: 14.3 s a cycle, 14.8 s per stow, 244 stows an hour |
| the heap | 50 parts in a tray with a 0.13 x 0.20 m interior, 0.02 m walls, a 30 degree lean, centred at (0.195, 0.0) m |
| the boxes | 4 to 8 boxes, interiors 0.07 to 0.10 m, depths 0.03 to 0.05 m. Rule R11 of MS7e fits the class fold at `BOX_VOLUME_PER_PART_VOLUME` = 2.0. |
| the catalog | 51 kinds in 8 classes: 29 base parts, 4 round, 8 LDraw, 10 flat. `parts_seed` 20260915 holds out 13 kinds. |
| the class stand-in | `Perceiver.classify` errs at `CLASS_ERROR_PROBABILITY` = 0.03. `ClassViews` makes half of that error persistent per part. |
| the cameras | wrist and overhead at 224 x 224 px. The overhead pixel covers about 3.7 mm of desk, the wrist pixel about 0.6 mm. |
| the passengers | The teacher lever T2 refuses a lift that carries a second part. The serving loop has no passenger signal. |
| the speed limits | The SDK profile clamps each command at 5.0 rad/s and caps the torque at 4 to 12 N m. `trajectory.TRANSPORT_SPEED` is 4.0. |

So the teacher itself ran under the required 300 an hour, and the policy ran at 167.

### 1.2 The decisions of the design

The owner took these decisions on 2026-09-30, in chat (MS7f design section 2):

| id | decision |
| --- | --- |
| F1 | Real-scale sorting. Assembly, a pixels-only loop and sort rules by instruction stay out. |
| F2 | Three axes: continuous operation and speed, a dense large heap and a large catalog. Parts coupled by their studs and flipped parts stay out. |
| F3 | A staged gate. Required on 150-part heaps from at least 50 kinds: at least 0.92 in a box of the part's category and at least 300 correct stows an hour. Stretch, with no bar: 0.95, 400 an hour, 1000 consecutive parts, at most 5 interventions. |
| F4 | Four days, 2026-10-01 to 2026-10-04, with no extension. |
| F5 | Parallel tracks first, then one joined teacher, the corpus, the retrains and the reads. |
| F6 | A deep tray first. If a measurement shows that it makes the pick harder, the current tray stays with a refill event. |
| F7 | Ten working assumptions. Among them: MuJoCo only, the MS7e system as the start, a box emptying event, and a throughput bar in place of the cycles bar. |
| F8 | Six categories as the sort key: brick, special, flat, round, joint and minifig. |
| F9 | Joint and minifig parts from the LDraw library, about 8 kinds each, each kind one rigid body. |
| F10 | A read draws from every kind. The kinds that neither the policy nor the classifier saw form the held-out row, with a bar of 0.85. |

### 1.3 The gate and the reads

A correct stow is a part at rest in a box of its category. The grader reads it at the episode end,
or at the moment a person empties its box. The throughput is 3600 x (correct stows) / ((summed
episode steps) / 30), in correct stows per simulated hour. Every step of an episode counts. The
steps after the last stow count too (MS7f design section 4.1). An event, a tray refill or a box
emptying, takes zero simulated time. The cycles per correct stow stay in the payload with no bar.

Rule R2 set the gate as the design wrote it. A checked read has three bars. It needs at least 2760
of 3000 correct stows, at least 300 an hour and a held-out row of at least 0.85. The official read
goes to the candidate whose checked read passes. With two passing candidates, the higher throughput
wins. The official read is one shot. It passes with at least 5520 of 6000, at least 300 an hour and
a held-out row of at least 0.85. If no checked read passes, the milestone fails, and the report
gives the decomposition.

Deviation D12 (10:38) is an owner decision. The owner asked for it at 10:35, after the screens and
with 2 of the 20 heaps of candidate 1's checked read visible: 220 of 300, 0.733, at 194.7 an hour.
The gate becomes the staged bars. They ask for at least 0.85 in a box of the category (checked 2550
of 3000, official 5100 of 6000). They also ask for at least 200 correct stows an hour and a held-out
row of at least 0.80. The design's bars stay as a stretch row in every verdict and in this report.
The MS8 numbers stay report-only. The reason given to the owner: the speed bar is out of reach this
week. MS7e's 50-part task read 167 an hour, the recording teacher 220 to 260, and only the clean
teacher with the true class about 340.

D12 also fixes three consequences:

- Candidate 1's checked read keeps its protocol and supervisor, so its token judges the old bars.
  The controller computes its staged verdict from its `gate_results.json`.
- Rule R11's retrain trigger becomes a miss of any staged bar by candidate 1's checked read. On that
  miss, candidate 2 gets a checked read.
- The official read runs on the first candidate whose checked read passes the staged bars. Its
  supervisor runs at `--threshold 0.85 --min_stows_per_hour 200 --heldout_threshold 0.80` through
  `bin/read_f2.sh`, a copy of `read_f.sh` with those bars.

| read | policy | seed and size | staged bars (D12) | stretch row |
| --- | --- | --- | --- | --- |
| screen | each retrain checkpoint | 20307000, draws 0 to 3, 4 heaps of 150, 600 parts | none, picks the candidate (rule R11) | none |
| checked | candidate 1, candidate 2 on the retrain trigger, candidate 3 on a D16 adoption, candidates 3 and 4 in the refill scene (D19) | 20307100, draws 0 to 19, 3000 parts | at least 2550 correct stows, at least 200 an hour, held-out row at least 0.80 | 2760, 300 an hour, 0.85 |
| official | the first candidate whose checked read passes the staged bars, in the refill scene since D19 | 20307200, 40 heaps, 6000 parts, one shot | at least 5100 correct stows, at least 200 an hour, held-out row at least 0.80 | 5520, 300 an hour, 0.85 |
| control | train_E2 150k, no retrain | 20307300, draws 0 to 6, 1050 parts | none, reported | none |
| stretch runs | the official candidate | run 20307400 + r, layout 20307450 + r, r = 0 and 1, 1000 parts each | none | reported against 0.95, 400 an hour, 1000 parts, 5 interventions |

Deviation D19 (06:40) is an owner decision. The owner asked for it at 06:35, after the stretch run,
as decision F6's own fallback. The gate scene becomes the refilled deep tray, `--deep_tray
--tray_refill`. The tray keeps the 0.045 m walls and holds 50 parts at the start. At 10 or fewer
parts, it refills to 50, and a heap still holds 150 parts. The bars stay D12's staged ones, with the
design's as the stretch row. The full-pile results stay in this report as the harder row. Section
9.18 gives the reads that D19 pre-registers.

The held-out row counts the parts whose kind is in the held-out split. The policy corpus and the
classifier's render set use only the train split, so neither model saw these kinds (MS7f design
section 4.4).

### 1.4 The read protocol

Rule R3 keeps the MS7e official protocol and adds `--snap_closed`, the MS7f base switches and the
switches of every adopted MS7f rung. `f/bin/read_f.sh` builds every MS7f read:

| group | flags |
| --- | --- |
| scene and serving rules | `--tray --desk_boxes 8 --tray_lean 30 --scripted_rake --part_failures --pick_snap --dwell_close --adopt_held --park_done --rake_turn --perceived` |
| the nine MS7d levers | `--box_filter_xy --release_gap --rearm_hold --hold_force --force_cap --marker_bias --wrist_fix --class_views --box_walls` |
| the MS7e rungs | `--families round,ldraw,flat,joint,minifig --box_draw 8 --release_lean --carry_via --passenger_check --deep_box_floor --thin_snap --wall_band_cont --snap_closed` |
| the MS7f rungs | `--categories --deep_tray --box_empty --classifier <model> --heap_k 150 --num_envs 1` |
| the bars | `--min_stows_per_hour 300 --heldout_threshold 0.85 --max_cycles_per_stowed 1000`, threshold 0.92 |
| client flags | `--select_at_release --replan_on_switch --squeeze_floor --square_wrist --max_wall_minutes 900` |

A read uses `--parts_split all`, 50 actions per forward and one env per draw. A cycles limit of 1000
switches the MS7e cycles bar off. Every read reports Wilson intervals, the held-out row and the rows
per category, per family and per box count. It also reports the throughput, the events and the
cycles per correct stow.

### 1.5 The seeds

Rule R1 fixes every seed before its use. No MS7f seed lies in an MS7e range. The part split keeps
`parts_seed` 20260915.

| use | seed |
| --- | --- |
| H0 speed probe | run 20300000 + i, layout 20300050 + i |
| H1 and H1' teacher reads | run 20300100 + i, layout 20300150 + i, i = 0 to 5 |
| H diagnostics, never a verdict row | run 20300200 + i, layout 20300250 + i |
| S0 census | the E2 policy on MS7e official seed 20282100 draws 0 to 3. The teacher on run 20288900 + i, layout 20288950 + i, i = 0 to 5. |
| S1 ladder | run 20301000 + i, layout 20301050 + i, i = 0 to 11 |
| S2 rows | 20301500, draws 0 to 5 |
| C render set, train kinds only | run 20302000 + i |
| C1 read set, every kind | run 20302500 + i |
| C2 row | 20302900, draws 0 to 3 |
| P lone-part tables | 20303000 + the part's catalog index |
| P heap reads | run 20303500 + i, layout 20303550 + i, i = 0 to 3 |
| wave 1 | run 20304000 + p, layout 20304100 + p |
| wave 2 | run 20305000 + p, layout 20305100 + p |
| wave 3, DAgger (D17) | run 20308000 + p, layout 20308100 + p, train split |
| wave 4, DAgger in the refill scene | run 20308200 + p, layout 20308300 + p, p = 0 to 89, train split |
| teacher reference read | run 20306000 + i, layout 20306050 + i, i = 0 to 5 |
| screens, checked, official, control | 20307000, 20307100, 20307200, 20307300 |
| stretch runs | run 20307400 + r, layout 20307450 + r, r = 0 and 1 |

Rule R1 grew five times, each time before any use. At 17:42 it added a render extension range for
the classifier: run 20310000 + i, i = 0 to 999. The range covers train kinds and the D4 extra kinds
only. At 03:21 the range grew to i = 0 to 1999, and i = 1000 to 1999 went to model v6. At 15:41 it
added the seeds of classifier v7: deep-tray render heaps on run 20311200 + i, i = 0 to 799, train
kinds and D4 extras only. A deep-tray C1 read set of every kind uses run 20312000 + i, i = 0 to 99.
At 21:34 deviation D17 added the seeds of classifier v8. Its deep-tray render heaps run 20311200 + i
for i = 12 to 399 and 424 to 799, train kinds and D4 extras. The deep-tray C1 set grows with run
20312000 + i, i = 24 to 99. At 08:32 the wave 4 pre-registration gave both ranges of p = 0 to 89 to
wave 4.

### 1.6 The rules that carry over and the hosts

Rule R14 carries over MS7e rule R9. A payload from the supervisor cache is not a read. Every read
records its SHA, `policy_id`, split, switches and knob scales. A READ_FAIL reads once more in a
fresh TAG. Every paired table takes its base on its own architecture, because x86 and aarch64
physics differ. The MS6 negatives and the generic abort on contact stay closed.

Rule R15 keeps `scripts/fleet.py` and `~/data/maker-arm-ms7/hosts.json`, with track aliases
`a_fN`, `b_fN` and `aarch64_fN`.

| host | hardware | MS7f work |
| --- | --- | --- |
| Machine A | 24 cores, RTX 5090 | the controller and the ledger, the P tables, the classifier training, the three retrains, read lanes. Half of wave 1, half of waves 3 and 4. |
| Machine B | 24 cores, RTX 5080 | read lanes (half of candidate 1's checked read, the D11, D13 and D14 screens), half of wave 1, most of wave 2. Half of waves 3 and 4, the D18 PGS screen, the refill-scene pair screens and v8 renders. |
| Machine C | 128 cores, A2000 12 GB | the night render set, the S0 teacher census, 12 processes of wave 2, the wave 2 conversions, the control read, the F2 screens |

Machine C runs at most 40 camera processes in all. The Thor runs MS5 and is not used.

## 2. The base

The base branch, `feat/ms7f-base`, landed at b4a2af1 (12:05) with 1496 tests passing. It holds the
plumbing of every MS7f switch, so the other branches meet only at their call sites.

The six categories (MS7f design section 6.1):

| category | members |
| --- | --- |
| brick | the rectangular bricks, MS7e classes tall, narrow and wide |
| special | the slopes and the LDraw parts |
| flat | the plates and the tiles |
| round | the round family |
| joint | the joint family |
| minifig | the minifig family |

Each episode draws a map from the categories onto the drawn boxes (MS7f design section 6.2). With
fewer boxes than categories, the R11 fold of MS7e puts several categories in one box. With more
boxes, the categories with the largest expected volume take a second box, the largest first. A part
in any box of its category is a correct stow. A part whose category has two boxes goes to the box
with less volume in it.

The base also records the throughput (`steps_total`, `stows_per_hour`), the held-out row, the events
and the four S0 intervals of every cycle. The intervals split a cycle at the hold detector's events:
selection to first close, first close to lift, lift to release, and release to the next selection or
the episode end. The four add up to the cycle's steps.

Heaps reach 150 parts. The heap client's cycle budget is ceil(1.3 K), 195 at 150 parts. The
recorder's episode length is 3030 s at 150 parts, and the eval client's 7150 s. A run at K 150 needs
`--max_wall_minutes`.

The base track recorded four choices that the design left open (12:05):

- More boxes than categories get the extra boxes in expected-volume order, and R11's floor applies
  there too.
- Each chooser, the loop and the teacher, keeps its own volume record per box.
- The category map draws on the env generator. So the heap drops do not pair between
  `--categories` on and off. The Perceiver and ClassViews streams do pair.
- MS7f reads pass a cycles rule of 1000, no bar.

With every switch off, a pinned MS7e draw stream and a K 6 teacher payload equal those of 29a36ef.
The smokes on run 7 and layout 8, split all, with the MS7e wave flags and `--categories`:

| smoke | result |
| --- | --- |
| teacher, K 50 | EVAL_OK. 48 of 50 in a box of their category, 0 in a wrong box. 19339 steps, 268.1 stows an hour, held-out row 8 of 9. |
| recorder, K 50 | BATCH_OK, 46 of 50 |

## 3. Track H: the heap and the events

### 3.1 H0, the physics speed

Rule R4: H0 measures the control steps per second of a 150-part scene against a 50-part scene. The
bar: the projected official read, 40 heaps of 150 at the E2 pace, finishes on the three hosts within
8 hours. If H0 misses, H1' is adopted without the H1 read.

The steps per second at the full heap, with the arm held and the read cameras on
(`tables/heap/h0.txt`):

| host | 150 parts, deep tray | 50 parts |
| --- | --- | --- |
| Machine A | 1.75 | 67 |
| Machine B | 2.56 | 112 |
| Machine C | 0.74 to 0.99 | 36 |

The physics alone runs at 2.3 steps/s on the full deep heap on Machine A. The scene has 558
contacts, and the Newton solve takes 56 ms a physics step. Over a whole episode, the mean step time
of the deep tray is 10 to 12 times the 50-part one: on Machine A 92.4 ms against 8.2 ms.

| model | what grows | official read, 40 heaps on 12 + 12 + 10 lanes |
| --- | --- | --- |
| additive | only the physics | 5.17 h |
| scaled, a bound | the whole lane time | 11.09 h |

Decision (11:45): H0 reads on the additive model. The policy forward runs once per 50-step chunk,
and the renders do not grow with the part count. So H0 passes at 5.17 h against 8 h, and the R4 read
decides H1 against H1'. A wave on the deep tray projects to about 4.8 h, bound by
Machine C. The final check (15:56): a deep-tray heap on Machine B with the booking rule ran 14.34 steps/s
over the whole episode, against 14.7 predicted.

### 3.2 H1 against H1'

The two arms (MS7f design section 5.3):

- H1, the deep tray. The same footprint and lean, with walls 0.045 m high (`INITIAL:`). All 150
  parts drop in at the start.
- H1', the current tray with a refill. 50 parts drop in at the start. When 10 or fewer remain
  between cycles, parts drop in until 50 are in the tray (both numbers `INITIAL:`). A waiting part
  is parked out of reach and out of contact, and the perceived state does not see it.

Rule R4: the teacher runs the same 6 heaps of 150 on both arms, on the MS7e box configuration with
the category map, on one architecture. H1 has two bars against H1'. Its share in a correct box must
be at least H1''s minus 0.02. Its seconds per correct stow must be at most 1.1 times H1''s. If H1
holds both bars, it is adopted.

The reads ran on x86, split all, with the teacher, `--categories --box_empty`, H1 on Machine B and H1' on
Machine A (`tables/heap/r4_verdict.txt`). This report computes the Wilson intervals.

| arm | correct box | Wilson 95 % | s per correct stow | hidden share at selection | s per correct stow without the long cycles |
| --- | --- | --- | --- | --- | --- |
| H1, deep tray | 842 of 900, 0.936 | 0.918 to 0.950 | 18.42 | 0.347 | 13.47 |
| H1', refill | 857 of 900, 0.952 | 0.936 to 0.964 | 19.10 | 0.226 | 12.33 |
| bar for H1 | at least 0.932 | | at most 21.0 | | |

H1 holds both bars, and decision F6 adopts the deep tray (15:55). Without the long cycles, H1 still
holds both bars at 1.09 times H1''s seconds. The hidden share is the share of parts that the
perceived state does not see at a selection.

Both arms carry a waste that the next section explains: H1' heaps 000 and 004 and H1 heap 000 hold
long cycles on one stray part. The deep tray costs about three times the compute of the refill. An
H1 heap took 4188 to 9464 s on Machine B. The ledger gives 1676 to 2229 s per H1' heap on Machine A.
`r4_verdict.txt` lists 7007 s and 4440 s for H1' heaps 000 and 004, the two heaps with the stray
loop. So the waves put most heaps on x86.

### 3.3 H2, the box emptying event

Between cycles, the loop sums the bounding volume of the parts in each box. When a box reaches its
capacity, its interior volume divided by `BOX_VOLUME_PER_PART_VOLUME`, a person empties it. The
grader reads every part in the box at that moment. Then the parts are parked, and the perceived
state forgets them. The payload counts the emptying events, and they are not interventions.

H2 runs on every MS7f read and has no bar of its own. In the R4 reads, a heap of 150 had 0 to 2
emptying events.

### 3.4 The stray loop and `--book_miss`

The heap track found the waste of the R4 reads (15:56). A part lies on the desk in a narrow gap by a
desk box. On the truth path, the teacher's grasp model has only the tray walls, so every descend
puts a finger on each box wall. The rig gives up after 3 x 3 lifts, marks the cell in its own memory
and delivers a neighbour. The loop's cycle then ends "stowed" with no failure booked, and the
selector takes the same stray part again.

`--book_miss` (truth path only) books a failure on the target of a cycle whose stows leave the
target out. It cuts H0 heap A0 from 42.9 to 16.4 s per stow and H0 heap C1 from 53.4 to 15.9 s. The
ledger adopted it for the teacher reads at 15:56. The R9 census of section 7 then removed it from
every teacher read (19:29). The recorder never ran it.

### 3.5 The wave blockers

Two defects stopped the first launch of wave 1 (section 8.1). Track H fixed both on
`feat/ms7f-heap-w1`, 4464b79 on 077c7f0 (19:27):

| commit | fix |
| --- | --- |
| 125d86e | `--abandon_cycle`: a jammed cycle ends, not the episode |
| 34da9f6 | `--near_tracking`, not used |
| 4c2575d | the scene spawns parts in rows of 50, to keep the camera clip |
| 4464b79 | the deep tray drop reads the pile top on the body poses it wrote |

Two checks ran before the relaunch. The check w1fix (125d86e, Machine A, 2 deep-tray heaps of 150,
cameras on, run 20300202 to 20300203) ended CORPUS_OK with 0 abandons. Both heaps exported, with 136
and 134 of 150 at 11.9 and 19.2 s per stow. The check e2_k150 (E2, K 150, refill, 20 cycles) stowed
15 parts, and its wrist frame shows the jaws.

## 4. Track P: the joint and minifig families

### 4.1 The kinds

`scripts/ldraw_parts.py` builds each kind from the LDraw library with the hull cap of 4. Every kind
is one rigid body. The final kinds (`tables/parts/final_kinds.txt`, feat/ms7f-parts 9754ee8):

| kind | catalog index | size, mm | mass, g | closing axes | split |
| --- | --- | --- | --- | --- | --- |
| joint_2780 | 51 | 16.0 x 6.4 x 6.24 | 0.500 | 0, 1, 2 | train |
| joint_6558 | 52 | 24.0 x 6.4 x 6.24 | 0.598 | 0, 1, 2 | train |
| joint_43093 | 53 | 15.8 x 6.4 x 6.4 | 0.500 | 0, 1, 2 | train |
| joint_3937 | 54 | 16.0 x 8.0 x 8.8 | 0.703 | 0, 1, 2 | train |
| joint_3938 | 55 | 16.0 x 8.0 x 6.4 | 0.511 | 0, 2 | train |
| joint_14417 | 56 | 16.0 x 15.2 x 6.4 | 0.971 | 0, 2 | train |
| joint_14418 | 57 | 22.0 x 8.0 x 6.2 | 0.681 | 0, 1, 2 | held out |
| joint_4081b | 58 | 15.2 x 8.0 x 6.4 | 0.500 | 1, 2 | held out |
| minifig_3626b | 59 | 10.4 x 10.4 x 9.6 | 0.648 | 0, 1, 2 | train |
| minifig_973 | 60 | 15.2 x 8.0 x 17.6 | 1.335 | 0, 1, 2 | train |
| minifig_3815c01 | 61 | 15.6 x 8.4 x 20.4 | 1.668 | 0, 1, 2 | train |
| minifig_3901 | 62 | 12.37 x 12.01 x 9.63 | 0.892 | 0, 1, 2 | held out |
| minifig_3624 | 63 | 14.28 x 13.07 x 6.35 | 0.740 | 0, 1, 2 | held out |
| minifig_3838 | 64 | 13.6 x 11.2 x 14.4 | 1.369 | 0, 1, 2 | train |
| minifig_4349 | 65 | 16.0 x 8.0 x 12.0 | 0.958 | 0, 1, 2 | train |

Each family holds out ceil(0.2 n) of its n kinds on its own generator, and the base split does not
move. With `--categories` off, `part_class` maps joint and minifig to "special".

### 4.2 The lone-part tables

Rule R5: 12 trials a kind on seeds 20303000 + the catalog index, the teacher as it is. A family
under 0.90 tries at most two levers, each behind its own switch. A family at 0.90 or more runs its
heap read. The tables ran on Machine A, x86 (`tables/parts/tables_sum.txt`):

| family | table | commit | stowed | notes |
| --- | --- | --- | --- | --- |
| joint | baseline | f2ebac1 | 84 of 96, 0.875 | joint_14417 4 of 12 and joint_4081b 10 of 12, both LIFT>REOPEN |
| joint | lever 1, "axes" | 64defa9 | 94 of 96, 0.979 | joint_14417 and joint_4081b 12 of 12 |
| joint | 3938 closes along x and z | c3ca2d6 | 95 of 96 | |
| minifig | baseline | cdff4e7 | 90 of 96, 0.938 | minifig_3838 9 of 12 (drops in EXTEND and RELEASE), minifig_3629 10 of 12 |
| minifig | final, 7 kinds | 9754ee8 | 81 of 84 | |

Both families reach 0.90. Joint needs one lever, and minifig none.

### 4.3 The heap reads and the census

Rule R5: 4 heaps of 50 on the "all" split, half from the new family, with the category map. The
bars: the family's own row stows at least 0.85 in a correct box, and the rest of the heap at least
0.90. A family that holds both bars is adopted. The reads ran on x86 (`tables/parts/heap_reads.txt`,
`heap_reads_final.txt`):

| TAG | own row | rest of the heap | all | s per stow | s per own-family stow |
| --- | --- | --- | --- | --- | --- |
| f_p_heap_nonew, no new family | | | 192 of 200, 73063 steps | | |
| f_p_joint_heap | 96 of 100 | 97 of 100 | 193 of 200, 168738 steps | 29.1 | 47.5 |
| f_p_minifig_heap | 95 of 100 | 96 of 100 | 191 of 200, 92615 steps | 16.2 | 18.0 |
| f_p_joint_heap2, 3938 fix | 96 of 100 | 97 of 100 | 193 of 200 | 14.2 | 15.2 |
| f_p_minifig_heap2, 7 kinds | 97 of 100 | 98 of 100 | 195 of 200 | 15.1 | 18.0 |

Both families hold both bars on every read, so both are adopted (12:09, kept at 12:51).

The first reads raised a throughput flag. The joint heaps ran 168738 steps for 193 stows, against
73063 for 192 without the new families. The census (12:24, `tables/parts/census.txt`) found the
waste in one kind each:

- joint_3938, the hinge 1x2 top, took 113877 steps in 56 cycles. Its underside leaves a 1.8 mm rib
  under the top slab, and a y grasp closes on the rib alone. Fix c3ca2d6 closes 3938 along x and z
  only. Every other joint kind reads 300 to 600 steps a cycle.
- minifig_3629, the cowboy hat, took 21 cycles at 1009 steps a cycle. The jaws ride over the
  curved brim.

Deviation D3 (12:24) drops minifig_3629 from the family. The box grasp model cannot hold an upright
hat, and the throughput bar charges every wasted cycle. The catalog holds 66 kinds. The hull cap of
4 holds, because no lane falls under half of its nonew lane.

After the fixes, minifig cycles still read 524 steps against 339 for the other kinds.

### 4.4 Open items of track P

- When a cycle stows a part other than its target, the target's failure memory is not marked. So
  the selector takes the same hard target again. This amplified joint_3938.
- Late close-edge ladders remain on joint_6558, minifig_973 and minifig_3815c01.

### 4.5 The classifier-only kinds: deviation D4

Deviation D4 (15:23) lets the classifier's render set add extra part kinds that are in no catalog
and no split. The network learns category cues from more kinds, and the catalog's held-out kinds
stay unseen by the policy and the classifier. The held-out row keeps its meaning.

Track P built 45 such kinds (16:36, feat/ms7f-cls-extra 690117f, `tables/parts/cls_extra.txt`):
brick 7, special 8, flat 8, round 4, joint 9 and minifig 9. They live in `parts.CLASSIFIER_EXTRA`,
never in `catalog()`, `split()` or `catalog_index`. Their lone-part lift checks read 9 to 12 of 12.
The set leaves out domed heads, arches with no side closing axis, degenerate hulls, and round plates
and tiles. It also leaves out the LDraw twins of held-out catalog kinds: 3040 and 4286 (slope_1x2,
slope_1x3) and 3749 (43093).

## 5. Track C: the classifier and the passenger signal

### 5.1 The views and the model

Each view of the MS7d vote calls the classifier on a rendered frame (MS7f design section 6.3):

1. The overhead view at selection. A narrow camera at the overhead camera's pose renders 224 x 224
   px around the target, at about 0.5 mm a pixel. This guess sets the box in the policy context
   before the lift.
2. The wrist view at the lift.
3. The wrist view during the carry, as the third vote.

The design trains a torchvision ResNet18 from ImageNet weights in the lerobot_stack venv and exports
a TorchScript file (MS7f design section 6.5). The heap client loads it with `torch.jit.load` and
runs it on the CPU. It has two heads: the category and, on wrist frames, the passenger count. The
render set labels each frame from the simulator. That is a recording tool, not a decision, so no
decision reads the truth.

The passenger response, `--passenger_drop`: if the wrist view at the lift reads a passenger over the
tray interior, the loop opens the jaws and ends the cycle. Over anything else, the loop ignores the
passenger reading.

### 5.2 The bars

Rule R6 (MS7f design section 6.7). C1 reads a render set of every kind, train and held-out, on its
own seeds:

- The wrist view at the lift reads the right category at least 0.97 of the time, over all parts.
- Every category reads right at least 0.90 of the time.
- The passenger head finds at least 0.8 of the passengers, with at most 0.02 false alarms.
- The held-out kinds and the overhead view are reported rows with no bar.

C has no fallback. If C1 misses, the classifier still serves, and the report gives its confusion
matrix.

### 5.3 The model ladder

| model | time | change | wrist at the lift | three-view rule `loop_sum` | held-out |
| --- | --- | --- | --- | --- | --- |
| v1 | 13:39 | 160 train render heaps, a fingertip crop, a burst of wrist frames, 16 epochs. C1 on 48 heaps of the 51-kind catalog. | 1832 of 2157, 0.849 | | lift 0.719 |
| v2 | 14:42 | 208 train render heaps, a wrist crop, a 56 mm overhead crop, 16 epochs. C1 on 72 heaps of all 66 kinds. | 2682 of 3122, 0.859 | 0.888, adopted as the serving rule | lift 0.685 |
| v3 | 15:06 | the perceived size as a second input | 0.861 | 2481 of 2771, 0.895 | lift 0.694, `loop_sum` 0.726 |
| v4a | 17:37 | a native 448 px wrist crop, about half the lifts | | 0.877, "fused" in the ledger | `loop_sum` 0.697 |
| v4 | 18:15 | native crop, 56 heaps of extra kinds, random erasing, label smoothing 0.1, early stop at epoch 8 | 0.832 | 2461 of 2799, 0.879 | `loop_sum` 0.807 |
| v5 | 23:42 | 528 train render files, 114,266 train frames with the D4 extras and the night set, early stop at a validation lift of 0.924 | 2819 of 3147, 0.896 | 2597 of 2799, 0.928 | lift 0.819, `loop_sum` 0.857 |
| v6a | 04:11 | ResNet50 on v5's data and recipe, epoch 14 kept | 2838 of 3147, 0.902 | 2629 of 2799, 0.939 | `loop_sum` 579 of 664, 0.872 |

`loop_sum` sums the log probabilities of the views where the lift read and the overhead read split.
The ledger fused v2's three views on 2771 lifts. The lift alone read 0.859, the MS7d vote 0.875, the
summed log probabilities 0.891 and `loop_sum` 0.888. The ledger adopted `loop_sum` behind
`--classifier` (0cdc7db).

The C1 counts change from 3122 lifts and 2771 three-view lifts (v2, v3) to 3147 and 2799 (v4 to
v6a). The ledger note of 05:42 gives the cause: the render set was rendered again at the native
crop, and the C1 set grew.

What the ladder found:

- The overhead view cannot see some categories (14:42). Flat parts that lie flat read 0.01 to 0.09,
  with no thickness from above. Technic bricks upright read 0.04 to 0.43, because the holes are on
  the sides. The size input of v3 fixes the flat parts upright (0.37 to 0.80), but not the
  brick-look held-out kinds. ldraw_3894 measures 48.0 x 8.0 x 9.6 mm, brick_1x6 47.8 x 7.8 x 9.6.
- 11 % of the C1 lifts hold a neighbour, not the target (15:20). The wrist frames already carry the
  held part's category, so no relabel is due. On target grasps, v2 reads 0.860 and v3 0.870. On
  neighbour grasps, v2 reads 0.853 and v3 0.787, because the size input is the target's.
- v2 overfits (17:37). It reads its own training heaps at 0.977 and its by-heap validation heaps at
  0.894. The error grid shows hard frames. A part is seen side-on or from below, or half hidden by
  the fingers. Pins and 1x1 parts are a few pixels wide, and some frames show a rider on top.
- The extra kinds carry category cues to unseen kinds (18:15). With them, v4's held-out `loop_sum`
  rises to 0.807 from v3's 0.726.

### 5.4 The C1 verdict

Model v5 (23:42, `tables/cls/model_v5/`) on the 72 C1 heaps of all 66 kinds:

| row | wrist at the lift | `loop_sum` |
| --- | --- | --- |
| all | 2819 of 3147, 0.896 | 2597 of 2799, 0.928 |
| train kinds | | 0.950 |
| held-out kinds | 0.819 | 0.857 |
| brick | 0.900 | 0.936 |
| special | 0.819 | 0.833 |
| flat | 0.927 | 0.980 |
| round | 0.984 | 1.000 |
| joint | 0.927 | 0.973 |
| minifig | 0.953 | 0.985 |
| passenger head | recall 55 of 124, 0.44, false alarms 17 of 3023, 0.006 | |

C1 misses three bars: 0.896 at the lift against 0.97, special 0.819 against 0.90, and the passenger
recall 0.44 against 0.8. By rule R6, model v5 serves every MS7f read that starts before v6a, and
`--passenger_drop` stays off.

The confusion matrix of v5 at the lift, over all 3147 lifts (`model_v5/c1.json`). A row is the true
category, and a column is the category that the classifier reads:

| true, read | brick | special | flat | round | joint | minifig | right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brick | 1158 | 85 | 28 | 2 | 3 | 11 | 1158 of 1287 |
| special | 108 | 619 | 7 | 3 | 14 | 5 | 619 of 756 |
| flat | 17 | 6 | 368 | 2 | 4 | 0 | 368 of 397 |
| round | 1 | 1 | 0 | 187 | 0 | 1 | 187 of 190 |
| joint | 6 | 1 | 7 | 0 | 203 | 2 | 203 of 219 |
| minifig | 3 | 6 | 0 | 1 | 4 | 284 | 284 of 298 |

Special read as brick (108) and brick read as special (85) are the two largest errors.

The five worst kinds at the lift are in `c1_views.txt`. ldraw_11211 reads 23 of 59 (0.390),
slope_1x2 31 of 48 (0.646) and ldraw_3894 45 of 66 (0.682). brick_2x3 reads 40 of 57 (0.702) and
ldraw_87087 25 of 35 (0.714). The file marks ldraw_11211, ldraw_3894 and ldraw_87087 as kinds with a
brick-look top face.

### 5.5 Model v6a

Track C reopened at 03:21. The screens showed the classifier behind 14 of 15 `wrong_box` cycles on a
heap, mostly special read as brick. The pre-registered v6 bar on the same 72 C1 heaps has four
parts:

- `loop_sum` at least 0.938, v5 + 0.01.
- Special `loop_sum` at least 0.863, v5 + 0.03.
- No category's `loop_sum` more than 0.02 under v5's.
- The held-out `loop_sum` at least 0.857, v5's value.

| row | v5 | v6a | bar |
| --- | --- | --- | --- |
| `loop_sum`, all | 0.928 | 2629 of 2799, 0.939 | at least 0.938 |
| `loop_sum`, special | 0.833 | 581 of 658, 0.883 | at least 0.863 |
| `loop_sum`, held-out | 0.857 | 579 of 664, 0.872 | at least 0.857 |
| `loop_sum`, brick | 0.936 | 0.931 | at least 0.916 |
| `loop_sum`, flat, round, joint, minifig | 0.980, 1.000, 0.973, 0.985 | 0.980, 1.000, 0.995, 0.988 | at most 0.02 under v5 |
| wrist at the lift | 2819 of 3147, 0.896 | 2838 of 3147, 0.902 | |
| passenger head | 0.44 at 0.006 | 62 of 124, 0.50, at 43 of 3023, 0.014 | |
| time a read, CPU | 34 ms | 61 ms | |

v6a passes, and the ledger adopts it (04:11, feat/ms7f-cls6 ab3008d). The branch adds `--arch`,
`--input_px` and `--ensemble` to the trainer, with every default unchanged. On the 2799 three-view
lifts, the lift read of brick falls from 0.903 to 0.871 and that of special rises from 0.810 to
0.854. The attempt v6f (v5 with a flip at test time) missed at 0.933 with special 0.842. The attempt
v6b (ResNet18 at 320 px) was stopped. The design names a ResNet18 (MS7f design section 6.5), so v6a
departs from it.

The ledger's v6a numbers come from the CPU read as served (`model_v6a/cpu/`). Its confusion matrix
at the lift, over all 3147 lifts:

| true, read | brick | special | flat | round | joint | minifig | right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brick | 1123 | 85 | 74 | 0 | 4 | 1 | 1123 of 1287 |
| special | 90 | 645 | 15 | 0 | 4 | 2 | 645 of 756 |
| flat | 8 | 7 | 380 | 1 | 1 | 0 | 380 of 397 |
| round | 0 | 0 | 0 | 189 | 0 | 1 | 189 of 190 |
| joint | 4 | 2 | 4 | 0 | 209 | 0 | 209 of 219 |
| minifig | 1 | 3 | 0 | 1 | 1 | 292 | 292 of 298 |

Against v5, special read as brick falls from 108 to 90, and brick read as flat rises from 28 to 74.

The ledger clarified the use of v6a at 04:11, before any warp or passenger-drop number. The D7 and
D9 screens pair with candidate 1's own screen, which read with v5, so they keep v5. v6a serves
candidate 1's checked read and every read after it. A read that started keeps its classifier.

### 5.6 C2, the classifier in the loop

Rule R6: C2 reads E2, train_E2 150k with no retrain, on the MS7e 50-part configuration with the
category map. It reads split all, seed 20302900 draws 0 to 3, once with the stand-in and once with
the classifier. It has no bar. This report computes the Wilson intervals.

| arm | model, commit, host | correct box | Wilson 95 % | wrong-box parts | cycles per correct stow |
| --- | --- | --- | --- | --- | --- |
| stand-in | 95c069e, Machine B | 907 of 1000, 0.907 | 0.887 to 0.923 | 61, 6.3 % of boxed parts | 1.155 |
| classifier | v2, 0cdc7db, Machine A | 758 of 1000, 0.758 | 0.730 to 0.784 | 194, 20.4 % of boxed parts | 1.343 |
| classifier, census | v2, 6a01b99, aarch64, CPU policy | 796 of 1000, 0.796 | 0.770 to 0.820 | 176 | |

With the stand-in, E2 already reads under 0.92 on the category task. The classifier arm reads 292
wrong overhead classes in 1018 selections, 127 wrong wrist classes in 840 lifts and 62 vote changes.

The census arm (18:33) runs an instrumented loop on Machine C, so it does not pair with the
x86 arms. Of its 124 wrong-box cycles, 122 had the wrong class:

| cause | cycles |
| --- | --- |
| the lift agreed with a wrong overhead read | 45 |
| the summed vote at a split was wrong | 50 |
| no lift read, and the overhead box stayed | 23 |
| the vote was lost | 4 |

The ledger counts 75 held-out kinds among these cycles. The loop's final class is right on 692 of
814 boxed cycles, 0.850, against C1's `loop_sum` of 0.888 for v2. C2 draws 31 % held-out parts
against 24 % of C1's lifts, and the grasps shift from the teacher's to the policy's. 52 more
wrong-box parts are riders or parts knocked into a box.

### 5.7 The passenger head

The passenger head learns the count of parts that rise with the jaws. Its recall was 0.22 for v1,
0.164 for v2 after a label fix (87d99fe), 0.44 for v5 and 0.50 for v6a. No model reached the 0.8
bar. `--passenger_drop` is not adopted at C1. Deviation D9 gives it one screen in the loop, and the
screen does not adopt it either (section 9.4).

### 5.8 Classifier v7 on the deep tray

The in-loop wrist read trails C1 (section 9.9). C1 and every render file use 50-part heaps in the
normal tray, but the gate serves 150-part heaps in the deep tray. Over each heap's first 40 cycles,
the in-loop wrist read is right on 0.771 (`tables/clsloop`).

The ledger pre-registered v7 at 15:41, before any number. v7 is v6a's recipe plus 150-part deep-tray
render heaps, the gate scene. It replaces v6a for every read that has not started on these bars:

- On a new deep-tray C1 set, its `loop_sum` is at least v6a's plus 0.02.
- On the old C1 set of 72 heaps, its `loop_sum` is at least v6a's 0.939 minus 0.01.
- On either set, no category reads more than 0.02 under v6a's, and the held-out rows do not read
  under v6a's.

At 15:45 the ledger found a blocker. `scripts/mj_render_set.py` never took the MS7f switches: no
branch gave it `--deep_tray`, `--categories` or `--box_empty`. So every render file and both C1 sets
so far used the normal tray. The authorized fix is a minimal patch on top of `feat/ms7f-levers2`
that mirrors the recorder's MS7f plumbing and adds `--abandon_cycle` to the renders. The deep-tray
heaps are bounded by `--cycle_budget 40`, the early pile where the in-loop read is worst, and by 300
wall minutes. The v7 bar stays as pre-registered.

The patch is 5e92758 on `feat/ms7f-levers2` (16:48). The render script takes `--categories`,
`--deep_tray`, `--tray_refill`, `--box_empty` and `--abandon_cycle`, and it refuses the serving
switches. 1559 tests pass under `MUJOCO_GL=disable`. A smoke on seed 20311999 ended RENDER_OK with
72 frames, the jaws visible over a deep pile, at about 550 steps a cycle.

The owner approved the syncs at 16:45. The render jobs:

| job | host | seeds | cycles a heap |
| --- | --- | --- | --- |
| f_cls_v7_c1d, the deep-tray C1 set | Machine C | 24 heaps, 20312000 to 20312023 | 20 |
| f_cls_v7_rs7a, train renders | Machine C | 20311200 to 20311211 | 20 |
| f_cls_v7_rs7t, train renders | Machine B, alias `b_fr5`, 8 lanes | 20311600 to 20311623 | 40 |

Job f_cls_v7_chain runs `tables/cls/bin/v7_pipeline.sh` after them. It trains v6a's recipe on the
render set plus the new renders, unweighted. It then reads v7 and v6a on the CPU on both C1 sets and
applies the bar (`v7_bar.py`).

The renders ended at 20:23, and the pipeline gave its verdict at 20:53. v7 trains on 119,291 frames
in 639 s and keeps the epoch with a validation lift of 0.888. This report computes the new deep-tray
frames as 119,291 less v6a's 114,266, about 5,000 frames, 4 % of the set.

The deep-tray C1 set holds 24 heaps (seeds 20312000 to 20312023, 20 cycles each), 407 lifts and 360
three-view lifts:

| row | v6a, deep tray | v7, deep tray | v6a, old C1 72 heaps | v7, old C1 72 heaps |
| --- | --- | --- | --- | --- |
| `loop_sum`, all | 323 of 360, 0.897 | 323 of 360, 0.897 | 2629 of 2799, 0.939 | 2544 of 2799, 0.909 |
| wrist at the lift | 0.816 | | 0.902 | |
| held-out | 0.806 | 0.778 | 0.872 | |
| brick | 0.918 | 0.884 | 0.931 | 0.852 |
| special | 38 of 55, 0.691 | 45 of 55, 0.818 | 0.883 | 0.904 |

The bar asks for at least 0.917 on the deep-tray set and at least 0.929 on the old set, and v7 reads
0.897 and 0.909. Brick falls more than 0.02 on both sets, and the deep-tray held-out row falls from
0.806 to 0.778. So v7 misses every check of the bar, and v6a stays (20:53).

The deep-tray read set shows the scene gap. v6a reads 0.897 on deep-tray heaps against 0.939 in the
normal tray, and special 0.691 against 0.883. v7 lifts special on both sets but loses brick.
Candidate 3's checked read serves v6a.

### 5.9 Classifier v8: deviation D17

Deviation D17 (21:34, section 9.11) makes classifier v8, the gate-scene classifier, one of its two
levers. Its render seeds are in section 1.5. The renders can use `--pile_pgs`, PR #153's PGS solver
for the full pile, because the classifier sees images. Every read stays on Newton. The v8 bar,
pre-registered at 21:34:

- On the deep-tray C1 set, with all of its heaps at the time of the v8 read, `loop_sum` is at least
  v6a's plus 0.03.
- On the same set, no category reads more than 0.03 under v6a's, and the held-out row does not read
  under v6a's.
- On the old C1 set of 72 heaps, `loop_sum` is at least v6a's 0.939 minus 0.01.

A v8 that passes then gets an in-loop screen against candidate 3 on seed 20307000, draws 0 to 3. It
serves on two conditions: at least candidate 3's screen plus 6 correct stows, and at least 0.97 of
its rate.

A subagent built the v8 tools on `feat/ms7f-render-v8` 0323927, on top of `feat/ms7f-pgs` (21:55).
The render script reports `pile_pgs` in its RENDER_READY line. The trainer takes `--repeat N
--repeat_match GLOB`, which repeats the matching render files on the training rows only. 1566 tests
pass. A smoke with `--pile_pgs` ran 8 cycles in 473 s with 122 frames and an intact deep pile. Its
Newton twin still ran at 25 minutes. The render jobs:

| job | host | solver | seeds | target |
| --- | --- | --- | --- | --- |
| f_cls_v8_c1d, the deep-tray C1 set | Machine C, 38 processes | Newton | 20312000 + i, i = 24 to 61, 62 heaps in all | the C1 read |
| f_cls_v8_rs8t, train renders | Machine B, 8 processes | PGS | 20311200 + i, i = 424 to 599, 40 cycles | the training rows |
| later train renders | Machine B, then Machine C | | i = 600 to 799, then i = 12 to 399 | the training rows |

The target was at least 40,000 deep-tray frames after `--cap 80`, weighted to about half the
training rows. The v8 verdict came at 06:17. The training set holds 27 rs7 and 207 rs8 deep-tray
files: 53,376 deep-tray frames at `--cap 80`, repeated 3 times to about half the training samples.
`tables/cls/model_v8.log` gives a deep share of 0.524. v8 trains on 244,962 frames, 158,728 of them
unique, in 2959 s, and keeps the epoch with a validation lift of 0.889. The deep-tray C1 set grew to
62 heaps and 928 three-view lifts, read on Newton:

| row | v6a, deep tray | v8, deep tray | v6a, old C1 72 heaps | v8, old C1 72 heaps |
| --- | --- | --- | --- | --- |
| `loop_sum`, all | 841 of 928, 0.906 | 842 of 928, 0.907 | 2629 of 2799, 0.939 | 2580 of 2799, 0.922 |
| wrist at the lift | 873 of 1036, 0.843 | 862 of 1036, 0.832 | 2838 of 3147, 0.902 | 2801 of 3147, 0.890 |
| held-out | 173 of 207, 0.836 | 174 of 207, 0.841 | 579 of 664, 0.872 | 567 of 664, 0.854 |
| brick | 0.926 | 0.888 | 0.931 | 0.868 |
| special | 0.726 | 0.879 | 0.883 | 0.915 |
| joint | 22 of 22 | 20 of 22 | 0.995 | 0.995 |

The bar asks for at least 0.936 on the deep-tray set and at least 0.929 on the old set, and v8 reads
0.907 and 0.922. Brick falls by 0.039 and joint by 0.091 on the deep-tray set. Only the held-out
check passes, so v8 misses 3 of the 4 checks, and v6a stays (06:17). The ledger gives the `loop_sum`
row and the deep-tray rows of held-out, brick, special and joint. The other cells come from
`tables/cls/model_v8.log`.

Like v7, the deep data moves the line between plain bricks and the brick-like specials instead of
sharpening it. At the lift on the old set, the bricks read as special grow from 85 to 160. The
ledger's reading: the classifier's capacity, not its scene, now limits it.

### 5.10 Classifier v9e, an ensemble of v6a and v8

The ledger pre-registered v9e at 07:22, with no v9e number in existence. It came after the interim
look of 07:22 at the refill-scene reads (section 9.18), which the ledger disclosed first. v9e is the
trainer's `--ensemble` of model v6a and model v8: the mean of their logits, with no flip. It comes
from the v8 worktree at 0323927, and nothing is trained. The reason: on the deep-tray C1 set, v8
lifts special from 0.726 to 0.879, and v6a holds brick at 0.926 against 0.888. So their errors
differ.

The C1 bar reads `loop_sum` on the CPU as served, on all 62 deep-tray heaps and the old 72:

- On the deep-tray set, the whole row is at least v6a's plus 0.01.
- On the deep-tray set, no category reads more than 0.03 under v6a's, and the held-out row does not
  read under v6a's.
- On the old set, the whole row is at least 0.939 minus 0.01.

On a C1 pass, a paired screen f_M_cap5_v9e_R_scr follows in the refill scene. It uses candidate 3's
flags, seed 20307000, draws 0 to 3, model v9e and the host of its pair f_M_cap5_R_scr. v9e is
adopted for every later read on two conditions: at least the pair's correct stows plus 6, and at
least 0.97 of the pair's correct stows an hour. A miss of C1 or of the screen ends the classifier
track, and v6a stays. The output is `tables/cls/model_v9e`.

The C1 verdict came at 07:58 (`tables/cls/model_v9e/bar.txt`, read on the CPU):

| row | v6a, deep tray | v9e, deep tray | v6a, old C1 72 heaps | v9e, old C1 72 heaps |
| --- | --- | --- | --- | --- |
| `loop_sum`, all | 841 of 928, 0.906 | 857 of 928, 0.9235 | 2629 of 2799, 0.939 | 2628 of 2799, 0.939 |
| wrist at the lift | 873 of 1036, 0.843 | 882 of 1036, 0.851 | 2838 of 3147, 0.902 | 2869 of 3147, 0.912 |
| held-out | 173 of 207, 0.836 | 173 of 207, 0.836 | 579 of 664, 0.872 | 584 of 664, 0.880 |
| brick | 0.926 | 0.921 | 0.931 | 0.906 |
| special | 114 of 157, 0.726 | 131 of 157, 0.834 | 0.883 | 0.918 |
| flat, round, joint | 1.000 | 1.000 | | |
| minifig | 44 of 46 | 46 of 46 | 0.989 | 1.000 |

The deep-tray row gains 0.017 against the bar's 0.01. Brick falls by 0.005 on the deep-tray set,
inside the 0.03 margin, and the held-out row stays equal. The old set reads 0.939 against a bar of
0.929. So v9e passes every check of its bar (V9E_BAR PASS). The deep-tray wrist row and the old
set's flat, round and joint rows come from `bar.txt`.

The paired screen started on `b_fr6:4` at `PORT_BASE` 17000 and ended at 08:31. Both reads serve
F2 150k with candidate 3's flags in the refill scene:

| screen | classifier | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong-box parts, `wrong_box` cycles | brick | special |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_M_cap5_R_scr, the pair | v6a | 468 of 600, 0.780 | 0.745 to 0.811 | 768 | 244,178 | 207.0 | 111 of 151 | 75, 45 | 168 of 202 | 109 of 148 |
| f_M_cap5_v9e_R_scr | v9e | 474 of 600, 0.790 | 0.756 to 0.821 | 767 | 237,919 | 215.2 | 104 of 151 | 69, 39 | 161 of 202 | 116 of 148 |

The rule asks for at least 474 correct stows and at least 200.8 an hour. v9e reads 474 at 215.2, so
it is adopted at the bar exactly, +6 (08:31). It serves every later read, and candidate 4's checked
read reads with it. Its held-out row reads 104 of 151 against the pair's 111. The rule does not test
that row.

## 6. Track S: the speed

### 6.1 S0, the cycle time census

Rule R7: S0 records the four intervals of every cycle and names the wait trims
(`tables/speed/SUMMARY.txt`).

The teacher, aarch64, pace 4.0, the MS7e reference seeds, 6 train heaps of 50:

| item | value |
| --- | --- |
| correct box | 291 of 300, 0.970 |
| seconds per correct stow | 12.47, 289 an hour |
| steps a cycle | 370 |
| DESCEND walk and settle | 93 steps, 25 % |
| CLOSE hold | 50 steps, 14 % |
| LIFT hold | 34 steps, 9 % |
| RELEASE | 20 steps, 5 % |
| the walks HOVER, LIFT, EXTEND, RETREAT | 145 steps, 39 % |

The policy, E2 on Machine A, MS7e official seed 20282100 draws 0 to 3, 1000 held-out parts:

| item | value |
| --- | --- |
| correct box | 884 of 1000, 0.884 |
| seconds per correct stow | 21.64, 166 an hour |
| seconds a cycle | 16.98 |
| cycles per correct stow | 1.275 |
| stowed cycles | 55 % of the steps. Intervals 165, 117, 93 and 27 steps. |
| failed cycles | 45 % of the steps: `close_miss` 18.2 %, `held_no_lift` 11.5 %, `lifted_lost` 10.2 % |

Against the teacher, the policy spends 45 more steps on the approach and stands still 58 steps after
its close. It saves 62 steps after the release, because it selects at the release and has no
retreat. The policy's 16.98 s a cycle follows its corpus. MS7e wave 1 ran 16.7 s a cycle. The
recorder recipe reads 16.38 s per stow on the S1 heaps, against 12.85 s for the read teacher.

### 6.2 S1, the teacher speed ladder

Rule R7: the rungs are `transport_speed` 5, 6 and 8, then the trims that S0 names. Each rung reads
the teacher on the same 12 heaps of 50 as its base, paired, one architecture per table. A rung
passes on three bars. Its share in the correct box is at least the base's minus 0.01. Its clamp
fraction is at most 0.005. Its `lifted_lost` cycles are at most the base's plus 1 per 100 cycles.
The fastest passing rung is adopted.

The pace ladder, Machine B x86, run 20301000 + i:

| pace | correct box | s per correct stow | clamp | `lifted_lost` per 100 cycles | verdict |
| --- | --- | --- | --- | --- | --- |
| 4.0 | 570 of 600, 0.950 | 12.85 | 0 | 1.53 | base |
| 5.0 | 570 of 600, 0.950 | 13.12 | 0 | 1.03 | passes, slower than the base |
| 6.0 | 560 of 600, 0.933 | 12.21 | 0 | 1.72 | fails the share |
| 8.0 | 562 of 600, 0.937 | 11.62 | 0 | 1.38 | fails the share |

No pace is adopted. The walk is not the limit.

The trims at pace 4.0, on the same heaps:

| trim | x86, Machine B | aarch64 | verdict |
| --- | --- | --- | --- |
| base | 570 of 600 at 12.85 s | 571 of 600 at 13.32 s | |
| `--descend_speed 2.0` | 570 at 10.96 s, passes | 562 at 11.93 s, fails the share | |
| `--descend_speed 3.0` | 560 at 11.01 s, fails the share | 570 at 11.39 s, passes | |
| `--lift_hold_steps 10` | 572 at 12.54 s, passes | | |
| `--release_steps 10` | 572 at 12.53 s, passes | | |
| `--close_steps 15` | 568 at 14.57 s, slower | | not adopted |
| comb3: d3 + l10 + r10 | 569 at 10.30 s, passes | 565 at 10.55 s, passes at the bar | adopted for the teacher |

Single trims flip between the architectures, so the bars sit near the noise of 12 heaps. comb3,
`--descend_speed 3.0 --lift_hold_steps 10 --release_steps 10`, passes on both and is adopted
(12:40).

### 6.3 S2, the serving waits

Rule R7: each serving trim is a switch with a paired row on E2, seed 20301500 draws 0 to 5. A trim
needs two bars for adoption. Its share in the correct box is at least the base's minus 0.01, and its
seconds per correct stow are lower. The rows are in `tables/speed/s2_table.txt`. This report
computes the Wilson intervals.

| row | correct box | Wilson 95 % | s per correct stow | an hour | s a cycle | cycles per correct stow |
| --- | --- | --- | --- | --- | --- | --- |
| f_s_s2_base | 1328 of 1500, 0.885 | 0.868 to 0.900 | 21.78 | 165 | 17.14 | 1.271 |
| f_s_s2_lb450, `--lift_budget 450` | 1270 of 1500, 0.847 | 0.828 to 0.864 | 20.15 | 179 | 13.80 | 1.460 |

`--lift_budget 450` is not adopted. The share falls 0.039. 532 cycles ended on the lift budget, and
each cut books a failure at the target's position. Both S2 rows and the S0 policy census ran without
`--snap_closed`.

### 6.4 The recorder recipe

The S1 rows read the teacher as the read client drives it. The waves record with the recorder
recipe, which adds the hover jitter band of 0.035 to 0.080, the perceived noise and the per-episode
draws. That recipe reads 16.38 s per stow on the S1 heaps, against 12.85 s for the read teacher
(12:40). A wave on it teaches about 220 stows an hour. So track S moved to the recipe's own pace.
The target was about 11 s per stow at a share within 0.01 of the recipe's.

The recorder census (14:15, aarch64, cameras off) found the extra steps. The recipe runs 467 steps a
cycle against the read teacher's 395:

- DESCEND settle, 28 steps more. The joint noise moves the pick point about 0.1 mm a step, and
  DESCEND's stall test waits for 0.05 mm over 12 steps.
- DESCEND walk, 24 more, and HOVER, 15 more, from the hover jitter and the gain draw of 0.7 to 1.0.
- More LIFT retries, 137.

The idle tail after DONE is 46 % of the recorder's simulated steps. Those steps are not recorded,
but they cost wall time.

The recorder rows on the S1 heaps, cameras off, with `--end_at_done`, pooled over x86 and aarch64
(15:33, `tables/speed/recorder_table.txt`):

| row | recipe | aarch64 | x86 | pooled |
| --- | --- | --- | --- | --- |
| M0 | the MS7e wave recipe | 564 of 600 at 16.43 s | 570 of 600 at 16.92 s | 1134 of 1200, 0.945, at 16.68 s |
| M1 | M0 + `--net_stall` | 565 at 15.38 s | 569 at 15.05 s | 1134, 0.945, at 15.22 s |
| M3 | M0 + comb3 + `--net_stall --close_steps 25 --close_guard` | 560 at 11.23 s | 555 at 12.17 s | 1115, 0.929, at 11.70 s |
| M5 | M0 + comb3 + `--landing_steps 15 --close_steps 25 --close_guard` | 554 at 11.39 s | 566 at 11.08 s | 1120, 0.933, at 11.24 s |

The seconds are per delivered part. The decision (15:33): the wave recipe is M5 with
`--end_at_done`. It keeps the joint noise, the gain draw, the hover draws, the jitter band and the
perceived knobs. It costs 0.012 of the teacher's share and saves 33 % of its time. `--end_at_done`
ends every episode once all teachers are DONE. Its exports are byte-equal on a test heap, with about
42 % fewer simulated steps a wave process.

Not adopted: `--quiet_landing`, `--gain_min 1.0`, `--lift_budget` and `--transport_speed 5.0` in the
recorder (0.912). With the noise off and the trims, the recorder reads 9.62 s, but that drops the
Datagen draws, so the ledger does not propose it.

Open at 15:33: the policy's own waits, 58 steps after its close and 45 extra on the approach, need
the retrain to move.

## 7. The teacher reference read

Rule R9 (MS7f design section 9.2): the teacher reads 6 train heaps of 150 on the full gate
configuration, less `--perceived`. The bars are at least 0.95 in a box of the part's category and at
most 10 s per correct stow. If it misses, a census and one round of teacher levers follow, at most
half a day. The corpus records in every case.

The read ran at 077c7f0 through the eval path without cameras, so the camera clip defect does not
touch it. Its flags (`f/w1/ref_flags.txt`) are the R4 H1 scene plus `--families
round,ldraw,flat,joint,minifig`, `--book_miss` and the M5 trims. Heaps 0 to 2 read on Machine A
and 3 to 5 on Machine B. This report computes the Wilson interval.

| item | value |
| --- | --- |
| TAG | f_ref |
| heaps in the verdict | 5 of 6 |
| correct box | 641 of 750, 0.855 |
| Wilson 95 % | 0.828 to 0.878 |
| seconds per correct stow | 11.27 |
| cycles per correct stow | 1.062 |
| per heap | 136, 121, 129, 129, 126 of 150 |
| outcome | both bars miss (0.95, 10 s) |

The verdict stays on five heaps. The sixth heap's payload exists (`reads/f_ref`, eval_005), but the
ledger never logged it (ledger note, 05:42).

The 17:45 entry counts 106 parts left. The ledger note of 05:42 corrects that to 750 - 641 = 109. Of
these, 70 lie in the tray and were never targeted. The R4 H1 arm had 37 such parts over 6 heaps.

The census (`tables/ref/README.txt`) pre-registered its rows at 17:51 on the H diagnostic seeds. Its
rule: a factor is a cause of the R9 miss on one of two results. Without the factor, the pooled share
rises by at least 0.01, or the seconds per correct stow fall by at least 0.5 s. The owner stopped
the census agent before 19:27, and two rows finished on heaps 1 to 3 (19:29):

| row | flags | correct box | s per correct stow | tray parts left | failure bookings |
| --- | --- | --- | --- | --- | --- |
| A | the reference flags | 382 of 450, 0.849 | about 11.1 | 57 | 58 |
| B | A without `--book_miss` | 422 of 450, 0.938 | about 10.6 | 23 | 13 |

The cause: `--book_miss` books a miss on the target's 5 cm failure-memory cell whenever a cycle
stows another part. Three bookings exclude the whole cell, so the selector never reaches the parts
in it. These are the census's never-targeted tray parts. The decision (19:29): `--book_miss` leaves
every teacher read. The R9 verdict stays a miss as pre-registered. The recorder never ran the
switch, so the waves are not affected. Without it, the teacher reads near the R9 bars, 0.938 at 10.6
s on three diagnostic heaps. No teacher lever round followed, for time. Rows C, D and E did not
finish.

## 8. The corpus

### 8.1 The first launch of wave 1

Rule R10 (MS7f design section 9.3): a wave is 40 train heaps of 150, split over the hosts, with
`--wall_band_cont` on. Every video is decode-checked before training. Every wave and checkpoint gets
a second copy on another host. The MS7e waves do not enter the corpus.

The ledger pre-registered wave 1 at 15:56. Each process records one heap, with run 20304000 + p and
layout 20304100 + p. It runs `--episode_length_s 3030` and the recorder's cycle budget of 195, with
cameras on. The recorder flags (`f/w1/flags.txt`) are the MS7e wave set and the two new families.
They add `--categories --deep_tray --box_empty`, the M5 trims, `--end_at_done` and the hover band of
0.035 to 0.080. Wave 1 ran on x86 only, because the deep tray runs about three times slower per step
on Machine C.

Wave 1 started at 16:01 at 077c7f0, Machine A with p 0 to 17 and Machine B with p 18 to 39. It stopped
at 17:04 after 55 minutes on two blockers:

1. The first two shards to end, Machine A proc_13 and Machine B proc_26, were abandoned by the rig's joint
   watchdog after 10 and 5 cycles. They stowed 4 and 2 of 150 and exported nothing. In the
   recorder an abandon ends the episode, and the export keeps only concluded episodes. So one trip
   lost a whole 150-part heap. On 50-part heaps a trip cost at most 50 parts.
2. At K over 50, the scene spawned part i at x = 0.04 i. MuJoCo's `stat.extent` grew to 7.22 m at
   K 150, and the camera near clip, znear times the extent, reached 72 mm. The jaws vanished from
   every wrist frame (17:43). The stopped frames were wrong in any case.

The fixes of section 3.5 removed both blockers.

### 8.2 Wave 1

Wave 1 relaunched at 19:27 at 4464b79, with the same seeds and `--abandon_cycle` added. Machine A
recorded p 0 to 19 and Machine B p 20 to 39.

| host | processes | delivered | cycles per delivered part | s per delivered part | cycle episodes |
| --- | --- | --- | --- | --- | --- |
| Machine A | p 0 to 19, 20 | 2583 of 3000, 0.861 | 1.172 | 16.23 | 2600 |
| Machine B | p 20 to 39, 20 | 2633 of 3000, 0.878 | 1.147 | | 2604 |
| wave 1 | 40 | 5216 of 6000, 0.869 | | | 5204 |

Machine A's cycle episodes are 5204 less 2604. Machine A's share recorded 0 abandons. Machine B's share
left 39 parts out of reach and ran 185 rakes. Its attempts by family: brick 990 of 1163, joint 295
of 350, LDraw 328 of 379 and minifig 250 of 305. Plate read 174 of 236, round 164 of 183, slope 209
of 226 and tile 139 of 179. Machine A's share ran about two hours behind, because orphans of stopped
agents held seven of its cores until 20:57.

The merge (23:16) gave f/w1/dataset with MERGE_OK 5204 cycle episodes and 1,790,156 frames. The
decode check gave CHECK_OK 83 videos.

### 8.3 Wave 2

Wave 2 records the wave 1 recipe at 4464b79 (`f/w2/flags.txt`), with run 20305000 + p and layout
20305100 + p. Machine B recorded p 20 to 39 from 22:02 and then p 12 to 19. Machine C recorded p
0 to 11, its cap of 12 processes. Its physics differ from x86, as in MS7e.

| host | processes | delivered | cycles per delivered part | s per delivered part | cycle episodes |
| --- | --- | --- | --- | --- | --- |
| Machine C | p 0 to 11, 12 | 1574 of 1800, 0.874 | 1.130 | 13.91 | 1570 |
| Machine B | p 12 to 39, 28 | 3581 of 4200, 0.853 | 1.167 | 14.81 | 3610 |
| wave 2 | 40 | 5155 of 6000, 0.859 | | | 5180 |

Every episode ends on its time-out, as in wave 1.

Machine B's disk filled at 01:48 (984 GB, 0 free), and its conversion of p 12 to 39 failed on ENOSPC in
every lane. Freeing MS7f intermediates on Machine B needed the owner. The 28 shards (91 GB) moved over
the LAN to Machine C in about 13 minutes and converted there (CONVERT_OK 3610 cycle
episodes, 03:46). Both wave 2 datasets then shipped to Machine A. The owner then approved two
cleanups:

| time | deleted on Machine B | free after |
| --- | --- | --- |
| 05:42 | the failed wave 2 conversion (7.5 GB) and wave 1's conversion lane copies (6.2 GB). Also deleted: the heap track's eight w1fix2 and w1fix3 smoke shards (22 GB). The merged dataset_b and the smokes' logs, meta and read files stay. | 35 GB |
| 05:55 | on the owner's request, the MS6 raw shards, conversion lanes, a smoke run and the datasets that Machine A also holds (187 GB) | 222 GB |

At 05:42 the read worktree 7f-r2 (03a5cd8) synced to Machine B as `b_fr2`, and model v6a was copied
there. So Machine B took lanes of candidate 1's checked read again (section 9.6).

The merge for retrain F2 (05:19) joins f/w1/dataset (5204) and both wave 2 halves (5180) into
f/w12/dataset: MERGE_OK 10,384 cycle episodes and 3,490,315 frames, CHECK_OK 164 videos.

### 8.4 Wave 3: deviation D17

Deviation D17 (21:34, section 9.11) adds wave 3, on-policy data for the missed grasps. The policy,
in candidate 3's configuration, drives 150-part deep-tray heaps, and the teacher takes over failed
cycles. Only the teacher's takeover frames are recorded, the DAgger scheme of MS4 and MS6. The seeds
are run 20308000 + p and layout 20308100 + p, on the train split. Wave 3 is not a third wave of the
wave 1 recipe, which D11 ruled out (section 9.8).

A subagent built the recorder on `feat/ms7f-dagger` 6abc2c4, on top of `feat/ms7f-pgs` (22:16).
`scripts/mj_eval_heap.py --dagger_dir` records the teacher's takeovers of a served heap. The
triggers are `no_close`, `close_far`, `closed_no_pinch`, `workspace`, `held_no_rise` and
`half_budget_no_lift`. A declined takeover works the teacher's own first-ranked part. A test pins
the off wiring, and 1572 tests pass. A smoke on K = 50 with 30 cycles gave 13 takeovers, all stowed,
and 3099 frames. It converted with CONVERT_OK 13 episodes, with features equal to f/w12/dataset.

The D18 amendment of 22:16 keeps the wave on Newton, because the policies and corpora are Newton's.
Job f_wave3_chain was to start wave 3 at the end of f_chk_F3: Machine A with p 0 to 44 on 14 lanes
and Machine B with p 45 to 89 on 10 lanes, 5 h each. Machine A's half started at 23:26 (unit
ms7f-wave3-Machine A, out f/w3/shards), while candidate 3's last Machine B heap still read. The waiting
chain was stopped before it started anything. The Machine B half started at 23:30, after f_chk_F3 ended.

Wave 3 ended at 06:17 (F2 150k with candidate 3's flags driving, Newton):

| host | heaps | boxed correct with the teacher's takeovers | takeovers | converted takeover episodes |
| --- | --- | --- | --- | --- |
| Machine A | 23, p 0 to 22 | 2881 of 3450 | 1357 | 1784, 5.5 GB |
| Machine B | 20, p 45 to 64 | 2476 of 3000 | 1090 | 1497, 4.9 GB |

This report computes the sums: 43 heaps, 5357 of 6450 boxed correct, 0.831, and 2447 takeovers.
Machine A pulled Machine B's half. The merge for retrain F3 (06:40) joins f/w12/dataset and both wave 3
halves into f/w123/dataset: MERGE_OK 13,665 episodes and 5,110,772 frames, CHECK_OK 235 videos.

## 9. The retrains and the reads

### 9.1 Retrain F1

Rule R11: retrain F1 warm starts from train_E2 150k on wave 1 and trains 200k steps with
`bin/train_warm.sh`, a checkpoint every 50k. It ran on Machine A's RTX 5090 from 23:21 at about 29
steps/s and ended TRAIN_OK at 01:03 (f/train_F1). The four checkpoints got second copies on Machine B
(01:21).

### 9.2 The screens of retrain F1

The ledger pre-registered the screens at 23:21. Each checkpoint reads seed 20307000, draws 0 to 3, 4
heaps of 150, one env each, through `read_f.sh` at d6f893d with model v5. The most correct stows
win, and a tie goes to the higher throughput. The first 50k screen ended READ_FAIL before any lane
started. A detached job had no ssh agent, so Machine A's self-ssh alias was refused. It read again
as f_F1_050000_scr2 on a local alias (23:47).

The screens (04:35, each `gate_results.json`). This report computes the cycles per correct stow:

| step | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box | cycles per correct stow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50k | 353 of 600, 0.588 | 0.549 to 0.627 | 714 | 396,117 | 96.2 | 82 of 151, 0.543 | 112 | 2.023 |
| 100k | 425 of 600, 0.708 | 0.671 to 0.743 | 685 | 349,191 | 131.4 | 99 of 151, 0.656 | 119 | 1.612 |
| 150k | 421 of 600, 0.702 | 0.664 to 0.737 | 664 | 318,985 | 142.5 | 99 of 151, 0.656 | 112 | 1.577 |
| 200k | 415 of 600, 0.692 | 0.654 to 0.727 | 689 | 352,969 | 127.0 | 91 of 151, 0.603 | 129 | 1.660 |

By category at 100k: brick 155 of 202, special 100 of 148, flat 56 of 86, round 29 of 40, joint 44
of 71 and minifig 41 of 53.

By rule R11, F1 100k has the most correct stows, so it is candidate 1. Every screen is far under
every bar: 0.92, 300 an hour and a held-out row of 0.85. One heap read in about 3 hours on
Machine A with 16 draws in parallel.

### 9.3 The warp screens: deviation D7

The ledger's reason (02:57): the policy copies the corpus's pace (section 6.1), and wave 1's teacher
ran 13.85 s a cycle. So no retrain on waves 1 and 2 reaches 300 an hour unless the share rises and
the cycle shortens together. Deviation D7 (02:57, before any warp number) adds a serving lever with
no retrain. `--action_warp X` plays each predicted 50-step chunk X times faster by linear
interpolation in time. X = 1.0 is byte-equal to the code before it.

The build (03:20, feat/ms7f-warp 03a5cd8 on d6f893d): the policy server resamples each chunk at t =
0, X, 2X and so on. `policy_id` gains `+warp<X>` only off 1.0. The supervisor forwards the switch,
and its cache splits on it. The server refuses it with temporal ensembling and outside [1.0, 3.0].
129 lerobot_stack tests and 1552 repo tests pass. The warp served as built (`policy_id` `+warp1.25`,
07:38).

The pre-registered rule: candidate 1 reads at X = 1.25 and at X = 1.5. Both screens use seed
20307000, draws 0 to 3, the four heaps of its own screen at X = 1.0. The TAGs are
f_F1_100000_w125_scr and f_F1_100000_w150_scr. A value of X qualifies with a correct-stow share at
most 0.02 under the X = 1.0 screen's. The qualified X with the most correct stows an hour wins, and
X = 1.0 always qualifies. One binomial standard error on 600 parts is about 0.015. The screens keep
model v5.

The extension (10:38, before any number) adds X = 2.0 on the same heaps, TAG f_F1_100000_w200_scr. X
= 2.0 replaces 1.5 for every read that has not started on two conditions. Its correct stows an hour
beat X = 1.5's 145.5, and its share is at least 0.708 - 0.02 = 0.688.

The screens ran on Machine A (08:49 and 13:08):

| X | TAG | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | f_F1_100000_scr | 425 of 600, 0.708 | 0.671 to 0.743 | 685 | 349,191 | 131.4 | 99 of 151, 0.656 | 119 |
| 1.25 | f_F1_100000_w125_scr | 406 of 600, 0.677 | 0.638 to 0.713 | 682 | 321,538 | 136.4 | 97 of 151, 0.642 | 133 |
| 1.5 | f_F1_100000_w150_scr | 428 of 600, 0.713 | 0.676 to 0.748 | 702 | 317,657 | 145.5 | 91 of 151, 0.603 | 114 |
| 2.0 | f_F1_100000_w200_scr | 361 of 600, 0.602 | 0.562 to 0.640 | 644 | 326,339 | 119.5 | 78 of 151 | 150 |

D7 verdict (08:49): X = 1.5 wins at 145.5 an hour, 11 % over X = 1.0, with a share of 0.713 within
0.02 of 0.708. X = 1.25 is out, because its share is 0.031 under. The extension verdict (13:08): X =
2.0 is rejected, because its rate and its share are both under X = 1.5's. X = 1.5 stays for every
later read.

### 9.4 The passenger-drop screen: deviation D9

Passengers cost about 9 % of the screen parts (section 10.6). Deviation D9 (04:07, before any
number) gives `--passenger_drop` one screen: candidate 1 at X = 1.0 with `--passenger_drop`, seed
20307000, draws 0 to 3, TAG f_F1_100000_pd_scr, model v5. It has two bars against the candidate's
own screen. Its correct stows are at least that screen's plus 6. Its correct stows an hour are at
least 0.97 of that screen's. A screen that holds both bars is adopted, and the switch then joins
every later read.

| row | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f_F1_100000_scr, the base | 425 of 600, 0.708 | 0.671 to 0.743 | 685 | 349,191 | 131.4 | 99 of 151 | 119 |
| f_F1_100000_pd_scr | 425 of 600, 0.708 | 0.671 to 0.743 | 741 | 385,971 | 118.9 | 92 of 151, 0.609 | 98 |
| bar | at least 431 | | | | at least 127.5 | | |

This report computes the bar from the base's 425 and 131.4. D9 verdict (08:49): `--passenger_drop`
is not adopted. Its 425 correct stows miss 431, and its 118.9 an hour is 0.90 of the base's rate. It
cuts the wrong-box parts from 119 to 98 and spends the gain on 56 more cycles.

### 9.5 The v6a diagnostic screen

The ledger pre-registered a diagnostic screen at 04:37: candidate 1 at X = 1.0 with model v6a, seed
20307000, draws 0 to 3, TAG f_F1_100000_v6_scr. No decision rides on it, because C1 already adopted
v6a. Paired with f_F1_100000_scr (v5), it measures the classifier lever inside the loop.

| row | model | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_F1_100000_scr | v5 | 425 of 600, 0.708 | 0.671 to 0.743 | 685 | 349,191 | 131.4 | 99 of 151 | 119 |
| f_F1_100000_v6_scr | v6a | 420 of 600, 0.700 | 0.662 to 0.735 | 690 | 342,213 | 132.5 | 97 of 151, 0.642 | 115 |

v6a reads 420 against v5's 425. Special rises from 100 of 148 to 108 of 148, and minifig falls from
41 of 53 to 34 of 53. C1's gain does not show inside the loop at this size of 4 heaps (08:49).

### 9.6 The Machine B lanes of the checked read

The checked read changed its lanes twice before it started, each time with the same rules and read:

| time | chain | lanes | why |
| --- | --- | --- | --- |
| 05:20 | f_chk_F1 | `a_fr2l:20` | Machine B's disk was full (D8). |
| 05:42 | f_chk_F1b | `a_fr2l:12` and `b_fr2:8` | The owner approved the Machine B cleanup (section 8.3). f_chk_F1 stopped before it read anything. |
| 08:17 | f_chk_F1c | `a_fr2l:10` and `b_fr2:10` | Machine B sat idle from 05:45, because its only queued work was the checked read. f_chk_F1b stopped before it read anything. |

### 9.7 Candidate 1's checked read

The ledger pre-registered the checked read at 02:20. Later entries amended it before it started:

- 02:20: candidate 1, seed 20307100, draws 0 to 19, 3000 parts, TAG f_F1_chk. It passes with at
  least 2760 correct stows, at least 300 correct stows an hour and a held-out row of at least 0.85.
- 03:20, deviation D8: it starts after the warp screens and reads at D7's winning X.
- 04:11: model v6a serves it.
- 05:20 to 08:17: the lanes of section 9.6.

f_F1_chk started at 08:36 with `--action_warp 1.5` and model v6a, without `--passenger_drop`. Machine B
finished its 10 heaps at 11:06 and Machine A its 10 at 11:54. D12 (10:38) set the staged bars
while it ran. The read keeps its protocol and its supervisor, whose token judges the stretch row.

| item | value |
| --- | --- |
| TAG | f_F1_chk |
| checkpoint | F1 100k |
| commit, classifier, warp | 03a5cd8, v6a, `--action_warp 1.5` |
| correct stows of 3000 | 2043, 0.681 |
| Wilson 95 % | 0.664 to 0.697 |
| cycles | 3408 |
| steps | 1,566,223 |
| correct stows an hour | 140.9 |
| held-out row | 471 of 785, 0.600, Wilson 0.565 to 0.634 |
| wrong-box parts | 621, from 318 `wrong_box` cycles, 291 of them the classifier's |
| cycles per correct stow, no bar | 1.668 |
| staged bars (D12) | FAIL on all three: 2043 against 2550, 140.9 against 200, 0.600 against 0.80 |
| stretch row | FAIL on all three: 2043 against 2760, 140.9 against 300, 0.600 against 0.85 |

By category: brick 784 of 1063, special 415 of 645, flat 267 of 419, round 148 of 199, joint 227 of
378 and minifig 202 of 296.

The read's `gate_results.json` gives the rows by family and by box count. This report computes the
fractions:

| family | brick | slope | round | LDraw | plate | tile | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 784/1063 | 175/284 | 148/199 | 240/361 | 167/256 | 100/163 | 227/378 | 202/296 |
| fraction | 0.738 | 0.616 | 0.744 | 0.665 | 0.652 | 0.613 | 0.601 | 0.682 |

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 97/150 | 499/750 | 537/750 | 492/750 | 418/600 |
| fraction | 0.647 | 0.665 | 0.716 | 0.656 | 0.697 |

The same file counts 13 box emptying events and no tray refill. Its clamp fraction is 0.0034.

The decomposition of the read's steps (11:54), by cycle outcome:

| cycle outcome | cycles | share of the steps |
| --- | --- | --- |
| `close_miss` | 624 | 0.295 |
| stowed | 1532 | 0.239 |
| `lifted_lost` | 535 | 0.194 |
| `held_no_lift` | 212 | 0.084 |
| `wrong_box` | 318 | 0.075 |
| `carrying` | 95 | 0.063 |
| `no_open` | 79 | 0.042 |

A stowed cycle takes 244 steps, 8.2 s, under the warp. If every cycle stowed, that pace gives 442 an
hour. `lifted_lost` ends 15.7 % of the cycles, against 9.9 % at X = 1.0 on the screen, so the faster
carry drops more parts. 1203 of the 3408 cycles end on the 800-step budget.

The verdict (11:54): f_F1_chk fails the staged bars on all three and the stretch row. Rule R11's
trigger, as D12 restated it, fires: candidate 2 gets a checked read.

### 9.8 Retrain F2 and its screens: deviations D6, D10, D11 and D13

Rule R11 set a trigger for retrain F2: candidate 1's checked read stows under 2790 of 3000 (0.93),
or under 320 correct stows an hour. D12 restates it as a miss of any staged bar. On the trigger, F2
warm starts from candidate 1 on waves 1 and 2. At most two retrains run.

Deviation D6 (02:20, before any screen number): retrain F2 starts at the end of the F1 screens. It
does not wait for candidate 1's checked read, because the deep-tray reads take hours and the GPU
sits idle. Its checkpoints count for the gate only on R11's trigger.

Retrain F2 started at 05:04 on Machine A from F1 100k on f/w12/dataset (section 8.3). It ended
TRAIN_OK at 07:38 (f/train_F2): 200k steps on 10,384 episodes, checkpoints 50k to 200k on
Machine A. The D10 job copies each checkpoint to Machine C before its screen
(`bin/screens_F2_aarch64.sh`). The ledger logs no other second copy.

Deviation D10 (05:19, before any F2 number): F2's screens run on Machine C as soon as each
checkpoint exists. Each reads seed 20307000, draws 0 to 3, through `read_f.sh` at d6f893d with model
v6a at X = 1.0 on `c_fr1:4`, TAGs f_F2_<step>_scr. They read the aarch64 physics with a CPU
policy, so they compare only with each other. The 50k screen started at 05:54, 100k at 06:29 and
150k at 07:05.

Deviation D11 (08:20, before any F2 number): the D10 screens differ from F1's screens in host and
classifier, so they cannot say whether wave 2 helped. On Machine B's spare cores, F2 100k and F2 200k
read the screen protocol on x86 with model v6a at X = 1.0, paired with f_F1_100000_v6_scr. D11 is a
diagnostic. Its rule: more data helps on a gain of at least 18 correct stows, 3 % of 600, by the
better F2 row over F1's. That rule decides whether a third wave records.

| checkpoint | TAG, host | correct stows | Wilson 95 % | cycles | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 100k | f_F1_100000_v6_scr, Machine A | 420 of 600, 0.700 | 0.662 to 0.735 | 690 | 132.5 | 97 of 151 | 115 |
| F2 100k | f_F2_100000_x86_scr, Machine B | 435 of 600, 0.725 | 0.688 to 0.759 | 681 | 140.8 | 91 of 151 | 102 |
| F2 200k | f_F2_200000_x86_scr, Machine B | 407 of 600, 0.678 | 0.640 to 0.714 | 683 | 121.3 | 92 of 151 | 117 |

D11 verdict (11:06): the better F2 row beats F1's by 15 correct stows, under the 18 that the rule
asks. Doubling the corpus with the same recipe buys at most a few points, so no third wave of the
same recipe records. As with F1, the 100k checkpoint reads better than the 200k one.

Deviation D13 (11:53, before any F2 50k or 150k number): the aarch64 F2 screens run about four times
slower than x86, about 6 h a heap. R11's pick of candidate 2 moves to x86 screens with model v6a at
X = 1.0. F2 50k and 150k read on Machine B (f_F2_<step>_x86_scr, `b_fr2:4` each). Candidate 2 is the
checkpoint with the most correct stows of the four, a tie to the higher rate. The D10 screens keep
running as a record.

The D13 screens ended at 15:40 (x86, model v6a, X = 1.0, seed 20307000, draws 0 to 3):

| checkpoint | TAG | correct stows | Wilson 95 % | cycles | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2 50k | f_F2_050000_x86_scr | 286 of 600, 0.477 | 0.437 to 0.517 | 780 | 66.4 | 67 of 151 | 81 |
| F2 100k | f_F2_100000_x86_scr | 435 of 600, 0.725 | 0.688 to 0.759 | 681 | 140.8 | 91 of 151 | 102 |
| F2 150k | f_F2_150000_x86_scr | 438 of 600, 0.730 | 0.693 to 0.764 | 693 | 134.7 | 106 of 151 | 109 |
| F2 200k | f_F2_200000_x86_scr | 407 of 600, 0.678 | 0.640 to 0.714 | 683 | 121.3 | 92 of 151 | 117 |

D13 pick: candidate 2 is F2 150k, with 438 correct stows against 435 for F2 100k. The ledger reads
F2 50k as the warm start's early transient.

The D10 record ended at 15:43 (aarch64, CPU policy, model v6a, X = 1.0, seed 20307000, draws 0 to 3,
no decision). The ledger sums the four payloads per checkpoint (18:30):

| checkpoint | correct stows | correct stows an hour |
| --- | --- | --- |
| F2 50k | 332 of 600 | 78.4 |
| F2 100k | 412 of 600 | 116.3 |
| F2 150k | 437 of 600 | 133.9 |
| F2 200k | 384 of 600 | 104.7 |

The x86 order of D13 agrees: F2 150k reads best on both architectures.

### 9.9 Two loop levers: deviation D14

At 11:07 two subagents took one lever each for the share bar of D12. Both branches start from
feat/ms7f-warp 03a5cd8. Their switches default off, read byte-equal off, and the recorder refuses
them. They reported at 12:54.

`--missed_lift` (feat/ms7f-riders 5715401, 1562 tests). Most "riders" are the cycle's own part in a
cycle that the hold detector never read as lifted. Such a cycle gets no wrist read, so the box stays
at the overhead read. It gets no release detection either, so it runs to the 800-step budget with a
stale row while its part is already boxed. On f_F1_100000_scr, 116 of 150 uncredited boxed parts
were another cycle's target or held part. About 36 over the 4 heaps were the part of no cycle, and
27 of those ended in a wrong box. These cycles are 22 to 25 a heap and 20 to 24 % of the steps. On
10 heaps of f_F1_chk they hold 198,563 of 828,292 steps. The switch adds three rules:

- A closed gripper command that carries the pick point 5 cm above its lowest closed point and off
  the tray counts as a lift. The wrist read and the vote then run.
- The cycle ends at a jaw gap 1 cm wider than the carry's narrowest gap. Over a box the part counts
  as placed, elsewhere as dropped.
- The next selection books a failure on a detection that it still sees within 15 mm of the target.

The subagent expects about 12,000 fewer steps a heap, 15 to 20 % more rate, and 2 to 5 fewer
wrong-box parts.

`--lift_view_steps N` (feat/ms7f-cls-loop 36b069b, 1554 tests). The in-loop wrist read is right on
1297 of 1584 target lifts, 0.819, against C1's 0.903. Over each heap's first 40 cycles it is right
on 0.771. At the detector's lift the part still rises, 36 mm in 3 steps. The switch reads the wrist
view N steps after the lift, and the carry view stays at 15 steps. On one instrumented 50-part
episode, the served rule reads 21 of 23 at N = 0 and 22 of 23 at N = 6. The subagent expects 3 to 4
fewer wrong-box parts a heap. Two causes stay open: neighbour grasps (the held part is not the
target, 8.4 % of lifts, final class right on 0.517) and overhead misreads in the early pile.

feat/ms7f-levers2 496f857 joins both (13:07, 1565 tests). It carries one fix outside the merge
conflicts: with both on, view 2 also reads on a carry that `--missed_lift` found. The read worktree
is 7f-r3, with the aliases `a_fr3l` and `b_fr3`.

Deviation D14 (13:07, before any number) screens the levers on candidate 1 at X = 1.5 with model
v6a. The screens read seed 20307000, draws 0 to 3, in job f_lever_screens. The rules against
f_L_base_scr:

- `--missed_lift` needs at least 1.05 times the base's correct stows an hour, and at least the
  base's correct stows minus 6. With both, it is adopted.
- `--lift_view_steps 6` needs at least the base's correct stows plus 6, and at least 0.97 times the
  base's rate. With both, it is adopted.
- If both are adopted, both serve. The exception: f_L_both_scr reads more than 6 correct stows or 3
  % of rate under the better single. Then the single with the higher correct stows an hour serves.

The D14 screens ended at 15:46 (candidate 1, X = 1.5, model v6a, seed 20307000, draws 0 to 3):

| screen | TAG, host | switches | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | f_L_base_scr, Machine A | none | 409 of 600, 0.682 | 0.643 to 0.718 | 691 | 322,533 | 137.0 | 97 of 151 | 136 |
| missed lift | f_L_ml_scr, Machine A | `--missed_lift` | 418 of 600, 0.697 | 0.659 to 0.732 | 709 | 214,486 | 210.5 | 95 of 151 | 110 |
| lift view | f_L_lv6_scr, Machine B | `--lift_view_steps 6` | 431 of 600, 0.718 | 0.681 to 0.753 | 703 | 335,091 | 138.9 | 100 of 151 | 111 |
| both | f_L_both_scr, Machine B | both | 421 of 600, 0.702 | 0.664 to 0.737 | 709 | 223,454 | 203.5 | 94 of 151 | 101 |

D14 verdict (15:46):

- `--missed_lift` is adopted: 418 is at least 409 - 6 = 403, and 210.5 is at least 1.05 x 137.0 =
  143.9.
- `--lift_view_steps 6` is adopted alone: 431 is at least 409 + 6 = 415, and 138.9 is at least 0.97
  x 137.0 = 132.9.
- Together, the rule keeps only the better single. The pair reads 421 correct stows, within 6 of
  418, but 203.5 an hour, under 0.97 x 210.5 = 204.2. So `--missed_lift` alone serves.

`--missed_lift` cuts the screen's steps from 322,533 to 214,486 for 9 more correct stows. This
report computes the seconds a cycle: 15.6 s on the base and 10.1 s with `--missed_lift`.

### 9.10 Candidate 2's checked read

Candidate 2's checked read waits for the D13 pick and the D14 screens (13:07). It reads with the
adopted D14 switches, X = 1.5 from D7 and model v6a, through `read_f2.sh` with the staged bars. It
reads seed 20307100, draws 0 to 19, on Machine A and Machine B. Job f_chk_F2 (`bin/chk_F2.sh`) applies
the D13 and D14 rules as written at 13:07.

f_F2_chk started at 15:45 on `a_fr3l:10` and `b_fr3:10`. Its worktree 7f-r3 sits at
496f857, whose tree equals 0a1b552 on the restacked branch. An early look at 17:28, with no
decision, read 648 of 900, 0.720, at 235.8 an hour on 6 of the 20 heaps. Every episode there ended
no_candidate with about 16 parts a heap unboxed, because the loop excludes a 5 cm cell after
`CELL_FAILURE_CAP` = 3 failures. The read ended at 18:05:

| item | value |
| --- | --- |
| TAG | f_F2_chk |
| checkpoint | F2 150k, candidate 2 |
| commit, classifier, switches | 496f857, v6a, `--action_warp 1.5 --missed_lift` |
| correct stows of 3000 | 2065, 0.688 |
| Wilson 95 % | 0.672 to 0.705 |
| cycles | 3627 |
| steps | 1,179,094 |
| correct stows an hour | 189.1 |
| held-out row | 508 of 785, 0.647, Wilson 0.613 to 0.680 |
| wrong-box parts | 550, from 307 `wrong_box` cycles, 264 of them the classifier's |
| cycles per correct stow, no bar | 1.756 |
| supervisor token | GATE_FAIL at the staged bars |
| staged bars (D12) | FAIL on all three: 2065 against 2550, 189.1 against 200, 0.647 against 0.80 |
| stretch row | FAIL on all three: 2065 against 2760, 189.1 against 300, 0.647 against 0.85 |

The first six heaps read 235.8 an hour, but the rest were slower. Against candidate 1's checked
read, candidate 2 gains 22 correct stows, 48 an hour (34 %) and 0.047 on the held-out row.

By category: brick 802 of 1063, special 415 of 645, flat 278 of 419, round 164 of 199, joint 192 of
378 and minifig 214 of 296. The read's `gate_results.json` gives the rows by family and by box
count. This report computes the fractions:

| family | brick | slope | round | LDraw | plate | tile | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 802/1063 | 177/284 | 164/199 | 238/361 | 174/256 | 104/163 | 192/378 | 214/296 |
| fraction | 0.754 | 0.623 | 0.824 | 0.659 | 0.680 | 0.638 | 0.508 | 0.723 |

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 98/150 | 532/750 | 513/750 | 519/750 | 403/600 |
| fraction | 0.653 | 0.709 | 0.684 | 0.692 | 0.672 |

The same file counts 17 box emptying events and no tray refill. Its clamp fraction is 0.0049.

The decomposition of the read's steps (18:06), by cycle outcome:

| cycle outcome | cycles | share of the steps | steps a cycle |
| --- | --- | --- | --- |
| stowed | 1818 | 0.363 | 235 |
| `close_miss` | 886 | 0.336 | 447 |
| `held_no_lift` | 201 | 0.085 | |
| `lifted_lost` | 309 | 0.076 | |
| `wrong_box` | 307 | 0.061 | |
| `carrying` | 43 | 0.039 | |

A stowed cycle takes 235 steps, 7.9 s. `close_miss` is now the largest sink. `lifted_lost` ends 9 %
of the cycles, against 16 % for candidate 1 without `--missed_lift`. 498 of the 3627 cycles end on
the 800-step budget, against 1203 of 3408 for candidate 1.

The verdict (18:06): f_F2_chk fails all three staged bars and the stretch row. No checked read
passes the staged bars at this point. The D16 screens start at once. Candidate 3 follows only on a
change that D16 or v7 adopts.

### 9.11 Two more switches and candidate 3: deviation D16

The early look at f_F2_chk (17:28) showed two more sinks: the cell exclusion after three failures,
and the classifier's vote. A subagent built two switches on `feat/ms7f-levers3`, both default off
and byte-equal off (17:58, 1561 tests under `MUJOCO_GL=disable`):

| commit | switch | what it does |
| --- | --- | --- |
| fbd45e9 | `--cell_failure_cap N` | The perceived selector's failure memory excludes a position after N failures. The default is 3, and the teacher keeps 3. |
| bcbbe02 | `--carry_votes K` | On a disputed lift, K more wrist reads between the lift view and the carry view join the vote. The vote still decides at lift + 15 steps. |

The read worktree is 7f-r6, with the aliases `a_fr6l` and `b_fr6`.

Deviation D16 (17:58, before any number) screens candidate 2 after f_F2_chk: F2 150k with
`--action_warp 1.5 --missed_lift` and model v6a, seed 20307000, draws 0 to 3, job f_lever3_screens.
f_M_base_scr and f_M_cap5_scr read on Machine A, f_M_cv2_scr and f_M_both_scr on Machine B. The rules
against f_M_base_scr:

- `--cell_failure_cap 5` needs at least the base's correct stows plus 6, and at least 0.90 times the
  base's rate. With both, it is adopted.
- `--carry_votes 2` needs at least the base's correct stows plus 6, and at least 0.97 times the
  base's rate. With both, it is adopted.
- If both are adopted, both serve. The exception: f_M_both_scr reads more than 6 correct stows or 3
  % of rate under the single with more correct stows. Then that single serves alone.

The D16 screens ended at 20:34 (candidate 2 = F2 150k, X = 1.5, `--missed_lift`, model v6a):

| screen | TAG, host | switches | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | f_M_base_scr, Machine A | none | 414 of 600, 0.690 | 0.652 to 0.726 | 706 | 223,635 | 199.9 | 87 of 151 |
| cap 5 | f_M_cap5_scr, Machine A | `--cell_failure_cap 5` | 435 of 600, 0.725 | 0.688 to 0.759 | 762 | 244,829 | 191.9 | 110 of 151 |
| carry votes | f_M_cv2_scr, Machine B | `--carry_votes 2` | 415 of 600, 0.692 | 0.654 to 0.727 | 721 | 226,728 | 197.7 | 93 of 151 |
| both | f_M_both_scr, Machine B | both | 424 of 600, 0.707 | 0.669 to 0.742 | 728 | 227,345 | 201.4 | 95 of 151 |

D16 verdict (20:34):

- `--cell_failure_cap 5` is adopted: 435 is at least 414 + 6 = 420, and 191.9 is at least 0.90 x
  199.9 = 179.9. It also lifts the held-out row from 87 to 110 of 151.
- `--carry_votes 2` is not adopted: 415 is under 420.
- With one switch adopted, the pair rule does not apply.

Candidate 3 is candidate 2 with every switch that D16 adopts. It also takes classifier v7 on a v7
pass of its bar (section 5.8). Candidate 3 gets a checked read only on a difference from candidate
2: seed 20307100, draws 0 to 19, through `read_f2.sh` with the staged bars.

Candidate 3 is candidate 2 with `--cell_failure_cap 5`: F2 150k, `--action_warp 1.5 --missed_lift
--cell_failure_cap 5` and model v6a, because v7 missed (section 5.8). Its checked read f_F3_chk
(`bin/chk_F3.sh`, job f_chk_F3, a user systemd unit) waited for the v7 verdict and started at 20:52.
It reads seed 20307100, draws 0 to 19, through `read_f2.sh` with the staged bars, in worktree 7f-r6
on `a_fr6l:10` and `b_fr6:10`, `PORT_BASE` 15400.

The read ended at 23:29, and the ledger recorded it at 00:52:

| item | value |
| --- | --- |
| TAG | f_F3_chk |
| checkpoint | F2 150k, candidate 3 |
| commit, classifier, switches | bcbbe02, v6a, `--action_warp 1.5 --missed_lift --cell_failure_cap 5` |
| correct stows of 3000 | 2181, 0.727 |
| Wilson 95 % | 0.711 to 0.743 |
| cycles | 3739 |
| steps | 1,242,893 |
| correct stows an hour | 189.5 |
| held-out row | 523 of 785, 0.666, Wilson 0.633 to 0.698 |
| wrong-box parts | 510, from 296 `wrong_box` cycles, 256 of them the classifier's |
| cycles per correct stow, no bar | 1.714 |
| supervisor token | GATE_FAIL at the staged bars |
| staged bars (D12) | FAIL on all three: 2181 against 2550, 189.5 against 200, 0.666 against 0.80 |
| stretch row | FAIL on all three: 2181 against 2760, 189.5 against 300, 0.666 against 0.85 |

It is the best checked read so far. Against candidate 2's checked read, candidate 3 gains 116
correct stows and 0.019 on the held-out row. Its rate stays the same, 189.5 against 189.1 an hour.

By category: brick 837 of 1063, special 431 of 645, flat 289 of 419, round 158 of 199, joint 251 of
378 and minifig 215 of 296. The read's `gate_results.json` gives the rows by family and by box
count. This report computes the fractions:

| family | brick | slope | round | LDraw | plate | tile | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 837/1063 | 186/284 | 158/199 | 245/361 | 179/256 | 110/163 | 251/378 | 215/296 |
| fraction | 0.787 | 0.655 | 0.794 | 0.679 | 0.699 | 0.675 | 0.664 | 0.726 |

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 91/150 | 551/750 | 566/750 | 565/750 | 408/600 |
| fraction | 0.607 | 0.735 | 0.755 | 0.753 | 0.680 |

Against candidate 2, joint gains the most, 192 to 251 of 378. Round loses 6 correct stows, 164 to
158 of 199 parts. The same file counts 16 box emptying events and no tray refill. Its clamp fraction
is 0.0058.

The decomposition of the read's steps (00:52), by cycle outcome:

| cycle outcome | cycles | share of the steps | steps a cycle |
| --- | --- | --- | --- |
| stowed | 1904 | 0.352 | 230 |
| `close_miss` | 942 | 0.340 | 449 |
| `held_no_lift` | 196 | 0.083 | |
| `lifted_lost` | 271 | 0.073 | |
| `wrong_box` | 296 | 0.058 | |
| `carrying` | 59 | 0.051 | |

A stowed cycle takes 230 steps, 7.7 s. `close_miss` stays the largest sink. 562 of the 3739 cycles
end on the 800-step budget, against 498 of 3627 for candidate 2.

The verdict (00:52): f_F3_chk fails all three staged bars (0.85, 200 an hour, held-out 0.80) and the
stretch row. No official read runs (D12).

Deviation D17 (21:34) is an owner decision. The owner was asked at 21:20, with candidate 3's checked
read running and no number of it seen. The campaign continues past candidate 3 with two levers in
parallel through 10-03. A checked read follows on 10-04, and the official read only on a pass:

- (a) Classifier v8, the gate-scene classifier (section 5.9).
- (b) Wave 3, on-policy data for the missed grasps (section 8.4), and retrain F3 on waves 1, 2 and 3
  (section 9.17).

### 9.12 The official read

The official read runs on seed 20307200, 40 heaps of 150, 6000 parts, one shot. Since D12, it reads
the first candidate whose checked read passes the staged bars. It passes with at least 5100 correct
stows, at least 200 correct stows an hour and a held-out row of at least 0.80. The stretch row is
5520, 300 an hour and 0.85. If no checked read passes the staged bars, the official read does not
run, and the milestone fails. Candidates 1, 2 and 3 fail the staged bars on the full pile, so no
official read runs in that scene. Since D19 (06:40), the official read runs in the refill scene,
through `read_f2.sh`. It reads the first checked read in that scene that passes the staged bars
(section 9.18). Candidates 3 and 4 fail the staged share and held-out bars in the refill scene, so
no official read runs yet.

### 9.13 The stretch runs: deviation D15

Rule R13: after the official read, the official candidate runs two episodes of 1000 parts, one env
each. The seeds are run 20307400 + r and layout 20307450 + r, r = 0 and 1. The tray refills from the
episode's pool, full boxes are emptied, and the category map stays fixed for the run.
`scripts/mj_stretch_run.py` runs them. An intervention is an event that the loop cannot resolve:

1. The episode ends with parts in the tray: 1 intervention.
2. A part rests on the desk outside the tray and every box: 1 per part.
3. Ten cycles in a row end with no correct stow: 1, and the loop restarts.

The runs report against four targets with no bar: 0.95 in a box of the category, 400 an hour, 1000
consecutive parts and 5 interventions per 1000 parts.

Deviation D15 (13:39, pre-registered): no checked read passes at 13:39, so no official read and no
R13 run can follow in time. A 1000-part run takes about a day of wall time, because a 150-part heap
takes 1.5 to 3 h. So one stretch run, r = 0 (run 20307400, layout 20307450), starts on Machine A.
It uses candidate 2's checkpoint and the flags of its checked read. The tray refills from the pool
(`--tray_refill`), because the deep tray holds 150 parts. Its numbers are a record of continuous
operation, report-only, not a gate row. If an official read runs, R13's two runs on the official
candidate replace it when time allows.

The stretch prep (14:02) found and fixed one defect. `mj_stretch_run.py` refused `--heap_k 1000`,
because it reuses the heap client's parser and its cap of 150. The fix on `feat/ms7f-levers2`,
fe33c0c, puts the cap on the parser class: the heap client keeps 150, and the stretch parser allows
1000. The smoke and the probe:

| run | setup | result |
| --- | --- | --- |
| smoke | K 60, F2 100k, X = 1.5, `--missed_lift`, `--lift_view_steps 6`, deep tray with refill | STRETCH_OK. 60 served, 49 correct (0.817), 5 in a wrong box. 24,261 steps, 218.1 an hour, one refill, 4 interventions (1 for parts left in the tray, 3 for strays). |
| probe | K 1000 | about 12 steps/s with 950 parked bodies, so a run takes about 10 to 13 h |

The run's worktree is 7f-r4 at fe33c0c, and job f_stretch_r0 (`f/stretch/stretch_chain.sh`) starts
it. The 15:46 entry starts it with candidate 2's checkpoint and flags, next to candidate 2's checked
read. The run went from 15:46 to 04:49, and the ledger recorded it at 06:19:

| item | value |
| --- | --- |
| policy | candidate 2's configuration: F2 150k, `--action_warp 1.5 --missed_lift`, model v6a |
| scene | the deep-tray walls with `--tray_refill`, 50 parts in the tray, refilled at 10 or fewer |
| seeds | run 20307400, layout 20307450 |
| token | STRETCH_OK |
| parts served | 1000 of 1000, complete |
| correct box | 801, 0.801 |
| wrong box | 129 |
| steps | 332,886, 3.08 simulated hours |
| correct stows an hour | 259.9 |
| interventions | 85: 79 strays put back, 6 tray removals, 0 miss streaks. 85 per 1000 parts. |
| events | 23 refills, 20 box emptyings |
| episodes | 6, each ended no_candidate |

| run | policy | correct box | correct stows an hour | consecutive parts | interventions per 1000 parts |
| --- | --- | --- | --- | --- | --- |
| r = 0, D15 | candidate 2, `--action_warp 1.5 --missed_lift`, v6a | 801 of 1000, 0.801 | 259.9 | 1000 | 85 |
| r = 1, R13 | the official candidate, only after an official read | TBD | TBD | TBD | TBD |

Against R13's report-only targets: 0.95, 400 an hour, 1000 consecutive parts (met) and 5
interventions per 1000 parts (85). The same policy reads 0.688 at 189.1 an hour on candidate 2's
checked read on the full 150-part deep pile. With at most 50 parts in the tray, it reads 0.801 at
259.9. This report computes 1230 cycles from the six episodes of `stretch_0.json`: 9.0 s a cycle and
1.536 cycles per correct stow. The owner decided D19 after this run (section 9.18).

TBD: R13's two stretch runs on the official candidate, after an official read.

### 9.14 The control read

Rule R12 (MS7f design section 10.3): E2, train_E2 150k with no retrain, reads 7 heaps of 150, 1050
parts, on the gate configuration. It has no bar and shows what the retrains buy. The ledger
pre-registered it at 00:18: `read_f.sh` at d6f893d with model v5, seed 20307300, draws 0 to 6,
`c_fr1:7` with a CPU policy, TAG f_control_E2. It reads the aarch64 physics.

The control read ended at 05:38:

| item | value |
| --- | --- |
| TAG | f_control_E2 |
| correct stows of 1050 | 768, 0.731 |
| Wilson 95 % | 0.704 to 0.757 |
| cycles | 1074 |
| steps | 548,351 |
| correct stows an hour | 151.3 |
| held-out row | 180 of 263, 0.684 |
| wrong-box parts | 219, 0.209 |
| outcome | no bar, reported |

By category: brick 293 of 360, special 139 of 216, flat 89 of 141, round 62 of 76, joint 75 of 123
and minifig 110 of 134.

The control read uses another seed and another host than the F1 screens, so it does not pair with
them. The ledger reads it at the level of F1's best screen, 0.708 and 131 an hour. So retrain F1 on
wave 1 bought no visible gain over the MS7e policy. This report computes its pace: 17.0 s a cycle
and 1.398 cycles per correct stow.

### 9.15 The owner's video

The owner asked for a video of the policy at work, and a subagent rendered it (12:54). The files are
in `~/data/maker-arm-ms7/f/video/`:

| item | value |
| --- | --- |
| video | `ms7f_F1_100k_w150_v6a_d0.mp4`, H.264, 1280 x 720, 60 fps, 5:10 at 8 times real time, 33.6 MB |
| stills | the first and the last frame, and a 4 x 3 contact sheet |
| policy | candidate 1, F1 100k, at X = 1.5 with model v6a |
| heap | seed 20307000, draw 0, the first heap of the screens, with 4 boxes |
| camera | the owner's third-person camera |
| cut | at about 100 boxed parts, simulated time 41:16 |
| state at the cut | 84 parts in their box, 21 in a wrong box, 148 cycles, 122 correct stows an hour, one box emptying |

The frames show the deep tray with the heap, the four boxes labeled with their categories, and the
arm at work. An overlay prints the simulated time, the running counts and the current target with
its classifier read and its box. It also prints how the last cycle ended, for example placed,
dropped or ended on the budget. The video stops at about 100 boxed parts, so it does not show the
end of the heap.

The owner asked at 14:40 why the arm hovers in mid-air in the video. The ledger answers from the
payloads (18:30). The hovering is a budget-ended cycle after a missed lift. The hold detector does
not read the carry as lifted, so the cycle gets no release detection either. The arm then waits with
a stale row up to the cycle's 800-step budget, while its part is already in a box. `--missed_lift`
reads such a carry as a lift and ends the cycle at the release (section 9.9).

The ledger compares the two screens on draw 3 of the screen seed, both at X = 1.5:

| screen | switches | budget-ended cycles | their steps | correct stows | correct stows an hour |
| --- | --- | --- | --- | --- | --- |
| f_F1_100000_w150_scr | none | 49 | 39,463 of 72,427, 54 % | 104 | 155.1 |
| f_L_ml_scr | `--missed_lift` | 9 | 7,200 of 37,195, 19 % | 105 | 304.9 |

On this heap the switch halves the steps for one more correct stow. On candidate 2's checked read,
498 of 3627 cycles end on the budget, against 1203 of 3408 for candidate 1 (section 9.10).

### 9.16 The PGS solver: deviation D18

Deviation D18 (21:46) came from an owner prompt: "why not use PR #153 to speed up the trial and
error?" The ledger pre-registered it before any number. `--pile_pgs` is a wall-time lever. With both
arms under the same solver, a comparison stays paired. `feat/ms7f-pgs` is #153's 496eb6f,
cherry-picked onto `feat/ms7f-levers3` as c657f61, with additive conflicts. The validation has two
parts:

1. A paired policy screen: candidate 3's configuration with `--pile_pgs`, seed 20307000, draws 0 to
   3, TAG f_M_cap5_pgs_scr, against f_M_cap5_scr (435 of 600, 191.9 an hour).
2. A paired teacher read on R4's six H1 heaps (run 20300100 + i, layout 20300150 + i, the H1 flags),
   with and without the switch.

PGS is adopted for every later screen, the wave 3 recording and the final checked and official reads
on two conditions. The screen (1) reads within 12 correct stows of 435 and within 5 % of 191.9 an
hour. The teacher's delivered share with PGS (2) is within 0.02 of Newton's, and its seconds per
delivered part within 10 %. Until then every read stays on Newton, and v8's renders can use PGS for
images only.

The D18 amendment (22:16) came before any D18 number. The wave records under Newton, because the
policies and corpora are Newton's. So the adoption of PGS for screens and reads rests on the paired
policy screen (1) alone. The teacher pair (2) is needed only to record a corpus under PGS, and it is
dropped.

The D18 verdict (22:59), candidate 3's configuration on seed 20307000, draws 0 to 3:

| solver | TAG | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row | wrong box | wall time a heap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Newton | f_M_cap5_scr | 435 of 600, 0.725 | 0.688 to 0.759 | 762 | 244,829 | 191.9 | 110 of 151 | 111 | 5208 to 6856 s, Machine A |
| PGS | f_M_cap5_pgs_scr | 449 of 600, 0.748 | 0.712 to 0.781 | 756 | 235,083 | 206.3 | 104 of 151 | 103 | 3446 to 4179 s, Machine B |

The rule asks for parity: within 12 correct stows and within 5 % of the rate. PGS reads 14 more
correct stows and a 7.5 % higher rate. So PGS is not adopted for reads. It moves the numbers beyond
the pre-registered tolerance, in the favourable direction, so a PGS read does not compare with the
Newton reads. A heap reads about 1.7 times faster in wall time on a policy read, on different hosts.
That is far less than the teacher's 4.5 times. PGS stays for the v8 renders.

### 9.17 Retrain F3: deviation D17

Lever (b) of D17 trains retrain F3, "retrain 3" in the ledger. It starts warm from F2 150k and
trains on waves 1, 2 and wave 3's frames for 200k steps, OUT f/train_F3. Rule R11 screens its
checkpoints against candidate 3 on seed 20307000, draws 0 to 3. The best one with at least candidate
3's screen plus 6 correct stows becomes candidate 4. At 02:53 the ledger chained the steps: the
conversions on both hosts, the merge, retrain F3 and the screens f_screens_F3 of F3 50k to 200k with
candidate 3's flags. They pair with f_M_cap5_scr, so candidate 4 needs at least 441 correct stows.

Retrain F3 started at 06:40 (job f_dag_after_w3_a). Its corpus f/w123/dataset holds 13,665
episodes and 5,110,772 frames, with CHECK_OK 235 videos (section 8.4). D19 moved its screens to the
refill scene before the full-pile chain f_screens_F3 read anything (section 9.18). Retrain F3 ended
at 08:26 with TRAIN_OK, 200,000 steps and the checkpoints 50k to 200k.

### 9.18 The refill scene: deviation D19

Deviation D19 (06:40) makes the refilled deep tray the gate scene (section 1.3). The ledger
pre-registered four reads at 06:40, before any number in the new scene:

1. Candidate 3's checked read in this scene: TAG f_F3R_chk, seed 20307100, draws 0 to 19,
   `read_f2.sh` plus `--tray_refill`, candidate 3's flags and model v6a. It reads in worktree 7f-r6
   on `a_fr6l:10` and `b_fr6:10`, `PORT_BASE` 15600.
2. The pair for retrain F3's screens: candidate 3 in this scene, seed 20307000, draws 0 to 3, TAG
   f_M_cap5_R_scr, on `b_fr6:4`, `PORT_BASE` 16900.
3. Retrain F3's screens move to this scene. The full-pile chain f_screens_F3 was stopped before it
   read anything. D17's rule stands against f_M_cap5_R_scr.
4. The official read runs in this scene on the first checked read there that passes: seed 20307200,
   40 heaps, one shot, `read_f2.sh`.

f_F3R_chk and f_M_cap5_R_scr started at 06:41. Job f_screens_F3R (`bin/screens_F3R.sh`, a user
systemd unit) screens each F3 checkpoint, 50k to 200k, in the refill scene on Machine A. Its TAGs
are f_F3_<step>_R_scr, on `PORT_BASE` 14400 to 14700, against f_M_cap5_R_scr. Each screen serves its
checkpoint with candidate 3's flags and model v6a.

The interim look (07:22). The ledger disclosed it as not a verdict, before the next rule, the v9e
pre-registration of section 5.10. Machine B's 8 finished heaps of f_F3R_chk read 987 of 1200, 0.823, at
253.7 an hour, with 134 wrong-box parts. That is 0.112 of the parts, the largest loss in the refill
scene. The 3 finished heaps of f_M_cap5_R_scr read 344 of 450.

The pair f_M_cap5_R_scr ended at 07:24: 468 of 600 at 207.0 an hour (section 5.10). So D17's bar for
retrain F3's screens is at least 474 correct stows. The screens of retrain F3 in the refill scene,
all on `a_fr6l:4`:

| screen | checkpoint | correct stows | Wilson 95 % | cycles | steps | correct stows an hour | held-out row |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f_M_cap5_R_scr, the pair | F2 150k | 468 of 600, 0.780 | 0.745 to 0.811 | 768 | 244,178 | 207.0 | 111 of 151, 0.735 |
| f_F3_050000_R_scr | F3 50k | 488 of 600, 0.813 | 0.780 to 0.842 | 752 | 224,859 | 234.4 | 108 of 151, 0.715 |
| f_F3_100000_R_scr | F3 100k | 482 of 600, 0.803 | 0.770 to 0.833 | 722 | 191,598 | 271.7 | 119 of 151, 0.788 |
| f_F3_150000_R_scr | F3 150k | 498 of 600, 0.830 | 0.798 to 0.858 | 707 | 185,258 | 290.3 | 116 of 151, 0.768 |
| f_F3_200000_R_scr | F3 200k | 477 of 600, 0.795 | 0.761 to 0.825 | 707 | 192,195 | 268.0 | 116 of 151, 0.768 |

Every F3 checkpoint passes D17's 474. F3 150k reads the most, 30 over the pair, at 1.40 times the
pair's rate.

Candidate 3's checked read in the refill scene is f_F3R_chk. It ran from 06:41 to 07:52 on Machine A
draws 0 to 9 and Machine B draws 10 to 19. The ledger recorded it at 07:53:

| item | value |
| --- | --- |
| TAG | f_F3R_chk |
| checkpoint | F2 150k, candidate 3 |
| commit, classifier, switches | bcbbe02, v6a, `--action_warp 1.5 --missed_lift --cell_failure_cap 5 --tray_refill` |
| correct stows of 3000 | 2297, 0.766 |
| Wilson 95 % | 0.750 to 0.780 |
| cycles | 3729 |
| steps | 1,211,739 |
| correct stows an hour | 204.7 |
| held-out row | 577 of 785, 0.735, Wilson 0.703 to 0.765 |
| wrong-box parts | 375, from 244 `wrong_box` cycles |
| unboxed parts | 328 |
| cycles per correct stow, no bar | 1.623 |
| events | 57 refills, 15 box emptyings |
| supervisor token | GATE_FAIL at the staged bars |
| staged bars (D12) | FAIL on the share, 2297 against 2550, and on the held-out row, 0.735 against 0.80. PASS on the rate, 204.7 against 200. |
| stretch row | FAIL on all three: 2297 against 2760, 204.7 against 300, 0.735 against 0.85 |

By category: brick 860 of 1063, special 467 of 645, flat 310 of 419, round 169 of 199, joint 252 of
378 and minifig 239 of 296. This report computes the fractions of the rows by family and by box
count from `gate_results.json`:

| family | brick | slope | round | LDraw | plate | tile | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 860/1063 | 205/284 | 169/199 | 262/361 | 192/256 | 118/163 | 252/378 | 239/296 |
| fraction | 0.809 | 0.722 | 0.849 | 0.726 | 0.750 | 0.724 | 0.667 | 0.807 |

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 126/150 | 574/750 | 608/750 | 558/750 | 431/600 |
| fraction | 0.840 | 0.765 | 0.811 | 0.744 | 0.718 |

The clamp fraction is 0.0059. The ledger names heaps 0, 5 and 19 as ending on the 195-cycle budget,
at 97, 81 and 86 correct stows. Heaps 5 and 19 take about 94k steps each. The ledger counts 13 of
the 20 heaps as ending no_candidate. The payloads give a larger count: 8 heaps end on the cycle
budget (0, 2, 3, 4, 5, 7, 14 and 19), and 12 end no_candidate.

The verdict (07:53): f_F3R_chk fails the staged share bar and the held-out bar, passes the rate bar,
and fails the stretch row. No official read runs. Against candidate 3 on the full pile (0.727 at
189.5), the refill scene adds 116 correct stows and 15.2 an hour.

The pick of candidate 4 was pre-registered at 08:00. That was after the F3 50k screen and f_F3R_chk,
and before any number of the 100k to 200k screens or of the v9e screen. Job f_chk_F4R
(`bin/chk_F4R.sh`, a user systemd unit) waits for the four F3 refill screens and f_M_cap5_v9e_R_scr:

- The checkpoint: the F3 refill screen with the most correct stows. It needs at least 474 (D17
  against f_M_cap5_R_scr), or else candidate 3's F2 150k stays.
- The classifier: model v9e on a screen of at least 474 correct stows and 200.8 an hour. Else model
  v6a.
- If neither lever moves, nothing is read, because f_F3R_chk is that read. Otherwise the read is TAG
  f_F4R_chk: seed 20307100, draws 0 to 19, the refill scene, candidate 3's flags, `read_f2.sh`,
  7f-r6, `a_fr6l:10` and `b_fr6:10`, `PORT_BASE` 15800.
- A pass starts item 4 of D19, the official read.

At 09:00 the rule gave candidate 4 = F3 150k, served with v9e. The two levers come from two
different screens: F3 150k read with v6a, and v9e read on F2 150k. No screen served F3 150k with v9e
before the checked read.

Candidate 4's checked read f_F4R_chk ran from 09:01 to 09:58 on Machine A draws 0 to 9 and Machine B draws
10 to 19. The ledger recorded it at 09:59:

| item | value |
| --- | --- |
| TAG | f_F4R_chk |
| checkpoint | F3 150k, candidate 4 |
| commit, classifier, switches | bcbbe02, v9e, `--action_warp 1.5 --missed_lift --cell_failure_cap 5 --tray_refill` |
| correct stows of 3000 | 2467, 0.822 |
| Wilson 95 % | 0.808 to 0.836 |
| cycles | 3453, 1.400 a correct stow |
| steps | 888,218 |
| correct stows an hour | 300.0 |
| held-out row | 617 of 785, 0.786, Wilson 0.756 to 0.813 |
| wrong-box parts | 376, from 231 `wrong_box` cycles |
| unboxed parts | 157 |
| events | 60 refills, 16 box emptyings |
| supervisor token | GATE_FAIL at the staged bars |
| staged bars (D12) | FAIL on the share, 2467 against 2550, 83 short, and on the held-out row, 0.786 against 0.80, 11 parts short. PASS on the rate, 300.0 against 200. |
| stretch row | FAIL on the share, 2467 against 2760, and on the held-out row, 0.786 against 0.85. The ledger reads the rate as met at 300.0. |

The payload's rate is 299.97 correct stows an hour, 0.03 under 300. The ledger rounds it to 300.0
and reads it as meeting the stretch rate.

By category: brick 872 of 1063, special 517 of 645, flat 340 of 419, round 176 of 199, joint 299 of
378 and minifig 263 of 296. This report computes the fractions of the rows by family and by box
count:

| family | brick | slope | round | LDraw | plate | tile | joint | minifig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct stows | 872/1063 | 227/284 | 176/199 | 290/361 | 210/256 | 130/163 | 299/378 | 263/296 |
| fraction | 0.820 | 0.799 | 0.884 | 0.803 | 0.820 | 0.798 | 0.791 | 0.889 |

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 125/150 | 604/750 | 620/750 | 609/750 | 509/600 |
| fraction | 0.833 | 0.805 | 0.827 | 0.812 | 0.848 |

The clamp fraction is 0.0031. 19 of the 20 heaps end no_candidate, and heap 3 ends on the 195-cycle
budget at 125 correct stows.

The verdict (09:59): f_F4R_chk fails the staged share bar (0.85, 83 correct stows short) and the
held-out bar (0.80, 11 short). It passes the rate bar (200) and meets the stretch rate (300). No
official read runs. Against candidate 3 in this scene, candidate 4 gains 170 correct stows and 95.3
an hour. The unboxed parts fall from 328 to 157, and the wrong-box parts stay at 375 and 376. Every
category gains, and special gains the most, from 467 to 517 of 645.

TBD: the official read in the refill scene, on the first checked read there that passes the staged
bars.

### 9.19 Wave 4 and retrain F4

The ledger pre-registered wave 4 at 08:32, with no F4 number in existence. It is the D17 DAgger
lever once more, now in the D19 gate scene:

- The driver is candidate 4 as f_chk_F4R picks it, checkpoint and classifier. On CHK_NONE it is
  candidate 3 with v6a. The driver is written to `f/w4/driver.txt`.
- The recorder is wave 3's. `tables/dag/bin/wave4_host.sh` is `wave3_host.sh` plus `--tray_refill`,
  from `feat/ms7f-dagger` 6abc2c4. It records the train split on 150-part heaps, run 20308200 + p
  and layout 20308300 + p (rule R1 amended, section 1.5).
- Machine A records p 0 to 44 on 6 lanes, and Machine B p 45 to 89 on 10 lanes, 5 h each, next to
  f_F4R_chk.
- The conversion follows wave 3. Retrain F4 joins f/w123/dataset and both wave 4 halves into
  f/w1234/dataset and trains 200k steps warm from the driver, OUT f/train_F4.
- The screens read each F4 checkpoint, 50k to 200k, in the refill scene. They use seed 20307000,
  draws 0 to 3, and candidate 4's flags and classifier. They pair with candidate 4's own served
  configuration on the same seed. If no such screen exists, the pair runs with them.
- Candidate 5 is the best F4 screen with at least the pair's correct stows plus 6. Its checked read
  (seed 20307100, draws 0 to 19, the refill scene) runs only on a failure of f_F4R_chk.
- A smoke on Machine B runs first: K = 50, 30 cycles, F2 150k, out `f/w4/smoke`.

The smoke passed at 09:00 on Machine B with F2 150k and v6a: EVAL_OK, 18 takeovers (16 stowed), 20 parts
stowed, WAVE4_OK. At K = 50 the tray holds every part, so no refill fired.

Wave 4 started at 09:05 with the driver of `f/w4/driver.txt`, candidate 4: F3 150k and model v9e.
Machine B's first share on 10 lanes failed every heap within 25 s on CUBLAS_STATUS_ALLOC_FAILED. The 10
policy servers of f_F4R_chk held 9 GB of the RTX 5080's 16 GB, so 10 more did not fit. The share was
stopped, and its 64 empty files moved to `f/w4/failed_0903`. No heap had recorded anything. It
relaunched on 5 lanes with the same job name, p 45 to 89, 5 h. Machine A's share runs on 6 lanes, p
0 to 44. The ledger records this as a deviation from the pre-registered lane counts only, and its
09:05 entry dates those counts 08:45.

TBD: wave 4's recording on both hosts, its takeovers and its conversion.

TBD: retrain F4 (f/train_F4 on f/w1234/dataset), its refill-scene screens against candidate 4's
pair, and candidate 5 by the rule of 08:32.

### 9.20 Three loop switches on candidate 4: deviation D20

Deviation D20 (10:00) was pre-registered before any D20 number. Candidate 4 has rate to spare, 300.0
an hour against the staged 200. D9 and D14 turned down two levers for their rate cost. So those two
get one more screen in the D19 scene, and D16's carry votes join them. Every screen serves candidate
4's configuration: F3 150k, model v9e, `--action_warp 1.5 --missed_lift --cell_failure_cap 5
--tray_refill`. Each reads seed 20307000, draws 0 to 3, through `read_f.sh` in 7f-r6 (bcbbe02), on 4
lanes:

| screen | host | switches |
| --- | --- | --- |
| f_F4c_R_scr, the pair and also the pair of the F4 screens | Machine A | none |
| f_F4c_pd_R_scr | Machine A | `--passenger_drop` |
| f_F4c_lv6_R_scr | Machine A | `--lift_view_steps 6` |
| f_F4c_cv2_R_scr | Machine B | `--carry_votes 2` |
| f_F4c_pdlv_R_scr | Machine B | `--passenger_drop --lift_view_steps 6` |

The rule: among the four, the screen with the most correct stows is adopted on two conditions. Its
correct stows are at least the pair's plus 6, and its correct stows an hour are at least 240. Ties
go to the higher rate. 240 is 1.2 times the staged bar, a margin for the checked read. An adopted
switch joins candidate 4's flags as candidate 4'. Candidate 4' then gets a checked read f_F4pR_chk
(seed 20307100, draws 0 to 19, the refill scene, `read_f2.sh`). The switch also joins the F4 screens
and their pair, and the pair runs again with it. If nothing is adopted, candidate 4's flags stand.

TBD: the D20 screens and their verdict.

TBD: the checked read of candidate 4' (f_F4pR_chk) or of candidate 5, against the staged bars and
the stretch row.

## 10. The decomposition

With no checked read that passes, rule R2 asks for the decomposition. The three checked reads on the
full pile miss every bar of both rows. The two checked reads in the refill scene of D19 pass the
staged rate and miss the share and the held-out row. This section answers where the parts and the
hours go across the five checked reads. The columns "candidate 3, refill" and "candidate 4, refill"
are f_F3R_chk and f_F4R_chk.

### 10.1 Where the parts go

Each checked read serves 3000 parts. This report computes the split from the ledger's counts. A part
without a `wrong_box` cycle of its own is a "rider" in the ledger's sense. Most of them are the
cycle's own part in a missed lift (section 9.9). An unboxed part ends outside every box: in the
tray, on the desk or on a rim.

| part | candidate 1 | candidate 2 | candidate 3 | candidate 3, refill | candidate 4, refill |
| --- | --- | --- | --- | --- | --- |
| correct box | 2043 | 2065 | 2181 | 2297 | 2467 |
| wrong box, the classifier's `wrong_box` cycles | 291 | 264 | 256 | 207 | 194 |
| wrong box, other `wrong_box` cycles | 27 | 43 | 40 | 37 | 37 |
| wrong box, riders without a `wrong_box` cycle | 303 | 243 | 214 | 131 | 145 |
| unboxed | 336 | 385 | 309 | 328 | 157 |
| short of the staged 2550 | 507 | 485 | 369 | 253 | 83 |
| short of the stretch 2760 | 717 | 695 | 579 | 463 | 293 |
| held-out row, against 0.80 | 0.600 | 0.647 | 0.666 | 0.735 | 0.786 |

The ledger gives the refill reads' wrong-box parts, `wrong_box` cycles and unboxed parts. This
report computes the classifier's share of their `wrong_box` cycles from the payloads, the same way
as for candidates 1 to 3. On candidate 4 that split comes from model v9e.

Five facts follow:

- The staged bar allows 450 misses of 3000. On the full pile, the classifier's `wrong_box` cycles
  alone take 256 to 291 of them, more than half. v6a reads 0.897 on deep-tray heaps against 0.939 in
  the normal tray (section 5.8), and v7 and v8 did not close that gap (section 5.9). In the refill
  scene they take 207 with v6a and 194 with v9e (section 5.10).
- `--missed_lift` cuts the riders from 303 to 243, and candidate 3 reads 214. The riders still cost
  0.07 of the parts.
- The unboxed parts rise from 336 to 385 on candidate 2. Its episodes end no_candidate with about 16
  parts a heap left in excluded cells. `--cell_failure_cap 5` brings the unboxed parts down to 309
  on candidate 3, and the correct stows gain 116.
- The refill scene cuts the riders from 214 to 131 for the same policy and flags. Candidate 3 gains
  116 correct stows there, but its unboxed parts rise from 309 to 328. 8 of its 20 heaps end on the
  195-cycle budget.
- Candidate 4 halves the unboxed parts, from 328 to 157, and only one heap ends on the cycle budget.
  Its wrong-box parts stay at 376, so the wrong boxes now hold 376 of its 533 misses. The staged bar
  needs 83 more correct stows.

### 10.2 Where the hours go

The share of each read's steps by cycle outcome (11:54, 18:06 and 00:52). This report computes the
rest, and all of the two refill columns:

| cycle outcome | candidate 1 | candidate 2 | candidate 3 | candidate 3, refill | candidate 4, refill |
| --- | --- | --- | --- | --- | --- |
| steps | 1,566,223 | 1,179,094 | 1,242,893 | 1,211,739 | 888,218 |
| cycles | 3408 | 3627 | 3739 | 3729 | 3453 |
| stowed | 0.239 | 0.363 | 0.352 | 0.380 | 0.492 |
| `close_miss` | 0.295 | 0.336 | 0.340 | 0.299 | 0.254 |
| `lifted_lost` | 0.194 | 0.076 | 0.073 | 0.066 | 0.067 |
| `held_no_lift` | 0.084 | 0.085 | 0.083 | 0.067 | 0.071 |
| `wrong_box` | 0.075 | 0.061 | 0.058 | 0.050 | 0.055 |
| `carrying` | 0.063 | 0.039 | 0.051 | 0.078 | 0.031 |
| `no_open` and `no_close` | 0.049 | 0.040 | 0.043 | 0.060 | 0.031 |
| cycles that end on the 800-step budget | 1203 | 498 | 562 | 519 | 268 |
| steps a stowed cycle | 244, 8.2 s | 235, 7.9 s | 230, 7.7 s | 224, 7.5 s | 196, 6.5 s |
| steps a `close_miss` cycle, of them before the first close | 740, 620 | 447, 392 | 449, 406 | 435, 399 | 373, 327 |
| correct stows an hour | 140.9 | 189.1 | 189.5 | 204.7 | 300.0 |

The payloads give the steps a cycle and the split before the first close. Six facts follow:

- A stowed cycle takes 6.5 to 8.2 s on every read. That is inside the 12 s of the stretch rate and
  the 18 s of the staged rate. The policy can run the cycle at the pace that the bars ask.
- `--missed_lift` moves the time from the budget-ended cycles to the stowed ones. The budget ends
  fall from 1203 to 498, and `lifted_lost` falls from 0.194 to 0.076 of the steps.
- `close_miss` becomes the largest sink on candidate 2, 0.336 of the steps, and stays so on
  candidate 3 at 0.340. Most of a `close_miss` cycle passes before its first close, 392 of 447 steps
  on candidate 2 and 406 of 449 on candidate 3. So the cost lies in the approach to the grasp, not
  in the close.
- `--cell_failure_cap 5` buys its 116 correct stows with 112 more cycles and 63,799 more steps, so
  the rate stays at 189.5 against 189.1. The budget ends rise from 498 to 562.
- In the refill scene, candidate 3's `close_miss` share falls from 0.340 to 0.299, and `carrying`
  and the `no_open` and `no_close` cycles grow. The rate rises to 204.7, over the staged 200.
- Candidate 4 spends 27 % fewer steps than candidate 3 in the same scene, 888,218 against 1,211,739,
  for 170 more correct stows. The stowed cycles hold 0.492 of its steps, and a stowed cycle takes
  6.5 s. `close_miss` stays the largest sink at 0.254, with 327 of its 373 steps a cycle before the
  first close.

### 10.3 The pace arithmetic for the staged bars

At 30 Hz, 200 correct stows an hour is 540 steps per correct stow, and 300 an hour is 360 steps.
This report computes the step budgets of a 3000-part checked read:

| row | correct stows | correct stows an hour | step budget | candidate 1 spent | candidate 2 spent | candidate 3 spent | candidate 3, refill, spent | candidate 4, refill, spent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| staged bars | 2550 | 200 | 1,377,000 | 1,566,223, 14 % over | 1,179,094, 14 % under | 1,242,893, 10 % under | 1,211,739, 12 % under | 888,218, 35 % under |
| stretch row | 2760 | 300 | 993,600 | 58 % over | 19 % over | 25 % over | 22 % over | 11 % under |

Candidate 1 misses both bars: it needs 507 more correct stows and 12 % fewer steps. Candidate 2's
steps already fit the staged budget. At its own 1,179,094 steps, 2550 correct stows read 233.6 an
hour. So on candidate 2 the share bar binds, and every correct stow it gains also lifts the rate.
With its 2065 correct stows, 200 an hour asks for at most 1,115,100 steps, 5 % fewer than it spent.

Candidate 3's steps also fit the staged budget. At its own 1,242,893 steps, 2550 correct stows read
221.6 an hour, so the share bar binds here too. With its 2181 correct stows, 200 an hour asks for at
most 1,177,740 steps, 5 % fewer than it spent.

The stretch row asks for more than this policy has. It needs 579 more correct stows on candidate 3
and 20 % fewer steps at the same time. On candidate 2 it needs 695 more and 16 % fewer. The wave
teacher itself delivers 222 to 259 parts an hour (section 10.4).

In the refill scene, candidate 3 spends 12 % under the staged budget, and 2550 correct stows at its
steps read 227.3 an hour. Candidate 4 spends 35 % under the staged budget and 11 % under the stretch
budget. At its own 888,218 steps, 2550 correct stows read 310.1 an hour, and 2760 read 335.6. So on
candidate 4 only the share and the held-out row bind, on both rows. Its 2467 correct stows at 300 an
hour ask for at most 888,120 steps, 98 fewer than it spent.

### 10.4 The throughput arithmetic

The stretch rate of 300 correct stows an hour is 12.0 s, or 360 control steps at 30 Hz, per correct
stow. The staged rate of 200 is 18.0 s, or 540 steps. The throughput is the product of two factors:
the seconds a cycle and the cycles per correct stow. This report computes the values from the
ledger's counts. For the wave rows, the columns count delivered parts in place of correct stows:

| row | s a cycle | cycles per correct stow | s per correct stow | correct stows an hour |
| --- | --- | --- | --- | --- |
| the stretch rate | | | 12.0 | 300 |
| the staged rate (D12) | | | 18.0 | 200 |
| candidate 4, checked read in the refill scene, F3 150k, v9e | 8.6 | 1.400 | 12.0 | 300.0 |
| candidate 3, checked read in the refill scene | 10.8 | 1.623 | 17.6 | 204.7 |
| stretch run r = 0, candidate 2's flags, the refill tray | 9.0 | 1.536 | 13.9 | 259.9 |
| candidate 3, checked read, X = 1.5, `--missed_lift --cell_failure_cap 5` | 11.1 | 1.714 | 19.0 | 189.5 |
| candidate 2, checked read, X = 1.5, `--missed_lift` | 10.8 | 1.756 | 19.0 | 189.1 |
| candidate 1, `--missed_lift` screen, X = 1.5 | 10.1 | 1.696 | 17.1 | 210.5 |
| candidate 1, checked read, X = 1.5 | 15.3 | 1.668 | 25.6 | 140.9 |
| candidate 1, warp screen, X = 1.5 | 15.1 | 1.640 | 24.7 | 145.5 |
| candidate 1, screen, X = 1.0 | 17.0 | 1.612 | 27.4 | 131.4 |
| control read, E2 on the gate configuration, aarch64 | 17.0 | 1.398 | 23.8 | 151.3 |
| E2 on the MS7e configuration, S0 | 16.98 | 1.275 | 21.64 | 166 |
| wave 1 teacher, Machine A share | 13.85 | 1.172 | 16.23 per delivered part | 222 delivered parts |
| wave 2 teacher, aarch64 share | | 1.130 | 13.91 per delivered part | 259 delivered parts |
| wave 2 teacher, Machine B share | | 1.167 | 14.81 per delivered part | 243 delivered parts |

Eight facts follow from the table:

- At candidate 1's 17.0 s a cycle without the warp, even one cycle per correct stow gives 212 an
  hour. The warp shortens the cycle to about 15 s, but the cycles per correct stow rise.
- E2 ran 16.98 s a cycle on the MS7e configuration, near its corpus's 16.7 s, and 17.0 s on the
  gate configuration. Candidate 1 also runs 17.0 s a cycle at X = 1.0, against 13.85 s for
  Machine A's share of its wave 1 corpus.
- The wave teacher delivers 222 to 259 parts an hour. A policy that copies its corpus teacher cycle
  for cycle reads under 300 an hour, even with no failed cycle. The staged rate of 200 lies under
  the teacher's pace.
- Every policy row needs at least 1.275 cycles per correct stow. At 15.3 s a cycle, 200 an hour
  needs at most 1.17 cycles per correct stow.
- `--missed_lift` cuts the cycle to 10.1 to 10.8 s. At candidate 2's 10.8 s a cycle, 200 an hour
  needs at most 1.66 cycles per correct stow, and candidate 2 reads 1.756.
- `--cell_failure_cap 5` lengthens the cycle to 11.1 s and cuts the cycles per correct stow to
  1.714. At 11.1 s a cycle, 200 an hour needs at most 1.62 cycles per correct stow.
- The stretch run serves candidate 2's policy in the refill tray. There it runs 9.0 s a cycle at
  1.536 cycles per correct stow, 259.9 an hour.
- Candidate 4 moves both factors at once in the refill scene: 8.6 s a cycle and 1.400 cycles per
  correct stow. That gives 12.0 s per correct stow, the stretch rate. Candidate 3 in the same scene
  reads 10.8 s and 1.623.

The ledger's arithmetic of the stretch row on 600 parts (05:56): 552 correct stows at 300 an hour is
198,720 steps, 57 % of what candidate 1's screen spent. At 311 steps a stowed cycle, that leaves
about 68,000 steps for every failure, against 246,245 now, a 72 % cut. This report computes the
staged row the same way: 510 correct stows at 200 an hour is 275,400 steps, 79 % of the screen's
349,191.

The design's target for the teacher was 10 s per correct stow, 360 an hour. With about 1.15 cycles
per correct stow, a policy at that pace reaches 12 s and 300 an hour (MS7f design section 8.2). The
read teacher came near that target: S1's comb3 reads 10.30 s on 50-part heaps, and the reference
read 11.27 s on 150-part heaps. The recorder recipe stays slower. M5 reads 11.24 s per delivered
part on 50-part heaps, and the deep-tray waves 13.91 to 16.23 s.

### 10.5 Where the steps go

The ledger splits candidate 1's screen, 4 heaps and 349,191 steps, by cycle outcome (05:56). The
intervals are the steps a cycle in select to close, close to lift, lift to release and release to
the next selection:

| cycle outcome | cycles | steps | share of the steps | steps a cycle | intervals |
| --- | --- | --- | --- | --- | --- |
| stowed | 331 | 102,946 | 0.295 | 311 | 185, 32, 73, 21 |
| `close_miss` | 119 | 92,070 | 0.264 | 774 | 643, 95, 6, 30 |
| `lifted_lost` | 68 | 43,233 | 0.124 | 636 | 250, 180, 56, 149 |
| `held_no_lift` | 49 | 35,763 | 0.102 | 730 | 463, 247, 3, 16 |
| `wrong_box` | 63 | 25,543 | 0.073 | 405 | 191, 112, 64, 38 |
| `no_open` | 29 | 23,696 | 0.068 | 817 | 817, 0, 0, 0 |
| `carrying` | 22 | 22,666 | 0.065 | 1030 | 344, 130, 479, 77 |
| `no_close` | 4 | 3,274 | 0.009 | 819 | |

259 of the 685 cycles end on the 800-step budget. A stowed cycle takes 311 steps, 10.4 s. With every
cycle stowed, that pace gives 347 an hour. A `close_miss` cycle spends 643 steps before its first
close.

The same table for the control read, E2 on 7 heaps on Machine C: a stowed cycle takes 406
steps, 13.6 s, 266 an hour at best, and `close_miss` holds 0.160 of the steps against candidate 1's
0.264. So the faster wave recipe (M5) made the policy's good cycles 23 % faster, and its failures
ate the gain (05:56).

Candidate 1's checked read at X = 1.5 shows the same picture on 3000 parts (section 9.7).
`close_miss` holds 0.295 of the steps and the stowed cycles 0.239. A stowed cycle takes 244 steps,
8.2 s. With every cycle stowed, that pace gives 442 an hour. `lifted_lost` ends 15.7 % of the cycles
against 9.9 % at X = 1.0. 1203 of 3408 cycles end on the 800-step budget.

Candidate 2's checked read adds `--missed_lift` (section 9.10). Its payload gives the four intervals
by outcome. This report computes the steps a cycle and the shares from them:

| cycle outcome | cycles | share of the steps | steps a cycle | intervals |
| --- | --- | --- | --- | --- |
| stowed | 1818 | 0.363 | 235 | 145, 24, 51, 15 |
| `close_miss` | 886 | 0.336 | 447 | 392, 44, 3, 8 |
| `held_no_lift` | 201 | 0.085 | 498 | 374, 117, 1, 7 |
| `lifted_lost` | 309 | 0.076 | 288 | 173, 42, 47, 26 |
| `wrong_box` | 307 | 0.061 | 233 | 141, 29, 49, 14 |
| `carrying` | 43 | 0.039 | 1076 | 247, 140, 663, 25 |
| `no_open` | 32 | 0.022 | 822 | 785, 8, 0, 29 |
| `no_close` | 31 | 0.018 | 691 | 596, 93, 1, 2 |

`close_miss` is now the largest sink, and most of its steps fall before the first close. The budget
ends fall from 1203 of 3408 cycles to 498 of 3627.

Candidate 3's checked read adds `--cell_failure_cap 5` (section 9.11). This report computes the same
table from its payload:

| cycle outcome | cycles | share of the steps | steps a cycle | intervals |
| --- | --- | --- | --- | --- |
| stowed | 1904 | 0.352 | 230 | 140, 24, 51, 16 |
| `close_miss` | 942 | 0.340 | 449 | 406, 33, 4, 6 |
| `held_no_lift` | 196 | 0.083 | 525 | 376, 119, 1, 29 |
| `lifted_lost` | 271 | 0.073 | 333 | 196, 64, 49, 24 |
| `wrong_box` | 296 | 0.058 | 242 | 151, 33, 41, 17 |
| `carrying` | 59 | 0.051 | 1074 | 271, 62, 732, 9 |
| `no_open` | 39 | 0.025 | 786 | 733, 0, 0, 54 |
| `no_close` | 32 | 0.019 | 730 | 620, 103, 2, 4 |

The picture of candidate 2 holds. `close_miss` keeps 0.340 of the steps, 406 of its 449 steps a
cycle before the first close. `carrying` grows from 0.039 to 0.051 of the steps, and its cycles
spend 732 steps between the lift and the release. 562 of the 3739 cycles end on the budget.

The two checked reads in the refill scene, computed the same way from their payloads:

| cycle outcome | candidate 3, refill: cycles | share of the steps | steps a cycle | intervals | candidate 4, refill: cycles | share of the steps | steps a cycle | intervals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stowed | 2058 | 0.380 | 224 | 136, 20, 55, 13 | 2228 | 0.492 | 196 | 121, 17, 44, 14 |
| `close_miss` | 831 | 0.299 | 435 | 399, 28, 3, 6 | 604 | 0.254 | 373 | 327, 35, 5, 7 |
| `held_no_lift` | 170 | 0.067 | 475 | 350, 87, 5, 34 | 129 | 0.071 | 488 | 371, 107, 2, 8 |
| `lifted_lost` | 248 | 0.066 | 321 | 168, 56, 64, 33 | 199 | 0.067 | 297 | 146, 68, 55, 29 |
| `wrong_box` | 244 | 0.050 | 247 | 150, 36, 47, 14 | 231 | 0.055 | 210 | 131, 29, 40, 9 |
| `carrying` | 86 | 0.078 | 1105 | 228, 117, 715, 45 | 26 | 0.031 | 1062 | 244, 115, 702, 0 |
| `no_open` | 48 | 0.032 | 800 | 783, 0, 0, 17 | 32 | 0.029 | 804 | 757, 22, 0, 25 |
| `no_close` | 44 | 0.029 | 792 | 756, 30, 0, 6 | 4 | 0.002 | 472 | 459, 5, 6, 2 |

On candidate 4 every outcome but `held_no_lift` and `no_open` takes fewer steps a cycle than on
candidate 3 in the same scene. The `carrying` cycles fall from 86 to 26, and the `no_close` cycles
from 44 to 4. 268 of its 3453 cycles end on the 800-step budget, against candidate 3's 519 of 3729.

The cycle classes of 13 screen heaps, 1950 parts, every F1 screen payload but the 50k draw 3
(04:07). This report computes the shares:

| class | cycles | share of the cycles |
| --- | --- | --- |
| stowed | 1022 | 0.460 |
| `close_miss` | 422 | 0.190 |
| `lifted_lost` | 221 | 0.099 |
| `wrong_box` | 208 | 0.094 |
| `no_open` | 144 | 0.065 |
| `held_no_lift` | 129 | 0.058 |
| `carrying` | 54 | 0.024 |
| `no_close` | 23 | 0.010 |
| all | 2223 | |

Fewer than half of the cycles stow their target. The 13 heaps stow 1296 parts with 1022 stowed
cycles, so other parts also reach their own box.

### 10.6 The wrong-box parts

On candidate 1's screen (05:56), 425 parts stow from 331 stowed cycles, so 94 correct parts ride
along. 119 parts end in a wrong box: 63 from `wrong_box` cycles, 57 of them the classifier's, and 56
"riders".

On the 13 screen heaps of 04:07, 379 of 1950 parts end in a wrong box, 0.194:

| source | parts | share of the 1950 parts |
| --- | --- | --- |
| `wrong_box` cycles | 208 | 0.107 |
| of them, the classifier's | 197 | 0.101 |
| of them, special read as brick | 74 | |
| of them, brick read as special | 51 | |
| of them, brick read as flat | 15 | |
| parts that reached a wrong box without a cycle of their own | 171 | 0.088 |
| all wrong-box parts | 379 | 0.194 |

On candidate 1's checked read, 621 of 3000 parts end in a wrong box. 318 come from `wrong_box`
cycles, 291 of them the classifier's. This report computes the other 303 as parts without a
`wrong_box` cycle of their own. On candidate 2's checked read, with `--missed_lift`, 550 parts end
in a wrong box. 307 come from `wrong_box` cycles, 264 of them the classifier's, and this report
computes the other 243. So `--missed_lift` cuts the parts without a cycle of their own from 303 to
243, and the classifier's cycles stay near 0.09 of the parts. On candidate 3's checked read, with
`--cell_failure_cap 5`, 510 parts end in a wrong box. 296 come from `wrong_box` cycles, 256 of them
the classifier's, and this report computes the other 214.

In the refill scene, candidate 3 puts 375 parts in a wrong box, from 244 `wrong_box` cycles. This
report computes 207 of the cycles as the classifier's and 131 parts without a cycle of their own.
Candidate 4, with v9e, puts 376 parts in a wrong box, from 231 cycles. This report computes 194 of
them as the classifier's and 145 parts without a cycle of their own. So the wrong boxes stay at
0.125 of the parts in the refill scene, and the classifier's cycles take 0.065 to 0.069 of the
parts.

The `--missed_lift` study (12:54, section 9.9) changes what these "riders" are. Most of them are the
cycle's own part in a cycle that the hold detector never read as lifted. Such a cycle gets no wrist
read, so the overhead read sets the box. It runs to the 800-step budget with its part already boxed.
On the screen, 116 of 150 uncredited boxed parts were another cycle's target or held part. About 36
were the part of no cycle, and 27 of those ended in a wrong box. So the same mechanism costs wrong
boxes and steps: these cycles hold 20 to 24 % of the steps.

The classifier's taken class is right on 1253 of 1505 lifted cycles, 0.833, with model v5. The
`--lift_view_steps` study puts the in-loop wrist read at 0.819 on target lifts, against C1's 0.903,
because the part still rises at the detector's lift. C1's `loop_sum` reads 0.928 on teacher
rollouts. The screen's heap of 03:15 shows the same split: 14 of its 15 `wrong_box` cycles are the
classifier's, 12 set by the wrist vote at the lift and 2 by the overhead read.

This report computes one bound from candidate 1's screen. Its 425 correct stows plus its 119
wrong-box parts make 544 of 600, 0.907. That is over the staged 0.85 and under the stretch 0.92. The
same bound on candidate 2's checked read is 2065 plus 550, 2615 of 3000, 0.872. On candidate 3's
checked read it is 2181 plus 510, 2691 of 3000, 0.897. In the refill scene it is 2672 of 3000,
0.891, for candidate 3 and 2843, 0.948, for candidate 4. So with every boxed part in its right box,
candidate 4 reads over the staged 0.85 and over the stretch 0.92. The screen misses 175 parts. Wrong
boxes hold 119 of them, and the other 56 end outside every box. With the class stand-in on the MS7e
50-part configuration, E2 reads 0.907 (C2).

### 10.7 The teacher's own share and pace

The teacher rows against the two rows of the gate:

| read | correct box or delivered | s per correct stow or delivered part |
| --- | --- | --- |
| R4 H1 teacher, 6 heaps, no `--perceived` | 842 of 900, 0.936 | 18.42 |
| reference read, 5 heaps, with `--book_miss` | 641 of 750, 0.855 | 11.27 |
| census row B, 3 diagnostic heaps, no `--book_miss` | 422 of 450, 0.938 | about 10.6 |
| wave 1 recorder | 5216 of 6000, 0.869 | 16.23 (Machine A share) |
| wave 2 recorder | 5155 of 6000, 0.859 | 13.91 and 14.81 |
| the staged bars | 0.85 | 18.0 |
| the stretch row | 0.92 | 12.0 |

Every teacher row but R4 H1 meets the staged bars. R4 H1 misses the staged rate at 18.42 s, because
of the stray loop of section 3.4. Only census row B meets the stretch row, on three diagnostic heaps
without `--book_miss` and without the recorder's draws. The recorder delivers 0.859 to 0.869 of its
parts at 13.91 to 16.23 s a part. Every wave episode ends on its time-out.

## 11. The code, the deviations and the defects

### 11.1 The stack

All branches start from 29a36ef (deviation D2). Deviation D5 sets the order base, heap, parts,
speed, classifier. The wave fixes and the extra kinds sit between speed and the classifier (20:57).
Two restacks made the chain linear:

- 13:20: `origin/feat/ms7f-heap` goes back to 3fa9f3f. Its four extra commits are on heap-w1: two
  identical, two equal in their added lines. A local backup branch keeps 9547db2. cls6 becomes
  7baf879 (ab3008d on 03a5cd8), and levers2 becomes 4cdf4df on it, with files byte-equal to
  496f857. The chain has 73 commits over main, and 1565 plus 130 tests pass.
- 14:58: another session found a CI failure on PR #153. Under CI's `MUJOCO_GL=disable`, the GL skip
  marker of `tests/test_render_set.py` had slid off the render test (cf3dbb4, 403c22d). The fix
  83ed51d lands on `feat/ms7f-classifier` as a fast-forward, and warp, cls6 and levers2 are picked
  again on top. Each new tip differs from its old one only in that test file, so every read keeps
  its code: d6f893d = 83ed51d, 03a5cd8 = d4ae5ec, 496f857 = 0a1b552 and fe33c0c = fbaa41f. The
  CI-style suite passes 1543 tests at the classifier tip and 1558 plus 130 at the levers2 tip.

The stack opened at 17:11 as GitHub stack #164, ten draft PRs. The first CI run passes on all ten.
The ledger and the branches give these tips:

| order | PR | branch | tips | what it adds |
| --- | --- | --- | --- | --- |
| 1 | #154 | `feat/ms7f-base` | b4a2af1 | 4c3b66f the six categories and their box map. 4f83b2f `--categories` through the env, the grader, the teacher and the loop. eabac0c the throughput, the held-out row and the S0 intervals. 2c5cce5 every MS7f switch declared. b4a2af1 heaps up to 150. |
| 2 | #155 | `feat/ms7f-heap` | 3fa9f3f, back from 9547db2 at 13:20 | 0623cc3 the deep tray and the refill. c59cbcc box emptying. ee8b24e the hidden share. 59d5e65 `--render`. 098e5a4 the stretch rules. e5d7175 the box-load reset. 9a9fb7e the recorder hook. 3762445 `scripts/mj_stretch_run.py`. 3fa9f3f `--book_miss`. |
| 3 | #156 | `feat/ms7f-parts` | 9754ee8, then 34c7e0e | the table's `--seed_base`, the joint and minifig families, two axis data fixes, `perception.OWN_AXES` for the new families, the hat dropped |
| 4 | #157 | `feat/ms7f-speed` | dd4a8bb, then 077c7f0 | `--transport_speed` in the heap client, the teacher waits, `--lift_budget`, `--close_guard`, `--gain_min`, `--quiet_landing`, `--landing_steps`, `--end_at_done`, `--net_stall` |
| 5 | #158 | `feat/ms7f-heap-w1` | 4464b79 | `--abandon_cycle`, `--near_tracking` (not used), parts in rows of 50, the pile top |
| 6 | #159 | `feat/ms7f-cls-extra` | 690117f, e2114d3, then 5bc55ae | the D4 extra kinds |
| 7 | #160 | `feat/ms7f-classifier` | 0322cfc, d6f893d, then 83ed51d | the narrow overhead camera, the render set, the trainer and the three-view vote. `--classifier`, `--passenger_drop`, the crops, the size input and the regularisation. 83ed51d the CI fix. |
| 8 | #161 | `feat/ms7f-warp` | 03a5cd8, then d4ae5ec | `--action_warp` (D7) |
| 9 | #162 | `feat/ms7f-cls6` | ab3008d, 7baf879, then 8bd326d | the trainer's `--arch`, `--input_px` and `--ensemble` (v6a) |
| 10 | #163 | `feat/ms7f-levers2` | 496f857, 4cdf4df, fe33c0c, fbaa41f, then 5e92758 | c7bf05d `--lift_view_steps`, 0a1b552 `--missed_lift` (D14), fbaa41f the stretch parser's cap of 1000, 5e92758 the deep-tray renders for v7 |
| 11 | #165 | `feat/ms7f-levers3` | bcbbe02 on 5e92758 | fbd45e9 `--cell_failure_cap`, bcbbe02 `--carry_votes` (D16) |
| top | #166 | `docs/ms7f-report` | on bcbbe02 | this report |
| off the stack | no PR | `feat/ms7f-pgs` | c657f61 on bcbbe02 | #153's 496eb6f, `--pile_pgs` (D18) |
| off the stack | no PR | `feat/ms7f-render-v8` | 0323927 on c657f61 | `pile_pgs` in RENDER_READY, the trainer's `--repeat N --repeat_match GLOB` (v8) |
| off the stack | no PR | `feat/ms7f-dagger` | 6abc2c4 on c657f61 | `scripts/mj_eval_heap.py --dagger_dir`, the wave 3 recorder (D17) |

The lever branches `feat/ms7f-riders` 5715401 and `feat/ms7f-cls-loop` 36b069b stay as the record.
Their commits sit in levers2 as 0a1b552 and c7bf05d. Another session's draft PR #153 holds
`--pile_pgs` (section 11.2).

The CPU test counts in the ledger:

| commit | repo tests | lerobot_stack tests |
| --- | --- | --- |
| b4a2af1, base | 1496 | |
| 9754ee8, parts | 1499 | |
| dd4a8bb, speed | 1508 | |
| 3fa9f3f, heap | 1509 | |
| 34c7e0e and 077c7f0, the first restack | 1512 and 1524 | |
| 4464b79, the wave tip | 1528 passed, 56 skipped | |
| d6f893d, the read tip | 1551 passed, 60 skipped | 128 |
| 03a5cd8, warp | 1552 | 129 |
| 36b069b, cls-loop | 1554 | |
| 5715401, riders | 1562 | |
| 496f857, levers2 | 1565 | |
| 4cdf4df, levers2 after the 13:20 restack | 1565 | 130 |
| 83ed51d, classifier after the CI fix, CI-style | 1543 | |
| fbaa41f, levers2 after the CI fix, CI-style | 1558 | 130 |
| 5e92758, the render patch, `MUJOCO_GL=disable` | 1559 | |
| bcbbe02, levers3, `MUJOCO_GL=disable` | 1561 | |
| 0323927, render-v8 | 1566 | |
| 6abc2c4, dagger | 1572 | |

The design merges a rung that is not adopted with its switch off. After the reads, the branches
restack into one stack with `GH_REPO=youtalk/maker-arm-lab gh stack`, with this report on top.

At 21:46 PR #165 (`feat/ms7f-levers3`) and PR #166 (`docs/ms7f-report`) joined the stack. Stack #164
holds 12 PRs, and CI passes on both new ones.

### 11.2 The deviations and the protocol changes

| time | ledger word | what happened | why |
| --- | --- | --- | --- |
| 10:16 | D1 | The run starts on 2026-09-30 10:13, one day before design day 1. The clock deadlines stay as upper bounds. | A phase starts as soon as its inputs exist. |
| 10:16 | D2 | Every branch starts from 29a36ef (PR #150), not main dbc6a8c. | The owner allowed `--snap_closed` while it merged. Resolved at 11:31: #150 and #151 merged into main 9893172, which adds only a doc, so no branch moves. |
| 11:31 | | The protocol carries `--snap_closed`. | The MS7e R15 row adopted it: 925 against 912 of 1000 at 1.135 against 1.242. |
| 11:45 | | H0 reads on the additive model. | The policy forward and the renders do not grow with the part count. |
| 12:24 | D3 | minifig_3629 leaves the family. | The box grasp model cannot hold an upright hat, and the throughput bar charges every wasted cycle. |
| 12:40 | | Track S moves to the recorder recipe's own pace. | The policy follows its corpus pace, and the recipe reads 16.38 s per stow. |
| 15:23 | D4 | The classifier's render set adds kinds in no catalog and no split. | More kinds teach category cues, and the held-out kinds stay unseen. |
| 15:33 | | The wave recipe is M5 with `--end_at_done`. | It costs 0.012 of share and saves 33 % of the time. |
| 15:34 | D5 | The stack order is base, heap, parts, speed, classifier. | The waves do not need the classifier, so the wave SHA can be fixed while C trains. |
| 15:56 | | `--book_miss` joins the teacher reads. | The stray loop of the R4 reads. Reversed at 19:29. |
| 15:56 | | Wave 1 records on x86 only. | The deep tray runs about three times slower per step on Machine C. |
| 17:04 | | Wave 1 stops and relaunches at 19:27 with the same seeds. | The abandon and the camera clip blockers. |
| 17:42, 03:21 | R1 | A render extension range, run 20310000 + i, i = 0 to 999, then 0 to 1999. | The classifier needs more distinct lifts. |
| 19:29 | | `--book_miss` leaves every teacher read. The R9 verdict stays a miss. | Its cell bookings exclude whole cells. |
| 23:03 | | Wave 2 records on Machine B and Machine C. | Machine A's share, p 12 to 19, waited for a free Machine A and moved to Machine B at 23:47. |
| 23:42 | R6 | C1 misses, and model v5 serves. | C has no fallback. |
| 23:47 | R14 | The 50k screen reads again in a fresh TAG. Reads use local Machine A aliases. | A detached job had no ssh agent. |
| 02:20 | D6 | Retrain F2 starts at the end of the F1 screens. | The deep-tray reads take hours, and the GPU sits idle. |
| 02:54 | | Retrain F2's data path moves off Machine B. | Machine B's disk is full. |
| 02:57 | D7 | `--action_warp`, a serving speed lever with no retrain, and its screens. | No retrain on the waves reaches 300 an hour unless the share and the cycle move together. |
| 03:20 | D8 | Candidate 1's checked read starts after the warp screens and reads at the winning X. | Machine A cannot run both, and Machine B is out. |
| 04:07 | D9 | `--passenger_drop` gets one screen. | Passengers cost about 9 % of the parts. |
| 04:11 | | Model v6a, a ResNet50, replaces v5 from the checked read on. The design names a ResNet18. | v6a passes the pre-registered v6 bar. |
| 05:19 | D10 | Retrain F2's screens run now, on Machine C. | Machine B's disk is full, and Machine A holds the other screens. |
| 05:42, 08:17 | D8 | The checked read takes Machine B lanes again: 12 + 8, then 10 + 10. | The owner approved the Machine B cleanup, and Machine B sat idle from 05:45. |
| 08:20 | D11 | F2 100k and F2 200k read the screen protocol on x86, paired with F1 100k on v6a. A diagnostic for a third wave. | The D10 screens differ from F1's in host and classifier. |
| 10:38 | D12 | Owner decision: the staged bars 0.85, 200 an hour and a held-out row of 0.80. The design's bars stay as a stretch row. R11's trigger becomes a miss of any staged bar. | The owner decided at 10:35, after the screens and with 2 of the 20 checked heaps visible. The speed bar is out of reach this week. |
| 10:38 | D7 | The extension adds X = 2.0 to the warp screens, paired with them. | Pre-registered before any number. It is rejected at 13:08. |
| 11:07 | | Two lever subagents for the share bar: the in-loop class read and the riders. | The share bar of D12. |
| 11:53 | D13 | The pick of candidate 2 moves to x86 screens with v6a at X = 1.0. F2 50k and 150k read on Machine B. | The aarch64 screens run about four times slower, about 6 h a heap. |
| 13:07 | D14 | Lever screens on candidate 1 at X = 1.5 with v6a: base, `--missed_lift`, `--lift_view_steps 6` and both. Candidate 2's checked read waits for them. | The lever reports of 12:54. |
| 13:20, 14:58 | | The stack restacks twice: a linear chain at 13:20, the CI fix 83ed51d at 14:58. Every read keeps its code. | The heap branch carried extra commits, and a GL skip marker had slid off a test. |
| 13:39 | D15 | One stretch run, r = 0, report-only, starts with candidate 2's checked read, with the tray refill. | No official read can come in time for two 1000-part runs of about a day each. |
| 14:11 | | `--pile_pgs` (draft PR #153 from another session) is noted and not adopted. | The remaining reads are pre-registered under Newton, and the policies trained under Newton. |
| 15:41 | R1 | The seeds of classifier v7: render heaps 20311200 + i, i = 0 to 799, and a deep-tray C1 set 20312000 + i, i = 0 to 99. | v7 needs deep-tray renders. |
| 15:41 | | Classifier v7 is pre-registered with its bar. | The in-loop read trails C1, and C1 used the normal tray. |
| 15:45 | | A minimal patch lets the render script take the MS7f switches (5e92758). | The render script never took the deep tray. |
| 15:46 | D14 | `--missed_lift` serves alone. `--lift_view_steps 6` passes alone and loses in the pair. | The pair reads 203.5 an hour, under 0.97 x 210.5. |
| 17:58 | D16 | Screens of `--cell_failure_cap 5` and `--carry_votes 2` on candidate 2. Candidate 3 gets a checked read only on a difference from candidate 2. | The early look at f_F2_chk: about 16 parts a heap left in excluded cells. |
| 17:58 | | f_lever3_screens runs as a user systemd unit over the local alias `a_fr6l`. | Machine A's ssh agent is gone, so Machine A self-ssh fails. |
| 18:30 | D10 | The aarch64 F2 screens end as a record: F2 150k reads best, as on x86. | No decision rides on them. |
| 20:34 | D16 | `--cell_failure_cap 5` is adopted, and `--carry_votes 2` is not. Candidate 3 is candidate 2 with the cap. | 435 of 600 against the base's 414, and 415 under the 420 of the rule. |
| 20:53 | | v7 misses every check of its bar, and v6a stays. f_F3_chk starts with v6a at 20:52. | v7 reads 0.897 on the deep-tray set and 0.909 on the old set. |
| 21:34 | D17 | Owner decision: continue past candidate 3 with two levers in parallel through 10-03, then a checked read on 10-04 and the official read only on a pass. Lever (a) is classifier v8 on the gate scene, with its bar. Lever (b) is wave 3, the teacher's takeovers of the served policy, and retrain F3 warm from F2 150k. | The owner was asked at 21:20, with candidate 3's checked read running and no number of it seen. |
| 21:34 | R1 | The seeds of v8. Deep-tray render heaps use 20311200 + i, i = 12 to 399 and 424 to 799. The deep-tray C1 set grows with 20312000 + i, i = 24 to 99. The renders can use `--pile_pgs`, and every read stays on Newton. | Amended before any use. The classifier sees images. |
| 21:46 | D18 | `--pile_pgs` (`feat/ms7f-pgs` c657f61) gets a paired policy screen and a paired teacher read. On parity, PGS serves every later screen, the wave 3 recording and the final reads. | The owner's prompt: "why not use PR #153 to speed up the trial and error?" A comparison stays paired under one solver. |
| 22:16 | D18 | Amended before any D18 number: wave 3 records under Newton, and the teacher pair is dropped. PGS's adoption rests on the paired policy screen alone. | The policies and corpora are Newton's. |
| 22:59 | D18 | PGS is not adopted for reads. It stays for the v8 renders. | The PGS screen reads 449 against 435 correct stows and 206.3 against 191.9 an hour, outside the tolerance of 12 stows and 5 %. |
| 00:52 | | f_F3_chk fails all three staged bars and the stretch row. No official read runs. | 2181 of 3000, 189.5 an hour, held-out row 0.666 (D12). |
| 06:17 | | v8 misses 3 of the 4 checks of its bar, and v6a stays. | v8 reads 0.907 on the deep-tray set against a bar of 0.936, and 0.922 on the old set against 0.929. |
| 06:40 | D19 | Owner decision: the gate scene becomes the refilled deep tray, `--deep_tray --tray_refill`, still 150 parts a heap. The bars stay D12's staged ones, and the full-pile results stay as the harder row. The official read runs on the first checked read in this scene that passes. Retrain F3's screens move to this scene. | Decision F6's own fallback. The owner was asked at 06:35, after the stretch run. |
| 07:22 | | An interim look at 8 heaps of f_F3R_chk and 3 of f_M_cap5_R_scr, disclosed as not a verdict before the next rule. | The ledger discloses every number seen before a rule. |
| 07:22 | | Classifier v9e, the ensemble of v6a and v8, is pre-registered with a C1 bar and a paired refill screen. | v8 lifts special and v6a holds brick, so their errors differ. |
| 07:53 | | f_F3R_chk fails the staged share and held-out bars and passes the rate bar. No official read runs. | 2297 of 3000, 204.7 an hour, held-out row 0.735. |
| 07:58 | | v9e passes its C1 bar. | 0.9235 on the deep-tray set against a bar of 0.916, and 0.939 on the old set. |
| 08:00 | | The pick of candidate 4 is pre-registered: the best F3 refill screen with at least 474 correct stows, and v9e on a passing screen. | After the F3 50k screen and f_F3R_chk, before any other F3 refill screen or v9e screen number. |
| 08:31 | | v9e is adopted for every later read. | 474 of 600 at 215.2 an hour, at the bar of 474 and 200.8 exactly. |
| 08:32 | R1 | Wave 4 and retrain F4 are pre-registered. Wave 4 takes run 20308200 + p and layout 20308300 + p, p = 0 to 89. | The D17 DAgger lever once more, in the D19 gate scene. |
| 09:00 | | Candidate 4 = F3 150k with v9e. | F3 150k reads 498 of 600, the most of the four F3 refill screens. |
| 09:05 | | Wave 4's Machine B share relaunches on 5 lanes, not 10. | Its first 10 lanes failed on CUBLAS_STATUS_ALLOC_FAILED next to f_F4R_chk's 10 policy servers. |
| 09:59 | | f_F4R_chk fails the staged share and held-out bars, passes the rate bar and meets the stretch rate. No official read runs. | 2467 of 3000, 300.0 an hour, held-out row 0.786. |
| 10:00 | D20 | `--passenger_drop`, `--lift_view_steps 6`, `--carry_votes 2` and the first two together get one screen each on candidate 4 in the refill scene. The best one is adopted with at least the pair's correct stows plus 6 and at least 240 an hour. | Candidate 4 has rate to spare, 300.0 an hour against the staged 200. |

The `--pile_pgs` switch (14:11) comes from another session at the owner's request, as draft PR #153
on `feat/ms7f-pile-pgs` from 03a5cd8. While the largest constraint island has over 600 degrees of
freedom, each physics step uses PGS at 50 iterations instead of Newton. The full deep-tray island
holds 846 of 908 degrees of freedom, and Newton spends 88 of 90 ms a step on it. The switch gives
about 4.5 times the steps an hour on the teacher at a full heap, and resets stay on Newton. MS7f
does not adopt it. The remaining reads are pre-registered under Newton, and the policies trained
under Newton. A switched read is a new baseline that first needs a paired teacher read. It does not
touch the stretch run's slowness, which comes from 950 parked bodies and the refill tray. Deviation
D18 (21:46) tested it again as a wall-time lever, and its paired policy screen read outside the
parity tolerance (section 9.16). So PGS serves no read, and it serves only the v8 renders.

The ledger corrects itself at 10:54. The stamps of the rules, the ledger and the agents were
estimates. They now carry the file times: rules 10:15, ledger 10:16, agents 10:20. Every later entry
takes its time from `date`.

The ledger notes of 05:42 record seven more findings from the first draft of this report:

1. The R9 reference read's sixth heap (`reads/f_ref`, eval_005) was never logged. The verdict used
   five heaps.
2. The R9 count of 106 parts left is 750 - 641 = 109.
3. At 03:15, 26,707 steps over 95 cycles gives 384 an hour, not 383. The difference is rounding.
4. D7's 13.85 s a cycle is wave 1's Machine A half only.
5. The C1 set grew from 3122 lifts (v2, v3) to 3147 (v4 on). The render set was rendered again at
   the native crop.
6. v6a is a ResNet50 where the design names a ResNet18. This is an unnumbered change that the v6 bar
   adopted.
7. `origin/feat/ms7f-heap` is 9547db2, 3fa9f3f plus the heap-w1 commits. The stack keeps heap-w1
   4464b79 as the wave base.

### 11.3 The defects found and fixed

| found | defect | effect | fix |
| --- | --- | --- | --- |
| 12:24 | A y grasp of joint_3938 closes on a 1.8 mm rib under the top slab. | 113877 steps in 56 cycles on one kind | c3ca2d6 closes 3938 along x and z |
| 15:56 | The rig gives up on a stray part and delivers a neighbour. The loop books no failure. | 40 cycles and 136k to 160k steps on one stray part | `--book_miss`, later out of the teacher reads |
| 17:04 | In the recorder, a watchdog abandon ends the episode, and the export drops it. | A whole 150-part heap lost per trip | 125d86e `--abandon_cycle` |
| 17:43 | At K over 50, the spawn row grows `stat.extent` to 7.22 m, and the wrist camera clips at 72 mm. | The jaws vanish from every wrist frame. | 4c2575d spawns parts in rows of 50 |
| 17:43 | The no-truth test's lift budget was too small for the tray scene's events. | | 4bc7e39 raises it from 150 to 750 |
| 19:27 | The deep tray drop read the pile top from stale body poses. | | 4464b79 |
| 19:29 | `--book_miss` books a miss on the target's whole 5 cm cell at every stow of another part. | The selector never reaches the parts in a cell with three bookings. | `--book_miss` leaves every teacher read |
| 20:57 | The no-truth scene's stub classifier read a passenger. | | it now reads none |

Seven process faults also cost time:

- 16:00: a tracked background call stops after 2 hours, and a deep-tray wave lasts longer.
  Machine A aliases now reach Machine A over ssh to its own address, so fleet.py detaches their
  jobs with setsid under sshd.
- 20:57: orphans of stopped agents held seven Machine A cores, and a census queue runner kept
  starting heaps. They were killed by their own command lines.
- 23:21: the first screen chain named a worktree that does not exist. A `pkill -f` pattern then
  matched a later argument of the same command line and killed the calling shell.
- 23:47: a detached job had no ssh agent, so the 50k screen ended READ_FAIL.
- 01:48: Machine B's disk filled (984 GB), and its wave 2 conversion failed. The owner approved two
  cleanups at 05:42 and 05:55, and Machine B then had 222 GB free (section 8.3).
- 17:58: Machine A's ssh agent is gone. Git runs over HTTPS through the override in
  `e/git_https_note.txt`, and the D16 screens run as a user systemd unit.
- 09:05: wave 4's first Machine B share asked for 10 more policy servers on the RTX 5080. f_F4R_chk's 10
  servers held 9 GB of its 16 GB, so every heap failed within 25 s on CUBLAS_STATUS_ALLOC_FAILED.
  The share relaunched on 5 lanes (section 9.19).

## 12. What the simulation does not promise for the real arm

MS7f runs in simulation only (MS7f design section 13):

- The events take zero simulated time. A real person takes time to refill a tray or empty a box,
  so a real hour holds fewer stows. The payload counts the events.
- The deep tray's walls of 0.045 m are `INITIAL:`. No real tray measured them.
- The classifier learns from renders only. A real category classifier needs real crops. The render
  accuracy promises no real accuracy.
- The narrow overhead camera assumes 0.5 mm a pixel at the heap. The real stream is compressed, and
  its focus and motion blur are not modeled.
- The renders show no prints. A real minifigure head or torso carries a printed design.
- The joint and minifig parts are convex hulls from CoACD, capped at 4 per part. A real Technic pin
  is hollow and has friction ridges, and a real torso is hollow.
- The speed trims keep the command clamp and the torque caps of the SDK profile. The simulation has
  no motor heating, no CAN latency and no current limit beyond the torque cap.
- The passenger head learns from the simulator's counts. A real passenger can hide behind the jaws.
- `--action_warp` plays a chunk faster than the corpus taught it. The SDK clamp still applies to
  every command, but no real arm tracked a warped chunk.
- The x86 and aarch64 builds do not share physics. A single simulator number is a number of one
  architecture.
- Every perceived input except the class keeps the MS7d truth-plus-noise stand-ins at nominal
  knobs.

## 13. What is next

The steps, in order. Each step names the sink of section 10 that it targets:

1. Cut the wrong boxes, now the largest loss. Candidate 4 puts 376 parts in a wrong box, 376 of its
   533 misses, and the staged bar needs only 83 more correct stows. v7 and v8 each missed their
   bars. More deep-tray renders moved the line between plain bricks and the brick-like specials
   instead of sharpening it. The ensemble v9e passed and serves candidate 4, but the wrong-box parts
   stay at 375 and 376 in the refill scene. The classifier's `wrong_box` cycles take 194 of them,
   and parts without a cycle of their own take 145. Neighbour grasps (8.4 % of lifts, final class
   right on 0.517) and early overhead misreads stay open beside it.
2. Fix the grasp approach behind `close_miss`, still the largest time sink. On candidate 4 it holds
   0.254 of the steps, and 327 of a `close_miss` cycle's 373 steps pass before the first close. A
   census of these approaches comes before any lever.
3. Read what the corpus under the loop that serves buys. Wave 3 (D17) was the first such corpus, and
   retrain F3 on it gave candidate 4: +170 correct stows and +95.3 an hour over candidate 3 in the
   refill scene, with v9e. Wave 4 repeats the lever in the refill scene with candidate 4 driving,
   and retrain F4 trains on it (section 9.19). More data of the wave 1 recipe buys little (D11).
4. Read the D20 switches (section 9.20). Candidate 4's rate leaves room for `--passenger_drop`,
   `--lift_view_steps 6` and `--carry_votes 2`, which the earlier screens turned down for their rate
   cost or their count.
5. Keep the unboxed parts in reach. `--cell_failure_cap 5` cuts the unboxed parts from 385 on
   candidate 2 to 309 on candidate 3. Candidate 4 leaves 157 in the refill scene, about 8 a heap.
6. Keep `--pile_pgs` (D18) out of the reads. Its paired screen reads outside the parity tolerance,
   so a PGS read is a new baseline. On a policy read it cuts the wall time a heap by about 1.7
   times. It still serves the renders, where the classifier sees only images.
7. Run R13's two stretch runs on any official candidate. The stretch run r = 0 is the record of
   continuous operation so far: 801 of 1000 at 259.9 an hour, with 85 interventions per 1000 parts
   (section 9.13).
8. Merge stack #164, 12 PRs with #165 for `feat/ms7f-levers3` and the report PR #166 on top.
   `feat/ms7f-pgs`, `feat/ms7f-render-v8` and `feat/ms7f-dagger` have no PR yet.

With a pass of a checked read in the refill scene of D19, the official read and R13's stretch runs
come before step 1.

The closing paragraph, written for the case where no checked read in the refill scene of D19 passes.
MS7f built the real-scale task in simulation and ran it in every read: 150-part heaps in a deep
tray, box emptying, six categories over 66 kinds and a classifier trained on renders. It did not
meet its gate. The owner's staged bars of D12 are missed on every checked read on the full pile:
candidate 1 at 2043 of 3000 and 140.9 an hour, candidate 2 at 2065 and 189.1, and candidate 3 at
2181 and 189.5. In the refilled deep tray of D19, candidate 3 reads 2297 at 204.7 and candidate 4
2467 at 300.0. Both pass the staged rate and miss the share and the held-out row. Candidate 4
reaches the design's stretch rate of 300 an hour (299.97 in the payload). It stays 83 correct stows
and 11 held-out parts under the staged bars. The levers bought rate and share. `--missed_lift` took
the policy from 140.9 to 189.1 an hour, and `--cell_failure_cap 5` added 116 correct stows at the
same rate. Retrain F3 on wave 3 with classifier v9e added 170 more in the refill scene. The wrong
boxes hold most of what is left: 376 of candidate 4's 533 misses. The next milestone starts from
these sinks.

TBD: the closing outcome. It waits for D20, wave 4 and retrain F4, and any checked read of candidate
4' or candidate 5 (sections 9.19 and 9.20). On a pass of a checked read in the refill scene, the
official read's outcome replaces the paragraph.
