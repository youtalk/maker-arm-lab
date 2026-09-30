# MS7e: more boxes, more part families and unseen box conditions

Date: 2026-09-30. Status: final. The official read passes both bars, and MS7e meets its gate with 4
to 8 drawn boxes, eight classes and all three new part families.

The design is `2026-09-29-ms7e-widened-task-design.md` in the claude-memory repository. This report
cites it as "design section N". The owner approved it section by section in chat. The campaign
ledger is `~/data/maker-arm-ms7/e/README.md`, and `STATE.md` beside it is the resume pointer. Every
rule below went into that ledger before its number. The measurement tables are under
`~/data/maker-arm-ms7/e/tables/`. Each read has a TAG, its name. The read payloads, the JSON results
of each read, are under `~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7e TAG starts with `e_`.

The counts that the ledger does not give come from those payloads. The cycle classes come from
`~/data/maker-arm-ms7/d/bin/row_report.py`. The cycles per family and the end places of lost parts
come from the E1 census script, `~/data/maker-arm-ms7/e/tables/e1_census/census.py`. That script
has its own end-place test, so its stow counts differ from the gate's by a few parts. This report
takes every stow count from the gate.

The ledger names its rules R1 to R14 and its deviations D1 to D11. The design names its decisions E1
to E9, and the two gate candidates are also named E1 and E2. This report writes "rule R4", "decision
E7" and "candidate E1" to keep them apart.

All times are PDT. Each time is the time of the ledger entry that records the event. Times from
15:50 to 23:57 are on 2026-09-29. Times after midnight are on 2026-09-30.

## 0. Summary

MS7d passed its gate on a desk with four boxes of one size, four part classes and a catalog of 29
bricks and slopes. MS7e widens that task in simulation along three axes (design section 1):

1. More destinations. The desk holds 4 to 8 boxes, and there are 8 classes.
2. Unseen box conditions. Each episode draws the box count, the interior size and aspect, and the
   rim height.
3. More part families. Round bricks, LDraw parts and flat plates and tiles join the catalog.

A rung is one change with a bar that the ledger fixed before its number. The teacher is the scripted
oracle that records the corpora. The teacher tries each rung first, and only the rungs that pass
their bar enter the corpus.

The results:

- The box ladder reaches rung B2, 4 to 8 boxes per episode. The side band of the design fits only 4
  boxes, so deviation D6 widens the B1 band to fit 6. Deviation D7 moves the back edge of the B2
  band from x -0.06 m to -0.07 m. That band places 8 boxes.
- The draw can put two large classes in one small box. Rule R11 adds a capacity fit to the map from
  classes to boxes. The teacher stows 275 of 300 with the fit, against 264 without it.
- All three families pass their lone-part tables and their heap reads, and all three join the
  corpus. A lone-part table runs one part through one teacher cycle, 12 times a part. A heap read
  runs the teacher on 50-part heaps. Round joins with collision studs and a rolling friction. Flat
  needs a deeper box floor collider, a fix of a simulation artifact. LDraw drops its hull cap from 8
  to 4 for speed (deviation D8).
- The first teacher reference read on the full wave configuration misses its bar. A reference read
  is a 150-part teacher read before a wave. It stows 127 of 150 parts, 0.847, against 0.90. A census
  (deviation D10) finds three mechanisms.
- Two new teacher levers remove two of the mechanisms. T1 is a carry path around the arm base. T2
  refuses a lift that carries a second part. The second reference read passes at 137 of 150, 0.913.
- Two corpus waves of 120 heaps each stow 0.923 and 0.917 with the teacher. Both lie within 0.03 of
  the reference read.
- Retrain 1 starts from the weights of train_P 200k and trains on wave 1. The screens, one 500-part
  read per checkpoint, pick train_E1 200k. That checkpoint is candidate E1.
- Candidate E1's checked read passes both bars. It stows 2614 of 3000 held-out parts in their own
  class's box, 0.871, at 1.275 cycles per correct stow. The bars are 2550 and 1.3.
- 1.275 is above the retrain trigger of 1.25 cycles per correct stow. So retrain 2 starts at 01:01,
  from candidate E1 on waves 1 and 2.
- Flat parts are the largest cost of candidate E1. On the checked read, plates stow 0.703 and tiles
  0.707. The E1 loss census puts 64 to 68 % of the excess cycles of the screens on flat parts. An
  excess cycle is a cycle beyond one per correct stow.
- Rule R13 names three serving levers. S2 snaps the pick point of a flat part to the desk. S3 tests
  the tray walls at the landing band of the continuous jaws. Both pass their rows and join the later
  reads. S1, a passenger check in the serving loop, cannot see the passengers with the overhead
  camera. It stays unbuilt.
- Retrain 2's screens pick train_E2 150k, candidate E2. With S2 and S3 on, its checked read stows
  2673 of 3000, 0.891, at 1.224 cycles per correct stow.
- Both checked reads pass with every family. Rule R2's tie-break, fewer cycles per correct stow,
  picks candidate E2 for the official read.
- The official read passes. It stows 5299 of 6000, 0.883 (Wilson 0.875 to 0.891), at 1.271 cycles
  per correct stow. It draws 4 to 8 boxes (rung B2) and uses the eight classes and all three new
  families.
- By box count, the official read stows 0.874 to 0.893. By family, it stows 0.670 (tile) to 0.952
  (round). Plates stow 706 of 925 and tiles 325 of 485. The flat parts use about two thirds of the
  excess cycles.
- The control read, train_P 200k with no retrain, stows 551 of 1000 at 2.076 cycles per correct stow
  on the base families. On its own MS7d task it stowed 0.920. So the widened boxes cost it 0.37, and
  the retrains buy that back.
- Every row of the unseen box ladder holds the stow bar. The unseen size row and the unseen layouts
  row also hold the cycles bar. The seen row and the unseen rim row go 0.006 and 0.010 over it.
- The campaign ran from 2026-09-29 15:45 to 2026-09-30 05:38, inside the four days of decision E5.
  It used the planned retrain and the one extra retrain that decision E5 allows.

The gate outcome:

| read | checkpoint | commit | correct stows | cycles per correct stow | outcome |
| --- | --- | --- | --- | --- | --- |
| candidate E1, checked | train_E1 200k | f8faeee | 2614 of 3000, 0.871 | 1.275 | PASS, the retrain trigger fires |
| candidate E2, checked | train_E2 150k | 3e9ee7b, with S2 and S3 | 2673 of 3000, 0.891 | 1.224 | PASS, picked by rule R2 |
| official | train_E2 150k, candidate E2 | 3e9ee7b, with S2 and S3 | 5299 of 6000, 0.883 | 1.271 | PASS |
| control A | train_P 200k | f8faeee | 551 of 1000, 0.551 | 2.076 | no bar, reported |

The official read is the milestone's number. It passes at 5100 of 6000 correct stows or more and at
most 1.3 cycles per correct stow. It reads 5299 at 1.271, so MS7e meets its gate.

## 1. The goal, the decisions and the rules

### 1.1 The starting point and the three axes

The MS7d task (design section 3) has four desk boxes with a 0.09 x 0.09 x 0.04 m interior. Each
episode draws every box centre in a side band, x 0.05 to 0.21 m and |y| 0.18 to 0.22 m. Each box
yaws by up to 45 degrees either way. The catalog holds 29 parts in four classes. The split with
`parts_seed` 20260915 holds out seven of them. Held-out parts are parts that no training corpus
contains. Flat parts and round bricks left the catalog on 2026-09-18.

MS7e keeps the MS7d protocol and adds the three axes of section 0. It uses eight classes, one per
family group (design section 5):

| class | rule | parts |
| --- | --- | --- |
| slope | family slope | 6 |
| tall | a brick taller than one brick height | 5 |
| narrow | a one-height brick one stud wide | 8 |
| wide | a one-height brick two or more studs wide | 10 |
| round | family round | 4 |
| special | family LDraw | 8 |
| plate | family plate | 6 |
| tile | family tile | 4 |

The class fold is the per-episode map from the eight classes to the boxes. When there are fewer
boxes than classes, it puts several classes in one box.

### 1.2 The decisions of the design

The owner took these decisions on 2026-09-29, in chat (design section 2):

| id | decision |
| --- | --- |
| E1 | A simulation milestone that widens the task (roadmap section 9.1), on the GPU hosts, in parallel with MS5 phase P. |
| E2 | Three axes: more classes and boxes, unseen box conditions and more part families. Larger heaps and continuous operation stay out, for MS8. |
| E3 | The families are flat plates and tiles, round bricks with collision studs, and LDraw parts. They join one at a time, with the flat parts last. |
| E4 | The MS7d system: `--perceived` and the nine MS7d levers at nominal knobs. The gate is at least 0.85 of the parts in their own class's box and at most 1.3 cycles per correct stow. |
| E5 | Four days, 2026-09-30 to 2026-10-03. One planned retrain and at most one more. No extension. |
| E6 | The interior size and aspect, the box count and the rim height vary per episode. The box color does not. |
| E7 | The box ladder: rung B1 in the measured side band, and rung B2 with a release lean and a camera shift for 7 to 8 boxes. |
| E8 | Eight classes by family. The base split stays, and each new family holds out 20 % of its parts. |
| E9 | The lone-part work of all three families starts in parallel on day 1. The flat parts keep the latest deadline. |

### 1.3 The gate and the reads

A correct stow is a part at rest in its own class's box. A cycle is one attempt on one target part.
The cycles per correct stow of a read is its cycle count divided by its correct stows.

Rule R2 (design sections 7.3 and 8.2) sets the gate. A checked read passes with at least 2550
correct stows of 3000 and at most 1.3 cycles per correct stow. The official read goes to the passing
checked read with the most families. A tie goes to fewer cycles per correct stow. The official read
passes with at least 5100 of 6000 and at most 1.3, and it is one shot. If no checked read passes,
the milestone fails, and the report gives the decomposition.

| read | policy | seed and size | pass |
| --- | --- | --- | --- |
| screen | each retrain checkpoint | 20281100, draws 0 and 1, 500 parts | picks the checkpoint |
| checked | candidate E1, and candidate E2 after retrain 2 | 20281700, draws 0 to 11, 3000 parts, a fresh TAG | at least 2550 correct stows, at most 1.3 cycles per correct stow |
| official | the checked read that rule R2 picks | 20282100, draws 0 to 23, 6000 parts, one shot | at least 5100 correct stows, at most 1.3 cycles per correct stow |
| control | A, on the base families | 20281700, draws 0 to 3, 1000 parts | none, reported |

The candidates (rule R8):

| candidate | policy | status |
| --- | --- | --- |
| E1 | retrain 1's best screen | a gate candidate |
| E2 | retrain 2's best screen | a gate candidate after retrain 2 only |
| A | train_P 200k with no retrain | a control, not a candidate |

Rule R14 (03:14) was set before candidate E2's checked read. Each candidate's official read uses
the protocol and the SHA of its own checked read. For candidate E1 that is f8faeee with no R13
lever. For candidate E2 it is 3e9ee7b with `--thin_snap --wall_band_cont`.

### 1.4 The read protocol

Rule R3 (design section 8.1) keeps the MS7d protocol of `d/bin/read_d.sh`. The serving loop runs the
policy and its serving rules. A serving rule changes the policy's command, or gives the env to the
teacher for a few steps. The protocol's scene flags are `--tray --desk_boxes 4 --tray_lean 30`. Its
serving switches are `--scripted_rake --part_failures --pick_snap --dwell_close --adopt_held
--park_done --rake_turn --perceived`. Its client flags are `--select_at_release --replan_on_switch
--squeeze_floor --square_wrist`.

Each read adds the nine MS7d levers at nominal knobs, with every noise scale at 1: `--box_filter_xy
--release_gap --rearm_hold --hold_force --force_cap --marker_bias --wrist_fix --class_views
--box_walls`. Each read also adds the switches of the adopted MS7e rungs. `e/bin/read_e.sh` builds
every MS7e read with `DESK 8` and this extra set: `--families round,ldraw,flat --box_draw 8
--release_lean --carry_via --passenger_check --deep_box_floor`. Candidate E2's checked read, the
official read and the ladder rows also add the two adopted R13 levers, `--thin_snap
--wall_band_cont` (section 6).

A read uses held-out parts, heaps of K = 50 and 50 actions per forward. A heap is K parts that the
scene drops into the tray. The actions per forward are the actions that one policy call returns.
Every read reports rows per class, per family and per box count.

### 1.5 The seeds

Rule R1 copies the seed table of design section 8.4:

| use | seed |
| --- | --- |
| part split | `parts_seed` 20260915, unchanged |
| lone-part tables | 20289000 + the part's catalog index |
| family heap reads | run 20288000 + i, layout 20288500 + i |
| teacher reference reads | run 20288900 + i, layout 20288950 + i |
| B2 envelope | 20289500 |
| count ceiling | 20290000 to 20290999 |
| wave 1 | run 20284000 + p, layout 20285000 + p |
| wave 2 | run 20286000 + p, layout 20287000 + p |
| screens | 20281100, draws 0 and 1 |
| checked | 20281700, draws 0 to 11 |
| official | 20282100, draws 0 to 23 |
| ladder rows | 20283100 (size), 20283200 (rim) |

The later rules add seeds that no other rule uses:

| rule | use | seed |
| --- | --- | --- |
| R11 | the box capacity fit | run 20289700 + i, layout 20289800 + i, i = 0 to 5 |
| D10 and R12 | the teacher census and the teacher levers | run 20289600 + i, layout 20289650 + i, i = 0 to 5 |
| R13 | the serving lever rows | 20281500, draws 0 to 5, 1500 parts |
| R8, set up at 04:21 | the seen ladder row | wave 1's run 20284000 + i and layout 20285000 + i, i = 0 to 19, the first env only, 1000 parts |

The lever smokes of rule R13 ran on the throwaway seeds 7 and 8. A smoke is a short run that tests a
build, and it is not a read.

### 1.6 The rules that carry over and the hosts

Rule R9 carries over MS7d rule R8 (design section 8.5). The supervisor, `scripts/run_gate_eval.py`,
runs a read and grades it. A payload from the supervisor cache is not a read. Every read records its
SHA, its `policy_id`, its split, its switches and its knob scales. A READ_FAIL is a read with no
gate token, and it reads once more in a fresh TAG. The MS6 negatives and the generic abort on
contact of the QDD spike stay closed.

Rule R10 keeps `scripts/fleet.py` and the host table `~/data/maker-arm-ms7/hosts.json`. At 15:52 the
owner asked that Machine A, Machine B and Machine C all carry the pipeline. So each track got
remote aliases, and each wave records on all three hosts. Machine C runs at most 40 camera
processes in all, and at most 12 recording processes a wave.

## 2. The box ladder

### 2.1 The rungs and the bars

| rung | what changes | adoption |
| --- | --- | --- |
| B1 | 4 to 6 boxes in the measured side band, drawn per episode. No rig change. | the base, always adopted |
| B2 | Up to 8 boxes. The release leg leans toward the box, and the overhead camera moves back along x. The band reaches back to `X_BACK`, the back edge of the B2 band. | the bars of rule R4 |

The rig is the code that turns the teacher's goals into arm motion. Rule R4 (design section 4.4) set
four measurements, each bar before its number:

1. BOX_MIN, the size floor of the box draw.
2. The B2 envelope, with the release lean on.
3. The camera shift for B2.
4. The count ceiling of each rung, the largest box count that the draw places.

If the envelope passes with an `X_BACK` whose count ceiling is 7 or more, B2 is adopted. If not,
MS7e ships at B1 with a ceiling of at most 6.

Three terms name the parts of a teacher cycle. EXTEND is the leg that carries the lifted part to the
release point over its box. RETREAT is the leg back from the box. A converged leg ends at its goal
within the rig's tolerance.

The envelope check, `scripts/mj_release_envelope_check.py`, drives a lone part from the tray to a
grid of box poses in a band. It records whether the part lands in the box and whether EXTEND and
RETREAT converge. Each part gets 80 trials.

### 2.2 BOX_MIN

The rule: the envelope check runs on the B1 band with a 0.05 m deep interior. The parts are
brick_2x4, brick_1x2x5, brick_2x7, brick_1x8 and brick_2x8. BOX_MIN is the smallest interior side,
from 0.07 m in 5 mm steps, at which every part lands 80 of 80 with EXTEND converged.

BOX_MIN is 0.07 m at the first step (16:20, `tables/boxes_phase1.txt`). The B1 band of the design is
x 0.05 to 0.21 m and |y| 0.18 to 0.22 m. On it, a 0.07 x 0.07 x 0.05 m interior lands all five parts
80 of 80 with EXTEND converged. The largest EXTEND residual is 1.20 mm and the largest RETREAT
residual 0.82 mm. No arm link touches a box.

### 2.3 The B1 band: deviation D6

The side band of the design fits only four boxes. On that band, count 5 places in 0 of 1000 seeds
with either sampler, at sides of 0.07 to 0.08 m. Three boxes a side do not fit a 0.16 m range of
centres. The rejection sampler of the design never packs 5 boxes on any band tried. So the B1
ceiling as written is 4.

Deviation D6 (16:20) changes the B1 band and the sampler:

- The band becomes x 0.03 to 0.25 m and |y| 0.18 to 0.20 m.
- The sequential sampler replaces rejection sampling. It draws the sizes first. Then it places the
  boxes of a side along x in order and clips each yaw to fit. It tries 200 size draws.
- Every footprint corner keeps x at -0.015 m or more (`FRAME_BACK_X`). A 10 cm box yawed 45 degrees
  at x 0.05 m left the frame by 2.1 px. The unseen rim row needs -0.010 m.

The envelope check certifies the new band at 0.07 m and 0.10 m interiors. All five parts land in 80
of 80 trials. EXTEND ends within 1.56 mm and RETREAT within 0.84 mm, and the tightest joint margin
is 0.032 rad. The band keeps 1 cm from a RETREAT failure at x 0.02 m. It keeps 2 cm from misses at x
0.27 m and |y| 0.20 m. With BOX_MIN 0.07 m, counts 4, 5 and 6 place in 1000 of 1000 seeds, and count
7 places in none. The B1 ceiling is 6.

### 2.4 The release lean and the B2 envelope

The rig pins the tool lean to world +x on every leg. The probe of design section 3.1 found that no
box can stand behind the band with that lean. The release lean, `--release_lean`, changes EXTEND
only. EXTEND leans toward `atan2(box_y, box_x)`, the direction from the arm base to the box centre.
RETREAT turns back to +x.

The probe of design section 3.1 ran again at 16:05 with brick_2x4. Its strip is x -0.07 to 0.00 m
and |y| 0.14 to 0.22 m. With the lean off, 0 of 40 trials land. EXTEND ends 240 to 296 mm from its
goal. With the lean on, 40 of 40 land with EXTEND converged. This is the probe, not the rule R4
envelope.

The rule R4 envelope ran at 16:50 on Machine A, seed 20289500, 0.05 m depth and 80 trials a part:

| band | interior | result |
| --- | --- | --- |
| x -0.06 to 0.25 m, \|y\| 0.18 to 0.20 m, the lean on | 0.07 m (BOX_MIN) | every part 80 of 80 landed, EXTEND and RETREAT converged. EXTEND at most 0.64 mm, RETREAT at most 1.03 mm, tightest joint margin 0.429 rad. |
| the B1 band, the lean on | 0.07 m and 0.10 m | 80 of 80 for every part, no regression |

The first B2 read at 16:09 used a 0.07 m placeholder interior, before the box track measured
BOX_MIN. It covered x -0.06 to 0.24 m and |y| 0.18 to 0.22 m. Every part landed 80 of 80, with
residuals of at most 0.60 mm for EXTEND and 0.82 mm for RETREAT.

### 2.5 The camera shift

The overhead camera moves back along x by the smallest shift that passes two projection tests. Every
rim corner of a B2 box stays at least 4 px inside the frame, and the tray stays inside the frame.
Both tests use the extremes of the camera draw, plus or minus 0.01 m and 2 degrees per axis.

For `X_BACK` -0.06 m, the shift is 0.129 m back. The nominal camera x moves from 0.3011 m to 0.1721
m. The box margin is 4.18 px and the tray margin 67.63 px. For another `X_BACK`, the shift is 0.129
m plus the distance from -0.06 m back to `X_BACK`. The frame's back limit for the shifted camera is
-0.132 m at `X_BACK` -0.06 m. A first read at 16:09 gave 0.130 m, a box margin of 4.16 px and a tray
margin of 67.62 px.

### 2.6 The count ceilings and deviation D7

The count ceiling is the largest count that the box draw places in at least 990 of 1000 seeds,
20290000 to 20290999. The B2 rows use the sequential sampler of the box track (71ff2ed) with BOX_MIN
0.07 m.

| band | back limit | count 7 places | count 8 places | ceiling | envelope |
| --- | --- | --- | --- | --- | --- |
| x -0.06 to 0.24 m, the design band | -0.132 m | 0 of 1000 | | 6 | |
| x -0.06 to 0.25 m | -0.132 m | 345 of 1000 | | 6 | passes |
| x -0.07 to 0.25 m | -0.142 m | 1000 of 1000 | 996 of 1000 | 8 | every part 80 of 80 with EXTEND and RETREAT, tightest joint margin 0.436 rad, camera shift 0.139 m |
| x -0.06 to 0.26 m | -0.132 m | | 996 of 1000 | 8 | |
| x -0.08 to 0.25 m | | | 1000 of 1000 | 8 | RETREAT goes unplanned on one grid point for four parts |

As written, with `X_BACK` -0.06 m, the ceiling is 6, and B2 is not adopted. The design marks
`X_BACK` as `INITIAL:`, and its own procedure moves it in steps of 0.01 m. One centimetre back gives
a ceiling of 8 with a clean envelope.

Decision D7 (16:50) adopts B2 at `X_BACK` = -0.07 m. The B2 rung is:

- `--release_lean`.
- The band x -0.07 to 0.25 m and |y| 0.18 to 0.20 m, with a frame back limit of -0.142 m.
- The camera 0.139 m back, at a nominal x of 0.1621 m.
- A box count uniform from 4 to 8.

A draw that cannot place its count must not crash an episode. At count 8 this happens in 4 of 1000
seeds. The draw then tries again, and the episode records the number of extra rounds. The B1
fallback stays: if the wave 1 teacher reference read stows under 0.90 on B2, B2 leaves, and MS7e
ships at B1.

### 2.7 The per-episode draw and the resize

The box track built the draw at 17:04 (feat/ms7e-boxes 4c90d78, 1344 tests). The scene builds eight
box slots and parks the unused boxes at x -1.0 m. Each episode writes the size and the pose of each
box.

- The resize refits `geom_rbound`, `geom_aabb` and the body BVH, the bounding volume hierarchy of
  the body. Without the refit, a 1x1 brick fell through a grown box floor.
- The redraw rule keeps the count. A side that exhausts 200 size draws draws again, for up to 10
  rounds. The episode records `box_draw_rounds`.
- With every switch off, the draws and a 20-part teacher payload equal those of the base.

The training draw (design section 4.2) takes each interior side from 0.07 to 0.10 m. The aspect is
at most 1.4, and the depth is 0.03 to 0.05 m. The unseen box rows draw outside that range. In the
size row, every box has a side over 0.10 m or an aspect over 1.4. Its sides go up to 0.12 m and its
aspect up to 1.7. The rim row draws an interior depth of 0.05 to 0.06 m. Their count ceilings:

| rung | unseen size | unseen rim |
| --- | --- | --- |
| B1 | 4 | 6 |
| B2 | 6 | 8 |

### 2.8 The box capacity and rule R11

The box track's teacher smoke at 17:20 found a capacity limit (c_e1, 4c90d78, six train heaps,
base families only). Without the box draw, the teacher stows 276 of 300, 0.920, at 1.007 cycles per
stow. With `--box_draw 6`, it stows 263 of 300, 0.877, at 1.129. By box count, 4 boxes stow 41 of
50, 5 boxes 86 of 100 and 6 boxes 136 of 150. 25 parts end outside the tray, against 9. 13 parts
count as stowed at their cycle's end but lie outside their box at the episode's end.

The losses sit at small, shallow boxes that got two classes. A box of 8.1 x 8.0 x 3.5 cm holds 13 of
20 parts. A box of 8.9 x 7.1 x 3.2 cm holds 16 of 23. The parts spill to the desk or onto the rims.
The eight-class fold can put two large classes in one box, and the box draw of the design has no
capacity rule. With all eight families, the expected load of a train heap is 9.2 parts of 50 for
narrow and 10.5 for wide. The other classes take 3.9 to 7.9.

Rule R11 (17:20): the box track builds the smallest capacity fix under the box draw switch. It tunes
the fix on its own seeds: run 20289700 + i and layout 20289800 + i, i = 0 to 5, train split. The
scene is the full wave configuration on x86. The bar: the fix must stow at least 0.90 in the
correct box there.

The fix (feat/ms7e-box-load 46a144c, 17:41) works under `--box_draw` only. It draws the eight-class
fold again, up to 200 uniform draws, until every box has enough room. Every box's interior volume
must be at least 2.0 times the summed bounding volume of the parts it receives. If no draw meets
this, the fit keeps the draw with the largest least ratio. `BOX_VOLUME_PER_PART_VOLUME` = 2.0 is
`INITIAL:`. The census gave it: boxes spilled at ratios of 1.14 to 1.35 and held at 1.76 and above.

The R11 reads use these flags: `--tray --desk_boxes 8 --tray_lean 30 --box_draw 8 --release_lean
--families round,ldraw,flat --deep_box_floor`.

| read | TAG | commit, host | correct box | cycles per correct stow | outside the tray | by box count 4, 5, 6, 7, 8 |
| --- | --- | --- | --- | --- | --- | --- |
| without the fit | e_boxload_base | 1ef4829, Machine B | 264 of 300, 0.880 | 1.042 | 13 | 85/100, 43/50, 48/50, 48/50, 40/50 |
| with the fit | e_boxload_fit | 46a144c, Machine A | 275 of 300, 0.917 | 1.044 | 5 | 94/100, 45/50, 48/50, 48/50, 40/50 |

The fit passes its bar of 0.90, and rule R11 adopts it. The fit drew the fold again in 2 of 6 heaps.
The other 4 heaps ran step for step the same on both hosts. Most of the remaining losses are parts
left in the tray, 17. Tile at 17 of 20 and LDraw at 42 of 45 are the weakest families.

## 3. The part families

### 3.1 The lone-part table

The lone-part table (design section 6.2, rule R5) runs `scripts/mj_part_table.py` on a list of
parts. One trial drops one part at a random pose in the tray and lets it settle. The teacher then
runs one cycle into one desk box, from SETTLE to the end of RETREAT. Each part gets 12 trials,
seeded 20289000 plus its catalog index. A trial takes about 0.5 s (feat/ms7e-part-table 6f3fffb).

A family with a pooled stow rate under 0.90 tries at most two levers, each behind its own switch. A
family whose table reaches 0.90 runs its heap read: 4 heaps of 50 parts on the "all" split. Half of
each heap is the new family, and half is the parts adopted so far. The seeds are run 20288000 + i
and layout 20288500 + i. The bars: the family's own row must stow at least 0.85 in the correct box,
and the rest of the heap at least 0.90.

The control row stows 84 of 84: brick_2x4, brick_1x2x5, brick_2x7, brick_1x8, brick_2x8, slope_2x2
and brick_1x1, 12 each. It reads the same on both CPU architectures.

The round baseline showed that the two CPU architectures do not share physics. Machine A and Machine B
(x86) agree bit for bit. Machine C differs in 41 of 48 round records. The rule of 16:16: a
lever table compares against a baseline from the same architecture, x86 unless a row says otherwise.

### 3.2 The classes and the split: deviation D5

The base branch (feat/ms7e-base 86263d6, 1302 tests) adds the families, the eight classes, the rows
per family and per box count, and five switches. The switches are `--families`, `--box_draw`,
`--release_lean`, `--box_unseen` and `--heap_family`.

Deviation D5 (16:18): each new family shuffles its split on its own generator,
`Random(f"{seed}:{family}")`. So a family adopted between waves cannot move another family's
held-out parts. Brick and slope keep the MS6 generator, and the seven base held-out parts do not
move. At `parts_seed` 20260915 the new held-out parts are round_2x2x2, plate_1x4, plate_2x2,
tile_1x4, and the LDraw parts 3894 and 11211.

### 3.3 Round

The round parts are round_1x1, round_2x2, round_2x2x2 and round_4x4. They left the catalog in MS6
because a jaw touch made a cylinder on its side roll away. Their studs were visual only.

The baseline table on x86 (6f3fffb, 16:16) stows 45 of 48, 0.938, and lifts 46 of 48. round_1x1
reads 10 of 12. In two trials it lies on its side with no grasp, two pushes fail, and it stays in
the tray. round_2x2 reads 11 of 12. In one trial it still rolls at 0.097 m/s after the hold.
Machine C reads 47 of 48. The 16:15 entry first gave that value for the table, and the 16:16
entry corrected it.

The lone-part table does not show the roll at heap time that retired the family. In a heap, a jaw
touches a neighbour. So the heap read is the real test for round.

The round rung (feat/ms7e-round 896e9b9, 16:58) has three commits:

- e6e048a keeps only the studs whose centre lies on a round part's disc. round_4x4 goes from 16 to
  12 studs. This is visual only with the switch off.
- b5558cb adds `round_physics`: collision studs, condim 6, friction 1.0, torsional friction 0.005
  and a rolling friction. Condim is MuJoCo's contact dimension, and condim 6 adds the rolling and
  torsional terms.
- 896e9b9 marks "round" ready under `--families round`. Deviation D4: the family has no switch of
  its own beyond `--families round`, so the physics comes with the family.

`ROUND_ROLLING_FRICTION` = 0.001 is `INITIAL:`. A roll probe set it. With condim 4, kicked parts
roll 83 to 134 mm at the 90th percentile. A rolling friction of 1e-3 holds them to 1.3 to 8.5 mm. At
3e-4 they rock, and at 2e-3 they balance on the rim.

The lone-part tables on x86 and the R1 seeds (`tables/baseline_round.json`,
`tables/round/table12_nophys_896e9b9.json`, `tables/round/table12_phys_896e9b9.json`):

| part | baseline, 6f3fffb | the rung without the physics, 896e9b9 | the rung with the physics, 896e9b9 |
| --- | --- | --- | --- |
| round_1x1 | 10/12 | 11/12 | 12/12 |
| round_2x2 | 11/12 | 11/12 | 9/12 |
| round_2x2x2 | 12/12 | 12/12 | 12/12 |
| round_4x4 | 12/12 | 12/12 | 12/12 |
| family | 45/48 = 0.938 | 46/48 | 45/48 = 0.938 |

Both 896e9b9 tables run with the eight classes. Both family rates reach 0.90, so the track tries no
lever. The heap reads ran on Machine A at 896e9b9 through `bin/teacher_heap.sh`. Their flags are
`--tray --desk_boxes 4 --tray_lean 30 --families round --heap_family round`, split all, K 50, i = 0
to 3:

| TAG | studs | round row | rest of the heap | all | cycles per correct stow |
| --- | --- | --- | --- | --- | --- |
| e_round_heap_nophys2 | visual | 86/100 | 96/100 | 182/200 | 1.038 |
| e_round_heap_phys2 | collision, with `round_physics` | 88/100 = 0.880 | 95/100 = 0.950 | 183/200 | 1.011 |

Both reads meet both bars of 0.85 and 0.90. The ledger adopts round with the physics on, and round
joins wave 1 that way (16:58).

### 3.4 LDraw

LDraw is an open library of LEGO part models. The design (section 6.5) pre-registers eight LDraw
parts:

| id | part | split |
| --- | --- | --- |
| 3700 | technic brick 1x2 | train |
| 3701 | technic brick 1x4 | train |
| 3894 | technic brick 1x6 | held out |
| 4490 | arch 1x3 | train |
| 3659 | arch 1x4 | train |
| 87087 | brick 1x1 with a side stud | train |
| 11211 | brick 1x2 with two side studs | held out |
| 3660 | inverted slope 2x2 | train |

The LDraw rung (feat/ms7e-ldraw 19411af, 17:10, 1323 tests) has these commits:

- 6778c77: the uv dependency group `ldraw`.
- 7f2f958: `scripts/ldraw_parts.py`, the generator. It expands each `.dat` file into triangles. It
  splits the collision mesh into convex hulls with CoACD, a convex decomposition library.
- 1bebbc4: the JSON parts and the notice, from `complete.zip` of 2026-08-31.
- 418d1af: `Part.hulls`.
- 054f350: the mesh hull geoms and the finger forces by body.
- 3ac1482: a `special` detection grasps along its own part's closing axes.
- c7a1869: the family is ready.
- 19411af: the data fix of 3660.

The LDraw track recorded three deviations at 17:10:

- The generator fills enclosed cavities. The 1.6 mm walls of an LDraw part let a squeezing jaw
  through a hollow hull.
- A part that fills 0.97 or more of its convex hull gets one hull.
- The library files carry CC BY 4.0. The design said CC BY 2.0.

The lone-part tables on Machine A (x86), R1 seeds, 12 trials a part (`e/ldraw/table_*.log`):

| part | hull cap 8, before the 3660 fix, c7a1869 | hull cap 8, 19411af | hull cap 4, 989e769 |
| --- | --- | --- | --- |
| 3700 | 12/12 | 12/12 | 12/12 |
| 3701 | 12/12 | 12/12 | 12/12 |
| 3894 | 12/12 | 12/12 | 12/12 |
| 4490 | 12/12 | 12/12 | 12/12 |
| 3659 | 11/12 | 11/12 | 12/12 |
| 87087 | 11/12 | 11/12 | 11/12 |
| 11211 | 12/12 | 12/12 | 12/12 |
| 3660 | 4/12 | 11/12 | 11/12 |
| family | 86/96 = 0.896 | 93/96 = 0.969 | 94/96 = 0.979 |

Before the fix, a thin CoACD wedge of 3660 crossed the box floor, and a side face had no closing
axis. Four trials of 3660 ended under the box floor.

The heap read ran on Machine B (x86) through `bin/teacher_heap.sh`. Its flags are `--tray --desk_boxes 4
--tray_lean 30 --families ldraw --heap_family ldraw`, run 20288000 + i, layout 20288500 + i, i = 0
to 3. Design section 6.5 adds a speed rule. The heap read records the steps per second of its lanes.
If they fall under half of the MS7d rate, the hull cap drops to 4 and the read runs again.

| TAG | commit | hull cap | LDraw row | rest of the heap | all | cycles per correct stow | steps per second against the same heaps without LDraw: pooled (lanes) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e_ldraw_heap | 19411af | 8 | 93/100 | 96/100 | 189/200 | 0.995 | 0.64 (0.78, 0.70, 0.48, 0.68) |
| e_ldraw_heap_cap4 | 989e769 | 4 | 87/100 = 0.87 | 96/100 = 0.96 | 183/200 | not recorded | 0.82 (0.88, 0.94, 0.64, 0.86) |

The baseline without LDraw is e_ldraw_heap_nold. Lane 2 alone reads 0.47 at cap 8.

Decision D8 reads the speed rule per lane: "the lanes fall under half". The pooled rate of 0.64 is
above half, but lane 2 falls under it. So the cap drops to 4, and the read runs again. At cap 4 the
LDraw row meets its bar of 0.85 and the rest meets 0.90. LDraw joins at cap 4 (feat/ms7e-ldraw
a11c755, `MAX_HULLS` 4). The JSON parts there are byte for byte those of 989e769. The 17:10 entry
records all three families as adopted, ahead of the wave 1 deadline.

### 3.5 Flat

The flat parts are plate 1x2, 1x4, 1x6, 2x2, 2x4 and 2x6, and tile 1x2, 1x4, 2x2 and 2x4. No 4-wide
flat part enters (design section 6.6). They left the catalog in MS6 for two reasons. LIFT squeezed a
3.2 mm part out of the jaws, and a 31.8 mm grasp slid off one side. In MS6, tiles lifted at 0.33
against 0.73 for plates.

The flat track's probe at 16:11 used scratch seeds, so it is not a table (`tables/flat_notes.txt`).
The baseline stows 80 of 120, 0.667, and lifts and carries 120 of 120. All 40 failures share one
mechanism:

- The part hangs at the 20 degree tool lean of the carry. It lands corner first from about 0.095 m.
- It then tunnels through the 5 mm box floor. The part falls at about 1.3 m/s, and one 240 Hz step
  moves it about 5.5 mm.
- Once its centre passes the centre of the floor box, the contact normal flips. The floor then
  pushes the part down onto the desk. A 3.2 mm part needs 4.1 mm of penetration for this. A brick
  needs 7.3 mm and never gets there.

The probe found three more things:

- A floor collider 10 mm deeper, with its top unchanged, stows 72 of 72 on the six plate footprints.
- A tile and a plate of one footprint are the same collision box, because the studs are visual. So
  their MS6 split was sampling.
- The continuous gripper shows no squeeze-out and no one-sided slide. At the LIFT gate, a 3.2 mm
  grasp holds at a gap of about 1.1 mm with about 0.5 N per jaw.

Lever 1 is the deeper floor collider behind its own switch, `--deep_box_floor` (8aff43a, later
5a27216). The flat lone-part tables on x86 and the R1 seeds (`tables/baseline_flat.json`,
`tables/lever1_flat.json`):

| part | baseline, 6f3fffb | `--deep_box_floor`, 58c4fc5 |
| --- | --- | --- |
| plate_1x2 | 6/12 | 12/12 |
| plate_1x4 | 10/12 | 12/12 |
| plate_1x6 | 12/12 | 12/12 |
| plate_2x2 | 7/12 | 12/12 |
| plate_2x4 | 5/12 | 12/12 |
| plate_2x6 | 10/12 | 12/12 |
| tile_1x2 | 9/12 | 12/12 |
| tile_1x4 | 10/12 | 12/12 |
| tile_2x2 | 7/12 | 12/12 |
| tile_2x4 | 10/12 | 12/12 |
| family | 86/120 = 0.717 | 120/120 = 1.000 |

Both tables lift 120 of 120. All 34 baseline failures end `own_box:under_floor`. The grip medians of
the two tables are the same, so the lever changes only where the released part comes to rest. The
table reaches 0.90 with one lever. These tables ran with the four base classes. Tables at the stack
top use eight, so they do not repeat these rows bit for bit (17:36).

The heap reads ran on Machine A at bdb55e9 (17:05), split all, i = 0 to 3:

| TAG | floor | flat row (plate, tile) | rest of the heap | all, cycles |
| --- | --- | --- | --- | --- |
| e_flat_heap_nofloor | 5 mm | 88/100 (51/58, 37/42) | 93/100 | 181/200, 182 cycles |
| e_flat_heap_deep | 5 mm plus a 10 mm deeper collider | 87/100 = 0.870 (48/58, 39/42) | 92/100 = 0.920 | 179/200, 183 cycles |

With the deep floor, the flat row meets its bar of 0.85 and the rest meets 0.90. Flat is adopted.

The heap read that the serving loop drives showed no fall-through, even without the deep floor. The
rig-driven recorder did show it (`reads/e_flat_recorder`). The recorder ran `mj_generate_batch.py`
with no cameras on the family seeds i = 0 and 1. Without the deep floor, plate_2x6 ended under a box
floor after 3 wasted cycles. With the deep floor, no part did. So flat joins the waves with
`--deep_box_floor`.

The deep floor fixes a simulation artifact and has no real counterpart. The real fingertip geometry
is not measured. So a flat pass here promises no real pick (design section 6.6, and section 8 of
this report).

## 4. The teacher on the wave configuration

### 4.1 The teacher smokes

Two sets of smokes measured the teacher on the drawn boxes before the reference read. The lean track
read these at 17:58, on six train heaps with the reference seeds, base families only:

| scene | commit | host | correct box | cycles per correct stow | by box count | unstowed: tray, outside, wrong box |
| --- | --- | --- | --- | --- | --- | --- |
| MS7d, `--desk_boxes 4` | 0dca08a | Machine A | 285/300 = 0.950 | 1.032 | | 5, 10, 1 |
| B1, `--box_draw 6` | 0dca08a | Machine A | 262/300 = 0.873 | 1.103 | 6: 135/150, 5: 88/100, 4: 39/50 | 16, 22, 3 |
| B2, `--box_draw 8 --release_lean`, before the hand-back fix | dc643b0 | | 237/300 = 0.790 | 1.101 | | |
| B2, after the hand-back fix | 0dca08a | Machine B | 259/300 = 0.863 | 1.116 | 8: 122/150, 6: 92/100, 5: 45/50 | 13, 16, 6 |

The unstowed counts cover the heaps without a time-out. So the drawn boxes cost the teacher on B1 as
well as on B2. The hand-back fix is a defect fix (section 7.3). The box track's smoke of section 2.8
is the other set.

### 4.2 The reference read misses

Rule R6 (design section 7.1) puts a teacher reference read before each wave. It reads 3 train-split
heaps on the wave's configuration, and it must stow at least 0.90 in the correct box. If it does
not, the most recently adopted rung leaves the configuration, and the reference reads again.

The ledger pre-registered the wave 1 reference read at 17:42, TAG e_w1_ref. It read at 6ffebd3 on
Machine A with the wave flags: `--tray --desk_boxes 8 --tray_lean 30 --box_draw 8 --release_lean
--families round,ldraw,flat --deep_box_floor`. Teacher reads leave `--perceived` out, because the
oracle driver refuses it.

The read misses at 17:50. It stows 127 of 150, 0.847, in the correct box, in 141 cycles:

| item | value |
| --- | --- |
| per heap | 42, 41, 44 of 50, all three at 8 boxes |
| by family | brick 69/75, slope 15/19, round 9/11, LDraw 13/19, plate 14/14, tile 7/12 |
| wrong box, per heap | 2, 3, 1 |

The R11 read e_boxload_fit gives a count split. At 4 to 7 boxes it reads 94/100, 45/50, 48/50 and
48/50. At 8 boxes it reads 40/50. So count 8 reads 0.80 to 0.85 in both reads, and counts 4 to 7
read 0.90 to 0.96.

### 4.3 The census: deviation D10

Rule R6 says that LDraw, the most recently adopted rung, leaves. Deviation D10 (17:50) runs a census
first. The losses spread over every family, and the evidence points at the box count. The census
reads the e_w1_ref heaps and diagnostic heaps on x86. The diagnostic seeds are run 20289600 + i and
layout 20289650 + i, which no rule uses. The census names the rung or parameter that leaves. The
reference read then runs again in a fresh TAG on its R1 seeds, with the same bar.

The census (18:23, `tables/boxload_diag/`) reads 6ffebd3 on six train heaps. Replays through its
wrapper repeat e_w1_ref step for step.

| arm | TAG | change against the wave flags | correct box of 300 | cycles per correct stow |
| --- | --- | --- | --- | --- |
| a | e_diag_a | none | 269, 0.897 | 1.059 |
| b | e_diag_b | `--box_draw 7` | 269 | 1.063 |
| c | e_diag_c | LDraw out | 267 | 1.060 |
| d | e_diag_d | the count forced to 8 | 256, 0.853 | 1.082 |
| e | e_diag_e | rake cap 4 and crowd | 269, null as in MS7a | 1.067 |
| f | e_diag_f | the B2 band with `--box_draw 6` | 266 | 1.030 |
| g | e_diag_g | B1 | 274, 0.913 | 1.040 |
| h | e_diag_h | the MS7d scene with the wave families | 271, 0.903 | |

The cycles per correct stow of arms b to g come from the census tables
(`tables/boxload_diag/fam_table.txt` and `sum_fg.txt`). Arm a reads by count 5: 92/100, 7: 45/50 and
8: 132/150. Arm c, with LDraw out, reads 2 fewer stows than arm a.

A census of 13 heaps at 8 boxes counts 650 parts and 555 correct stows. It places the lost parts:

- 62 in the tray, 52 of them never a target.
- 22 in another box. The census traced 21 of the 22 as passengers. A passenger is a part other than
  the target that rides out of the tray with it.
- 7 on the desk or on a rim.
- 4 still moving at the end.

The census names three mechanisms:

1. The carry near the arm base, on B2 only. The EXTEND carry is a straight line from the pick point
   to the box centre. When it passes within 0.08 m of the arm base axis, the part drops: 20 of 20
   cycles. From 0.08 to 0.09 m, 5 of 12 drop, and at 0.09 m or more, 11 of 483. Boxes at x -0.07 m
   served from the tray's far side do this. Counts 7 and 8 always put two boxes there. The envelope
   check never saw it, because its lone part sits at the tray centre.
2. Parts stuck at the tray walls, on every rung: 15 to 21 of 300. The share by family is tile 0.128,
   plate 0.106, slope 0.085, LDraw 0.063, brick 0.045 and round 0.017. 47 of 59 rakes moved the part
   less than 3 mm. A rake is the teacher's scripted push that moves a part off a wall.
3. Passengers into another box.

### 4.4 The teacher levers: rule R12

Rule R12 (18:23) set three teacher levers before any number, each behind its own switch. Parallel
worktrees built them from 6ffebd3:

- T1: a via-point or arc EXTEND carry that keeps the path at least 0.10 m from the base axis, under
  `--release_lean` only. It targets mechanism 1.
- T2: a passenger check at the lift. It targets mechanism 3.
- T3: a rake that frees thin parts at the tray walls. It targets mechanism 2.

The tuning seeds are the D10 seeds, on x86. A lever is adopted on two conditions. It reads more
correct stows than without it on arm a (the wave flags) and on arm d (count 8). Its target count
falls. B2 stays on two conditions. Arm a with the adopted levers reads at least 0.90. It also reads
no more than 6 parts under arm g (B1) with the same levers. Otherwise MS7e ships at B1.

| lever | switch, branch, commit | arm a, correct of 300 at cycles per correct stow | arm d | target count, lever against none | verdict |
| --- | --- | --- | --- | --- | --- |
| none | 6ffebd3 | 269 at 1.059 (e_diag_a) | 256 at 1.082 (e_diag_d) | | the base |
| T2 | `--passenger_check`, feat/ms7e-passenger 46441f4 | 276, 0.920, at 1.087 (e_t2_a) | 270, 0.900, at 1.078 (e_t2_d) | wrong-box parts: arm a 0 against 9, arm d 0 against 8 | adopted, 18:44 |
| T1 | `--carry_via`, feat/ms7e-carry-via eae4007 | 272, 0.907, at 1.029 (e_t1_a) | 265, 0.883, at 1.008 (e_t1_d) | `lifted_lost` cycles: arm a 4 against 14, arm d 2 against 21 | adopted, 18:48 |
| T3 | `--thin_rake`, feat/ms7e-thin-rake fd3a5fa | aarch64: 262 on against 264 off. x86: 270 on against 269 off. | aarch64: 268 on against 269 off | wall-stuck parts: 21 to 22 on arm a, 16 to 18 on arm d | not adopted, 19:10 |

A `lifted_lost` cycle lifts its target and then loses it.

T2 (1388 tests). At the LIFT gate, a part other than the target can rise more than 0.02 m within
0.06 m of the pick point. Then the lift counts as not lifted, and the existing REOPEN ladder tries
again. The two thresholds are `INITIAL:`. Arm a rejected 8 lifts and arm d 14. The desk and rim
losses do not rise: 3 against 6 and 1 against 1. Heaps with no reject repeat the census step for
step. The recorder's episode meta does not count the rejects, and the read payload does.

T1 (0fb2b80, da5a5c9, eae4007, 1388 tests). If the straight line passes within `CARRY_MIN_RADIUS_M`
= 0.10 m of the base axis, the rig walks EXTEND along an arc about that axis. The radius is
`INITIAL:`. The client refuses
`--carry_via` without `--release_lean`. On arm a, the `lifted_lost` cycles to boxes at x under -0.05
m fall from 11 to 0. The envelope check gained `--pick X,Y` for far-side picks. It ran with
brick_2x4, seed 20289500 and a 0.07 m box:

| pick point | back +y points landed without T1 | with T1 |
| --- | --- | --- |
| (0.15, -0.08) m | 0/16 | 16/16 |
| (0.20, -0.09) m | 6/16 | 16/16 |

The T1 branch also makes the supervisor cache compare `MS7E_OFF | payload["ms7e"]`. So a payload
block that predates a switch reads that switch as off.

T3 (19:10). A rake of a part no taller than 4 mm sweeps its tips 0.8 mm over the desk. The switch
also reaches the loop's `--scripted_rake`. On Machine C, against its own off baseline at
6ffebd3, both arms read fewer correct stows. The trace found three causes:

- The tips ride 2.4 to 2.7 mm over the desk at `PUSH_Z`, and flat parts moved only 0.2 to 5.4 mm.
- The teacher declines many thin-part rakes before they sweep, because teacher reads lack
  `--rake_turn`.
- The grasp model's wall test carries the 8 mm `JAW_HOLD_BAND` of the binary gripper under the
  continuous gripper. No read tested it then, and it became serving lever S3 (section 6.4).

### 4.5 The combined check and the second reference read

The adopted levers stack as carry-via (0fb2b80, da5a5c9, eae4007) under passenger 3f5629b, on
box-load 6ffebd3 (1394 tests). At 19:14 the ledger pre-registered the combined check of rule R12 at
3f5629b, on the R12 seeds, train split, x86:

| arm | TAG | scene | host | correct box of 300 | cycles per correct stow | wrong box | by box count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a2 | e_r12_a2 | B2 with the wave flags, `--carry_via --passenger_check` | Machine A | 277, 0.923 | 1.029 | 0 | 5: 93/100, 7: 47/50, 8: 137/150 |
| g2 | e_r12_g2 | B1 (`--box_draw 6`, no lean, no carry_via), `--passenger_check` | Machine B | 281, 0.937 | 1.036 | 0 | 4: 90/100, 5: 48/50, 6: 143/150 |

a2 reads at least 270, and at least g2 - 6 = 275. So B2 stays (19:24). The wave configuration is B2
with `--carry_via --passenger_check`.

The ledger pre-registered the second reference read at 19:24: TAG e_w1_ref2, 3f5629b, Machine A,
`bin/teacher_heap.sh`, the R1 reference seeds. Its flags are `--tray --desk_boxes 8 --tray_lean 30
--box_draw 8 --release_lean --carry_via --passenger_check --families round,ldraw,flat
--deep_box_floor`. The bar is 0.90, 135 of 150. If it misses, LDraw leaves, as rule R6 orders.

| item | e_w1_ref | e_w1_ref2 |
| --- | --- | --- |
| commit | 6ffebd3 | 3f5629b |
| correct box | 127/150 = 0.847 | 137/150 = 0.913 |
| cycles | 141 | 142 |
| per heap | 42, 41, 44 | 46, 46, 45 |
| box count | 8 in all three heaps | 8 in all three heaps |
| brick | 69/75 | 67/75 |
| slope | 15/19 | 16/19 |
| round | 9/11 | 11/11 |
| LDraw | 13/19 | 19/19 |
| plate | 14/14 | 14/14 |
| tile | 7/12 | 10/12 |
| wrong box | 2, 3, 1 | |
| passenger rejects | | 3, 5, 1 |
| outcome | misses the bar of 0.90 | passes |

The second reference read passes at 19:28. A wave's yield is the teacher's stow rate over the wave's
heaps. Each wave's yield must lie within 0.03 of 0.913.

## 5. The corpus, the retrains and the reads

### 5.1 Wave 1

Wave 1 started at 19:28 at 3f5629b in the w1 worktrees. `bin/wave_host.sh` calls
`bin/run_corpus_e.sh` with the recorder flags of `e/w1/flags.txt`: `--tray --desk_boxes 8
--tray_lean 30 --perceived --box_draw 8 --release_lean --carry_via --passenger_check --families
round,ldraw,flat --deep_box_floor`. Each process records 3 train heaps of 50 parts. The episode
length is 1500 s (deviation D9). The seeds are run 20284000 + p and layout 20285000 + p.

| host | processes | heaps | stowed | cycles per stow | cycle episodes after conversion | size |
| --- | --- | --- | --- | --- | --- | --- |
| Machine C | p 0 to 11, 12 | 36 | 1671/1800 = 0.928 | 1.068 | 1654 | 4.2 GB |
| Machine B | p 12 to 25, 14 | 42 | 1934/2100 | 1.07 | 1903 | 4.9 GB |
| Machine A | p 26 to 39, 14 | 42 | 1934/2100 = 0.921 | 1.072 | 1916 | 4.9 GB |
| wave 1 | 40 | 120 | 5539/6000 = 0.923 | | 5473 | |

Wave 1 ended at 21:06. Its yield of 0.923 lies within 0.03 of the reference read, 0.913, so the
yield rule holds. On Machine A, the attempts by family were brick 872/947, LDraw 329/362, plate
160/193, round 143/158, slope 252/255 and tile 138/159. The video decode check passed with CHECK_OK
30 videos on Machine B and 30 on Machine A.

### 5.2 Wave 2

The ledger pre-registered wave 2 at 20:08. Design section 7.1 says that wave 2 records in every
case. All three families joined before it, so its configuration is that of wave 1. Its reference
read is e_w1_ref2. It records 120 train heaps as 30 processes of 4 heaps, run 20286000 + p and
layout 20287000 + p. Machine B keeps its CPUs for retrain 1.

| host | processes | stowed | cycles per stow | cycle episodes after conversion |
| --- | --- | --- | --- | --- |
| Machine A | p 12 to 29, 18 | 3323/3600 = 0.923 | 1.067 | 3306, 8.4 GB |
| Machine C | p 0 to 11, 12 | 2179/2400 = 0.908 | 1.075 | 2152 |
| wave 2 | 30 | 5502/6000 = 0.917 | | 5458 |

Wave 2 started on Machine A at 20:34 and ended at 23:57. Its yield of 0.917 lies within 0.03 of
0.913 (rule R6).

The merge for retrain 2 (00:54, Machine B) joins e/w1/dataset (5473), e/w2/dataset_a (3306) and
e/w2/dataset_c (2152). MERGE_OK gives 10931 episodes and 4,879,196 frames, and CHECK_OK 188
videos (e/w2/dataset_all).

### 5.3 Retrain 1 and its screens

Rule R7 (design section 7.2): retrain 1 warm-starts from train_P 200k on wave 1 alone. A warm start
begins training from the weights of an existing checkpoint. The recipe is the MS7a arm A recipe,
`bin/train_warm.sh`, 200k steps, with a checkpoint every 50k.

The ledger pre-registered retrain 1 at 21:25 on Machine B. `bin/merge_train.sh` merged dataset_b
(1903), dataset_a (1916) and dataset_c (1654) into e/w1/dataset. MERGE_OK gave 5473
episodes and 2,432,249 frames, and CHECK_OK 84 videos. LeRobot trained at 38.8 steps/s, about 85
minutes for 200k (21:30). Retrain 1 ended TRAIN_OK at 23:03 (e/train_E1).

Second copies (rule R6). Machine A and aarch64 datasets stay on their hosts. dataset_b copies
to Machine C. Each checkpoint shipped to Machine A for its screen, and that is its second
copy.

Each screen reads seed 20281100, draws 0 and 1, 500 parts, with the full protocol
(`e/bin/read_e.sh`). The most correct stows win, and a tie goes to fewer cycles.

| checkpoint | TAG | commit | correct stows of 500 | cycles per correct stow | by box count 4, 5, 6, 7, 8 | by family: brick, slope, round, LDraw, plate, tile |
| --- | --- | --- | --- | --- | --- | --- |
| 50k | e_E1_050000_scr | 3f5629b, payload sum (D11) | 423, 0.846 | 1.343 | 43/50, 123/150, 84/100, 87/100, 86/100 | 166/180, 79/95, 23/25, 76/80, 65/95, 14/25 |
| 100k | e_E1_100000_scr | 3f5629b, payload sum (D11) | 411, 0.822 | 1.333 | 45/50, 128/150, 82/100, 75/100, 81/100 | 165/180, 72/95, 25/25, 68/80, 69/95, 12/25 |
| 150k | e_E1_150000_scr | 3f5629b, payload sum (D11) | 418, 0.836 | 1.215 | 42/50, 126/150, 84/100, 83/100, 83/100 | 165/180, 76/95, 25/25, 71/80, 65/95, 16/25 |
| 200k | e_E1_200000_scr | f8faeee, b_r1:2, a clean GATE_OK | 428, 0.856 | 1.276 | | |

The supervisors of the 50k, 100k and 150k screens crashed on a defect after they wrote the payloads
(section 7.3). Deviation D11 sums those screens from their own EVAL_OK payloads with
`e/bin/payload_sum.py`. No payload moves between reads. Rule R9 asks for a read again in a fresh
TAG, so D11 departs from it. The three screens ran in the w1 worktrees at 3f5629b. The 200k screen,
candidate E1's checked read, the control read and E2's screens run at f8faeee. The two commits
differ only in the supervisor's per-box sum, so the simulation code of all four screens is the same.

By rule R7, train_E1 200k has the most correct stows, so it is candidate E1 (00:14).

### 5.4 Candidate E1's checked read

The ledger pre-registered the checked read at 00:14: TAG e_E1_chk, train_E1 200k, f8faeee, the rule
R3 protocol with no R13 lever. It reads seed 20281700, draws 0 to 11, 3000 held-out parts, on
`a_r1:6` and `b_r1:6` with `PORT_BASE` 9600. The same entry repeats the retrain 2 trigger
of rule R7. The trigger fires on a missed bar, on fewer than 2610 correct stows, or on more than
1.25 cycles per correct stow.

| item | value |
| --- | --- |
| TAG | e_E1_chk |
| checkpoint | train_E1 200k |
| commit | f8faeee |
| correct stows of 3000 | 2614, 0.871 |
| Wilson 95 % | 0.859 to 0.883 |
| cycles per correct stow | 1.275 |
| stows over the bar of 2550 | 64 |
| cycles per correct stow under the bar of 1.3 | 0.025 |
| wrong-box parts | 167 |
| outcome | both bars met, a gate candidate with every family |

A Wilson 95 % interval is a 95 % confidence interval for a rate.

By box count:

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 474/550 | 615/700 | 658/750 | 264/300 | 603/700 |
| fraction | 0.862 | 0.879 | 0.877 | 0.880 | 0.861 |

By family:

| family | brick | slope | round | LDraw | plate | tile |
| --- | --- | --- | --- | --- | --- | --- |
| correct stows | 1126/1185 | 446/540 | 171/185 | 445/485 | 281/400 | 145/205 |
| fraction | 0.950 | 0.826 | 0.924 | 0.918 | 0.703 | 0.707 |

This report computes the fractions from the counts. The box count moves the stow rate by less than
0.02. The family moves it by 0.25. Brick stows at 0.950 and the flat families at about 0.70.

The 12 draws stow 208 to 226 parts each. The payloads give the failed cycles by class. A failed
cycle is a cycle that does not end with its target stowed. Section 5.8 sets these classes beside
those of the later reads.

### 5.5 The trigger and retrain 2

The retrain 2 trigger fires at 01:01. 2614 is at least 2610, but 1.275 is above 1.25 cycles per
correct stow. Retrain 2 starts at 01:01 on Machine B (job e_train_E2). It warm-starts from train_E1 200k
on e/w2/dataset_all, waves 1 and 2, 10931 episodes. It trains 200k steps with `bin/train_warm.sh`,
with a checkpoint every 50k (e/train_E2). Retrain 2 ended TRAIN_OK at 02:43.

The ledger pre-registered E2's screens at 01:43. As rule R7 asks, they read draws 0 and 1 of seed
20281100, and the most correct stows win. They read f8faeee with the rule R3 protocol and no R13
lever, as E1's screens did.

| checkpoint | TAG | lane | correct stows of 500 | Wilson 95 % | cycles per correct stow | by box count 4, 5, 6, 7, 8 | by family: brick, slope, round, LDraw, plate, tile | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50k | e_E2_050000_scr | `a_r1:2` | 418, 0.836 | 0.801 to 0.866 | 1.373 | 45/50, 126/150, 82/100, 83/100, 82/100 | 165/180, 82/95, 25/25, 74/80, 57/95, 15/25 | 41 |
| 100k | e_E2_100000_scr | `a_r1:2` | 427, 0.854 | 0.820 to 0.882 | 1.279 | 42/50, 129/150, 85/100, 87/100, 84/100 | 168/180, 80/95, 24/25, 71/80, 69/95, 15/25 | 42 |
| 150k | e_E2_150000_scr | `a_r1:2` | 433, 0.866 | 0.833 to 0.893 | 1.210 | 45/50, 131/150, 76/100, 88/100, 93/100 | 172/180, 74/95, 24/25, 69/80, 76/95, 18/25 | 41 |
| 200k | e_E2_200000_scr | `b_r1:2` | 413, 0.826 | 0.790 to 0.857 | 1.334 | 41/50, 127/150, 78/100, 82/100, 85/100 | 167/180, 82/95, 24/25, 73/80, 55/95, 12/25 | 36 |

By rule R7, train_E2 150k has the most correct stows, so it is candidate E2 (03:18). It also has
the fewest cycles per correct stow. The 200k screen reads fewer stows than the 150k screen. Plates
move the most: 76 of 95 at 150k and 55 of 95 at 200k. A screen has no bar. Its supervisor still
prints the gate token with the gate's arithmetic on 500 parts.

### 5.6 Candidate E2's checked read

Rule R14 (03:14) set the protocol of this read before it started. It reads 3e9ee7b with the full
protocol plus `--thin_snap --wall_band_cont`, the R13 levers adopted at 02:55 (section 6.4). It
reads seed 20281700, draws 0 to 11, 3000 held-out parts, on `a_r2:6` and `b_r2:6` with
`PORT_BASE` 9900. The bars are 2550 correct stows and 1.3 cycles per correct stow.

| item | value |
| --- | --- |
| TAG | e_E2_chk |
| checkpoint | train_E2 150k |
| commit and switches | 3e9ee7b, the full protocol plus S2 and S3 |
| correct stows of 3000 | 2673, 0.891 |
| Wilson 95 % | 0.879 to 0.902 |
| cycles per correct stow | 1.224 |
| stows over the bar of 2550 | 123 |
| cycles per correct stow under the bar of 1.3 | 0.076 |
| wrong-box parts | 188 |
| outcome | GATE_OK, PASS at 04:00, a gate candidate with every family |

By box count:

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 499/550 | 621/700 | 682/750 | 270/300 | 601/700 |
| fraction | 0.907 | 0.887 | 0.909 | 0.900 | 0.859 |

By family:

| family | brick | slope | round | LDraw | plate | tile |
| --- | --- | --- | --- | --- | --- | --- |
| correct stows | 1135/1185 | 477/540 | 178/185 | 450/485 | 293/400 | 140/205 |
| fraction | 0.958 | 0.883 | 0.962 | 0.928 | 0.733 | 0.683 |

Against candidate E1's checked read, candidate E2 gains 59 stows and saves 0.051 cycles per correct
stow. Slope gains the most, from 0.826 to 0.883. Tile loses 5 parts. Count 8 is the weakest box
count at 0.859. The wrong-box parts rise from 167 to 188.

### 5.7 The control read

The ledger pre-registered the control read at 01:02 (rule R8, design section 8.2). A is train_P 200k
with no retrain, on the base families with no `--families`. It uses the adopted box rung and levers
of the protocol: `--box_draw 8 --release_lean --carry_via --passenger_check --deep_box_floor`. It
reads f8faeee on seed 20281700, draws 0 to 3, 1000 held-out parts. Its TAG is e_A_control, on
`a_r1:4` with `PORT_BASE` 9800. It has no pass rule.

A sees a camera pose outside its training draws, because the B2 camera sits 0.139 m back.

| item | value |
| --- | --- |
| TAG | e_A_control, read at 01:57 |
| correct stows of 1000 | 551, 0.551 |
| Wilson 95 % | 0.520 to 0.582 |
| cycles per correct stow | 2.076 |
| by box count 4, 5, 6, 7, 8 | 264/400, 121/200, 90/250, 47/100, 29/50 |
| by family | brick 435/755, slope 116/245 |
| wrong-box parts | 80 |
| outcome | GATE_FAIL on the gate's arithmetic, no bar, reported |

The MS7d official read, d_A_official, stowed 0.920 at 1.123 with this policy on the MS7d task. The
widened boxes cost it 0.37 of the parts. They add the B2 camera 0.139 m back and the drawn boxes
behind the arm base. The retrains buy the loss back. Candidate E1 stows 0.871 with every family.

The payloads show where the control loses. It has 327 `lifted_lost` cycles on 1000 parts, the
target lifted and then lost. The official read has 540 on 6000 parts, 90 per 1000.

### 5.8 The official read

Rule R2 gives the official read to the passing checked read with the most families. Candidates E1
and E2 both carry every family, the base ones and all three new ones. So the tie-break decides:
fewer cycles per correct stow. Candidate E2 reads 1.224 against 1.275, so rule R2 picks it. The
ledger pre-registered the read at 04:00. It uses seed 20282100, draws 0 to 23, 6000 held-out parts,
one shot. Rule R14 gives it the protocol of candidate E2's checked read.

| item | value |
| --- | --- |
| candidate and checkpoint | candidate E2, train_E2 150k, picked by the tie-break of rule R2 |
| TAG | e_E2_official |
| commit and switches | 3e9ee7b, the full protocol plus `--thin_snap --wall_band_cont` |
| lanes | `a_r2:12` (draws 0 to 11) and `b_r2:12` (draws 12 to 23), `PORT_BASE` 10000 |
| correct stows of 6000 | 5299, 0.883 |
| Wilson 95 % | 0.875 to 0.891 |
| cycles per correct stow | 1.271, 6737 cycles |
| stows over the bar of 5100 | 199 |
| cycles per correct stow under the bar of 1.3 | 0.029 |
| wrong-box parts | 414 |
| outcome | GATE_OK, PASS at 05:00 |

By box count:

| boxes | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| correct stows | 1159/1300 | 893/1000 | 1095/1250 | 1059/1200 | 1093/1250 |
| fraction | 0.892 | 0.893 | 0.876 | 0.882 | 0.874 |

By family:

| family | brick | slope | round | LDraw | plate | tile |
| --- | --- | --- | --- | --- | --- | --- |
| correct stows | 2156/2290 | 817/920 | 471/495 | 824/885 | 706/925 | 325/485 |
| fraction | 0.941 | 0.888 | 0.952 | 0.931 | 0.763 | 0.670 |

The official read passes both bars. It stows 5299 of 6000 against 5100, at 1.271 cycles per correct
stow against 1.3. MS7e meets its gate with 4 to 8 drawn boxes (rung B2), the eight classes and every
family. The box count moves the stow rate by less than 0.02. The family moves it by 0.28.

The flat parts carry most of the excess cycles. The official read has 6737 cycles for 5299 correct
stows, so 1438 cycles are excess. Plates take 1303 cycles for 706 stows, 1.85 cycles per correct
stow. Tiles take 695 cycles for 325 stows, 2.14. Together the flat parts hold 967 of the 1438 excess
cycles, 0.67. Without them, the read uses 1.11 cycles per correct stow. The census script counts the
cycles per family. Slope reads 0.888, level with its 0.886 on MS7d.

The failed cycles by class, from the payloads:

| class | E1 checked | E2 checked | official |
| --- | --- | --- | --- |
| cycles | 3333 | 3271 | 6737 |
| `close_miss`, the jaws close on nothing | 383 | 287 | 691 |
| `lifted_lost`, the target is lifted and then lost | 236 | 259 | 540 |
| `held_no_lift`, the target is held but not lifted | 220 | 223 | 495 |
| `wrong_box` | 54 | 51 | 137 |
| `carrying`, the cycle ends while it carries | 50 | 52 | 114 |
| `no_close` and `no_open` | 10 | 5 | 27 |

S2 targets the `close_miss` cycles. Candidate E2's checked read adds S2 and retrain 2 together. Its
`close_miss` cycles are 9.6 % of the parts, against 12.8 % for candidate E1. The official read has
11.5 %, and the MS7d official read had 6.1 %.

The wrong-box parts are 6.9 % of the parts in the official read, against 4.7 % on MS7d. In the E1
census, most wrong-box parts were passengers that no cycle held (section 6.1). S1 cannot catch them
(section 6.3).

### 5.9 The unseen box ladder

Rule R8 (design section 8.3) reads the official winner. With no winner, it reads the best checked
read. Each row reports its Wilson interval and whether it holds the gate bars. The ladder does not
change the gate. Training draws the whole count range, so no row holds a count out. The per-count
rows of the checked read stand in for one.

The ledger set up the ladder at 04:21. `feat/ms7e-ladder` 1acb329 on 3e9ee7b adds `--seen_seeds
RUN0,LAYOUT0` to the supervisor (1424 tests). The switch is off by default. With it, draw i reads
run RUN0 + i and layout LAYOUT0 + i in one env, and the pair must be in `ledger.json`. Every row
reads candidate E2, train_E2 150k, with `--thin_snap --wall_band_cont` through `e/bin/ladder_e.sh`.

| row | TAG | box conditions | seed and size | lanes | correct stows | Wilson 95 % | cycles per correct stow | holds the bars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seen | e_ladder_seen | the boxes of wave 1, with held-out parts | 20284000 and 20285000, draws 0 to 19, 1000 parts | `c_lad:10` | 869, 0.869 | 0.847 to 0.889 | 1.306 | the stow bar holds, the cycles bar misses by 0.006 |
| unseen layouts | e_E2_chk | the training distribution on a fresh seed | 20281700, 3000 parts | `a_r2:6`, `b_r2:6` | 2673, 0.891 | 0.879 to 0.902 | 1.224 | both hold |
| unseen size | e_ladder_size | sides up to 0.12 m and an aspect up to 1.7, 4 to 6 boxes | 20283100, 1000 parts | `a_lad:2`, `b_lad:2` | 896, 0.896 | 0.876 to 0.913 | 1.230 | both hold |
| unseen rim | e_ladder_rim | an interior depth of 0.05 to 0.06 m, 4 to 8 boxes | 20283200, 1000 parts | `a_lad:2`, `b_lad:2` | 881, 0.881 | 0.859 to 0.900 | 1.310 | the stow bar holds, the cycles bar misses by 0.010 |

By box count and by family:

| row | by box count | brick | slope | round | LDraw | plate | tile | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seen | 4: 131/150, 5: 87/100, 6: 173/200, 7: 304/350, 8: 174/200 | 336/363 | 136/145 | 66/69 | 145/155 | 143/190 | 43/78 | 79 |
| unseen size | 4: 364/400, 5: 310/350, 6: 222/250 | 320/340 | 154/175 | 72/75 | 181/195 | 91/115 | 78/100 | 65 |
| unseen rim | 4: 127/150, 5: 221/250, 6: 93/100, 7: 229/250, 8: 211/250 | 381/400 | 114/130 | 72/75 | 132/145 | 118/160 | 64/90 | 67 |

Every row holds the stow bar of 0.85. The seen and rim rows go 0.006 and 0.010 over the cycles bar.
A 1000-part read cannot tell a gap that small from noise. The larger boxes of the size row read no
worse than the training boxes: 0.896 against 0.891. The flat parts again carry the excess cycles:
0.72 of them in the seen row and 0.69 in the rim row. The tiles of the seen row stow only 43 of 78.

The seen row uses one fact of the box draw. In an eval draw, the first env draws the exact boxes
that wave 1 recorded in its own first env. All 40 wave 1 processes match bit for bit. Later envs
depend on the heap before them. So the row reads the first env of wave 1's processes 0 to 19. Three
caveats apply:

- The parts are held out, so only the boxes are seen, not the heaps.
- The class map can differ from the recorded one, because the capacity fit of rule R11 weighs the
  held-out heap. The map differs in 4 of 14 checked episodes.
- The row read on Machine C, with the policy on its CPU. The box draw does not depend on
  the architecture, but the physics does. So the seen row does not pair step for step with the
  other rows, which read on x86.

The unseen size and rim rows lie outside the certification of the envelope check. That check used
0.07 m and 0.10 m interiors at 0.05 m depth. The frame margins of the two rows are 5.9 px and 6.2
px. The rim row has its own back limit of -0.137 m. The smokes on seed 7 (E1 200k, throwaway) read
45 of 50 seen, 223 of 250 size and 231 of 250 rim, all GATE_OK.

## 6. The policy census and the serving levers

### 6.1 The E1 loss census

The E1 loss census (23:32, `tables/e1_census/`) reads the screens of 50k to 150k against the MS7d
official read d_A_official.

Flat parts cost 64 to 68 % of the excess cycles, the cycles beyond one per stow. Plates use 1.74 to
2.03 cycles per stow and tiles 1.75 to 2.92. Without the flat parts, the heap reads 1.09 to 1.15
cycles per stow.

The lost parts, of 500 per screen:

| end place | 50k | 100k | 150k |
| --- | --- | --- | --- |
| another class's box | 40 | 46 | 44 |
| of them, never held by any cycle (passengers) | 25 | 29 | 28 |
| the tray wall band | 25 | 28 | 24 |
| the desk | 7 | 7 | 11 |

The serving loop has no passenger check. The rig's T2 acts only in the scripted rakes, where the rig
drives.

At 50k, the first close on a flat part sits above the part's top in 0.60 of the misses. The
`--pick_snap` floor puts the pick point at row z + 3 mm. On a 3.2 mm part, that is 1.4 mm over the
top.

The failed cycle classes as a share of the parts, against MS7d. A `close_miss` cycle closes the jaws
on nothing:

| class | 50k | 100k | 150k | MS7d official |
| --- | --- | --- | --- | --- |
| `close_miss` | 17 % | 17 % | 10 % | 6.1 % |
| `lifted_lost` | 10 % | 8 % | 9 % | 3.8 % |

Wrong-box parts are 8 to 9 % of the parts on the screens, against 4.7 % on MS7d. Brick stows at the
MS7d level. Slope reads 0.76 to 0.83 against 0.886.

### 6.2 Rule R13

Rule R13 (23:32) set three serving levers before any number. Each is a switch, built on the stack
top f8faeee and stacked on it:

| lever | switch | what it does | target |
| --- | --- | --- | --- |
| S1 | `--loop_passenger` | Under `--perceived`, the loop perceives at the lift. Another detection can rise more than 0.02 m within 0.06 m of the pick point. Then the jaws open at lift height, and the cycle goes on. | passengers into another box |
| S2 | `--thin_snap` | A served row with a half height of at most 2 mm snaps to desk contact, not to row z + 3 mm. | flat first closes above the part |
| S3 | `--wall_band_cont` | The grasp model's wall test uses the landing band of the continuous gripper, not the binary 8 mm `JAW_HOLD_BAND`. | parts at the tray walls |

The rows read candidate E1 with the full protocol. They use a tuning seed that no other rule uses:
20281500, draws 0 to 5, 1500 held-out parts. The rows are base, +S1, +S2, +S3 and +all adopted. To
be adopted, a lever's row must meet three conditions:

- It reads at least as many correct stows as the base row.
- It reads no more than 0.02 cycles per correct stow above the base row.
- It gains at least 8 stows or saves at least 0.02 cycles.

The `all` row must not read worse than the best single row by 20 stows or 0.03 cycles.

Adopted levers join candidate E2's checked read and the official read. This is a deviation from the
frozen protocol of rule R3, as the MS7d levers were.

### 6.3 S1 cannot work with the overhead camera

The S1 track built S1 as specified at 00:30: feat/ms7e-loop-passenger 764827d, an overhead look at
the policy's lift through the Perceiver (1407 tests). The smoke ran on seed 7, draw 0, throwaway,
with E1 50k on `a_s1:1`:

| item | value |
| --- | --- |
| result | GATE_OK 222/250 at 1.203 |
| wrong-box parts | 8 |
| looks at the lift | 225 |
| rejects | 1, a false positive |
| true riders | 11 |
| riders caught | 0 |

A rider is a passenger at the lift. The overhead camera cannot see a part within 0.04 m of the tool
(`TOOL_OCCLUSION_RADIUS_M`), and riders sit there. So the look as specified cannot work.

The S1 track then measured a proprioceptive variant before any build (01:17, `tables/s1/`). The test
compares the measured jaw gap at the lift with the served row's width. The traces ran E1 200k on the
throwaway seeds 7 and 8, draw 0, with S1 not refusing:

| item | value |
| --- | --- |
| lifts | 453 |
| lifts with a true rider | 27 |
| riders | 28: 20 touch one finger, 4 pinched, 4 on top |
| finger riders beside the carried part, along the jaw faces | 18 of 24. They do not widen the gap. |
| clean lifts refused at the best threshold, gap minus served width over 1 mm | 7 of 426 |
| riders caught at that threshold | 6 of 27, 0 of 4 pinched |
| largest rider distance to the pick point | 0.0356 m, inside the 0.04 m tool occlusion |

To see the riders, the loop needs a wrist-camera look after the lift or a load signal. Both lie
outside MS7e. S1 leaves rule R13 unbuilt. feat/ms7e-loop-passenger 764827d stays pushed, with its
switch off, as the record. Rule R13 keeps two lever rows, S2 and S3, and the combined row.

### 6.4 S2 and S3

S3 (00:39): feat/ms7e-wall-band 4ca0e49 on f8faeee, 1410 tests.

- 114d1bc sets the wall test at the landing gap of the continuous jaws. The band is w/2 + 4 mm +
  `CONT_HOLD_BAND` 1 mm, `INITIAL:`, instead of w/2 + 4 + 8 mm.
- 4ca0e49 adds `--wall_band_cont` to the clients, the recorder and the gate.
- A probe found 19,234 newly allowed grasps. None puts a finger in a wall at the grip band, and no
  flat grasp does even at the desk.
- The smoke (seed 7, draw 0, throwaway, E1 50k) ends GATE_OK 226/250 at 1.274, with 6 parts at the
  walls.

The teacher with S3 on the R12 seeds (x86, e_s3_teacher_a and e_s3_teacher_b) stows 289 of 300,
0.963, at 1.024. e_r12_a2 stowed 277 of 300 at 1.029. The wall-stuck parts fall from 16 to 4, and
the rakes from 41 to 10. This gain is information only, because the waves are recorded. The R13 rows
decide the serving switch.

S2 (00:54): feat/ms7e-thin-snap c67d471. A served row with a half height of at most 2 mm,
`INITIAL:`, snaps to desk contact at 2.1 mm. The smoke (seed 7, draw 0, throwaway, E1 50k) reads
218/250 at 1.284. S2 and S3 stack as f8faeee, then wall-band 4ca0e49, then thin-snap 3e9ee7b (1418
tests). The r2 worktrees on all three hosts sit at 3e9ee7b.

The three smokes read one throwaway draw each, and no base smoke ran on that draw. So they do not
compare as rows.

The ledger pre-registered the R13 rows at 00:54. They read 3e9ee7b with E1 = train_E1 200k, seed
20281500, draws 0 to 5, the full protocol plus the row's lever. In every row, draws 0 to 3 read on
x86, Machine A or Machine B, which share their physics. Draws 4 and 5 read on Machine C. So
the rows pair draw for draw.

| row | TAG | lever switches | correct stows of 1500 | Wilson 95 % | cycles per correct stow | against the base row | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | e_r13_base | none | 1290, 0.860 | 0.842 to 0.877 | 1.341 | | the base |
| S2 | e_r13_s2 | `--thin_snap` | 1303, 0.869 | 0.851 to 0.885 | 1.269 | +13 stows, -0.072 cycles | adopted, 02:51 |
| S3 | e_r13_s3 | `--wall_band_cont` | 1339, 0.893 | 0.876 to 0.907 | 1.291 | +49 stows, -0.050 cycles | adopted, 02:51 |
| all | e_r13_all | `--thin_snap --wall_band_cont` | 1320, 0.880 | 0.863 to 0.895 | 1.308 | +30 stows, -0.033 cycles | holds its bar, 02:55 |
| S1 | no row | `--loop_passenger` | | | | | unbuilt, see section 6.3 |

S2 and S3 each meet all three conditions. Neither loses stows, both save cycles, and both gain at
least 8 stows. The best single row is S3. So the `all` row must read at least 1319 correct stows and
at most 1.321 cycles per correct stow. It reads 1320 at 1.308, and rule R13 closes with S2 and S3
adopted. The `all` row reads 19 stows under S3 alone. That gap lies inside the noise of two
1500-part reads, and the rule stands. From 03:14, candidate E2's checked read and the official
read carry `--thin_snap --wall_band_cont` at 3e9ee7b (rule R14).

By box count and by family:

| row | by box count 4, 5, 6, 7, 8 | brick | slope | round | LDraw | plate | tile | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 381/450, 306/350, 260/300, 42/50, 301/350 | 540/565 | 143/180 | 127/135 | 214/235 | 181/245 | 85/140 | 74 |
| S2 | 387/450, 317/350, 252/300, 40/50, 307/350 | 538/565 | 162/180 | 128/135 | 203/235 | 177/245 | 95/140 | 84 |
| S3 | 402/450, 304/350, 272/300, 45/50, 316/350 | 540/565 | 159/180 | 130/135 | 220/235 | 184/245 | 106/140 | 81 |
| all | 392/450, 316/350, 256/300, 40/50, 316/350 | 535/565 | 157/180 | 128/135 | 217/235 | 179/245 | 104/140 | 88 |

The payloads name each lever's mechanism. The `close_miss` counts come from `row_report.py`, and the
other counts from the census script:

| count | base | S2 | S3 | all |
| --- | --- | --- | --- | --- |
| `close_miss` cycles, all parts | 227 | 164 | 223 | 216 |
| `close_miss` cycles on plates and tiles | 188 | 121 | 172 | 158 |
| cycles on plates and tiles | 601 | 523 | 572 | 570 |
| lost parts in the tray wall band | 79 | 60 | 41 | 37 |

S2 cuts the `close_miss` cycles on flat parts from 188 to 121, and the cycles on flat parts from 601
to 523. S3 cuts the lost parts in the tray wall band from 79 to 41. In the `all` row, the wall band
count falls to 37, but the flat `close_miss` cycles fall only to 158. The `all` row does not repeat
the S2 gain in full, and this report has no cause for that.

## 7. The code, the deviations and the defects

### 7.1 The stack

All branches start from main c728747 (deviation D1). The ledger records the tips in the third
column. The rebase with `--update-refs` at 23:03 moved every branch above boxes. The ledger names
only the new top, f8faeee. The fourth column gives each branch's tip in its pull request (PR).

The final stack is GitHub stack #148 (05:19). It holds 13 draft PRs, #135 (`feat/ms7e-base`) to
#147 (`feat/ms7e-ladder`). Each branch sits at the SHA that its reads recorded. CI passed on all 13
at the first run, with no fix commit. The report PR goes on top of #147, with `feat/ms7e-ladder` as
its base.

| order | PR | branch | tips in the ledger | tip in the PR | what it adds |
| --- | --- | --- | --- | --- | --- |
| 1 | #135 | `feat/ms7e-base` | 86263d6 | 86263d6 | c74ff38 the families, the classes and the rows. 86263d6 the five switches. |
| 2 | #136 | `feat/ms7e-boxes` | a388321, then a74045b | a74045b | e6d02b6 the per-episode draw. 419973b the envelope `--interior` through the env. f5e8b14 the deep floor, taken from the flat track. 4c90d78 the resize keeps the deep floor. a388321 the recorder's box fields. a74045b the gate's per-box sum. |
| 3 | #137 | `feat/ms7e-release-lean` | dc643b0, then 0dca08a | 1f9add2 | ca24a41 the rig lean. 77414e7 the envelope reads. 3ad2e1f the camera back under `--release_lean`. b4b9eb3 the clients and the gate. def4361 tests. dc643b0 the B2 draw, `B2_CEILING` 8. 0dca08a the takeover hand-back fix. |
| 4 | #138 | `feat/ms7e-round` | e55ca19, then 90ce982 | 6af5cd2 | P1: the disc studs, `round_physics`, the family ready |
| 5 | #139 | `feat/ms7e-ldraw` | 89ddeb2, then 9ee7d37 | b7220fc | P2: the generator, the JSON parts and their notice, the hull geoms, the closing axes, `MAX_HULLS` 4 |
| 6 | #140 | `feat/ms7e-flat` | eb1589a, then b5c60e4 | 7cfa885 | P3: the family ready. beecc81 the gate pieces of `--deep_box_floor`, forwarded and cached by the supervisor. A combined-families scene test. |
| 7 | #141 | `feat/ms7e-part-table` | 99e88f4, then 0773c54 | 0798144 | `scripts/mj_part_table.py` (deviation D3) |
| 8 | #142 | `feat/ms7e-box-load` | 6ffebd3 | bb3e3ac | 1ef4829 `box_poses` and `box_sizes` in each episode record. 46a144c the fit of rule R11. c40bb22 docs. |
| 9 | #143 | `feat/ms7e-carry-via` | eae4007 | b51d9b2 | T1: 0fb2b80 the arc carry, da5a5c9 `--carry_via`, eae4007 the envelope's `--pick` |
| 10 | #144 | `feat/ms7e-passenger` | 3f5629b, then f8faeee | f8faeee | T2: 46441f4 taken as 3f5629b. f8faeee differs from 3f5629b only in a74045b. |
| 11 | #145 | `feat/ms7e-wall-band` | 4ca0e49 | 4ca0e49 | S3 |
| 12 | #146 | `feat/ms7e-thin-snap` | 3e9ee7b | 3e9ee7b | S2 |
| 13 | #147 | `feat/ms7e-ladder` | 1acb329 | 1acb329 | the supervisor's `--seen_seeds` for the seen ladder row |

Two branches stay pushed without a PR, with their switches off: `feat/ms7e-loop-passenger` 764827d
(S1) and `feat/ms7e-thin-rake` fd3a5fa (T3). Design section 10.1 merges a rung that is not adopted
with its switch off. S1 and T3 are levers of rules R12 and R13, and the ledger does not apply that
rule to them. They stay as the record (section 9). The CPU
suite runs 1302 tests at 86263d6, 1381 at c40bb22, 1394 at 3f5629b and f8faeee, 1418 at 3e9ee7b and
1424 at 1acb329.

The design restacks its seven branches in one order (design section 10.1). The campaign adds the
part table (D3), the box load fit (R11), two teacher levers (R12), two serving levers (R13) and the
ladder switch.

### 7.2 The deviations and the protocol changes

Every entry below is in the ledger at the time given.

| time | ledger word | what happened | why |
| --- | --- | --- | --- |
| 15:50 | D1 | Every branch starts from main c728747, not 92c9e91. | The two later merges, #130 (the Thor timing rows) and #131 (the MuJoCo EGL test default), touch no sim code. |
| 15:50 | D2 | The run starts on 2026-09-29 15:45, about half a day before design day 1. The clock deadlines stay. | Wave 1 can start earlier once the box rung and the round and LDraw tracks have their verdicts. |
| 15:50 | D3 | `scripts/mj_part_table.py` goes on its own branch, `feat/ms7e-part-table`. | The part table grows in parallel with the base and restacks onto it. The round and flat parts are already constructible. |
| 15:50 | D4 | The round family has no switch of its own beyond `--families round`. | The collision studs and condim 6 come with the family. |
| 15:56 | | Each wave records on all three hosts, and Machine B records 14 processes of wave 1. The design gives the recording to Machine C and Machine A. | The owner asked at 15:52. |
| 16:16 | RULE | A lever table compares against a baseline from the same architecture. | Machine C's physics differs from x86 in 41 of 48 round records. |
| 16:18 | D5 | Each new family shuffles its split on its own generator. | A family adopted between waves cannot move another family's held-out parts. |
| 16:20 | D6 | The B1 band becomes x 0.03 to 0.25 m and \|y\| 0.18 to 0.20 m. A sequential sampler replaces rejection sampling. `FRAME_BACK_X` is -0.015 m. | The band as written has a ceiling of 4, and the rejection sampler never packs 5. |
| 16:50 | D7 | B2 is adopted at `X_BACK` -0.07 m. | As written, the ceiling is 6. The design marks `X_BACK` `INITIAL:` and moves it in 0.01 m steps. |
| 17:10 | DEVIATIONS | The LDraw generator fills enclosed cavities. A part that fills 0.97 of its hull gets one hull. The library files carry CC BY 4.0. | A squeezing jaw passes through a hollow hull. The design said CC BY 2.0. |
| 17:10 | D8 | The LDraw hull cap drops to 4. | D8 reads the speed rule per lane, and lane 2 falls under half. |
| 17:20 | R11 | A capacity fit under the box draw switch. | Small, shallow boxes with two classes spill. The design's draw has no capacity rule. |
| 17:36 | D9 | The waves pass `--episode_length_s 1500`. | The recorder formula of design section 9, 1030 s = 30 + 20 x 50, ended a smoke with 8 parts left. A B2 cycle takes about 22 s. The read client is not affected. |
| 17:36 | | Teacher reads leave `--perceived` out. | The oracle driver refuses it. |
| 17:50 | D10 | A census runs before a rung leaves. | The losses spread over every family, and the evidence points at the box count. |
| 18:23 | R12 | Three teacher levers, each behind a switch, and a rule for B2 to stay. | The census mechanisms. |
| 19:28, 20:34 | | Wave 1 starts at 19:28 and wave 2 at 20:34, both on 2026-09-29. The design schedules them for 10-01 and 10-02 at 12:00. | D2 allows an earlier wave 1. Wave 2 records in every case (design section 7.1), and its configuration equals wave 1's. |
| 20:08 | | Wave 2 records on Machine C and Machine A only. | Machine B keeps its CPUs for retrain 1. |
| 23:03 | D11 | The ledger sums the screens of 50k, 100k and 150k from their own EVAL_OK payloads. Rule R9 asks for a read again in a fresh TAG. | Their supervisors crashed after they wrote the payloads. |
| 23:32 | R13 | Three serving levers. The adopted ones join candidate E2's checked read and the official read. | The E1 loss census. The levers deviate from the frozen protocol of rule R3, as the MS7d levers did. |
| 01:17 | | S1 leaves rule R13 unbuilt. | The overhead camera cannot see the riders, and the jaw gap catches 6 of 27. |
| 03:14 | R14 | Each candidate's official read uses the protocol and the SHA of its own checked read. | Candidate E1's checked read ran at f8faeee with no R13 lever. Candidate E2's ran at 3e9ee7b with S2 and S3. |
| 04:21 | | The supervisor gains `--seen_seeds` (1acb329) for the seen ladder row. The row reads on Machine C. | Only the first env of an eval draw repeats wave 1's boxes. The box draw does not depend on the architecture. |

The ledger also corrects itself three times:

- 16:16: the round baseline of 16:15 came from Machine C. The x86 baseline reads 45 of 48.
- 16:30 and 17:13: the log times of the entries from 15:50 to 17:10 were first estimates. Every
  entry now takes its time from `date` or from the files it cites.
- 17:36: the flat tables of 16:23 ran with the four base classes, so tables at the stack top do not
  repeat them bit for bit.

### 7.3 The defects found and fixed

| found | defect | effect | fix |
| --- | --- | --- | --- |
| 17:04 | A per-episode resize left stale collision bounds (design section 11, the first risk). | A 1x1 brick fell through a grown box floor. | The resize refits `geom_rbound`, `geom_aabb` and the body BVH. 4c90d78 keeps the deep floor on a resize. |
| boxes phase 2 | The recorder's shard writer dropped `initial_state/desk_box_sizes`, `box_count` and `box_draw_rounds` under `--box_draw` at 4c90d78. | A corpus shard lost its drawn boxes. | a388321. 1ef4829 then adds `box_poses` and `box_sizes` to each episode record. |
| 17:41 | The rig never handed back an abandoned takeover, after a watchdog or a NaN guard. | The teacher loop waited to the episode's end. A B2 smoke heap lost 34 of 50 parts. The B2 smoke read 237 of 300 before the fix and 259 after. | 0dca08a on the lean branch. The controller rebased the upper branches onto it. |
| 20:11 | The controller edited `bin/wave_host.sh` while it ran. Bash reads a running script again by offset. | The launcher failed with "unexpected EOF" (JOB_EXIT rc=2) on Machine B, and later on Machine C. The corpus step was over, so no shard was lost. | The wave scripts change only between runs. |
| 23:03 | The gate supervisor's per-box sum in `run_gate_eval.aggregate_gate` assumed that every episode lists the same number of boxes. The box draw breaks that. | Every screen lane wrote its payload and then crashed with an IndexError. The 50k, 100k and 150k screens of retrain 1 ended READ_FAIL. | a74045b sums each box slot over the episodes that have it. The test fails without the fix. New top f8faeee. D11 covers the three crashed screens. |

The hand-back fix 0dca08a changes every takeover read: the teacher, the rake and the scripted place.
It changes a read only where a run used to wait to its time-out after an abandon. The family heap
reads of 16:58 to 17:10 ran without it.

Three process faults also cost time:

- 16:17: the shared Machine A venv carried `maker_arm_rs` e154151, not main's pin 6141b84, and 13
  tracking tests failed. `uv sync --group dev --inexact` in the main clone installed 6141b84. The
  diff adds dynamics and tracking only, so no sim number moves.
- 16:30: the controller's brief put `--scripted_rake` in the teacher's flags. The oracle driver
  rakes by itself, so the first round heap reads ended EVAL_FAIL on usage. They are not reads, and
  they ran again through `bin/teacher_heap.sh`.
- 16:58: Machine A's SSH agent socket went away. Git now goes over HTTPS through the gh credential
  helper, with no repo configuration change (`e/git_https_note.txt`).

## 8. What the simulation does not promise for the real arm

MS7e runs in simulation only. Several of its fixes and constants have no real counterpart or no real
measurement:

- The flat deep floor is a fix of a simulation artifact. At 240 Hz, a 3.2 mm part falling at about
  1.3 m/s tunnels through the 5 mm box floor. A real floor does not let a part through, so the deep
  floor has no real counterpart. The flat pass rests on the simulator's contact model.
- The real fingertip geometry is not measured (design section 6.6). In simulation, a 3.2 mm grasp
  holds at about 1.1 mm of gap with about 0.5 N per jaw. During CLOSE the tips press 2 mm on the
  desk. So a flat pass here promises no real pick of a plate or a tile.
- `ROUND_ROLLING_FRICTION` = 0.001 is `INITIAL:`. A roll probe in simulation set it, not a
  measurement of real ABS on the real desk. The torsional friction of 0.005 is also a simulation
  value.
- The LDraw parts are convex hulls from CoACD, capped at 4 per part. The generator fills enclosed
  cavities, and a part that fills 0.97 of its hull gets one hull. The studs stay visual. A real part
  has 1.6 mm walls and a hollow underside that a jaw can enter. The hull model does not show that.
- The capacity fit reads the bounding volumes of the heap's own parts before the episode. Its
  constant, `BOX_VOLUME_PER_PART_VOLUME` = 2.0, is `INITIAL:`. A real rig does not know its heap in
  advance, and no real box measured the constant.
- T2, the passenger check, reads the simulator's part poses and runs in the teacher only. S1 shows
  that the overhead camera cannot see the riders. A real rig needs a wrist-camera look after the
  lift or a load signal.
- `CARRY_MIN_RADIUS_M` = 0.10 m and the passenger thresholds of 0.02 m and 0.06 m are `INITIAL:`. So
  are `CONT_HOLD_BAND` = 1 mm and the thin snap of 2 mm.
- B2 moves the overhead camera 0.139 m back. A real rig must move its camera mount the same way. If
  not, the policy sees a camera pose outside its training draws.
- The x86 and aarch64 builds do not share physics. So a single simulator number is a number of one
  architecture.
- Every perceived input keeps the MS7d truth-plus-noise stand-ins at nominal knobs. Their values are
  optimistic until MS5a measures them.

## 9. Next steps

MS7e meets its gate on its one official read. The margins are 199 stows and 0.029 cycles per correct
stow. Every read of candidate E2 holds the stow bar with room, and two ladder rows miss the cycles
bar. So the cycles bar binds first.

The steps, in order:

1. Merge stack #148 and the report PR on top of #147. The owner decides whether the T3 branch and
   the S1 branch get PRs with their switches off, as design section 10.1 does for a rung that is not
   adopted.
2. Treat the cycles bar as the binding bar. The seen row reads 1.306 and the rim row 1.310, against
   1.3. The official read passes by 0.029. A next lever must save cycles before it adds stows.
3. Work on the flat parts first. They are the weakest family, and they cost the most cycles. In the
   official read, plates stow 706 of 925 (0.763) and tiles 325 of 485 (0.670). The flat parts hold
   967 of the 1438 excess cycles. Without them, the read uses 1.11 cycles per correct stow. S2 cut
   the flat `close_miss` cycles of its row from 188 to 121. The official read still has 486
   `close_miss`, 350 `lifted_lost` and 320 `held_no_lift` cycles on flat parts. A study of the flat
   grasp in the serving loop comes before any new lever.
4. Give the loop a passenger signal. The wrong-box parts are 6.9 % of the official read, against
   4.7 % on MS7d. The E1 census traced most of them to passengers. The overhead camera cannot see a
   rider (S1), and the jaw gap catches only 6 of 27. A wrist-camera look after the lift or a load
   signal is the next test. The QDD torque of the contact spike is one candidate for the load
   signal. Both options lie outside MS7e.
5. Record the next corpus with the teacher's own wall band. S3 raised the teacher from 277 to 289 of
   300 on the R12 seeds, and it cut the wall-stuck parts from 16 to 4. The waves ran without it. The
   recorder already takes `--wall_band_cont` (4ca0e49), so a future corpus can turn it on.
6. Carry the MS7e results to the real arm only with real measurements. MS5 phase P runs on the real
   arm in parallel (decision E1). Section 8 lists what the simulation does not promise. Before an
   MS7e rung moves to the real arm, three items need a real measurement: the fingertip geometry on
   flat parts, a passenger signal, and a camera mount that matches the B2 shift of 0.139 m.

The open items:

- T3's causes stay open. The rake tips ride 2.4 to 2.7 mm over the desk at `PUSH_Z`, and teacher
  reads lack `--rake_turn`.
- The recorder's episode meta does not count the passenger rejects. The read payload does.
- The x86 and aarch64 physics differ, so every lever table needs a baseline on its own architecture.
  The seen ladder row read on Machine C, so it does not pair with the x86 rows.
- The unseen size and rim rows lie outside the certification of the envelope check. Both read at
  least 0.881, but no envelope check covers boxes over 0.10 m or a depth over 0.05 m.
- Count 8 is the weakest box count of both checked reads, at 0.861 and 0.859. In the official read,
  count 8 reads 0.874, level with count 6.

MS7e widened the MS7d task in simulation along all three axes of the design. It met its gate with
the planned retrain, one extra retrain, two teacher levers and two serving levers, each behind a
switch. The flat parts and the passengers remain the two largest costs. The cycles bar, not the
stow bar, is the next limit.