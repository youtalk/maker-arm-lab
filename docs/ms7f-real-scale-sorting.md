# MS7f: real-scale sorting in simulation

Date: 2026-10-01. Status: draft. The campaign still runs. This draft uses the ledger up to its entry
of 05:38 on 2026-10-01. A line that starts with `TBD:` waits for a read or a decision that had no
number at that time. The controller fills those lines.

The design of MS7f is the document that the owner reviewed and approved in chat on 2026-09-30. This
report cites it as "MS7f design section N" and restates every rule that it uses. The campaign ledger
is `~/data/maker-arm-ms7/f/README.md`, and `STATE.md` beside it is the resume pointer. Every rule
below went into that ledger before its number. The measurement tables are under
`~/data/maker-arm-ms7/f/tables/`. Each read has a TAG, its name. The read payloads are under
`~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7f TAG starts with `f_`.

The ledger names its rules R1 to R15 and its deviations D1 to D10. The design names its decisions F1
to F10. The tracks are H (the heap), C (the classifier), P (the part families) and S (the speed).
The retrains are F1 and F2, and their checkpoints are written as "F1 100k". This report writes "rule
R4", "decision F6" and "retrain F1" to keep them apart.

All times are PDT. Each time is the time of the ledger entry that records the event. Times from
10:13 to 23:47 are on 2026-09-30. Times after midnight are on 2026-10-01.

Numbers come from the ledger or from the files that it names. This report also computes some values
from the ledger's counts: fractions, rates an hour and Wilson intervals that the ledger does not
give. The text marks each of them.

## 0. Summary

MS7e met its gate. Its official read stowed 5299 of 6000 held-out parts in their own class's box,
0.883, at 1.271 cycles per correct stow. A correct stow took 21.5 simulated seconds, 167 an hour.
MS7f makes the task real-scale along three axes (MS7f design section 1):

1. Continuous operation and speed. The bar is at least 300 correct stows per simulated hour.
2. A dense large heap. An episode holds 150 parts, against 50 in MS7e.
3. A large catalog sorted by category. Two new families, joint and minifig, join the catalog. The
   parts go to six categories, and a classifier trained on rendered images replaces the class
   stand-in.

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
- Every screen lies far under every bar of the gate: 0.92 in a box of the category, 300 correct
  stows an hour and 0.85 on the held-out row.
- The control read, E2 with no retrain, stows 768 of 1050, 0.731, at 151.3 correct stows an hour.
  That is the level of candidate 1's screen, so retrain F1 bought no visible gain.

The decomposition of section 10 finds four facts:

- On one heap of candidate 1, the failed cycles hold 65 % of the steps.
- On 13 screen heaps, 0.194 of the parts end in a wrong box. The classifier sets 197 of the 208
  `wrong_box` cycles, and 171 more parts reach a wrong box as passengers.
- The wave teacher delivers a part every 13.91 to 16.23 s. This report computes that as 222 to 259
  parts an hour, under the 300 bar.
- Candidate 1 runs 17.0 s a cycle, so even one cycle per correct stow reads under 300 an hour.

The gate outcome:

| read | checkpoint | commit, classifier | correct stows | correct stows an hour | held-out row | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| candidate 1, screen | F1 100k | d6f893d, v5 | 425 of 600, 0.708 | 131.4 | 99 of 151, 0.656 | no bar, picked by rule R11 |
| candidate 1, checked | F1 100k | 03a5cd8, v6a | TBD | TBD | TBD | TBD |
| candidate 2, checked | TBD | TBD | TBD | TBD | TBD | TBD |
| official | TBD | TBD | TBD | TBD | TBD | TBD |
| control | train_E2 150k | d6f893d, v5, aarch64 | 768 of 1050, 0.731 | 151.3 | 180 of 263, 0.684 | no bar, reported |

TBD: candidate 1's checked read f_F1_chk (correct stows of 3000, an hour, held-out row, outcome).

TBD: the trigger of retrain F2 (fires or not) and, on a fire, candidate 2's checked read.

TBD: the official read, or the statement that the milestone fails by rule R2 with no passing checked
read.

TBD: the one-paragraph outcome of the milestone.

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

Rule R2 sets the gate. A checked read has three bars. It needs at least 2760 of 3000 correct stows,
at least 300 an hour and a held-out row of at least 0.85. The official read goes to the candidate
whose checked read passes. With two passing candidates, the higher throughput wins. The official
read is one shot. It passes with at least 5520 of 6000, at least 300 an hour and a held-out row of
at least 0.85. If no checked read passes, the milestone fails, and the report gives the
decomposition.

| read | policy | seed and size | pass |
| --- | --- | --- | --- |
| screen | each retrain checkpoint | 20307000, draws 0 to 3, 4 heaps of 150, 600 parts | none, picks the candidate (rule R11) |
| checked | candidate 1, and candidate 2 on the retrain trigger | 20307100, draws 0 to 19, 3000 parts | at least 2760 correct stows, at least 300 an hour, held-out row at least 0.85 |
| official | the candidate that rule R2 picks | 20307200, 40 heaps, 6000 parts, one shot | at least 5520 correct stows, at least 300 an hour, held-out row at least 0.85 |
| control | train_E2 150k, no retrain | 20307300, draws 0 to 6, 1050 parts | none, reported |
| stretch | the official candidate | run 20307400 + r, layout 20307450 + r, r = 0 and 1, 1000 parts each | none, reported against 0.95, 400 an hour, 1000 parts, 5 interventions |

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
| teacher reference read | run 20306000 + i, layout 20306050 + i, i = 0 to 5 |
| screens, checked, official, control | 20307000, 20307100, 20307200, 20307300 |
| stretch runs | run 20307400 + r, layout 20307450 + r, r = 0 and 1 |

Rule R1 grew twice, each time before any use. At 17:42 it added a render extension range for the
classifier: run 20310000 + i, i = 0 to 999. The range covers train kinds and the D4 extra kinds
only. At 03:21 the range grew to i = 0 to 1999, and i = 1000 to 1999 went to model v6.

### 1.6 The rules that carry over and the hosts

Rule R14 carries over MS7e rule R9. A payload from the supervisor cache is not a read. Every read
records its SHA, `policy_id`, split, switches and knob scales. A READ_FAIL reads once more in a
fresh TAG. Every paired table takes its base on its own architecture, because x86 and aarch64
physics differ. The MS6 negatives and the generic abort on contact stay closed.

Rule R15 keeps `scripts/fleet.py` and `~/data/maker-arm-ms7/hosts.json`, with track aliases
`a_fN`, `b_fN` and `aarch64_fN`.

| host | hardware | MS7f work |
| --- | --- | --- |
| Machine A | 24 cores, RTX 5090 | the controller and the ledger, the P tables, the classifier training, both retrains, read lanes, half of wave 1 |
| Machine B | 24 cores, RTX 5080 | read lanes, half of wave 1, most of wave 2 |
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
v6a). The ledger does not record why.

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
bar. `--passenger_drop` is not adopted at C1. Deviation D9 gives it one screen in the loop (section
9.4).

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

TBD: the sixth heap of f_ref and the six-heap totals, or a note that the verdict stays on five
heaps.

The ledger counts 106 parts left, 70 of them in the tray and never targeted. The R4 H1 arm had 37
such parts over 6 heaps.

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

Every episode ends on its time-out, as in wave 1. Machine B's disk filled at 01:48 (984 GB, 0 free), and
its conversion of p 12 to 39 failed on ENOSPC in every lane. Freeing it needs the owner. The 28
shards (91 GB) moved over the LAN to Machine C in about 13 minutes and converted there
(CONVERT_OK 3610 cycle episodes, 03:46). Both wave 2 datasets then shipped to Machine A.

The merge for retrain F2 (05:19) joins f/w1/dataset (5204) and both wave 2 halves (5180) into
f/w12/dataset: MERGE_OK 10,384 cycle episodes and 3,490,315 frames, CHECK_OK 164 videos.

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
129 lerobot_stack tests and 1552 repo tests pass.

The pre-registered rule: candidate 1 reads at X = 1.25 and at X = 1.5. Both screens use seed
20307000, draws 0 to 3, the four heaps of its own screen at X = 1.0. The TAGs are
f_F1_100000_w125_scr and f_F1_100000_w150_scr. A value of X qualifies with a correct-stow share at
most 0.02 under the X = 1.0 screen's. The qualified X with the most correct stows an hour wins, and
X = 1.0 always qualifies. One binomial standard error on 600 parts is about 0.015. The screens keep
model v5.

| X | TAG | correct stows | correct stows an hour | held-out row | verdict |
| --- | --- | --- | --- | --- | --- |
| 1.0 | f_F1_100000_scr | 425 of 600, 0.708 | 131.4 | 99 of 151 | the base |
| 1.25 | f_F1_100000_w125_scr | TBD | TBD | TBD | TBD |
| 1.5 | f_F1_100000_w150_scr | TBD | TBD | TBD | TBD |

TBD: the two warp screens' rows, the winning X by the D7 rule, and what the cycle intervals show.

### 9.4 The passenger-drop screen: deviation D9

Passengers cost about 9 % of the screen parts (section 10.3). Deviation D9 (04:07, before any
number) gives `--passenger_drop` one screen: candidate 1 at X = 1.0 with `--passenger_drop`, seed
20307000, draws 0 to 3, TAG f_F1_100000_pd_scr, model v5. It has two bars against the candidate's
own screen. Its correct stows are at least that screen's plus 6. Its correct stows an hour are at
least 0.97 of that screen's. A screen that holds both bars is adopted, and the switch then joins
every later read.

| row | correct stows | correct stows an hour | wrong box | passenger drops | verdict |
| --- | --- | --- | --- | --- | --- |
| f_F1_100000_scr, the base | 425 of 600 | 131.4 | 119 | | |
| f_F1_100000_pd_scr | TBD | TBD | TBD | TBD | TBD |
| bar | at least 431 | at least 127.5 | | | |

This report computes the bar from the base's 425 and 131.4.

TBD: the passenger-drop screen's row and its verdict by the D9 rule.

### 9.5 The v6a diagnostic screen

The ledger pre-registered a diagnostic screen at 04:37: candidate 1 at X = 1.0 with model v6a, seed
20307000, draws 0 to 3, TAG f_F1_100000_v6_scr. No decision rides on it, because C1 already adopted
v6a. Paired with f_F1_100000_scr (v5), it measures the classifier lever inside the loop.

| row | model | correct stows | correct stows an hour | wrong box | `wrong_box` cycles set by the classifier |
| --- | --- | --- | --- | --- | --- |
| f_F1_100000_scr | v5 | 425 of 600 | 131.4 | 119 | |
| f_F1_100000_v6_scr | v6a | TBD | TBD | TBD | TBD |

TBD: the v6a diagnostic screen's row and its pairing with the v5 screen.

### 9.6 Candidate 1's checked read

The ledger pre-registered the checked read at 02:20. Two later entries amended it, and a third set
its job:

- 02:20: candidate 1, seed 20307100, draws 0 to 19, 3000 parts, TAG f_F1_chk. It passes with at
  least 2760 correct stows, at least 300 correct stows an hour and a held-out row of at least 0.85.
- 03:20, deviation D8: it starts after the warp screens and reads at D7's winning X. Machine A
  cannot run the 8 warp heaps and the 20 checked heaps together, and Machine B's disk is full. Its lanes
  are a_fr2l at 03a5cd8, which reads byte-equal to d6f893d at X = 1.0.
- 04:11: model v6a serves it.
- 05:20: job f_chk_F1 waits for the D7 and D9 screens, applies their rules from the
  `gate_results.json` files and starts f_F1_chk on `a_fr2l:20`, `PORT_BASE` 15000.

| item | value |
| --- | --- |
| TAG | f_F1_chk |
| checkpoint | F1 100k |
| commit, classifier | 03a5cd8, v6a |
| added flags | TBD: `--action_warp X` and `--passenger_drop` as the D7 and D9 rules set them |
| correct stows of 3000 | TBD |
| Wilson 95 % | TBD |
| correct stows an hour | TBD |
| held-out row | TBD |
| cycles per correct stow, no bar | TBD |
| wrong-box parts | TBD |
| outcome | TBD |

TBD: the rows by category, by family and by box count of f_F1_chk, and its events.

### 9.7 Retrain F2 and its screens: deviations D6 and D10

Rule R11 sets a trigger for retrain F2: candidate 1's checked read stows under 2790 of 3000 (0.93),
or under 320 correct stows an hour. On the trigger, F2 warm starts from candidate 1 on waves 1 and
2. At most two retrains run.

Deviation D6 (02:20, before any screen number): retrain F2 starts at the end of the F1 screens. It
does not wait for candidate 1's checked read, because the deep-tray reads take hours and the GPU
sits idle. Its checkpoints count for the gate only on R11's trigger. Without the trigger, they are
reported and set aside.

Retrain F2 started at 05:04 on Machine A. It merged f/w12/dataset (section 8.3) and trains 200k
steps from F1 100k, a checkpoint every 50k (f/train_F2). At 05:19 it ran at about 25.5 steps/s next
to the 16 screen lanes.

Deviation D10 (05:19, before any F2 number): retrain F2's screens run as soon as each checkpoint
exists, on Machine C. They count only on the trigger. Each checkpoint 50k to 200k reads
seed 20307000, draws 0 to 3, through `read_f.sh` at d6f893d. The reads use model v6a at X = 1.0 on
`c_fr1:4`, with TAGs f_F2_<step>_scr. Rule R11's rule picks candidate 2 among them. They read
the aarch64 physics with a CPU policy, so they compare only with each other.

| step | TAG | correct stows | Wilson 95 % | correct stows an hour | held-out row | wrong box |
| --- | --- | --- | --- | --- | --- | --- |
| 50k | f_F2_050000_scr | TBD | TBD | TBD | TBD | TBD |
| 100k | f_F2_100000_scr | TBD | TBD | TBD | TBD | TBD |
| 150k | f_F2_150000_scr | TBD | TBD | TBD | TBD | TBD |
| 200k | f_F2_200000_scr | TBD | TBD | TBD | TBD | TBD |

TBD: retrain F2's TRAIN_OK time and the second copies of its checkpoints.

TBD: whether R11's trigger fires on f_F1_chk (under 2790 or under 320 an hour).

TBD: on the trigger, candidate 2 by rule R11 and its checked read with its pre-registration.

### 9.8 The official read

Rule R2: the official read runs on seed 20307200, 40 heaps of 150, 6000 parts, one shot. It reads
the candidate whose checked read passes. With two passing candidates, it reads the one with the
higher throughput. It passes with at least 5520 correct stows, at least 300 correct stows an hour
and a held-out row of at least 0.85. If no checked read passes, the official read does not run, and
the milestone fails.

TBD: the official read's pre-registration (candidate, commit, classifier, lanes), its rows and its
outcome, or the statement that it did not run.

### 9.9 The stretch runs

Rule R13: after the official read, the official candidate runs two episodes of 1000 parts, one env
each. The seeds are run 20307400 + r and layout 20307450 + r, r = 0 and 1. The tray refills from the
episode's pool, full boxes are emptied, and the category map stays fixed for the run.
`scripts/mj_stretch_run.py` (3762445) runs them. An intervention is an event that the loop cannot
resolve:

1. The episode ends with parts in the tray: 1 intervention.
2. A part rests on the desk outside the tray and every box: 1 per part.
3. Ten cycles in a row end with no correct stow: 1, and the loop restarts.

The runs report against four targets with no bar: 0.95 in a box of the category, 400 an hour, 1000
consecutive parts and 5 interventions per 1000 parts.

| run | correct box | correct stows an hour | consecutive parts | interventions per 1000 parts |
| --- | --- | --- | --- | --- |
| r = 0 | TBD | TBD | TBD | TBD |
| r = 1 | TBD | TBD | TBD | TBD |

TBD: the stretch runs, or the statement that they did not run because no official read ran.

### 9.10 The control read

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

## 10. The decomposition

With no passing checked read, rule R2 asks for the decomposition. The screens already place the
gaps, so this section gives the decomposition that the ledger has.

### 10.1 The throughput arithmetic

The bar of 300 correct stows an hour is 12.0 s, or 360 control steps at 30 Hz, per correct stow. The
throughput is the product of two factors: the seconds a cycle and the cycles per correct stow. This
report computes the values from the ledger's counts. For the wave rows, the columns count delivered
parts in place of correct stows:

| row | s a cycle | cycles per correct stow | s per correct stow | correct stows an hour |
| --- | --- | --- | --- | --- |
| the bar | | | 12.0 | 300 |
| candidate 1, screen (F1 100k) | 17.0 | 1.612 | 27.4 | 131.4 |
| control read, E2 on the gate configuration, aarch64 | 17.0 | 1.398 | 23.8 | 151.3 |
| E2 on the MS7e configuration, S0 | 16.98 | 1.275 | 21.64 | 166 |
| wave 1 teacher, Machine A share | 13.85 | 1.172 | 16.23 per delivered part | 222 delivered parts |
| wave 2 teacher, aarch64 share | | 1.130 | 13.91 per delivered part | 259 delivered parts |
| wave 2 teacher, Machine B share | | 1.167 | 14.81 per delivered part | 243 delivered parts |

Three facts follow from the table:

- At candidate 1's 17.0 s a cycle, even one cycle per correct stow gives 212 an hour. To reach 300
  at 17.0 s a cycle, the cycles per correct stow must fall to 0.71, under one.
- E2 ran 16.98 s a cycle on the MS7e configuration, near its corpus's 16.7 s, and 17.0 s on the
  gate configuration. Candidate 1 also runs 17.0 s a cycle, against 13.85 s for Machine A's share
  of its wave 1 corpus.
- The wave teacher delivers 222 to 259 parts an hour. A policy that copies its corpus teacher cycle
  for cycle reads under 300 an hour, even with no failed cycle.

The design's target for the teacher was 10 s per correct stow, 360 an hour. With about 1.15 cycles
per correct stow, a policy at that pace reaches 12 s and 300 an hour (MS7f design section 8.2). The
read teacher came near that target: S1's comb3 reads 10.30 s on 50-part heaps, and the reference
read 11.27 s on 150-part heaps. The recorder recipe stays slower. M5 reads 11.24 s per delivered
part on 50-part heaps, and the deep-tray waves 13.91 to 16.23 s.

### 10.2 Where a screen's steps go

The 100k screen's draw 3 is one heap of 150 (03:15). It stows 114 of 150 in 172 cycles and 77,390
steps, 159.1 correct stows an hour. The cycle intervals split its steps by cycle outcome. This
report computes the shares and the steps a cycle:

| cycle outcome | cycles | steps | steps a cycle | share of the steps |
| --- | --- | --- | --- | --- |
| stowed | 95 | 26,707 | 281 | 0.345 |
| `close_miss` | 29 | 21,994 | 758 | 0.284 |
| `lifted_lost` | 17 | 11,446 | 673 | 0.148 |
| `wrong_box` | 15 | 5,159 | 344 | 0.067 |
| `held_no_lift` | 8 | 4,857 | 607 | 0.063 |
| `carrying`, `no_close`, `no_open` and the rest | 3, 3 and 2 | 7,227 | | 0.093 |

Failed cycles hold 65 % of the steps. 52 cycles ended on the 800-step budget. A stowed cycle takes
281 steps, 9.4 s, under the 12 s of the bar. A failed `close_miss` cycle takes 758 steps, almost
three stowed cycles. On the MS7e configuration, E2's failed cycles held 45 % of the steps (S0).

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
cycles, so passengers also reach their own box.

### 10.3 The wrong-box parts

On the same 13 heaps, 379 of 1950 parts end in a wrong box, 0.194. They split in two groups:

| source | parts | share of the 1950 parts |
| --- | --- | --- |
| `wrong_box` cycles | 208 | 0.107 |
| of them, the classifier's | 197 | 0.101 |
| of them, special read as brick | 74 | |
| of them, brick read as special | 51 | |
| of them, brick read as flat | 15 | |
| passengers, parts that reached a wrong box without a cycle of their own | 171 | 0.088 |
| all wrong-box parts | 379 | 0.194 |

The classifier's taken class is right on 1253 of 1505 lifted cycles, 0.833 (model v5). C1's
`loop_sum` reads 0.928 on teacher rollouts. The 100k heap of section 10.2 shows the same split: 14
of its 15 `wrong_box` cycles are the classifier's, 12 set by the wrist vote at the lift and 2 by the
overhead read. 6 are special read as brick and 3 brick read as special. Its other 11 wrong-box parts
rode along.

This report computes one bound from candidate 1's screen. Its 425 correct stows plus its 119
wrong-box parts make 544 of 600, 0.907, still under 0.92. The screen misses 175 parts. Wrong boxes
hold 119 of them, and the other 56 end outside every box. With the class stand-in on the MS7e
50-part configuration, E2 reads 0.907 (C2).

### 10.4 The teacher's own share and pace

The teacher rows against the gate's two numbers:

| read | correct box or delivered | s per correct stow or delivered part |
| --- | --- | --- |
| R4 H1 teacher, 6 heaps, no `--perceived` | 842 of 900, 0.936 | 18.42 |
| reference read, 5 heaps, with `--book_miss` | 641 of 750, 0.855 | 11.27 |
| census row B, 3 diagnostic heaps, no `--book_miss` | 422 of 450, 0.938 | about 10.6 |
| wave 1 recorder | 5216 of 6000, 0.869 | 16.23 (Machine A share) |
| wave 2 recorder | 5155 of 6000, 0.859 | 13.91 and 14.81 |
| the gate | 0.92 | 12.0 |

Only census row B meets both numbers, on three diagnostic heaps without `--book_miss` and without
the recorder's draws. The recorder delivers 0.859 to 0.869 of its parts, under 0.92, at 13.91 to
16.23 s a part, over 12 s. Every wave episode ends on its time-out.

TBD: the decomposition of candidate 1's checked read, and of candidate 2's and the official read
where they ran.

## 11. The code, the deviations and the defects

### 11.1 The stack

All branches start from 29a36ef (deviation D2). Deviation D5 sets the order base, heap, parts,
speed, classifier. The wave fixes and the extra kinds sit between speed and the classifier (20:57).
The ledger records these tips:

| order | branch | tips in the ledger | what it adds |
| --- | --- | --- | --- |
| 1 | `feat/ms7f-base` | b4a2af1 | 4c3b66f the six categories and their box map. 4f83b2f `--categories` through the env, the grader, the teacher and the loop. eabac0c the throughput, the held-out row and the S0 intervals. 2c5cce5 every MS7f switch declared. b4a2af1 heaps up to 150. |
| 2 | `feat/ms7f-heap` | 3fa9f3f | 0623cc3 the deep tray and the refill. c59cbcc box emptying. ee8b24e the hidden share. 59d5e65 `--render`. 098e5a4 the stretch rules. e5d7175 the box-load reset. 9a9fb7e the recorder hook. 3762445 `scripts/mj_stretch_run.py`. 3fa9f3f `--book_miss`. |
| 3 | `feat/ms7f-parts` | 9754ee8, then 34c7e0e | the table's `--seed_base`, the joint and minifig families, two axis data fixes, `perception.OWN_AXES` for the new families, the hat dropped |
| 4 | `feat/ms7f-speed` | dd4a8bb, then 077c7f0 | `--transport_speed` in the heap client, the teacher waits, `--lift_budget`, `--close_guard`, `--gain_min`, `--quiet_landing`, `--landing_steps`, `--end_at_done`, `--net_stall` |
| 5 | `feat/ms7f-heap-w1` | 4464b79 | `--abandon_cycle`, `--near_tracking` (not used), parts in rows of 50, the pile top |
| 6 | `feat/ms7f-cls-extra` | 690117f, then e2114d3, then 5bc55ae | the D4 extra kinds |
| 7 | `feat/ms7f-classifier` | 0322cfc, then d6f893d | the narrow overhead camera, the render set, the trainer and the three-view vote. `--classifier`, `--passenger_drop`, the crops, the size input and the regularisation. |
| 8 | `feat/ms7f-warp` | 03a5cd8 on d6f893d | `--action_warp` (D7) |
| 9 | `feat/ms7f-cls6` | ab3008d on d6f893d | the trainer's `--arch`, `--input_px` and `--ensemble` (v6a) |

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

The design merges a rung that is not adopted with its switch off. After the reads, the branches
restack into one stack with `GH_REPO=youtalk/maker-arm-lab gh stack`, with this report on top.

TBD: the final stack, its PR numbers and the branch tip of each PR.

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

The ledger corrects itself once. At 10:54, the stamps of the rules, the ledger and the agents were
estimates. They now carry the file times: rules 10:15, ledger 10:16, agents 10:20. Every later entry
takes its time from `date`.

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

Five process faults also cost time:

- 16:00: a tracked background call stops after 2 hours, and a deep-tray wave lasts longer.
  Machine A aliases now reach Machine A over ssh to its own address, so fleet.py detaches their
  jobs with setsid under sshd.
- 20:57: orphans of stopped agents held seven Machine A cores, and a census queue runner kept
  starting heaps. They were killed by their own command lines.
- 23:21: the first screen chain named a worktree that does not exist. A `pkill -f` pattern then
  matched a later argument of the same command line and killed the calling shell.
- 23:47: a detached job had no ssh agent, so the 50k screen ended READ_FAIL.
- 01:48: Machine B's disk filled (984 GB), and its wave 2 conversion failed. Freeing MS7f intermediates
  on Machine B needs the owner.

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

The ledger names these open items:

1. The policy runs slower than its teacher. The wave recorder delivers at 13.91 to 16.23 s a part.
   Candidate 1 runs 17.0 s a cycle at 1.612 cycles per correct stow. A faster policy needs a
   faster corpus or a serving lever, and D7's warp screens test the serving lever.
2. Retrain F1 on wave 1 bought no visible gain. The control read, E2 with no retrain, reads 0.731
   at 151.3 an hour, at the level of candidate 1's screen.
3. The failed cycles hold 65 % of a candidate-1 heap's steps. `close_miss` and `lifted_lost` take
   the most.
4. The classifier sets 197 of 208 `wrong_box` cycles on 13 screen heaps, mostly special and brick
   confused. v6a raises special to 0.883 on C1, and the diagnostic screen measures it in the loop.
5. Passengers put 171 of 1950 parts in a wrong box. The passenger head finds 0.50 of them at best.
   D9 tests `--passenger_drop`.
6. When a cycle stows another part, the loop's failure memory does not mark the target, and the
   selector takes the same hard target again. `--book_miss` books the whole cell and hides parts.
   The census pre-registered row E1, the teacher with `--part_failures`, which books the target
   part and not its 5 cm cell. The policy reads already carry that switch. The ledger records that
   rows C, D and E did not finish.
7. The policy's own waits remain: 58 steps after its close and 45 extra on the approach.
8. Machine B's disk is full and needs the owner.

TBD: the steps in order, after the checked read, the trigger, the official read and the stretch
runs.

TBD: the closing paragraph of the milestone.
