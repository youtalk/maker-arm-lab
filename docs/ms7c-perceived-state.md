# MS7c: the perceived-state protocol

Date: 2026-09-25. Status: final. No checked read passes. MS7c fails its gate at nominal noise on the
cycles bar, and the official read does not run.

The design is the MS7c design. The owner approved it section by section in chat. The campaign ledger
is `~/data/maker-arm-ms7/c/README.md`, and `STATE.md` beside it is the resume pointer. Every rule
below went into that ledger before its number. The read payloads are under
`~/data/maker-arm-ms7/reads/<TAG>/`. The counts that the ledger does not give come from those
payloads. The requirement table comes from `~/data/maker-arm-ms7/bin/req_table.py`, which reads the
sweep payloads.

All times are PDT. Each time is the time of the ledger entry that records the event. Times after
midnight are on 2026-09-25, and all other times are on 2026-09-24.

## 0. Summary

MS7a passed in simulation, but its serving loop reads the simulator truth. It selects targets on the
true part poses. It builds the target row and the marker from the true part centre. It reads the
hold from contact forces and the lift from the true part height. It takes the class of the held part
from its identity, and it ends each cycle on a true in-box test. A real rig has none of these
inputs.

MS7c puts one boundary between the simulator and the serving loop. Behind the switch `--perceived`,
the loop reads only a perceived state and a hold signal. The perceived state is the truth passed
through six noise and failure models. Each model has a knob that MS5a calibrates on real frames. The
hold signal comes from the jaw gap alone, which the real gripper motor reports. The grader, the
census and the tracker still read the truth, so every MS7c number compares with MS7a.

The pass rule has two parts. At nominal noise, a checkpoint must stow at least 2550 of 3000 held-out
parts in their own class's box. It must use at most 1.3 cycles per correct stow. The official read
must then repeat the pass on a fresh seed. MS7c also delivers a requirement table. For each knob,
the table gives the perception accuracy that the real rig needs to hold the bars.

The gate outcome: no checked read passes. Four checked reads ran on three checkpoints. Every read
after the release fix meets the stow bar and misses the cycles bar:

| candidate | checkpoint | commit | correct stows of 3000 | cycles per correct stow | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | r2 A2 100k, no retrain | 0cfb61b | 2527, 0.842 | 1.450 | both bars missed |
| 1b | r2 A2 100k, no retrain | eae617a | 2603, 0.868 | 1.414 | the cycles bar missed |
| 2 | train_P 150k, the best screen | eae617a | 2628, 0.876 | 1.341 | the cycles bar missed |
| 3 | train_P 200k, a declared deviation | a5482d6 | 2646, 0.882 | 1.329 | the cycles bar missed |

The milestone fails its gate at nominal noise on the cycles bar. Every candidate after the release
fix meets the stow bar and misses 1.3 cycles per correct stow. The official read does not run, and
the official seed 20262100 stays unused. The paired read on true inputs stows 2770 of 3000, 0.923,
at 1.160.

The other results:

- The hold detector passes its pre-registered bar at the design thresholds on 142 teacher cycles.
  The median offset of the first hold is 5.0 steps, exactly the bar.
- A trace found a defect in the perceived release end. The fix d51c495 waits for the jaws to open
  before the loop switches the row. After the fix, the structure alone costs nothing: with every
  knob at 0, draws 0 and 1 stow 466 of 500, against 464 on true inputs.
- The retrain on a perceived corpus gains 25 to 43 parts and cuts the cycles per correct stow from
  1.414 to 1.341 and 1.329. That is not enough for the bar.
- The position knob uses the whole cycles margin by itself. With only position noise on, the cycles
  per correct stow go from 1.146 to 1.292. The bar is 1.3.
- The position split puts the cost on the lateral (xy) error, not on the height error.
- `--pick_snap` turns the lateral error into the grasp point. It still stays on, because every
  snap-off screen reads 19 to 91 fewer parts.
- The requirement table marks two knobs tight: the position and the class. On the cycles bar, the
  knobs share one budget. Every knob except `miss` reaches 1.3 cycles at a scale of 1.02 to 1.11.
- A close diagnostic on development seeds rejects a search pattern on the snap's retries. Only 18 of
  71 `close_miss` cycles meet its rule, against a bar of half. The failing closes land high, because
  a perception error of more than 4 mm along the closing axis puts a finger on the part.
- A wider landing gap on the snap's descent (`--snap_clearance`) does not beat today's gap on a
  tuning seed, so nothing is adopted.

## 1. The goal, the gate and the rules

The ledger fixed rules R1 to R10 at 15:46, before any MS7c number. The design gives MS7c three days,
from 2026-09-25 to 2026-09-27, and at most one retrain. There is no extension.

| read | seed and draws | pass |
| --- | --- | --- |
| checked, per candidate | 20261700, draws 0 to 11, 3000 parts, a fresh TAG | at least 2550 correct stows, at most 1.3 cycles per correct stow |
| official, only the better passing checked read | 20262100, draws 0 to 23, 6000 parts, 120 episodes | at least 5100 correct stows, at most 1.3 cycles per correct stow |

A correct stow is a part at rest in its own class's box. The design names two candidates: r2 A2 100k
under `--perceived` with no retrain, and the best screen of the retrain. The official read is the
milestone's number. If no checked read passes, the milestone fails, and the report gives the
decomposition of section 8. The design names seed 20261900 for the official read, but an MS7a read
already used that seed (section 9).

The read protocol (R4). Every MS7c read passes `--tray --desk_boxes 4 --tray_lean 30`. It serves the
nine MS7a switches `--select_at_release --replan_on_switch --squeeze_floor --square_wrist
--scripted_rake --part_failures --pick_snap --dwell_close --adopt_held`. It adds the tail fixes
`--park_done --rake_turn` of PR #112, and `--perceived`. It uses held-out parts (`--parts_split
heldout --parts_seed 20260915`), heaps of K = 50 and 50 actions per forward. The knob scales are 1
unless a sweep row says otherwise.

The fallback (R5). If the hold detector misses its bar, the stop flag of the serving rules reads the
contacts again. Everything else stays perceived. A pass under that fallback is `PASS (contact
hold)`, a separate outcome. The detector met its bar, so the fallback does not apply.

The control rule (R6). A control screen at the MS7c commit reads draw 0 of seed 20261700 with
`--perceived` off. The paired read on true inputs is TAG r2A2_100000_L30_tail_chk at 4946432. It
stows 2770 of 3000, with 232 on draw 0. If the control differs from 232 by more than 15 parts, the
true-input checked read runs again at the MS7c commit.

The seeds (R1, R2):

| use | seed | draws |
| --- | --- | --- |
| screen | 20261700 | draw 0 (250 parts) |
| sweep, isolation and position split rows | 20261700 | draws 0 and 1 (500 parts) |
| checked | 20261700 | draws 0 to 11 (3000 parts), a fresh TAG |
| official | 20262100 | draws 0 to 23 (6000 parts, 120 episodes), not used |
| hold calibration, process i | run seed 20267700 + i, layout seed 20268700 + i, i = 0 to 2 | one heap per process |
| corpus process p | run seed 20272000 + p, layout seed 20273000 + p, p = 0 to 39 | three heaps per process, never a read seed |
| close diagnostic | development seeds: run seed 7 with layout seed 8, and run seed 9 with layout seed 10 | 5 envs each, never a read seed |
| snap-clearance tuning | 20269900 | draws 0 to 3 (1000 parts), never a gate seed |

Every corpus seed entered `~/data/maker-arm-ms7/ledger.json` before the wave, as `ms7c_proc_0` to
`ms7c_proc_39`.

The rules that carry over (R10). A payload from the supervisor cache is not a read. Every read
records the SHA, the `policy_id`, the split, the serving switches and the knob scales. A
mid-training checkpoint gets its `action_space.json` marker before a read. The MS6 negatives stay
closed: no noise-scaled corpus, no round-1 DAgger frames, no 25-action horizon and no close gate.
The 17:08 entry adds one rule. A READ_FAIL, a read with no gate token, runs once more in a fresh TAG
with the same settings.

## 2. The hosts

The hosts keep their MS7a roles and the MS7a host rules. `scripts/fleet.py` and the host table
`~/data/maker-arm-ms7/hosts.json` stay.

| host | MS7c work |
| --- | --- |
| Machine A, RTX 5090 | The controller session and the ledger. The hold calibration on CPU. 8 lanes of checked reads 1, 1b and 2, and 4 lanes of candidate 3. The structure-only row and the scale-2 rows of the sweep. The snap-off screens. The position split rows c_z_xy1 and c_z_xy2. The clearance tuning reads c_tune_k0 and c_tune_k2. |
| Machine B, RTX 5080 | The control screen and the first structure-only row. 4 lanes of each checked read. The training arm and its screens. The position split row c_z_z2. The clearance tuning read c_tune_k4. |
| Machine C | The corpus wave and its conversion. The scale-4 rows of the sweep and the isolation rows, on CPU lanes. |

Four host events:

- Machine A sync to 0cfb61b and the sync of all hosts to eae617a used `--force`. Each came after a
  check that no process had its cwd in a campaign worktree.
- At 20:14, five sweep reads shipped the same checkpoint to Machine C at the same time. The
  post-copy file-list check of `fleet.py` failed on the colliding copies. The five reads ended
  READ_FAIL before any lane ran (section 8). The resume pointer now says to ship the checkpoint
  first, or to stagger the reads.
- At 22:12, Machine A and Machine B moved to a5482d6. Machine C kept eae617a until its pos=4
  row ended, and it then moved to a5482d6 for the isolation rows.
- At 02:05, every host moved to 8458f66 for the clearance tuning.

## 3. The perceived state

### 3.1 The module and the knobs

`core/perception.py` uses only torch and the standard library, like the rest of `core/`. A
`Perceiver` per env turns the truth into a `PerceivedState`. The state holds the detected parts and
the box poses. Each detection has a perceived centre, orientation, footprint size and class. An
undetected part is absent.

A detection keeps the simulator's part index in a private field, `sim_index`. Only the metrics and
the recorder read it. Within one selection, the selector sees it only as a slot. Nothing that lives
longer than a selection keys on it. The failure memory and the rake counter key on the perceived
position (section 5).

Each knob is a scale on a nominal value. A scale of 0 turns that model off, and 1 is the nominal.
Every nominal carries an `INITIAL:` marker, and MS5a measures it on real frames.

| knob | model | nominal | basis |
| --- | --- | --- | --- |
| `pos` | a calibration bias per episode, plus noise per detection | bias sigma 2 mm on each of x, y and z. Noise sigma 1.5 mm in x and y, 2 mm in z. | The iPhone wide camera covers the 0.9 m desk with 1920 px, about 0.5 mm per pixel. The tag-board extrinsics add the bias. |
| `heading` | noise per detection on the yaw about world z | sigma 4 degrees | the principal axis of a mask |
| `size` | noise per detection on the body extents | sigma 1 mm. The extent nearest world z gets twice this, so the half height carries 1 mm. | the mask extent and the depth |
| `box` | a bias per episode on each box pose | sigma 3 mm in x and y, 2 degrees in yaw | a box fit or tags on the boxes |
| `miss` | a part dropped at one perceive call | probability 0.05 | |
| `class` | a wrong class, drawn uniformly from the other three, at each classification: overhead at selection and in the wrist after a lift | probability 0.03 | the MS7b classifier gate of 0.97 |

The loop perceives at each target selection and on every step of a look-again wait (section 5). The
perceiver draws the box poses once per episode.

### 3.2 The structural rules

Two rules apply at every scale, zero included:

1. If another part covers the centre of a part from above, the perception does not detect that part.
   The rule tests the part centre against the footprint of every part whose centre is higher.
2. If the centre of a part lies within 40 mm in xy of the tool (`TOOL_OCCLUSION_RADIUS_M`), the
   perception does not detect that part. The arm hides it from the overhead camera. The radius
   carries an `INITIAL:` marker.

### 3.3 Seeding and paired draws

Each env's generator takes its seed from the run seed and the env index. At reset, the episode's
position bias comes first, then each box's bias. The per-detection draws follow in call order. So a
read with the same seeds and the same knobs repeats exactly.

The draws are paired across knobs. Every call draws the same values in the same order, whatever the
scales and whatever the structural rules remove. A scale only multiplies the drawn value. So two
sweep rows that differ in one knob see the same underlying noise.

### 3.4 The pos_z diagnostic knob

The sweep showed that position noise costs the most (section 8). The `pos` knob scales x, y and z
together, so it cannot tell the lateral error from the height error. a5482d6 on `feat/ms7c-pos-z`
(PR #118) adds the diagnostic knob `pos_z`. It multiplies the z axis of the position bias and
noise on top of `pos`. So `pos_z=0` reads the xy error alone, and `pos_z=2` doubles only the height
error.

`pos_z` multiplies the same draws, so every read at `pos_z=1` repeats the eae617a read exactly.
`all=` leaves it alone. The payload writes it only at values other than 1. So the recorded scales of
every earlier read stay the same. The CPU suite passes at a5482d6 with 1167 tests.

## 4. The hold signal

### 4.1 The detector

`core/hold.py` reads only proprioception. It uses the measured jaw gap, the gap that the previous
command asked for and the height of the FK pick point. On the real arm, the RS00 gripper motor
reports its position. The transmission from motor angle to jaw gap has no measured value yet. MS5a
measures it.

- Hold. Three conditions hold on 3 steps in a row (`CONFIRM_STEPS`). The measured gap is at least 2
  mm wider than the commanded gap (`MIN_EXCESS`). The measured gap is at least 3 mm (`MIN_GAP`). The
  measured gap spreads at most 0.2 mm over its last 3 readings (`STABLE`, `STABLE_STEPS`). The third
  condition keeps the closing ramp itself from reading as a hold.
- Exit. If the excess or the width condition fails on 2 steps in a row (`EXIT_STEPS`), the hold
  ends. So a carry that jiggles the gap keeps its hold.
- Lift. The FK pick point rises 5 cm (`LIFT_MIN_RISE`) above its height at the first step of the
  hold. The lift stays latched until the hold ends.
- Lost. A hold ends while it is lifted. This is a drop in the carry, or the release.

Each threshold carries an `INITIAL:` marker. The teacher closes to 6 mm under the part width, and
`--squeeze_floor` gives the policy the same close. So the design expected a real hold to stall about
6 mm wider than the command.

With `--perceived` off, the detector runs as a census only, and no decision reads it. Each census
row records `hold_detected`, `lift_detected`, `first_hold_at` and `first_held_at` beside the contact
truth. Each payload has a `hold_agreement` block over all its cycles.

### 4.2 The calibration

The bar (R5) came before the calibration. On the teacher's cycles of three calibration heaps,
precision and recall must be at least 0.95 for two events. The events are "held", a hold at least
once in the cycle, and "lifted", a lift at least once. The median offset of the first hold must be
at most 5 steps. The truth is the contact census. The rule allows a threshold change on these cycles
only, and the thresholds freeze before any read under `--perceived`.

At 16:04, a 12-part smoke of the teacher read hold and lift precision and recall of 1.0. But its
median first-hold offset read 6 steps. After the first contact, the soft contact still closes the
gap by 0.2 to 0.3 mm per step for about 4 steps. So the stability window opens late. The stall
excess read 2.3 to 5.0 mm, not the 6 mm of the design. The smoke is not a calibration heap.

At 16:06, while the calibration ran, the ledger added a sweep rule for a miss. Its first candidate
was `STABLE` 0.5 mm. After that came a grid over `STABLE`, `CONFIRM_STEPS` and `MIN_EXCESS`, and the
passing setting with the fewest changes won. The rule was not needed. The ledger first gave these
entries times from a wrong clock guess. A correction at 16:08 set them to the file and commit times,
and the order stayed the same.

The calibration ran at 9ff58a2 on Machine A CPU. The teacher drove with `--oracle_driver` on the
rung 2 lean scene, one heap per process, with `--hold_trace`. The results, over 142 cycles:

| event | true positives | false positives | false negatives | precision | recall |
| --- | --- | --- | --- | --- | --- |
| held | 140 | 1 | 0 | 0.993 | 1.000 |
| lifted | 138 | 0 | 2 | 1.000 | 0.986 |

The teacher stowed 47, 47 and 47 of 50 on the three heaps. The median first-hold offset is 5.0 steps
against the bar of 5. The payload census and the trace replay agree, and both print HOLD_OK.

The thresholds stay at the design values, frozen for every read under `--perceived`. They are
`MIN_EXCESS` 2 mm, `MIN_GAP` 3 mm, `STABLE` 0.2 mm over 3 readings, `CONFIRM_STEPS` 3 and
`EXIT_STEPS` 2. The offset has no margin.

Every read payload records the same agreement on the policy's cycles, as a diagnostic. The bar
applies to the teacher's cycles only, so these numbers change no rule. Over the 12 draws of each
checked read:

| read | hold precision | hold recall | lift precision | lift recall | median first-hold offset per draw |
| --- | --- | --- | --- | --- | --- |
| candidate 1b | 0.909 | 0.985 | 0.953 | 0.983 | 6 or 7 steps |
| candidate 2 | 0.918 | 0.989 | 0.955 | 0.981 | 6 or 7 steps |
| candidate 3 | 0.910 | 0.988 | 0.956 | 0.985 | 6 steps |

## 5. The loop under `--perceived`

The switch is off by default. Under it, three accessors are the only truth reads of the loop.
`perception_truth` is the input of `perceive()` at a selection. `wrist_class` gives the class of the
part between the jaws at a lift, the stand-in for the MS7b wrist classifier. `truth_reads` feeds the
grader, the tracker and the census.

A CPU test runs a perceived rollout twice. The second time, every old truth accessor raises or
returns noise. The test asserts identical decisions, and it caught three planted leaks.

| decision | on true inputs | under `--perceived` |
| --- | --- | --- |
| target selection and grasp ranking | the true poses of all parts | the detections only, with the perceived sizes and the closing axes of the perceived class. The tray walls are known geometry. |
| the candidate filter | true positions inside the true boxes | A detection inside a perceived box footprint is not a candidate. |
| the target row | the true centre and the ranked grasp at selection | the detection and its ranked grasp at selection, frozen for the cycle |
| the marker | the true live centre on every frame | the detected centre at selection, frozen for the cycle (`env.marker_points`) |
| the box row | the box of the true class, at the true box pose | the box of the class perceived overhead, at the perceived box pose |
| the class after a lift | `--adopt_held`: the identity of a lifted part that is not the target | At the detector's lift, the class of the part between the jaws passes through the class confusion. The box row follows the drawn class, for the target and for any other part (`lift_class_box`). |
| the stop flag of `--square_wrist`, `--pick_snap` and `--dwell_close` | contacts of at least 1 N on both jaws, or a true rise of 5 cm | the detector's hold or lift |
| the close gap of `--squeeze_floor` and `--dwell_close` | the true width from the row | the perceived width from the row |
| the cycle end | the true in-box test at bin entry | a completed release, `placed` over a perceived box footprint and `dropped` anywhere else, or the step budget |
| the out-of-reach end | the true resting position | removed. The step budget ends such a cycle. |
| the failure memory | the part index | the perceived position of the target |
| the rake | the true poses | the detections. The takeover reads the detector's hold and the perceived scene. |
| the episode end | `all_stowed` from the grader, or `no_candidate` | `no_candidate` among the detections after the look-again wait, or the cycle allowance |

More detail on these rules:

- A failed cycle, `dropped` or `budget`, books one failure at the target's perceived position
  (`PositionMemory`). At each selection, the nearest detection within 15 mm of a record takes over
  its count, and the record moves to that detection. The rake counter works the same way. The 15 mm
  radius carries an `INITIAL:` marker.
- A `placed` cycle books no failure.
- The carry grace of `--select_at_release` keeps a cycle open past its budget while the detector
  holds.
- A lift with no part between the jaws draws no class and keeps the row. The payload counts it as an
  empty lift.
- `--adopt_held` has no effect under `--perceived`. The payload records both switches.
- The env's own termination on success stays. It only ends a finished episode early, and it changes
  no stow. The loop then closes the open cycle as `stowed`.
- At the episode end, a `placed` cycle whose last contact-held part is stowed counts as a stow. This
  is a metric read, and no decision reads it.
- `--perceived` refuses the switches that read the truth: the DAgger takeover, `--scripted_place`
  and `--row_refresh`. `--contact_hold` is the fallback of R5.
- The payload's `perception` block counts the selections, the wrong overhead classes, the classified
  lifts, the wrong wrist classes, the empty lifts and the look-again waits.

### 5.1 The look-again rule

The design has no such rule (section 9). A selection can find neither a target nor a rake. Under
`--perceived`, such a selection ends the episode `no_candidate` in two cases only. Either the FK
tool point is more than 40 mm from the tray footprint at the rim, or the empty streak reaches 300
steps. That cap is the cap of the retreat wait. While the loop waits, the row and the marker stay,
the policy drives, and the loop perceives again on every step.

Without the rule, the tool occlusion rule ended an episode while the tool sat over the last parts
after a rake. A real rig moves the arm away and looks again before it calls the heap empty. The rule
is 3a79bae. On candidate 1b, the loop waited 60 times, for 1166 steps in total, and 54 waits found a
target or a rake.

### 5.2 The release end

The first perceived loop, at 0cfb61b, ended a cycle on the step that the detector's lifted hold
ended. Candidate 1 read under that rule.

At 17:31, the structure-only row showed 40 `dropped` cycles, and 25 of them had the census class
`carrying`. At the cycle end, the contacts still held the part. At 18:02, a trace on development
seeds (run seed 7, layout seed 8) read these cycles as a label issue. Each one ended on the step
that the policy commanded the jaws open at the release height, 0.096 to 0.100 m. The contacts held
the part 1 to 2 steps longer. If the FK pick point lay outside every perceived box footprint at that
step, the loop named the release `dropped`. The grader found 61 of the 174 such parts stowed at the
end. The ledger then planned no loop change.

At 18:07, the same trace showed the real defect. The structure-only row left 13 parts on the desk
outside every box on draws 0 and 1, and the true-input read left none there. The hold ended on the
first step of the release. The commanded gap was 67 to 90 mm, but the measured gap was 8 to 9 mm.
The loop then selected the next target and switched the row. `--replan_on_switch` served a new
chunk, and that chunk closed the jaws again at about 14 mm while the part was still pinched.

The true-input loop switches at the same point only after the measured opening reads 0.8
(`SELECT_OPEN_FRACTION` in `released_after_lift`). That is a proprioceptive read. The fix d51c495 on
`feat/ms7c-perceived-loop` adds `perceived_release`:

1. The end of a lifted hold starts a release, and the release keeps the box test of that step.
2. A new detected hold cancels the release.
3. The release completes at a measured opening of 0.8.

The corpus branch moved onto the fix as eae617a. The recorder does not use this rule, so the corpus
wave stands. Candidate 1b and every later read use the fix (section 7).

The fix has a known cost in wall time. At 20:49, a check of the candidate 1b payloads found 272
`lifted_lost` cycles that ended on the budget. All 272 ran the full 800 steps. For 244 of them, the
last-held part lies in its own box at the end, and for 28 it does not. In these cycles, the policy
released the part into the box. But it opened the jaws less than 0.8 of the full gap, 84 of 104.8
mm. So the perceived release never completed, and the cycle ran on to its budget. This costs wall
time, not stows or cycles, and the rule stays as it is for every read.

## 6. The corpus and the retrain

### 6.1 The recorder

The teacher still acts on the truth. Under `--perceived`, the rig perceives once per teacher
selection and records the result:

- The target row is the detection's centre and a grasp ranked on the perceived scene for that part
  alone. The rig takes the best clear grasp, else the top-ranked one. When no grasp ranks, it takes
  the teacher's grasp.
- The recorder burns the marker into the overhead frames at the perceived centre.
- The box row serves the perceived pose of the true class's box. The corpus holds no class
  confusion, because a confused box row teaches the policy to ignore the box row.
- For a part that the perception misses, the row falls back to the true centre plus the episode's
  bias, with the teacher's grasp.

The row and the marker hold until the next selection. They hold through HOVER's adoption of the
pre-selection, a REOPEN and an adoption of a held part. After such an adoption, only the box row
switches. The teacher picks the true part with the true grasp, so the policy can learn to correct
the noisy row from the images.

At 16:20, a 12-part smoke recorded 4140 frames with 12 selections and 0 fallbacks. It converted with
CONVERT_OK.

### 6.2 The wave

| item | value |
| --- | --- |
| commit | 85685be, the recorder branch on the perception branch |
| host | Machine C, in its own worktree `~/src/maker-arm-lab-ms7c` |
| scene | `--tray --desk_boxes 4 --tray_lean 30 --perceived`, nominal scales |
| processes, heaps | 40 processes of 3 heaps each, 120 train-split heaps |
| host check | 0 MiB GPU in use, no policy server or sim process, 869 GB free |
| start, end | 16:20, CORPUS_OK at 18:08 |
| correct stows | 5372 of 6000, 0.895 (Wilson 95 % 0.887 to 0.903) |
| cycles per stow | 1.084 |
| parts out of reach, rakes | 107, 693 |
| yield check | The lean wave r2L1 read 0.869. The difference is 0.026, inside the 0.03 rule, so the check passes. |
| conversion | on Machine C at eae617a: CONVERT_OK, 5294 segments, CHECK_OK on every video |
| transfer | to Machine B over the LAN. The copy matches, 89 files and 14389162153 bytes. The aarch64 copy stays as the second copy. |

The recorder code is the same at 85685be, 0cfb61b and eae617a. The heaps of this wave and of r2L1
differ, so the two yields are not paired.

### 6.3 The training

| item | value |
| --- | --- |
| run | train_P |
| host | Machine B, RTX 5080, 38.5 steps per second |
| start | r2 A2 100k |
| data | the perceived corpus alone (`c/dataset`, 5294 segments), with the MS7a arm A recipe (`bin/train_warm.sh`) |
| steps | 200k, a checkpoint every 50k |
| launch, end | 19:02, TRAIN_OK at 20:34 |
| loss | 0.138 at 26k, 0.124 at 101k, 0.103 at 176k |
| copies | every checkpoint from 50k to 200k is on Machine B and on Machine A |

The retrain is the one retrain that the design allows. After it, only serving changes remain inside
the time box (decision C3).

## 7. The reads

The policy id of r2 A2 100k is
`act:~/data/maker-arm-ms7/r2/train_A2/checkpoints/100000/pretrained_model+nas50+ctx16+gap`.
The train_P checkpoints read as
`act:~/data/maker-arm-ms7/c/train_P/checkpoints/<step>/pretrained_model+nas50+ctx16+gap`.

### 7.1 The control screen

The ledger pre-registered the control screen at 16:55. It read at 17:31 as c_ctrl_scr, at 0cfb61b on
Machine B, with `--perceived` off. It stows 233 of 250 at 1.133 cycles per correct stow. The paired read
gave 232 on the same draw. The difference is inside 15 parts, so the paired true-input read stays
r2A2_100000_L30_tail_chk. Its draws 0 and 1 stow 464 of 500 at 1.151.

### 7.2 The structure-only rows

Every knob is at 0 in these rows. So only the hold signal, the frozen marker, the perceived cycle
end and the two occlusion rules act.

| read | commit | stowed of 500 | Wilson 95 % | cycles per correct stow | placed | budget | dropped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c_struct | 0cfb61b | 441 (213 + 228), 0.882 | 0.851 to 0.907 | 1.215 | 423 | 73 | 40 |
| c_sw2_all0 | eae617a | 466 (233 + 233), 0.932 | 0.906 to 0.951 | 1.146 | 420 | 88 | 26 |
| true inputs, draws 0 and 1 of r2A2_100000_L30_tail_chk | 4946432 | 464 (232 + 232), 0.928 | 0.902 to 0.948 | 1.151 | | | |

At 0cfb61b, the structure cost 23 parts, 0.046. After the release fix, it costs nothing. On
c_struct, the look-again rule fired 8 times, and 5 waits found a part. The hold agreement on the
policy's cycles read hold precision 0.956 and 0.929 on the two draws. Lift precision read 0.969 and
0.983, and every recall read 0.99 or more.

The 18:10 entry names the second row c_struct_b. The 19:03 entry names it c_sw2_all0, inside the
sweep.

### 7.3 The retrain screens

The ledger pre-registered these screens at 20:14. Each train_P checkpoint went to Machine A as the
second copy. Each screen reads at eae617a under `--perceived` at nominal scales, snap on, on seed
20261700, draw 0, 250 parts. It uses one Machine B lane beside the training. The best screen by correct
stows goes to the checked read, and a tie goes to fewer cycles.

| checkpoint | TAG | stowed of 250 | cycles per correct stow | rakes | wrong-box parts |
| --- | --- | --- | --- | --- | --- |
| r2 A2 100k, draw 0 of candidate 1b | c_r2A2_100000_chk_b | 229 | 1.262 | 69 | 14 |
| train_P 50k | c_P_050000_scr | 220 | 1.250 | 50 | 16 |
| train_P 100k | c_P_100000_scr | 218 | 1.216 | 52 | 21 |
| train_P 150k | c_P_150000_scr | 229 | 1.349 | 46 | 7 |
| train_P 200k | c_P_200000_scr | 227 | 1.185 | 50 | 18 |

At 20:58, the rule picked train_P 150k, with the most correct stows. The 200k screen had the fewest
cycles per correct stow, but the rule reads stows alone.

### 7.4 The snap-off screens

At 20:15, the ledger pre-registered a serving test. `--pick_snap` moves the commanded pick point
onto the served row's centre on the final approach. Under `--perceived`, it drives the tool onto the
noisy perceived position. So it can undo what the retrain learns from the images, because the
teacher grasps the true part while its recorded row carries the noise.

A snap-off screen leaves out `--pick_snap` and keeps every other switch (`bin/read_c2.sh SNAP=0`).
It reads at nominal scales on seed 20261700, draw 0, on one Machine A lane. The rule: the retrain's
checked read serves snap-on, unless snap-off reads at least 5 more correct stows than snap-on on the
best retrain checkpoint's screen. If snap-off wins, the checked read of that checkpoint runs
snap-off and says so.

| checkpoint | snap-on stowed of 250 | snap-off TAG | snap-off stowed of 250 | snap-off cycles per correct stow | snap-off `close_miss` cycles | snap-off minus snap-on |
| --- | --- | --- | --- | --- | --- | --- |
| r2 A2 100k | 229 | c_r2A2_100000_nosnap_scr | 138 | 2.355 | 134 | -91 |
| train_P 50k | 220 | c_P_050000_nosnap_scr | 149 | 2.161 | 82 | -71 |
| train_P 100k | 218 | c_P_100000_nosnap_scr | 199 | 1.633 | 81 | -19 |
| train_P 150k | 229 | c_P_150000_nosnap_scr | 161 | 2.019 | 105 | -68 |
| train_P 200k | 227 | c_P_200000_nosnap_scr | 140 | 2.321 | 148 | -87 |

The snap stays on (21:24). `close_miss` is the largest failure class on every snap-off screen. The
per-axis row error at nominal scales is about 2.5 mm. Even so, the snap onto the noisy row centre
beats the policy's own final approach by far. The retrain did not learn to replace the snap.

### 7.5 The checked reads

The ledger pre-registered candidate 1 at 17:08, candidate 1b at 18:10, candidate 2 at 21:24 and
candidate 3 at 22:13. Each reads at nominal scales with the R4 protocol, snap on, on seed 20261700,
draws 0 to 11.

Candidate 3 is a declared deviation, because the design names two candidates (section 9). The cycles
bar binds on both earlier checked reads. The 200k screen had the fewest cycles per correct stow, and
the screen rule picked 150k on stows alone. A pass of candidate 3 still needed the official read on
the fresh seed, which guards against a selection on the checked seed. If candidates 2 and 3 had both
passed, the better checked read took the official read. Candidates 1 and 1b are the same checkpoint.
If both passed, the 18:10 rule gave the official read to 1b.

| item | true inputs | candidate 1 | candidate 1b | candidate 2 | candidate 3 |
| --- | --- | --- | --- | --- | --- |
| checkpoint | r2 A2 100k | r2 A2 100k | r2 A2 100k | train_P 150k | train_P 200k |
| TAG | r2A2_100000_L30_tail_chk | c_r2A2_100000_chk | c_r2A2_100000_chk_b | c_P_150000_chk | c_P_200000_chk |
| commit | 4946432 | 0cfb61b | eae617a | eae617a | a5482d6, `pos_z=1` |
| read at | MS7a | 17:58 | 19:02 | 22:12 | 00:01 |
| stowed of 3000 | 2770, 0.923 | 2527, 0.842 | 2603, 0.868 | 2628, 0.876 | 2646, 0.882 |
| Wilson 95 % | 0.913 to 0.932 | 0.829 to 0.855 | 0.855 to 0.879 | 0.864 to 0.887 | 0.870 to 0.893 |
| cycles | 3212 | 3665 | 3681 | 3523 | 3517 |
| cycles per correct stow | 1.160 | 1.450 | 1.414 | 1.341 | 1.329 |
| cycles over the bar at this stow count | | | 298 | 107 | 78 |
| lanes | | Machine A 8, Machine B 4 | Machine A 8, Machine B 4 | Machine A 8, Machine B 4 | Machine A 4, Machine B 4 |
| Machine A draws, Machine B draws | | 1670 (0 to 7), 857 (8 to 11) | 1733 (0 to 7), 870 (8 to 11) | 1738 (0 to 7), 890 (8 to 11) | 1317 (0 to 5), 1329 (6 to 11) |
| rakes | 580 | 446 | 463 | 500 | 479 |
| wrong-box parts | 75 | 179 | 147 | 201 | 189 |
| episodes that end on the cycle allowance | 1 of 60 | 28 of 60 | 27 of 60 | 12 of 60 | 15 of 60 |
| outcome | above both bars | GATE_FAIL, both bars missed | GATE_FAIL, the cycles bar missed | GATE_FAIL, the cycles bar missed | GATE_FAIL, the cycles bar missed |

The pos_z knob multiplies the same draws, so candidate 3 at `pos_z=1` reads the same perception
draws as eae617a.

Per draw, 250 parts each:

| read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true inputs | 232 | 232 | 229 | 241 | 228 | 230 | 219 | 230 | 232 | 238 | 236 | 223 |
| candidate 1 | 217 | 211 | 212 | 227 | 200 | 196 | 190 | 217 | 209 | 212 | 225 | 211 |
| candidate 1b | 229 | 220 | 220 | 227 | 198 | 204 | 217 | 218 | 218 | 202 | 225 | 225 |
| candidate 2 | 225 | 215 | 210 | 228 | 214 | 207 | 219 | 220 | 221 | 223 | 221 | 225 |
| candidate 3 | 229 | 221 | 224 | 229 | 210 | 204 | 217 | 220 | 227 | 213 | 226 | 226 |

Per class, as stowed parts and fractions:

| class | parts | true inputs | candidate 1 | candidate 1b | candidate 2 | candidate 3 |
| --- | --- | --- | --- | --- | --- | --- |
| slope | 910 | 780 (0.857) | 681 (0.748) | 718 (0.789) | 724 (0.796) | 733 (0.805) |
| tall | 900 | 853 (0.948) | 797 (0.886) | 817 (0.908) | 815 (0.906) | 815 (0.906) |
| narrow | 415 | 388 (0.935) | 353 (0.851) | 360 (0.867) | 362 (0.872) | 380 (0.916) |
| wide | 775 | 749 (0.966) | 696 (0.898) | 708 (0.914) | 727 (0.938) | 718 (0.926) |

The cycle ends. On true inputs, a cycle ends `stowed`, `released`, `out_of_reach` or `budget`. Under
`--perceived`, it ends `placed`, `dropped` or `budget`. The one `stowed` end of candidates 2 and 3
is an episode that the env's own success ended with every part in its box.

| read | stowed | released | out_of_reach | placed | dropped | budget |
| --- | --- | --- | --- | --- | --- | --- |
| true inputs | 2743 | 22 | 9 | | | 438 |
| candidate 1 | | | | 2523 | 336 | 806 |
| candidate 1b | | | | 2356 | 184 | 1141 |
| candidate 2 | 1 | | | 2594 | 75 | 853 |
| candidate 3 | 1 | | | 2570 | 72 | 874 |

The census of all cycles:

| read | cycles | stowed | close_miss | held_no_lift | lifted_lost | carrying | wrong_box | no_close | no_open |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true inputs | 3212 | 2743 | 264 | 128 | 27 | 24 | 1 | 21 | 4 |
| candidate 1 | 3665 | 2350 | 562 | 326 | 109 | 262 | 35 | 18 | 3 |
| candidate 1b | 3681 | 2244 | 525 | 366 | 405 | 45 | 77 | 17 | 2 |
| candidate 2 | 3523 | 2439 | 467 | 273 | 180 | 37 | 94 | 32 | 1 |
| candidate 3 | 3517 | 2416 | 524 | 253 | 178 | 40 | 87 | 19 | 0 |

The census of the budget cycles only:

| read | budget cycles | close_miss | held_no_lift | lifted_lost | carrying | other |
| --- | --- | --- | --- | --- | --- | --- |
| true inputs | 438 | 260 | 126 | 6 | 24 | 22 |
| candidate 1 | 806 | 459 | 289 | 33 | 3 | 22 |
| candidate 1b | 1141 | 458 | 336 | 272 | 45 | 30 |
| candidate 2 | 853 | 403 | 254 | 114 | 37 | 45 |
| candidate 3 | 874 | 460 | 233 | 107 | 40 | 34 |

The census counts cycles, and the grader counts parts at rest at the episode end, so the two stowed
counts differ. The perception counts:

| count | candidate 1 | candidate 1b | candidate 2 | candidate 3 |
| --- | --- | --- | --- | --- |
| selections | 3665 | 3681 | 3523 | 3517 |
| wrong overhead class | 153 (0.042) | 143 (0.039) | 156 (0.044) | 134 (0.038) |
| lifts classified | 2691 | 2717 | 2686 | 2650 |
| wrong wrist class | 92 (0.034) | 64 (0.024) | 86 (0.032) | 69 (0.026) |
| empty lifts | 159 | 165 | 156 | 163 |
| look-again waits, steps, waits that found a target or a rake | 50, 1193, 45 | 60, 1166, 54 | 51, 1790, 46 | 50, 959, 45 |

The overhead class is wrong on 0.038 to 0.044 of the selections, above the nominal 0.03. The ledger
gives no reason.

Candidate 1 misses both bars. Against the paired read on the same draws, the perceived state costs
243 parts, 0.081. The structure-only row puts 0.046 of that on the structure. The 262 `carrying`
cycles are the release defect of section 5 at scale. The 18:02 entry lists where the 243 parts went:

- Wrong-box parts rose from 75 to 179. The wrist class draw was wrong on 92 of 2691 lifts, 0.034,
  against the nominal 0.03.
- Parts left in the tray rose from 136 to 205.
- The `close_miss` budget cycles rose from 260 to 459, and the `held_no_lift` budget cycles from 126
  to 289.
- 28 of 60 episodes ended on the cycle allowance, against 1.

Candidate 1b meets the stow bar with 53 parts to spare, but it uses 1.414 cycles per correct stow.
The release fix is worth 76 parts over candidate 1. The perceived state still costs 167 parts
against the paired read. The budget cycles rose from 438 to 1141. `close_miss` and `held_no_lift`
make 794 of them, against 386 on true inputs. Most of the 272 `lifted_lost` budget cycles are the
slow releases of section 5, which cost wall time only. For 244 of them, the part lies in its own box
at the end. Wrong-box parts are 147, against 75.

Candidate 2 gains 25 stows over candidate 1b and uses 158 fewer cycles. The 22:12 entry counts the
cycles that do not end `placed`. Against candidate 1b, `close_miss` falls from 492 to 425,
`held_no_lift` from 365 to 270 and `lifted_lost` from 392 to 147. Wrong-box parts rise to 201. Only
12 episodes end on the cycle allowance, against 27. It misses the cycles bar by 107 cycles.

Candidate 3 is the best checked read of MS7c: 2646 correct stows at 1.329 cycles per correct stow.
It misses the cycles bar by 78 cycles. Of its cycles that do not end `placed`, 473 are `close_miss`,
249 `held_no_lift` and 150 `lifted_lost`. Wrong-box parts are 189, against 75 on true inputs.

### 7.6 The gate outcome

The 00:01 entry records the outcome. No checked read passes. The three candidates after the release
fix, 1b, 2 and 3, meet the stow bar with 0.868, 0.876 and 0.882. All three miss the cycles bar with
1.414, 1.341 and 1.329. By the gate of design section 8, the milestone fails, and the official read
does not run. The official seed 20262100 stays unused. The detector passed its bar, so no `PASS
(contact hold)` outcome applies either.

The checked seed is also the selection seed. So the checked reads filter only, and a pass needed the
one-shot official read on a fresh seed.

The clearance tuning of section 8 adopts no lever, so no candidate 4 exists. At 02:53, the ledger
records the final gate outcome. No checked read passes, and the milestone fails its gate at nominal
noise on the cycles bar.

## 8. The decomposition

### 8.1 The sweep

The ledger pre-registered the sweep at 19:03, at eae617a, with the TAG prefix `c_sw2_`. Every row
reads r2 A2 100k on seed 20261700, draws 0 and 1, 500 parts, on 2 lanes. Machine A reads the
structure-only row and the six scale-2 rows, at most 4 at once. Machine C reads the six
scale-4 rows at once, on 12 CPU lanes. The nominal row is candidate 1b's draws 0 and 1. The
true-input row is draws 0 and 1 of the paired read. Neither row needs a new read. The sweep was
complete at 23:32.

| row | TAG | scales | stowed of 500 | Wilson 95 % | cycles per correct stow |
| --- | --- | --- | --- | --- | --- |
| true inputs | r2A2_100000_L30_tail_chk, draws 0 and 1 | `--perceived` off | 464, 0.928 | 0.902 to 0.948 | 1.151 |
| structure only | c_sw2_all0 | every scale 0 | 466, 0.932 | 0.906 to 0.951 | 1.146 |
| nominal | c_r2A2_100000_chk_b, draws 0 and 1 | every scale 1 | 449, 0.898 | 0.868 to 0.922 | 1.292 |
| pos at 2 | c_sw2_pos2 | pos=2 | 354, 0.708 | 0.667 to 0.746 | 1.836 |
| pos at 4 | c_sw2_pos4 | pos=4 | 256, 0.512 | 0.468 to 0.556 | 2.539 |
| heading at 2 | c_sw2_heading2 | heading=2 | 430, 0.860 | 0.827 to 0.888 | 1.407 |
| heading at 4 | c_sw2_heading4 | heading=4 | 419, 0.838 | 0.803 to 0.868 | 1.487 |
| size at 2 | c_sw2_size2 | size=2 | 428, 0.856 | 0.823 to 0.884 | 1.437 |
| size at 4 | c_sw2_size4 | size=4 | 446, 0.892 | 0.862 to 0.916 | 1.374 |
| box at 2 | c_sw2_box2 | box=2 | 449, 0.898 | 0.868 to 0.922 | 1.365 |
| box at 4 | c_sw2_box4 | box=4 | 414, 0.828 | 0.792 to 0.859 | 1.541 |
| miss at 2 | c_sw2_miss2 | miss=2 | 458, 0.916 | 0.888 to 0.937 | 1.290 |
| miss at 4 | c_sw2_miss4 | miss=4 | 439, 0.878 | 0.846 to 0.904 | 1.387 |
| class at 2 | c_sw2_class2 | class=2 | 420, 0.840 | 0.805 to 0.870 | 1.386 |
| class at 4 | c_sw2_class4 | class=4 | 411, 0.822 | 0.786 to 0.853 | 1.484 |

Position noise costs the most. At pos=2, all 10 episodes end on the cycle allowance, and at pos=4
the loop runs no rake. At class=4, wrong-box parts rise to 63. The size row reads higher at scale 4
than at scale 2, and the two intervals overlap. The miss row at scale 2 reads 9 parts above the
nominal row, and the intervals overlap too.

At 20:14, five scale-4 rows ended READ_FAIL before any lane ran (section 2). They are pos4,
heading4, size4, box4 and class4. The failed logs moved to `reads/_failed_ship`. The five rows ran
again under the same TAGs, with the same settings, at eae617a. c_sw2_miss4 got through the ship on
the first try.

### 8.2 The isolation rows

The sweep changes one knob with the others at 1, so it cannot say how the knobs share the cycles. At
23:32, the ledger pre-registered six isolation rows to split that budget. Each row reads r2 A2 100k
on draws 0 and 1 with one knob at 1 and every other knob at 0 (`all=0,<knob>=1`). The rows ran at
a5482d6 on aarch64 CPU lanes, 2 lanes each, TAG `c_iso_<knob>`. They are descriptive, not gate
reads.

| row | TAG | stowed of 500 | Wilson 95 % | stows against structure only | cycles per correct stow | cycles against structure only |
| --- | --- | --- | --- | --- | --- | --- |
| structure only | c_sw2_all0 | 466, 0.932 | 0.906 to 0.951 | | 1.146 | |
| pos alone | c_iso_pos | 459, 0.918 | 0.891 to 0.939 | -7 | 1.292 | +0.146 |
| heading alone | c_iso_heading | 448, 0.896 | 0.866 to 0.920 | -18 | 1.161 | +0.015 |
| size alone | c_iso_size | 454, 0.908 | 0.879 to 0.930 | -12 | 1.227 | +0.081 |
| box alone | c_iso_box | 453, 0.906 | 0.877 to 0.929 | -13 | 1.216 | +0.070 |
| miss alone | c_iso_miss | 450, 0.900 | 0.871 to 0.923 | -16 | 1.171 | +0.025 |
| class alone | c_iso_class | 446, 0.892 | 0.862 to 0.916 | -20 | 1.224 | +0.078 |
| sum of the six changes | | | | -86 | | +0.415 |
| all six at 1 (nominal) | c_r2A2_100000_chk_b, draws 0 and 1 | 449, 0.898 | 0.868 to 0.922 | -17 | 1.292 | +0.146 |

Position noise alone takes the cycles from 1.146 to 1.292, the whole margin to the bar of 1.3. The
other knobs add 0.015 to 0.081 each. The effects do not add: all six at 1 read 1.292 on these draws,
the same as position alone. On all 12 draws, the nominal reads 1.414. Each change in stows lies
inside the interval of a 500-part read. The class row puts 27 parts in a wrong box, against 15 with
every knob at 0.

### 8.3 The position split

At 22:12, the ledger pre-registered the position split with the `pos_z` knob of section 3. It is a
diagnostic, not a gate read. Each row reads r2 A2 100k on draws 0 and 1 at a5482d6, with the other
knobs at 1.

| row | TAG | scales | xy scale | z scale | stowed of 500 | Wilson 95 % | cycles per correct stow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| structure only | c_sw2_all0 | every scale 0 | 0 | 0 | 466, 0.932 | 0.906 to 0.951 | 1.146 |
| xy only | c_z_xy1 | pos_z=0 | 1 | 0 | 446, 0.892 | 0.862 to 0.916 | 1.262 |
| nominal | c_r2A2_100000_chk_b, draws 0 and 1 | every scale 1 | 1 | 1 | 449, 0.898 | 0.868 to 0.922 | 1.292 |
| z doubled | c_z_z2 | pos_z=2 | 1 | 2 | 437, 0.874 | 0.842 to 0.900 | 1.442 |
| xy doubled, no z | c_z_xy2 | pos=2,pos_z=0 | 2 | 0 | 380, 0.760 | 0.721 to 0.795 | 1.663 |
| both doubled | c_sw2_pos2 | pos=2 | 2 | 2 | 354, 0.708 | 0.667 to 0.746 | 1.836 |

The pre-registered rule has two branches. If c_z_xy1 reads 460 or more and c_z_xy2 reads 420 or
more, the height error is the driver. Then a serving change for the grasp height is worth building.
If c_z_xy2 reads under 400, the lateral error is the driver, and no serving change on the height
follows.

c_z_xy2 reads 380, under 400. So the lateral error is the driver (23:30). The height error costs
cycles more than stows. Doubling it costs 12 stows and adds 0.150 cycles per correct stow. Removing
it costs 3 stows and saves 0.030.

A floor-prior switch was written while the split ran: `feat/ms7c-floor-prior`, 9588748, off by
default. Under it, a detection that reads near the floor takes the floor plane plus half its
vertical extent as its z. The split rule does not allow a height change, so the switch is not used,
not read, not pushed and not shipped.

### 8.4 The requirement table

For each knob, the table gives the largest scale at which the stowed fraction stays at 0.85 or more
(R7). It also gives the largest scale at which the cycles stay at 1.3 per correct stow or fewer.
Both columns interpolate linearly between the measured scales. A knob whose scale-2 read is under
0.85 is tight. A 500-part read has a Wilson 95 % interval of about plus or minus 0.03.

Each cell gives k/n, the fraction, the Wilson 95 % interval and the cycles per correct stow. The
scale-1 cell of every knob is the nominal row, 449/500 = 0.898 [0.868, 0.922], 1.292.

| knob | nominal | scale 2 | scale 4 | largest scale at 0.85 stowed | in physical units | largest scale at 1.3 cycles | in physical units | tight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pos` | bias sigma 2 mm per axis, noise sigma 1.5 mm in xy and 2 mm in z | 354/500 = 0.708 [0.667, 0.746], 1.836 | 256/500 = 0.512 [0.468, 0.556], 2.539 | 1.25 | bias sigma 2.5 mm per axis, noise sigma 1.9 mm in xy and 2.5 mm in z | 1.02 | bias sigma 2.0 mm per axis, noise sigma 1.5 mm in xy and 2.0 mm in z | yes |
| `heading` | sigma 4 degrees | 430/500 = 0.860 [0.827, 0.888], 1.407 | 419/500 = 0.838 [0.803, 0.868], 1.487 | 2.91 | sigma 11.6 degrees | 1.07 | sigma 4.3 degrees | no |
| `size` | sigma 1 mm | 428/500 = 0.856 [0.823, 0.884], 1.437 | 446/500 = 0.892 [0.862, 0.916], 1.374 | 4 or more | sigma 4 mm or more | 1.06 | sigma 1.06 mm | no |
| `box` | sigma 3 mm and 2 degrees per box | 449/500 = 0.898 [0.868, 0.922], 1.365 | 414/500 = 0.828 [0.792, 0.859], 1.541 | 3.37 | sigma 10.1 mm and 6.7 degrees | 1.11 | sigma 3.3 mm and 2.2 degrees | no |
| `miss` | probability 0.05 per perceive call | 458/500 = 0.916 [0.888, 0.937], 1.290 | 439/500 = 0.878 [0.846, 0.904], 1.387 | 4 or more | probability 0.20 or more | 2.20 | probability 0.11 | no |
| `class` | probability 0.03 per classification | 420/500 = 0.840 [0.805, 0.870], 1.386 | 411/500 = 0.822 [0.786, 0.853], 1.484 | 1.83 | probability 0.055 | 1.09 | probability 0.033 | yes |

Notes on the table:

- The cycles bar is shared by the knobs, not set by one. The nominal row sits at 1.292 on these
  draws, only 0.008 under the bar. So every knob except `miss` reaches 1.3 between scales 1.02 and
  1.11. The isolation rows show that position alone uses the whole margin (section 8).
- The `pos` knob scales the bias and the per-detection noise together, so each scale applies to
  both. The `box` knob scales the position and the yaw of the box bias together. Since the 01:20
  correction, `req_table.py` prints these scaled values.
- The `heading` and `size` reads at scale 2 are under 0.88, and their intervals reach under 0.85. By
  the stow rule, they are not tight.
- The table describes r2 A2 100k. The design reads the tight knobs again on a retrain that wins the
  gate. No retrain won, so that re-read does not run.

The tight knobs for MS5a are the position and the class. The first MS5a task measures them on real
frames. On the cycles bar, a real rig at the nominal values has almost no margin.

### 8.5 Where the cycles go

The 00:01 entry names two costs:

1. The lateral position error. `--pick_snap` turns it into the grasp point. Snap-off loses 19 to 91
   parts per 250 (section 7), so the snap stays. The position split puts the cost on xy, not z.
2. The class errors. Wrong-box parts are 189 on candidate 3, against 75 on true inputs.

A serving change cannot remove either cost without a new perception input. Two such inputs are a
wrist-camera position fix at the final approach, and a calibration that removes the per-episode
bias. Both are outside the MS7c model.

At 01:12, the ledger pre-registered one more diagnostic, on development seeds, not a read seed. For
each failed cycle, it finds where each close lands against the row and against the true part. It
decides whether a search pattern on the snap's retries is worth building. A search pattern is a
real-rig strategy under position uncertainty. The rule came before its numbers. The search pattern
is built only on one condition. At least half of the `close_miss` cycles must close two or more
times with the tool within 2 mm of the row centre. In those closes, the true centre must lie 2 mm or
more from the row centre, in the part's across-jaw direction. A retry at the same point cannot
succeed there.

The diagnostic ran on train_P 200k with an observation-only close trace (`diag/ms7c-close-trace`,
77d51e2). It used two sets of development seeds, 5 envs each: run seed 7 with layout seed 8, and run
seed 9 with layout seed 10. The policy stowed 462 of 500 at 1.251 cycles per correct stow.

The rule does not hold (01:59). Only 18 of 71 `close_miss` cycles meet it, 0.25 against the bar of
0.50. So no search pattern is built. The trace shows three other facts:

- The failing closes land high. The pick point sits 4 mm above the true centre at the first close
  and 10 mm above it at later closes. Stowed first closes sit 0.6 mm below it.
- On 35 % of the `close_miss` first closes, the perception error along the jaw closing axis is over
  4 mm. On the stowed first closes, the share is 9 %. The jaws descend at the teacher's landing gap,
  w + 8 mm, which is 4 mm per side. So such an error lands a finger on the part, and the tool closes
  high. The opening clears the part by a median of -5.4 mm in `close_miss` cycles, against -0.1 mm
  in stowed ones.
- The target moves a median of 3 mm between retries, while the row stays frozen.

### 8.6 The snap clearance

The trace led to a serving lever, pre-registered at 01:59. `--snap_clearance K` widens the landing
gap that `--pick_snap` holds on the descent by K mm per side, to w + 8 + 2K mm. The gap never
exceeds the fully open jaws. K = 0 keeps today's gap exactly. The real rig can command this gap. The
lever is 8458f66 on `feat/ms7c-snap-clearance`, and the CPU suite passes there with 1170 tests.
Every host moved to 8458f66 at 02:05.

The tuning reads train_P 200k at K = 0, 2 and 4 on seed 20269900, draws 0 to 3, 1000 parts, on 4
lanes each. No gate read uses that seed. The rule has two conditions. The better K of 2 and 4, by
cycles per correct stow, must beat K = 0 by 0.04 or more. It must also lose at most 10 correct stows
against K = 0. If both conditions hold, that K is adopted, and candidate 4 is train_P 200k with that
K. Candidate 4 then gets a checked read on seed 20261700, draws 0 to 11, in a fresh TAG, with the
same bars.

| K | TAG | landing gap | stowed of 1000 | Wilson 95 % | cycles per correct stow | per draw | close_miss | held_no_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | c_tune_k0 | w + 8 mm | 892, 0.892 | 0.871 to 0.910 | 1.275 | 223, 224, 227, 218 | 124 | 59 |
| 2 | c_tune_k2 | w + 12 mm | 871, 0.871 | 0.849 to 0.890 | 1.333 | 223, 208, 227, 213 | 149 | 81 |
| 4 | c_tune_k4 | w + 16 mm | 904, 0.904 | 0.884 to 0.921 | 1.284 | 227, 226, 224, 227 | 133 | 81 |

The `close_miss` and `held_no_lift` columns count the cycles that do not end `placed`, as the 22:12
entry does.

Neither K beats K = 0 on cycles per correct stow (02:53). So nothing is adopted, and no candidate 4
exists. The branch stays pushed as the record of these reads, with no PR.

## 9. Deviations from the design

| deviation | reason |
| --- | --- |
| The campaign starts on 2026-09-24 at 15:46. The design schedules it from 2026-09-25 to 2026-09-27. | The ledger gives no reason. |
| The official read uses seed 20262100, draws 0 to 23, not seed 20261900 (R2). | The MS7a vertical-wall read of TAG r2_vertical_confirm (15:05) already read seed 20261900 with r2 A2 100k. No gate read used 20262100 before. MS6 j used 20262000 + n as run seeds, and a gate seed maps to the run seeds seed * 1000 + i. |
| The MuJoCo hooks `marker_points` and `perceived_box_poses` (01172b6) land on `feat/ms7c-perception`. The design puts the frozen marker on `feat/ms7c-perceived-loop`. | The recorder branch starts on top of the perception branch, beside the loop branch. The recorder and the loop both use the hooks. |
| The hold calibration runs on Machine A CPU, not on Machine C. | The ledger gives no reason. |
| The corpus wave records at 85685be, the recorder branch on the perception branch, before the stack rebase. | The recorder code is the same after the rebase. The rebase adds only the loop branch, which does not touch the teacher's own paths. |
| The look-again rule (3a79bae). A selection can find neither a target nor a rake. Such a selection ends the episode in two cases only: the tool is clear of the tray, or the empty streak reaches 300 steps. | Without it, the tool occlusion rule ended an episode while the tool sat over the last parts after a rake. A real rig moves the arm away and looks again. |
| Candidate 1 reads the first release end of 0cfb61b. A cycle ends on the step that the detector's lifted hold ends. | This is the first stage of the release end. At 18:07, a trace found that the row switched while the part was still pinched (section 5). |
| The release fix d51c495, restacked as eae617a on the corpus branch. A release completes at a measured opening of 0.8. | This is the second stage. It uses the proprioceptive read of the true-input loop's `released_after_lift`. |
| A second checked read of r2 A2 100k, candidate 1b, on the same draws. If candidates 1 and 1b both passed, 1b took the official read. | The release fix changes the serving loop. The design names one checked read per candidate. |
| The 18:02 sweep at 0cfb61b stopped at 18:07 before any row finished. None of its rows is a read. The sweep ran again at eae617a with a new structure-only row. | The release defect. The stopped rows lie in `reads/_stopped_sweep_0cfb61b`. |
| Five scale-4 sweep rows ran again under the same TAGs, not in fresh TAGs. | They ended READ_FAIL at 20:14 on a failed checkpoint ship, before any lane ran. The failed logs moved to `reads/_failed_ship`. |
| Snap-off screens leave out `--pick_snap`. The rule: if snap-off reads at least 5 more correct stows than snap-on on the best retrain screen, the retrain's checked read runs snap-off. | `--pick_snap` drives the tool onto the perceived, noisy row centre, so it can undo what the retrain learns from the images. The snap stayed on, because snap-off lost 19 to 91 parts. |
| Candidate 3, train_P 200k, is a third checked read. The design names two candidates. | The cycles bar binds on both earlier checked reads. The 200k screen had the fewest cycles per correct stow, and the screen rule picked 150k on stows alone. A pass still needed the official read on a fresh seed. |
| The diagnostic knob `pos_z` (a5482d6) and the position split. | The sweep marked position as tight, but the `pos` knob cannot split the lateral error from the height error. |
| Six isolation rows, one knob at 1 and the rest at 0. The design has no such rows. | The sweep showed that every knob reaches 1.3 cycles near scale 1. The isolation rows split that shared budget for the report. |
| The floor-prior switch (9588748) was written during the position split. It is not used and not shipped. | The pre-registered split rule found the lateral error to be the driver, so no height change follows. |
| The close diagnostic on development seeds, with an observation-only close trace (`diag/ms7c-close-trace`, 77d51e2). The branch is local only, not pushed and not shipped. | The isolation rows put the cycles on the position error. The diagnostic tests whether a search pattern on the snap's retries is worth building. Its rule rejects the search pattern. |
| The snap-clearance lever `--snap_clearance K` (`feat/ms7c-snap-clearance`, 8458f66), tuned on seed 20269900. It is not adopted. The branch stays pushed with no PR, as the record of the tuning reads. | The close diagnostic found fingers that land on the part. On 35 % of the failed first closes, the error along the closing axis exceeds the landing clearance of 4 mm per side. |

## 10. The code

The code is one stack of one-concern branches on main 8272d9e. Every new switch is off by default,
so the MS7a protocol stays reproducible. Every script keeps the token contract. The stack, bottom to
top:

| order | branch | tip | what it adds | PR |
| --- | --- | --- | --- | --- |
| 1 | `feat/ms7c-perception` | 01172b6 | `core/perception.py` (75828c1): the `PerceivedState`, the `Perceiver`, the six knobs, the two structural rules and `PositionMemory`. The MuJoCo hooks `marker_points` and `perceived_box_poses` (01172b6). | #113 |
| 2 | `feat/ms7c-hold-signal` | 9ff58a2 | `core/hold.py`, the detector and its agreement (505738c). The hold census in the loop, `--hold_trace` and `scripts/hold_agreement.py` (9ff58a2). | #114 |
| 3 | `feat/ms7c-perceived-loop` | b099d4b | The perceived rake takeover in the rig (7fe0a2b). `--perceived` in the loop (7794875). The client and gate flags and the cache key (86a48ea). The look-again rule (3a79bae). The release fix (d51c495). A test fix for CI (b099d4b). | #115 |
| 4 | `feat/ms7c-perceived-corpus` | cbf1b77 | the recorder under `--perceived`, in `OracleRig` and `scripts/mj_generate_batch.py` | #116 |
| 5 | `feat/ms7c-pos-z` | a13f6e7 | the diagnostic knob `pos_z` in the perception, the client and the gate | #118 |
| 6 | `docs/ms7c-perceived-state` | | this report | #119 |

PRs #113 to #116, #118 and #119 form GitHub stack #117. The reads ran at eae617a, a5482d6 and
8458f66. After them, the test-only commit b099d4b went into branch 3, because CI has no EGL and one
usage-token test asked for it. Branches 4 and 5 were rebased onto it, so their tips are now cbf1b77
and a13f6e7, with the same code as eae617a and a5482d6. The full CPU suite passes at the top of the
stack with 1161 tests and 6 skipped, under `MUJOCO_GL=disable` as in CI. `run_gate_eval.py` forwards
`--perceived`, `--noise_scales` and `--contact_hold`, and its cache refuses a payload read with
other values.

Three branches stay off the stack. `feat/ms7c-floor-prior` (9588748) is local only and not pushed.
`diag/ms7c-close-trace` (77d51e2), the observation-only close trace, is local only.
`feat/ms7c-snap-clearance` (8458f66, on a5482d6, now b09a611 on a13f6e7) is pushed with no PR, as
the record of the tuning reads. The CPU suite passed at 8458f66 with 1170 tests.

## 11. What is still open

- The hold offset has no margin. It reads 5.0 steps on the teacher against a bar of 5, and 6 or 7
  steps on the policy's cycles.
- The overhead class is wrong on 0.038 to 0.044 of the selections, against the nominal 0.03. The
  ledger gives no reason.
- A slow release runs its cycle to the budget (section 5). It costs wall time, not stows or cycles.
- The real arm. The gripper transmission from motor angle to jaw gap has no measured value. Every
  knob value is an `INITIAL:` estimate. MS5a measures the tight knobs on real frames.

## 12. Conclusion

MS7c fails its gate at nominal noise on the cycles bar. Every candidate after the release fix meets
the stow bar, with 0.868, 0.876 and 0.882. Each one misses 1.3 cycles per correct stow, with 1.414,
1.341 and 1.329. The official read does not run, and seed 20262100 stays unused.

MS7c still delivers these results:

- The perceived-state boundary behind one switch, `--perceived`. A CPU test shows that no decision
  reads the truth under it.
- A proprioceptive hold detector that meets its calibration bar on the teacher's cycles.
- The release-end fix. After it, the structure costs nothing: with every knob at 0, the loop stows
  466 of 500, against 464 on true inputs.
- The requirement table. Position and class are tight, and the knobs share one cycles budget.
- The isolation budget. Position noise alone takes the cycles per correct stow from 1.146 to 1.292,
  the whole margin to the bar.

The reads give these findings for MS5a and the next milestone:

- The overhead lateral position error must be at or under about the nominal. That is a `pos` scale
  of about 1.0 for the cycles bar and 1.25 for the stow bar. The nominal is a bias sigma of 2 mm per
  axis plus a noise sigma of 1.5 mm in xy.
- The class error must be at or under about 5.5 % per classification for the stow bar, and about 3.3
  % for the cycles bar.
- The retrain, the snap-off serving and the snap clearance did not remove the position cost. A
  position fix at the final approach is the input that no MS7c serving change replaced. A wrist
  camera or a calibration that removes the per-episode bias can give that fix.

The first MS5a task measures the position and class errors of a baseline detector on real frames
against these limits.
