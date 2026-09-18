# MS6 scene and grasp rework: the pre-registered ladder

Date: 2026-09-17. Design: `2026-09-17-ms6-scene-and-grasp-rework-design.md` in the
`claude-memory` repository. The owner wrote it after watching one heap episode in the MuJoCo
GUI on this branch's parent, `main` at `33e70fb`. The ladder ran on `~/data/maker-arm-ms6/`. R0
sits at `d/probe/base_readflags_{0,1,2}` (campaign d, already on record from
`docs/ms6-lever-verification.md`). R1 through R5 sit at `e/r{1,2,3,4,5}_{0,1,2}`. Every rung
runs the same three held-out heaps of 50 parts, `--run_seed 2026100n --layout_seed 2026101n`
for n = 0, 1, 2. Every rung uses the read flags of `MS6 rework section 2`. Each rung therefore
grades 150 parts. Every number in this report comes from `LADDER-EVIDENCE.md`, written by the
controller who ran the campaign. Nothing here is recomputed.

## 1. What the GUI inspection saw

The owner watched heap 0 of the current read campaign in the viewer (`MS6 rework section 1`).
The episode stowed 15 of 50 parts, in 27 cycles. 38 of those cycles were pushes and 5 were
forced picks. The episode ended at the 30900-step timeout with 26 of 50 parts outside the tray.
Four things were visible, and the event log holds all four:

1. The tray walls block the parts near the edge. The owner asked for a round dish instead of a
   box.
2. The teacher grips a part, the gate calls it a miss, and the jaws open and drop it. The log
   holds 8 such events against 26 real holds on this one heap.
3. Flat parts are picked with the wide face up and the pick fails. Plates fail 13 of 24 lift
   reports on this heap, the worst of the four families.
4. The destination bin overflows near the end of the episode. Its interior is 90 by 90 by
   40 mm, and the heap holds 159 cm3 of solid parts.

The design changes four things in response, one per pull request. Each change is measured on
the same pre-registered ladder:

- the lift decision (`MS6 rework section 4`)
- the tray shape (`MS6 rework section 5`)
- the bin size (`MS6 rework section 6`)
- the pick primitive for flat parts (`MS6 rework section 7`, with the tilted approach as
  `MS6 rework section 7.1`)

## 2. The pinch measurement

`MS6 rework section 3` is the measurement the owner asked for before the gate changed.
`scripts/mj_inspect_heap.py` records the three pinch factors, the two lift factors and their
floors, on every LIFT report, as telemetry beside the untouched gate. The instrumented run of
heap 0 reproduced the original episode exactly: 15 stowed, 27 cycles, 30900 steps.

The gate is `pinch_ok and lifted`. `pinch_ok` is a jaw stall floor, a force floor on the weaker
jaw and a limit of 0.30 on the left-right asymmetry. `lifted` is a rise above 50 mm and a
horizontal error under 30 mm.

These are the 8 reports on heap 0 that the pinch factors rejected while the part rose in the
jaws:

| family | grasp width | stall / floor (mm) | left / right (N) | force floor (N) | asymmetry | rise (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| plate | 3.2 mm | 1.97 / 1.00 | 0.00 / 1.98 | 0.42 | 1.00 | 98.9 |
| brick | 15.8 mm | 4.53 / 4.80 | 5.16 / 2.32 | 2.00 | 0.55 | 99.7 |
| brick | 15.8 mm | 4.80 / 4.80 | 4.85 / 4.84 | 2.00 | 0.00 | 104.1 |
| brick | 15.8 mm | 4.80 / 4.80 | 4.80 / 4.80 | 2.00 | 0.00 | 100.1 |
| brick | 31.8 mm | 15.48 / 4.80 | 1.26 / 1.25 | 2.00 | 0.01 | 94.6 |
| slope | 23.8 mm | 1.48 / 3.90 | 1.48 / 1.48 | 1.62 | 0.00 | 91.8 |
| plate | 31.8 mm | 0.97 / 1.60 | 0.98 / 0.97 | 0.67 | 0.00 | 90.4 |
| plate | 7.8 mm | 4.30 / 1.40 | 0.11 / 0.03 | 0.58 | 0.74 | 101.4 |

Three facts decided the change (`MS6 rework section 3`):

- The rejections split across all three factors. Four fail the force floor and three fail the
  asymmetry limit. Two sit exactly on the stall floor, where the 2x4 brick stalls at 4.80 mm
  against a floor of 4.80 mm. A single relaxed threshold rescues part of the set and leaves the
  rest.
- The 27 true misses on this heap fail by a wide margin. 24 fail all three factors and 3 fail
  two of them. Nothing sits between the two groups.
- `lifted` separates the episode on its own. Every real hold rose at least 90.4 mm with a
  horizontal error of at most 8.4 mm, against floors of 50 mm and 30 mm.

`core/oracle.py` therefore decides the LIFT transition on `result.lifted` alone
(`MS6 rework section 4`). `pinch_ok` and the three factors stay as telemetry on the phase
result and in the inspection log. The refused-pinch information is not lost. It is the
`rescued` count in the ladder below.

## 3. The six rungs

Every rung ran the three read heaps at the flags of `MS6 rework section 2`, so each rung grades
150 parts. Four numbers are read at every rung: the stowed fraction, the cycles per stowed
part, the parts outside the dish at the last scan, and the refused lifts. A refused lift is a
lift report with `lifted` true that the old pinch factors reject.

| rung | contents | stowed | fraction | Wilson 95% | cycles per stowed part | outside at the last scan | refused lifts that rose |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | the shipped teacher | 43/150 | 0.2867 | [0.220, 0.364] | 2.023 | 60 | 54 of 101 |
| R1 | R0 plus the lift decision | 50/150 | 0.3333 | [0.263, 0.412] | 1.780 | 53 | 20 of 57 |
| R2 | R1 plus the round dish | 42/150 | 0.2800 | [0.214, 0.357] | 1.714 | 57 | 18 of 53 |
| R3 | R2 plus the wide bin | 49/150 | 0.3267 | [0.257, 0.405] | 1.673 | 62 | 16 of 56 |
| R4 | R3 plus the tip-up | 16/150 | 0.1067 | [0.067, 0.166] | 1.625 | 66 | 7 of 21 |
| R5 | R4 plus the tilted pick | 16/150 | 0.1067 | [0.067, 0.166] | 1.625 | 66 | 7 of 21 |

R0 did not re-run for this ladder. `~/data/maker-arm-ms6/d/probe/base_readflags_{0,1,2}`
already holds the shipped teacher at the read flags on these three heaps, from campaign d.
Heap 0 of that data reads 53 lift reports and 8 refused lifts. That is the table of section 2
of this report exactly, so the flags and the seeds match. Every other rung ran fresh into
`~/data/maker-arm-ms6/e/r{1,2,3,4,5}_{0,1,2}`.

Every Wilson interval overlaps every other except R4 and R5, which sit clear below R1 and R3.

### R1, the lift decision: ADOPTED

The only rung that improves on the rung below it and stays inside its own interval.

The design measured 8 refused lifts of 26 real holds on heap 0 alone (section 2 above). Across
the three heaps R0 refuses 54 of 101 holds that rose. The old pinch gate refused more than half
of every hold that actually lifted. R1 stops refusing them and reads 0.3333 against 0.2867. The
cycles per stowed part fall from 2.023 to 1.780.

### R2, the round dish: a regression that the diagnostic explains

R2 reads 0.2800 against R1's 0.3333, which the ladder rule reports as a regression.

The controller debugged it before reporting it. All three R2 heaps ended with `reason` done and
`pushes` at exactly 50, which is `PUSH_BUDGET = CYCLE_BUDGET = 50`. They stopped at 24950,
29458 and 27153 control steps against a step budget of 30900. R1 used 34, 43 and 25 pushes and
ran to the timeout on all three heaps. The dish drives the teacher into far more pushes. The
push budget binds where it was slack on the rectangle, and the episode ends early with time on
the clock.

A diagnostic run with `PUSH_BUDGET = 200` and nothing else changed reads 47 of 150, 0.3133,
Wilson [0.245, 0.391]. All three heaps run to the timeout again, with 71, 54 and 59 pushes.
That diagnostic is NOT a rung. It lived in a scratch worktree on no branch, and the edit was
reverted. It is reported beside R2 and labeled a diagnostic, not a rung.

So about 0.033 of R2's 0.053 shortfall is the budget artifact. The remaining 0.020 sits inside
the noise at this sample size.

The rim works. Re-entries are parts that left the dish and came back. They rise from 4 on R1 to
21 on R2 and to 36 on R3. That is the leaning rim returning parts that slid out.

The dish also costs pushes. That is the ceiling the `ponytail:` comment in
`grasp.tray_wall_obstacles` already names. The rim is modeled as a vertical stop at its xy
shadow, so it refuses grasps that the 45 degree slope allows.

Parts still leave during the settle, 23 on R2 against 28 on R1. 50 parts crowd an interior that
is 7 percent smaller than the rectangle it replaced.

R2 read below the rung under it. The owner decides whether it stays.

### R3, the wide bin: ADOPTED

R3 reads 0.3267, which recovers the whole R2 shortfall and sits within noise of R1's 0.3333.
R3 also holds the best cycles per stowed part of any rung, at 1.673, and the most re-entries,
at 36. Two of its three heaps still end at the 50-push budget, so R3 carries the same artifact
as R2 and is still under-read.

### R4, the tip-up: a large regression with a named cause

R4 reads 0.1067 against R3's 0.3267. Its interval does not overlap R3's.

The cause is direct. The tip fires 54 times across the three heaps and succeeds ONCE. `tips`
reads 19, 16 and 19 per heap, and `tips_ok` reads 0, 0 and 1.

The motion is not the problem. Every tip leg converges: 54 TIP_HOVER converged, 52 TIP_DESCEND
converged with 2 stalled, and 54 TIP_SWEEP converged. The arm goes where the planner asks. The
part does not end up standing.

The cost compounds because of what the trigger replaces. A flat plate's best grasp grips
3.2 mm of face, under the 5 mm threshold, so every flat part is diverted into a tip. Those same
parts were being PICKED on R3, imperfectly but often. Section 2 of this report shows plates
failing 13 of 24 lift reports on one heap, so roughly half succeeded there. R4 trades a pick
that works about half the time for a tip that works about one time in fifty.

State this plainly: the tip-up as built does not stand a part up. The trigger diverts parts the
teacher can otherwise pick. R4 read below the rung under it, by the largest margin in this
ladder. The owner decides whether it stays.

### R5, the tilted pick: ships as wiring only, with a measured negative result

R5 reads exactly the same numbers as R4 on every one of the four pre-registered quantities. The
rung ships `PICK_TILT_RAD = 0.0`, so the jaw plane is the horizontal plane again. That identity
is the evidence that the rung is inert as shipped, not a copying error.

The rung ships the wiring because both non-zero values the ladder tried cost more than they
bought.

At `math.radians(25.0)` the commanded pose moves. `oracle_runtime` builds every carrying leg
with `max(goal.tilt_rad, GRASP_TILT_RAD)`. `robots/frames.py` sets `GRASP_TILT_DEG` to 20.0,
from the D3 sweep's 18.42 degree worst case. Two driven tests broke at this value. The
bottom-pick probe needed a second DESCEND leg, so its first grasp attempt failed. A driven heap
episode completed one pick instead of two in the same step budget.

At `math.radians(20.0)` the pose does not move at all, because 20 degrees is the floor the
runtime already holds. The suite was green on that value in isolation. The six rungs were then
rebased into stack order. Three tests failed at the top of the stack, and a commit-by-commit
bisect landed all three on the 20-degree commit. The decisive one is
`tests/test_mj_heap_oracle.py::test_the_heap_teacher_stows_three_spread_parts`, a driven
capability test. It stows all three parts at a zero tilt. At 20 degrees it reports `succeeded`
False, with 2 of 3 sort attempts, 12 retries and ZERO unplanned legs. Zero unplanned legs means
the solver still reaches everything. The teacher cannot find an acceptable grasp.

The mechanism: `grasp.AXIS_MAX_TILT_RAD` is 15 degrees. When the jaw plane was assumed level,
that band was calibrated at 15 degrees. Tilting the plane 20 degrees, while keeping the
15 degree band, refuses every closing axis lying along the tilt direction. That removes an
orientation class of parts. Making the plane truthful without re-measuring the band makes the
model STRICTER than the hardware, rather than more accurate. Two constants have to move
together, and only one did.

This closes the hope of `MS6 rework section 7.1`. `candidate_grasps` keeps an axis only within
`AXIS_MAX_TILT_RAD` of the jaw plane. A lying plate's 3.2 mm thickness axis enters the
candidate set only once the jaw plane leans at least 75 degrees. That is far outside this arm's
30 degree tool cone. A tilted approach cannot grip a plate's wide faces on this arm. The tip-up
is the only route to that grip, not an alternative.

R5 ships as wiring, not as a behavior change, with a measured negative result on both tried
tilt values.

## 4. The rung-to-rung differences

R1 and R3 are the two rungs that earned their place. Each reads at or above the rung under it,
and each stays inside its own Wilson interval against the rung it builds on.

R2 and R4 are the two rungs that read below the rung under them. The pre-registered rule says
the owner decides whether either stays.

- R2, the round dish, reads 0.2800 against R1's 0.3333. About 0.033 of that 0.053 shortfall is
  a push-budget artifact the round dish exposes. `PUSH_BUDGET = CYCLE_BUDGET = 50` binds where
  it was slack on the rectangle. The remaining 0.020 sits inside the noise of a 150-part
  sample. R3 recovers the whole shortfall on top of R2.
- R4, the tip-up, reads 0.1067 against R3's 0.3267, the largest drop in this ladder. It is also
  the only rung whose Wilson interval does not overlap the rung below it. The tip fires 54
  times and succeeds once. Every leg of the motion converges, so the loss is not a planning or
  tracking failure. The trigger diverts flat parts that R3 was picking, imperfectly but about
  half the time, into a mechanism that works about one time in fifty.

R5 changes nothing measured. It ships the tilt wiring at `PICK_TILT_RAD = 0.0`, so its four
numbers match R4's exactly. Both non-zero tilt values that were tried cost more than they
bought. One broke two driven tests outright. The other left the ladder's own oracle unable to
find an acceptable grasp on a driven capability test. The cause is `grasp.AXIS_MAX_TILT_RAD`.
It was calibrated for a level jaw plane and was not re-measured for a tilted one. R5 therefore
ships as wiring only, carrying a measured negative result rather than a behavior change.

## 5. Defects and surprises found on the way

1. The R2 push-budget artifact above. `PUSH_BUDGET = CYCLE_BUDGET = 50` binds on the dish and
   ends episodes early. The owner decides whether it moves.
2. The heap grader used the six-brick bin height. `layout.bin_interior_centers` computes its z
   from `BIN_INTERIOR[2]`. Both `HeapOracle._stowed` and `core/stowed.py::per_brick_stowed`,
   the dataset label and the gate grader, resolved through it. The center was 0.175, where the
   heap bin's own center is 0.185. It did not misgrade, because the checked band was
   [0.055, 0.295] against a true interior of [0.155, 0.215], but grading rested on that
   coincidence. Fixed in the R3 rung, with a test that pins the grader's center to
   `layout.heap_bin_center`.
3. When the margin exceeded the radius, `grasp.clamp_to_tray` reflected a point through the
   dish center. The old rectangular clamp degraded sanely instead. No caller reached it. Fixed.
4. The rectangular tray probes were retired. `scripts/mj_heap_drop_sweep.py`,
   `scripts/heap_sweep_read.py` and `tests/test_drop_sweep_scripts.py` are deleted. That sweep
   works by patching `TRAY_INTERIOR` and re-reading `tray_static_boxes()`. The dish removes the
   first name and changes the arity of the second. Every number the sweep produced is archived
   in `docs/ms6-lever-verification.md` under PR #59 and PR #62, which stays unedited.
   `scripts/reach_map.py` keeps a local `RECT_TRAY_INTERIOR` for its historical comparison.
5. A tip did not count toward `pushes_in_a_row`. After three pushes and a diverted forced pick,
   the next obstructed part lost its own grace period. Fixed: a tip now counts like a push.
6. The plan's own test for the rim top was arithmetically impossible. It asserted
   0.298 <= 0.290. The bound became `SCATTER_MAX_RADIUS + TRAY_WALL_HEIGHT`.
7. The plan's push-heading test asserted a 90 degree bound at a point where the code draws from
   a 95.1 degree half-arc, so it fails for about 94 percent of seeds. The existing wall test
   was retargeted to the rim instead, where the half-arc is exactly 90 degrees.
8. Without a widened guard in `HeapOracle.next_goal`, a scheduled tip ended the episode instead
   of running, because the guard only recognized `PUSH_HOVER`.

## 6. Not run

The six-brick re-baseline of `MS6 rework section 4` did NOT run. `scripts/oracle_check.py`
fails in the Isaac venv with `ImportError: cannot import name 'Kinematics' from
'maker_arm_rs'`. That venv's `maker_arm_rs` exposes Arm, clamp_command, encode_mit,
parse_frame and profile, and no Kinematics, while the project venv exposes it.
`core/global_ik.py` on main imports Kinematics at module scope. Every Isaac script that reaches
the solver already fails there, on main, before this work. The remedy is to install the
pyproject's pinned `maker-arm-rs` rev `e1541514` into `~/src/IsaacLab/.venv`. That is an owner
decision about a shared environment. So the MS4 gate of 0.840 keeps its old-gate provenance
until someone runs it.
