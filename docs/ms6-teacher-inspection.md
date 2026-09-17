# MS6 phase A follow-up: the heap teacher inspection and the teacher-logic fixes

Date: 2026-09-16. Code: this branch on top of `main` at 4549f22 (merge of PR #52).
Design: `2026-09-16-ms6-teacher-inspection-design.md` in the claude-memory repo. Origin: the
phase A read (`docs/ms6-heap-teacher.md`) stowed 34 of 6000 parts against a rule of 0.90.

This report records four things. What the inspection tool saw. Which pre-registered
hypotheses held. Which fixes landed. What the fixed teacher reads on the same heaps. The
owner asked for the work to proceed by small live episodes, one change at a time, instead of
the plan's separate ladder campaign. The numbers therefore come from single-env episodes and
lone-part reads, not from the 24-episode rungs of the design. The design's repeated
120-episode read did not run.

## 1. Summary

The teacher failed for four separate reasons. The read's own instruments hid all four.

1. The clearance model rejected almost every grasp. It measured two jaw boxes at the closed
   width against the circumscribed disc of every neighbor. On a 12-part heap in the 13 x 20 cm
   tray, every part read a clearance between -12 and -50 mm. The teacher pushed three times
   and gave up at step 918 with zero pick cycles. The read's 2.05 cycles per episode were the
   heap tops that happened to clear.
2. The physics does not match that model. The jaws descend fully open. Each finger tip
   lands `OPEN_HALF_GAP` = 52.4 mm out from the part center and then sweeps inward. A brick
   target is stowed with a brick or a plate anywhere in that sweep band or under the landing
   spot. The landing does not survive a tray wall. A tip whose inner face lands past a wall's
   inner face never picks. A tip that lands 0.6 mm inside picks. It picks even with the hull resting on the wall top. On this tray about 40 percent of the parts lie with every closing axis unpickable.
3. Every plate and tile was lost the same way. At the level descend the tips hold 0.9 mm
   above the desk and reach 2.4 mm down a 3.2 mm side face. The pinch closes over the part's
   top edge and presses it into the desk. The part center ends at 0.37 mm and the jaws stall
   at 0.3 mm. Solref 2 ms and 1 ms fail the same way, so contact stiffness is not the cause. Tips pressed 2 mm under the desk pinch the 1-wide and 2-wide flats at 3.8 to 5.7 N
   per jaw.
4. The LIFT gate rejected real pinches on narrow parts. The jaw drive is a position
   actuator, so its force is the stall times its stiffness. A 15.8 mm brick stalls at 6.1 mm
   with 6.1 N. A 7.8 mm part stalls at 1.8 mm with 1.8 N. The fixed 2 N force floor failed
   every 7.8 mm part while it rose 100 mm in the jaws. The brick-derived 4.8 mm stall floor
   failed every real plate pinch, which stalls at 4.4 mm.

Four fixes landed, all teacher logic. A landing-spot clearance model with the walls as hard
stops. A push that moves the best wall-blocked part toward the tray center. A pressed-tip
descend for flat parts with a part-aware pinch gate. The design's F1 (tray filter) and F2
(forced pick, and a release before moving on). Three K = 50 held-out heaps ran with the
read's flags. The unchanged teacher stows 0, 0 and 0 of 50. The fixed teacher stows 15, 11
and 17 of 50. All three episodes reach the time limit, with 27 to 30 pick cycles and 28 to
39 pushes each.

That is 0.29 against a rule of 0.90. The rest is not teacher logic. Section 7 names it.

## 2. The tool

`scripts/mj_inspect_heap.py` builds one heap env exactly as the batch script does. It
drives the env with a logging subclass of `HeapOracle` and writes two files. The first is
`events.jsonl`. It holds a `scan` per selection, with every part's tray membership, closing
axes, landing clearance, wall clearance and swept count. It holds a `phase` per report. When a part crosses the tray interior, past a 5 mm hysteresis band, it holds an `exit` or
an `enter`. It holds one `end`. The second file is a free-camera video with the teacher's state stamped
on every frame. `scripts/heap_inspect_read.py` sums the logs into one table and an
`INSPECT_READ` token. `tests/test_inspect_heap_script.py` tests both.

A K = 12 episode runs in 3 to 40 seconds. The unchanged teacher gives up early, so its K = 50 episode
runs in 12 seconds. The fixed teacher runs to the time limit, in 220 seconds.

## 3. What the log and the frames showed

### 3.1 The clearance model (mechanism 1)

The first K = 12 episode (run seed 20261000, layout seed 20261010) ran on the unchanged
teacher. Its first scan had 11 candidates and every best clearance was negative. The
smallest loss was -12.4 mm, on a 2x2 brick 11 mm from a corner. The largest was -47.5 mm, on
a 1x6 plate 19 mm from a 2x8 plate. The frame shows a roomy tray with visible gaps between
most parts. The disc model turned a 63.8 x 15.8 mm plate into a 33 mm radius. A neighbor
34 mm away therefore read as overlapping the jaw box.

### 3.2 The open jaw and the walls (mechanism 2)

The finger geometry was measured in the MuJoCo model, in the grasp frame, with the jaws
open. Each finger's collision hull ends in a flat tip 8 mm across the face and 2 mm along
jaw travel. The inner face is vertical at 52.41 mm from the tool axis. The hull widens by
about 0.46 mm of thickness and 0.32 mm of half face per mm of height. It reaches its full
27.5 x 40 mm plate only 57 mm above the tips.

A lone 2x4 brick, with the jaws closing along the tray's long axis toward a wall:

| part center to wall face | tip inner face to wall face | descend | pick |
| --- | --- | --- | --- |
| 70 mm | 17.6 mm inside | converged | stowed |
| 60 mm | 7.6 mm inside | converged, 2 wall contacts | stowed |
| 55 mm | 2.6 mm inside | stalled at 5.7 mm tip height | stowed |
| 50 mm | 2.4 mm past | stalled at 16.6 mm | lost |
| 40 mm | 12.4 mm past | converged, one jaw outside the tray | lost |
| 20 mm | 32.4 mm past | one jaw outside the tray | lost |

The short walls read the same. A part 8 mm off the tray center line picks x-closing. A part
13 mm off does not. A y-closing grasp therefore needs the part within 45 mm of the center
line, out of 100. An x-closing grasp needs it within 10 mm, out of 65. Across the three
K = 50 heaps, 17 to 23 of the 50 parts start with every closing axis wall-blocked.

Neighbors are far less of an obstacle than the walls. Four reads placed a 2x2 brick or a 2x4
plate next to a 2x4 brick target. The neighbor sat 30 mm away in the sweep band, or 62 mm
away under the landing spot. The target was stowed in all four cases. The brick neighbor was stowed
after it.

### 3.3 Flat parts (mechanism 3)

Lone parts on the tray floor, one cycle each, with three retries allowed. The table gives
the outcome per descend.

| descend | 2x4 brick | 1x2 brick | 2x4 plate | 2x8 plate | 1x4 plate | 1x6 tile | 4x4 tile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| level, tips 0.9 mm up (shipped) | stowed | lost, 1.8 N | lost | lost | lost | lost | lost |
| tips at the desk | stowed | stowed | lost | stowed | stowed | stowed | lost |
| tips 2 mm under the desk | stowed | stowed | pinched 3.8 N, 4.4 mm stall | stowed | stowed | stowed | lost |

The 1x2 brick and the 1x2 slope are lost at the shipped descend only through the LIFT gate.
Their stall is 1.77 mm against a 1.4 mm floor, their force 1.77 N against a 2.0 N floor, and
they rise to 104 mm. The "pinched" cell is the same story: a real pinch that the 4.8 mm
stall floor rejects.

The frames (`plate.mp4` in the inspection scratch) show the two wedge tips meeting above the
plate at the end of CLOSE. The plate is beneath them, pressed into the desk. Stiffer contacts
(solref 2 ms and 1 ms against the shipped 5 ms) do not change the outcome. The cause is the
descend geometry, not the contact model.

The 4x4 plate and tile (31.8 mm both ways) are lost at every descend tried. With the fixes
in place, the lone reads stow six parts at yaw 0. They are the 2x4 brick, the 1x2 brick,
the 2x8 plate, the 1x4 plate, the 1x6 tile and the 2x6 tile. The 2x4 plate and the 2x4 tile are stowed at yaw 0
and lost at yaw 0.8. The 4x4 plate and tile are lost at both.

### 3.4 Ejections (mechanism 4 of the design, H-D)

After the drop and the settle, the three K = 50 heaps start with 9, 11 and 8 of 50 parts
outside the tray. That is the design's early-stop trigger, a mean above 5
of 50. The drop and the settle alone cost 16 to 22 percent. Whatever the teacher does, the rule of 0.90 cannot pass inside the teacher-logic scope. F1 still landed as a defect
fix.

The exit stream needs the hysteresis band. A part resting on a wall top jitters across the
wall face by a millimeter for thousands of steps. With the band, the routes over the three
fixed-teacher K = 50 heaps are in the table below. "carried" means the target left the tray
in the jaws on its way to the bin. That is not an ejection.

| route | unchanged teacher, 3 heaps | fixed teacher, 3 heaps |
| --- | --- | --- |
| settle | 28 | 28 |
| push | 2 | 14 |
| grasp | 0 | 7 |
| transit | 0 | 12 |
| carried | 0 | 49 |
| other | 1 | 3 |

## 4. Hypothesis verdicts

| hypothesis | verdict | evidence |
| --- | --- | --- |
| H-A off-tray targets | supported | the illustration episode's two targets, which `_candidates` counted but did not exclude |
| H-B the give-up idles the budget | supported | unchanged teacher, K = 50: DONE at steps 1037 to 2313 of 30900, 3 pushes, 0 to 3 cycles, 28587 to 29863 idle steps |
| H-B2 pushes do not raise clearance | supported for the old push | K = 12: best clearance -41.7, -43.5, -43.2, -42.9 mm across the three densest-cell pushes |
| H-C1 flat parts on the desk, the pick closes above the part | supported | section 3.3, every plate and tile at the level descend |
| H-C2 flat parts on the heap, tips foul the part beneath | undecided | not separated, because every flat pick failed on the desk already |
| H-C3 flat parts slip in transit | not supported | no flat part passed LIFT at the shipped descend |
| H-D1 the drop ejects | supported, early stop | 8 to 11 of 50 outside at step zero |
| H-D2 the push ejects | see the exit table | |
| H-D3 the jaws eject | see the exit table | |
| not pre-registered: the clearance model itself | supported, dominant | section 3.1 |
| not pre-registered: the open jaw against the walls | supported, dominant after the model | section 3.2 |
| not pre-registered: the LIFT gate on narrow and flat parts | supported | section 3.3 |

## 5. The fixes

The fixes live in `core/grasp.py`, `core/heap_oracle.py`, `core/global_ik.py`,
`core/checker.py` and the rig's LIFT gate. The tests are `tests/test_grasp.py`,
`tests/test_heap_oracle.py` and `tests/test_mj_heap_oracle.py`.

- Landing-spot clearance. `jaw_clearance` returns `(clearance, swept, wall)`. The landing
  spots are the two tip footprints `OPEN_HALF_GAP` out along the closing axis. Each spot
  grows with the neighbor's height by the measured hull taper. The target's grip band is the top `MIN_GRIP_OVERLAP` = 5 mm of its side faces. A neighbor
  whose top reaches into that band stands. Standing neighbors in the sweep band are counted, not blocking. The tray walls are
  rectangles. The whole stroke from the part face to the landing spot must clear them.
  Footprints are exact convex hulls of the projected box corners. The model's boundary matches the measured one, so `CLEARANCE_MIN` is 0. Grasps rank by 5 mm height band,
  then by swept count, then by clearance.
- The push. When no grasp clears, `_plan_push` takes the best wall-blocked part. It plans a
  sweep from behind that part, on its far side from the tray center, through its center and
  toward the tray center. The start is 4 mm off the part's footprint, clear of every other
  footprint, with the tip center at least 4 mm inside the walls. When the straight approach
  has no room, the start swings up to 115 degrees to either side. Every push must bring the
  part at least 5 mm closer to the center. With no wall-blocked part, the densest-cell sweep
  remains, now clamped inside the tray. Pushes have their own budget, `PUSH_BUDGET` = 50.
  After three pushes in a row, the best grasp whose tips land inside the tray is a forced
  pick (F2). A part wedged in a corner has no push.
- F1. A part whose center is outside the tray is counted and never a target.
- F2 and the release. DONE comes only from a budget or an empty candidate set. A cycle that
  fails its last retry goes through REOPEN before the next HOVER. A part riding in the
  pocket then drops in place instead of being carried off. At K = 12 a tile and a plate rode
  out of the tray during the next push's hover. They came back 360 steps later.
- Flat parts. Below `FLAT_HALF_HEIGHT` = 2 mm the descend asks for the tips
  `FLAT_TIP_PRESS` = 2 mm under the desk, at the level tilt only. The desk stops them at zero
  height, and the DESCEND stall is the intended end of the leg.
- The LIFT gate. `pinch_min_stall(width, part_min_size)` used the 2x4 brick's smallest dimension. It now
  uses the part's own. The six-brick task is unchanged.
  `pinch_min_force` scales the 2 N floor with the stall floor. The heap oracle exposes the
  target's smallest size beside its grasp width.

## 6. The K = 12 ladder and the K = 50 read

One K = 12 heap (run seed 20261000, layout seed 20261010, held-out split, the read's flags)
ran once per change. The episode limit is 270 s = 8100 control steps.

| step | teacher | stowed of 12 | cycles | pushes | end |
| --- | --- | --- | --- | --- | --- |
| 0 | unchanged | 0 | 0 | 3 | DONE at step 918 |
| 1 | exact footprints in the old jaw-box model | 0 | 0 | 3 | DONE at step 918 |
| 2 | + F1, F2, width-scaled force floor | 1 | 5 | 14 | time limit |
| 3 | + landing-spot model with walls | 2 | 7 | 8 | DONE at step 7257 |
| 4 | + push toward the center | 3 | 8 | 10 | time limit |
| 5 | + flat-part descend and part-aware gate | 3 | 8 | 9 | time limit |

Three K = 50 heaps ran on the held-out split with the read's flags, one env each. The run
seeds are 20261000 to 20261002 and the layout seeds 20261010 to 20261012.

| heap | teacher | stowed | cycles | pushes | forced picks | outside at start | wall-blocked at start | end | idle steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | unchanged | 0/50 | 3 | 3 | 0 | 9 | 0 | done | 28587 |
| 1 | unchanged | 0/50 | 0 | 3 | 0 | 11 | 0 | done | 29749 |
| 2 | unchanged | 0/50 | 0 | 3 | 0 | 8 | 0 | done | 29863 |
| 0 | fixed | 15/50 | 27 | 38 | 5 | 9 | 23 | timeout | 0 |
| 1 | fixed | 11/50 | 30 | 39 | 5 | 11 | 17 | timeout | 0 |
| 2 | fixed | 17/50 | 30 | 28 | 1 | 8 | 26 | timeout | 0 |

Over the three heaps the fixed teacher stows 43 of 150 = 0.29. That is 2.02 cycles per
stowed part, with 105 pushes and 11 forced picks. The per-family rows follow. A LIFT
row counts LIFT reports, so one cycle with three retries counts four times.

| family | stowed | LIFT passed | LIFT failed |
| --- | --- | --- | --- |
| brick | 25 | 25 | 92 |
| slope | 5 | 5 | 12 |
| plate | 12 | 16 | 32 |
| tile | 1 | 1 | 9 |

## 7. What is left, and where it is

- The walls. About 40 percent of the parts start wall-blocked on every closing axis. The
  push brings them in a few at a time, at 200 to 300 control steps per push. The tray
  interior (design H1) is 130 x 200 mm and the open jaw spans 105 mm. One way out is a
  larger tray, which is outside the D2 scope and an owner decision. The other is a pick
  primitive whose jaws open at the bottom instead of descending open. The second is teacher
  logic, but it changes what the dataset records during HOVER and DESCEND. It is a design
  decision, not a fix.
- The drop. 8 to 11 of 50 parts start outside the tray. The design's early stop fired on
  the first heap. The drop band and the settle are outside the D2 scope.
- Wide flat parts. The 4x4 plate and tile are lost at every descend tried. The 2-wide
  flats depend on the yaw. When the tips are pressed onto the desk, the physical pinch works at 3.8 to 5.7 N. The remaining loss is in the LIFT and the transit. It is not read
  yet.
- Cycle time. The fixed teacher spends the whole 1030 s episode on about 30 cycles and 35
  pushes. The rule's 1.3 cycles per stowed part reads 2.3 over the three heaps and the
  K = 12 episode. The retries, three per failed cycle at about 800 steps, are the largest
  cost.
- The corner. A part within about 12 mm of two walls has no push start inside the tray and
  no landing spot. It stays.

## 8. Open items for the owner

1. Whether to reopen the tray footprint or the pick primitive (section 7, first item).
2. Whether to reopen the drop band and the settle (second item).
3. Whether the repeated 120-episode read runs on this bundle at 0.29, or waits for 1 and 2.
