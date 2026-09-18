# MS6 matched jaw opening: the pick, the release and the grip

Date: 2026-09-18. The owner watched one heap episode in the MuJoCo viewer on `main` at
`5726ed5` and again after each rung. Every defect in this report was seen by eye first and
measured afterwards. The three rungs ship on `feat/ms6-matched-jaw-opening`.

Every rung runs the same three held-out heaps of 50 parts the MS6 ladder has used
throughout. The seeds are `--run_seed 2026100n --layout_seed 2026101n` for n = 0, 1, 2, at the
read flags of `MS6 rework section 2`. Each rung therefore grades 150 parts. The logs are at
`~/data/maker-arm-ms6/f/r6_{0,1,2}` for rung 1, `g/r7_{0,1,2}` for rung 2 and `h/r8_{0,1,2}`
for rung 3. R3 is the shipped teacher, already on record at `e/r3_{0,1,2}`. Every number below
is recomputable from those `events.jsonl` files.

## 0. The one defect behind all three rungs

The gripper channel is binary by contract. A raw action at or above zero opens the jaws and
one below zero closes them. So every phase drove the jaws to an extreme, and the teacher never
asked for the opening the part actually needed. That one fact had three faces:

- the pick descended fully open, 52.4 mm out on each side (rung 1)
- the release opened fully, about 44 mm of travel per jaw, over the bin (rung 2)
- the grip closed all the way shut, with nothing to stop it (rung 3).

All three are fixed by asking for the part's own width. All three are heap only, behind
`EnvOracle.matched_jaw_opening`, so the six-brick task and its gated corpus are untouched.

## 1. The ladder

| rung | contents | stowed | fraction | Wilson 95% | per heap | carried but never delivered | flat-part lifts | cycles per stowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3 | the shipped teacher | 49/150 | 0.3267 | [0.257, 0.405] | 14, 14, 21 | 7 of 56 | 0.25 | 1.673 |
| rung 1 | the pick opening | 55/150 | 0.3667 | [0.294, 0.446] | 19, 18, 18 | 18 of 73 | 0.43 | 1.545 |
| rung 2 | the release | 76/150 | 0.5067 | [0.427, 0.586] | 26, 23, 27 | 4 of 80 | 0.40 | 1.276 |
| rung 3 | the grip | 73/150 | 0.4867 | [0.408, 0.566] | 26, 17, 30 | 7 of 80 | 0.47 | 1.288 |

Rung 2 is the only rung whose Wilson interval clears the rung below it. Rung 3 reads below the
rung under it, so the pre-registered rule hands that decision to the owner (section 5).

## 2. Rung 1, the pick opening: ADOPTED

`core.grasp` modeled both jaw landing spots 52.4 mm out from the part center whatever the
part, because the jaws descended fully open. `layout.TRAY_RADIUS` is 0.088. A part more than
about 66 mm from the dish center therefore had NO heading whose stroke cleared the rim. That
makes 44 percent of the dish area a geometric dead zone.

Measured on the R3 logs, of every in-dish part sighting:

- 84.1 percent were refused at the landing spots
- 67.4 percent of all refusals were the rim alone
- 1.7 percent of the parts that had a graspable axis were selectable at all.

With nothing to pick the teacher pushes. 64 percent of its decisions were pushes, and every
heap ended by exhausting the 50-push budget.

`grasp.landing_half_gap(width)` replaces the constant, and `OracleRig` holds that opening
through HOVER and DESCEND. For a 2x4 brick that is 11.9 mm against 52.4 mm.

The rung reads 0.3667 against 0.3267. That gain sits inside the noise of a 150-part sample.
The MECHANISM moved much further than the yield did, and that is the useful result:

| | R3 | rung 1 |
| --- | --- | --- |
| refusals caused by the rim | 78.8 % | 41.6 % |
| refusals caused by a neighbor | 19.5 % | 55.9 % |
| selectable sightings | 1.7 % | 2.5 % |
| pushes as a share of decisions | 64 % | 62 % |

The dish wall is no longer what gates the teacher. The heap's own density is. A second
constraint of nearly the same size sat behind the first. The next lever therefore aims at
neighbor blocking, not at the dish.

## 3. Rung 2, the release: ADOPTED

25 percent of the parts the rung 1 teacher lifted never reached the bin, against 12 percent
before it. Every traced one landed on the DESK, 60 to 200 mm SHORT of the release point. They
were thrown back toward the arm, not dropped. One instrumented cycle left the jaws at
(0.417, 0.002, 0.300) and came to rest at (0.246, 0.018, 0.230).

Two causes, and the rung fixes both.

RELEASE drove the jaws fully open over the bin, with the tool tilted 80.2 degrees. That is
about 44 mm of travel per jaw from a pinched grip, at roughly 3.3 mm per control step. RELEASE
now
joins `MATCHED_OPENING_PHASES`. A part needs `JAW_APPROACH_CLEARANCE` per side to fall out,
not the whole opening.

RELEASE also has no pose of its own. It HOLDS what EXTEND left, and `extend_goal` asks for no
jaw heading. The plane the jaws opened in was therefore inherited from the heading the PICK
happened to need. Measured over one instrumented episode, it stood at 5.3, 56.4, 78.0 and 71.6 degrees
off horizontal on four consecutive releases. The near-horizontal one dropped its part 13.3 mm
in ONE control step, which is a free fall from the whole release height. The steep ones moved
1.3 mm, with the part sliding off the lower plate. After the change the same four releases all
stand at 80.1 degrees, which is the geometric ceiling at `RELEASE_TILT_RAD`. That first part
now falls 0.3 mm instead of 13.3 mm.

The roll cannot be asked for through `GraspTarget.jaw_heading`. That residual row is
`jaw[1].atan2(jaw[0])`, the jaw axis's HORIZONTAL heading. A steep axis has almost no
horizontal component, so the row goes singular exactly where it is needed. This is the same
failure the lean azimuth has at zero tilt, which `core.global_ik` already records. Driven,
commanding the lean azimuth as the jaw heading gave a 2.6 degree plane, and commanding its
quarter turn gave 0.00007 degrees.

`link_007_joint` answers it instead, and exactly. Its rotation axis IS the tool axis. Rolling
it moves `grasp_center` by 0.00 mm and the tool axis by 0.00 degrees. The jaw axis follows one
for one, at a measured ratio of -1.000. `PICK_OFFSET_MEASURED` is (0, 0, -0.0321),
which lies on that axis, so the pick point does not move either. EXTEND's plan therefore gains
waypoints that turn that one joint and nothing else.

Nothing is solved in those waypoints and nothing can fail to solve. That is why the roll is
NOT asked of `_plan_carry`. That method's own docstring records three six-brick episodes lost
to an EXTEND held at a heading the solver refused to walk.

The owner had approved a change to `maker_arm_rs.Kinematics` to add the missing constraint.
The wrist measurement made that unnecessary, and no Rust change ships.

## 4. Rung 3, the grip: the owner decides

Trace one control step at a time on a LONE plate_2x4, with nothing around it. CLOSE ends with
2.9 N on both jaws and the part in the grip. LIFT then keeps closing:

| jaw opening (mm) | 5.18 | 4.81 | 4.21 | 3.70 | 3.19 | 2.69 |
| --- | --- | --- | --- | --- | --- | --- |
| force per jaw (N) | 2.9 | 1.0 | 1.25 | 0.9 | 0.33 | 0.00 |

The plate is squeezed out like a pip. The jaws end at 0.02 mm holding nothing, and the part is
back on the desk. A 9.6 mm brick stalls them at 5.9 mm and is held. A 3.2 mm plate has nothing
to stall them on.

So the LIFT gate's 0.00 N on a flat part was never a failed grasp. It samples AFTER the part
is gone. Reading it as a failed grasp cost three refuted hypotheses. Section 6 records them.

The carrying legs now hold the jaws at the grasp's half width less `CLOSE_SQUEEZE_M`. On the
lone-part wall table plate_2x4 goes from 0.00 N on both closing axes to 2.32 N and 1.89 N
held. plate_4x8's x axis also recovers.

The rung reads 0.4867 against rung 2's 0.5067. The intervals overlap almost completely. The
whole difference is heap 1, which drops from 23 to 17 while heap 2 gains from 27 to 30.

What the rung targeted did improve. Flat-part lifts read 0.47, the best of the four rungs, and
plate goes from 0.38 to 0.73. But tile goes from 0.43 to 0.33 and cancels it, and the 31.8 mm
grasps barely move, from 0.28 to 0.30. `CLOSE_SQUEEZE_M = 3 mm` was chosen from a 12-trial
lone-part table, and the 150-part read does not support it as optimal.

## 5. What this branch does not fix

The largest remaining group is flat parts gripped across 31.8 mm: 16 of 54. Their failure is a
SECOND thin-part mechanism, distinct from the extrusion rung 3 fixes.

Trace a lone plate_4x8. The closing jaws contact one side first, at 2.70 N against 0.05 N on
the other, and the part slides 1.5 mm away instead of being pinched. The heap failures carry
the matching asymmetry of 1.00. Both the successes and the failures stall at the same 12.9 mm,
which is the rung 3 floor. The difference is only whether a part is between the jaws.

Four-stud-wide parts cannot choose a narrower axis, because their other side exceeds
`MAX_GRASP_WIDTH`. 81 of 88 flat-part lifts had only one usable axis. Where a choice did
exist, the narrow axis lifted 3 of 3 and the wide axis 0 of 4. That is 7 cycles and not a
lever. Fixing this needs a design decision, not a constant.

## 6. Refuted hypotheses, recorded so nobody pays for them twice

- The flat descend's press depth. `FLAT_TIP_PRESS` was swept from 2 mm to 6 mm on a lone
  plate_2x4. Every value read 0.00 N. The knob is connected: `descend_pick_offset` moves 2 mm
  per 2 mm of press.
- The half height that selects the flat branch. A lying plate correctly reports 1.6 mm, which
  is under `FLAT_HALF_HEIGHT`, so the flat branch does fire.
- The hover jitter band. On a heap of flat parts only, lifts read 4 of 11 with the band and 5
  of 15 without it.

One measurement was also withdrawn during the work. A breakdown of thin-part lifts by
`retries_used` appeared to show the first attempt succeeding every time. The log writes that
field AFTER `report()` runs. A value of zero therefore means the attempt succeeded, and the
statistic is circular.

## 7. Defects found on the way

1. The probe's `_matched_opening_command` override, added by rung 1 on this branch, made
   `--shipped` run a fully-open descend. Every control row in `scripts/mj_probe_bottom_pick.py`
   was therefore reading the pre-rung-1 teacher. It now stands aside only while the probe
   drives the jaws itself.
2. `HeapOracle._release_target` takes its y from `bl.bin_ys`, the six-brick color slots, whose
   values reach 0.21. The heap bin spans 0.15. A destination at an outer slot therefore aims
   outside the bin. This did not fire on the three read heaps, which drew y near zero, so it
   is reported and not fixed here. It is the same class as the grader defect the MS6 rework
   fixed, on the release side.

## 8. Testing

788 tests pass and ruff is clean. New CPU tests pin `grasp.landing_half_gap`, the dish dead
zone it removes, and `global_ik.steepest_jaw_roll`. New driven MuJoCo tests pin the held
opening through the pick, the bounded opening at the release, and a lone flat part surviving
the lift. Each driven test was confirmed to fail against the old behavior before its fix
landed.
