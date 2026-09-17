# MS6 lever verification: the tray, the drop and the pick primitive

Date: 2026-09-17. Campaign `d` under `~/data/maker-arm-ms6/`. Design:
`2026-09-17-ms6-lever-verification-design.md` in the `claude-memory` repository. This report
answers open items 1 and 2 of `docs/ms6-teacher-inspection.md` section 8.

## 1. Summary

The recommended lever is the pick primitive, not the tray. The primitive opens the jaws at the
bottom of the descend instead of descending with the jaws open. It needs a grasp ranking model
matched to that opening. With both parts in place the teacher stows 67 of 150 parts on three
heaps of 50 parts each. The fixed teacher stows 43 of 150 on the same three heaps. The tray lever is nearly
dead inside this arm's driven reach. The drop lever misses its pre-registered bar at every band
tested. The cheapest ejection lever measured is a taller tray wall, and it is not one of the
two levers the owner asked about.

The two headline numbers:

- The matched bottom-open pick stows 67 of 150, a fraction of 0.447. The fixed teacher stows 43
  of 150 on the same heaps, a fraction of 0.287. That is a factor of 1.56.
- The widest tray with a driven-valid near edge grows the short axis from 130 to 160 mm. At
  that tray the wall blockage falls from 23.8 to 20.2 of 50 parts. The parts outside the tray
  fall from 6.6 to 3.6 of 50.

Interpretation: 0.447 on three heaps is not a pass. The pre-registered phase A rule is 0.90,
and the fixed teacher's repeated read over 121 episodes is 0.2676. A 1.56 factor on three heaps
does not close that gap. The recommendation is about which lever to open next, not about a
passing read.

Both owner decisions are stated as options in section 9. Decision 1 is the tray footprint
against the pick primitive. Decision 2 is the drop band and the settle.

## 2. What was measured

Three experiments, from the design. Two more reads were added in the middle of the round, each
pre-registered before its code ran.

| experiment | what it measures | script | pull request | branch commit |
| --- | --- | --- | --- | --- |
| 0, the reach map | where the solver can pick, on a grid | `scripts/reach_map.py` | #58 | `f03f16c` |
| 1, the drop and tray sweep | the heap at step zero, over tray, drop, settle and wall | `scripts/mj_heap_drop_sweep.py`, `scripts/heap_sweep_read.py` | #59 | `9da4400` |
| 2, the pick probe | the bottom-open pick, on a lone part and on a heap | `scripts/mj_probe_bottom_pick.py` | #57 | `c59ba15` |

Added in the middle of the round:

- The driven reach band, design section 6.1. The reach map queries the solver. The comment
  above `SCATTER_MAX_RADIUS` in `core/layout.py` says that bound is a driven-tracking
  measurement. It also says that a re-measurement must drive real descends. It must not query
  the solver. The band read is a mode of the probe. It drives a real descend and pick at each
  point of a radius and azimuth grid.
- The matched clearance model, design section 6.2. The first heap read of the probe lost
  against the fixed teacher. `jaw_clearance` in `core/grasp.py` still modeled a fully open
  jaw, so the teacher kept rejecting the grasps the new primitive can make. The
  `--matched_clearance` flag gives the ranking the probe's own landing half gap.

The campaign did not run from `main`. It ran from a local integration worktree,
`~/src/maker-arm-lab-ms6d`, which merges the three branches without pushing anything.
Integration commit `723bea3` merges `feat/ms6-reach-map` at `f03f16c`, `feat/ms6-drop-sweep` at
`0fe7029` and `feat/ms6-bottom-pick-probe` at `b9b1a27`. Integration commit `539091f` folds in
two later fix rounds, and brings `feat/ms6-drop-sweep` to `9da4400` and
`feat/ms6-bottom-pick-probe` to `782b4e4`. The corrected heap reads at `8a388e8` and `e9dca80`,
and the matched read at `c59ba15`, ran from a second worktree on the probe branch. Pull
requests 57, 58 and 59 are open. None is merged. Merging them is the owner's decision.

No experiment changed production behavior. Nothing under `source/maker_arm_lab/core/` changed.
Every script is a probe or a measurement script.

## 3. The reach

Two kinds of number appear here, and the difference decides the tray question. A solver claim
comes from `core/global_ik.py`. It says a pose has a joint solution inside the tolerance. A
driven number comes from a real descend and pick in MuJoCo. It says the arm tracked the plan
and the part ended stowed.

The comment above `SCATTER_MAX_RADIUS` in `core/layout.py` is why the distinction exists. It
records that 0.280 m is the largest radius that tracked clean at every azimuth measured. It
also instructs whoever re-measures the bound to drive real descends. The re-measurement must
not query the solver. The first coarse run of the reach map returned a rectangle with a corner
at 0.334 m. A solver read is not allowed to widen the band. Every rectangle below is therefore
clipped to 0.280 m, and the near edge of every tray candidate comes from a driven read.

### 3.1 The solver map

Solver claims. The grid step is 5 mm. It covers x in [0.05, 0.32] m and y in [-0.20, 0.20] m.
Each point runs eight headings, at the pick height of a 2x4 brick. 4455 grid points: 3975
pickable at every heading, 192 pickable at some heading, 288 at none.

| group | pick | size (m) | center x (m) | area against the current tray |
| --- | --- | --- | --- | --- |
| all headings | overall | 0.150 by 0.180 | 0.133 | 1.04x |
| all headings | square | 0.115 by 0.115 | 0.115 | 0.51x |
| any heading | overall | 0.160 by 0.375 | 0.128 | 2.31x |
| any heading | square | 0.210 by 0.210 | 0.152 | 1.70x |
| any heading | current ratio | 0.130 by 0.200 | 0.112 | 1.00x |

The all-heading cut needs every heading to converge at every point of the rectangle. The
any-heading cut needs one. The teacher ranks the grasps a part offers and takes the best one
that clears. The any-heading cut is therefore the weaker test a real tray needs. Every corner
of every rectangle above sits inside 0.280 m.

### 3.2 The driven band, far side

Driven numbers, 65 points, one 2x4 brick on a bare desk, no tray. The control is the same read
at radius 0.20 m.

| radius (m) | -40 deg | -20 deg | 0 deg | +20 deg | +40 deg |
| --- | --- | --- | --- | --- | --- |
| 0.28 | stowed | stowed | stowed | stowed | stowed |
| 0.30 | stowed | lost | lost | lost | stowed |
| 0.32 | stowed | lost | lost | lost | stowed |
| 0.34 | no solution | no solution | no solution | no solution | no solution |

At plus and minus 40 degrees the arm stows the part at 0.30 and 0.32 m. The tracking residual
there is 0.030 to 0.031 rad. The control reads 0.0354 to 0.0356 rad at the same azimuths. The
residual sits below the control's, so the loss at the center is not a tracking loss at those
radii.

The two loss causes are separated. At 0.30 m and plus or minus 20 degrees the residual is 0.024
to 0.026 rad. The control reads 0.025 rad. The part was lost at the grasp. At 0.30 m and 0
degrees the residual is 0.139 rad, 6.6 times the control's 0.021 rad, which is the tracking
failure the constant describes.

Interpretation: the reach is a wedge, not a circle. It holds to 0.32 m at the outer azimuths
and to 0.28 m at the center. A single radius bound assumes a circle, and this data does not
support that shape.

### 3.3 The driven band, near side

Driven numbers, 60 points, radii from 0.04 to 0.14 m plus the 0.20 m control.

The smallest radius that stows the part at every azimuth is 0.10 m. At 0.09 m the pick fails at
plus and minus 40 degrees, with the brick correctly placed and the residual at the control
level. That is a grasp failure, not a tracking failure.

14 of the 60 rows carry a placement drift above 2 mm. The cause is the arm's own body. It
occupies the desk close to its base, so the brick does not rest where the script put it. At
0.04 m the drift runs from 76 to 163 mm. The rows at 0.04 m and 0.05 m are not reach readings
at all and must not be quoted as such. The token's own `min_stowed_radius_m` of 0.04 is that
artifact.

### 3.4 The tray candidates

Every candidate takes its near edge at the driven 0.10 m and keeps its far corners inside 0.280
m.

| candidate | interior (mm) | center x (m) | far corner (m) | note |
| --- | --- | --- | --- | --- |
| current | 130 by 200 | 0.195 | 0.279 | the baseline |
| wide | 160 by 200 | 0.180 | 0.279 | the short axis grows by 30 mm |
| square | 165 by 165 | 0.1825 | 0.278 | the largest square inside both bounds |
| long | 120 by 340 | 0.160 | 0.278 | the largest area, at a narrower short axis |

The short axis is the axis the jaw span binds. The open jaw spans 105 mm. The current tray
leaves 12.5 mm on each side of it. The wide candidate leaves 27.5 mm.

## 4. The tray and the drop sweep

240 samples, 48 combinations, 5 layout seeds per combination, K = 50, held-out split, parts
seed 20260915. No teacher runs here. Each sample drops the parts, settles the physics and
measures step zero. Every number below is a mean over the 5 seeds, recomputed from
`d/sweep/samples.jsonl`.

### 4.1 The tray axis

At the current drop band, settle 480 and wall 20 mm.

| tray | outside | wall_blocked | of which no_grasp | blocked | stacked |
| --- | --- | --- | --- | --- | --- |
| 130x200@195 (current) | 6.6 | 32.0 | 8.2 | 42.2 | 32.4 |
| 160x200@180 | 3.6 | 30.8 | 10.6 | 45.6 | 31.4 |
| 165x165@182 | 4.6 | 31.0 | 9.2 | 45.2 | 35.2 |
| 120x340@160 | 0.8 | 31.6 | 7.0 | 47.6 | 32.0 |

`wall_blocked` counts every part with no grasp whose wall-only clearance clears the threshold.
`no_grasp` counts the parts inside it that have no grasp candidate at all, for any reason. The
difference is the count of parts that have a grasp a wall still rules out. It is 23.8 of 50 at
the current tray and 20.2 of 50 at the 160 mm one.

### 4.2 The drop axis

At the current tray. A drop band of `30:1.2` is a height uniform in [0.03, 0.03 + 0.0012 K]
metres. A band of `20:0` is a fixed 0.02 m with no jitter.

| wall mm | drop | settle | outside | stacked | wall_blocked | speed max |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 30:1.2 (current) | 480 | 6.6 | 32.4 | 32.0 | 0.024 |
| 20 | 15:0.6 | 480 | 10.6 | 28.8 | 29.4 | 0.019 |
| 20 | 20:0 | 480 | 20.4 | 21.2 | 22.4 | 0.016 |
| 40 | 30:1.2 | 480 | 4.0 | 29.8 | 30.6 | 0.009 |
| 40 | 15:0.6 | 480 | 6.0 | 29.4 | 33.2 | 0.012 |
| 40 | 20:0 | 480 | 6.8 | 30.2 | 35.6 | 3.811, unstable |

The last row is not a reading. Two of its five samples left the physics unsettled. The residual
part speed reads 19.02 m/s at settle 480 and 149.28 m/s at settle 960. Everywhere else it reads
about 0.02 m/s. Both sit at layout seed 1. The residual-speed field is what caught them.

Settle length changes nothing measured. Over all 24 cells the largest difference in mean
`outside` between settle 480 and settle 960 is 0.2 parts.

### 4.3 What each lever buys

- A 30 mm wider short axis buys 3.6 of 50 fewer wall-blocked parts, out of 23.8. That is about
  15 percent of the blockage. It also buys 3.0 of 50 fewer ejections.
- The long tray buys almost all the ejections, 6.6 down to 0.8 of 50, and none of the blockage.
  Its short axis is narrower than the current one.
- A 40 mm wall at the current tray and the current drop buys 2.6 of 50 fewer ejections and does
  not raise the blockage.
- A lower or fixed drop buys nothing. It raises the ejections.

Interpretation: no tray inside the driven reach removes the wall blockage. The blockage is not
mostly a tray-size problem at this heap density. It is a problem of what the jaw needs next to
a part.

## 5. The pick primitive

The probe reuses the inspection tool's single-episode driver. It descends with the jaws at a
narrow opening, opens them to the full span at pick depth, and closes. The narrow opening is
the part width plus twice the tip thickness plus 4 mm. For a 2x4 brick that is a landing half
gap of 17.5 mm, against the fully open 52.4 mm.

### 5.1 The lone-part wall table

One 2x4 brick at a measured distance from a tray wall face, x-closing and y-closing. The
control is the shipped fully-open descend through the same driver.

| distance (mm) | bottom-open, y | bottom-open, x | shipped, y | shipped, x |
| --- | --- | --- | --- | --- |
| 70 | stowed | stowed | stowed | stowed |
| 60 | stowed | stowed | stowed | stowed |
| 55 | stowed | stowed | stowed | stowed |
| 50 | stowed | stowed | lost | lost |
| 40 | stowed | stowed | lost | lost |
| 20 | stowed | stowed | lost | lost |

The bottom-open descend stows 12 of 12. The shipped descend stows 6 of 12 and loses everything
at 50 mm and closer. This reproduces the inspection report section 3.2 row for row on the
control side. The lift stall reads 5.9 mm and the lift force 5.89 N on every stowed row.

One limit on that clean sweep. Below 55 mm the jaws open into the tray wall and lever the arm
up. The pick point ends at 11.9 mm at the 20 mm row, against a nominal 4.8 mm. The pick holds,
but it presses on the tray. Interpretation: on a real arm that is a durability question, and it
bounds how far this table can be trusted as a primitive read.

### 5.2 The first heap read, and the defect that withdrew it

The first heap read returned 12 of 50 on one heap. It is withdrawn. Two defects made it
invalid.

- `parse_args` overwrote `--run_seed` and `--layout_seed` with its own pinned constants in heap
  mode. Three runs at three seed pairs produced three byte-identical episodes, down to the
  `push_parts` list. The flags were accepted and discarded.
- The heap mode never set `--episode_draws`, one of the read's own flags. That flag redraws the
  run-level jitter at reset, and it draws from the same random generator the heap drop uses.
  Without it the first scan sits 100.6 mm mean and 270.3 mm maximum away from the inspection's
  log. With it the two agree to 0.007 mm.

The 12 of 50 was measured on a heap that is not the inspection's heap. The fix makes the
constants defaults, honors a flag the caller passes, and adds a test.

### 5.3 The corrected read

Three heaps, run seeds 20261000 to 20261002 and layout seeds 20261010 to 20261012, at the
read's flags. The fixed teacher's column is the inspection's own run on the same heaps.

| heap | bottom-open | fixed teacher | cycles | pushes | forced | end |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10/50 | 15/50 | 19 | 50 | 9 | push budget at step 28476 |
| 1 | 8/50 | 11/50 | 21 | 23 | 4 | timeout |
| 2 | 17/50 | 17/50 | 33 | 44 | 7 | timeout |
| total | 35/150 = 0.233 | 43/150 = 0.287 | | | | |

The primitive wins every lone-part wall row and loses on a heap. Heap 0's parts were never
attempted. The per-cycle stow rate is unchanged, 0.53 against 0.56, and the descend converges
more often. What collapsed is the supply of grasps the teacher selects. 73 percent of scans
ended in a push, and the push budget ended the episode.

### 5.4 The matched-clearance read

`jaw_clearance` in `core/grasp.py` models the landing spots of a jaw that descends fully open.
The probe changed the descend and left the model alone. The teacher therefore kept rejecting,
as unpickable, exactly the grasps the new primitive can make. The `--matched_clearance` flag
patches that constant at run time, for the heap mode only. It sets the landing half gap the
ranking uses to the probe's own 17.5 mm. Nothing under `core/` changes.

| heap | matched | fixed teacher | bottom-open, shipped model | cycles | pushes | end |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 18/50 | 15/50 | 10/50 | 37 | 26 | timeout |
| 1 | 22/50 | 11/50 | 8/50 | 40 | 34 | timeout |
| 2 | 27/50 | 17/50 | 17/50 | 42 | 38 | timeout |
| total | 67/150 = 0.447 | 43/150 = 0.287 | 35/150 = 0.233 | | | |

No matched run came near the 50-push budget. All three ran to the clock. Cycles per stowed part
fall to 1.78, against the fixed teacher's 2.02 on the same heaps.

The mechanism is the model change itself, and it reads directly off the first scan of each
episode. The matched and the unmatched runs start from identical part poses on all three heaps.
The count of parts whose best grasp lands past a wall falls from 25, 17 and 26 to 6, 3 and 11.
That is a mean of 22.7 against 6.67 on the same physical heaps. The campaign ledger records
6.67 and 22.0 for the same pair, from the read script's own count.

The yield was not bought by knocking parts out. Count the ejections that are neither the settle
nor a carried target. They read 36 for the fixed teacher, 20 for the unmatched probe and 11 for
the matched one. The grasp-phase bucket is zero across all three matched episodes.

### 5.5 Two mechanisms that were wrong

The round records this because it is what the round learned. After the corrected heap read the
implementer floated two mechanisms for the loss. The first was the design's own named risk: a
narrow-opening descend fouls the neighbors it passes on the way down. The second was the
opposite. The shipped descend lands 52.4 mm out and sweeps a space clear around every target.
The gentler descend then leaves a denser and more tangled tray.

Both were wrong. The matched read settles it. The supply of grasps collapsed because the
selection model rejected them, not because the heap was in a worse physical state. The physical
state was identical at step zero, and the model change alone recovers the yield.

Interpretation: the logs cannot separate either mechanism. The log records part poses only in
`scan` events, once per selection. A displacement between two scans covers DESCEND, CLOSE,
LIFT, the carry, the release and the settle together. A phase-attributed read needs per-phase
part poses. `events.jsonl` does not carry them.

## 6. The pre-registered readings

Written before any code, in design section 7 and in sections 6.1 and 6.2. No wording is revised
to fit the data.

| reading | verdict | evidence |
| --- | --- | --- |
| R-A, the tray can grow | not fired | The widest tray with a driven near edge is 165 mm in the short axis. The reading asked for 170 mm. `wall_blocked` does not fall near 0.10 at any tray in the sweep. |
| R-B, the tray cannot grow | not fired | Rectangles wider than 0.15 m in the short axis do exist inside the band: 0.160 m overall and 0.210 m square, both at the any-heading cut. |
| R-C, the pick primitive works | not fired | The first half fired: the wall table stows a lone brick at 20 mm from a wall, past the reading's 40 mm bar. The second half asked for at least 25 of 50 on the heap the fixed teacher stowed 15 of. The matched run returns 18 of 50 there. |
| R-D, neither lever alone is enough | fired on its literal terms, conclusion does not follow | R-A failed and R-C failed, which is exactly what R-D asked for. See the note below. |
| R-E, the drop is fixable | not fired | No drop band at the current tray reaches `outside` under 2 of 50. The lowest measured is the current band, at 6.6. |
| R-F, the drop trades against stacking | not fired | No cell met the precondition, because no cell has `outside` under 2 of 50. |
| R-G, the band holds at 0.280 m | not fired | Two points beyond 0.280 m stow the part with a residual below the control's, at plus and minus 40 degrees, at 0.30 and 0.32 m. |
| R-H, the band is wider than 0.280 m | not fired | 0.30 m is lost at -20, 0 and +20 degrees. |
| R-I, the near side bounds the tray | fired | The driven near-side sweep gives 0.10 m as the smallest radius that stows at every azimuth. No number in the repository covered the near side before. |
| R-J, the primitive lifts the heap | fired | The matched run stows 67 of 150, against the threshold of 43, the fixed teacher's count on the same three heaps. |
| R-K, the primitive does not lift the heap | not fired | 67 of 150 is above 43 of 150. |

R-D needs a note, because its literal firing is misleading. R-D was written to mean that both
levers had a fair test and both lost. R-C did not give the primitive a fair test. It ran a
changed descend against an unchanged selection model, so the teacher rejected the grasps the
changed descend makes. R-J is the fair test, and the primitive wins it. The report therefore
does not draw R-D's conclusion. That conclusion sends the owner to a smaller K, a different
part catalog or a different task. R-D stands as written and as fired, and the conclusion it
points at does not follow from this data.

## 7. Findings nobody pre-registered

Each item below is a finding the design did not pre-register.

- Not pre-registered: a lower and less jittered drop ejects more parts, not fewer. The fixed 20
  mm drop takes `outside` from 6.6 to 20.4 of 50 at the current tray, while it lowers `stacked`
  from 32.4 to 21.2.
- Not pre-registered: the tray wall height is the cheapest ejection lever measured. 20 to 40 mm
  takes `outside` from 6.6 to 4.0 at the current tray and the current drop, and it does not
  raise `wall_blocked`.
- Not pre-registered: the settle length changes nothing measured. Settle 480 and settle 960
  read the same mean `outside` in all 24 cells, within 0.2 parts.
- Not pre-registered: two cells are physically unstable, both at a 40 mm wall with the fixed 20
  mm drop, at layout seed 1. The residual part speed reads 19.02 m/s and 149.28 m/s against
  about 0.02 m/s elsewhere. Their part counts are not readings. The residual-speed field is
  what caught them.
- Not pre-registered: the driven reach band is a wedge, not a circle. It holds to 0.32 m at
  plus and minus 40 degrees and to 0.28 m at the center.
- Not pre-registered: a brick will not rest on the desk close to the arm's base, because the
  arm's own body occupies that space. 14 of 60 near-side rows carry a placement drift above 2
  mm, and the 0.04 m rows drift 76 to 163 mm.
- Not pre-registered: the matched primitive lowers ejections rather than raising them, 36 to 11
  over three heaps, with a grasp-phase bucket of zero.
- Not pre-registered: a full heap episode is reproducible byte for byte. A repeat of heap 0 at
  the same seeds returned the same stowed count, the same cycle count and the same push list.

## 8. What the data does not separate, and what bounds the recommendation

- The matched clearance is a single scalar of 17.5 mm. The real opening depends on the part
  width. 17.5 mm is the measured held gap for a 15.8 mm part. It is pessimistic for narrower
  parts and optimistic for wider ones. Near the 45 mm maximum grasp width the true held gap is
  about 26.5 mm. The gain spreads over all four part families, so no single size carries it. A
  width-aware model is the honest next version, and it will not read 67 of 150 again without
  being measured.
- The rectangles of the reach map are conservative by about half a grid step. The rectangle
  builder pads a corner outward by half a step before the clip tests it. The current tray's own
  far corner is 0.2786 m unpadded and 0.2818 m padded, against the 0.280 m bound. The padded
  clip therefore excludes the current tray at every grid step tried. That is a property of a
  grid read, not a claim about the arm.
- The reach map is a solver read. The arm's own collisions with the desk and with a tray wall
  are not in it. The driven band covers a lone brick on a bare desk, not an arm reaching into a
  tray full of parts.
- Three heaps are three heaps. 67 of 150 has a wide interval and it is not a campaign. The
  fixed teacher's own read over 121 episodes is 0.2676, and the three-heap read of the same
  teacher is 0.287. The two agree, which is some evidence that three heaps are not
  pathological, and it is not a substitute for a campaign.
- The sweep measures step zero only. It does not measure what a teacher does with a heap over
  an episode. A tray that starts better can still run worse.
- The wall table is a lone part against a bare wall. It does not cover a part against a wall
  with neighbors around it.
- The probe's `--part` flag runs a flat-part wall table that no test exercises and that this
  report does not read.
- No read here measures the dataset. Whether a bottom-open pick can be recorded as a
  demonstration is not answered by any number above.

## 9. The owner decision

### Decision 1: the tray footprint against the pick primitive

Option 1A, open the pick primitive. Cost: it changes what a demonstration records during HOVER
and DESCEND. The inspection report already named that as a design question and not as a fix. It
also needs the ranking model changed with it, which is a change under `core/`. The number stays
untrustworthy until the width-aware version of the clearance lands. The 17.5 mm scalar is a
probe, not a design. Buys: 67 of 150 against 43 of 150 on three heaps, cycles per stowed part
1.78 against 2.02, and ejections cut to under a third. It also makes every lone-part wall row
work down to 20 mm.

Option 1B, grow the tray. Cost: a new tray in sim and a physical twin later. It also needs a
bin layout check, because the far corner sits 1 mm inside the measured reach bound. Buys: 3.6
of 50 fewer wall-blocked parts and 3.0 of 50 fewer ejections, at the 160 by 200 mm candidate.
Interpretation: this does not pay for itself against option 1A.

Option 1C, do both. Cost: the sum of the two. Buys: unmeasured. No sweep cell ran a wider tray
with the matched primitive.

Option 1D, do neither and change the task. Cost: the phase A design changes. Buys: nothing
measured here. R-D pointed at this option and R-J removed its premise.

Recommendation: option 1A, with the width-aware clearance model as its first piece of work.

### Decision 2: the drop band and the settle

Option 2A, keep the current drop band and the current settle. Cost: 6.6 of 50 parts start
outside the tray, which caps the stowed fraction near 0.87 before the teacher moves. Buys: no
work.

Option 2B, raise the tray wall to 40 mm. Cost: a tray change, and a jaw that has more wall to
hit on the way in. The sweep says the blockage does not rise, 32.0 to 30.6 at the current drop.
Buys: `outside` from 6.6 to 4.0 of 50. Interpretation: this is the cheapest ejection lever
measured, and it is not one of the two levers the question named.

Option 2C, lower the drop or fix it. Cost: the sweep says it is worse. `outside` rises to 10.6
or 20.4 of 50. Buys: less stacking, 32.4 down to 21.2 at the fixed drop.

Option 2D, lengthen the settle. Cost: run time. Buys: nothing measured. Settle 960 reads the
same as settle 480 in all 24 cells.

Recommendation: option 2A now. A later tray change for another reason then carries option 2B.
R-E did not fire. No band measured reaches the pre-registered bar of 2 of 50. The drop is not
the binding lever at this heap density.

## 10. Reproduction

Every command below ran from the integration worktree, not from `main`. Run them from a
checkout that carries all three branches.

```bash
# Experiment 0, the reach map. CPU only, no MuJoCo. About one minute.
.venv/bin/python scripts/reach_map.py --out ~/data/maker-arm-ms6/d/reach/reach_map.json

# Experiment 1, the drop, settle and tray sweep. 240 samples.
MUJOCO_GL=egl uv run python scripts/mj_heap_drop_sweep.py \
    --out_dir ~/data/maker-arm-ms6/d/sweep --seeds 5 --heap_k 50 \
    --trays 130x200@195 --trays 160x200@180 --trays 165x165@182 --trays 120x340@160 \
    --wall_heights 20 --wall_heights 40 \
    --drop_bands 30:1.2 --drop_bands 15:0.6 --drop_bands 20:0 \
    --settle_steps 480 --settle_steps 960
uv run python scripts/heap_sweep_read.py ~/data/maker-arm-ms6/d/sweep/samples.jsonl

# Experiment 2, the lone-part wall table and its shipped-descend control.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode wall_table \
    --out_dir ~/data/maker-arm-ms6/d/probe/wall
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode wall_table --shipped \
    --out_dir ~/data/maker-arm-ms6/d/probe/wall_control

# The driven reach band, far side at the script's own grid, then the near side.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode reach_band \
    --out_dir ~/data/maker-arm-ms6/d/probe/band_far
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode reach_band \
    --radii_mm 40,50,60,70,80,90,100,110,120,130,140,200 \
    --out_dir ~/data/maker-arm-ms6/d/probe/band_near

# Experiment 2, the corrected heap read. One command per heap.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --episode_draws --run_seed 20261000 --layout_seed 20261010 --out_dir /tmp/ed_0
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --episode_draws --run_seed 20261001 --layout_seed 20261011 --out_dir /tmp/ed_1
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --episode_draws --run_seed 20261002 --layout_seed 20261012 --out_dir /tmp/ed_2

# Experiment 2, the matched clearance read, the same three heaps.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --matched_clearance --run_seed 20261000 --layout_seed 20261010 --out_dir /tmp/mc_0
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --matched_clearance --run_seed 20261001 --layout_seed 20261011 --out_dir /tmp/mc_1
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --matched_clearance --run_seed 20261002 --layout_seed 20261012 --out_dir /tmp/mc_2
```

The runs used the worktree's own virtual environment, with `PYTHONPATH` on the worktree's
`source` directory. A worktree then tests the code it carries, not the primary checkout's
editable install. The fixed teacher's column comes from campaign `b`, under
`~/data/maker-arm-ms6/b/k50/fixed_0` to `fixed_2`.
