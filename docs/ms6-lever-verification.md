# MS6 lever verification: the tray, the drop and the pick primitive

Date: 2026-09-17. Campaign `d` under `~/data/maker-arm-ms6/`. Design:
`2026-09-17-ms6-lever-verification-design.md` in the `claude-memory` repository. This report
answers open items 1 and 2 of `docs/ms6-teacher-inspection.md` section 8.

## 1. Summary

The recommended lever is the pick primitive, not the tray. The primitive opens the jaws at the
bottom of the descend instead of descending with the jaws open. It needs a grasp ranking model
matched to that opening. Its measured gain depends on one corpus-level randomization, the hover
jitter band, and the report gives both readings. The tray lever is nearly dead inside this
arm's driven reach. The drop lever moves the ejections only through the tray and the wall, not
through the drop band alone.

The two headline numbers, both on the same three held-out heaps of 50 parts:

- At the read's flags, which is the configuration the corpus runs, the matched primitive stows
  45 of 150. The fixed teacher stows 43 of 150. That is a gain of 2 parts of 150.
- Without the hover jitter band, the matched primitive stows 67 of 150 and the fixed teacher
  stows 32 of 150. That is a doubling, in a configuration the project does not use.

Interpretation: the primitive is the stronger of the two levers the owner asked about, and its
measured gain is configuration-dependent. The tray lever removes 3.6 of the 23.8 wall-blocked
parts of 50 at the best candidate. Both of those numbers come from the drop sweep, at step
zero. The grasp ranking model moves the same blockage far more. That read uses a different
script and a different dataset, and section 5.4 carries it beside its own baseline. The size of
the yield that follows is what the hover jitter band decides. Retuning that band for the new
primitive is a measurable follow-up, not a decision.

Interpretation: neither reading is a pass. The pre-registered phase A rule is 0.90. The fixed
teacher's repeated read over 121 episodes is 0.2676. 45 of 150 is 0.300 and 67 of 150 is 0.447.
Both sit far below the rule.

Both owner decisions are stated as options in section 9. Decision 1 is the tray footprint
against the pick primitive. Decision 2 is the drop band and the settle.

## 2. What was measured

Three experiments, from the design. Two more reads were added in the middle of the round, each
pre-registered before its code ran. A set of flag controls was added after the first draft of
this report, because the first heap comparison was not like for like.

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
  against the fixed teacher. `jaw_clearance` in `core/grasp.py` still modeled a fully open jaw,
  so the teacher kept rejecting the grasps the new primitive can make. The
  `--matched_clearance` flag gives the ranking the probe's own landing half gap.

Added after the first draft:

- The flag controls. The first heap comparison ran the probe at its argparse defaults and the
  fixed teacher at the read's flags. The two differ. Every heap number in this report now comes
  from a run at a named flag set. The fixed teacher is measured under the same flag set as the
  probe. Section 5.4 gives the table and section 5.5 gives the isolation.

The campaign did not run from `main`. It ran from a local integration worktree,
`~/src/maker-arm-lab-ms6d`, which merges the three branches without pushing anything.
Integration commit `723bea3` merges `feat/ms6-reach-map` at `f03f16c`, `feat/ms6-drop-sweep` at
`0fe7029` and `feat/ms6-bottom-pick-probe` at `b9b1a27`. Integration commit `539091f` folds in
two later fix rounds, and brings `feat/ms6-drop-sweep` to `9da4400` and
`feat/ms6-bottom-pick-probe` to `782b4e4`. The heap re-runs and the flag controls ran at
integration commit `60e6130`. Pull requests 57, 58 and 59 are open. None is merged. Merging
them is the owner's decision.

Every heap episode this report reads is archived under `~/data/maker-arm-ms6/d/probe/`, one
directory and one log per episode. Section 5 names the directory of every number. Every
number in this report was recomputed from those files for this report.

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

The candidates were not required to sit inside a rectangle the reach map returned, and none of
them does. Every candidate runs out to x = 0.26 m, while the largest any-heading rectangle
stops at x = 0.2075 m. The current production tray does not sit inside a rectangle either, for
the same reason. `rectangle_info` pads a corner outward by half a grid step before the clip
tests it. The current tray's far corner therefore reads 0.2818 m padded, against a 0.280 m
bound. A separate test drives the real solver over the current tray's own footprint and finds
all four corners pickable at every heading. So the rectangles bound nothing about these four
candidates. The bounds that do apply to them are the driven near edge of 0.10 m and the 0.280 m
far-corner limit, and both were applied.

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
the current tray, 20.2 at the 160 mm one, 21.8 at the square and 24.6 at the long one.

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

### 4.3 The cells that clear 2 ejections of 50

Twelve of the 48 cells read `outside` under 2 of 50. Ten of the twelve also hold `stacked` at
or below the current cell's 32.4. The best cell is the long tray at a 40 mm wall, at the
current drop band: `outside` 0.4 and `stacked` 30.0.

| tray | wall mm | drop | outside | stacked |
| --- | --- | --- | --- | --- |
| 120x340@160 | 40 | 30:1.2 (current) | 0.4 | 30.0 |
| 120x340@160 | 20 | 30:1.2 (current) | 0.8 | 32.0 |
| 160x200@180 | 40 | 30:1.2 (current) | 1.2 | 30.6 |
| 160x200@180 | 40 | 20:0 | 1.2 | 30.8 |
| 120x340@160 | 40 | 20:0 | 1.8 | 28.8 |

Every one of the twelve cells changes the tray, the wall, or both. Not one of them sits at the
current tray with the current wall. The drop band alone therefore did not move the ejections.
The best cell holds the current drop band unchanged.

### 4.4 What each lever buys

- A 30 mm wider short axis buys 3.6 of 50 fewer wall-blocked parts, out of 23.8. That is about
  15 percent of the blockage. It also buys 3.0 of 50 fewer ejections.
- The long tray buys almost all the ejections, 6.6 down to 0.8 of 50, and none of the blockage.
  Its short axis is narrower than the current one.
- A 40 mm wall at the current tray and the current drop buys 2.6 of 50 fewer ejections and does
  not raise the blockage.
- A lower or fixed drop band buys nothing. It raises the ejections.

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
control is the shipped fully-open descend through the same driver. Source:
`d/probe/wall_table/bottom_open.json` and `shipped.json`.

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

### 5.2 The other part families, at a clear wall

The campaign also holds one row per part family at 70 mm from the wall, under both descends.
Source: `d/probe/wall_table/wall_table_<part>.json` and the `_shipped` file beside it.

| part | bottom-open | shipped |
| --- | --- | --- |
| brick_1x2 | stowed | stowed |
| plate_1x4 | lost | lost |
| plate_2x4 | lost | lost |
| plate_2x8 | stowed | stowed |
| tile_1x6 | stowed | stowed |

Both descends agree on every family. Two of the five parts are lost at 70 mm, which is a
distance where no wall blocks anything. The inspection named the flat-part descend as a
separate mechanism, and this table says the bottom-open primitive does not touch it. That bears
on the owner's decision, so section 9 carries it.

### 5.3 What the first heap read got wrong, twice

The first heap read returned 12 of 50 on one heap, and it is withdrawn. Two defects made it
invalid.

- `parse_args` overwrote `--run_seed` and `--layout_seed` with its own pinned constants in heap
  mode. Three runs at three seed pairs produced three byte-identical episodes, down to the
  `push_parts` list. The flags were accepted and discarded.
- The heap mode never set `--episode_draws`, one of the read's own flags. That flag redraws the
  run-level jitter at reset, and it draws from the same random generator the heap drop uses.
  Without it the first scan sits 100.6 mm mean and 270.3 mm maximum away from the inspection's
  log. With it the two agree to 0.007 mm.

A third defect survived both fixes and is the reason this report was rewritten. The probe ran
at its argparse defaults, while the fixed teacher it was compared against ran at the read's
flags. The probe's heap mode pins only `heap_k`, `parts_split`, `run_seed`, `layout_seed` and
`episode_draws`. It does not pin `--nearest_first_pick_order`, the hover jitter band,
`--frame_jitter` or `--cycle_budget`. So the earlier 67 of 150 against 43 of 150 compared two
different configurations. Pull request #57 now records the resolved flag set in the probe's
own `PROBE_OK` token and in its end event. A future log answers this question without a
re-run. Section 5.4 is the like-for-like table.

### 5.4 The heap read, under both flag sets

All six configurations, all on the same three held-out heaps of 50 parts, run seeds 20261000 to
20261002 and layout seeds 20261010 to 20261012. `outside_at_start` reads 9, 11 and 8 in every
run, which are the inspection's own numbers, so every row starts from the same physics.

| teacher and model | with the hover jitter band | without the band |
| --- | --- | --- |
| fixed teacher, shipped descend, shipped model | 43/150 (15, 11, 17) | 32/150 (10, 6, 16) |
| bottom-open descend, shipped model | 35/150 (12, 9, 14) | 35/150 (10, 8, 17) |
| bottom-open descend, matched model | 45/150 (15, 15, 15) | 67/150 (18, 22, 27) |

Archived at `d/probe/`: `base_readflags_*` and `base_defaults_*` for the first row,
`unmatched_*` and `minimal_unmatched_*` for the second, `matched_*` and `iso_nojitter_*` for
the third. Every cell of the table has an archived log for all three heaps.

The first row at the read's flags reproduces the inspection's 15, 11 and 17 exactly. The heaps,
the scripts and the grader therefore all agree with campaign `b`.

The corpus runs at the read's flags. The like-for-like answer is therefore 45 against 43, a
gain of 2 parts of 150. The 67 against 32 belongs to a configuration the project does not use.

The summary rows below come from one source, `scripts/heap_inspect_read.py`, run over the
archived `events.jsonl` of each group.

| configuration | stowed | cycles per stowed | wall-blocked at start | ejections |
| --- | --- | --- | --- | --- |
| fixed teacher, read's flags | 43/150 | 2.02 | 22.0 | 36 |
| fixed teacher, no band | 32/150 | 2.75 | 22.0 | 37 |
| bottom-open, shipped model, read's flags | 35/150 | 2.29 | 22.0 | 22 |
| bottom-open, matched, read's flags | 45/150 | 2.51 | 6.67 | 21 |
| bottom-open, matched, no band | 67/150 | 1.78 | 6.67 | 11 |

`wall-blocked at start` is the mean count of parts whose best grasp lands past a wall, at the
first scan. `ejections` counts tray exits that are neither the settle nor a carried target.

The matched model is what moves the wall blockage, in every configuration, by 15.3 parts of 50.
The per-heap counts are 23, 17 and 26 under the shipped model. They are 6, 3 and 11 under the
matched one. An independent recount off the same `scan` events agrees on every heap but one. On
heap 0 under the shipped model it reads 25 against the script's 23. The table uses the script
for both sides.

The per-cycle stow rate does not improve. At the read's flags it reads 0.449 for the
bottom-open descend against 0.506 for the fixed teacher. That rate divides the successful
entries of the logged `cycle_outcomes` list by the length of that list, which is 78 and 85. The
matched model reads 0.405 at the read's flags and 0.573 without the band. So the matched model
buys its yield through more cycles at the read's flags, 113 against 87, and not through a
better cycle. Those two cycle counts come from `extra_stats.cycles`, a different field. It runs
one to two higher per group than the length of the outcome list. The two denominators differ,
and no reading here mixes them. An earlier draft of this report said the per-cycle rate was
unchanged at 0.53 against 0.56. That pair does not reproduce from the archived logs and is
withdrawn.

The DESCEND convergence fraction is 0.425 for the fixed teacher and 0.429 for the bottom-open
descend at the read's flags. An earlier claim that the bottom-open descend converges much more
often does not hold at the read's flags. It reads 0.53 for the matched model without the band,
against 0.425.

### 5.5 The isolation: the hover jitter band is the flag

Matched bottom-open, all three heaps, at the read's flags with one flag dropped.

| configuration | heaps | total | archive |
| --- | --- | --- | --- |
| read's flags, all of them | 15, 15, 15 | 45/150 | `matched_*` |
| read's flags without `--nearest_first_pick_order` | 15, 15, 15 | 45/150 | `iso_nonearest_*` |
| read's flags without the hover jitter band | 18, 22, 27 | 67/150 | `iso_nojitter_*` |
| the probe's argparse defaults, heap 0 only | 18 | | `minimal_matched_0` |

The hover jitter band of 0.035 to 0.080 m is the flag. Dropping nearest-first ordering changes
nothing at all: `iso_nonearest_*` matches `matched_*` in stowed count, cycles and pushes on
every heap. Dropping the band recovers the full 67, and `minimal_matched_0` reproduces
`iso_nojitter_0` exactly.

The same band moves the fixed teacher in the opposite direction. It reads 43 with the band and
32 without it. `base_nojitter_*` matches `base_defaults_*` exactly, so nearest-first ordering
and frame jitter change nothing for it either.

So one corpus-level randomization lifts the shipped teacher by about a third and cuts the
matched primitive by about a third. This round does not separate why. The band exists for
policy training, and its interaction with a bottom-open descend is unmeasured.

### 5.6 The two mechanisms that were floated, and what the data supports

The round records this because it is what the round learned. After the corrected heap read the
implementer floated two mechanisms for the loss. The first was the design's own named risk: a
narrow-opening descend fouls the neighbors it passes on the way down. The second was the
opposite. The shipped descend lands 52.4 mm out and sweeps a space clear around every target.
The gentler descend then leaves a denser and more tangled tray.

The matched read shows that the model change alone is sufficient to recover the yield. It does
not show that either mechanism was absent. The logs cannot separate them. The log records part
poses only in `scan` events, once per selection. A displacement between two scans covers
DESCEND, CLOSE, LIFT, the carry, the release and the settle together. A phase-attributed read
needs per-phase part poses. `events.jsonl` does not carry them. So both mechanisms stay
unmeasured, and neither is needed to explain the numbers.

What the data does support: the model change is what moves the wall blockage. All three
configurations at the read's flags start from identical part poses on all three heaps, checked
pose by pose at the first scan. The wall-blocked count at that scan falls from 22.0 to 6.67
across the model change.

## 6. The pre-registered readings

Written before any code, in design section 7 and in sections 6.1 and 6.2. No wording is revised
to fit the data.

| reading | verdict | evidence |
| --- | --- | --- |
| R-A, the tray can grow | not fired | The first clause holds: experiment 0 returns an any-heading square of 0.210 m, which is 80 mm wider than 0.13 m. The second clause fails. `wall_blocked` reads 30.8 to 31.6 of 50 at every candidate against 32.0 at the current tray, nowhere near the 0.10 the reading asked for. The reading's own 0.40 baseline is off too, because the sweep measures 0.64 at the current tray. The verdict does not change. |
| R-B, the tray cannot grow | not fired | Rectangles wider than 0.15 m in the short axis do exist inside the band: 0.160 m overall and 0.210 m square, both at the any-heading cut. |
| R-C, the pick primitive works | not fired | The first half fired: the wall table stows a lone brick at 20 mm from a wall, past the reading's 40 mm bar. The second half asked for at least 25 of 50 on the heap the fixed teacher stowed 15 of. The matched run returns 15 of 50 there at the read's flags and 18 of 50 without the hover jitter band. |
| R-D, neither lever alone is enough | fired on its literal terms, conclusion does not follow | R-A failed and R-C failed, which is exactly what R-D asked for. See the note below. |
| R-E, the drop is fixable | fired | Twelve sweep cells read `outside` under 2 of 50, and ten of them hold `stacked` at or below the current cell's 32.4. The best is the long tray at a 40 mm wall, at `outside` 0.4 and `stacked` 30.0. See the note below. |
| R-F, the drop trades against stacking | not fired | Its precondition is met by twelve cells, and the trade it predicted does not appear. Ten of the twelve lower the stacking instead of raising it by half. |
| R-G, the band holds at 0.280 m | not fired | Two points beyond 0.280 m stow the part with a residual below the control's, at plus and minus 40 degrees, at 0.30 and 0.32 m. |
| R-H, the band is wider than 0.280 m | not fired | 0.30 m is lost at -20, 0 and +20 degrees. |
| R-I, the near side bounds the tray | fired | The driven near-side sweep gives 0.10 m as the smallest radius that stows at every azimuth. No number in the repository covered the near side before. |
| R-J, the primitive lifts the heap | fired | The matched run stows 45 of 150 at the read's flags. The threshold is 43, the fixed teacher's count on the same three heaps under the same flags. The margin is 2 parts of 150. Without the hover jitter band it stows 67 against that same threshold. |
| R-K, the primitive does not lift the heap | not fired | 45 of 150 is above 43 of 150 at the read's flags, and 67 of 150 is above it without the band. |

R-D needs a note, because its literal firing is misleading. R-D was written to mean that both
levers had a fair test and both lost. R-C did not give the primitive a fair test. It ran a
changed descend against an unchanged selection model, so the teacher rejected the grasps the
changed descend makes. R-J is the fair test, and it fires by 2 parts of 150 at the corpus
flags. What removes R-D's premise is that R-C tested an unmatched selection model, not the size
of that win. The report therefore does not draw R-D's conclusion. That conclusion sends the
owner to a smaller K, a different part catalog or a different task. R-D stands as written and
as fired, and the conclusion it points at does not follow from this data.

R-E needs a note too. Its wording asks experiment 1 for a drop band and settle with `outside`
under 2 of 50 and `stacked` no higher than the current cell. Twelve cells meet it. Every one of
those cells also changes the tray or the wall. The drop band alone is not what moved the
ejections, and the best cell holds the current drop band unchanged. An earlier draft of this
report marked R-E as not fired, which was wrong.

## 7. Findings nobody pre-registered

Each item below is a finding the design did not pre-register.

- Not pre-registered: the hover jitter band decides the primitive's measured gain. It lifts the
  shipped teacher from 32 to 43 of 150 and cuts the matched primitive from 67 to 45. This round
  does not separate why.
- Not pre-registered: nearest-first pick ordering changes nothing for either teacher, on any
  heap, in stowed count, cycles or pushes.
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
- Not pre-registered: the matched primitive lowers ejections rather than raising them, 36 to 21
  at the read's flags and 36 to 11 without the band.
- Not pre-registered: the bottom-open descend does not fix the flat-part loss. The 1x4 and 2x4
  plates are lost at 70 mm from a wall under both descends.
- Not pre-registered: a full heap episode is reproducible byte for byte. A repeat of heap 0 at
  the same seeds returned the same stowed count, the same cycle count and the same push list.

## 8. What the data does not separate, and what bounds the recommendation

- The hover jitter band's interaction with the primitive is unmeasured. The band is a
  corpus-level randomization of the hover height, and this round shows only that it helps one
  descend and hurts the other. No read here says why, and no read here tries a retuned band.
- The matched clearance is a single scalar of 17.5 mm. The real opening depends on the part
  width. 17.5 mm is the measured held gap for a 15.8 mm part. It is pessimistic for narrower
  parts and optimistic for wider ones. Near the 45 mm maximum grasp width the true held gap is
  about 26.5 mm. The gain spreads over all four part families, so no single size carries it. A
  width-aware model is the honest next version, and its yield is unmeasured. The 17.5 mm value
  is a probe setting and not a design. No code under `core/` carries it.
- Three heaps are three heaps. 45 of 150 against 43 of 150 is a margin of 2 parts, which three
  episodes cannot resolve. The report states the direction and not a significant difference.
  The fixed teacher's own read over 121 episodes is 0.2676 and its three-heap read is 0.287.
  That is some evidence that these three heaps are not pathological.
- The rectangles of the reach map are conservative by about half a grid step. The rectangle
  builder pads a corner outward by half a step before the clip tests it. The current tray's own
  far corner is 0.2786 m unpadded and 0.2818 m padded, against the 0.280 m bound. The padded
  clip therefore excludes the current tray at every grid step tried.
- The reach map is a solver read. The arm's own collisions with the desk and with a tray wall
  are not in it. The driven band covers a lone brick on a bare desk, not an arm reaching into a
  tray full of parts.
- The sweep measures step zero only. It does not measure what a teacher does with a heap over
  an episode. A tray that starts better can still run worse.
- The wall table is a lone part against a bare wall. It does not cover a part against a wall
  with neighbors around it.
- No read here measures the dataset. Whether a bottom-open pick can be recorded as a
  demonstration is not answered by any number above.

## 9. The owner decision

### Decision 1: the tray footprint against the pick primitive

Option 1A, open the pick primitive. Cost: it changes what a demonstration records during HOVER
and DESCEND. The inspection report already named that as a design question and not as a fix. It
also needs the ranking model changed with it, which is a change under `core/`. The number stays
untrustworthy until the width-aware version of the clearance lands. Buys: at the configuration
the corpus runs, 45 of 150 against 43 of 150, a gain of 2 parts. Without the hover jitter band,
67 of 150 against 32 of 150. It also makes every lone-part wall row work down to 20 mm, and it
cuts the ejections from 36 to 21. It does not fix the flat-part loss: the 1x4 and 2x4 plates
are lost under both descends even at a clear wall.

Option 1B, grow the tray. Cost: a new tray in sim and a physical twin later. It also needs a
bin layout check, because the far corner sits 1 mm inside the measured reach bound. Buys: 3.6
of 50 fewer wall-blocked parts and 3.0 of 50 fewer ejections, at the 160 by 200 mm candidate.
Interpretation: this does not pay for itself against option 1A.

Option 1C, do both. Cost: the sum of the two. Buys: unmeasured. No sweep cell ran a wider tray
with the matched primitive.

Option 1D, do neither and change the task. Cost: the phase A design changes. Buys: nothing
measured here. R-D pointed at this option. Its premise fails for one reason only: R-C tested a
primitive whose selection model was not matched to it. The size of R-J's win is not that
reason.

Option 1E, open the primitive and measure the hover jitter band with it. Cost: option 1A plus a
band ladder, which is three heaps per rung. Buys: the one number this round does not supply,
which is the primitive's gain at a band tuned for it. Interpretation: this is the cheapest way
to find out whether the 2-part gain or the 35-part gain is the real one.

Recommendation: option 1E. The primitive is the stronger lever. Its gain at the corpus
configuration is 2 parts of 150, and the band is the reason. Measure the band before the
dataset cost is paid.

### Decision 2: the drop band and the settle

Option 2A, keep the current drop band and the current settle. Cost: 6.6 of 50 parts start
outside the tray, which caps the stowed fraction near 0.87 before the teacher moves. Buys: no
work.

Option 2B, raise the tray wall to 40 mm. Cost: a tray change, and a jaw that has more wall to
hit on the way in. The sweep says the blockage does not rise, 32.0 to 30.6 at the current drop.
Buys: `outside` from 6.6 to 4.0 of 50. Interpretation: this is the cheapest ejection lever
measured, and it is not one of the two levers the question named.

Option 2C, take the long tray at a 40 mm wall, at the current drop band. Cost: a 120 by 340 mm
tray, which is a bigger change than option 2B and narrows the short axis by 10 mm. Buys:
`outside` from 6.6 to 0.4 of 50 and `stacked` from 32.4 to 30.0. This is the cell that fires
R-E.

Option 2D, lower the drop or fix it. Cost: the sweep says it is worse. `outside` rises to 10.6
or 20.4 of 50. Buys: less stacking, 32.4 down to 21.2 at the fixed drop.

Option 2E, lengthen the settle. Cost: run time. Buys: nothing measured. Settle 960 reads the
same as settle 480 in all 24 cells.

Recommendation: option 2B. A later tray change for another reason then carries option 2C. R-E
fired, and it fired through the tray and the wall. The drop band itself is not the lever, and
lowering it makes the ejections worse.

## 10. Reproduction

Every command below ran from the integration worktree, not from `main`. Run them from a
checkout that carries all three branches. The read's flags are the ones campaign `a` runs, in
`~/data/maker-arm-ms6/a/run_read.sh`.

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
    --out_dir ~/data/maker-arm-ms6/d/probe/wall_table
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode wall_table --shipped \
    --out_dir ~/data/maker-arm-ms6/d/probe/wall_table

# The driven reach band, far side at the script's own grid, then the near side.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode reach_band \
    --out_dir ~/data/maker-arm-ms6/d/probe/band_far
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode reach_band \
    --radii_mm 40,50,60,70,80,90,100,110,120,130,140,200 \
    --out_dir ~/data/maker-arm-ms6/d/probe/band_near

# The heap read. READ_FLAGS is what campaign a runs and what the corpus runs.
READ_FLAGS="--heap_k 50 --cycle_budget 50 --parts_split heldout --parts_seed 20260915 \
    --nearest_first_pick_order --hover_jitter_min 0.035 --hover_jitter_max 0.080 \
    --episode_draws --frame_jitter --no_cameras"

# After pull request #57 merges, --mode heap resolves the nearest-first flag, the frame
# jitter and the hover jitter band by itself, on top of the five arguments it already pins.
# --no_read_flags skips that block. Passing READ_FLAGS is exact before and after. Dropping
# one read flag needs --no_read_flags plus the rest, because leaving a flag out no longer
# drops it. mj_inspect_heap.py is not affected by any of this.

# The fixed teacher, one command per heap, n = 0, 1, 2.
MUJOCO_GL=egl uv run python scripts/mj_inspect_heap.py --no_video $READ_FLAGS \
    --run_seed 2026100n --layout_seed 2026101n \
    --out_dir ~/data/maker-arm-ms6/d/probe/base_readflags_n

# The bottom-open probe at the read's flags, shipped model then matched model.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    $READ_FLAGS --run_seed 2026100n --layout_seed 2026101n \
    --out_dir ~/data/maker-arm-ms6/d/probe/unmatched_n
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    $READ_FLAGS --matched_clearance --run_seed 2026100n --layout_seed 2026101n \
    --out_dir ~/data/maker-arm-ms6/d/probe/matched_n

# The isolation. Each run is the read's flags with exactly one of them dropped.
NO_NEAREST="--heap_k 50 --cycle_budget 50 --parts_split heldout --parts_seed 20260915 \
    --hover_jitter_min 0.035 --hover_jitter_max 0.080 --episode_draws --frame_jitter \
    --no_cameras"
NO_JITTER="--heap_k 50 --cycle_budget 50 --parts_split heldout --parts_seed 20260915 \
    --nearest_first_pick_order --episode_draws --frame_jitter --no_cameras"

MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --no_read_flags $NO_NEAREST --matched_clearance --run_seed 2026100n \
    --layout_seed 2026101n --out_dir ~/data/maker-arm-ms6/d/probe/iso_nonearest_n
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --no_read_flags $NO_JITTER --matched_clearance --run_seed 2026100n \
    --layout_seed 2026101n --out_dir ~/data/maker-arm-ms6/d/probe/iso_nojitter_n

# The shortest command that produces the no-band row, at the probe's own defaults.
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --no_read_flags --matched_clearance \
    --out_dir ~/data/maker-arm-ms6/d/probe/minimal_matched_0
MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode heap --no_video \
    --no_read_flags --out_dir ~/data/maker-arm-ms6/d/probe/minimal_unmatched_0

# The fixed teacher without the hover jitter band. It runs through mj_inspect_heap.py, so
# #57 leaves it alone.
MUJOCO_GL=egl uv run python scripts/mj_inspect_heap.py --no_video $NO_JITTER \
    --run_seed 2026100n --layout_seed 2026101n \
    --out_dir ~/data/maker-arm-ms6/d/probe/base_nojitter_n
#   base_defaults_n: the same teacher at its own default flag set, with no read flags. It
#   reads the same as base_nojitter_n on every heap.

# Every group above sums through one reader.
python3 scripts/heap_inspect_read.py ~/data/maker-arm-ms6/d/probe/matched_*/events.jsonl
```

The runs used the worktree's own virtual environment, with `PYTHONPATH` on the worktree's
`source` directory. A worktree then tests the code it carries, not the primary checkout's
editable install. The fixed teacher's campaign `b` column is at
`~/data/maker-arm-ms6/b/k50/fixed_0` to `fixed_2`, and `base_readflags_*` reproduces it.
