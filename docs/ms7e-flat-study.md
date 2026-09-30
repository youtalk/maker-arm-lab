# MS7e flat study: why the policy wastes cycles on plates and tiles

Date: 2026-09-30. Status: final for x86. The R15 row's aarch64 pair (draws 4 and 5) still reads.
All times are PDT.

At 06:50 the owner asked to resolve the flat plates in the remaining MS7e time. A later message set
the end to the early afternoon of 2026-09-30. The campaign ledger is
`~/data/maker-arm-ms7/e/README.md`, entries 08:20 to 10:36. The scripts and the data of the study
are under `~/data/maker-arm-ms7/e/flat/`. This report cites `docs/ms7e-widened-task.md` as "MS7e
report section N". Terms: a flat part is a plate or a tile. The teacher is the scripted oracle that
records the corpora. The policy is candidate E2, train_E2 150k. A cycle is one pick attempt on one
target part. The loop is the serving code between the policy and the arm, `core/loop.py`.

## 0. Summary

Flat parts are the largest cost of the MS7e policy. In the official read, e_E2_official, they use
1998 cycles for 1031 stows, 1.94 cycles per stow.

The study found the cause in the loop, not in the policy's grasp. The policy closes the jaws on the
flat part. Then the loop rule `snap_hold` opens them again as the lift starts. The rule keeps the
jaws open until the hold detector reads a hold. On a thin plate the fingers sink into the part, the
measured gap creeps, and the detector reads no hold.

The fix is the switch `--snap_closed` (feat/ms7e-snap-closed, PR #150). It ends the hold once the
measured jaw gap is under the part's width plus 2 mm. The diagnostic pairs, 4 seeds and 1000 parts
an arm, pooled:

| arm | flat stows per cycle | other stows per cycle | parts stowed of 1000 | cycles per stow |
| --- | --- | --- | --- | --- |
| base | 0.497 | 0.891 | 857 | 1.321 |
| fix | 0.750 | 0.925 | 903 | 1.145 |

The pairs are a diagnostic, not a read. The pre-registered R15 row reads the switch on the official
protocol. Its four x86 draws read 1000 parts an arm. The switch stows 925 against 912, at 1.135
against 1.242 cycles per correct stow. All three conditions of rule R13 hold. The aarch64 pair
still reads. The census also over-counts the flat losses. Most of the 350
flat `lifted_lost` cycles of the official read are stows. The waste lies in the 806 flat
`close_miss` and `held_no_lift` cycles.

The owner's toe pad does not help in the simulation. More finger friction reads 0.715 flat stows per
cycle against 0.750, not worth modeling. A soft pad reads 0.388, and the study rejects it.

With the fix, 47 of 275 flat parts stay out of their boxes. The part selector refuses them: 22 lie
in another class's box, and 18 were never a target. True slips out of closed jaws are rare, 13 of
229 flat cycles. The next levers are the teacher's wall band, a smaller wall margin and the failure
memory radius.

## 1. The evidence path

### 1.1 The official read's flat census

`flat/flat_census.py` splits the flat cycles of e_E2_official by cycle class. A close edge is one
step where the commanded jaw gap drops under the cycle's close threshold.

| cycle class | meaning | flat cycles | close edges per cycle, mean |
| --- | --- | --- | --- |
| `stowed` | at the cycle's end, the part is in its box | 727 | 2.01 |
| `close_miss` | the jaws close on nothing | 486 | 8.42 |
| `lifted_lost` | the part rises, then leaves the jaws | 350 | 3.33 |
| `held_no_lift` | the jaws hold the part, and no lift follows | 320 | 8.32 |
| `wrong_box` | the part goes into another class's box | 46 | 3.09 |
| other classes | `carrying`, `no_close`, `no_open` | 69 | |

A failed flat cycle closes the jaws 8.42 or 8.32 times on average, and a stowed one 2.01 times. So
in the failed cycles the policy closes again and again on one part.

### 1.2 The teacher against the policy

Six diagnostic runs read held-out heaps on run seeds 7000 and 7001, layouts 8000 and 8001, at
1acb329 on Machine A. P_held_a/b and P2_held_a/b serve the policy. T_held_a/b run the teacher.

| measure | teacher | policy |
| --- | --- | --- |
| flat cycles that stow | 155 of 157 | 0.39 to 0.41 stows per flat cycle |
| share of 3-step windows with a still gap | 0.85 on flat parts | 0.71 on flat parts (q10 0.29), 0.96 on bricks |
| squeeze command on a flat part of width w | w - 6.0 mm | w - 6.2 mm median, w - 8.7 mm at q10 |

The teacher stows almost every flat cycle. So the plates are not hard to grip in the simulation.

### 1.3 Who opens the jaws

The P2 runs record two gripper commands at each step: the raw policy command and the final command
after the loop. This recorder shows who opens the jaws.

| item, P2 runs | value |
| --- | --- |
| both fingers on the part by step 5 | 181 of 222 first flat closes |
| the loop forced the jaws open before a lift | 111 of 222, 46 steps after the close, as the lift starts |
| the policy's gripper command at that step | still closed, median 7.7 mm against a 15.3 mm threshold |

Only one loop rule raises the jaw gap: `snap_hold`, a part of `pick_snap`. `pick_snap` lowers the
served pick point onto the part. Meanwhile `snap_hold` keeps the jaws at the landing gap, the width
plus 8 mm. It holds while the policy's command is above the snap floor and no hold is read. The lift
raises the command above the floor. With no hold read, `snap_hold` then opens the jaws, and the part
drops.

### 1.4 Why the hold detector misses

The hold detector, `core/hold.py`, tests the measured jaw gap. It needs a still gap, less than
0.2 mm of change over 3 steps (`hold.STABLE`). A gap under `hold.MIN_GAP`, 3 mm, ends a hold. In
run P2_held_a, the loop opened 36 first flat closes. In 19 of them the detector never read a
hold, because the gap was not still. In 15 it read a hold that ended before the lift, because the
gap fell under 3 mm.

The gap creeps because the fingers sink into the plate. The simulation uses soft contact, which lets
two touching solids overlap a little. On a plate 3.2 mm thick and 7.8 mm wide, each finger overlaps
it by 1.9 to 2.5 mm. The measured gap then reads 2.9 mm, under the 3 mm floor. The policy also
squeezes deeper than the teacher. The loop's squeeze floor, `--squeeze_floor`, sets the closed
command from the perceived width. So the policy's command sits 6.2 mm under the width at the median
and 8.7 mm under it at q10. The teacher's command sits 6.0 mm under the width.

## 2. The census correction

The switch `select_at_release` ends the cycle at the release, while a released flat part still
settles. So the census classes the cycle `lifted_lost`, and the part then comes to rest in its box.
In e_E2_official, 727 flat cycles are classed `stowed`. But at the episode end, the gate counts 1031
flat parts in their boxes, and 379 flat parts not stowed. In P2, 23 of 25 policy releases in flat
`lifted_lost` cycles were over the part's own box. All 23 parts came to rest in it. So the 350 flat
`lifted_lost` cycles of the official read are mostly stows. The waste lies in the 806 flat
`close_miss` and `held_no_lift` cycles.

## 3. The fix

### 3.1 The diagnostic test

The ledger fixed the test and its bar at 08:20, before any number. `flat/snap_test_client.py` wraps
`scripts/mj_eval_heap.py`. With `SNAP_FIX=1`, `snap_hold` returns False once the measured jaw gap is
under the served width plus 2 mm. `SNAP_FIX=0` is the paired base. Each run reads the E2 official
protocol on 5 envs, 250 parts, with the layout seed at the run seed plus 1000. Seeds 7000 and 7001
ran on Machine A at 3e9ee7b, and seeds 7002 and 7003 on Machine B at 1acb329. 3e9ee7b is 1acb329
without the `--seen_seeds` flag, which is off by default. The bar pools the 4 pairs. The flat stows
per cycle must rise by at least 0.15. The other stows per cycle must fall by at most 0.03. The
pairs, from `flat/sum_all.txt`:

| run seed | arm | flat stowed | flat cycles | other stowed | other cycles | stowed of 250 | cycles per stow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7000 | base | 41/60 | 78 | 178/190 | 188 | 219 | 1.215 |
| 7000 | fix | 51/60 | 63 | 176/190 | 184 | 227 | 1.088 |
| 7001 | base | 80/110 | 159 | 126/140 | 152 | 206 | 1.510 |
| 7001 | fix | 88/110 | 119 | 128/140 | 145 | 216 | 1.222 |
| 7002 | base | 49/70 | 98 | 173/180 | 178 | 222 | 1.243 |
| 7002 | fix | 62/70 | 83 | 172/180 | 184 | 234 | 1.141 |
| 7003 | base | 22/35 | 51 | 188/215 | 228 | 210 | 1.329 |
| 7003 | fix | 27/35 | 39 | 199/215 | 217 | 226 | 1.133 |
| pooled | base | 192/275, 0.497 a cycle | | 665/725, 0.891 a cycle | | 857 of 1000 | 1.321 |
| pooled | fix | 228/275, 0.750 a cycle | | 675/725, 0.925 a cycle | | 903 of 1000 | 1.145 |

The test is SUPPORTED (09:04). The flat stows per cycle rise by 0.253 against a bar of 0.15, and the
other stows per cycle rise too. Every pair gains flat stows (+10, +8, +13, +5) and cuts its cycles
per stow. In each fix run the rule ended the hold on 1392 to 2552 steps.

### 3.2 The switch

`--snap_closed` is off by default and needs `--pick_snap`. Under the switch, the loop passes
`snap_hold` the measured jaw gap. A gap under the served row's width plus `SNAP_CLOSED_MARGIN` ends
the hold. The margin is 2 mm, an `INITIAL:` value. The landing gap is the width plus 8 mm, so a
descent keeps its hold.

Commit 221ede5 adds the rule to `core/loop.py`, with `tests/test_snap_closed.py`. Commit 29a36ef
takes the flag through `core/batch_loop.py`, `scripts/mj_eval_heap.py` and the gate, and into the
`ms7e` block and the cache key. The branch sits on main dbc6a8c, with 1433 tests and a clean lint.
With the switch off, every `snap_hold` call gets no gap, and a test shows a byte-identical payload.
A unit test reproduces the failure with a hold detector that never reads a hold. Off, it stows 2 of
3 parts. On, it stows 3 of 3 in 2 cycles.

## 4. The R15 row

Rule R15 (09:33) fixed the row before any number. The policy is E2, train_E2 150k, at 29a36ef. The
protocol is the official one: `read_e.sh` with DESK 8, the MS7e set and `--thin_snap
--wall_band_cont`. The seed is 20281900, draws 0 to 5, 1500 held-out parts an arm. Two TAGs read the
same draws: e_r15_base with the switch off and e_r15_snapc with `--snap_closed`. Draws 0 to 3 read
on x86, a_r1 and b_r1, and draws 4 and 5 on c_r1. So the rows pair draw for draw.
Both TAGs started at 09:39. For adoption, e_r15_snapc must meet the three conditions of rule R13
against e_r15_base:

- It reads at least as many correct stows.
- It reads no more than 0.02 cycles per correct stow above the base.
- It gains at least 8 stows or saves at least 0.02 cycles.

| row | TAG | correct stows of 1500 | Wilson 95 % | cycles per correct stow | plate | tile | against the base row | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | e_r15_base | x86: 912 of 1000, 0.912 | 0.893 to 0.928 | 1.242 | 146/180 | 63/85 | | the base |
| `--snap_closed` | e_r15_snapc | x86: 925 of 1000, 0.925 | 0.907 to 0.940 | 1.135 | 154/180 | 80/85 | +13 stows, -0.107 cycles | the three conditions hold on x86 |

The table holds the four x86 draws, 0 to 3, read at 10:30. The aarch64 draws 4 and 5 serve the
policy on the CPU and take hours. When that pair ends, the full-row verdict adds it. On x86 the tiles
gain the most, 63 to 80 of 85, and the plates gain 146 to 154 of 180.

## 5. The owner's toe pad

The owner proposed a rubber or foam pad on the finger tips. The simulation already grips at pad-like
contact. A finger-part pair takes the larger sliding friction, finger 1.0 or part 0.8 to 1.2. The
torsional coefficient of 0.005 m is about what a 4 to 5 mm rubber patch gives (mu_t ~ 2/3 mu r).

So two rows test what the simulation does not have yet. Both add to the snap fix, on the four seeds
of section 3.1. The row pmu sets the finger friction to 1.5. The row psoft adds a softer finger
contact: a `solref` response time of 0.02 s, mixed with the part's 0.005 s. To be worth modeling, a
pad must raise the pooled flat stows per cycle of the fix arm, 0.750, by at least 0.05. The other
stows per cycle must fall by at most 0.03 from 0.925. Two runs of one seed are not bit-exact: P_held
and P2_held read 43 and 48 of 60. The rows, from `flat/sum_pad_partial.txt`:

| arm | seeds | flat stows per cycle | flat parts stowed | other stows per cycle | stowed | cycles per stow | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fix | 7000 to 7003 | 0.750 | 228/275 | 0.925 | 903/1000 | 1.145 | the reference |
| pmu | 7000 to 7003 | 0.715 | 223/275 | 0.951 | 906/1000 | 1.137 | not worth modeling |
| fix | 7002, 7003 | | 89/105 | | | 1.137 | the reference |
| psoft | 7002, 7003 | 0.388 | 69/105 | 0.885 | 424/500 | 1.366 | rejected |

pmu moves the flat stows per cycle by -0.035 against a bar of +0.05. psoft cannot reach the bar even
with perfect Machine A pairs, so the campaign stopped its two Machine A runs at 09:51. A soft pad lets
the fingers sink further into a flat part. For the real arm, the pad is still the way to reach the
simulation's friction of 1.0 to 1.2. The simulation does not reward more friction than that.

## 6. What remains

### 6.1 The selector study

In the fix runs, 47 flat parts stayed out of their boxes. At the end of each episode, the selector
found no part to take (`no_candidate`). The study sorts the 47 by place and by the rules that refuse
them (`flat/select_study/f1_left.txt`). A part can meet more than one rule.

| where the part lies | parts | never a target | box filter | out of reach | full failure memory | raked |
| --- | --- | --- | --- | --- | --- | --- |
| in another class's box | 22 | 8 | 22 | 19 | 0 | 0 |
| on the tray floor or at a tray wall | 16 | 7 | 0 | 2 | 3 | 7 |
| on the desk | 9 | 3 | 0 | 5 | 2 | 0 |
| all | 47 | 18 | 22 | 26 | 5 | 7 |

The box filter refuses a part inside a box. The failure memory refuses a part after 3 failed cycles,
and it can inherit a neighbour's failures within 15 mm. A rake is the scripted push that moves a
part away from a wall. On the tray floor, the tray-wall grasp test refuses flat parts:
`grasp.wall_allows`, with `WALL_BAND_MARGIN` 5 mm, `INITIAL:`. On the desk, 6 of the 9 parts lie 6.2
to 13.8 mm from a desk box wall. Some parts lie under others.

The teacher leaves 16 flat parts on seeds 7000 and 7001, all tray-wall refusals at `JAW_HOLD_BAND`
8 mm. Its reads and the corpus waves ran without `--wall_band_cont`. A yaw sweep counts, of 180
yaws, those with no grasp as the wall rules relax (`flat/select_study/yaw_variants.txt`). The 1 mm
band of the continuous jaws moves the counts of three teacher parts. They go from 90 to 0, 180 to
90 and 180 to 180. A wall margin of 0 mm then takes the second part to 0. For one policy tile, the
count falls only from 127 to 111 over the whole sweep.

### 6.2 The true slips

True slips out of closed jaws are rare: 13 of 229 flat cycles in P2. Two came right after the
squeeze tightened. A part that tilts 20 degrees in the jaws turns out of them. The policy's carries
accelerate twice as hard as the teacher's, 12.7 against 6.1 rad/s2 at the median. But the
acceleration is equal in the slip cycles and the carried cycles, so it does not explain the slips.

### 6.3 The levers and their order

1. Adopt `--snap_closed`. The x86 pairs of the R15 row meet all three conditions. The aarch64 pair
   still reads.
2. Record the next corpus with the teacher's wall band, `--wall_band_cont`. The teacher's 16 left
   flat parts are all wall refusals at the 8 mm band.
3. Do not set `WALL_BAND_MARGIN` to 0 mm on this evidence. The row wm0 (10:35, `flat/sum_wm.txt`)
   adds it to the snap fix on the four seeds. It reads 0.761 flat stows per cycle against 0.750,
   under its bar of +0.05. All parts read 923 of 1000 at 1.118 against 903 at 1.145. That gain lies
   inside the rerun noise of one seed.
4. Study the failure memory radius, 15 mm, which lets a part inherit a neighbour's failures.

## 7. Defects found on the way

- The fleet sync lock race. `fleet.py sync` to two remote aliases in one call raced on
  `~/data/maker-arm-ms7/sync.bundle.lock`. The bundle step exited 128 for the second host. A sync of
  that one alias then passed.
- The orphaned first R15 launch. The first launch at 09:3x ran from a subshell that exited. The
  local fleet processes lost their parent, and this host sweeps such processes. After about 3
  minutes the campaign stopped both TAGs on all three hosts, removed their directories and
  relaunched them. It read no payload from the first launch.
- A stale number in MS7e report section 9. Step 3 says that the official read still has 350
  `lifted_lost` cycles on flat parts. Section 2 of this report shows that most of them are stows.
  The step needs a correction that points at the 806 `close_miss` and `held_no_lift` cycles.
