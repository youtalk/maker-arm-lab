# MS6 bare-desk heap: the ladder to a 0.80 stowed fraction

Date: 2026-09-18. The owner set the goal on this day: training data at a success rate of 0.80
or more, so the learning pipeline can start soon. The owner also relaxed three rules of the
MS6 design. If other shapes replace them, the thin parts can leave the catalog. The parts
need no tray or box. They can lie on the desk in front of the arm and to its left and right.
The jaws can carry more than one part at a time.

The branch is `feat/ms6-bare-desk-heap`. Every read below runs three heaps of 50 parts
through `scripts/mj_inspect_heap.py --no_video --num_envs 1`. The flags are the read flags of
`MS6 rework section 2`: `--heap_k 50 --cycle_budget 50 --parts_seed 20260915
--nearest_first_pick_order --hover_jitter_min 0.035 --hover_jitter_max 0.080 --episode_draws
--frame_jitter --no_cameras`. The held-out heaps add `--parts_split heldout --run_seed
2026100n --layout_seed 2026101n`. The train-split heaps add `--parts_split train --run_seed
2026200n --layout_seed 2026201n`. In both, n is 0, 1 or 2. The starting point's logs are at
`~/data/maker-arm-ms6/i/rA_{0,1,2}`. The rungs' logs are at
`~/data/maker-arm-ms6/j/<rung>_{0,1,2}` for the held-out heaps and `j/<rung>t_{0,1,2}` for the
train-split heaps. `scripts/heap_inspect_read.py` recomputes every number from those
`events.jsonl` files.

The metric is the stowed fraction: the parts at rest in the one bin at the episode's end,
over the parts dropped. A rung reads against the rung below it with a Wilson 95 percent
interval.

## 0. Read this first

The goal is met. Rung b4 stows 140 of 150 held-out parts, 0.933, Wilson [0.882, 0.963]. It
stows 141 of 150 train-split parts, 0.940, Wilson [0.890, 0.968]. It spends 1.05 pick cycles
per stowed part. The starting point stowed 95 of 150, 0.633. Four levers carried the ladder. The
teacher no longer pushes. The drop spreads over the whole scatter zone, in batches. The
catalog has no round bricks and no wedge slopes. A backstop wall stands behind the bin. Rung b5 adds the held-part carry
and reads the same 140 and 141.

The training corpus recorded from rung b5 into `~/data/maker-arm-ms6/k/corpus`: eight
processes of five train-split heaps of 50 parts each, cameras on. It stowed 1905 of 2000
parts, 0.9525. Section 6 gives the
conversion and training commands. One converter defect had to go first. It dropped almost
every heap episode.

## 1. The starting point: the bare desk at 0.633

A previous session on this day shipped two commits before this ladder. `9f49908` removed the
plates and tiles from the catalog and added five tall bricks and three round bricks.
`f93e010` removed the round dish. The parts are dumped on the bare desk over the old 0.088 m
disc. The teacher gates its candidates on the arm's reach band (`layout.in_reach`), and the
wall model of `core.grasp` is gone. Its read, `i/rA`, stowed 95 of 150 held-out parts,
0.633, Wilson [0.554, 0.706].

The `events.jsonl` logs of that read named the levers:

- All three episodes ran to the 30900-step timeout with 17 to 19 parts left. Pushes took
  34 percent of every episode's control steps, 118 pushes against 119 picks.
- The forced picks those pushes delayed stowed 20 of 21. A forced pick takes the top-ranked
  grasp although a neighbor overlaps a landing spot.
- `round_1x1` lifted 15 of 44 times, the worst part of the heap. `slope_1x1` and
  `slope_2x1` have one stud along the run. They are pure wedges whose front edge has no
  height, and they lifted 26 of 47 times. Bricks lifted 68 of 83.
- The stowed parts over the pick cycles already read 0.80. The clock and the pushes capped
  the read, not the grasp.

That session also left one uncommitted fix: the desk is a hard contact surface. A 3 g part
under the jaws' 2.9 N grip sank 1.9 mm into the desk, and 13 mm under 20 N. It ships here as
`665de1d`, and section 5 records a defect in it.

## 2. The ladder

| rung | commit | contents | held-out stowed | fraction | Wilson 95% | train-split stowed |
| --- | --- | --- | --- | --- | --- | --- |
| rA | f93e010 | the bare desk, the previous session's read | 95/150 | 0.633 | [0.554, 0.706] | not read |
| b1 | 7e0e09d | hard desk, no 1x1 round, no wedges, a slope closes across its sides | 103/150 | 0.687 | [0.609, 0.755] | not read |
| b2 | 64dc35f | b1 and no pushes: a blocked scan is a forced pick at once | 109/150 | 0.727 | [0.650, 0.791] | not read |
| b3 | 7d8ec8a | b2 and the drop spread over the whole scatter zone | 118/150 | 0.787 | [0.714, 0.845] | 120/150 = 0.800 |
| b4 | 1c7b34b | b3 and the four losses the b3 read named (section 3.4) | 140/150 | 0.933 | [0.882, 0.963] | 141/150 = 0.940 |
| b5 | 1b2468c | b4 and the LIFT accepts any part the jaws hold | 140/150 | 0.933 | [0.882, 0.963] | 141/150 = 0.940 |

b1 changed the catalog, so the held-out set changed between rA and b1. rA is therefore a
starting point and not a rung of the same set. From b1 to b3 the three held-out heaps hold
the same eight part types: five bricks, one round brick and two slopes. From b4 on, the
round brick is gone and the set holds five bricks and two slopes.

The cycles per stowed part fell along the ladder: 1.233 at b1, 1.229 at b2, 1.144 at b3 and
1.057 at b4. Every b3 and b4 episode ended before the clock ran out.

## 3. The rungs

### 3.1 b1: the parts and the grasp axis

The catalog drops `round_1x1`, `slope_1x1` and `slope_2x1`. It never ships `round_4x4`,
whose 31.8 mm grasp closes on one side first (`docs/ms6-matched-jaw-opening.md` section 5).
It keeps the 2x2 round brick in one and two brick heights. Every retired part stays
constructible through `parts.by_name`, so an older shard replays.

A slope offers one closing axis, body y, across its two flat side faces. Closing along x put
one jaw on the slope face and squeezed the part out. The narrower-width tie of the rank chose
exactly that axis on the wedges. `Part.closing_axes` carries the rule, and
`grasp.rank_grasps` takes one axis tuple per part, so every box part keeps all three axes.

The rung reads 0.687 against 0.633. The intervals overlap. The three episodes still ran 51
pushes, and two of the three ended at the cycle budget or the timeout.

### 3.2 b2: no pushes

`MAX_PUSHES_IN_A_ROW` goes from 3 to 0. When no grasp clears, the teacher takes the
top-ranked grasp at once. The push machinery stays, and its test sets the constant back to 3
to drive it.

The rung reads 0.727 against 0.687, with 0 pushes and 48 forced picks over the three heaps.
Every episode ended DONE with time on the clock: 17423, 18661 and 24930 steps against a
budget of 30900.

### 3.3 b3: the whole scatter zone

The heap was dumped on a disc of radius 0.088 m in front of the arm, 243 cm2 for 50 parts.
The drop now draws uniformly over the six-brick scatter zone, `SCATTER_X` by `SCATTER_Y`
clipped at `SCATTER_MAX_RADIUS`, about 470 cm2. That is the zone Task D3 drove and the zone
the overhead camera frames, so no reach or camera number moves. `layout.in_reach` gains the
zone's y span plus a slack as a bound.

The rung reads 0.787 against 0.727, Wilson [0.714, 0.845], with 42 forced picks. On the
train split it reads 120 of 150, 0.800. Every episode ended DONE at 17094 to 20677 steps.

The 32 unstowed held-out parts fell into three groups, read off the last scan of each
episode:

- 16 were out of reach. Six of them ended on the desk at x 0.50 to 0.69 m after one pick.
  Every one of the six ended RELEASE with the `budget` outcome, already 5 to 15 cm past the
  bin's far wall. The tool holds 80 degrees over at RELEASE and the jaws open in a
  near-vertical plane. The part slides off the lower plate along the tool axis, toward +x,
  and clears the 0.06 m wall. Two round bricks rolled off the desk during the settle and two
  rolled under the arm. Six sat at 0.280 to 0.291 m from the base before a single pick.
- 11 were slopes with no closing axis. A 1xN slope is 7.8 mm wide under a 9.6 mm height. It
  comes to rest on a side face, so body y points up.
- 5 were late arrivals: parts in the bin and still moving at the moment RETREAT graded
  them. Four of the five were round bricks rolling.

`round_2x2` lifted 21 of 43 times on the held-out heaps, and `round_2x2x2` 5 of 15 on the
train split.

### 3.4 b4: the four losses

Each group above gets a change.

- `SLOPE_AXES` gains body z. On its side, a slope is gripped between its flat bottom and its
  flat top. Upright, it still offers y alone, because `core.grasp` keeps only the axes near
  the jaw plane.
- The round family retires (`parts.RETIRED_FAMILIES`). Studs are visual only, so nothing
  stops a cylinder rolling. The catalog is 29 parts and the held-out split 7. Collision
  studs are the way back.
- The bin's far wall becomes a 0.18 m backstop (`layout.HEAP_BIN_BACKSTOP`), above
  `RELEASE_Z`, so the slide hits it and drops into the bin. The other three walls and the
  grader's z band do not move. A heap release setback of 0.035 m was tried with it and
  dropped. Driven, that setback opened the jaws 8 mm past the matched ceiling at RELEASE,
  with either wall height. The six-brick setback of 0.02 m stayed at the ceiling with both.
- `HEAP_DROP_INSET` goes from 0.01 to 0.02, and `layout.in_reach` follows the driven wedge.
  Past 25 degrees of azimuth, a part out to 0.295 m is still a target. The D3 table tracked
  0.295 m at 25 degrees and 0.308 m at 30 degrees. Campaign d stowed at 0.30 and 0.32 m at
  plus and minus 40 degrees. The drop never places a part there.

Two more changes come from the branch review (section 5). The drop releases the parts in
batches of five, and `in_reach` gains an azimuth cut.

The rung reads 0.933 against 0.787. Its interval, [0.882, 0.963], sits clear above b3's. On
the train split it reads 0.940. Bricks lifted 111 of 134 times and slopes 36 of 37. The 10
unstowed held-out parts are 8 out of reach and 2 late arrivals. Two of the six episodes
ended at the cycle budget with 44 and 47 stowed, and no episode ran out of time.

### 3.5 b5: the LIFT accepts any part the jaws hold

On the round-dish heaps of 2026-09-17, 16 of 154 LIFT reports declared a miss with a
non-target part in the jaws. That is 10.4 percent, each 100 to 144 mm above the desk. REOPEN dropped
every one back. With one bin for every part, a held part is worth carrying. When the target
did not rise, `OracleRig._lift_result` now looks for another part within `LIFT_MAX_XY_ERROR`
of the pick point and above `LIFT_MIN_RISE`, and `HeapOracle.adopt_held_part` re-points the
cycle at it. `held_swaps` counts it. The six-brick oracle keeps its gate:
`accepts_any_held_part` is False there.

The rung reads the same 140 and 141 as b4. The swap fired 3 times in the 300 parts, all on
the held-out heaps: two episodes gained one part and one lost one. Under the
pre-registered rule, a rung that reads at the rung below it stays.

## 4. The train-split reads

The corpus records the train split, 22 part types against the held-out 7. The stowed
fraction the learning pipeline sees is therefore the train-split one. b3 read 120 of 150 there and b4
141 of 150. The nine unstowed b4 parts on the train split are 4 out of reach, 3 late
arrivals and 2 candidates left at the cycle budget. Bricks lifted 119 of 134 times there and
slopes 28 of 34.

## 5. Defects found on the way

A read-only review of the branch at `7d8ec8a` found two defects and one measured risk.
`6a0c799` and `1c7b34b` fix all three.

1. The desk's hardness rode on a geom priority. MuJoCo hands the pair the whole contact set
   of the higher-priority geom. Every part-desk pair therefore read condim 3 instead of
   `BRICK_CONDIM` 4. It also read the desk's friction 1.0 instead of the per-episode draw of
   `mj/env.py`. That held in the six-brick task too. Measured at brick friction 1.2: the pair read
   dim 3 and friction 1.0 on the branch, against dim 4 and 1.2 on main. A `solmix` weight
   of 1e6 pulls the pair's solref and solimp to the desk's values and touches nothing else.
   The pressed-part test now pins the pair's condim and friction.
2. `layout.in_reach` cut the annulus only on |y| and an upper x, so the half-plane behind the
   arm read as reachable: (-0.25, 0.0) passed. A measured drop settled a part at
   (-0.075, 0.203), 3 mm from a target at azimuth 112 degrees. `in_reach` now cuts at the
   zone's corner azimuth plus 5 degrees.
3. All 50 parts were written into the air at once and interpenetrated. The solver flung
   them apart, and 8 of 150 settled out of reach, up to 80 mm out. An A/B ruled the desk
   contact out as the cause. Parts now drop in batches of five, 60 physics steps apart. Each
   batch's points are at least the two half diagonals plus 5 mm apart
   (`layout.sample_heap_batch`, largest parts first). 300 catalog batches place with no
   overlap. Batches of ten did not: 22 of 300 fell back to an overlapping draw.

The review also found stale text in the README's heap paragraph and in a probe comment.
Section 8 lists what it checked and found clean.

Two more defects surfaced in the reads themselves.

4. `scripts/mj_inspect_heap.py` read `stowed_at_rest` after the step that ended the
   episode. That step also resets the env, so an episode that ended on the env's own
   termination read 0 there. The count is now read every 30 steps before the step.
5. With the whole half diagonal as the drop inset, a 4x8 brick had a sliver 5 cm wide to
   land in, so two of them did not keep apart. The inset is now a quarter of the half
   diagonal. The zone's edge bounds the part's centre and is not a wall.

## 6. The learning pipeline

A heap shard recorded on this branch converts and loads. One converter defect had to go
first. `lerobot_stack/convert_to_lerobot.py` skipped every episode whose `success` attribute
is False before the positive-actor cut ran. A heap run records every episode on purpose
(design H7) and marks each pick cycle through `obs/actor`. A 50-part episode sets `success`
only after all 50 parts stow. Measured on a 2-episode shard: 8 cycles kept, and 15 with the
attribute forced true. `fcbb5c2` on `feat/ms6-heap-converter` drops that filter under
`--dagger`, reports `read` and `skipped` on the `CONVERT_OK` line, and adds the `camera_box`
context. That context is the overhead camera pose followed by the destination box pose, as
one 9-wide `observation.environment_state`. The same shard converts to 15 episodes and 6059
frames with `--split_cycles --dagger --context camera_box`.

The commands, from the repository root:

```bash
# record: PROCS processes of ENVS envs, one 50-part train-split heap per env, cameras on
WT=~/src/maker-arm-lab-read-b5 PROCS=8 ENVS=5 K=50 bash ~/data/maker-arm-ms6/k/run_corpus.sh
# convert, keeping the cycles that stowed
lerobot_stack/.venv/bin/python lerobot_stack/convert_to_lerobot.py \
    --shards ~/data/maker-arm-ms6/k/corpus/proc_*.hdf5 --root ~/data/maker-arm-ms6/k/dataset \
    --split_cycles --dagger --context camera_box
# train
DATASET_ROOT=~/data/maker-arm-ms6/k/dataset OUT=~/data/maker-arm-ms6/k/train/h1 \
    STEPS=600000 BATCH_SIZE=8 JOB_NAME=act-heap ACTION_SPACE=absolute bash lerobot_stack/train_act.sh
```

Three facts about the pipeline, verified in code. The target marker rides inside the
overhead image, so no flag selects it. `obs/images/overhead_depth` is recorded and not
converted. `--dagger` requires `--split_cycles` and refuses `--lead_keep`.

The corpus itself read above the ladder. Eight processes recorded 40 episodes of 50
train-split parts each with cameras on, at the read flags, in about 100 minutes of wall
clock. `scripts/heap_yield_read.py` sums the sidecars to 1905 of 2000 parts stowed, 0.9525,
Wilson [0.942, 0.961], at 1.036 cycles per stowed part, with 51 parts out of reach. Bricks
stowed 1642 of 1717 attempts and slopes 246 of 257. The worst episode stowed 34 and eleven
stowed all 50. The shards hold 42 GB under `~/data/maker-arm-ms6/k/corpus`.
The conversion keeps 1921 pick cycles from the 40 episodes, 783279 frames at 30 Hz, and reads
`CONVERT_OK episodes=1921 total=1921 read=40 skipped=0`. The dataset is 3.4 GB under
`~/data/maker-arm-ms6/k/dataset`, with `action` (7,), `observation.state` (7,), the wrist and
overhead videos, and `observation.environment_state` (9,) from `--context camera_box`. The
ledger `train_act.sh` asks for is `~/data/maker-arm-ms6/k/ledger.json`, with a target of
1921. A 100-step training smoke on that dataset prints `TRAIN_OK` at about 40 steps per
second with a batch of 8, so a 600k-step run takes about four hours. The launchers are
`~/data/maker-arm-ms6/k/run_corpus.sh`, `convert.sh` and `train.sh`. `train_act.sh` finds its
venv beside itself, so a worktree needs `lerobot_stack/.venv` linked to the main checkout's.

## 7. Testing

779 tests pass on the branch and ruff is clean. New CPU tests pin the retired parts and
their replay. They pin the slope's closing axes upright and on its side. They pin the reach
wedge, the azimuth cut, and the backstop wall's height and clearance from the tool. They pin
the batch sampler's separation and its escape, and the no-push forced pick. They pin the
held-part swap on a heap and its refusal on the six-brick oracle. New driven MuJoCo tests pin the part-desk pair's condim and
friction under the hard desk, and the spread of the drop across the zone. The release test
and the lone flat-part test are unchanged, and the bisect of section 3.4 ran their bodies.

## 8. What the review checked and found clean

No stale caller of any removed symbol in `source/`, `scripts/`, `tests/`, `lerobot_stack/`,
`README.md` or `docs/`. The `core/grasp.py` signature changes reach every call site. The
closing-axis tuples index the same part order as the sizes. The slope hull's flat faces are
the two y faces, so body y is a closing axis. The six-brick task is untouched outside the
desk contact, which section 5 fixed. `sample_heap_point` cannot loop: the lowest acceptance
rate over the catalog and the run draw's `bin_x` range is 0.50.
`tests/test_report_unverified_constants.py` and `tests/test_spec_citations.py` pass.

## 9. Open items

- The round bricks return only with collision studs, or another stop against rolling.
- The push and tip machinery stays in `core/heap_oracle.py` behind two constants.
  `grasp.clamp_to_reach` clamps the radius only and does not apply `in_reach`'s azimuth and
  y cuts, so a return of either needs that clamp widened first.
- `HEAP_CENTER` and `HEAP_RADIUS` remain for the push heading's bias and nothing else.
- The flat families are constructible for replays and never in the catalog. Their lift is
  marginal. The lone-plate driven test passes on its fourth to ninth attempt. Which attempt
  depends on scene details that do not touch the plate.
- Two b4 episodes ended at the 50-cycle budget with parts left. A budget above the part
  count lets the teacher retry them.
