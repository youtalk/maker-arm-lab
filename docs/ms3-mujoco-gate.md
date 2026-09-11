# MS3 — MuJoCo first: the port, the re-baseline, and the first two levers

Milestone: MS3. Spec: `2026-09-10-ms3-mujoco-design.md`. That spec supersedes the execution
order of `2026-09-10-ms3-recipe-transplant-design.md` and keeps its gate, decisions, seeds,
instruments and read rules. Executed 2026-09-10 to 2026-09-11 on the RTX 5090 host (MuJoCo
3.13, LeRobot 0.5.1, maker-arm-rs at its `feat/kinematics` merge). Every number below is read
off a file named beside it. The running record is `~/data/maker-arm-ms3/README.md` on that
host, and this report was assembled from it with `~/data/maker-arm-ms3/report_tables.py`.

## 1. Summary

**The MuJoCo port reproduces the Isaac gate number.** The MS2 recipe re-run on MuJoCo scored
**55/720 = 0.076 per-brick**, Wilson 95% [0.059, 0.098] (section 3.6). The criterion is
**≥ 0.70**. The recipe is a 1000-episode success-only corpus of the staircase oracle with
randomized pick order. It trains plain ACT for 600k steps, selects by rule L3, and gates on
120 official episodes. Isaac's MS2 read 52/720 = 0.072 [0.055, 0.093] on the same recipe and protocol.
The two simulators agree to within the interval. The levers measured on MuJoCo from here read
against 0.076, and the simulator switch does not confound them. The gate **FAILED**, as the
design expected of a re-baseline. Corpus to gate took 11.7 hours. The Isaac campaign was
budgeted at three GPU days.

Three things landed beside the re-baseline, each on a pre-registered rule:

- The planner moved to Rust (section 4). The oracle's global IK spent about 80 % of its time
  in torch dispatch overhead. `maker_arm_rs.Kinematics` runs the same solver 200 to 300 times
  faster with parity at 1e-12. One six-brick episode fell from 88 s to 14 s.
- The smooth via-point oracle (section 5). The waypoint staircase became a minimum-jerk walk
  with a transport-speed scale. The speed ladder adopted 4.0. The cycle fell from 815 to 484
  control steps per brick at yield 0.98, and the commanded acceleration p90 from 14.6 to 3.3
  rad/s².
- The recipe corpus (section 6). Corrective approaches (`hover_jitter`) and the lead trim were
  ported. The yield probes adopted `ns2.0 sp0.2 induce0.12 hover_jitter 0.005-0.035 rpo` at
  yield 0.92. Ten lanes generated 1209 source episodes (7263 per-brick sub-episodes) in 1.6
  hours. Its training, ladder and gate are in section 6.4.

## 2. The port

### 2.1 What was built

`source/maker_arm_lab/mj/` (PR #18, stacked on #17) has three modules. `scene.py` builds an
`MjSpec` from the vendored URDF and `core.layout`: desk, shelf, 20 bin boxes, six free bricks,
two cameras, lights, actuators, armature, and gravity compensation. `env.py` implements the
duck-typed environment surface that `OracleRig`, the batch loop and the eval loop read. It
holds one `MjModel` and `MjData` per env, steps them in lockstep, auto-resets a done env, and
passes every arm target through the Rust clamp. `recorder.py` writes the HDF5 shards that
`convert_to_lerobot.py` reads. The batch loop and the rollout loop moved into
`core/batch_loop.py` and `core/eval_loop.py`, which both backends share. `generate_dataset.py`
and `run_gate_eval.py` gained `--backend {isaac,mujoco}`.

Five physics settings are not MuJoCo defaults. Each cost a failed oracle run to find:

- the `implicitfast` integrator, because Euler let the jaw actuator's explicit damping
  oscillate,
- 2 ms joint-limit stiffness, because the 20 N jaw pushed 9 mm past the stops at the default,
- no robot self-collision, because the finger hulls stalled the jaws 27 mm short,
- 5 ms contact stiffness, because the default sank the jaws 5 mm into a 3 g brick (5 ms gives
  a 6.1 mm stall at 6.1 N per jaw, against Isaac's 7.4-7.7 mm at 7.6 N),
- a jaw target ramped at the URDF's 0.1 m/s.

### 2.2 Parity checks (design section 3)

| check | bar | result |
| --- | --- | --- |
| FK identity, `grasp_center` under MuJoCo against `GlobalIK.fk` | ≤ 1e-6 m and rad | 1e-16 at HOME and 20 random joint vectors |
| GRASP and ORACLE | pinch and stow | the unchanged `OracleRig` stows 2/2 and 6/6 bricks with 0 retries, in 4334-4360 control steps per six-brick episode (Isaac 4316-4436) |
| ORACLE yield probe (`probes/p1`, 50 attempts, 12 envs, cameras on, MS2 recipe) | success ≥ 0.70, clamp ≤ 0.01 | **1.000**, clamp 0.0, no time-outs or abandons, 87.0 s/episode per process (Isaac 128 s/episode for the whole batch) |
| converter | `CONVERT_OK`, count equal | one cameras-on episode gave `CONVERT_OK episodes=1` |
| training smoke | `TRAIN_OK` | 100 steps on lane 0's first 24 episodes, 29 steps/s |
| eval smoke, the MS2 Isaac checkpoint on MuJoCo (`eval/smoke_ms2ckpt`) | `EVAL_OK` | 0/72 per-brick, first close 282-573 steps, clamp fraction 0.387 (cross-simulator, reported only), 27.4 s/episode at 12 envs |

The CI failed at collection on the ubuntu runner, which has no libEGL. Pytest now runs with
`MUJOCO_GL=disable` there, and the one render test skips itself without a GL context.

### 2.3 Throughput, and why the corpus runs as lanes

One batch process is one core. The py-spy profile of the Isaac datagen put 82 % of the time
in the planner's finite-difference IK. On MuJoCo the same planner bounds a process at 87 s per
episode, whatever the env count. The corpus therefore ran as ten parallel lanes. Each lane is
one `generate_dataset.py` supervisor with one dataset root. Its seed is `20261007 + 100 ×
lane`. LeRobot's `aggregate_datasets` merges the lane datasets. The lane ledgers are
concatenated. Under the ten-lane load a lane runs at 133-193 s per attempt, because SMT
siblings share the 12 physical cores.

## 3. The re-baseline (design section 4)

### 3.1 Corpus (`t1/`)

Ten lanes × 5 runs × 25 attempts of the MS2 recipe: `--randomize_pick_order`, the staircase
oracle, `noise_scale 1.0`, start-pose offset 0.05. It ran from 19:53 to 01:48 PDT.

| lane | seed | runs | exported/attempts | rate | time_out | watchdog | abandons | clamp | hover_lowered | s/attempt |
|---|---|---|---|---|---|---|---|---|---|---|
| lane_0 | 3081137816 | 5 | 116/125 | 0.928 | 0 | 8 | 9 | 0.0e+00 | 15 | 149 |
| lane_1 | 3081237816 | 5 | 108/125 | 0.864 | 0 | 14 | 17 | 0.0e+00 | 14 | 144 |
| lane_2 | 3081337816 | 5 | 114/125 | 0.912 | 1 | 8 | 10 | 4.6e-06 | 10 | 152 |
| lane_3 | 3081437816 | 5 | 116/125 | 0.928 | 0 | 8 | 9 | 1.6e-06 | 17 | 157 |
| lane_4 | 3081537816 | 5 | 117/125 | 0.936 | 0 | 5 | 8 | 1.5e-06 | 10 | 162 |
| lane_5 | 3081637816 | 5 | 119/125 | 0.952 | 0 | 4 | 6 | 3.0e-06 | 15 | 167 |
| lane_6 | 3081737816 | 5 | 119/125 | 0.952 | 0 | 5 | 6 | 0.0e+00 | 20 | 165 |
| lane_7 | 3081837816 | 5 | 119/125 | 0.952 | 0 | 5 | 6 | 0.0e+00 | 15 | 154 |
| lane_8 | 3081937816 | 5 | 120/125 | 0.960 | 0 | 3 | 5 | 1.5e-06 | 16 | 158 |
| lane_9 | 3082037816 | 5 | 115/125 | 0.920 | 0 | 6 | 10 | 0.0e+00 | 13 | 155 |
| **all** | | | **1163/1250** | **0.930** [0.915, 0.943] | 1 | 66 | 86 | | 145 | |

Merged: `MERGE_OK lanes=10 episodes=1163 frames=4978803`, 20 GB. The 66 watchdog abandons are
5.3 % of attempts (MS2 had 56 in 1150). They include a false positive found later, section
5.2.

### 3.2 Training

`train_act.sh` ran 600k steps with checkpoints every 25k, wandb `maker-arm-lab` /
`ms3-act-t1-600k`. It took 4.6 hours (01:49 to 06:27 PDT): about 17 steps/s while the recipe
lanes ran, 50 steps/s after. The final loss was 0.046. Loss is a liveness instrument only
(MS2 report section 4).

### 3.3 Selection ladder (rule L3)

Fifteen rungs from 250k to 600k every 25k ran on draws 0 to 2 of seed 20261017, 36 episodes
and 216 bricks per rung. Four rungs ran at a time on separate policy-server ports. The ladder
took 49 minutes.

| rung | per-brick (draws 0-2, 216 bricks) |
|---|---|
| 250k | 0/216 = 0.000 |
| 275k | 8/216 = 0.037 |
| 300k | 13/216 = 0.060 |
| 325k | 13/216 = 0.060 |
| 350k | 13/216 = 0.060 |
| 375k | 0/216 = 0.000 |
| 400k | 0/216 = 0.000 |
| 425k | 8/216 = 0.037 |
| 450k | 31/216 = 0.144 **L3 candidate** |
| 475k | 21/216 = 0.097 |
| 500k | 8/216 = 0.037 |
| 525k | 11/216 = 0.051 |
| 550k | 21/216 = 0.097 |
| 575k | 25/216 = 0.116 |
| 600k | 12/216 = 0.056 |

The ladder is non-monotone, as MS2's was. The candidate's neighbours read 0.037 and 0.097. By
the rule's definition it is not an isolated spike.

### 3.4 Check step

The top two rungs ran on all five draws of the selection seed, 60 episodes and 360 bricks
each: 575k 46/360 = 0.128, 450k 45/360 = 0.125. The ladder spike of 450k regressed to the
mean on draws 3 and 4 (14/144). The rule gates the higher checked rate: **575k**.

### 3.5 Official gate, evaluated once

The gate command was `run_gate_eval.py --backend mujoco --threshold 0.70 --processes 10
--num_envs 12 --seed 20261027 --n_action_steps 1 --temporal_ensemble_coeff 0.01`. The
checkpoint was `train/act_t1_600k/checkpoints/575000`. Five draws ran at a time, then one
aggregating pass. It is
pure policy. The `--unstick` supervisor of D1 is not ported, and MS2's 0.072 was measured the
same way. Results are in `eval/official/gate_results.json`.

| | MuJoCo MS3 | Isaac MS2 |
| --- | --- | --- |
| per-brick | **55/720 = 0.076**, Wilson [0.059, 0.098] | 52/720 = 0.072, [0.055, 0.093] |
| episodes with any brick | 41/120 (histogram 0: 79, 1: 29, 2: 10, 3: 2) | 42/120 |
| never closed the jaws | 45/120 | 3/120 |
| first close, median (p10, p90) | 563 (289, 1829) | 1249 (476, 2792) |
| clamp fraction | 0.168 | 7.4e-4 |
| wrong bin | 1 of 59 releases | 13 of 90 held |
| wall | 24.5 s/episode at five concurrent draws | 25 s/episode |

### 3.6 Reading the gate

The factor account of the recipe design reads ① engagement 427/720 = 0.593, ② first-attempt
accuracy 23/427 = 0.054, ③ delivery 55/23 = 2.39. A delivery over one means the pinch grader
misses holds. `PINCH_MIN_FORCE` and `PINCH_MIN_STALL` were tuned on Isaac's 7.6 N grip, and
MuJoCo grips at 6.1 N. So 55 bricks were stowed by grasps, and only 23 of those grasps were
graded as holds. Factors ② and ③ are therefore not comparable across simulators. The product
is comparable, and it is the gate number. Two MuJoCo-specific reads travel into the recipe
gate. The policy never closes in 37 % of episodes (Isaac 3 %). Its commands are clamped on
17 % of steps (Isaac under 0.1 %). Both say the MuJoCo policy fails by not acting. The Isaac
policy failed by acting and missing.

## 4. The planner in Rust (maker-arm-rs PR #5, maker-arm-lab PRs #20 and #21)

Measured on the torch `GlobalIK` before the port: one Levenberg-Marquardt iteration cost 0.5
ms, of which the batched 12-probe Jacobian was 0.22 ms and one residual 0.21 ms. A seeded
solve cost 3.3 ms. `solve(restarts=24)` cost 1.18 s, and every HOVER plan and every
`solve_path` fallback pays it. Moving the Jacobian alone measured at under 2x, so the owner
chose the whole solver. `maker_arm::kinematics` holds FK over a caller-supplied chain and the
six-scalar grasp residual. It also holds the central-difference Jacobian, the LM refine with
the lab's damping schedule, branch selection, and the continuity-constrained path walk. Python sees it
as `maker_arm_rs.Kinematics`. `GlobalIK` parses the URDF, draws the fixed-seed restart spread
with torch, and converts the dataclasses.

Parity against the torch solver on the Maker Arm chain: FK agrees to 1.1e-16 m. The Jacobian
differs by 3.4e-8 at the median and 5.3e-7 at most, which is finite-difference rounding.
Refine from seeds within 0.05 rad converges on both sides 300/300 with |dq| under 1.2e-12.
`solve(restarts=24)` returns the same branch on 100 of 100 random targets with |dq| under
1.4e-13. Speed on one core: the 24-restart solve fell from 1.18 s to 8-9 ms. The seeded solve fell
from 3.3 ms to 0.03 ms, and FK from 0.17 ms to 0.008 ms. End to end, one six-brick MuJoCo episode
at one env fell from 88 s to 14.1 s with cameras and 6.7 s without. The renders, not the
planner, are now the ceiling. The lab suite (498 tests, the stub-rig oracle replays included)
passed on the Rust-backed solver. A new test pins the 24-restart solve under 0.5 s. One
trap for future parity scripts: `torch.tensor(list_of_floats)` is float32 and fakes a 1e-7
mismatch.

## 5. The smooth via-point oracle (PR #22; design section 5 item 1)

### 5.1 What changed

One six-brick MuJoCo episode of the staircase oracle took 4419 steps: HOVER 865, DESCEND 672,
CLOSE 270, LIFT 678, RAISE 181, EXTEND 738, RELEASE 119, RETREAT 656, SETTLE 240. The commanded
target stood still on 46 % of frames (dwell 2). Its acceleration read p50 6.1, p90 14.6, max 42
rad/s². The cycle was set by the walking pace (3 mm per two steps, 45 mm/s), not by dwells.

The walk is now one C² time law per leg: the minimum-jerk quintic over the leg's waypoint
polyline (`core/trajectory.py`), sampled every control step, at rest at both ends.
`leg_resolution` keeps the waypoint count for the IK continuity cap. It returns the walk
length from the old pace times the episode's gain draw times `--transport_speed`. The scale
applies to HOVER, LIFT, RAISE, EXTEND and RETREAT, and never to DESCEND, which is stall-gated
and grasp-critical. An acceleration cap of four times the peak speed per second binds only on
short legs. Holds, phase budgets and `core.oracle` are unchanged. `RUN_STATS` gained the
probe's reads: control steps per episode, and the commanded target's static-frame fraction
and acceleration quantiles.

### 5.2 A watchdog false positive

The watchdog abandoned the first smooth episode on its fifth HOVER. Trim (max 0.0001 rad) and
tracking lag (max 0.019 rad) were not the cause. The leg's end pose sat 0.0008 rad from
`link_003_joint`'s stop, which the solver accepts (margin ≥ 0). The minimum-jerk deceleration
kept the arm inside the 0.05 rad watchdog guard for 29 steps. The staircase crossed that band
in about 15 steps. When the arm is off its commanded target by 0.02 rad or more
(`WATCHDOG_TRACKING_TOL_RAD`), a near-limit step is a strike. Otherwise it is not. Past-the-limit and speed strikes are
unchanged. The recipe corpus of section 6 ran with 0 watchdog abandons in 1250 attempts. The
re-baseline had 66 and MS2 had 56, and both likely include this false positive.

### 5.3 Speed ladder, pre-registered

The ladder ran on MuJoCo with 12 envs, 50 attempts, cameras off, the MS2 recipe, and the same
seeds per rung. The rule, written before the run: adopt the fastest rung with yield ≥ 0.95
and a cycle at most 0.85x the speed-1.0 control.

| probe | transport speed | yield | clamp | steps/brick | static frames | accel p90 | time_outs | abandons |
|---|---|---|---|---|---|---|---|---|
| p1 (staircase) | none | 1.000 | 0.0000 | 815 | 0.46 (target, one episode) | 14.6 (one episode) | 0 | 0 |
| smooth_s1.0 | 1.0 | 0.960 | 0.0000 | 935 | 0.17 | 0.38 | 1 | 1 |
| smooth_s2.0 | 2.0 | 1.000 | 0.0000 | 665 | 0.24 | 1.06 | 0 | 0 |
| smooth_s3.0 | 3.0 | 1.000 | 0.0000 | 539 | 0.29 | 2.24 | 0 | 0 |
| smooth_s4.0 | 4.0 | 0.980 | 0.0000 | 484 | 0.33 | 3.29 | 0 | 1 |

**Rule: adopt 4.0.** Its yield 0.980 is at least 0.95, and its cycle 484/935 = 0.52 is at
most 0.85. Against the staircase's 815 it is 41 % shorter, past the design's 15 % bar. 3.0 is
the neighbour with a perfect yield at an 11 % longer cycle. The 1.0 control is slower than the
staircase, because the staircase dropped its dwell to one step on budget-capped long legs. 4.0
is the default of `--transport_speed` and `OracleRig`.

## 6. The recipe corpus (PRs #23 and #25; design section 5 item 2)

### 6.1 Ports

`hover_jitter` (recipe design 5.1, PR #23) is a per-pick lateral offset of the HOVER target.
The radius is uniform in the range, at a uniform angle, drawn on every HOVER entry, retries
included. DESCEND keeps the brick's true pose. The knob is off by default. When it is off, it draws
nothing from the rng. Both batch scripts take `--hover_jitter_min/max`. `RUN_META` records `hover_jitter`, and
`RUN_STATS` and each sidecar record `hover_jitters`. When the flags are set, the supervisor forwards
them. Otherwise it omits them.

The lead trim (recipe design 5.3, PR #25) replaces the metadata-only reshape of that design.
That reshape existed because the MS2 shards were gone. A corpus generated on MuJoCo keeps
`obs/cycle_phase` and `obs/cycle_brick` in its shards. The converter's `--split_cycles` does
the per-brick split at conversion time, and a new `--lead_keep 60` keeps the last 60 SETTLE
frames on each source's first cycle. A two-episode shard checked it: 12 sub-episodes, exactly
the 2 × 240 SETTLE frames dropped, sidecars written. The metadata-only pass stays unwritten.

### 6.2 Yield probes, pre-registered (recipe design 6.3), on the smooth walk at 4.0

| probe | recipe | yield | clamp | steps/brick | jitters/picks | induced | off-schedule | read |
|---|---|---|---|---|---|---|---|---|
| P1 | `ns2.0 sp0.4 induce0.12 hover_jitter 0.005-0.035 rpo` | 0.920 | 0.0137 | 587 | 323/295 | 37 | 49/50 | acceptance FAIL on clamp > 0.01 |
| P2 | P1 with `sp0.2` | 0.920 | 0.0000 | 534 | 317/291 | 35 | 49/50 | acceptance PASS, **ADOPT** (≥ 0.50) |

P1's clamp came from the 0.4 rad start-pose offset alone. The recipe design expected a yield
near 0.5 on this recipe. The Maker Arm oracle on the smooth walk reads 0.92.

### 6.3 Corpus (`t2/`)

Ten lanes × 5 runs × 25 attempts ran at the P2 recipe with seeds `20261037 + 100 × lane`,
converted with `--split_cycles --lead_keep 60 --sidecar_dir`. They ran from 01:49 to 03:23
PDT, beside the re-baseline's training.

| lane | seed | runs | exported/attempts | rate | time_out | watchdog | abandons | clamp | hover_lowered | s/attempt |
|---|---|---|---|---|---|---|---|---|---|---|
| lane_0 | 3081167816 | 5 | 121/125 | 0.968 | 0 | 0 | 4 | 0.0e+00 | 102 | 21 |
| lane_1 | 3081267816 | 5 | 122/125 | 0.976 | 1 | 0 | 2 | 2.2e-06 | 89 | 21 |
| lane_2 | 3081367816 | 5 | 122/125 | 0.976 | 0 | 0 | 3 | 1.6e-04 | 128 | 21 |
| lane_3 | 3081467816 | 5 | 119/125 | 0.952 | 0 | 0 | 6 | 4.3e-06 | 83 | 21 |
| lane_4 | 3081567816 | 5 | 122/125 | 0.976 | 0 | 0 | 3 | 3.6e-05 | 92 | 21 |
| lane_5 | 3081667816 | 5 | 121/125 | 0.968 | 0 | 0 | 4 | 2.2e-05 | 79 | 20 |
| lane_6 | 3081767816 | 5 | 122/125 | 0.976 | 0 | 0 | 3 | 2.2e-06 | 80 | 20 |
| lane_7 | 3081867816 | 5 | 120/125 | 0.960 | 0 | 0 | 5 | 1.3e-05 | 102 | 20 |
| lane_8 | 3081967816 | 5 | 121/125 | 0.968 | 0 | 0 | 4 | 1.5e-04 | 118 | 21 |
| lane_9 | 3082067816 | 5 | 119/125 | 0.952 | 0 | 0 | 6 | 3.2e-05 | 105 | 20 |
| **all** | | | **1209/1250** | **0.967** [0.956, 0.976] | 1 | 0 | 40 | | 978 | |

Merged: `MERGE_OK lanes=10 episodes=7263 frames=3441541`, 6.0 sub-episodes per source, mean
474 frames. That is 31 % fewer frames than the re-baseline on 4 % more sources. Every abandon
is a `grasp_retries_exhausted`, the recipe's own induction and correctives. The 21 s per
attempt against the re-baseline's 150 is the Rust planner and the smooth walk.

### 6.4 Training, ladder and gate

Pending. The chain trains `train/act_t2_600k` (600k steps, from 07:35 PDT) and runs the same
ladder, check and official gate under `eval_t2/`. This section is filled in after the gate.

## 7. Levers not pulled

- The privileged-state BC ceiling (design section 5 item 3) needs per-frame brick poses in the
  shard and an `observation.environment_state` feature in the dataset. Neither exists yet.
- DAgger with the oracle as expert (item 4) needs the rig's resume-from-state entry.
- Cross-leg blending of the smooth walk (RAISE into EXTEND, RETREAT into HOVER) is left for a
  later probe. The speed ladder cleared the cycle bar without it.
- The `--unstick` supervisor (D1 Option A), the close-event instrument (recipe design 5.2) and
  the BCE gripper head are not ported. Every gate here is pure policy.

## 8. Reproduction

```
# Code: maker-arm-lab PRs #17, #18 (the port), #20, #21 (Rust planner), #22 (smooth walk),
# #23 (hover_jitter), #25 (lead_keep); maker-arm-rs PR #5. Drivers and ledger under
# ~/data/maker-arm-ms3/ (each driver writes a `done` file; launch with setsid).

# Yield probe (design section 3) and the speed ladder (section 5).
bash probes/run_probe_mj.sh p1
bash probes/run_probe_smooth.sh smooth_s4.0 4.0        # and 1.0, 2.0, 3.0
bash probes/run_probe_smooth.sh recipe_p2 4.0 --run_seed 3081134816 --layout_seed 3081135316 \
  --noise_scale 2.0 --start_pose_offset_rad 0.2 --induce_prob 0.12 \
  --hover_jitter_min 0.005 --hover_jitter_max 0.035

# Re-baseline: ten lanes, merge, train, ladder + check + gate (the chain driver does all four).
bash t1/launch_lanes.sh 10 100 && bash t1/run_chain_mj.sh

# Recipe corpus: the t2 chain waits for the t1 lanes and the t1 gate on its own.
bash t2/run_chain_t2.sh

# Tables of this report.
python3 report_tables.py all
```
