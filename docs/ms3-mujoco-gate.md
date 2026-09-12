# MS3 — MuJoCo first: the port, the levers, and the gate pass

Milestone: MS3. Spec: `2026-09-10-ms3-mujoco-design.md`. That spec supersedes the execution
order of `2026-09-10-ms3-recipe-transplant-design.md` and keeps its gate, decisions, seeds,
instruments and read rules. Executed 2026-09-10 to 2026-09-12 on the RTX 5090 host (MuJoCo
3.13, LeRobot 0.5.1, maker-arm-rs at its `feat/kinematics` merge). Every number below is read
off a file named beside it. The running record is `~/data/maker-arm-ms3/README.md` on that
host, and this report was assembled from it with `~/data/maker-arm-ms3/report_tables.py`.

## 1. Summary

**MS3 passes.** The Phase 4 corpus t4 scored **605/720 = 0.840 per brick**, Wilson 95%
[0.812, 0.865], on the pre-registered second seed 20261047 (section 8.4). The whole interval
clears the 0.70 criterion. R-GATE passes, and the milestone closes on this report. Two
campaigns followed the recipe gate of 0.449. A visual inspection found that the t2 policy
memorized the 50 run-level bin and camera layouts of its corpus. It also found that the MuJoCo
light rig hid brick colors in shadow. The t3 corpus added many short runs, a new rig, a
nearest-first pick order and a close-event instrument. It scored **484/720 = 0.672** (section
7.6), and its wrong-bin deposits fell from 82 of 549 releases to 1 of 617. That read fired the
pre-registered second tranche. The failed closes of the t3 gate selected wide corrective
approaches, and t4 added 375 episodes of them (section 8).

**The MuJoCo port reproduces the Isaac gate number, and the recipe corpus lifts it 5.9 times**.
The MS2 recipe re-run on MuJoCo scored **55/720 = 0.076 per-brick**, Wilson 95% [0.059, 0.098]
(section 3.6). Isaac's MS2 read 52/720 = 0.072 [0.055, 0.093] on the same recipe and protocol.
The two simulators agree to within the interval. The levers measured on MuJoCo from here read
against 0.076, and the simulator switch does not confound them. The recipe corpus scored
**323/720 = 0.449**, Wilson [0.413, 0.485] (section 6.7). It is the openyam-lab recipe on the
smooth walk, cut into per-brick sub-episodes with a 60-frame lead. The criterion is **≥ 0.70**.
Both gates **FAILED**. The recipe gate reads under 0.50, so the read rule R-GATE stops the
milestone for an owner decision (section 6.8). Each corpus is 1000 success-only sources, plain
ACT for 600k steps, rule L3 with the check step, and one official gate on 120 episodes. The
re-baseline took 11.7 hours from corpus to gate. The recipe corpus took 6.4 hours of compute.
The Isaac campaign was budgeted at three GPU days.

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
  hours. A host reboot cut its first 600k run at step 83k, and the same chain re-ran it from
  scratch. The gate read 0.449 (section 6.7).

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

### 6.4 Training, interrupted once

The training smoke passed at 07:35 PDT (`TRAIN_OK`, 100 steps on the 7263-sub-episode corpus).
`train_act.sh` accepted the corpus against the ledger target of 1000, which counts sources. The
first 600k run started at 07:35 (wandb run `iccsupub`). At 08:07:33 PDT the host shut down for
a reboot, a clean systemd shutdown with the same kernel before and after. The run stopped at
step 83,325. `train/done_t2` was never written, so the chain's guard held and nothing ran under
`eval_t2/`. The killed run is parked as `train/act_t2_600k.killed-0807`. The same chain driver
was relaunched at 08:37 on the owner's instruction. Every finished stage skipped itself on its
completion file, and a fresh 600k run started into `train/act_t2_600k` (wandb
`ms3-act-t2-600k`). A fresh run rather than a resume keeps the protocol identical to the
re-baseline's, and `train_act.sh` has no resume path. It took 3 h 44 min (08:37 to 12:21 PDT)
at about 45 steps/s, alone on the GPU. The final loss was 0.066 after 1.39 epochs over 3.44 M
frames, a liveness read only.

One defect surfaced in the killed run. Its log ends with `TRAIN_OK out=.../act_t2_600k
steps=600000` although it stopped at 83k. `train_act.sh` prints its token from an EXIT trap.
Bash runs that trap on the shutdown signal, and the status it sees is the previous command's
zero, not the killed trainer's. The token had no consumer, because the chain reads `done_t2`,
which `run_train_t2.sh` writes only after the wrapper returns. The fix is one trap line for
TERM, INT and HUP that exits non-zero, so the EXIT trap prints `TRAIN_FAIL`. It is not in this
report's branch, which carries documents only.

### 6.5 Selection ladder (rule L3)

The same ladder as section 3.3: fifteen rungs on draws 0 to 2 of seed 20261017, 216 bricks per
rung, four rungs at a time, 20.6 s per episode. It took 49 minutes (12:21 to 13:10 PDT).

| rung | per-brick (draws 0-2, 216 bricks) |
|---|---|
| 250k | 55/216 = 0.255 |
| 275k | 73/216 = 0.338 |
| 300k | 59/216 = 0.273 |
| 325k | 75/216 = 0.347 |
| 350k | 108/216 = 0.500 |
| 375k | 69/216 = 0.319 |
| 400k | 78/216 = 0.361 |
| 425k | 105/216 = 0.486 |
| 450k | 100/216 = 0.463 |
| 475k | 88/216 = 0.407 |
| 500k | 104/216 = 0.481 |
| 525k | 50/216 = 0.231 |
| 550k | 74/216 = 0.343 |
| 575k | 116/216 = 0.537 **L3 candidate** |
| 600k | 83/216 = 0.384 |

Every rung reads above the re-baseline's best rung (0.144). The ladder is non-monotone again:
525k dips to 0.231 between 0.481 and 0.343. The candidate's neighbours read 0.343 and 0.384,
so it is not an isolated spike by the rule's definition. A blind `last` takes 600k at 0.384.

### 6.6 Check step

The top two rungs ran on all five draws of the selection seed, 60 episodes and 360 bricks
each: 575k 162/360 = 0.450, 350k 150/360 = 0.417. Both ladder reads regressed on draws 3 and 4
(575k 46/144 = 0.319, 350k 42/144 = 0.292). The rule gates the higher checked rate: **575k**,
the same rung the re-baseline gated.

### 6.7 Official gate, evaluated once

The gate command was the one of section 3.5 with `--out_dir eval_t2/official` and the t2
ledger. The checkpoint was `train/act_t2_600k/checkpoints/575000`. Five draws ran at a time,
then one aggregating pass. It is pure policy, no supervisor. Results are in
`eval_t2/official/gate_results.json`. It took 65 minutes from the end of training to the gate
read (12:21 to 13:27 PDT), ladder included.

| | MuJoCo recipe corpus (t2) | MuJoCo re-baseline (t1) | Isaac MS2 |
| --- | --- | --- | --- |
| per-brick | **323/720 = 0.449**, Wilson [0.413, 0.485] | 55/720 = 0.076, [0.059, 0.098] | 52/720 = 0.072, [0.055, 0.093] |
| episodes with any brick | 102/120 | 41/120 | 42/120 |
| histogram, 0 to 6 bricks | 18, 20, 21, 18, 21, 9, 13 | 79, 29, 10, 2, 0, 0, 0 | 78 sorted none, none sorted six |
| all six bricks | 13/120 | 0/120 | 0/120 |
| never closed the jaws | 0/120 | 45/120 | 3/120 |
| first close, median (p10, p90) | 198 (152, 306) | 563 (289, 1829) | 1249 (476, 2792) |
| graded grasp attempts | 1602, one per 398 steps | 427, one per 1518 steps | 693, one per 935 steps |
| clamp fraction | 0.019, on joints 4 and 6 | 0.168, all on joint 6 | 7.4e-4, all on joint 6 |
| wrong bin | 82 of 549 releases | 1 of 59 releases | 13 of 90 held |
| wall | 24.7 s/episode at five concurrent draws | 24.5 s/episode | 25 s/episode |

### 6.8 Reading the gate

**R-GATE says STOP**. The recipe design's read rule (section 8.4 there) has three bands. It
passes at 0.70, fires the contingent second tranche between 0.50 and 0.70, and stops under
0.50 for an owner decision. The gate reads 0.449 and its interval tops out at 0.485, so the
second tranche does not fire. The lever is still the largest step from one campaign on either
arm: 0.076 to 0.449 on the same simulator, protocol and seeds. openyam-lab's largest was 0.468
to 0.758. The corpus carries six openyam-lab levers at once, plus the smooth walk: widened
noise, induction, correctives, randomized pick order, the per-brick split and the lead trim.
The step is therefore not attributable to any one of them. That was the known cost of decision
D2.

The factor account reads ① 1602/720 = 2.23, ② 128/1602 = 0.080, ③ 323/128 = 2.52. Two of the
three are not factors on MuJoCo. ① is above one because the policy attempts 2.2 graded grasps
per brick, where the definition assumed at most one. ③ is above one because the pinch grader
misses holds at MuJoCo's 6.1 N grip (section 3.6): 323 bricks were stowed by grasps, and the
grader credited 128 of those grasps. Only the product is read, and it is the gate number. A
grader retuned on MuJoCo is the prerequisite for reading ② and ③ apart, and it is not done.

What changed against the re-baseline is the cadence. The re-baseline policy never closed in 45
episodes and first closed at a median of 563 steps. The recipe policy closes in every episode,
first at a median of 198 steps, which is the lead trim's effect (openyam-lab: 877 to 299). It
then attempts a grasp every 398 steps, 13.4 per episode, against the demonstrated cycle of 534
steps per brick. It sorts all six bricks in 13 episodes and at least one in 102. Its commands
are clamped on 1.9 % of steps (re-baseline 16.8 %), now split between joints 4 and 6. The loss
is in the misses and in the sorting: 1602 attempts for 323 stowed bricks, and 82 wrong-bin
deposits against 549 queued releases (0.149, MS2's Isaac policy 0.16 of detected releases).
The corpus was generated with randomized pick order. The color-schedule shortcut that
openyam-lab's pick-order fix removed (0.415 to 0.058) is therefore not open to this policy. The
wrong-bin loss has no measured mechanism here. The close-event instrument, which gives the
misses and the wrong bins a geometry, was not ported at that gate. It is now PR #29 (section 7.2).

## 7. The visual inspection and the t3 corpus

### 7.1 What the inspection found

After the recipe gate, the owner watched the t2 policy in the MuJoCo viewer (PR #27). Two
reads came out of it. Dark robot shadows hid the color of the bricks, and grasped bricks went
to a wrong bin. Instrumented probes on four official draws then separated three causes, in this
order of weight:

1. **Layout memorization.** The t2 corpus has 50 runs, so it has 50 combinations of run-level
   bin positions and overhead camera pose. On an unseen combination the policy sends a brick to
   a neighboring bin. On official draw 4 at nominal bin and camera geometry, the same checkpoint
   stowed 50 of 72 bricks instead of 14. Its wrong releases fell from 22 to 2.
2. **The light rig.** The MuJoCo port put most of the light in a hard sun. With shadows off,
   draw 0 stowed 31 of 72 instead of 16.
3. **Approach precision.** Half of all closes landed 2 cm or more from the nearest brick, and a
   close that far away rarely holds.

### 7.2 Levers, and the owner decision

The owner composed four levers into one corpus. The gate then cannot separate them:

| lever | change | where |
| --- | --- | --- |
| 1, many short runs | `--attempts_per_run 12` on 12 envs, one wave per run: about 100 layouts for 1200 attempts | lane script only |
| 2, light rig | hemisphere dome and mild sun, calibrated on Isaac frames | PR #28 |
| 3, instrument | `[brick, bin]` per release, nearest-brick distance and a windowed hold per close | PR #29 |
| 5, recipe knobs | `--nearest_first_pick_order` instead of the randomized order, `--induce_prob` 0.12 to 0.06, `hover_jitter` 0.005-0.035 to 0.005-0.020 | PR #30 |

Nearest-first takes the candidate brick closest to the tool in the desk plane. The desk scatter
is random, so the color order still varies per episode, and the target becomes a function of
the scene the cameras see. The oracle refuses a draw that asks for both pick orders.

### 7.3 Corpus (`t3/`)

Ten lanes, lane seed `20261057 + 100 * lane`, target 100 each, the t2 recipe otherwise.

| lane | runs | exported/attempts | rate | abandons | hover_lowered | s/attempt |
| --- | --- | --- | --- | --- | --- | --- |
| lane_0 | 9 | 105/108 | 0.972 | 3 | 73 | 17 |
| lane_1 | 9 | 102/108 | 0.944 | 6 | 82 | 18 |
| lane_2 | 9 | 104/108 | 0.963 | 4 | 88 | 17 |
| lane_3 | 9 | 103/108 | 0.954 | 5 | 76 | 17 |
| lane_4 | 9 | 105/108 | 0.972 | 3 | 83 | 17 |
| lane_5 | 9 | 107/108 | 0.991 | 1 | 77 | 16 |
| lane_6 | 9 | 103/108 | 0.954 | 5 | 91 | 15 |
| lane_7 | 9 | 107/108 | 0.991 | 1 | 81 | 17 |
| lane_8 | 9 | 107/108 | 0.991 | 1 | 86 | 17 |
| lane_9 | 9 | 105/108 | 0.972 | 3 | 74 | 17 |
| **all** | **90** | **1048/1080** | **0.970** [0.958, 0.979] | 32 | 811 | |

The merge holds 6289 sub-episodes and 2,924,212 frames. That is 15 % fewer frames than t2 for
about the same source count, because the lower induction produces fewer retry cycles. All 24
color-slot permutations appear.

### 7.4 Three interruptions

- **A moved checkout.** The first launch ran from the main checkout. At 16:19 PDT another
  session switched that checkout to a docs branch, and every lane died on its next run with
  `unrecognized arguments: --nearest_first_pick_order`. All 30 finished runs carried the
  pre-registered knobs, so the lanes resumed. Every later stage ran from a dedicated worktree
  whose `PYTHONPATH` puts its own `source/` ahead of the editable install.
- **Two training runs died at step 36823.** Both runs failed at the same step in a DataLoader
  worker with `Could not push packet to decoder: Invalid data found when processing input`.
  The trainer seeds its shuffle, so the failure was deterministic. A replay of the dataloader
  with the decoder instrumented named the item: frames 4388 and 4389 of one lane's wrist video,
  in episode 1445. A complete `ffmpeg -v error` decode of all 180 videos found no other bad
  file.
- **The repair.** The converter had deleted the shard HDF5, so a reconversion of the video was
  impossible. A plain re-encode drops the undecodable packets and returns 35138 frames instead
  of 35144, and that shifts every later frame index. The re-encode with `-vsync cfr -r 30`
  refills the gap by duplication and keeps the count, the frame rate and the duration. The
  third run trained 600k steps in 3 h 43 min.

### 7.5 Selection ladder (rule L3)

| rung | per-brick (draws 0-2, 216 bricks) |
| --- | --- |
| 250k | 100/216 = 0.463 |
| 275k | 80/216 = 0.370 |
| 300k | 107/216 = 0.495 |
| 325k | 131/216 = 0.606 |
| 350k | 127/216 = 0.588 |
| 375k | 119/216 = 0.551 |
| 400k | 134/216 = 0.620 |
| 425k | 142/216 = 0.657 |
| 450k | 129/216 = 0.597 |
| 475k | 140/216 = 0.648 |
| 500k | 158/216 = 0.731 |
| 525k | 150/216 = 0.694 |
| 550k | 148/216 = 0.685 |
| 575k | 160/216 = 0.741 **L3 candidate** |
| 600k | 135/216 = 0.625 |

The candidate is not an isolated spike. The check step on draws 0 to 4 read 575k 272/360 =
0.756 against 500k 261/360 = 0.725 and gated 575k.

### 7.6 Official gate, evaluated once

Seed 20261027, checkpoint 575000, pure policy, `eval_t3/official/gate_results.json`.

| | t3 | t2 |
| --- | --- | --- |
| per-brick | **484/720 = 0.672**, Wilson [0.637, 0.706] | 323/720 = 0.449, [0.413, 0.485] |
| episodes with any brick | 115/120 | 102/120 |
| histogram, 0 to 6 bricks | 5, 12, 15, 11, 16, 21, 40 | 18, 20, 21, 18, 21, 9, 13 |
| all six bricks | 40/120 | 13/120 |
| never closed the jaws | 0/120 | 0/120 |
| first close, median (p10, p90) | 216 (163, 317) | 198 (152, 306) |
| graded grasp attempts | 1345, one per 447 steps | 1602, one per 398 steps |
| wrong bin | **1 of 617 releases** | 82 of 549 releases |
| closes within 2 cm | 720 of 1338 (53.8 %) | not instrumented |
| windowed hold, all closes and within 2 cm | 0.524 and 0.860 | not instrumented |
| clamp fraction | 0.115, all on joint 6 | 0.019, on joints 4 and 6 |
| wall | 24.5 s/episode | 24.7 s/episode |

### 7.7 Reading the gate

**R-GATE fires the second tranche.** The read is 0.672, inside the 0.50 to 0.70 band. It is
1.50 times t2 on the same simulator, protocol and seeds.

The wrong-bin loss is gone. One error in 617 releases (`green->yellow`) replaces 82 in 549,
as the layout ablations predicted for lever 1. The grasp itself did not
change. Holds by nearest-brick distance read 92 % under 1.5 cm and 68 % at 1.5 to 2 cm. They read
28 % at 2 to 3 cm, 8 % at 3 to 5 cm and 0 % beyond. That profile matches the t2 probes. What remains is where
the closes land.

The tranche-2 table of the recipe design picks its lever from the anatomy of the failed closes.
Of 637 failed closes, 45.1 % missed by 3.5 cm or more. That is above the 35 % line, so the
table selects wide corrective approaches (`hover_jitter 0.035-0.080`). A coverage read supports that choice.
The t3 corpus demonstrates lateral corrections up to about 2.5 cm: the jitter range, the induced
offsets and `lateral_offset` together. Of the failed closes, 69.4 % missed by more than that. The
policy's dominant failure lies outside the error band its demonstrations ever null. Lever 5's
narrowing pushed the wrong way for this failure mode.

The clamp fraction rose from 0.019 to 0.115, all on joint 6. It is not explained, and it did not
hold the gate back.

## 8. Phase 4: wide corrective approaches (`t4/`)

### 8.1 Probe P3, pre-registered

Branch 1 reverses a knob that t3 had just adopted, so the owner asked for a probe first. Three
arms differed only in `hover_jitter`, each 24 attempts on the same paired seeds (20261003 and
20261004), `--probe`, the t3 recipe otherwise. The rule, written before the run: take the wide
arm if its yield is 0.85 or better.

| arm | `hover_jitter` | yield | corrections per episode | s/episode |
| --- | --- | --- | --- | --- |
| N | 0.005-0.020 (t3) | 24/24 | 6.3 | 8.1 |
| M | 0.020-0.050 | 24/24 | 6.2 | 7.8 |
| W | 0.035-0.080 (branch 1) | 24/24 | 6.1 | 7.3 |

The oracle nulls an 8 cm offset as reliably as a 2 cm one. The rule took W.

### 8.2 Corpus

Ten lanes, lane seed `20261037 + 100 * lane`, target 35 each, the t3 recipe with
`hover_jitter 0.035-0.080`.

| lane | runs | exported/attempts | rate | abandons | hover_lowered |
| --- | --- | --- | --- | --- | --- |
| lane_0 | 3 | 36/36 | 1.000 | 0 | 42 |
| lane_1 | 3 | 36/36 | 1.000 | 0 | 47 |
| lane_2 | 4 | 45/48 | 0.938 | 3 | 73 |
| lane_3 | 3 | 36/36 | 1.000 | 0 | 53 |
| lane_4 | 4 | 45/48 | 0.938 | 3 | 61 |
| lane_5 | 3 | 35/36 | 0.972 | 1 | 57 |
| lane_6 | 3 | 36/36 | 1.000 | 0 | 47 |
| lane_7 | 3 | 36/36 | 1.000 | 0 | 55 |
| lane_8 | 3 | 35/36 | 0.972 | 1 | 56 |
| lane_9 | 3 | 35/36 | 0.972 | 1 | 45 |
| **all** | **32** | **375/384** | **0.977** [0.956, 0.988] | 9 | 536 |

The lanes took 25 minutes. The merge of t3's ten lanes and these ten holds 8539 sub-episodes and
4,002,347 frames over 122 layouts. The wide correctives are 27 % of the frames. The merge passes
`video_files_size_in_mb=1`, so every video is a byte copy of its lane source. Before training,
the chain decoded all 244 videos with `ffmpeg -v error` and stops on any error. It passed in
under four minutes. Training ran 600k steps in 3 h 41 min with no failure.

### 8.3 Selection ladder (rule L3)

| rung | per-brick (draws 0-2, 216 bricks) |
| --- | --- |
| 250k | 161/216 = 0.745 |
| 275k | 113/216 = 0.523 |
| 300k | 178/216 = 0.824 |
| 325k | 146/216 = 0.676 |
| 350k | 154/216 = 0.713 |
| 375k | 177/216 = 0.819 |
| 400k | 166/216 = 0.769 |
| 425k | 177/216 = 0.819 |
| 450k | 187/216 = 0.866 **L3 candidate** |
| 475k | 155/216 = 0.718 |
| 500k | 140/216 = 0.648 |
| 525k | 176/216 = 0.815 |
| 550k | 184/216 = 0.852 |
| 575k | 172/216 = 0.796 |
| 600k | 185/216 = 0.856 |

The candidate is not an isolated spike. The check step on draws 0 to 4 read 600k 315/360 =
0.875 against 450k 310/360 = 0.861 and gated 600k. On the same five selection draws t3's gated
rung read 0.756.

### 8.4 Official gate, evaluated once

The pre-registered second seed 20261047, checkpoint 600000, pure policy,
`eval_t4/official/gate_results.json`. t3 and t2 were gated on seed 20261027, so the columns do
not share draws.

| | t4 | t3 | t2 |
| --- | --- | --- | --- |
| per-brick | **605/720 = 0.840**, Wilson [0.812, 0.865] | 484/720 = 0.672, [0.637, 0.706] | 323/720 = 0.449, [0.413, 0.485] |
| episodes with any brick | 117/120 | 115/120 | 102/120 |
| histogram, 0 to 6 bricks | 3, 2, 3, 11, 12, 18, 71 | 5, 12, 15, 11, 16, 21, 40 | 18, 20, 21, 18, 21, 9, 13 |
| all six bricks | 71/120 | 40/120 | 13/120 |
| never closed the jaws | 0/120 | 0/120 | 0/120 |
| first close, median (p10, p90) | 252 (189, 317) | 216 (163, 317) | 198 (152, 306) |
| graded grasp attempts | 1196, one per 457 steps | 1345, one per 447 steps | 1602, one per 398 steps |
| wrong bin | 9 of 764 releases | 1 of 617 releases | 82 of 549 releases |
| closes within 2 cm | 824 of 1191 (69.2 %) | 720 of 1338 (53.8 %) | |
| windowed hold, all closes and within 2 cm | 0.714 and 0.951 | 0.524 and 0.860 | |
| clamp fraction | 0.029, all on joint 6 | 0.115, all on joint 6 | 0.019 |
| wall | 24.1 s/episode | 24.5 s/episode | 24.7 s/episode |

### 8.5 Reading the gate

**R-GATE passes, and the milestone closes.** The read is 0.840, and the whole interval clears
0.70. After Phase 4 the milestone stops whatever the outcome (recipe design section 9), and this
outcome is a pass.

The wide correctives moved closes into the band that holds. Closes within 2 cm rose from 53.8 %
to 69.2 %, and the hold rate there from 0.860 to 0.951. Holds by distance read 97 %, 82 %, 42 %,
12 % and 0 %. The failed closes kept their shape (median 3.36 cm, 69.5 % beyond 2.5 cm), and
there are 47 % fewer of them (341 against 637). The policy sorts all six bricks in 71 of 120
episodes.

Wrong bin reads 9 of 764 releases, up from t3's 1 but 13 times under t2. With more releases and
different draws, it is recorded and not read as a regression. The clamp fraction fell back to
0.029.

The factor account reads 1.66, 0.216 and 2.34. As in t2 and t3, the pinch grader under-credits
holds on MuJoCo, so only the product is a number. The windowed instrument of PR #29 is the
readable grasp measure.

Attribution. The passing corpus carries many short runs, the new rig, the instrument,
nearest-first order, lower induction and the wide correctives. The gates in this report cannot
say how much each one contributes.

## 9. Levers not pulled

- The `train_act.sh` token contract on a signal death (section 6.4): one trap line, on its own
  branch, with a test that kills the wrapper and expects `TRAIN_FAIL`.
- The pinch grader on MuJoCo. `PINCH_MIN_FORCE` and `PINCH_MIN_STALL` were tuned on Isaac's
  7.6 N grip and credit 128 holds for 323 stowed bricks at MuJoCo's 6.1 N. Until they are
  re-measured there, factors ② and ③ have no separate read on MuJoCo.
- The privileged-state BC ceiling (design section 5 item 3) needs per-frame brick poses in the
  shard and an `observation.environment_state` feature in the dataset. Neither exists yet.
- DAgger with the oracle as expert (item 4) needs the rig's resume-from-state entry.
- Cross-leg blending of the smooth walk (RAISE into EXTEND, RETREAT into HOVER) is left for a
  later probe. The speed ladder cleared the cycle bar without it.
- The `--unstick` supervisor (D1 Option A) and the BCE gripper head are not ported. Every gate
  here is pure policy. The close-event instrument is ported (PR #29).
- A decode check in the converter. `generate_dataset.py` deletes the shard HDF5 before anything
  decodes the video it wrote, so a bad encode cannot be reconverted (section 7.4). The t4 chain
  checks the merged videos, but the check belongs before the delete.
- DONE hold frames. The corpus has no frames after the last brick, so the policy repeats the
  pick cycle on an empty desk. It costs nothing at the gate and is due before a real-arm demo.
- The joint-6 clamp. It rose to 0.115 on t3 and fell to 0.029 on t4, with no measured mechanism.

Hand-off. The milestone passed at 0.840 (section 8.5), so the pre-registered escalations did
not fire. The privileged-state ceiling, DAgger and the pinch grader retune carry over to the
post-gate roadmap.

## 10. Reproduction

```
# Code: maker-arm-lab PRs #17, #18 (the port), #20, #21 (Rust planner), #22 (smooth walk),
# #23 (hover_jitter), #25 (lead_keep), #27 (viewer), #28 (light rig), #29 (close-event
# instrument), #30 (nearest-first order); maker-arm-rs PR #5. Drivers and ledger under
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

# After the 2026-09-11 08:07 PDT reboot (section 6.4) the killed run and its log were parked
# and the same chain relaunched. Finished stages skip themselves. 3 h 44 min of training, then
# 65 min of ladder, check and gate; the result is eval_t2/official/gate_results.json.
mv train/act_t2_600k train/act_t2_600k.killed-0807 && mv train/train_t2.log train/train_t2.killed-0807.log
setsid bash -c 'exec bash t2/run_chain_t2.sh' < /dev/null > /dev/null 2>&1 &

# t3: the visual-inspection levers, from a dedicated worktree (section 7.4). Lanes resume by
# themselves on a relaunch. Failed training runs are parked before a relaunch, as for t2.
git worktree add ~/src/maker-arm-lab-t3 feat/ms3-nearest-first-pick
setsid bash -c 'exec bash t3/run_chain_t3.sh' < /dev/null > /dev/null 2>&1 &

# Phase 4: probe P3, then the tranche-2 chain (lanes, merge with t3, video check, train, ladder,
# check, gate on 20261047).
bash probes/p3_hover_width/run_p3.sh
setsid bash -c 'exec bash t4/run_chain_t4.sh' < /dev/null > /dev/null 2>&1 &

# Tables of this report.
python3 report_tables.py all
```
