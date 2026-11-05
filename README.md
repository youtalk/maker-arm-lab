# maker-arm-lab

Research layer for the [MakerMods Maker Arm](https://github.com/makermods-robotics/maker-arm-sdk): simulation, system identification, model learning and gated milestone experiments. The aim is sim-to-real. The simulation uses the vendor URDF and the arm's own joint coordinates. Thus a trained policy can run on the physical arm through the [maker-arm-rs](https://github.com/youtalk/maker-arm-rs) SDK.

Gains, torque caps and the command clamp come from `maker_arm_rs.profile()` at import time. This repository does not restate them. Thus every recorded action is a command that the real arm accepts. There is no fallback table, and a missing `maker_arm_rs` fails at import.

## Milestones

Two tracks run in this repository. The MA milestones prepare and identify the physical arm. The MS milestones build a sorting task in simulation and train policies on it. MS1 to MS4 sort six bricks into bins by color. MS6 to MS7f sort a heap of mixed parts into desk boxes by class. MS5 is the real-arm milestone and has no report yet.

| Milestone | Task | Result | Report |
| --- | --- | --- | --- |
| MA0 | Bench preparation | A record. It was written before MS1 and not read again against it. | [`ma0-bench-prep.md`](docs/ma0-bench-prep.md) |
| MS1 | The LegoSort environment on Isaac Lab | Failed: the oracle sorts 30/50 = 0.600 against 0.70. | [`ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md) |
| MS2 | The first ACT training and rollout gate | Failed: 52/720 = 0.072 against 0.15. | [`ms2-act-gate.md`](docs/ms2-act-gate.md) |
| MS3 | The MuJoCo port and the corpus recipe | Passed: 605/720 = 0.840 against 0.70. | [`ms3-mujoco-gate.md`](docs/ms3-mujoco-gate.md) |
| MS4 | Corpus levers, then DAgger with the oracle as expert | Passed: DAgger round 1 reads 660/720 = 0.9167 against 0.90. | [`ms4-corpus-levers.md`](docs/ms4-corpus-levers.md), [`ms4-dagger.md`](docs/ms4-dagger.md) |
| MS6 phase A | A teacher for a heap of 50 mixed parts | Goal met: 0.933 held-out and 0.940 train-split stowed against 0.80. | [`ms6-bare-desk-heap.md`](docs/ms6-bare-desk-heap.md) and the five reports before it |
| MS6 phase B | An ACT policy on that heap | Goal met: 2759/3000 = 0.9197 stowed at 1.198 cycles per stowed part against 0.80. | [`ms6-square-wrist.md`](docs/ms6-square-wrist.md) and the two reports before it |
| MS7a | The heap sorted into four desk boxes by four classes | Passed: 5205/6000 = 0.8675 at 1.134 cycles per correct stow. | [`ms7-sorting.md`](docs/ms7-sorting.md) |
| MS7c | The loop reads only a perceived state | Failed on cycles: 2646/3000 = 0.882 at 1.329 against 1.3. | [`ms7c-perceived-state.md`](docs/ms7c-perceived-state.md) |
| MS7d | Nine perception levers on the MS7c gate | Passed: 5521/6000 = 0.920 at 1.123. | [`ms7d-perception-levers.md`](docs/ms7d-perception-levers.md) |
| MA2 | A MuJoCo bench for the tracking controller | `BENCH_FAIL`: 8 of 10 bars pass. | [`ma2-tracking-bench.md`](docs/ma2-tracking-bench.md) |
| MS7e | 4 to 8 boxes, 8 classes, round, LDraw and flat parts | Passed: 5299/6000 = 0.883 at 1.271. | [`ms7e-widened-task.md`](docs/ms7e-widened-task.md), [`ms7e-flat-study.md`](docs/ms7e-flat-study.md) |
| MS7f | 150-part heaps, six categories, a part classifier, speed | Failed, closed by owner decision D25: 2522/3000 = 0.841 at 308.0 correct stows an hour, held-out 0.790. | [`ms7f-real-scale-sorting.md`](docs/ms7f-real-scale-sorting.md) |
| MS7h | A learned policy that beats the oracle head-to-head: residual RL on ACT and a π0.5 track | Failed: the π0.5 candidate's checked read stowed 2464/3000 = 0.821 at 244.8 correct stows an hour, held-out 0.753. S stowed 2525 (D = -61, paired SE 31.5). No residual RL screen beat S. | [`ms7h-beyond-the-oracle.md`](docs/ms7h-beyond-the-oracle.md) |
| MS7j | Real-arm readiness: a stress suite for S, and a π0.5 LoRA on the RTX 5090 | A record, no gate. S passes every suite row at the levels that the rig expects and is the candidate for the first real test. The LoRA's 30k save places 2500/3000 on the screen at 280.0 correct stows an hour, against 2533 at 369.5 for S. Chunked π0.5 fails an inference delay of 6 steps, and real-time chunking passes it. | [`ms7j-real-arm-readiness.md`](docs/ms7j-real-arm-readiness.md) |
| MS7k | π0.5 accuracy and sim2real against S, with the last Colab credit | A record against two rules, both met. The π0.5 LoRA C (35k steps) places 5126 of 6000 on two screens under real-arm serving, against 5073 for S (D +53, SE 44.15). C places 208.0 and 225.1 correct stows an hour, against 332.2 and 343.9 for S. On the sim2real rows, C passes every expected row. It also passes an uncorrected overhead tilt of 6 degrees and a +5 cm shift, where S fails. | [`ms7k-pi05-accuracy-sim2real.md`](docs/ms7k-pi05-accuracy-sim2real.md) |

[Reports](#reports) at the end describes each report.

## Layout

```
source/maker_arm_lab/
  robots/            frames.py (measured constants, isaaclab-free) + maker_arm.py (ArticulationCfg)
  actuators/         MIT-mode actuator; gains and caps from maker_arm_rs.profile()
  assets/            urdf_prep.py (Z-up derived URDF), GravityCompensatedArticulation
  core/              isaaclab-free logic: layout, oracle, global_ik, trajectory, grader (stowed,
                     checker), gate, trace, profile; heap teacher (parts, grasp, heap_oracle);
                     loops (batch_loop, eval_loop, dagger_loop, loop); desk_boxes, perception,
                     hold, torque, dynamics_model, overlay
  mj/                MuJoCo backend: env, scene, recorder, depth, context_names;
                     bench and bench_gate (the MA2 tracking bench)
  real/              MS5 real rig: env (RealEnv), arm (maker_arm_rs Tracking), perceive, sources,
                     camera_model, rig, pacing, frames, latency; ros.py is the only rclpy import
  tasks/manager_based/lego_sort/
                     Isaac env cfgs and the five task ids, mdp/, datagen, oracle_runtime
                     (OracleRig), vision
assets/vendor/maker_arm_description/   pinned copy of maker-arm-sdk urdf/maker_arm (+ VENDOR.md)
assets/parts/                          LDraw part hulls and studs as JSON (scripts/ldraw_parts.py)
assets/housing_mass_model.json         housing mass model for maker_arm_rs.Dynamics
assets/usd/                            generated maker_arm.usd, NOT committed
lerobot_stack/                         LeRobot venv: HDF5 conversion, policy protocol and server,
                                       ACT train wrapper, part classifier trainer
perception_stack/                      perception worker wire format (numpy + stdlib only)
scripts/                               setup, smokes, checks, generation, eval clients, supervisors
tests/                                 CPU-only pytest (no isaaclab import at collection time)
docs/                                  milestone reports
```

## Environment setup

This repository uses three separate environments. `uv run pytest` does not collect `lerobot_stack/`.

```bash
# Isaac Sim 5.1 + Isaac Lab v2.3.2 (Python 3.11) + the pinned maker_arm_rs wheel + this repo, editable
export ISAACLAB_PATH="$HOME/src/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES      # the script refuses to accept NVIDIA's licence on your behalf
bash scripts/setup_isaac_env.sh

# LeRobot 0.5.1 (Python 3.12), for dataset conversion and training
bash lerobot_stack/setup_lerobot_env.sh

# CPU tests and lint (Python 3.12); this is what CI runs
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

`uv sync` builds maker-arm-rs from git, so it needs `cargo`. The tests use MuJoCo's EGL backend unless `MUJOCO_GL` names another one. CI sets `MUJOCO_GL=disable`. Without a GL context, each render test skips.

Activate the Isaac venv before every Isaac command below. Without it, `isaaclab.sh -p` fails with `python: command not found`:

```bash
source "$ISAACLAB_PATH/.venv/bin/activate"
```

The MuJoCo scripts need no Isaac. They run under the uv venv with `uv run python`.

## Simulation asset build

The Isaac simulator loads `assets/usd/maker_arm.usd`. This file is generated and git-ignored:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/convert_urdf_to_usd.py --headless    # CONVERT_OK
```

Run this once after you clone the repository. If `scripts/convert_urdf_to_usd.py`, `source/maker_arm_lab/assets/urdf_prep.py` or the vendored URDF changes, run it again. Nothing tests the freshness of the USD. A stale USD loads without an error and then behaves wrongly in the physics.

The vendored URDF is committed and is never edited. To vendor a new upstream commit, bump `PIN_SHA` in the script. Then run the script and the URDF test again:

```bash
bash scripts/vendor_maker_arm_description.sh
uv run pytest tests/test_urdf_validation.py
```

## Registered task ids

The last column shows what the `--task` default of each script selects. It comes from the scripts, not from the original spec. The two diverged at the move of the oracle. Since MS4 the Isaac task is frozen at its MS3 features and only gets smoke tests. Every new capability goes to the MuJoCo backend.

| Task id | Arm action | Cameras | Runs on it today |
| --- | --- | --- | --- |
| `MakerArm-LegoSort-v0` | clamped absolute joint position | none | `smoke_lego_sort.py` (default), `scripted_grasp_check.py` (default) |
| `MakerArm-LegoSort-IK-Rel-v0` | task-space delta pose (differential IK) | none | no check by default. It is registered and smoke-tested via `--task`. |
| `MakerArm-LegoSort-IK-Rel-Vision-v0` | task-space delta pose | wrist, overhead | no check by default. It is registered and smoke-tested via `--task`. |
| `MakerArm-LegoSort-Datagen-v0` | clamped joint position + vision + domain randomization + recorders | wrist, overhead | `generate_batch.py`, `release_envelope_check.py`, `oracle_check.py` (all three by default) |
| `MakerArm-LegoSort-Eval-v0` | clamped joint position + vision + eval terminations | wrist, overhead | `eval_policy.py`, and smoke-tested via `--task` |

In MS1 the oracle moved onto `core/global_ik.py`, and `Datagen-v0` moved with it from the IK-Rel action space to clamped joint position. The envelope and oracle checks moved with it. Thus the two IK-Rel ids no longer back a gate, although the spec first gave them one. They stay registered and smoke-tested, so the differential-IK path stays available and cannot break unseen. The dataset contract did not change: the recorded action stream was always the joint target after IK. See `docs/ms1-maker-arm-sim.md` section 4.3.

## Checks

Every GPU script prints one token line (`<NAME>_OK` or `<NAME>_FAIL`) and exits through `os._exit` with a real code. It does this on every exit path, argparse usage errors included. A tokenless exit is a defect.

```bash
# SMOKE_OK: joint/body counts, articulation root, Z-up base orientation
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_usd.py --headless

# SMOKE_OK: the id constructs and steps, no NaN, tool axis within 30 deg of world -z
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless [--task <id>] [--enable_cameras]

# the FK sweep that pinned HOME_JOINT_POS
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/find_home_pose.py --headless

# GRASP_OK: hover / descend / close / lift, gated on pinch evidence
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/scripted_grasp_check.py --headless

# ENVELOPE_OK: EXTEND convergence and RELEASE stow; one process per layout, 0..4
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/release_envelope_check.py --headless --layout_index N

# ORACLE_OK: full sorts, every brick, every phase
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/oracle_check.py --headless --trials 2

# ORACLE_OK: the same, exercising the HDF5 image streams
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/oracle_check.py --headless --enable_cameras --with_cameras --trials 1
```

Run `release_envelope_check.py` as one process per layout and read the five tokens. Its single-process loop over the five layouts hangs after the second `gym.make`. The script fixes the pick at (0.20, 0.0) by design. Thus it samples `core.layout.SCATTER_MAX_RADIUS` at one azimuth only and does not give a second, independent test of that bound. If you change what it sweeps, sweep the tracking, not the IK feasibility.

## Data generation and conversion

One batch, for a probe or a smoke:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/generate_batch.py --headless --enable_cameras \
    --out_dir /tmp/batch0 --attempts 10 --num_envs 12 --probe        # BATCH_OK exported=<n> attempts=<m>
```

A full campaign runs generation and LeRobot conversion run by run. It keeps a ledger and a crash marker per run, so an interrupted campaign resumes without duplicate episodes. `generate_dataset.py` is stdlib-only `python3` and needs no virtualenv of its own. It launches `generate_batch.py` under `isaaclab.sh` (or `mj_generate_batch.py` with `--backend mujoco`) and `convert_to_lerobot.py` under the LeRobot venv. This host sweeps long background jobs, and `setsid` alone does not always prevent it. Daemonize the campaign:

```bash
setsid bash -c 'cd "$PWD" && python3 scripts/generate_dataset.py \
  --out_dir ~/data/maker-arm-ms1/work --dataset_root ~/data/maker-arm-ms1/dataset \
  --target 50 --attempts_per_run 50 --num_envs 12 --seed 20260905 \
  > ~/data/maker-arm-ms1/supervisor.log 2>&1' < /dev/null &
# per-run BATCH_OK, then CONVERT_OK, then CAMPAIGN_OK total=<n> runs=<m>
```

To continue an interrupted campaign, add `--resume`. If a run was killed between its conversion and its ledger entry, the campaign stops and does not duplicate the run.

A campaign reads the checkout for hours. Run it from its own git worktree, so that a branch switch in the main checkout cannot break it.

### MS4 corpus levers

MS4 adds a set of corpus levers. They are MuJoCo only, and the Isaac batch script accepts and ignores them.

- `--episode_draws` draws the bins, the lights and the cameras again at every episode reset, not once per run.
- `--frame_jitter` samples a small overhead-camera jitter again on every rendered frame.
- `--spatial_scale` multiplies the jitter of the camera pose and of `bin_x`, in the run draw and in every episode draw. `--spatial_scale 3.0` on `mj_eval_policy.py` or `run_gate_eval.py` scores a policy far outside its training distribution.
- `--context {none,camera,privileged}` on the same two scripts picks the extra channel that the env serves to the policy: the overhead camera pose or the six brick poses. It must match the checkpoint.

`mj_generate_batch.py` and `generate_dataset.py` take the first two flags. On the conversion side:

- `--context {camera,privileged,camera_box}` writes that channel into the dataset.
- `--no_images` drops both image features, for the privileged-state ceiling.
- `--action_space {absolute,delta_prev}` sets what the six joint columns of `action` hold. The trainer takes the same choice as `ACTION_SPACE`, refuses a dataset that disagrees, and records the choice next to the checkpoints.

One MS4 campaign is converted three times. Thus `generate_dataset.py --keep_shards` keeps every converted shard on disk, so the later conversions can read them.

### MS4 DAgger

MS4 round c2 adds DAgger with the oracle as expert. `scripts/mj_dagger_batch.py` rolls a policy out and lets the oracle rig take an env over on a trigger. There are five triggers:

- No jaw close for 400 steps.
- A jaw close far from every brick.
- Closed jaws with no pinch.
- A grasp center outside the workspace.
- A random trigger at a mean interval of 300 steps, which doubles each round.

A takeover runs one brick cycle from the state that the policy reached, then gives the env back. The recorder keeps only takeover frames. `obs/actor` holds the takeover ordinal of each kept frame. If a cycle does not stow its brick, its ordinal goes negative. `convert_to_lerobot.py --dagger --split_cycles` keeps the positive-actor cycles. The ceiling lanes of c2 record with `generate_dataset.py --no_cameras --context privileged --no_images`.

```bash
uv run python scripts/mj_dagger_batch.py --num_envs 12 --run_seed 1 --layout_seed 2 \
    --permutation 0,1,2,3 --episode_draws --frame_jitter --context camera \
    --server tcp://127.0.0.1:5700 --round 1 --out_dir /tmp/dag --shard_name proc_0
```

### MS6 heap

MS6 phase A adds a heap of mixed parts on the desk. It is MuJoCo only.

- `--heap_k K` on `mj_generate_batch.py` replaces the six bricks with K parts from a 29-part parametric catalog. The catalog holds bricks from 1x1 to 4x8, tall bricks up to 1x2x5, and 45-degree slopes with two or more studs along the run.
- Plates, tiles and round bricks left the catalog on 2026-09-18 (`docs/ms6-bare-desk-heap.md`). They stay constructible for replays. MS7e adds round, LDraw and flat families again behind `--families`.
- The parts drop in batches of five onto the bare desk and settle. They spread over the six-brick scatter zone in front of the arm and to its left and right. A part that slides out of the reach band stays there.
- `--parts_split {train,heldout,all}` picks the side of the stratified 20 percent held-out split, seeded by `--parts_seed`.
- Every part goes to one destination bin that is drawn per episode, and colors carry no meaning. The far wall of the bin is a 0.18 m backstop. It catches a part that slides off the tilted jaws at the release.

The teacher is `core/heap_oracle.py`. It ranks grasps on the part geometry (`core/grasp.py`) and takes the highest clear one. If no grasp clears, it takes the top-ranked one. It remembers failures per 5 cm cell. If a LIFT comes up with a part other than its target, the teacher carries that part to the bin.

A heap run exports every episode. `obs/actor` holds the ordinal of each pick cycle, negative for a cycle that did not stow. Thus `convert_to_lerobot.py --dagger --split_cycles` keeps the good cycles. `--dagger` also drops the success-attribute filter. A heap episode sets that attribute only after all its parts stowed. A heap run's `BATCH_OK` count therefore gives the episodes written, not the episodes where every part stowed.

The heap contexts are wider:

- `--context camera_box` writes the overhead camera pose and then the destination box pose as one 9-wide `observation.environment_state`.
- `--context camera_box_target` appends the 7-wide target row, for 16 in all. The row (`obs/context_target`) holds the center of the target part at the cycle start and the teacher's grasp. The grasp is the closing-axis heading as cos and sin, the width and the half height. The heap loop writes the row at selection, and the rig stashes it with the cycle part.

Heap shards also carry `obs/images/overhead_depth` (uint16 millimeters, 0 is a hole), `obs/target_part`, `obs/context_box`, a `parts` attribute and `initial_state/destination_bin`. The overhead image carries a white marker on the target part.

`scripts/heap_yield_read.py` sums a set of `*.meta.json` sidecars into the phase A read.

```bash
uv run python scripts/mj_generate_batch.py --out_dir /tmp/h0 --attempts 12 --num_envs 12 \
    --heap_k 50 --parts_split heldout --probe                          # BATCH_OK exported=12 attempts=12
python3 scripts/heap_yield_read.py --meta '/tmp/h*/shard.meta.json' --out /tmp/read.json   # YIELD_OK {...}
```

### Dataset schema

Both sides use this schema (spec section 4.2, `docs/ms1-maker-arm-sim.md`):

| key | shape | content |
| --- | --- | --- |
| `observation.state` | (7,) | 6 URDF joint angles [rad] + gripper opening fraction (0 closed, 1 open) |
| `action` | (7,) | 6 URDF joint targets [rad] + binary gripper (`>= 0` open, `< 0` close) |
| `observation.images.wrist`, `observation.images.overhead` | 224x224x3 | uint8 RGB |
| `observation.images.overhead_depth` | 224x224 | uint16 millimeters, 0 is a hole (heap shards) |
| `observation.environment_state` | (6,), (66,), (9,) or (16,) | only with `--context`. (6,) is the camera pose (x, y, z, roll, pitch, yaw). (66,) holds per brick the position, rotation, target and stowed mask. (9,) is the camera pose and the destination box (x, y, yaw). (16,) adds the target row. |

Under `--action_space delta_prev`, the six joint columns of `action` hold this frame's clamped target minus the previous frame's target. On the first frame of a sub-episode they hold the measured joints. The converter records the choice in `meta/action_space.json`. At inference the eval client sends its previous clamped command as the `anchor` array, and the policy server adds it back. Thus the wire and the real arm still carry absolute joint targets. Both sides refuse the c1 definition (`delta`, the target minus the measured joints of the same frame).

Under `--gripper continuous` (MS6 phase B and later), the seventh action column is the jaw gap in meters: `0.0` closed and `0.104825` open. The dataset names it `gripper_gap_m`. The `action_space.json` of the checkpoint and the server handshake carry `gripper`. Every corpus before phase B is `binary`.

To convert a shard by hand:

```bash
lerobot_stack/.venv/bin/python lerobot_stack/convert_to_lerobot.py \
    --shards /tmp/batch0/shard.hdf5 --root ~/data/maker-arm-ms1/dataset   # CONVERT_OK episodes=<n> total=<n> read=<n> skipped=<n>
```

## Evaluation and gating

`scripts/run_gate_eval.py` is the gate supervisor. It is stdlib-only `python3`. It starts one policy server under the LeRobot venv and N eval clients, then sums their payloads. A six-brick read (`--heap_k 0`) runs `mj_eval_policy.py`, or `eval_policy.py` on Isaac. A heap read runs `mj_eval_heap.py`. The seeds of a read must be disjoint from the ledger. If the policy, the inference configuration and every switch match, the supervisor reuses a cached `eval_NNN.json`.

The inference protocol depends on the task:

- Six-brick reads serve under the MS3 protocol, `--n_action_steps 1 --temporal_ensemble_coeff 0.01`.
- Heap reads serve open loop with no temporal ensembling. On the heap, the MS3 protocol read 0.1868 against 0.3230 open loop (`docs/ms6-heap-policy.md` section 6).
- Since `docs/ms6-target-context.md`, heap reads execute 50 actions per forward (`--n_action_steps 50`).

### The heap gate

The heap gate drives `scripts/mj_eval_heap.py` through `core/loop.py`. A `HeapOracle` selects each target, the policy drives, and the privileged grader ends the cycle. With `--oracle_driver`, the rig drives every cycle and the read needs no policy server.

```bash
uv run python scripts/mj_eval_heap.py --num_envs 1 --heap_k 50 --parts_split heldout \
    --run_seed 20261300 --layout_seed 20261310 --oracle_driver --out /tmp/preflight.json
python3 scripts/run_gate_eval.py --backend mujoco --checkpoint <pretrained_model> \
    --ledger ~/data/maker-arm-ms6/l/ledger.json --threshold 0.85 --max_cycles_per_stowed 1.3 \
    --seed 20261400 --processes 8 --num_envs 5 --heap_k 50 --parts_split heldout \
    --episode_draws --frame_jitter --context camera_box --gripper continuous \
    --n_action_steps 50 --out_dir ~/data/maker-arm-ms6/l/gate
python3 scripts/heap_cycle_census.py \
    --evals "$HOME/data/maker-arm-ms6/l/gate/eval_[0-9][0-9][0-9].json"   # CENSUS_OK {...}
```

`heap_cycle_census.py` sums the cycle census of a read. It gives how far each failed cycle got and the stows of parts other than the target. It also gives the first close distance on a continuous run and the step count per end reason.

The heap loop switches go to the client through `--client_flags '...'`. The supervisor lists each switch in `LOOP_SWITCHES`, so its cache refuses a payload read with other switches. `run_gate_eval.py` forwards the MS7e and MS7f switches directly.

### The hold census (MS7c)

A real arm has no contact sensors. `core/hold.py` reads the hold and the lift from three signals only: the measured jaw gap, the commanded jaw gap and the FK pick point. The loop runs this detector beside the contact truth on every step. With `--perceived` off, no decision reads the detector.

Each census row records `hold_detected`, `lift_detected`, `first_hold_at` and `first_held_at`. Each payload has a `hold_agreement` block. `mj_eval_heap.py --hold_trace PATH` also writes a per-step trace as gzipped JSON lines.

`scripts/hold_agreement.py` reports the agreement over a set of payloads, and it can replay the detector over traces at other thresholds. It tests the pre-registered bar. Precision and recall must be 0.95 or more for the hold and for the lift. The median first-hold offset must be 5 steps or fewer.

```bash
uv run python scripts/hold_agreement.py --payloads '/tmp/gate/eval_*.json'   # HOLD_OK or HOLD_FAIL {...}
uv run python scripts/hold_agreement.py --traces '/tmp/gate/*.trace.jsonl.gz' --stable 0.0004
```

### The perceived state (MS7c)

`--perceived` puts one boundary between the simulator and the decisions of the loop. The decisions read only the perceived state of `core/perception.py` and the hold detector.

- The selector ranks the detected parts.
- The target row, a frozen marker and the box row come from the detection of the target at selection.
- The end of a lifted hold ends the cycle: `placed` over a perceived box, `dropped` anywhere else. The step budget also ends it.
- At a lift, the box row follows the class drawn for the part between the jaws.

The grader, the tracker and the census still read the truth, so the numbers compare with MS7a.

`--noise_scales` sets a scale per perception knob (`pos`, `heading`, `size`, `box`, `miss`, `class`, or `all`). A scale of 0 turns that model off, and 1 is the nominal. `--contact_hold` is the pre-registered fallback: the stop flag of the serving rules reads the contacts again. `--perceived` needs `--desk_boxes`, `--gripper continuous` and a policy server. Each payload has a `perception` block. `run_gate_eval.py` forwards the three flags, and its cache refuses a payload read with other values.

```bash
python3 scripts/run_gate_eval.py ... --tray --desk_boxes 4 --tray_lean 30 --perceived --noise_scales heading=2
```

### The perception levers (MS7d)

MS7d adds nine levers to the MS7c protocol. Each lever has its own switch on `mj_eval_heap.py`, and every switch is off by default. All switches except `--hold_force` and `--force_cap` need `--perceived`. The passing read uses all nine (`docs/ms7d-perception-levers.md`).

| Lever | Switch | What it removes |
| --- | --- | --- |
| L1 | `--box_filter_xy` | the in-box re-picks |
| L2 | `--release_gap` | the dead steps of a slow release |
| L3 | `--marker_bias` | the lateral bias per episode, with a marker on the gripper |
| L4 | `--wrist_fix` | the bias and most of the noise at the grasp, with a wrist camera look |
| L5 | `--class_views` | the parts in a wrong box, with a third class view |
| L6 | `--hold_force` | the false holds, from the torques of the arm's own motors (`core/torque.py`) |
| L7 | `--force_cap` | finger presses of up to 45 N |
| L8 | `--rearm_hold` | the dead steps after a false or failed hold |
| L9 | `--box_walls` | the failed re-picks beside a desk box |

### The widened task (MS7e)

MS7e widens the task. These switches are in `core/batch_loop.py`, and `run_gate_eval.py` forwards them:

- `--box_draw N` draws the box count of each episode, uniform from 4 to N.
- `--families round,ldraw,flat` adds the round parts, the LDraw parts of `assets/parts/` and the flat parts (plates and tiles). Any family also turns on the eight classes.
- `--deep_box_floor` extends the collider of each box floor under the desk.
- `--carry_via` and `--passenger_check` are adopted teacher levers. `--thin_snap` and `--wall_band_cont` are adopted serving levers.
- `--snap_closed`, with `--pick_snap`, ends the landing hold once the measured jaw gap is under the served width plus a margin. It fixes the flat pinches that the hold detector misses (`docs/ms7e-flat-study.md`).

`scripts/ldraw_parts.py` builds the LDraw JSON files from LDraw `.dat` files. It needs the `ldraw` dependency group (CoACD and trimesh), so run it in a throwaway venv. The runtime reads only the committed JSON. `scripts/mj_part_table.py` drops one part at a time into the tray and reads whether the teacher lifts and stows it.

### Real-scale sorting (MS7f)

MS7f adds 150-part heaps, six categories over 66 kinds, a classifier trained on renders, and speed. Each switch defaults off, and off is MS7e. The supervisor refuses a switch that is not in `MS7F_READY`. The main switches:

- `--categories` sorts by the six categories over the drawn boxes. It needs `--box_draw`.
- `--deep_tray` raises the tray walls and drops every part at the start. `--tray_refill` fills the tray with a share of the heap and refills it between cycles.
- `--classifier PATH` reads a TorchScript category classifier on three views. `--lift_view_steps`, `--carry_votes` and `--passenger_drop` tune the views and the vote.
- `--missed_lift` reads a carry that the hold detector missed. `--cell_failure_cap 5` sets the failure cap of the perceived selector.
- `--action_warp 1.5` on the supervisor makes the policy server play each action chunk faster.
- `--pile_pgs` solves a large pile with PGS and every other island with Newton. It stays out of the reads, because its paired screen falls outside the parity tolerance. It serves only the renders.

The classifier is trained on rendered frames and loaded in-process by the heap client on the CPU. `lerobot_stack/policy_server.py` serves only the ACT policy.

```bash
# 1. Render the classifier's overhead, lift and carry views on teacher cycles: RENDER_OK
MUJOCO_GL=egl uv run python scripts/mj_render_set.py --out_dir /tmp/rs --num_envs 1 --heap_k 50 \
    --parts_split train --run_seed 20302000 --layout_seed 20302000 ...   # full flag list: its docstring
# 2. Write the perceived size beside each render file: SIZES_OK
uv run python scripts/render_set_sizes.py /tmp/rs/render_*.npz
# 3. Train the classifier and export TorchScript: CLS_OK
lerobot_stack/.venv/bin/python lerobot_stack/train_part_classifier.py \
    --train /tmp/rs/render_*.npz --read /tmp/c1/render_*.npz --out /tmp/cls/model_v1
```

`scripts/mj_stretch_run.py` is the MS7f stretch run. It chains rollouts on one env that never resets and serves a pool of `--heap_k` parts through the tray refill. It counts the interventions per 1000 parts (`STRETCH_OK`).

### Heap DAgger

`mj_eval_heap.py --dagger_dir` serves a heap and records the takeover frames only (MS7f wave 3). Convert with `--dagger --split_cycles --context camera_box --gripper <mode>`.

### Reads across hosts

`scripts/fleet.py` runs a campaign across hosts. It uses only the standard library. It reads the host table `<campaign>/hosts.json`, and the default campaign is `~/data/maker-arm-ms7`.

- `sync` puts the campaign worktree of every host on one SHA from GitHub and runs a fast test subset.
- `ship` copies a directory or a file with rsync and compares the file lists at both ends.
- `run` starts a logged job in the worktree of a host.
- `read` splits one gate read across hosts by their lane caps. It pulls the payloads into one new TAG directory, refuses a payload older than the read, and then runs the aggregate pass.
- `status` prints the last token line of every job log.

A payload records the checkpoint path as its policy id. Thus every path must be the same absolute path on every host.

```bash
python3 scripts/fleet.py sync --sha "$(git rev-parse origin/main)"             # SYNC_OK {...}
python3 scripts/fleet.py ship <dir> aarch64:<dir> --streams 4                   # SHIP_OK {...}
python3 scripts/fleet.py run aarch64 <name> -- <command>                        # RUN_OK {...}
python3 scripts/fleet.py read --ckpt <abs pretrained_model> --tag <TAG> --seed 20261700 \
    --draws 0-11 --hosts c,a,b [-- <more run_gate_eval.py flags>]  # GATE_OK {...}
python3 scripts/fleet.py status                                                 # STATUS_OK
```

## Tracking-controller bench (MA2)

The bench tests the compliant tracking controller of maker-arm-rs (`maker_arm_rs.Tracking`) on an arm-only MuJoCo plant before the real arm runs it. `mj/bench.py` is the plant: the vendored URDF with gravity, a 1 ms step and the RobStride MIT law. `mj/bench_gate.py` holds the pre-registered scenarios and the ten bars. Two timing scripts run on the Jetson Thor under plain `python3`, with only stdlib and `maker_arm_rs`.

```bash
uv run --no-sync python scripts/mj_track_bench.py --out ~/data/maker-arm-tracking/gate-20260925.json   # BENCH_OK or BENCH_FAIL
python3 scripts/track_timing.py                     # Thor, T1: update() timing at 200 Hz; TIMING_OK
python3 scripts/track_soak.py --minutes 10 --rt 80:5  # Thor, T2: real 200 Hz loop, FIFO 80 on CPU 5; SOAK_OK
```

`scripts/export_housing_mass_model.py` wrote `assets/housing_mass_model.json` once. It needs trimesh: `uv run --no-sync --with trimesh python scripts/export_housing_mass_model.py`.

## Real arm (MS5)

`source/maker_arm_lab/real/` puts a real Maker Arm and its cameras behind the same env surface the heap loop already reads (MS5 design section 4.3), so `core/loop.py` and the served switches run unchanged on `RealEnv`. The arm runs `Tracking`, the SDK's contact-aware rung A, through `maker_arm_rs.Arm` (`real/arm.py`); the SDK refuses rung F, and the other bench rungs need a Python command every tick, so a real motor only turns on rung A until the owner rules on the rest. Perception is a separate worker over ZMQ (`perception_stack/perception_protocol.py`, the wire; `real/perceive.py`, the client), and the two cameras' latest frames are warped to the simulator's trained views by `real/camera_model.py`.

```bash
uv sync --group dev --group real     # adds PyYAML (rclpy) and opencv-python-headless (the wrist camera)
source /opt/ros/jazzy/setup.zsh      # zsh only; setup.bash silently drops PYTHONPATH there
```

The loop reads the overhead view and its depth from Conduit (MS5 design section 4.2, the Conduit design `2026-09-26-ar-aligned-depth` section 5): namespace `/conduit`, frame `camera_back_optical_frame`, and four topics, `camera/wide/image_raw/compressed` (JPEG), `camera/wide/camera_info`, `camera/wide/depth/image_raw` (16UC1 mm, 0 invalid) and `camera/wide/depth/camera_info`. `real/ros.py` is the only module that subscribes to them.

```bash
# The sheet: where the operator places the four desk boxes, and which class goes to which box.
uv run python scripts/real_episode_sheet.py --seed 20265000 --episode 0 \
    --out ~/data/maker-arm-ms5/sheets/e000.json   # SHEET_OK, then the sheet as JSON

# The heap gate client: one process runs one real episode under the MS7d served switches.
source /opt/ros/jazzy/setup.zsh
uv run python scripts/real_eval_heap.py --rig ~/data/maker-arm-ms5/rig.json \
    --sheet ~/data/maker-arm-ms5/sheets/e000.json --heap_k 10 --dilation 0.5 \
    --run_seed 20265100 --out ~/data/maker-arm-ms5/reads/e000.json   # EVAL_OK or EVAL_FAIL

# P7: the Conduit stream's stamp and photon latency methods.
uv run python scripts/conduit_latency.py stamp --seconds 300 \
    --out ~/data/maker-arm-ms5/p7/stamp.json
uv run python scripts/conduit_latency.py photon --on_cmd '<light on>' --off_cmd '<light off>' \
    --roi 900,600,1020,720 --cycles 30 --out ~/data/maker-arm-ms5/p7/photon.json   # LATENCY_OK
```

The rig's geometry and session settings are one JSON file, `real.rig.RigConfig` (`--rig`; the default is the nominal config). Five fields are placeholders a named step still measures: `overhead_pos`, `overhead_quat`, `wrist_rot` (the hand-eye rotation) and `wrist_device` wait on A2, and `tracking_params` waits on A1. `overhead_k` and `overhead_size` may stay unset, because Conduit's camera info supplies them. `wrist_k` and `wrist_size` are not calibrated yet: unset, the wrist source expects the simulated 224x224 view, so a real UVC camera needs both set, or every frame of another size is dropped and the frame-age watchdog holds the arm. `PICK_FLOOR_MARGIN_M` in `real/env.py` and the gripper motor's linear angle-to-opening map in `real/arm.py` are placeholders the same way, on A0 and A1 respectively.

A frame older than 300 ms (counted from its arrival) pauses the episode instead of ending it: the arm holds its last row, and the loop resumes once both cameras deliver a fresh frame, or holds and ends with `EVAL_FAIL reason=watchdog_frame` after a 2 s pause (`STALE_WAIT_S` in `real/env.py`). The payload's `real.stale_pauses` counts the pauses. Over the home Wi-Fi the Conduit stream drops past 300 ms about once every few minutes.

`real_eval_heap.py --release` controls what happens to the motors once the token is printed: `prompt` (the default) holds the arm until the operator types `RELEASE` or closes stdin, and a Ctrl-C at the prompt only re-asks — it never releases; `now` disables at once, for the SDK simulator and tests. A Ctrl-C during the episode itself ends it (`EVAL_FAIL reason=interrupted`) and holds the arm the same way a watchdog would, through the same prompt.

`RealEnv` is a blind scene (MS5 design section 4.3): it has no simulator truth, so its stowed/cycle metrics never fire and a payload's `real.graded` is always `false`. The payload is ungraded until the operator counts the boxes against the episode sheet.

## Training smoke

```bash
STEPS=100 OUT=~/data/maker-arm-ms1/train/smoke JOB_NAME=act-smoke WANDB_ENABLE=false \
  bash lerobot_stack/train_act.sh                                    # TRAIN_OK out=<dir> steps=<n>
```

The default run is online: `WANDB_ENABLE=true` with `WANDB_MODE=online`. For an online run, set `WANDB_ENTITY` explicitly. If the run is online and `WANDB_ENTITY` is empty, the wrapper refuses to launch, so it never falls through to a default entity. With `WANDB_ENABLE=false` or `WANDB_MODE=offline`, as in the smoke above, an empty entity is accepted.

## Reports

The reports are in milestone order.

- [`docs/ma0-bench-prep.md`](docs/ma0-bench-prep.md): MA0, the bench preparation. It was written before MS1 and not read again against it. Read it as the MA0 record. If it disagrees with MS1 about the arm, the measured constants of MS1 are the later reading.
- [`docs/ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md): MS1, the LegoSort simulation on the vendor URDF. It records the asset pipeline, the measured robot constants, the validation ladder and the acceptance gate. The gate failed on the oracle success rate, 30/50 = 0.600 against a 0.70 bar. The report also lists the sim/real divergences and the unverified constants for the hardware bring-up.
- [`docs/ms2-act-gate.md`](docs/ms2-act-gate.md): MS2, the first ACT training and rollout gate on Isaac. It uses a 1034-episode pick-order corpus, plain ACT 600k and rule L3 selection. The official gate failed at 52/720 = 0.072 against 0.15. The report gives the factor account and the ranked levers.
- [`docs/ms3-mujoco-gate.md`](docs/ms3-mujoco-gate.md): MS3, the MuJoCo port and its parity checks. The MS2 recipe failed there at 55/720 = 0.076 against 0.70, inside the interval of Isaac. The recipe corpus failed at 323/720 = 0.449. The t3 corpus read 484/720 = 0.672, and the Phase 4 corpus t4 passed at 605/720 = 0.840. The report also records the planner in Rust and the smooth via-point oracle.
- [`docs/ms4-corpus-levers.md`](docs/ms4-corpus-levers.md): MS4, three corpus and schema levers on one 1039-episode corpus, against a 0.90 gate. The camera context is adopted: 600/720 = 0.833, and 260/360 = 0.722 at `--spatial_scale 3.0` against 0.600 for the base. The `delta` action space is rejected at 58/720 = 0.081. The privileged ceiling reads 8/360 = 0.022 and is not a valid ceiling.
- [`docs/ms4-dagger.md`](docs/ms4-dagger.md): MS4 round c2. DAgger round 1 passes at 660/720 = 0.9167, the first `GATE_OK` of MS4. Round 2 reads 680/720 = 0.9444, 2 bricks short of its adoption bar, so the rounds stop. The `delta_prev` action space is rejected at 93/720 = 0.129.
- [`docs/ms6-heap-teacher.md`](docs/ms6-heap-teacher.md): MS6 phase A, the heap teacher on a reach-corrected tray. The pre-registered teacher read failed: 34/6000 = 0.0057 stowed against at least 0.90, at 7.24 cycles per stowed part against at most 1.3. After the fixes of the inspection, a repeated read on fresh seeds failed again at 1619/6050 = 0.2676 and 2.176 cycles per stowed part.
- [`docs/ms6-teacher-inspection.md`](docs/ms6-teacher-inspection.md): the inspection that followed, with a decision log and a video per episode. It names four mechanisms: the clearance model, the open jaw against the tray walls, the flat-part descend and the LIFT gate on narrow parts. The fixed teacher reads 15, 11 and 17 of 50, against 0 of 50 three times before.
- [`docs/ms6-lever-verification.md`](docs/ms6-lever-verification.md): the study of the tray, the drop and the pick primitive. The tray lever buys 3.6 of 50 fewer wall-blocked parts. The bottom-open pick stows 45 of 150 against 43 of 150 at the corpus flags. Without the hover jitter band, it stows 67 of 150 against 32 of 150.
- [`docs/ms6-scene-and-grasp-rework.md`](docs/ms6-scene-and-grasp-rework.md): a six-rung scene and grasp ladder at 150 parts per rung. The lift decision (R1, 50/150 = 0.3333) and the wide bin (R3, 49/150 = 0.3267) are adopted. The tip-up (R4) and the tilted pick (R5) read 16/150 = 0.1067 and ship disabled.
- [`docs/ms6-matched-jaw-opening.md`](docs/ms6-matched-jaw-opening.md): a teacher ladder that matches the jaw opening to the part. The pick opening (55/150 = 0.3667) and the release opening (76/150 = 0.5067) are adopted. The grip rung reads 73/150 = 0.4867, under the release rung.
- [`docs/ms6-bare-desk-heap.md`](docs/ms6-bare-desk-heap.md): the ladder from the bare desk at 0.633 to 0.933 held-out and 0.940 train-split stowed, against a goal of 0.80. It removes the pushes, drops in batches over the whole scatter zone, retires round bricks and wedge slopes, and adds a backstop wall. It also records the converter defect that dropped heap episodes.
- [`docs/ms6-heap-policy.md`](docs/ms6-heap-policy.md): MS6 phase B, ACT on the bare-desk heap with the continuous gripper. The best read is 0.4297 on 3000 parts against a 0.85 gate, a fail. DAgger round 1 misses its adoption bar. The report finds the inference protocol defect and rules that the heap serves open loop.
- [`docs/ms6-target-context.md`](docs/ms6-target-context.md): the second phase B lever. It adds the 16-wide `camera_box_target` context, moves the cycle boundary to the teacher's, and serves 50 actions per forward. The 400k checkpoint reads 0.6320 on 3000 held-out parts against a goal of 0.80. It records the stale-row mechanism.
- [`docs/ms6-square-wrist.md`](docs/ms6-square-wrist.md): campaign n, the third phase B lever. It adds the next target at the release, a squeeze floor and a wrist squaring that serve from the target row. The checked read meets the goal of 0.80 at 2711/3000 = 0.9037, and the official seed reads 2759/3000 = 0.9197.
- [`docs/ms7-sorting.md`](docs/ms7-sorting.md): MS7a. Rung 1 stows a tray heap in one desk box, and rung 2 sorts it into four desk boxes by four part classes. The teachers pass at 0.920, 0.963 and 0.905 on 600 parts. On a tray whose walls lean out 30 degrees, the official read passes at 5205/6000 = 0.8675 at 1.134 cycles per correct stow. On vertical walls, the same checkpoint reads 2513/3000 = 0.838. The lean is a deviation from the design. The report also records `scripts/fleet.py` on three hosts.
- [`docs/ms7c-perceived-state.md`](docs/ms7c-perceived-state.md): MS7c, the loop on a perceived state and a hold detector from the jaw gap. The best read, candidate 3, stows 2646 of 3000 (0.882) at 1.329 cycles per correct stow. It fails the 1.3 cycles bar, so the official read did not run. The same read on true inputs gives 2770 of 3000 (0.923) at 1.160.
- [`docs/ms7d-perception-levers.md`](docs/ms7d-perception-levers.md): MS7d, nine levers behind their own switches on the MS7c gate. The official read passes at 5521/6000 = 0.920 at 1.123 cycles per correct stow, with no retrain.
- [`docs/ma2-tracking-bench.md`](docs/ma2-tracking-bench.md): MA2, the bench gate of the tracking controller. It prints `BENCH_FAIL` with 8 of 10 bars passed. The overshoot reads 0.3575 mm against 0.1713 mm, and the press force 1.1269 N against 1.1847 to 1.7771 N. An erratum traces both to joint and tip friction. Both Thor bars pass: the p99 of `update()` is 26.815 us, and the largest tick interval is 5.1792 ms.
- [`docs/ms7e-widened-task.md`](docs/ms7e-widened-task.md): MS7e, 4 to 8 drawn boxes, 8 classes and the round, LDraw and flat families. The official read of candidate E2 passes at 5299/6000 = 0.883 at 1.271 cycles per correct stow. The flat parts take about two thirds of the excess cycles.
- [`docs/ms7e-flat-study.md`](docs/ms7e-flat-study.md): why the plates and tiles cost cycles. If the hold detector misses a thin part, the serving rule `snap_hold` opens the jaws again. `--snap_closed` fixes it: 925 against 912 at 1.135 against 1.242 cycles per correct stow in the pre-registered row. The toe pad is rejected.
- [`docs/ms7f-real-scale-sorting.md`](docs/ms7f-real-scale-sorting.md): MS7f, 150-part heaps, six categories over 66 kinds, a classifier trained on renders, and speed. The staged bars are 0.85 share, 200 correct stows an hour and 0.80 held-out. The final candidate 6 reads 2522 of 3000 (0.841) at 308.0 an hour in the refilled deep tray, and 620 of 785 (0.790) held-out. The owner closed MS7f as failed (D25), and the official read did not run. Wrong boxes hold 342 of the 478 misses.
- [`docs/ms7j-real-arm-readiness.md`](docs/ms7j-real-arm-readiness.md): MS7j, the readiness suite and the π0.5 LoRA. S passes every row at the levels that the rig expects, so no retrain ran. The report gives the calibration and mount limits for MS5, for example an uncorrected overhead tilt of 1 degree at most. The LoRA trains on the RTX 5090 in 30 h 46 min. Its 30k save equals the full fine-tune on the screen and is faster. Chunked π0.5 fails an inference delay of 6 and 12 steps on both candidates. Real-time chunking passes at 12 steps for the fresh 30k and at 6 steps for the LoRA.
- [`docs/ms7k-pi05-accuracy-sim2real.md`](docs/ms7k-pi05-accuracy-sim2real.md): MS7k, π0.5 against S on accuracy and sim2real. Stage 1 found no training lever better than a plain 5000-step continuation of the LoRA 30k. A 10000-step Colab run past it gave no gain. The continuation, C, beats S on correct placements under real-arm serving by 1.2 pooled SE and loses less than S on the sim2real rows. In the suite, C tolerates an uncorrected overhead camera up to 6 degrees and +5 cm off, where S needs 1 degree and 1 cm. The report also records the Colab LoRA path on LeRobot 0.6.1 and the MS5 tolerance columns for both policies.
