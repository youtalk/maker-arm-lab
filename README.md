# maker-arm-lab

Research layer for the [MakerMods Maker Arm](https://github.com/makermods-robotics/maker-arm-sdk): simulation, system identification, model learning, and gated milestone experiments. The aim is sim-to-real — an Isaac Lab environment built on the vendor URDF, generating data and training policies in the Maker Arm's own joint coordinates so the result can be deployed on the physical arm through the [maker-arm-rs](https://github.com/youtalk/maker-arm-rs) SDK. Gains, torque caps and the command clamp are read from `maker_arm_rs.profile()` at import time rather than restated here, so a recorded action is by construction a command the real arm accepts; there is no fallback table, and a missing `maker_arm_rs` fails at import. Two tracks run in this repository: **MA0–MA5**, the hardware bring-up and identification milestones (the arm arrives October 2026), and **MS1**, the LegoSort simulation environment, which is complete — see [`docs/ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md) for its results, including a **failed** acceptance gate — followed by **MS2**, the first ACT training and rollout gate on that environment, also complete and also a **failed** gate (per-brick 0.072 against 0.15; [`docs/ms2-act-gate.md`](docs/ms2-act-gate.md)), and [`docs/ma0-bench-prep.md`](docs/ma0-bench-prep.md) for the bench preparation. **MS3** followed: the MuJoCo port of that environment, with the MS2 recipe re-baselined on it at per-brick 0.076 against 0.70, within Isaac's interval. The openyam-lab recipe corpus then lifted the number to 0.449, still a **failed** gate. Many short runs, a new light rig and a nearest-first pick order lifted it to 0.672, and a second tranche of wide corrective approaches to 0.840, a **passed** gate. Its report, [`docs/ms3-mujoco-gate.md`](docs/ms3-mujoco-gate.md), also covers the planner in Rust and the smooth via-point oracle.

## Layout

```
source/maker_arm_lab/
  robots/            frames.py (measured constants, isaaclab-free) + maker_arm.py (ArticulationCfg)
  actuators/         MIT-mode actuator; gains and caps from maker_arm_rs.profile()
  assets/            urdf_prep.py (Z-up derived URDF), GravityCompensatedArticulation
  core/              layout, oracle state machine, global_ik, checker, gate, profile, trace,
                     parts, grasp, heap_oracle
  mj/                MuJoCo backend: env, scene, recorder, overlay, depth
  tasks/manager_based/lego_sort/
                     env cfgs and the five task ids, mdp/, oracle_runtime, recorders, vision
assets/vendor/maker_arm_description/   pinned copy of maker-arm-sdk urdf/maker_arm (+ VENDOR.md)
assets/usd/                            generated maker_arm.usd -- NOT committed
lerobot_stack/                         HDF5 -> LeRobot conversion, policy protocol/server, train wrapper
scripts/                               setup, vendor, convert, smokes, checks, batch generation
tests/                                 CPU-only pytest (no isaaclab import at collection time)
docs/                                  milestone reports
```

## Environment setup

Three environments, deliberately separate. `lerobot_stack/` is not collected by `uv run pytest`.

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

Every GPU command below needs the Isaac venv activated first — `isaaclab.sh -p` fails with `python: command not found` otherwise:

```bash
source "$ISAACLAB_PATH/.venv/bin/activate"
```

## Simulation asset build

The simulator loads `assets/usd/maker_arm.usd`, which is **generated and git-ignored**:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/convert_urdf_to_usd.py --headless    # CONVERT_OK
```

Run this once after cloning, and again whenever `scripts/convert_urdf_to_usd.py`, `source/maker_arm_lab/assets/urdf_prep.py` or the vendored URDF changes. Nothing checks freshness for you: a stale USD loads without complaint and then misbehaves physically rather than failing.

The vendored URDF itself is committed and is **never edited**. To re-vendor at a new upstream commit, bump `PIN_SHA` in the script, re-run it, and re-run the validation test:

```bash
bash scripts/vendor_maker_arm_description.sh
uv run pytest tests/test_urdf_validation.py
```

## Registered task ids

The last column is what each script's `--task` default actually selects, read off the scripts rather than off the spec's original assignment — the two diverged when the oracle migrated. Since MS4 the Isaac task is frozen at its MS3 features and smoke-tested only; every new capability lands on the MuJoCo backend.

| Task id | Arm action | Cameras | Runs on it today |
| --- | --- | --- | --- |
| `MakerArm-LegoSort-v0` | clamped absolute joint position | — | `smoke_lego_sort.py` (default), `scripted_grasp_check.py` (default) |
| `MakerArm-LegoSort-IK-Rel-v0` | task-space delta pose (differential IK) | — | no check by default; registered, and smoke-tested via `--task` |
| `MakerArm-LegoSort-IK-Rel-Vision-v0` | task-space delta pose | wrist, overhead | no check by default; registered, and smoke-tested via `--task` |
| `MakerArm-LegoSort-Datagen-v0` | clamped joint position + vision + domain randomisation + recorders | wrist, overhead | `generate_batch.py`, `release_envelope_check.py`, `oracle_check.py` — all three by default |
| `MakerArm-LegoSort-Eval-v0` | clamped joint position + vision + eval terminations | wrist, overhead | nothing yet; reserved for the next milestone's rollout gate, smoke-tested via `--task` |

`Datagen-v0` moved from the IK-Rel action space to clamped joint position in MS1, when the oracle moved onto `core/global_ik.py`, and the envelope and oracle checks moved with it — which is why the two IK-Rel ids no longer back a gate even though the spec originally assigned them one. They stay registered and smoke-tested so the differential-IK path remains available and cannot rot unnoticed. The dataset contract is unchanged by the move: the recorded action stream was always the post-IK joint target. See `docs/ms1-maker-arm-sim.md` section 4.3.

## Checks

Every GPU script prints one token line (`<NAME>_OK` / `<NAME>_FAIL`) and exits through `os._exit` with a real code, on **every** exit path including argparse usage errors. A tokenless exit is a defect.

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

`release_envelope_check.py`'s single-process five-layout loop wedges after the second `gym.make`, so run one process per layout and read the five per-layout tokens. It also fixes the pick at (0.20, 0.0) by design, so it samples `core.layout.SCATTER_MAX_RADIUS` at one azimuth only and is not an independent confirmation of that bound — and if you change what it sweeps, sweep **tracking**, not IK feasibility.

## Data generation and conversion

One batch, for a probe or a smoke:

```bash
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/generate_batch.py --headless --enable_cameras \
    --out_dir /tmp/batch0 --attempts 10 --num_envs 12 --probe        # BATCH_OK exported=<n> attempts=<m>
```

A full campaign — generation and LeRobot conversion run by run, with a ledger and per-run crash markers so an interrupted campaign resumes without duplicating episodes. `generate_dataset.py` is stdlib-only `python3` and needs no virtualenv of its own; it launches `generate_batch.py` under `isaaclab.sh` and `convert_to_lerobot.py` under the LeRobot venv. Daemonise it — this host sweeps long background jobs, and `setsid` alone is not always sufficient:

```bash
setsid bash -c 'cd "$PWD" && python3 scripts/generate_dataset.py \
  --out_dir ~/data/maker-arm-ms1/work --dataset_root ~/data/maker-arm-ms1/dataset \
  --target 50 --attempts_per_run 50 --num_envs 12 --seed 20260905 \
  > ~/data/maker-arm-ms1/supervisor.log 2>&1' < /dev/null &
# per-run BATCH_OK, then CONVERT_OK, then CAMPAIGN_OK total=<n> runs=<m>
```

Add `--resume` to continue an interrupted campaign; it refuses rather than duplicating if a run was killed between converting and ledgering.

MS4 adds a set of corpus levers. They are MuJoCo-only. `--episode_draws` redraws the bins, the lights and the cameras at every episode reset instead of once per run. `--frame_jitter` resamples a small overhead-camera jitter on every rendered frame. Both are flags of `mj_generate_batch.py` and of `generate_dataset.py`, and the Isaac batch script accepts and ignores them. `--spatial_scale` multiplies the camera-pose and `bin_x` jitter of the run draw and of every per-episode draw, so `--spatial_scale 3.0` on `mj_eval_policy.py` or on `run_gate_eval.py` scores a policy well outside the distribution it trained on. `--context {none,camera,privileged}` on the same two picks which extra channel the env serves to the policy, the overhead camera pose or the six brick poses, and it has to match the checkpoint. On the conversion side `--context {camera,privileged,camera_box}` writes that channel into the dataset, `--no_images` drops both image features for the privileged-state ceiling, and `--action_space {absolute,delta_prev}` chooses what the six joint columns of `action` hold. The trainer takes the same choice as `ACTION_SPACE`, refuses a dataset that disagrees with it, and records it next to the checkpoints. Because one MS4 campaign is converted three times over, `generate_dataset.py --keep_shards` leaves every converted shard on disk instead of deleting it, so the later conversions have something to read.

MS4 round c2 adds DAgger with the oracle as expert. `scripts/mj_dagger_batch.py` rolls a policy out and lets the oracle rig take an env over on a trigger. Two triggers watch the jaws: no close for 400 steps, or a close far from every brick. Two more watch the grasp: closed jaws with no pinch, or the grasp center outside the workspace. A fifth fires at random, at a mean interval of 300 steps that doubles each round. A takeover runs one brick cycle from the state the policy reached, then hands the env back. The recorder keeps only takeover frames. `obs/actor` holds the takeover ordinal for each kept frame. When a cycle does not stow its brick, its ordinal goes negative. `convert_to_lerobot.py --dagger --split_cycles` keeps the positive-actor cycles. `lerobot_stack/probe_held.py` is the realizability probe that gates round 1. The ceiling lanes of c2 record with `generate_dataset.py --no_cameras --context privileged --no_images`.

```bash
uv run python scripts/mj_dagger_batch.py --num_envs 12 --run_seed 1 --layout_seed 2 \
    --permutation 0,1,2,3 --episode_draws --frame_jitter --context camera \
    --server tcp://127.0.0.1:5700 --round 1 --out_dir /tmp/dag --shard_name proc_0
```

MS6 phase A adds a heap of mixed parts on the desk, MuJoCo only. `--heap_k K` on `mj_generate_batch.py` replaces the six bricks with K parts drawn from a 29-part parametric catalog: bricks from 1x1 to 4x8, tall bricks up to 1x2x5 and 45-degree slopes with two or more studs along the run. Plates, tiles and round bricks left the catalog on 2026-09-18 (`docs/ms6-bare-desk-heap.md`) and stay constructible for replays. The parts drop in batches of five onto the bare desk, spread over the six-brick scatter zone in front of the arm and to its left and right, and settle. There is no dish and no box: a part that slides out of the reach band stays there. `--parts_split {train,heldout,all}` picks the side of the stratified 20 percent held-out split, seeded by `--parts_seed`. Every part goes to one destination bin drawn per episode, and colors carry no meaning. The bin's far wall is a 0.18 m backstop that catches a part sliding off the tilted jaws at the release. The teacher is `core/heap_oracle.py`: it ranks grasps on the part geometry (`core/grasp.py`), takes the highest clear one, takes the top-ranked one anyway when nothing clears, and remembers failures per 5 cm cell. A LIFT that comes up holding a part other than its target carries that part to the bin. A heap run exports every episode. `obs/actor` holds the pick cycle ordinal, negative for a cycle that did not stow, so `convert_to_lerobot.py --dagger --split_cycles` keeps the good cycles. `--dagger` also drops the success-attribute filter, because a heap episode only sets that attribute when every one of its parts stowed. `--context camera_box` writes the overhead camera pose followed by the destination box pose as one 9-wide `observation.environment_state`. `--context camera_box_target` appends the 7-wide target row (`obs/context_target`: the target part's centre when the cycle opened, the closing-axis heading of the teacher's grasp as cos and sin, its width and half height), 16 wide; the heap loop writes the row at selection and the rig stashes it with the cycle part. Heap shards also carry `obs/images/overhead_depth` (uint16 millimeters, 0 is a hole), `obs/target_part`, `obs/context_box`, a `parts` attribute and `initial_state/destination_bin`, and the overhead image carries a white marker on the target part. `scripts/heap_yield_read.py` sums a set of `*.meta.json` sidecars into the phase A read. `scripts/mj_inspect_heap.py` replays one heap episode with the teacher's decisions logged to `events.jsonl` and a free-camera video, and `scripts/heap_inspect_read.py` sums those logs (`docs/ms6-teacher-inspection.md`).

```bash
uv run python scripts/mj_generate_batch.py --out_dir /tmp/h0 --attempts 12 --num_envs 12 \
    --heap_k 50 --parts_split heldout --probe                          # BATCH_OK exported=12 attempts=12
python3 scripts/heap_yield_read.py --meta '/tmp/h*/shard.meta.json' --out /tmp/read.json   # YIELD_OK {...}
uv run python scripts/mj_inspect_heap.py --out_dir /tmp/i0 --heap_k 12 --parts_split heldout   # INSPECT_OK {...}
python3 scripts/heap_inspect_read.py /tmp/i0/events.jsonl                                      # INSPECT_READ {...}
```

Design H7 exports every episode and marks one actor ordinal per pick cycle. The converter's positive-actor rule then keeps only the cycles that stowed. A heap run's `BATCH_OK` count therefore reports episodes written, not episodes where every part stowed.

Dataset schema, both sides (spec section 4.2; `docs/ms1-maker-arm-sim.md`):

| key | shape | content |
| --- | --- | --- |
| `observation.state` | (7,) | 6 URDF joint angles [rad] + gripper opening fraction (0 closed, 1 open) |
| `action` | (7,) | 6 URDF joint targets [rad] + binary gripper (`>= 0` open, `< 0` close) |
| `observation.images.wrist`, `observation.images.overhead` | 224x224x3 | uint8 RGB |
| `observation.images.overhead_depth` | 224x224 | uint16 millimeters, 0 is a hole (MS6 heap shards, converted in phase B) |
| `observation.environment_state` | (6,), (66,) or (9,) | camera pose x, y, z, roll, pitch, yaw, or per brick position, rotation, target, and stowed mask, or camera pose plus destination box x, y, yaw, only with `--context` |

Under `--action_space delta_prev` the six joint columns of `action` are this frame's post-clamp target minus the previous frame's target, and the measured joints on the first frame of a sub-episode. The converter records the choice in `meta/action_space.json`. At inference the eval client sends its previous clamped command as the `anchor` array and the policy server adds it back, so the wire and the real arm still carry absolute joint targets. The c1 definition (`delta`, target minus the measured joints of the same frame) is refused by both sides.

Under `--gripper continuous` (MS6 phase B) the action's seventh column is the jaw gap in meters, `0.0` closed and `0.104825` open. The dataset names it `gripper_gap_m`. The checkpoint's `action_space.json` and the server handshake carry `gripper`. Every corpus before phase B is `binary`.

To convert a shard by hand:

```bash
lerobot_stack/.venv/bin/python lerobot_stack/convert_to_lerobot.py \
    --shards /tmp/batch0/shard.hdf5 --root ~/data/maker-arm-ms1/dataset   # CONVERT_OK episodes=<n> total=<n> read=<n> skipped=<n>
```

## Evaluation and gating

The heap gate (MS6 phase B) drives `scripts/mj_eval_heap.py` through `core/loop.py`. A `HeapOracle` selects each target. The policy drives. The privileged grader ends the cycle. With `--oracle_driver`, the rig drives every cycle instead of the policy.

```bash
uv run python scripts/mj_eval_heap.py --num_envs 1 --heap_k 50 --parts_split heldout \
    --run_seed 20261300 --layout_seed 20261310 --oracle_driver --out /tmp/preflight.json
python3 scripts/run_gate_eval.py --backend mujoco --checkpoint <pretrained_model> \
    --ledger ~/data/maker-arm-ms6/l/ledger.json --threshold 0.85 --max_cycles_per_stowed 1.3 \
    --seed 20261400 --processes 8 --num_envs 5 --heap_k 50 --parts_split heldout \
    --episode_draws --frame_jitter --context camera_box --gripper continuous \
    --n_action_steps 1 --temporal_ensemble_coeff 0.01 --out_dir ~/data/maker-arm-ms6/l/gate
python3 scripts/heap_cycle_census.py \
    --evals "$HOME/data/maker-arm-ms6/l/gate/eval_[0-9][0-9][0-9].json"   # CENSUS_OK {...}
```

Every gate read and every DAgger rollout serves the policy under the MS3 inference protocol, `--n_action_steps 1 --temporal_ensemble_coeff 0.01`. The supervisor passes the two flags to the policy server. A reused draw must report the same inference configuration in its payload. Because the temporal ensemble state lives in the server, a DAgger round starts one policy server per client. `heap_cycle_census.py` sums the cycle census of a read: how far each failed cycle got, the stows of a part other than the target, the first close distance on a continuous run and the step count per end reason.

Round k rolls out the previous round's policy on train-split heaps and records takeover frames only. The conversion flags are `--dagger --split_cycles --context camera_box --gripper <mode>`.

```bash
uv run python scripts/mj_dagger_heap.py --num_envs 5 --heap_k 50 --parts_split train \
    --run_seed 20261510 --layout_seed 20261610 --episode_draws --frame_jitter \
    --context camera_box --gripper continuous --server tcp://127.0.0.1:5700 --round 1 \
    --out_dir ~/data/maker-arm-ms6/l/dagger/r1 --shard_name proc_0
```

`scripts/fleet.py` runs the MS7a campaign across hosts. It uses only the standard library. It reads the host table `<campaign>/hosts.json`, and the default campaign is `~/data/maker-arm-ms7`. `sync` puts the campaign worktree of every host on one SHA from GitHub and runs a fast test subset. `ship` copies a directory or a file with rsync and compares the file lists at both ends. `run` starts a logged job in the worktree of a host. `read` splits one gate read across hosts by their lane caps. It pulls the payloads into one new TAG directory and refuses a payload older than the read. Then it runs the aggregate pass. `status` prints the last token line of every job log. A payload records the checkpoint path as its policy id, so every path must be the same absolute path on every host.

```bash
python3 scripts/fleet.py sync --sha "$(git rev-parse origin/main)"             # SYNC_OK {...}
python3 scripts/fleet.py ship <dir> aarch64:<dir> --streams 4                   # SHIP_OK {...}
python3 scripts/fleet.py run aarch64 <name> -- <command>                        # RUN_OK {...}
python3 scripts/fleet.py read --ckpt <abs pretrained_model> --tag <TAG> --seed 20261700 \
    --draws 0-11 --hosts c,a,b [-- <more run_gate_eval.py flags>]  # GATE_OK {...}
python3 scripts/fleet.py status                                                 # STATUS_OK
```

## Training smoke

```bash
STEPS=100 OUT=~/data/maker-arm-ms1/train/smoke JOB_NAME=act-smoke WANDB_ENABLE=false \
  bash lerobot_stack/train_act.sh                                    # TRAIN_OK out=<dir> steps=<n>
```

For an **online** run — the default, `WANDB_ENABLE=true` with `WANDB_MODE=online` — set `WANDB_ENTITY` explicitly. In that case only, the wrapper refuses to launch with an empty `WANDB_ENTITY` rather than silently falling through to a default entity; with `WANDB_ENABLE=false` or `WANDB_MODE=offline` (as in the smoke above) an empty entity is fine and the guard does not fire.

## Reports

- [`docs/ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md) — MS1: the LegoSort simulation environment on the vendor URDF. Asset pipeline, the measured robot constants, the full validation ladder, the acceptance gate (**failed** on the oracle success rate, 30/50 = 0.600 against a 0.70 bar) with its diagnostic hand-off, and the sim/real divergences and unverified constants the October hardware bring-up needs.
- [`docs/ms2-act-gate.md`](docs/ms2-act-gate.md) — MS2: the first ACT training and rollout gate on the Isaac environment. A 1034-episode pick-order corpus, plain ACT 600k and rule L3 selection. The official gate **failed** at 52/720 = 0.072 against 0.15, and the report carries its factor account and ranked levers.
- [`docs/ms3-mujoco-gate.md`](docs/ms3-mujoco-gate.md) — MS3: the MuJoCo port of the LegoSort scene and its parity checks. The re-baseline of the MS2 recipe there **failed** at 55/720 = 0.076 against 0.70, within Isaac's interval. The report also records the planner in Rust and the smooth via-point oracle adopted on a speed ladder. The recipe corpus gate is there too: 323/720 = 0.449, **failed** against 0.70. The t3 corpus then read 484/720 = 0.672, and the Phase 4 corpus t4 **passed** at 605/720 = 0.840, which closes MS3.
- [`docs/ma0-bench-prep.md`](docs/ma0-bench-prep.md) — MA0: bench preparation. Predates MS1 and was **not** re-verified against it, so read it as the MA0 record rather than as current: where the two disagree about the arm, MS1's measured constants are the later reading.
- [`docs/ms6-heap-teacher.md`](docs/ms6-heap-teacher.md) — MS6 phase A: the heap teacher (a part catalog, grasp synthesis and an `EnvOracle` subclass with a failure memory and a push) over a mixed-part heap on a reach-corrected tray (design H1). The pre-registered teacher read **failed** on both conditions: 34/6000 = 0.0057 stowed against a rule of at least 0.90, and 7.24 cycles per stowed part against a rule of at most 1.3. The report names three candidate mechanisms and stops the plan before phase B. After the teacher-logic fixes from `docs/ms6-teacher-inspection.md`, the repeated read on fresh seeds **failed** again: 1619/6050 = 0.2676 stowed and 2.176 cycles per stowed part, against the same rule.
- [`docs/ms6-teacher-inspection.md`](docs/ms6-teacher-inspection.md) — the inspection that followed: a decision log and a video per episode, four mechanisms (the clearance model, the open jaw against the tray walls, the flat-part descend, the LIFT gate on narrow parts), the teacher-logic fixes, and the K = 50 read of the fixed teacher (0 of 50 three times before, 15, 11 and 17 of 50 after). The rest of the gap is the tray, the drop and the pick primitive, all owner decisions.
- [`docs/ms6-lever-verification.md`](docs/ms6-lever-verification.md) — the verification that decides those two owner questions. It runs a solver reach map, a driven reach band, a 240-sample drop and tray sweep, and a probe of a bottom-open pick. The tray lever buys 3.6 of 50 fewer wall-blocked parts. The drop lever moves the ejections only through the tray and the wall. The pick primitive, with a matched grasp ranking model, stows **45 of 150** against the fixed teacher's **43 of 150** at the corpus flags. Without the hover jitter band it stows **67 of 150** against **32 of 150**. The report recommends the primitive, names the band as the reason the gain is configuration-dependent, and prices its dataset cost.
- [`docs/ms6-target-context.md`](docs/ms6-target-context.md) — the second phase B lever: a 16-wide `camera_box_target` context that hands the policy its target's centre and the teacher's grasp, the heap loop's cycle boundary moved to the teacher's (bin entry, then a wait for the retreat), and a 50-action serving horizon. The 400k checkpoint reads 0.6250 on 3000 held-out parts against the previous 0.4297 and a goal of 0.80. It records the stale-row mechanism, the wait constants that measured lower, and the census of the two checks.
- [`docs/ms6-square-wrist.md`](docs/ms6-square-wrist.md) — campaign n, the third phase B lever: the next target at the release on both the teacher and the eval loop, a squeeze floor and a wrist squaring that serve from the target row, on a warm start trained on the re-recorded corpus. The checked read passes the goal of 0.80 at 2711/3000 = 0.9037, and the official seed confirms it at 2759/3000 = 0.9197. It records the depth forensic that found the jaw-angle mechanism and the levers that read lower.
- [`docs/ms6-bare-desk-heap.md`](docs/ms6-bare-desk-heap.md) — the ladder from the bare desk at 0.633 to a 0.933 held-out and 0.940 train-split stowed fraction: no pushes, a batched drop over the whole scatter zone, a catalog without round bricks or wedge slopes, a slope that closes on its side, and a backstop wall. It records the desk contact and reach defects found on the way, the converter defect that dropped heap episodes, and the corpus and training commands.
- [`docs/ms7-sorting.md`](docs/ms7-sorting.md): MS7a. Rung 1 stows a tray heap in one desk box, and rung 2 sorts it into four desk boxes by four part classes. The teachers pass at 0.920, 0.963 and 0.905 on 600 parts. The rung 2 checked read passes on a tray whose walls lean out 30 degrees: 2610/3000 = 0.870 at 1.138 cycles per correct stow. On vertical walls, the same checkpoint reads 2513/3000 = 0.838 on the same draws. The lean is a deviation from the design. Rung 1 fails at 2533/3000 = 0.844. The report records `scripts/fleet.py` on three hosts, the serving rules, the forensics of the lost parts and the host lessons. The official read passes on the leaned tray: 5205/6000 = 0.8675 at 1.134 cycles per correct stow.
