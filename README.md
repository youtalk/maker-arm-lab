# maker-arm-lab

Research layer for the [MakerMods Maker Arm](https://github.com/makermods-robotics/maker-arm-sdk): simulation, system identification, model learning, and gated milestone experiments. The aim is sim-to-real — an Isaac Lab environment built on the vendor URDF, generating data and training policies in the Maker Arm's own joint coordinates so the result can be deployed on the physical arm through the [maker-arm-rs](https://github.com/youtalk/maker-arm-rs) SDK. Gains, torque caps and the command clamp are read from `maker_arm_rs.profile()` at import time rather than restated here, so a recorded action is by construction a command the real arm accepts; there is no fallback table, and a missing `maker_arm_rs` fails at import. Two tracks run in this repository: **MA0–MA5**, the hardware bring-up and identification milestones (the arm arrives October 2026), and **MS1**, the LegoSort simulation environment, which is complete — see [`docs/ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md) for its results, including a **failed** acceptance gate, and [`docs/ma0-bench-prep.md`](docs/ma0-bench-prep.md) for the bench preparation.

## Layout

```
source/maker_arm_lab/
  robots/            frames.py (measured constants, isaaclab-free) + maker_arm.py (ArticulationCfg)
  actuators/         MIT-mode actuator; gains and caps from maker_arm_rs.profile()
  assets/            urdf_prep.py (Z-up derived URDF), GravityCompensatedArticulation
  core/              layout, oracle state machine, global_ik, checker, gate, profile, trace
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

| Task id | Arm action | Cameras | Used by |
| --- | --- | --- | --- |
| `MakerArm-LegoSort-v0` | clamped absolute joint position | — | smoke, the scripted grasp check, policy rollout |
| `MakerArm-LegoSort-IK-Rel-v0` | task-space delta pose (differential IK) | — | release envelope check |
| `MakerArm-LegoSort-IK-Rel-Vision-v0` | task-space delta pose | wrist, overhead | oracle check with cameras |
| `MakerArm-LegoSort-Datagen-v0` | clamped joint position + vision + domain randomisation + recorders | wrist, overhead | `generate_batch.py` |
| `MakerArm-LegoSort-Eval-v0` | clamped joint position + vision + eval terminations | wrist, overhead | the next milestone's rollout gate |

`Datagen-v0` moved from the IK-Rel action space to clamped joint position in MS1, when the oracle moved onto `core/global_ik.py`. The dataset contract is unchanged by that move — the recorded action stream was always the post-IK joint target. See `docs/ms1-maker-arm-sim.md` section 4.3.

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

Dataset schema, both sides (spec section 4.2; `docs/ms1-maker-arm-sim.md`):

| key | shape | content |
| --- | --- | --- |
| `observation.state` | (7,) | 6 URDF joint angles [rad] + gripper opening fraction (0 closed, 1 open) |
| `action` | (7,) | 6 URDF joint targets [rad] + binary gripper (`>= 0` open, `< 0` close) |
| `observation.images.wrist`, `observation.images.overhead` | 224x224x3 | uint8 RGB |

To convert a shard by hand:

```bash
lerobot_stack/.venv/bin/python lerobot_stack/convert_to_lerobot.py \
    --shards /tmp/batch0/shard.hdf5 --root ~/data/maker-arm-ms1/dataset   # CONVERT_OK episodes=<n> total=<n>
```

## Training smoke

```bash
STEPS=100 OUT=~/data/maker-arm-ms1/train/smoke JOB_NAME=act-smoke WANDB_ENABLE=false \
  bash lerobot_stack/train_act.sh                                    # TRAIN_OK out=<dir> steps=<n>
```

For an online run, set `WANDB_ENTITY` explicitly. The wrapper refuses to launch with an empty `WANDB_ENTITY` rather than silently falling through to a default entity.

## Reports

- [`docs/ms1-maker-arm-sim.md`](docs/ms1-maker-arm-sim.md) — MS1: the LegoSort simulation environment on the vendor URDF. Asset pipeline, the measured robot constants, the full validation ladder, the acceptance gate (**failed** on the oracle success rate, 30/50 = 0.600 against a 0.70 bar) with its diagnostic hand-off, and the sim/real divergences and unverified constants the October hardware bring-up needs.
- [`docs/ma0-bench-prep.md`](docs/ma0-bench-prep.md) — MA0: bench preparation. Predates MS1 and was **not** re-verified against it, so read it as the MA0 record rather than as current: where the two disagree about the arm, MS1's measured constants are the later reading.
