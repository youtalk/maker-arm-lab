# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Research layer for the MakerMods Maker Arm. It holds a LegoSort pick-and-place simulation on two backends (Isaac Lab and MuJoCo). A scripted oracle generates demonstrations, a LeRobot ACT wrapper trains on them, and gated milestone evaluations score the result. `README.md` holds the full command reference, the task id table and the dataset schema. `docs/ms*.md` are the milestone reports. The reports record what was measured and why. Read the latest one before you change the oracle, the grader or the physics.

## Commands

Three separate environments. Do not mix them.

```bash
# CPU tests and lint (uv, Python 3.12). This is what CI runs.
uv sync --group dev                   # needs cargo: maker-arm-rs is built from git
uv run pytest
uv run pytest tests/test_mj_env.py -k shard      # one file, one test pattern
uv run ruff check . && uv run ruff format --check .

# lerobot_stack tests run in their own venv and are NOT collected by `uv run pytest`
bash lerobot_stack/setup_lerobot_env.sh          # once
lerobot_stack/.venv/bin/python -m pytest lerobot_stack/

# MuJoCo backend scripts run under the uv venv, no GPU or Isaac needed
uv run python scripts/mj_generate_batch.py --out_dir /tmp/b0 --attempts 12 --num_envs 12 --probe
uv run python scripts/mj_eval_policy.py --num_envs 12 --run_seed 1 --layout_seed 2 \
    --permutation 0,1,2,3 --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json
uv run python scripts/mj_dagger_batch.py --num_envs 12 --run_seed 1 --layout_seed 2 \
    --permutation 0,1,2,3 --episode_draws --frame_jitter --context camera \
    --server tcp://127.0.0.1:5700 --round 1 --out_dir /tmp/dag --shard_name proc_0

# Isaac Lab scripts need the Isaac venv and isaaclab.sh (see README "Environment setup")
export ISAACLAB_PATH="$HOME/src/IsaacLab"; export OMNI_KIT_ACCEPT_EULA=YES
source "$ISAACLAB_PATH/.venv/bin/activate"
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless

# Campaign and gate supervisors are stdlib-only and run under system python3
python3 scripts/generate_dataset.py --backend mujoco --out_dir ... --dataset_root ... --resume
python3 scripts/run_gate_eval.py --backend mujoco --checkpoint ... --ledger ... --threshold 0.70 --seed ...
```

Long GPU or campaign jobs get swept on this host. Daemonize them with `setsid bash -c '...' < /dev/null &` and capture stdout to a log.

`cargo` lives at `~/.cargo/bin/cargo` and can be off PATH. `uv sync` fails without it.

## Architecture

### Two backends, one set of rules

The Isaac Lab task lives in `source/maker_arm_lab/tasks/manager_based/lego_sort/` and the MuJoCo backend in `source/maker_arm_lab/mj/`. Both build the same scene from the same vendored URDF and `core.layout`. Everything that decides what a run records or what a rollout measures lives once, in `core/`. That code reads the environment through a duck-typed surface with Isaac Lab's names, for example `env.scene["robot"].data.joint_pos`, `env.obs_buf["images"]` and `env.step`. The full member list is the docstring of `mj/env.py`. The Isaac backend is frozen at its MS3 feature set (roadmap decision D2, 2026-09-12). New env members are MuJoCo only, and the shared code in `core/` reads them with `getattr(env, name, default)` so the Isaac smokes keep passing. The Isaac batch script accepts and ignores `--episode_draws` and `--frame_jitter`. The gate supervisor sends `--spatial_scale` and `--context` only to the MuJoCo client. Photorealism is not the lever the sim-to-real literature ranks first, and a Gaussian-splat twin bolts onto MuJoCo later if a real gate needs it.

- `core/batch_loop.py` is the generation run. `scripts/generate_batch.py` (Isaac) and `scripts/mj_generate_batch.py` (MuJoCo) only build the env and honor the token contract.
- `core/eval_loop.py` is the gate rollout. `scripts/eval_policy.py` and `scripts/mj_eval_policy.py` are the two thin clients.
- `core/stowed.py` and `core/checker.py` are the grader. Driver bookkeeping is never the metric.
- `core/gate.py` and `core/trace.py` are the rollout tracker and per-step trace.
- `core/dagger_loop.py` is the DAgger rollout: the policy drives, and the oracle rig takes over on a trigger for one brick cycle. It records only takeover frames. `scripts/mj_dagger_batch.py` is its thin client.
- `core/parts.py`, `core/grasp.py` and `core/heap_oracle.py` are the MS6 heap teacher: the part catalog, grasp synthesis on part geometry, and an `EnvOracle` subclass with a failure memory and a push. `mj/env.py` builds a heap when it gets `parts`, and `OracleRig` takes an `oracle_factory` so the heap script installs the subclass without a rig subclass. `mj/overlay.py` stamps the target marker and `mj/depth.py` is the noisy depth model. Heap runs export every episode and mark cycles through `obs/actor`.

### The oracle is split into WHAT and HOW

`core/oracle.py` is the phase state machine and only says what to do next (a `Goal`). `tasks/manager_based/lego_sort/oracle_runtime.py` holds `OracleRig`, which turns a `Goal` into a joint-space plan. The plan comes from `core/global_ik.py` (a global multi-start solver) and `core/trajectory.py` (minimum-jerk walk). The solver's docstring records why it replaced differential IK. Both are torch-only and the MuJoCo scripts import `OracleRig` directly. `OracleRig.actions()` and `observe()` are the two halves of `step()`. An idle env belongs to another actor. `resume()` enters the machine from a policy's state: LIFT on a pinched brick, HOVER otherwise.

### The `core/` package is isaaclab-free

Every module under `core/` is torch and stdlib only, so the solver, the grader, the gate arithmetic and the oracle have CPU unit tests. `pytest` collects nothing that imports `isaaclab`. Tests that must inspect an Isaac-only script (for example the token contract of `scripted_grasp_check.py`) parse it as source with `ast` instead of importing it. Keep it that way.

### Gains and the clamp come from the Rust SDK, with no fallback

`core/profile.py` reads gains, torque caps and the command clamp from `maker_arm_rs.profile()` at import time. A missing `maker_arm_rs` fails at import. Never add a constant table, an optional import or a try/except that lets the sim run without it, not even to get a test green. The recorded action is by construction a command the real arm accepts.

### The two-venv boundary in lerobot_stack

`lerobot_stack/` has its own Python 3.12 venv with LeRobot 0.5.1 and cannot import `maker_arm_lab`. `policy_protocol.py` is the ZMQ wire format, numpy and pyzmq only. Both the LeRobot-side `policy_server.py` and the eval clients import it, so it must never gain a torch or lerobot import. Constants that both sides need (gripper travel, joint names, the 7-wide state) are pinned by `tests/test_lerobot_schema_pin.py`, which reads the lerobot_stack source as text.

### Token contract

Every script prints exactly one terminal token line on every exit path, including argparse errors, and exits through `os._exit`. Examples: `BATCH_OK`, `EVAL_OK`, `GATE_FAIL`, `SERVER_READY`. Supervisors parse these tokens. A tokenless exit reads as a hang and is a defect.

## Conventions that tests enforce

- Joint coordinates everywhere are the vendored URDF's, ordered `link_002_joint` to `link_007_joint`. Gripper: `0.0` closed, `0.0524125` open. Raw gripper action `>= 0` opens, `< 0` closes.
- Dataset schema: `observation.state` is (7,) = 6 joint angles + gripper opening fraction. `action` is (7,) = 6 joint targets + binary gripper.
- `assets/vendor/maker_arm_description/` is never edited. Re-vendor by bumping `PIN_SHA` in `scripts/vendor_maker_arm_description.sh`. `assets/usd/` is generated and git-ignored, and nothing tests its freshness.
- Constants marked `# INITIAL:` are placeholders that a named later task measures. `<NAME>_UNVERIFIED` flags in `source/` must also appear in the divergence list of `docs/ms1-maker-arm-sim.md` (`tests/test_report_unverified_constants.py`).
- Do not write an unqualified `section 5.x` spec citation in code or scripts. Only `MS2 spec section 5.x` and `MS2 report section 5.x` are valid (`tests/test_spec_citations.py`).
- A number carried over from openyam-lab unchanged must say so and name what re-validates it on this arm.
- Ruff: line length 100, target py312, `assets/vendor` excluded, F403 enabled.
- Every committed artifact is English only. Commit with `git commit -s`. Stage named paths, never `git add -A`.
- The six-brick task is the special case of every heap rule: `pinch_min_stall(0.0158)` is `PINCH_MIN_STALL`, `Goal.half_height` defaults to the 2x4's, and `per_brick_stowed` grades by color unless the env has `destination_bins`. Part sizes come from `core/parts.py`, never from a new constant table. The tray interior stays inside `SCATTER_MAX_RADIUS` (`tests/test_layout.py`).
