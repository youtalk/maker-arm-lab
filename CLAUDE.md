# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Research layer for the MakerMods Maker Arm. It holds pick-and-place simulations on two backends (Isaac Lab and MuJoCo). A scripted teacher (the oracle) generates demonstrations, a LeRobot ACT wrapper trains on them, and gated milestone reads score the result. MS1 to MS4 sort six bricks into bins by color. MS6 to MS7f sort a heap of mixed parts into desk boxes by class. A second track (MA0, MA2) prepares the real arm. `README.md` holds the milestone status table, the command reference, the task id table and the dataset schema. `docs/ms*.md` and `docs/ma*.md` are the milestone reports. The reports record what was measured and why. Read the latest one (`docs/ms7f-real-scale-sorting.md`) before you change the oracle, the grader, the loop or the physics.

## Commands

Three separate environments. Do not mix them.

```bash
# CPU tests and lint (uv, Python 3.12). This is what CI runs, with MUJOCO_GL=disable.
uv sync --group dev                   # needs cargo: maker-arm-rs is built from git
uv run pytest
uv run pytest tests/test_mj_env.py -k shard      # one file, one test pattern
uv run ruff check . && uv run ruff format --check .

# lerobot_stack tests run in their own venv and are NOT collected by `uv run pytest`
bash lerobot_stack/setup_lerobot_env.sh          # once
lerobot_stack/.venv/bin/python -m pytest lerobot_stack/

# MuJoCo backend scripts run under the uv venv, no Isaac needed
uv run python scripts/mj_generate_batch.py --out_dir /tmp/b0 --attempts 12 --num_envs 12 --probe
uv run python scripts/mj_eval_policy.py --num_envs 12 --run_seed 1 --layout_seed 2 \
    --permutation 0,1,2,3 --server tcp://127.0.0.1:5555 --out /tmp/eval_000.json
uv run python scripts/mj_eval_heap.py --num_envs 1 --heap_k 50 --parts_split heldout \
    --run_seed 20261300 --layout_seed 20261310 --oracle_driver --out /tmp/preflight.json

# Isaac Lab scripts need the Isaac venv and isaaclab.sh (see README "Environment setup")
export ISAACLAB_PATH="$HOME/src/IsaacLab"; export OMNI_KIT_ACCEPT_EULA=YES
source "$ISAACLAB_PATH/.venv/bin/activate"
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless

# Campaign, gate and fleet supervisors are stdlib-only and run under system python3
python3 scripts/generate_dataset.py --backend mujoco --out_dir ... --dataset_root ... --resume
python3 scripts/run_gate_eval.py --backend mujoco --checkpoint ... --ledger ... --threshold 0.70 --seed ...
python3 scripts/fleet.py status       # multi-host campaigns, host table <campaign>/hosts.json
```

`tests/conftest.py` sets `MUJOCO_GL=egl` unless the caller names a backend. A render test probes for a GL context and skips without one.

`cargo` lives at `~/.cargo/bin/cargo` and can be off PATH. `uv sync` fails without it.

### Long runs

- Run a campaign from its own git worktree (`git worktree add ~/src/maker-arm-lab-<tag> <branch>`), not from this checkout. A campaign re-reads the code for hours, and a branch switch by another session kills every lane.
- In that worktree, symlink `.venv` and `lerobot_stack/.venv` to the main checkout. Run scripts as `MUJOCO_GL=egl PYTHONPATH=$PWD/source .venv/bin/python ...`, the form the script docstrings use. Without `PYTHONPATH`, the editable install imports the main checkout's code.
- Do not `uv run` in a worktree with a symlinked venv. uv re-points the shared editable install at that worktree. To repair it, run `uv sync --group dev --inexact` in the main checkout.
- Long GPU or campaign jobs get swept on this host. If the launching session stays alive, daemonize them with `setsid bash -c '...' < /dev/null &` and capture stdout to a log. If a tool call launches them, start every child with `&` and `wait` inside one background call.
- Do not edit a launcher script while a copy of it runs. Bash reads the file by offset, so the run dies at its next line.

## Architecture

### Two backends, one set of rules

The Isaac Lab task lives in `source/maker_arm_lab/tasks/manager_based/lego_sort/` and the MuJoCo backend in `source/maker_arm_lab/mj/`. Both build the same scene from the same vendored URDF and `core.layout`. Everything that decides what a run records or what a rollout measures lives once, in `core/`. That code reads the environment through a duck-typed surface with Isaac Lab's names, for example `env.scene["robot"].data.joint_pos`, `env.obs_buf["images"]` and `env.step`. The full member list is the docstring of `mj/env.py`. The Isaac backend is frozen at its MS3 feature set (roadmap decision D2, 2026-09-12). New env members are MuJoCo only, and the shared code in `core/` reads them with `getattr(env, name, default)` so the Isaac smokes keep passing. The Isaac batch script accepts and ignores `--episode_draws` and `--frame_jitter`. The gate supervisor sends `--spatial_scale` and `--context` only to the MuJoCo client. Every heap milestone (MS6 onward) is MuJoCo only.

- `core/batch_loop.py` is the generation run. `scripts/generate_batch.py` (Isaac) and `scripts/mj_generate_batch.py` (MuJoCo) only build the env and honor the token contract. It also holds the shared MS7e and MS7f command line (`add_ms7f_arguments`, `MS7F_OFF`, `MS7F_READY`, `check_ms7f`).
- `core/eval_loop.py` is the six-brick gate rollout. `scripts/eval_policy.py` and `scripts/mj_eval_policy.py` are the two thin clients.
- `core/loop.py` is the heap outer loop and the largest module. A `HeapOracle` selects each target, the policy drives the cycle, and the privileged grader ends it. `rollout(env, rig, policy, ...)` returns the eval payload. With `policy=None` the rig drives every cycle, and `takeover=` turns on the heap DAgger. The module also holds the serving rules (gripper and wrist rules applied to the served action) and the per-cycle census. Its docstring lists every switch by milestone. `scripts/mj_eval_heap.py` and `mj_stretch_run.py` are its thin clients.
- `core/stowed.py` and `core/checker.py` are the grader. Driver bookkeeping is never the metric.
- `core/gate.py` and `core/trace.py` are the rollout tracker and per-step trace.
- `core/dagger_loop.py` is the six-brick DAgger rollout: the policy drives, and the oracle rig takes over on a trigger for one brick cycle. It records only takeover frames. `scripts/mj_dagger_batch.py` is its thin client.
- `core/parts.py`, `core/grasp.py` and `core/heap_oracle.py` are the heap teacher: the part catalog, grasp synthesis on part geometry, and an `EnvOracle` subclass with a failure memory per 5 cm cell. `core/desk_boxes.py` draws the MS7a desk boxes. With `parts`, `mj/env.py` builds a heap. `OracleRig` takes an `oracle_factory`, so the heap scripts install the subclass without a rig subclass. `core/overlay.py` stamps the target marker and `mj/depth.py` is the noisy depth model. Heap runs export every episode and mark cycles through `obs/actor`.
- The LDraw parts are committed as `assets/parts/<id>.json`. `scripts/ldraw_parts.py` generated them once in a throwaway venv with the `ldraw` dependency group. The runtime reads only the JSON, so CI never installs that group.
- `real/` (MS5) is the rig behind the same env surface: `real/env.py` (`RealEnv`, a blind scene for the truth metrics), `real/arm.py` (`Tracking` through `maker_arm_rs.Arm`, one row per step), `real/perceive.py` (the perception worker's client, the loop's `perception_backend`), `real/sources.py`, `real/camera_model.py` (the homography warps to the simulator's views), `real/pacing.py` (ticks, caps) and `real/rig.py` (the rig JSON, box labels, the episode sheet). `core/loop.py` reaches perception only through `backend_of(env)`.

### The oracle is split into WHAT and HOW

`core/oracle.py` is the phase state machine and only says what to do next (a `Goal`). `tasks/manager_based/lego_sort/oracle_runtime.py` holds `OracleRig`, which turns a `Goal` into a joint-space plan. The plan comes from `core/global_ik.py` (a global multi-start solver) and `core/trajectory.py` (minimum-jerk walk). The solver's docstring records why it replaced differential IK. Both are torch-only and the MuJoCo scripts import `OracleRig` directly. `OracleRig.actions()` and `observe()` are the two halves of `step()`. An idle env belongs to another actor. `resume()` enters the machine from a policy's state: LIFT on a pinched brick, HOVER otherwise.

### The perceived boundary

A real arm has no contact sensors. Under `--perceived` (MS7c onward), the heap loop makes its decisions from two sources only. `core/perception.py` is the perceived state: the truth passed through noise and failure models (`--noise_scales`). `core/hold.py` reads the hold and the lift from the jaw gap, the jaw command and the FK pick point. The grader, the tracker and the census still read the truth, so the numbers compare across milestones. The loop reaches the truth only through `perception_truth`, `wrist_class` and `truth_reads`, and `tests/test_perceived_loop.py` enforces this. A new decision rule must not read contacts or part poses directly.

The MS7f part classifier sits on the same boundary. `scripts/mj_render_set.py` renders its views on teacher cycles. `lerobot_stack/train_part_classifier.py` trains on the renders and exports TorchScript. The heap client loads that file in-process on the CPU through `core.perception.PartClassifier` (`--categories --classifier PATH`). `policy_server.py` serves only the ACT policy.

### The gate supervisor

`scripts/run_gate_eval.py` starts one policy server under `lerobot_stack/.venv` and N eval clients, then sums their payloads. A heap read (`--heap_k` > 0) runs `scripts/mj_eval_heap.py`, and a six-brick read runs `mj_eval_policy.py` or the Isaac `eval_policy.py`. Seeds must be disjoint from the ledger. If the policy, the inference configuration and every switch match, the supervisor reuses a cached `eval_NNN.json`. If you add a switch, keep these rules:

- A new heap loop switch reaches the client through `--client_flags`. Add it to `LOOP_SWITCHES`, or the cache mixes reads with and without it.
- A new MS7f switch needs its entry in `MS7F_READY`. A switch outside it fails with its token and never runs as a no-op (`tests/test_ms7f_stack.py`).
- The supervisor imports no `maker_arm_lab`. It keeps its own copies of `PERCEPTION_KNOBS`, `CONTROL_HZ`, `INTERVAL_NAMES`, `MS7F_OFF`, `MS7F_READY` and `MAX_HEAP_K`. Tests pin each copy to its source.

Six-brick reads serve under the MS3 protocol, `--n_action_steps 1 --temporal_ensemble_coeff 0.01`. Heap reads serve open loop with no temporal ensembling (ruling of 2026-09-20, `docs/ms6-heap-policy.md` section 6), at 50 actions per forward since `docs/ms6-target-context.md`.

### The tracking bench (MA2)

`mj/bench.py` is an arm-only MuJoCo plant for the tracking controller of maker-arm-rs (`maker_arm_rs.Tracking`). `mj/bench_gate.py` holds the pre-registered scenarios and bars, and `scripts/mj_track_bench.py` runs them. `scripts/track_timing.py` and `scripts/track_soak.py` run on the Jetson Thor under plain `python3`. They, `core/dynamics_model.py` and `assets/urdf_prep.py` import only stdlib and `maker_arm_rs` (`tests/test_track_scripts.py`, `tests/test_dynamics_model.py`).

### The `core/` package is isaaclab-free

Every module under `core/` is torch and stdlib only. Thus the solver, the grader, the gate arithmetic, the loop and the oracle have CPU unit tests. `pytest` collects nothing that imports `isaaclab`. Tests that must inspect an Isaac-only script (for example the token contract of `scripted_grasp_check.py`) parse it as source with `ast` instead of importing it. Keep it that way. `real/ros.py` is the only module that imports rclpy; tests never import it, and scripts load it inside functions (`tests/test_real_ros_boundary.py`). `perception_stack/perception_protocol.py` is the worker's wire, numpy and the stdlib only.

### Gains and the clamp come from the Rust SDK, with no fallback

`core/profile.py` reads gains, torque caps and the command clamp from `maker_arm_rs.profile()` at import time. A missing `maker_arm_rs` fails at import. Never add a constant table, an optional import or a try/except that lets the sim run without it, not even to get a test green. The recorded action is by construction a command the real arm accepts. `pyproject.toml` pins maker-arm-rs by git revision.

### The two-venv boundary in lerobot_stack

`lerobot_stack/` has its own Python 3.12 venv with LeRobot 0.5.1 and cannot import `maker_arm_lab`. `policy_protocol.py` is the ZMQ wire format, numpy and pyzmq only. Both the LeRobot-side `policy_server.py` and the eval clients import it, so it must never gain a torch or lerobot import. Constants that both sides need (gripper travel, joint names, the 7-wide state) are pinned by `tests/test_lerobot_schema_pin.py`, which reads the lerobot_stack source as text.

### Token contract

Every script prints exactly one terminal token line on every exit path, including argparse errors, and exits through `os._exit`. Examples: `BATCH_OK`, `EVAL_OK`, `GATE_FAIL`, `SERVER_READY`. Supervisors parse these tokens. A tokenless exit reads as a hang and is a defect.

## Conventions that tests enforce

- Joint coordinates everywhere are the vendored URDF's, ordered `link_002_joint` to `link_007_joint`. Gripper: `0.0` closed, `0.0524125` open. Raw gripper action `>= 0` opens, `< 0` closes. Under `--gripper continuous` the seventh action column is the jaw gap in meters instead.
- Dataset schema: `observation.state` is (7,) = 6 joint angles + gripper opening fraction. `action` is (7,) = 6 joint targets + binary gripper.
- `assets/vendor/maker_arm_description/` is never edited. Re-vendor by bumping `PIN_SHA` in `scripts/vendor_maker_arm_description.sh`. `assets/usd/` is generated and git-ignored, and nothing tests its freshness.
- Constants marked `# INITIAL:` are placeholders that a named later task measures. `<NAME>_UNVERIFIED` flags in `source/` must also appear in the divergence list of `docs/ms1-maker-arm-sim.md` (`tests/test_report_unverified_constants.py`).
- Do not write an unqualified `section 5.x` spec citation in code or scripts. Only `MS2 spec section 5.x` and `MS2 report section 5.x` are valid (`tests/test_spec_citations.py`).
- A number carried over from openyam-lab unchanged must say so and name what measures it again on this arm.
- Ruff: line length 100, target py312, `assets/vendor` excluded, F403 enabled.
- Every committed artifact is English only. Commit with `git commit -s`. Stage named paths, never `git add -A`.
- The six-brick task is the special case of every heap rule: `pinch_min_stall(0.0158)` is `PINCH_MIN_STALL`, `Goal.half_height` defaults to the 2x4's, and `per_brick_stowed` grades by color unless the env has `destination_bins`. Part sizes come from `core/parts.py`, never from a new constant table. The tray interior stays inside `SCATTER_MAX_RADIUS` (`tests/test_layout.py`).
