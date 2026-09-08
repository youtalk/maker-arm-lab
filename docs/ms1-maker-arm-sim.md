# MS1 — Maker Arm simulation environment (LegoSort on the vendor URDF)

Milestone: MS1. Spec: `2026-09-05-ms1-maker-arm-sim-design.md`. Executed 2026-09-05 to 2026-09-07 on an RTX 5090 host with Isaac Lab v2.3.2 / Isaac Sim 5.1. Every number below is read off a log named in the task that produced it.

**If you are editing this file, two couplings to know about.** Section 4.3's list of UNVERIFIED constants is asserted by `tests/test_report_unverified_constants.py` in the CPU suite: every module-level `<NAME>_UNVERIFIED` flag under `source/` must have its constant named in that section, so a flagged constant added without documenting it fails `uv run pytest`. That bites **locally only** — see section 7 on the state of this repository's CI. And the pull request that introduced this report deliberately repeats section 1's and section 6's headline numbers in its body, so that the failed gate is visible without opening a file — editing those numbers here does **not** update that body, and the two need changing together.

## 1. Summary

**The pre-registered acceptance gate of spec section 5 FAILED.** The acceptance run — the campaign supervisor `generate_dataset.py --attempts_per_run 50 --num_envs 12` driving `generate_batch.py` through its own `--attempts 50`, run once, untuned — delivered **30/50 = 0.600** successful episodes against a **≥ 0.70** criterion. The gate's other two criteria passed on the same run (`clamp_fraction` 8.23e-05 against a ≤ 0.01 bar; zero NaN abandonments), and the milestone's stated goal was reached — the LegoSort environment stands up on the vendor URDF, the scripted oracle produces success-only episodes, `convert_to_lerobot.py` builds a 63-episode / 267,456-frame dataset with the (7,)/(7,) schema, and `lerobot-train` starts and completes 100 steps on it — but neither of those facts changes the gate's verdict, and neither is offered here to soften it. Nothing was tuned, re-rolled or re-seeded at any point, before or after the run. The gate was evaluated on run 0 and it failed; a second campaign run of the same command is reported below as an independent estimate (33/50 = 0.660) and explicitly not as a second attempt.

## 2. Method carried over from openyam-lab

The M0–M7 OpenYAM campaigns are not merged into this repository; what carried over is the method. Six practices, each with the instance in this milestone that earned its keep.

- **Pre-registered gates and read rules.** The thresholds in spec section 5 were fixed before any GPU ran, and the acceptance run was read against them as launched. When run 0 came in at 0.600 the campaign's second run was read as an independent estimate rather than as a re-gate, because re-gating on whichever run happens to clear the bar is exactly what pre-registration exists to prevent.
- **Null replicates and measured standard errors before reading any arm.** The gate miss was checked against its own null before being explained: 30/50 sits 3.20 standard deviations below the probe rate that sized the gate, `P(X ≤ 30 | p = 0.7857, n = 50) = 0.0023`, which is what turned "an unlucky draw" into "the probe was wrong". The counterweight was computed with the same discipline — the Wilson 95% interval for 30/50 is [0.462, 0.724] and contains 0.70.
- **Select-then-verify.** Constants were selected on one instrument and confirmed on another before being pinned: the home pose was chosen by a numpy FK of the vendored URDF and then reproduced by live Isaac to every printed digit; the grasp tilt was swept analytically and confirmed on GPU; the pinch floor was derived from brick geometry *before* anyone looked at which trials it admitted, and it cost a point when it was applied.
- **Build the instrument before pulling the lever.** Three of this milestone's defects were found in the measuring apparatus rather than in the arm: a torque decomposition that did not close because `computed_torque` and `joint_pos` came from opposite phases of a period-2 oscillation; a `clamp_fraction` that read 0.0000 because the rig pre-clamped its own command, making the criterion arithmetically incapable of firing; and a mid-run throughput estimate derived from HDF5 byte growth that read zero bytes over 300 s while the child burned 1.2 cores. Each was diagnosed before any lever was pulled on the strength of it.
- **Token-line protocol for GPU scripts.** Every GPU script prints a single `<NAME>_OK` / `<NAME>_FAIL` line and exits through `os._exit` with a real code, so a supervisor never has to parse prose. This milestone had to make that contract *total* rather than nominal: `scripted_grasp_check.py` had four tokenless exit paths (a module-scope constant, a default argument evaluated at import, `AppLauncher` argument validation, and argparse itself), and `train_act.sh` and `policy_server.py` had none at all. A tokenless exit is indistinguishable from a hang to the supervisor that reads it.
- **Venv isolation and the editable-install hazard.** Three environments are kept deliberately separate — the Isaac Sim venv (Python 3.11.15, built by `scripts/setup_isaac_env.sh`), the LeRobot venv (Python 3.12, `lerobot_stack/setup_lerobot_env.sh`), and uv's CPU test environment — and `lerobot_stack/` is not collected by `uv run pytest`. The editable install of this project into the Isaac venv is the hazard: it forced `requires-python = ">=3.11"` with a `.python-version` of 3.12, and the bootstrap's own `cd` made a relative `$0` resolve against Isaac Lab's `scripts/` directory instead of this repo's, so the setup script installed Isaac Lab over itself until the path resolution was captured before the `cd`.

## 3. Asset

- **Vendored source:** `makermods-robotics/maker-arm-sdk`, directory `urdf/maker_arm`, pinned commit `b30d05a23d72e8c155a8e00f807aba8e8c705f68` ("Add Maker Arm URDF with symmetric sliding gripper", 2026-09-05), Apache-2.0. Copied by `scripts/vendor_maker_arm_description.sh` into `assets/vendor/maker_arm_description/`; `VENDOR.md` carries the commit, a per-file SHA-256 block (self-excluded), the size, the license, and every enumerated upstream limitation. **The vendored URDF is never edited.**
- **Model contents, as pinned by `tests/test_urdf_validation.py`:** robot `maker_arm_gripper_v1`, 10 links, 9 joints, six revolute arm joints `link_002_joint`..`link_007_joint` (= motor ids 1..6), two prismatic gripper joints `gripper_left_joint` / `gripper_right_joint` (0..0.0524125 m, `<mimic>` multiplier 1), and the fixed `grasp_center` frame on `link_007` at −0.169638704618 m along the tool axis. **52 unique mesh files**, not the spec's 63: verified independently in the working tree as 102 raw `<mesh>` tags, 52 unique filenames, 52 STLs on disk and zero referenced meshes missing. The spec's 63 was a plan-time figure that was never measured; the pin, the vendored bytes and the test literal now agree at 52.
- **Derived Z-up URDF (plan refinement 1).** The vendored URDF keeps the CAD's Y-up base. Rather than rotating the articulation's initial state, `source/maker_arm_lab/assets/urdf_prep.py::write_zup_urdf` writes a derived copy into the git-ignored `assets/usd/` build directory with one extra fixed link, `mount_link` → `base_link`, rotated by `MOUNT_RPY`. `scripts/smoke_usd.py` then reads `BASE_Y_IN_WORLD` as `[~0, ~0, 0.99999994]`, i.e. the URDF base's +y within a small fraction of a degree of world +z, with no `ArticulationCfg.init_state.rot` flip needed.
- **`MOUNT_RPY = (π/2, 0.0, π)` — the yaw term is measured, not decorative.** At all-zero joints the arm faced **−x** (`FORWARD_XY` −0.2281, −0.0000) while the workspace (scatter band, bins, shelf) sits at +x, so the π yaw was added and the asset reconverted. `GRASP_CENTER_W` moved from `[−0.2269, ~0, 0.1092]` to `[+0.2269, ~0, 0.1092]` and Z-up survived the yaw (`BASE_Y_IN_WORLD` unchanged).
- **Mimic → two independent joints.** Both jaws are ordinary driven prismatic joints, commanded to the same target by `BinaryJointPositionAction`. The Isaac Lab flag that produces this is `convert_mimic_joints_to_normal_joints=False` — **the field is the inverse of its own name.** Isaac Sim's own URDF importer tests settle it: `parse_mimic = True` asserts the joint *has* `PhysxMimicJointAPI`, `parse_mimic = False` asserts it does not, and the Isaac Lab cfg field is a straight pass-through to `set_parse_mimic`. Setting it `True` (the spec's literal instruction) produced a right jaw that overshot to 0.0624 m on open against its own 0..0.0524125 limits.
- **Body and joint list in the USD:** 11 bodies (10 URDF links + `mount_link`) and 8 joints (6 arm + the 2 jaws). PhysX exposes 10 Jacobian rows, the missing root being `mount_link`; `grasp_center` is row 7 and is the only body whose centre of mass coincides with its link origin — exactly because it is massless — which is what makes the differential-IK Jacobian and the measured frame pose agree on this asset.

## 4. Robot configuration

### 4.1 Profile gains (from `maker_arm_rs.profile()`, pin `8f511c81f983323a1d33a3a213c18d5341566750`)

No gain literals exist in this repository; `core/profile.py` reads them from the Rust SDK at import and a missing `maker_arm_rs` fails at import with no fallback table.

| URDF joint | motor id | model | kp | kd | `tau_max` (feed-forward cap, N·m) | `effort_limit` (motor rating, N·m) |
| --- | --- | --- | --- | --- | --- | --- |
| `link_002_joint` | 1 | RS00 | 60.0 | 4.0 | 4.0 | 14.0 |
| `link_003_joint` | 2 | RS02 | 150.0 | 4.5 | **12.0** | 17.0 |
| `link_004_joint` | 3 | RS02 | 90.0 | 3.0 | **10.0** | 17.0 |
| `link_005_joint` | 4 | RS00 | 30.0 | 2.0 | 4.0 | 14.0 |
| `link_006_joint` | 5 | RS00 | 30.0 | 2.0 | 4.0 | 14.0 |
| `link_007_joint` | 6 | RS00 | 30.0 | 2.0 | 4.0 | 14.0 |

The two bolded caps were **6.0 each** when this milestone started. The sim caught the defect before the hardware did: at the home pose j2's gravity torque is 6.34 N·m against a 6.0 N·m feed-forward cap, i.e. the cap is below the arm's own gravity load essentially everywhere in its workspace, including at rest. The owner raised them upstream (`maker-arm-rs`, merged) sized from the *visited* peak of 9.48 N·m at j2 and 8.198 N·m at j3, explicitly not from a 13.257 N·m swept-envelope figure whose implied 0.55 m COM lever out of a 0.607 m maximum reach is not physical for a distributed arm. This is a change to the shared profile, not a sim/real divergence — the sim and the real arm still read the same numbers.

### 4.2 Home pose and grasp geometry

`HOME_JOINT_POS` (URDF coordinates, rad): `link_002_joint` 0.0, `link_003_joint` −1.79, `link_004_joint` 2.05, `link_005_joint` −1.3, `link_006_joint` −0.22, `link_007_joint` 0.0, both gripper joints at `GRIPPER_OPEN_POS = 0.0524125`. Forward kinematics, agreeing between a numpy FK of the vendored URDF and the live Isaac reading: `grasp_center` at **(0.2749, 0.0371, 0.1646)** with the tool axis **23.85°** off world −z and a minimum joint margin of **0.30 rad**, on `link_005_joint` — the tightest of the six by a factor of three, and the joint that limits every attitude demand downstream.

The spec's "about 0.25 m above the desk" is **geometrically unreachable** under the spec's own tool-cone (≤ 30°) and joint-margin (≥ 0.30 rad) constraints: the measured ceiling is z = 0.188 m. `find_home_pose.py`'s `--z_range` default was lowered to (0.15, 0.25) rather than relaxing either functional constraint, because the desk top is at world z = 0 exactly and "0.25 m" was a comfort target, while the 30° cone and the 0.30 rad margin are what keep the oracle's motions and the watchdog sane.

| constant | value | provenance |
| --- | --- | --- |
| `PICK_OFFSET_MEASURED` | `(0.0, 0.0, −0.0321)` m | Task D3, swept over −0.038..−0.010 m on the global-IK path with a near-level descend. The x component is a measurement, not an omission: centring the brick on the jaw plate's collision box looked right and was tried, and on GPU the jaws closed on nothing and pushed the brick to −81.9 mm, because that box is the whole finger body including bracket and rail while the gripping surface sits near the tool axis. |
| `DESCEND_CLEARANCE_MEASURED` | `0.0` m | Zero by construction on the global-IK path: the descend commands a position, so the depth is entirely expressed by `PICK_OFFSET_MEASURED` and a second additive term would be degenerate with it. |
| `WRIST_ROLL_SIGN` | `+1.0` | Measured `d(jaw_yaw)/d(wrist roll) = 0.9478`. **Flips** relative to the OpenYAM relation (−1). |
| `BASE_YAW_SIGN` | `−1.0` | Measured `d(jaw_yaw)/d(base yaw) = −0.9999962`, and independently derivable: `link_002_joint`'s URDF axis is (0, 0, −1) in a frame that `MOUNT_RPY` carries onto world −z. **Flips** relative to the OpenYAM relation (+1). Both signs are now pinned by CPU tests that were verified to fail when a sign is flipped back. |
| `GRASP_TILT_DEG` | `20.0` | The **approach** tilt. This arm cannot hold a level tool over the scatter band at hover height; the worst band corner needs 18.42°. At the brick the arm is level to within 0.42° at every radius, so the tilt is an approach cost and never a grasp cost. |
| `core.global_ik.MIN_DESCEND_TILT` | `0.5°` | The descend runs near level, never exactly level: at exactly vertical the lean azimuth is undefined and the solver's azimuth constraint row goes singular, which failed the continuity-constrained descend at waypoint 39 of 40 on five of six bricks. |
| `core.layout.SCATTER_MAX_RADIUS` | `0.280` m | The measured **trackable** radius, enforced as an invariant on the scatter sampler (a sample that violates it is pulled back along its radius) rather than as a sampling preference. The bound is a radius and not a box: trimming `SCATTER_X` or `SCATTER_Y` alone drops none of the uncovered region. It landed in two steps — 0.315 first, to exclude the one band position in 35 that is IK-infeasible at every jaw heading, costing 2.87% of band area against 15.9% / 25.0% for the equivalent box trims; then 0.280, the largest radius that tracked clean at every azimuth measured, costing 19.18%. Trials are not comparable seed-for-seed across the clip, because it changes the scatter RNG stream; bucket counts are. |
| `core.oracle.RELEASE_TILT_RAD` | `1.40` (was 0.70) | Task D4: at 0.70 the release pose sits 117–160 mm above this arm's reach ceiling. `core.layout` needed **no numeric change** at any point — the failure was in z, not in distance. |
| `core.oracle.RAISE_TILT_RAD` | `1.05` (new) | Task D5 decoupled RAISE's attitude from RELEASE's. At the old shared 80.2° value EXTEND could not plan to *any* of the four bins; at 60.2° it plans to all four. |
| `core.action_scale.IK_SCALE` | `(0.002, 0.002, 0.002, 0.01, 0.01, 0.01)` | Carried over from openyam-lab, unchanged. Halving the translation entries was tried under the brief's authority and refuted — it made the tracking error worse (216 → 294 mm), because it reduced the servo's authority to recover rather than reducing a torque demand. |

### 4.3 Sim/real divergences and unverified constants

This is the list the October 2026 hardware bring-up needs, and it is the one prose section in this report with a machine-checked coupling behind it. `tests/test_report_unverified_constants.py` walks `source/` with `ast`, collects every module-level `<NAME>_UNVERIFIED` flag, and asserts each one's constant is named in this section — so a future edit that adds a flagged constant without documenting it here fails the CPU suite. **Read that as a local guarantee, not a CI one:** this repository's CI has not run a test since the `maker-arm-rs` dependency was introduced (section 7), so the assertion protects whoever runs `uv run pytest` and nobody else. The reverse direction (a flag removed from source while this section still names it) is deliberately not pinned: it would need a third copy of the roster to maintain, and a section that still names a constant somebody has since measured is stale in the safe direction, unlike one that omits a constant nobody has measured at all.

- **`ARM_ARMATURE = 0.02` on all six arm joints — UNVERIFIED, and not a measured rotor inertia.** It is a bracketed stability value. The vendored URDF omits armature; at 0.0 every wrist joint sat in a period-2 limit cycle at the Nyquist frequency of the 120 Hz physics rate — `link_005_joint` reversing sign every substep at ±4.2 rad/s with the applied torque flipping ±9 N·m while its mean position error was 0.25 mrad. The governing bound is `kd·dt/M_eff ≥ 2`; for kd = 2.0 at dt = 1/120 the analytic marginal value is 0.00833, and the measured bracket straddles it (0.005 still chatters, 0.01 clean, 0.02 shipped at ~2.4× margin). The real value is rotor inertia × gear ratio², which neither the URDF nor the SDK publishes. **The failure is two-sided:** too small buzzes, too large silently adds fictitious inertia the real arm does not have. A measured value must still clear the bound; if it comes in below ~0.0083 for the wrist joints, raise the physics rate or lower kd rather than keeping 0.02.
- **Wrist-camera pose — UNVERIFIED, but bounded by a stated hardware fact.** `WRIST_CAM_POS_IN_WRIST = (0.030, 0.0, −0.118)` in `link_007`, pitched 30° toward −x, FOV 70°. The bound is the owner's statement that on the physical arm the camera sits **in front of** the wrist motor, so the motor cannot appear in the image; the original placeholder at (−0.05, 0, −0.06) sat 52 mm *inside* the motor housing's z span and filled half to two thirds of every rendered frame with the arm's own body. The replacement satisfies three measured constraints (z past the motor's −0.1128 front face; clear of the jaw sweep volume `|y| ≤ 0.0826` once the jaws' own ±0.0524 of travel is added to the static mesh bounds; `grasp_center` centred in frame) and renders 0.0% of 1428 sampled motor collision vertices into frame against 18.4% for the old pose. The real lens, intrinsics and bracket remain an arrival re-measurement item, and the 51.6 mm standoff makes the view unavoidably oblique — that is forced by the geometry, not chosen. **The consequence downstream, for the next milestone rather than for the bring-up:** a policy trained on this wrist stream is looking through a lens whose real counterpart has not been measured, so re-measuring the mount is a prerequisite for transferring anything learned from it.
- **`GRIPPER_EFFORT_LIMIT_N = 20.0`, the vendored URDF's placeholder, after 40 N was measured INERT.** Task D3 step 4.3 authorised the raise for the condition "the pinch will not hold"; the A/B showed the affected trial bit-identical at 20 N and 40 N. The arithmetic explains it: the gripper's `ImplicitActuatorCfg` carries `stiffness = 1000.0` N/m, the jaws are commanded to 0.0 and stall on the brick at ~7.6 mm, so the demand is ~7.6 N and never reached even the 20 N cap. **`stiffness`, not `effort_limit_sim`, is what sets the grip on this arm** — and `stiffness = 1000` N/m is itself a sim-only PD number with no real-arm counterpart. The raise was reverted so that a constant does not imply a strengthening that changes no outcome; the knowledge lives in the comment and here.
- **`maker-arm-rs` `tau_max` raise, merged upstream by the owner:** j2 6.0 → 12.0 N·m, j3 6.0 → 10.0 N·m, both under the RS02 17 N·m rating so `config.rs`'s `tau_max < t_max` invariant still holds. Recorded here because `config.rs`'s own governance rule is that these may be tuned "never silently".
- **Gain deviations: none.** Spec section 4.4 authorises raising kp within `kp_max = 200` and recording the deviation. It was applied exactly as scoped (j2 150 → 200, j3 90 → 200) and **REFUTED by measurement**: tool-attitude error 7.87° → 86.05°, j2 effort saturation 22% → 61%, settle drift 35.5 → 84.4 mm. Reverted. The mechanism generalises — raising kp raises the PD *demand* on joints already clipping at their effort limit, so the extra demand becomes more saturation rather than more torque. The shipped gains are the profile's, unmodified.
- **Owner-approved spec deviation: `scripted_grasp_check.py` runs on the joint-position action space, not on `MakerArm-LegoSort-IK-Rel-v0` as spec section 5 item 4 specifies.** Isaac Lab's damped-least-squares differential IK is correctly wired on this asset — verified against a kinematic finite difference at both the home pose and the jammed pose, agreeing to ≤ 2e-4 relative on the linear half of the Jacobian and ≤ 3e-4 on the angular half — and nonetheless spends `link_005_joint`'s 0.30 rad of margin instead of turning the base, for motions the base alone could deliver, and cannot back off once pinned. The oracle now plans joint targets with `core/global_ik.py` (Levenberg–Marquardt with multi-start; with the lean azimuth pinned the task is 6 constraints on 6 joints, so the solution set is discrete and this is branch selection rather than optimisation) and drives them through the joint-position action space. **The dataset contract is unaffected:** the recorded action stream was always the post-IK joint-position target, so what changed is how the oracle *produces* those targets, not what is stored or what a policy learns. The five registered task ids are unchanged and the IK-Rel variants remain available.

## 5. Validation ladder

| step | invocation | token | key numbers |
| --- | --- | --- | --- |
| CPU tests | `uv run pytest` | — | **320 passed, zero skips** at this branch's head (318 at the acceptance run's head `9be229d`, plus the two report-coupling tests added in section 4.3); `lerobot_stack/.venv/bin/python -m pytest lerobot_stack/` **64 passed**, suites kept split; `ruff check` and `ruff format --check` clean. **All local** — see section 7 on CI. |
| `smoke_usd.py` | `isaaclab.sh -p scripts/smoke_usd.py --headless` | `SMOKE_OK` | JOINTS 8/8, BODIES 11/11, `BASE_Y_IN_WORLD [~0, ~0, 0.99999994]`, MAX_DRIFT 0.031008; re-run after the `MOUNT_RPY` yaw with `GRASP_CENTER_W [+0.2269, ~0, 0.1092]` |
| `find_home_pose.py` | `isaaclab.sh -p scripts/find_home_pose.py --headless` | — | 12 candidates at a 10° step; best margin 0.300 rad, tool 23.6°, `grasp_center` (0.2745, 0.0375, 0.1628); pinned rounded, giving the 4.2 numbers |
| `smoke_lego_sort.py` ×6 | one per registered id, plus `--num_envs 4 --num_bricks 8`; the three vision ids add `--enable_cameras` | `SMOKE_OK` ×6, exit 0 ×6 | `TOOL_AXIS_DEG` ≤ 26.2° everywhere (30° bound), `CLAMPED_AT_RESET` False in every env of every run, `IMAGE_STD` well above 1.0 on both streams, `brick_poses` scaling to (4, 56) confirming `--num_bricks` |
| `scripted_grasp_check.py` | `isaaclab.sh -p scripts/scripted_grasp_check.py --headless` | **`GRASP_OK`** | **10/10 carried**, every failure bucket empty. Lift 99.92–100.07 mm at 7.60–7.62 N with 7.43–7.71 mm of jaw stall — **every carry clears the derived 4.8 mm pinch floor by more than 50%**. The floor is `min(BRICK_SIZE)/2`, derived from brick geometry *before* the trials were consulted, and it cost a point relative to the inherited 3.0 mm value. |
| `release_envelope_check.py` | `isaaclab.sh -p scripts/release_envelope_check.py --headless --layout_index N`, one process per layout | **`ENVELOPE_OK`** ×5 | **40/40 trials, 40/40 stowed, 40/40 EXTEND converged**, nothing abandoned or timed out; EXTEND residual 0.09–1.66 mm, attitude residual 0.04–0.38°, tightest joint `link_006_joint` at 0.755–1.291 rad from a stop. **Re-run in full under the shipped joint-position control model** (ruling D5-2), and the re-run was not a formality: before the RAISE-tilt fix the same check failed both outer bins at every layout-0 trial, a defect `oracle_check.py` cannot see because it scatters bricks at random yaw while this check places every brick at identity yaw. |
| `oracle_check.py` | `--headless --trials 2`; then `--headless --enable_cameras --with_cameras --trials 1` | **`ORACLE_OK`** ×2 | Headless 2/2 `terminated_by=success`, LIFT 12/12 pinch_ok. Camera 1/1 with `obs/images/{wrist,overhead}_rgb` at (4194, 224, 224, 3) uint8, `actions` (4194, 7) holding joint targets, `clamped_mean 0.0000` |
| `generate_batch.py --probe` | `--headless --enable_cameras --out_dir /tmp/batch0 --attempts 10 --num_envs 12 --probe` | **`BATCH_OK exported=7 attempts=12`** | successes 7, time_outs 3, abandons 2 (both `grasp_retries_exhausted`), watchdog 0, `clamp_fraction` 4.63e-05, 126.06 s/episode. Pooled over four probe runs: 33/42 = 0.786, 8 clamped env-steps in 225,660 = 3.5e-05 |
| `generate_dataset.py` acceptance | `--target 50 --attempts_per_run 50 --num_envs 12 --seed 20260905` | run 0 `BATCH_OK exported=30 attempts=50`; campaign `CAMPAIGN_OK total=63 runs=2` | See section 6 |
| `convert_to_lerobot.py` | driven by the supervisor per run | `CONVERT_OK` | Final dataset **63 episodes, 267,456 frames**, 30 fps, `robot_type maker_arm`, 479 MB, both image streams |
| train smoke | `STEPS=100 ... bash lerobot_stack/train_act.sh` | **`TRAIN_OK out=... steps=100`** | `step:100 smpl:800 ep:0 epch:0.00 loss:10.508 grdn:208.918 lr:1.0e-05 updt_s:0.025 data_s:0.004`; `dataset.num_episodes=63`, `dataset.num_frames=267456`; checkpoint written (`model.safetensors`, 197.1 MB, 51,599,239 learnable parameters) |

Layout changes made by Task D4: **none to `core.layout`**. The reach problem it was built to find was in z rather than in distance, and what moved was `core.oracle.RELEASE_TILT_RAD` (0.70 → 1.40 rad), a constant the brief explicitly authorises adjusting with the state machine untouched, as spec section 5 item 5 requires. The one layout constant this milestone did add, `SCATTER_MAX_RADIUS`, was pulled forward into Task D3 because the 10/10 grasp gate was structurally unreachable while a brick could spawn where the arm provably cannot go.

Two limitations of the ladder, stated so a later reader does not over-read a pass. `release_envelope_check.py` fixes the pick at (0.20, 0.0) by design, so it samples `SCATTER_MAX_RADIUS` at exactly one azimuth and is **not** an independent confirmation of that bound. And its single-process five-layout loop wedges after the second `gym.make` (100% CPU, no GPU work, no I/O), so the cross-layout `ENVELOPE_OK` is read off five per-layout tokens from five separate processes.

## 6. Acceptance gate

### 6.1 Pre-registered thresholds against measured values

| criterion (spec section 5) | required | measured, run 0 | verdict |
| --- | --- | --- | --- |
| oracle success rate, 12 envs, `--attempts 50` | **≥ 0.70** | **30/50 = 0.600** | **FAIL** |
| clamp fraction | ≤ 0.01 | 8.23e-05 (19 clamped steps of 19,246) | PASS |
| NaN abandonments | 0 | 0 (`{watchdog: 6, grasp_retries_exhausted: 7}`) | PASS |
| `convert_to_lerobot.py` builds a (7,)/(7,) dataset | yes | 63 episodes, 267,456 frames, `observation.state` [7], `action` [7] | PASS |
| `lerobot-train` with `STEPS=100` starts and completes | yes | reached step 100, checkpointed | PASS |

Run 0's full `RUN_STATS`: `attempts 50, successes 30, time_outs 7, abandons 13, abandon_reasons {watchdog: 6, grasp_retries_exhausted: 7}, watchdog_trips 6, control_steps 19246, clamped_steps 19, clamp_fraction 8.226817693719907e-05, wall_seconds 6355.79, deadline_hit false`. The campaign then closed `CAMPAIGN_OK total=63 runs=2`. Nothing was tuned, re-rolled or re-seeded: the base seed 20260905 reproduces the ledger's recorded per-run seeds exactly through `(seed*1000 + run_index) % 2**32`, and `MAX_GRASP_RETRIES`, `BATCH_WALL_MINUTES` and the three dependency pin sites are unchanged from before the run.

### 6.2 What the failure does and does not establish

| sample | rate | Wilson 95% | contains 0.70? |
| --- | --- | --- | --- |
| run 0 — **the gate** | 30/50 = **0.600** | [0.462, 0.724] | yes |
| run 1 — estimate only, **not a re-gate** | 33/50 = 0.660 | [0.522, 0.776] | yes |
| both acceptance runs | 63/100 = 0.630 | [0.532, 0.718] | yes (just) |
| + D6's probe | 96/142 = 0.676 | [0.595, 0.748] | yes |

**No sample excludes 0.70.** Every interval above contains it, and at n = 50 or even n = 100 the data cannot rule out a true rate of 0.70. What the run does is fail a pre-registered point criterion, which is a weaker and different claim than "the oracle is below 0.70". Both statements belong in the record.

**D6's four short probes over-estimated the oracle, and that is a finding about probe design.** The gate's ~7% pre-registered miss risk was computed against D6's pooled 33/42 = 0.7857; run 0 sits 3.20 standard deviations below that mean with `P(X ≤ 30 | p = 0.7857, n = 50) = 0.0023`. The two acceptance runs agree with each other (difference 0.060, se 0.097, **z = 0.62**) and jointly refute the probe rate: `P(X ≤ 63 | p = 0.7857, n = 100) = 0.00027`. Four short runs (42 episodes total, `--attempts 10` class) on a handful of layout draws is not enough to size a gate, and the same mistake is available to the next milestone.

### 6.3 Diagnostic hand-off

This is the part written for whoever picks the oracle up next.

| mode | run 0 | run 1 | bricks delivered of 6 | retries |
| --- | --- | --- | --- | --- |
| `time_out` | **7** | **7** | run 0: 4,5,5,5,5,5,5 (mean 4.86); run 1: 5,5,5,5,5,5,5 | all 0 |
| `grasp_retries_exhausted` | **7** | **7** | run 0: 1,2,3,3,4,5,5 (mean 3.29); run 1: 0,2,3,4,4,4,4 | **all exactly 2** |
| `watchdog` | 6 | 3 | run 0: 0,2,2,2,3,5 (mean 2.33); run 1: 0,3,5 | all 0 |

- **The two large modes are near-deterministic at these settings, and the watchdog carries the variance.** `time_out` and `grasp_retries_exhausted` each occurred exactly 7 times in *both* runs — 14/100 apiece. The entire run-to-run difference in successes (30 vs 33) is the watchdog delta (6 vs 3). That makes the two stable modes the tractable targets and the watchdog the noisy one, which inverts the ordering suggested by severity alone.
- **All 14 time-outs delivered 4–5 of 6 bricks with zero retries**, 13 of them exactly 5/6 and run 1's seven unanimously so. They ran out of step budget, not capability. **Corrected in section 9:** an instrumented re-run showed this reading was wrong. A time-out is a brick lost after a completed carry, followed by ~1500 idle control steps in DONE, and the step budget was never the binding constraint.
- **All 14 grasp-retry failures ended at `retries = 2`, never 0 or 1.** `MAX_GRASP_RETRIES = 2` is what terminates them by construction, so that mode is a parameter question rather than an observation about the arm. **Corrected in section 9:** the retries were three identical attempts at a pick the planner could not solve at the drawn hover height, so no retry count would have helped; the parameter was the hover height.
- **Failures cluster in a minority of episodes rather than being uniform per-brick flakiness.** Brick-level delivery is **513/600 = 0.855** across both runs against a 0.630 episode rate; under independence 0.855⁶ = 0.391 would be far below what was observed. All 30 of run 0's successes delivered 6/6 with zero retries and exactly 6 pick-and-place cycles. The useful question is what makes an *episode* go bad, not a brick.
- **Bounded ceilings, and they are ceilings rather than predictions.** Converting every run-0 `time_out` to a success gives 37/50 = **0.740**; converting every one-brick-short failure gives 39/50 = **0.780**. Neither was acted on. A time-out at 5/6 is no evidence that the sixth brick would have *completed* rather than stalled, and any step-budget increase trades directly against the 180-minute batch deadline that run 0 finished inside with 41% headroom.
- **The strongest failure correlate found so far** is `unplanned_legs` — 2.27 per success against 7.56 per failure in D6's batches — not `direct_legs`, which came in at 0.43 per episode and is not material. RAISE remains unplannable on roughly 40% of legs at any tilt; an unplannable leg now ends on the next control step (1 step instead of 60 idle steps) and EXTEND covers it from the LIFT pose, which is EXTEND's best start.

### 6.4 Throughput, for planning the next campaign

`deadline_hit: false`; run 0 took `wall_seconds = 6355.8` = 105.9 min against the 180-minute deadline, at **127.1 s per episode**, inside D6's measured 122.9–138.3 s band. `episodes_per_env` summed to exactly 50, not the `attempts + num_envs − 1 = 61` worst case the deadline was sized against, so 180 minutes was about 22% more pessimistic than reality. Camera overhead is inside the noise at 12 envs (126.1 s with cameras against 126.8 s without) — the planner is CPU-bound and dominates. Run 1 took 93.9 min.

## 7. Open items (need the physical arm, October 2026 or later)

Spec section 7's table, reproduced verbatim:

| Item | Where it lands |
| --- | --- |
| `direction/offset` calibration per joint; re-expressing `q_lo/q_hi` in URDF coordinates | maker-arm-rs (MA1) |
| j3 range width: SDK 4.07 rad vs URDF 3.14 rad | maker-arm-rs + maker-arm-lab docs |
| Wrist camera pose, model, and field of view | `robots/maker_arm.py` wrist `OffsetCfg`, `vision.py` intrinsics |
| Gripper motor-angle → opening-fraction transmission (linear map is provisional) | §4.2, real-side interface |
| Which end of the profile gripper range (−2.092..−0.039 rad) is open | §4.3 gripper slot mapping |
| Inertia, friction, and actuator identification | MA2 |

Discovered during this milestone, and added to the same list:

- **`ARM_ARMATURE = 0.02` — replace with rotor inertia × gear ratio², subject to the `kd·dt/M_eff ≥ 2` bound.** Full derivation and the two-sided failure mode in section 4.3.
- **Gripper `stiffness = 1000 N/m` is what sets the grip in sim, and has no measured real-arm counterpart.** `GRIPPER_EFFORT_LIMIT_N` is inert at any value the oracle reaches.
- **`PINCH_MAX_ASYMMETRY` stays inherited and unre-measured**, deliberately: this milestone never produced a lopsided-but-real grip, so the bound has never been exercised from the side it exists to reject.
- **The tracking wedge at |az| ≤ 10°.** The trackable region is smaller than the IK-feasible one, and it is **not a disc**: the arm tracks to 0.3082 m at +30° azimuth and to ≥ 0.295 m at |az| ≥ 25°, but gives out between 0.280 and 0.290 m near azimuth 0. `SCATTER_MAX_RADIUS = 0.280` is the largest radius that tracked clean at every azimuth **measured**, which is not the same as the largest that would track clean at every azimuth. A radius clip pays for the whole annulus to remove a wedge; the alternative — freeing `lean_azimuth`, which is pinned to 0.0 and therefore makes the descend lean in a fixed *world* direction — is a plausible shape of an explanation and was not implemented. Note for whoever re-derives the bound: `release_envelope_check.py` **must sweep tracking, not IK feasibility**, or it will conclude the arm reaches further than it tracks and re-derive a bound that was never binding.
- **The third unnamed defect, outside the clipped scatter band.** With the arm quiet, unsaturated, the integral trim off, the IK residual at 0.00 µm and the path continuous, the arm still stops 12–50 mm short and sits 0.02–0.12 rad off, with one joint holding ~12 N·m against a 17 N·m limit. Something takes that load and it was not found; it is deliberately written up **without** a cause. Excluded by measurement: gravity compensation, the feed-forward cap, the effort limit, the Rust clamp, the action term's scale/offset and joint ordering, static torque over the band, and the trim. Three ranked next measurements: a contact query at the stalled pose; the existing MIT-law decomposition script pointed at a stalled pose; stall height against step budget, to separate "blocked" from "moving very slowly".
- **`trace.py` still writes an eight-wide state fallback** (the articulation joint count), not the new (7,). Left alone because no eval script exists yet, but whoever writes the eval path must choose deliberately rather than inherit 8.
- **CI has not run a test in this repository since the `maker-arm-rs` dependency was introduced, so every test result in this milestone's record is a LOCAL result.** The `test` job dies after ~17 seconds inside `uv sync --group dev` with `failed to fetch commit 8f511c81… could not read Username for 'https://github.com'`: `maker-arm-rs` is a **private** repository and the GitHub Actions runner has no credentials for it, so `pytest` never starts. This is not specific to one branch — `docs/ms1-report`, `feat/ms1-lerobot` and `feat/ms1-oracle-datagen` all fail identically. Consequences worth stating plainly, because they are easy to misread: **no PR in this milestone has been validated by CI**; a red check on any of these PRs says nothing about the code; and the section 4.3 coupling test above, like every other test here, currently protects only whoever runs the suite locally. The fix is repository infrastructure rather than code — a deploy key, a PAT secret in the workflow, or making `maker-arm-rs` public — and it is the owner's to make. It belongs beside the citation sweep below as something the next milestone inherits, and it should be fixed **before** anything relies on a green check.
- **~19–20 dangling `§5.x` spec citations across 13 Part B/C/D files** (`mit_law.py`, `mit_mode.py`, `checker.py`, `lego_sort_env_cfg.py`, `vision.py`, `oracle.py`, `oracle_runtime.py`, `datagen.py`, `recorders.py`, `gate.py`, `eval_cfg.py`, `smoke_lego_sort.py`, `scripted_grasp_check.py`). These carry OpenYAM's spec numbering; this milestone's spec section 5 is an unsubdivided numbered list, so **`section 5 item N` is a valid form** and is used correctly in at least eight places, while `section 5.4` and `section 5.3` are dangling. The sweep is mechanical and documentation-only but must be its own branch off the merged `main` — the files span merged code and an open PR. **The inventory recorded in `task-E2-report.md` is known incomplete and must be re-derived**, not trusted: an independent grep found six sites it omits, an undercount of more than 2× on exactly the half of the ledger that proves `section 5 item N` is legitimate and therefore not blanket-fixable. **Swept in MS2 Phase 0:** the re-derived inventory was 22 sites in 13 files (`mit_law.py`, `mit_mode.py`, `checker.py`, `oracle.py`, `gate.py`, `lego_sort_env_cfg.py`, `eval_cfg.py`, `vision.py`, `oracle_runtime.py`, `datagen.py`, `recorders.py`, `smoke_lego_sort.py`, `scripted_grasp_check.py`); each was rewritten to the MS1 spec's real section (2.2, 2.3, 4.2, 4.4, or 5 item 4), and the two references to OpenYAM's contact-escalation ladder, which the MS1 spec never had, now name openyam-lab explicitly. `tests/test_spec_citations.py` pins that no `section 5.<digit>` string remains under `source/`, `scripts/`, `lerobot_stack/`, or `tests/`.
- **`feat/ms1-env` still exists on the remote** at `49476a6`, a commit already in `main`'s history, despite PR #2 having been merged. It should be deleted; that is a mutation on a shared repository and is left to the owner.
- **Observability gaps carried from Part E:** `CONVERT_OK` is parsed and discarded, so a *successful* convert leaves no log and the ledger's `converted_total` is its only durable trace; and dropping `exec` from `train_act.sh` (required for its EXIT trap to run at all) means a SIGTERM addressed to the wrapper's PID alone would orphan `lerobot-train`, though a process-group signal still reaches both.

## 8. Reproduction

```bash
# 0. Environments. The Isaac Sim venv is Python 3.11; the LeRobot venv is 3.12; uv's CPU
#    environment is 3.12. They are deliberately separate.
export ISAACLAB_PATH="$HOME/src/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES          # the setup script refuses to accept this for you
bash scripts/setup_isaac_env.sh          # Isaac Sim 5.1 + Isaac Lab v2.3.2 + the pinned maker_arm_rs wheel + this repo, editable
bash lerobot_stack/setup_lerobot_env.sh  # lerobot 0.5.1 into lerobot_stack/.venv
uv sync --group dev                      # CPU tests
uv run pytest && uv run ruff check . && uv run ruff format --check .

# Every GPU command below needs the Isaac venv activated first:
source "$ISAACLAB_PATH/.venv/bin/activate"

# 1. Vendor the URDF (already committed; re-run only to re-vendor at a new PIN_SHA).
bash scripts/vendor_maker_arm_description.sh
uv run pytest tests/test_urdf_validation.py

# 2. Build the USD. Not committed, and nothing checks its freshness: a stale USD loads
#    without complaint and then misbehaves physically.
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/convert_urdf_to_usd.py --headless   # CONVERT_OK

# 3. Smokes.
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_usd.py --headless                                                     # SMOKE_OK
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless                                               # MakerArm-LegoSort-v0
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless --task MakerArm-LegoSort-IK-Rel-v0
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless --enable_cameras --task MakerArm-LegoSort-IK-Rel-Vision-v0
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless --enable_cameras --task MakerArm-LegoSort-Datagen-v0
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless --enable_cameras --task MakerArm-LegoSort-Eval-v0
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/smoke_lego_sort.py --headless --num_envs 4 --num_bricks 8

# 4. Checks.
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/scripted_grasp_check.py --headless                                          # GRASP_OK, 10/10
for i in 0 1 2 3 4; do                                                                                              # one process per layout:
  "$ISAACLAB_PATH/isaaclab.sh" -p scripts/release_envelope_check.py --headless --layout_index "$i"                  # the single-process loop wedges
done                                                                                                                # ENVELOPE_OK x5, 40/40
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/oracle_check.py --headless --trials 2                                       # ORACLE_OK
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/oracle_check.py --headless --enable_cameras --with_cameras --trials 1        # ORACLE_OK
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/generate_batch.py --headless --enable_cameras \
    --out_dir /tmp/batch0 --attempts 10 --num_envs 12 --probe                                                        # BATCH_OK

# 5. The acceptance campaign. Generation and LeRobot conversion, run by run, crash-guarded.
#    Daemonise it: this host sweeps long background jobs, and setsid alone is not always enough.
setsid bash -c 'cd '"$PWD"' && python3 scripts/generate_dataset.py \
  --out_dir ~/data/maker-arm-ms1/work --dataset_root ~/data/maker-arm-ms1/dataset \
  --target 50 --attempts_per_run 50 --num_envs 12 --seed 20260905 \
  > ~/data/maker-arm-ms1/supervisor.log 2>&1' < /dev/null &
# -> per-run BATCH_OK, then CONVERT_OK, then CAMPAIGN_OK total=<n> runs=<m>
# generate_dataset.py is stdlib-only python3 and needs no virtualenv of its own; it launches
# generate_batch.py under isaaclab.sh and convert_to_lerobot.py under the LeRobot venv.

# 6. Train smoke.
STEPS=100 OUT=~/data/maker-arm-ms1/train/smoke JOB_NAME=act-smoke WANDB_ENABLE=false \
  bash lerobot_stack/train_act.sh                                                                                   # TRAIN_OK steps=100
```

## 9. Post-closure: the gate re-run on the oracle planning fixes (2026-09-07, later the same day)

This section was written after the milestone closed, on the branch `fix/ms1-oracle-gate`. It does not change the record above: section 6's gate was pre-registered, was evaluated once on run 0 at commit `9be229d`, and failed. What follows is a diagnosis of *why*, two fixes, and one new acceptance-class run of the same command on the fixed code. That run is a measurement of the fixed oracle, not a re-gate of the closed milestone, and it is reported with the same discipline: one run, launched once, nothing re-rolled.

### 9.1 Method: instrument the oracle, replay the planner on CPU

Section 6.3's hand-off described the failure modes from the sidecar's counters alone, which was all the acceptance run recorded. The diagnosis started by recording more. A throwaway driver (`/tmp/ms1diag/diag_batch.py`, not committed: a `DiagRig(OracleRig)` subclass over the same `generate_batch.py` construction, `EXPORT_NONE`, cameras off) logged one JSON line per phase end — ending phase and outcome, the target brick's pose and whether it was stowed, the LIFT pinch evidence, the arm's measured joint vector, and whether the *next* leg could be planned — plus every watchdog trip with joint margins, and a pre-step brick snapshot for each episode's end. Twelve envs, run 0's own seeds (`run_seed 3081035816`, `layout_seed 3081036316`, permutation `0,3,1,2`), 23 episodes before it was stopped: 18 successes, 2 time-outs, 1 watchdog, 2 grasp-retry abandons — every mode the acceptance runs produced, each at least once.

The second half of the method is what made the fixes cheap. `core.global_ik` and the runtime's planning code are CPU torch over the vendored URDF, and `tests/test_oracle_runtime_frames.py` already carried a stub env (`_stub_rig`) that constructs a real `OracleRig` without Isaac Sim. Feeding a logged joint vector into that rig and calling `_plan_leg` reproduces the GPU planning decision **exactly** — same plan length, same end pose to three decimals, same failure — so every hypothesis below was tested in seconds on CPU before any GPU minute was spent, and the fixes are pinned as CPU tests on the recorded vectors.

### 9.2 What each failure mode actually was

| mode | section 6.3 reading | measured mechanism |
| --- | --- | --- |
| `time_out` | "ran out of step budget" | **A brick lost after a completed carry, then ~1500 idle steps.** Both time-outs were near picks (radius 0.155 m and 0.140 m, azimuth −11°) bound for the far −y bin. RAISE's Cartesian ladder failed there and `_plan_direct` solved the endpoint 1.25 rad away in base yaw — inside `MAX_LEG_JOINT_JUMP = 1.5` — on a branch with the wrist near its stop. From that pose EXTEND had no continuous path at any of the eleven heading rungs and no direct fallback; RELEASE opened the jaws over the scatter band, the brick landed flat on the desk at radius 0.17–0.20 m, RETREAT marked it sorted (`core/oracle.py`'s RETREAT branch was unconditional), and the oracle sat in DONE from step ≈3900 to the 5400-step time-out with a brick in reach. Replayed on CPU: EXTEND fails at waypoint 4 of ≈115 from the RAISE-end pose and plans at once, at the held heading, from the LIFT-end pose. |
| `watchdog` | "the noisy one" | **Downstream of the same contorted RAISE pose.** The one watchdog abandon followed an unplanned EXTEND on brick 2; the next HOVER's joint-space walk from that pose ran `link_005_joint` to 0.018 rad from its stop for 30 consecutive steps. |
| `grasp_retries_exhausted` | "a parameter question" (`MAX_GRASP_RETRIES`) | **An unplannable pick retried three times against the same target.** `unplanned_legs` of 9 in every such episode is HOVER + DESCEND + LIFT × 3 attempts. Two shapes: a band-edge brick (radius 0.274 m, azimuth +40°) with **no hover IK solution at the drawn 0.138 m hover height at 24, 64 or 256 restarts**; and a near brick (radius 0.146 m) whose hover solved at 0.129 m only on a branch with the wrist 0.18 rad from its stop, from which no descend rung was walkable. Both plan at 0.10 m above the brick, descend included, with 0.28–0.34 rad of hover margin. The retry count was never the lever. |

Two earlier readings are therefore withdrawn: the "step budget" explanation for time-outs, and `MAX_GRASP_RETRIES` as the grasp-retry lever. Section 6.3's bounded ceilings stand as arithmetic but no longer describe distinct modes — all three were the planner.

### 9.3 The fixes, both in `oracle_runtime.OracleRig._plan_leg`

1. **RAISE lookahead** (commit `5c02669`). After planning RAISE, the runtime dry-runs the EXTEND leg — `EnvOracle.extend_goal()`, the same heading ladder and fallback the real EXTEND will use — from the candidate plan's end pose, and treats the RAISE as unplanned when that fails. RAISE is the one leg the episode can do without: the runtime already advanced straight to EXTEND on an unplannable RAISE, and Task D5 measured the LIFT pose as EXTEND's best start. A veto is counted in the new `raise_vetoed` episode stat. `_plan_carry` now returns whether it used the direct fallback instead of counting it, so a dry-run plan that is never walked cannot inflate `direct_legs`.
2. **Hover-height ladder with a DESCEND lookahead** (commit `ef3562d`). HOVER tries the episode's drawn hover height first, then `HOVER_HEIGHT_LADDER_M = (0.10, 0.08, 0.06)` m above the brick, taking the first rung whose hover solves *and* from whose end pose the descend ladder plans (`EnvOracle.descend_goal(yaw)`, at the yaw the HOVER goal carries, with `DESCEND_CLEARANCE_MEASURED` applied as `_enter_phase` applies it). A lowered rung moves `goal.target_pos[2]` down with it, because `_phase_result` grades arrival against the goal and a hover walked to 0.10 m can never converge on a target at 0.13 m. Counted in the new `hover_lowered` stat. The hover height is a via pose and a dataset-diversity draw (`OracleNoise.hover_height`), not a grasp parameter — the descend commands the brick's own z whatever the hover was — so a lowered rung costs the dataset a little variety on the affected picks and nothing else.

Neither fix changes a measured constant, the state machine's phase order, or the spec's action space. `core/oracle.py` gains two public goal constructors split out of `next_goal` so the lookahead and the real leg ask for the same thing. CPU suite: **325 passed** (320 + 5 new), `ruff check` and `ruff format --check` clean, all local (section 7's CI note still applies).

### 9.4 The re-run

`generate_batch.py --headless --enable_cameras --attempts 50 --num_envs 12 --probe`, run 0's seeds and permutation, `--max_wall_minutes 240`, at commit `ef3562d`. Log `/tmp/ms1gate/run0.log`, sidecar `/tmp/ms1gate/run0/shard_000.meta.json`.

| criterion (spec section 5) | required | acceptance run 0 (`9be229d`) | **re-run (`ef3562d`)** |
| --- | --- | --- | --- |
| oracle success rate, 12 envs, `--attempts 50` | ≥ 0.70 | 30/50 = 0.600 | **42/50 = 0.840** |
| clamp fraction | ≤ 0.01 | 8.23e-05 | **9.30e-05** (22 clamped env-steps of 236,640) |
| NaN abandonments | 0 | 0 | **0** |

Full `RUN_STATS`: `attempts 50, successes 42, time_outs 4, abandons 4, abandon_reasons {watchdog: 4}, watchdog_trips 4, control_steps 19720, clamped_steps 22, clamp_fraction 9.2968e-05, wall_seconds 7183.6, sim_steps_per_sec 2.745, deadline_hit false, success_rate 0.84, wall_sec_per_episode 143.67`. Wilson 95% interval for 42/50: [0.715, 0.917] — the first sample in this milestone whose interval **excludes 0.70 from below**. Against run 0, the difference is 0.240 with a two-proportion z of 2.67 (p ≈ 0.008); against the pooled acceptance rate of 0.630 over 100 episodes, z = 2.65. `grasp_retries_exhausted` went from 7/50 to **0/50**; `time_out` from 7 to 4; `watchdog` from 6 to 4. The new counters fired 7 times (`raise_vetoed`, seven episodes) and 3 times (`hover_lowered`, three episodes) in 50 episodes, so the lookaheads act on a minority of legs and the majority of the dataset is generated exactly as before. The price is wall clock: 143.7 s per episode against 127.1 s, 13% more, from the extra EXTEND and DESCEND plans.

### 9.5 What remains, and one prepared change that was not applied

The eight failures of the re-run are four `watchdog` abandons and four `time_out`s. All four time-outs are again `cycles = 6, sort_attempts = 6, bricks_delivered = 4–5` — a brick lost after a carry and then written off — and two of them are episodes where a RAISE was vetoed, so the loss happened on some other leg (an EXTEND from the LIFT pose with no plan, or a drop in transit); the re-run has no per-phase log to say which. The watchdog abandons were not instrumented either. Both are the next diagnostic targets, and the instrumented driver of section 9.1 is the instrument to use.

The lost-brick half of the time-out mode has a prepared, tested, **GPU-unvalidated** change on the local branch `wip/ms1-repick` (commit `2e914c8`): RETREAT adds a brick to `_sorted` only when the scene shows it stowed, so a brick still inside the top-down band is picked again with a fresh retry budget, and one out of reach abandons at once under the existing `no_reachable_brick` reason instead of idling 1500 steps. It is a change to `core/oracle.py`'s state machine, which is why it was kept out of the measured branch: the re-run above validates commits `5c02669` and `ef3562d` and nothing else, and folding an unmeasured third change into a passing branch would make the measurement describe code that was not run. Applying it costs one more acceptance-class run.
