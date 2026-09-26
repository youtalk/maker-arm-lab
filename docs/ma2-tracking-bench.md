# MA2: the tracking bench gate

Date: 2026-09-26. This document pre-registers the gate before the first full run. After the
gate runs, only the Result section changes.

## 1. Purpose

The compliant tracking controller lives in maker-arm-rs, on the `feat/tracking` stack
(`feat/loop-rt`, `feat/dynamics`, `feat/tracking`). It replaces the plain profile PD with a
model feed-forward and a compliant response to contact. An arm-only MuJoCo bench in this
repository gates the controller before the real arm runs it. The design is
`~/src/claude-memory/youtalk/maker-arm-lab/specs/2026-09-25-compliant-tracking-controller-design.md`
("tracking design" below). Task 11 runs the gate with `scripts/mj_track_bench.py` and records
the result here.

## 2. Controller summary

The controller reads a plan of time-stamped rows, six joint targets and a gripper target per
row. It interpolates the plan at the current time and holds the nearest sample before the
first row and after the last row. It filters each joint with a critically damped second-order
filter, which turns the plan into a smooth reference: position `q_d`, velocity `dq_d` and
acceleration `ddq_d`. The filter integrates over the measured interval between ticks, not a
fixed 5 ms, so a late tick does not corrupt it.

A momentum observer estimates the external torque `r` at each joint, from the model and the
measured torque. A deadband ignores `r` below a per-joint threshold `tau_th`. Past the
deadband, an admittance moves the reference along the push, with a yield stiffness
`K_a = kp / 4` (the owner's ratio). When the push ends, the admittance pulls the reference
back to the plan.

A feed-forward term adds the model torque to the MIT command. The gravity term stays on at
all times. The inertial, Coriolis, friction and velocity terms follow a contact gate. The gate
closes over 20 ms while any joint's observed torque stays above its deadband. It opens over
100 ms once every joint stays under its deadband for 50 ms. An error clamp keeps the MIT
position command within `e_max = 0.03 rad` of the measured position, at all times.

On the Jetson Thor, the loop thread can opt into a `SCHED_FIFO` priority and a CPU pin. This
is off by default. Python and its own threads keep the default scheduling in every case.

## 3. Plant

`source/maker_arm_lab/mj/bench.py` builds the plant. It is the vendored URDF through the
scene's compiler block and the Z-up mount, with gravity on and `gravcomp` off. The two jaw
joints are removed. The jaw bodies sit at a fixed opening instead (`dynamics_model.JAW_FRACTION`,
half travel). The loop does not model the gripper.

- Timestep `PHYSICS_DT` = 1 ms, integrator `implicitfast`.
- The MIT law runs at every physics step. The plant holds each command for one `TICK_S` = 5 ms
  tick (`STEPS_PER_TICK` = 5 physics steps per tick). The total torque at each joint saturates
  at the motor rating, `MOTOR_RATING_NM`: 14 N m for an RS00 joint, 17 N m for an RS02 joint.
- Armature `ARM_ARMATURE` = 0.02. This constant is UNVERIFIED, not INITIAL, and it is already
  tracked in `docs/ms1-maker-arm-sim.md` section 4.3.
- Joint friction: Coulomb `COULOMB_NM` (0.1 N m on an RS00 joint, 0.2 N m on an RS02 joint,
  INITIAL) and viscous `VISCOUS_NMS` = 0.02 N m s/rad (INITIAL).
- The controller sees `q`, `dq` and `tau_m` from the tick before. `tau_m` is the applied MIT
  torque plus Gaussian noise, sigma `TAU_NOISE_SIGMA` = 0.2 N m, from the plant noise seed.
- Every command goes through `maker_arm_rs.clamp_command` with the sim profile, in the same
  order as the loop's own output stage.
- A massless fingertip sphere at `grasp_center`, radius `TIP_RADIUS` = 0.008 m (INITIAL), is
  the only geom that collides. It uses the sim's own contact settings.

## 4. Models

- Exact: the adapter's URDF bodies (`dynamics_model.urdf_bodies()`), the same bodies the plant
  uses.
- Housing: the housing mass model asset, `assets/housing_mass_model.json`. It is a one-time
  export from the housing study (`scripts/export_housing_mass_model.py`, provenance recorded
  in the file), at the vendored URDF's pinned commit. Its motor mass is the RS02 datasheet
  value, 0.380 kg, not 0.405 kg. The housing study's `effects.py` used 0.380 kg for its
  5.36 N m and 3.36 N m figures at the desk pose (tracking design section 3). The bench model
  matches the study it comes from.
- Report only: a plus or minus 30 percent sweep of the total mass on the exact model
  (`dynamics_model.scaled`). The sweep scales every mass and inertia and keeps every center of
  mass fixed.

The RNEA check against MuJoCo has two parts, both in `tests/test_dynamics_model.py`. The
algorithm agrees with `mujoco.mj_inverse` on MuJoCo's own compiled inertias within 1e-9 N m.
The adapter's URDF-derived inertias agree within 1e-6 N m. MuJoCo's compiler stores each
inertia as principal moments plus a rotation, and the scene also sets `balanceinertia`. Its
tensor therefore differs from the URDF's own numbers by a small amount. The measured worst
difference between the two is 5.7e-8 N m.

## 5. Rungs

- B0: the profile PD, `tau = 0`, `p` = the raw plan target, `v = 0`. This is today's upstream
  behavior.
- B1: B0 plus the exact gravity feed-forward. This is what the training sim assumes with
  `gravcomp = 1`, so B1 always uses the exact model, never the housing model.
- F: the full controller without the observer, the admittance, the gate or the error clamp.
- A: the full controller.

## 6. Scenarios

Four poses drive the scenarios: `desk` (`(0.0, -2.03, 1.71, -1.17, 0.0, 0.0)`), `reach`
(`(0.0, -1.2, 0.6, -0.5, 0.0, 0.0)`), `hover` and `release`. Hover is the IK solution for
`grasp_center` over the tray center, at height `0.12 + BRICK_SIZE[2] / 2` m. This is the
nominal hover height over a 2x4 on the desk. It uses tilt `GRASP_TILT_RAD` (20 degrees), jaw
heading 0, and lean `LEAN_AZIMUTH` (0.0 rad). Release is the IK solution at the mid-point of
the first desk box's band, `DESK_BOX_RELEASE_Z` = 0.10 m high, heading 0, lean `LEAN_AZIMUTH`.

Ruling: the release pose flies at `GRASP_TILT_RAD`, not `PICK_TILT_RAD`. The level release
asks for tilt `PICK_TILT_RAD` (0.0). At exactly zero tilt, the lean azimuth is undefined, and
the release point has no IK solution. The oracle runtime reads every carry leg's tilt as
`max(goal tilt, GRASP_TILT_RAD)`, so it flies the release leg at `GRASP_TILT_RAD` (20 degrees)
too. The bench uses `RELEASE_TILT = max(PICK_TILT_RAD, GRASP_TILT_RAD)`, the tilt the oracle
actually flies. "Level" in "level release" names the jaw-plane roll, not the tool's tilt.

1. Static holds: start at rest on each of the four poses. Push a one-row plan of that pose at
   `t = 0`. Hold for 3.0 s. Sag is the tip distance from the commanded tip position, read at
   `t = 2.0 s`, in mm. The metric is the worst sag of the four poses.
2. Transport: the oracle's transport leg from hover to release, one row every 1/30 s at
   `TRANSPORT_SPEED`. Push the whole leg as one plan at `t0 = 0.5 s`. Run to the last row's
   time plus 1.5 s. Sway is the norm of the tip's per-axis range (`max - min`) over the 0.5 s
   from the last row's time, in mm. Overshoot is the largest distance the tip travels past its
   final position, along the direction from the start tip to the final tip. It reads every
   tick from `t = 0.5 s` on, and it clips at 0, in mm.
3. Policy-like transport, report only: scenario 2's rows, cut into chunks of 50. Each chunk
   gets its own per-joint offset, drawn once from `N(0, 0.005)` rad (INITIAL) and added to
   every row in the chunk. Chunk `c` is pushed at its own first row's time. The transport leg
   is 37 rows, fewer than one chunk, so scenario 3 has exactly one chunk and one constant
   offset for the whole leg. It has no chunk boundary inside the leg. Its overshoot reads
   against scenario 2's own, un-offset tip positions, so the reported number carries the
   offset's static bias. This is a report row, not a gate bar.
4. Wall hit: a rigid box wall stands at the tip position of scenario 2's mid-leg row, face
   normal to the horizontal part of the hover-to-release direction. Push scenario 2's plan
   through it. Metrics: the tip contact force at its peak, and the force at the tick nearest
   0.3 s after first contact. If the tip never touches the wall, the run is a bench defect and
   the suite raises.
5. Press: the tip descends onto the desk plane over the tray center, at `GRASP_TILT_RAD`. The
   contact height `z_c` is `TIP_RADIUS` (the sphere touches `z = 0`). Depth `d` comes from a
   closed-form check, with no physics rollout. Let `q_c` be the IK solution at `z_c` and `q_d`
   be the IK solution at `z_c - d`. A bisection finds the first `d`, in `[0, 0.05]` m to a
   tolerance of `1e-5` m, where `max_j kp_j |q_d - q_c|_j / tau_th_j` reaches 0.7. This check
   uses the exact model's calibrated `tau_th`. The plan is `n = ceil((0.03 + d) / 0.003)`
   equally spaced waypoints, at most 3 mm apart, from `z_c + 0.03` m down to `z_c - d`. The
   waypoints are solved as one continuous IK chain, the same continuity rule the oracle's own
   path solver uses. The solve
   is seeded from a standalone solve of the top waypoint. The plan walks that chain with a
   minimum-jerk timing law, at the base pace. Start at rest on the first waypoint, push at
   `t0 = 0.5 s`, run to the last row's time plus 1.0 s. Metric: the contact force 0.5 s into
   the hold.
6. False yield: a free-motion walk from the hover pose through 8 random targets, drawn in a
   box (`x` in `[0.10, 0.26]`, `y` in `[-0.20, 0.20]`, `z` in `[0.08, 0.16]` m) at
   `GRASP_TILT_RAD`. This walk uses its own seed and is disjoint from the calibration walk.
   Push the whole walk at `t0 = 0.5 s`, run to the last row's time plus 1.5 s. Metric: the
   largest tip displacement, over every tick, that the admittance offset puts on the
   reference.

Growing oscillation: for a run, compare two peaks of `max_j |dq_j|`, the 0.5 s before the run
ends against the 0.5 s before that. When the later peak is larger than the earlier one, and
larger than 0.02 rad/s, the run reads "growing".

Ruling: the housing growing-oscillation gate bar reads every scenario, scenario 3 included.
This ruling came from a later review, after scenario 3 was fixed as a report row. It makes the
bar stricter. A policy-like offset that only shows up as growing oscillation still fails the
housing model.

## 7. Calibration rule

This rule is fixed before the gate run. It runs on rung A with `tau_th = infinity`, so the
observer has no deadband. It runs separately for the exact model, the housing model, and each
mass-sweep model, each with its own `tau_th`:

- Run the calibration walk (its own seed, disjoint from every gate walk) from the hover pose,
  pushed at `t0 = 0.5 s`, to the last row's time plus 1.0 s. Read `max|r_j|` over every tick,
  including the 0.5 s before the push.
- Run a 2.5 s hold at the desk pose. Read `sigma_r_j`, the standard deviation of `r_j` over
  ticks with `t` in `[0.5, 2.5]` s.
- `tau_th_j = max(1.25 * max|r_j|, 3 * sigma_r_j)`.
- The plant noise seed is the suite seed, for every run, calibration included.

## 8. Gate bars

Bars for rung A. "rel" and "band" read against B1 on the exact model.

| model | metric | bar |
|---|---|---|
| exact | static sag, worst of 4 poses | <= 1 mm |
| exact | residual sway after the stop | <= 0.3 x B1 |
| exact | overshoot past the final pose | <= 0.3 x B1 |
| exact | false-yield tip drift | <= 1 mm |
| exact | wall force 0.3 s after contact | <= 0.5 x B1 |
| exact | wall force peak | <= B1 |
| exact | press force below the deadband | B1 +- 20 % |
| housing | residual sway after the stop | <= 0.5 x B1 |
| housing | false-yield tip drift | <= 3 mm |
| housing | growing oscillation | none in any scenario |
| housing | static sag | report only |
| Thor | `update()` p99 (T1) | <= 100 us |
| Thor | largest tick interval, FIFO and pin, 10 min under load (T2 b) | <= 10 ms |

`source/maker_arm_lab/mj/bench_gate.py`'s `BARS` computes the first 10 rows and the suite's
`pass` field covers them. The housing static sag row is a report number, not a pass or fail
check. The two Thor rows come from the Thor procedure below and go in this document by hand.

Report rows, not gated: the plus or minus 30 percent mass sweep, B0, F, scenario 3, a Machine A
run of T1, T2 run a and the ACT run.

## 9. INITIAL values

| constant | value | file | measured by |
|---|---|---|---|
| `COULOMB_NM` | RS00 0.1 N m, RS02 0.2 N m | `mj/bench.py` | MA2 |
| `VISCOUS_NMS` | 0.02 N m s/rad | `mj/bench.py` | MA2 |
| `TIP_RADIUS` | 0.008 m | `mj/bench.py` | MA2 |
| `DESK_BOX_RELEASE_Z` | 0.10 m | `core/desk_boxes.py` | MA2 |
| `CHUNK_SIGMA` | 0.005 rad | `mj/bench_gate.py` | MA2 |
| `f_r` (reference filter corner) | 10 Hz | maker-arm-rs `TrackingParams` | MA2 |
| `k_o` (observer gain) | 50 1/s | maker-arm-rs `TrackingParams` | MA2 |
| `f_y` (admittance corner) | 2 Hz | maker-arm-rs `TrackingParams` | MA2 |
| `e_max` (error clamp) | 0.03 rad, every joint | maker-arm-rs `TrackingParams` | MA2 |
| gate fall time | 0.020 s | maker-arm-rs `TrackingParams` | MA2 |
| gate rise time | 0.100 s | maker-arm-rs `TrackingParams` | MA2 |
| gate quiet time | 0.050 s | maker-arm-rs `TrackingParams` | MA2 |
| loop thread priority | 80 | Thor settings | the Thor session |
| loop thread CPU core | 5, 8 or 11 | Thor settings | the Thor session |

`bench_gate.TRACKING_INITIAL` is a copy of the `TrackingParams` defaults above, because the
Python binding has no getter for them. A test pins the copy against the installed build.

## 10. Thor procedure

The owner runs this section, at the machine, after the Rust pull requests merge. `sudo` needs
a password, and this host cannot reach the Thor today.

1. Record `uname -a` and `nvpmodel -q`.
2. Set the power mode and clocks: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
3. One-time setup for the rtprio limit:
   `echo "youtalk - rtprio 80" | sudo tee /etc/security/limits.d/99-maker-arm-rtprio.conf`.
   Log in again after this step, then check the limit with `ulimit -r`.
4. T1: run `python3 scripts/track_timing.py`.
5. T2 run a: start `stress-ng --cpu 14 &`, then run
   `python3 scripts/track_soak.py --minutes 10`.
6. T2 run b: run the same soak with `--rt 80:5` (or core 8 or 11 in place of 5).

Each run prints one token line (`TIMING_OK`/`TIMING_FAIL`, `SOAK_OK`/`SOAK_FAIL`) with a JSON
summary. Record each token line in the Result section below.

## Development run disclosure

While `bench_gate.py` was under development, the implementer ran the full suite once to prove
that it completes, about 5 seconds. That run read FAIL, on the exact model's overshoot and its
press force. No bar, INITIAL value or definition changed because of that run. The release-tilt
ruling in section 6 came from an IK failure found before that run, not from its result. The
growing-oscillation ruling in section 6 came from a later review, and it makes a bar stricter.
This document does not list that run's numbers. Task 11 records the official run.

## 11. Result

Task 11 ran the gate on 2026-09-26 with seed 20260925. The command was:

```
uv run --no-sync python scripts/mj_track_bench.py --out ~/data/maker-arm-tracking/gate-20260925.json
```

The full JSON lives at `~/data/maker-arm-tracking/gate-20260925.json`. It has the keys
`initial`, `tau_th`, `press_depth_m`, `setup`, `results`, `report`, `gate` and `pass`.

### Verdict

The suite printed `BENCH_FAIL`. Two of the ten gated bars failed: the exact model's overshoot
and the exact model's press force. The other eight bars passed.

### Deadbands (`tau_th`, N m, joints `link_002_joint` to `link_007_joint`)

| model | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| exact | 0.8009 | 0.7688 | 0.6381 | 0.5585 | 0.4730 | 0.6348 |
| housing | 0.7676 | 1.4808 | 1.1621 | 0.6683 | 0.4799 | 0.6356 |
| exact x0.7 (report only) | 0.8644 | 2.2290 | 1.7937 | 0.6123 | 0.4784 | 0.6357 |
| exact x1.3 (report only) | 0.7456 | 2.1857 | 1.6665 | 0.7353 | 0.4843 | 0.6337 |

### Press depth

The closed-form check found `d = 0.0010193 m` (about 1.02 mm) on the exact model.

### Gate table (rung A, "rel" and "band" rows read against B1 on the exact model)

| model | metric | value | reference | bar | pass |
|---|---|---|---|---|---|
| exact | static sag (mm) | 8.10e-10 | 9.29e-10 | <= 1 mm | true |
| exact | sway after the stop (mm) | 0.1648 | 1.0738 | <= 0.3x B1 | true |
| exact | overshoot past the final pose (mm) | 0.3575 | 0.5710 | <= 0.3x B1 | **false** |
| exact | false-yield tip drift (mm) | 0.0 | 0.0 | <= 1 mm | true |
| exact | wall force 0.3 s after contact (N) | 10.428 | 62.981 | <= 0.5x B1 | true |
| exact | wall force peak (N) | 70.625 | 80.880 | <= B1 | true |
| exact | press force below the deadband (N) | 1.1269 | 1.4809 | B1 +- 20% | **false** |
| housing | sway after the stop (mm) | 0.1593 | 1.0738 | <= 0.5x B1 | true |
| housing | false-yield tip drift (mm) | 0.0 | 0.0 | <= 3 mm | true |
| housing | growing oscillation | false | false | none | true |

For the overshoot row, rung A's tip travels 0.3575 mm past its final pose on the exact model.
B1's own overshoot is 0.5710 mm, so the bar sits at 0.1713 mm (0.3 times B1). Rung A's number is
about twice the bar.

For the press force row, rung A holds the tip at 1.1269 N of contact force below the deadband on
the exact model. The B1 +- 20% band around the reference of 1.4809 N runs from 1.1847 N to
1.7771 N. Rung A's value sits under the lower edge of that band.

### Report rows

B0 (profile PD, exact model):

| metric | value |
|---|---|
| static sag, worst of 4 poses (mm) | 19.586 |
| static sag by pose (mm) | desk 16.227, reach 19.586, hover 15.000, release 15.042 |
| sway after the stop (mm) | 1.038 |
| overshoot past the final pose (mm) | 0.0 |
| wall force peak / 0.3 s after contact (N) | 73.273 / 62.002 |
| press force below the deadband (N) | 17.072 |
| false-yield tip drift (mm) | 0.0 |
| growing oscillation | false |

F (the full controller without the observer, the admittance, the gate or the error clamp):

| model | static sag (mm) | sway (mm) | overshoot (mm) | wall force peak / 0.3 s (N) | press force (N) | false yield (mm) | growing |
|---|---|---|---|---|---|---|---|
| exact | 8.10e-10 | 0.1648 | 0.3575 | 74.352 / 62.794 | 1.1269 | 0.0 | false |
| housing | 3.189 | 0.1593 | 0.9559 | 73.910 / 62.900 | 0.0 | 0.0 | false |

Scenario 3, policy-like transport (chunk sway and chunk overshoot, one chunk for the whole leg):

| model / rung | chunk sway (mm) | chunk overshoot (mm) |
|---|---|---|
| exact / B0 | 1.028 | 0.0 |
| exact / B1 | 1.062 | 0.0 |
| exact / F | 0.1617 | 0.0 |
| exact / A | 0.1617 | 0.0 |
| housing / F | 0.1554 | 0.0 |
| housing / A | 0.1554 | 0.0 |
| exact x0.7 / A | 0.2656 | 0.0 |
| exact x1.3 / A | 0.2195 | 0.0 |

Mass sweep, plus or minus 30 percent on the exact model, rung A:

| sweep | static sag (mm) | sway (mm) | overshoot (mm) | wall force peak / 0.3 s (N) | press force (N) |
|---|---|---|---|---|---|
| x0.7 | 5.811 | 0.2646 | 0.0 | 76.647 / 9.931 | 5.177 |
| x1.3 | 5.828 | 0.2232 | 1.425 | 72.774 / 9.853 | 0.0 |

Housing static sag (report only, rung F and A agree): 3.189 mm, worst of 4 poses. By pose: desk
2.270 mm, reach 3.189 mm, hover 2.653 mm, release 2.389 mm.

Machine A T1 (`scripts/track_timing.py`, report row, not the Thor):

```
TIMING_OK {"ticks": 100000, "p50_us": 2.808, "p99_us": 11.896, "max_us": 361.732, "uname": {"system": "Linux", "node": "machine-a", "release": "7.0.0-31-generic", "version": "#31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2", "machine": "x86_64", "processor": "x86_64"}, "python": "3.12.3", "bar_us": 100.0}
```

Machine A soak, one minute, no FIFO and no CPU pin (`scripts/track_soak.py --minutes 1`, report row):

```
SOAK_OK {"pass": true, "enabled": true, "minutes_requested": 1.0, "elapsed_s": 60.000179280061275, "rt": null, "telemetry": {"max_interval": 0.006173641000000174, "late_ticks": 0, "gaps": 0}, "uname": {"system": "Linux", "node": "machine-a", "release": "7.0.0-31-generic", "version": "#31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2", "machine": "x86_64", "processor": "x86_64"}}
```

### Thor rows

T1 (`update()` p99 on the Thor) and T2 b (largest tick interval under load, FIFO and pin) are
pending the owner's Thor session.

### Erratum and analysis

This subsection comes from the final branch review, on 2026-09-26, after the official run. It
holds exploratory checks outside the pre-registration. They do not change the recorded verdict:
`BENCH_FAIL` stands. No bar, definition, INITIAL value or plant setting changes, and this
document proposes no retune.

The exploratory numbers come from runs of this branch's bench code with the official `tau_th`,
seed 20260925 and maker-arm-rs 6141b84. Each item names the plant change that its run makes. The
run scripts are not in the repository.

#### Erratum

1. Contact settings (section 3). Section 3 says that the fingertip sphere "uses the sim's own
   contact settings". This is wrong. `mj/bench.py` applies only `CONTACT_SOLREF` (0.005, 1.0) to
   the tip, the desk plane and the wall. It does not apply the sim's `DESK_SOLIMP` (0.99, 0.999,
   0.001, 0.5, 2.0), `DESK_SOLMIX` (1e6), elliptic cone, `impratio` 10 or `noslip_iterations` 3.
   The bench plant has the MuJoCo defaults for these: solimp (0.9, 0.95, 0.001, 0.5, 2.0), a
   pyramidal cone, `impratio` 1 and no noslip iterations. With the sim's desk settings on the
   press plant, B1's press force is 5.993 N, not 1.481 N. The plant of the official run does not
   change.
2. The exact model (section 4). Section 4 says that the exact model has "the same bodies the
   plant uses". The bodies are the same, but the plant adds two terms that the exact model does
   not have. The first term is the armature, `ARM_ARMATURE` = 0.02 kg m^2 per joint, and
   `maker_arm_rs.Dynamics` has no armature term. The second term is the joint friction
   (`COULOMB_NM` and `VISCOUS_NMS`), and the bench builds the exact model with zero friction. At
   the hover pose, the diagonal of the rigid-body mass matrix is 0.0385, 0.1313, 0.0730, 0.0176,
   0.0043 and 0.00057 kg m^2 (J1 to J6). The armature is 0.52, 0.15, 0.27, 1.13, 4.66 and 35.0
   times these values. With the torque noise off, the largest |r| per joint of the exact model
   is 0.18 to 0.52 N m on the calibration walk. It is 0.21 to 0.62 N m on the gate walk. The
   exact deadband is 0.47 to 0.80 N m, so the model error alone takes a large part of it.
3. The INITIAL table (section 9). The "measured by" column says MA2 for `TIP_RADIUS`,
   `DESK_BOX_RELEASE_Z` and `CHUNK_SIGMA`. MA2 identifies the arm's model and friction, and it
   does not measure these three. They are bench and scene choices. The table also leaves out
   `TAU_NOISE_SIGMA` (0.2 N m, `mj/bench.py`), the torque noise of the bench. The risk table of
   the tracking design treats this noise as INITIAL. The code now marks it `# INITIAL:`, and its
   value does not change.
4. The inertia wording (section 4). Section 4 says that "the scene also sets `balanceinertia`"
   and that the tensor "therefore" differs from the URDF. MuJoCo's principal-axis storage causes
   the difference. `balanceinertia` changes no body of this URDF: a compile with it off gives
   the same `body_inertia`, `body_iquat` and `body_mass`. The 5.7e-8 N m figure stands. A stray
   line break in scenario 5 of section 6 is cosmetic only.

#### What the two failed bars show

On the exact model, rungs A and F give the same value on every gated row except the two wall
rows. The only other difference is the scenario 3 chunk sway, 0.161733 mm on A against 0.161725
mm on F. In the official press run, the gate of A stays at 1.0 and its admittance offset stays
at 0.0. Its largest |r| reaches 0.756 of `tau_th`, at J3. So the contact side (the observer, the
admittance, the gate and the clamp) has no effect on the two failed rows. Both failures come
from the motion side that F also runs, the reference filter and the feed-forward, on this plant.

Overshoot. The overshoot of A is a real transient. It peaks at t = 1.73 s, 30 ms after the last
row at t = 1.70 s. At the end of the run, 0.0555 mm of it remains. The overshoot of B1 peaks at
t = 1.79 s. Without the INITIAL joint Coulomb friction of the plant, A reads 0.0545 mm and B1
reads 0.1803 mm, a ratio of 0.30. The bar for that run is 0.0541 mm, so A still misses it, by
less than 0.001 mm. The friction multiplies A by 6.6 and B1 by 3.2. So most of the 0.3575 mm of
A comes from the INITIAL Coulomb friction, which the exact model does not have.
The risk table of the tracking design answers the INITIAL plant values with relative bars on the
same plant. This row shows that a relative bar does not remove that risk.

Press. The press row asks whether A goes soft below the deadband. It does not, because the
contact side never acts. A reads 1.1269 N, and F reads the same value to the last digit. The gap
to B1 comes from the friction between the tip and the desk:

- The tip slides on the desk during the hold. From the last row to the end of the run, it moves
  2.304 mm across the desk on B1 and 0.591 mm on A.
- The force falls through the probe time. From the probe (0.5 s into the hold) to the end of the
  run, B1 falls from 1.481 N to 1.179 N. A falls from 1.127 N to 1.007 N.
- The verdict depends on the probe time. A/B1 is 0.761 at 0.5 s into the hold (fail), 0.818 at
  0.75 s (pass) and 0.854 at the end of the run (pass).
- With a frictionless contact between the tip and the desk, A/B1 is 1.029 (0.807 N against 0.784
  N), inside the band.
- With a frictionless contact and no joint Coulomb friction, B1 holds 0.762 N at 0.25, 0.5, 0.75
  and 1.0 s into the hold. A holds 0.784 to 0.790 N at these times, and A/B1 is 1.031.
- Without the joint Coulomb friction but with the contact friction, A/B1 is 0.620 (1.1114 N
  against 1.7927 N), outside the band.
- With the sim's own desk contact settings (erratum item 1), A/B1 is 0.393 (2.353 N against
  5.993 N).

These runs point to the friction between the tip and the desk. During the landing, the contact
friction locks in a sideways load along the desk. The size of that load depends on how each rung
lands. The deadband behavior, which the design meant this row to test, is not the cause.

#### Report-only notes

Wall scenario: a wrist crawl that no metric reports. On rung A and the exact model, the tip
first touches the wall at t = 1.100 s. The gate starts to close at t = 1.110 s. It reaches 0.0 at
t = 1.125 s and stays at 0.0 to the end of the run. The wall does not block J6 (the wrist roll).
J6 falls up to 0.448 rad behind its reference, at t = 1.590 s. From t = 1.3 s to t = 2.5 s, it
crawls at 0.356 to 0.361 rad/s. Its lag drops under 0.03 rad only at t = 2.805 s. The error
clamp limits this crawl: with the full clamp error, the PD drives J6 at most
`(kp e_max - F_c) / (kd + viscous)` = `(30 x 0.03 - 0.1) / (2.0 + 0.02)` = 0.396 rad/s. The
growing row reads false, because the late peak of 0.3600 rad/s is below the early peak of 0.3607
rad/s. Both peaks are 18 times the 0.02 rad/s floor. The housing model shows the same crawl
(0.3583 and 0.3590 rad/s). If the real arm acts like the bench, a fingertip touch during a carry
can stall the wrist for over a second.

Scenario 3 has no chunk switch. The transport leg has 37 rows and `CHUNK` is 50, so scenario 3
runs one chunk, as section 6 states. Its chunk overshoot reads 0.0 on every run. So scenario 3
tests one constant offset and no chunk switch.

Housing and x1.3 rows. On the housing model and the x1.3 sweep, the overshoot is mostly a static
offset along the travel direction. The gravity error of the model causes this offset.
On housing A, 0.756 mm of the 0.956 mm overshoot remains at the end of the run. On x1.3 A, 1.285
mm of the 1.425 mm overshoot remains. The press rows of these runs read 0.0 N because the tip
never touches the desk. The lowest point of the tip sphere stays 1.550 mm above the desk on the
housing model (F and A). It stays 3.319 mm above the desk on x1.3 A. The press depth is 1.02 mm.

#### Questions for the owner

- Do you want new press and overshoot definitions in a later, pre-registered gate?
- Do you want an armature term, a per-joint diagonal, in `maker_arm_rs.Dynamics`?
- Do you want a per-joint gate and clamp, so that the joints that a contact does not block keep
  their plan?

#### Reproducibility recheck

The recheck ran on 2026-09-26 with maker-arm-rs 6141b84 (`feat/tracking`) and the lab code of
this branch at commit 3c3d8ff. The command was:

```
uv run --no-sync python scripts/mj_track_bench.py --out ~/data/maker-arm-tracking/gate-20260925-recheck.json
```

It printed the same `BENCH_FAIL` token line as the official run, byte for byte. The recheck
JSON, `~/data/maker-arm-tracking/gate-20260925-recheck.json`, is byte-identical to the official
JSON: 437 values compared, no difference. So the fixed builds reproduce the official numbers.
This recheck is not a new verdict.
