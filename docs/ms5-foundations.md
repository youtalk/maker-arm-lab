# MS5: real foundations

Date: 2026-10-16. This document pre-registers the arm sessions of the MS5 revision before any
powered session runs. Every rule below is fixed before its number exists. After a session, only
the Results section changes, and it sits below the pre-registrations. The design is the MS5
real foundations spec (claude-memory, `specs/2026-10-15-ms5-real-foundations-design.md`). "MS5
revision section N" below means a section of that spec.

Values marked `INITIAL` are placeholders. The owner can change them before session 1. After
session 1 starts, they are fixed.

## 1. Exposure steps (pre-registered 2026-10-16)

Amended 2026-10-16 before session 1.

Source: MS5 revision section 8.7. The twin step is gone. The steps run after the mock and
`SimArm` runs, which need no arm.

| Step | Arm state | Torque caps (`caps` in the rig) |
|---|---|---|
| 1 | One joint moves. The other joints hold. | 0.25 of the profile on all joints (`INITIAL`) |
| 2 | Full arm | 0.5 of the profile (`INITIAL`) |
| 3 | Full arm | Nominal profile caps |

A reduced cap can only lower the profile cap, never raise it. The rig's `caps` scale three keys
of the profile per joint: `tau_max`, `kp` and `kd`. The controller's error clamp `e_max` follows
kp so that the spring's largest torque, `kp * e_max`, stays the profile's: `e_max_j = 0.03 /
kp_scale_j` rad (0.03 rad is the SDK's default), written into the session's tracking
parameters (ruling R33). An explicit `e_max` in the parameters wins only where it does not
exceed that value. Without this rule a cap of 0.25 cuts the spring to a quarter while the
gravity feed-forward keeps its model error: on the bench, with the model at 0.65 of the plant's
mass, j2 ran 1.5 rad to its stop; with the rule it drifts 37 to 46 mrad. The reduced `tau_max`
still bounds the whole command.

Joint frame gate (ruling R32). The tracking controller's model (gravity, FK) works in URDF joint
coordinates. The only calibration the session accepts today is the identity, so the model reads
the motor frame's angles, and the profile's limits are the motor frame's too (j3 [3.882, 7.955]
rad against the URDF's [0, 3.14]; j5 and j6 disjoint). Until maker-arm-rs re-expresses the
limits through `direction` and `offset`, a session on the real arm runs only the SDK's hold
controller (`scripts/real_session.py --controller hold`: no model, PD at the start pose, every
tick logged), and `--controller tracking` is refused before connect. Every model-based step of
sessions 2 to 9 (gravity compensation, excitation, identification, deadband, jitter baseline,
the A/B) waits on that change. A real session also refuses an input that is not a terminal,
since the end of input releases the arm (ruling R34).

Promotion rule from one step to the next (`INITIAL`). All three must hold. Each names the
counter it reads:

- No fault bit on any motor during the step: `faults` of the `SESSION_OK` line (ticks with a
  fault bit), and no `SESSION_FAIL` with reason `fault`.
- Clamped ticks are under 1 % of the ticks: `clamped` over `ticks` of the `SESSION_OK` line.
- The 60 s hold drifts under 2 mrad per joint: `hold_drift` of `scripts/real_motion_read.py`
  (per joint, the mean q over the last 1 s of the hold less the mean over the first 1 s after
  the 1 s settle, absolute; the largest over the holds).

A step that fails the rule repeats at the same caps after the cause is found. It never promotes.

Standing conditions for every powered session (MS5 revision section 8.7):

- A person stays at the e-stop. The Thor runs no Kachaka or Autoware stack.
- The first gravity-compensation session starts with the arm physically supported.
- The observer compensation and the learned term pass through the clamp of `Tracking`. Each has
  its own per-joint output clamp and its own switch, off by default.
- The hold must survive the loop's worst tick, because motors disable themselves after 200 ms
  without frames (`CAN_TIMEOUT`).

## 2. Bring-up rules, session 2 (pre-registered 2026-10-16)

Amended 2026-10-16 before session 1.

Source: MS5 revision section 8.2. Under the joint frame gate of section 1, session 2 runs on the
hold controller.

- A 10-minute telemetry capture through the Rust stack has no fault and no dropped tick. It
  reads `faults` (zero), `dropped` (zero: ticks the 4000-tick rings lost between drains) and
  `late_ticks` (zero: gaps between consecutive logged ticks over 1.5 tick periods, 7.5 ms) of
  the `SESSION_OK` line and the log's attrs. `max_interval_s`, the largest such gap, is
  reported next to them, with no bar of its own.
- A 60 s hold drifts under 2 mrad per joint: `hold_drift`, as in section 1.
- The base registration is set so that FK agrees with the overhead camera's view of the gripper
  tag. Its RMS disagreement is reported in mm. There is no pass bar on it. It is the baseline.

## 3. The identification rule, sessions 3 and 4 (pre-registered 2026-10-16)

Amended 2026-10-16 before session 1.

Source: MS5 revision section 8.3.

- Data: ten excitation trajectories under reduced caps, logged by the telemetry ring. One run
  seed S = 20261016 (`INITIAL`, the owner can change it before session 3) writes them as
  `exc_00` to `exc_09`.
- Held-out set: `exc_08` and `exc_09` of that seed (`INITIAL`). The fit never sees them.
- Rule: the identified model's residual torque RMS is lower than the URDF model's on every
  joint, scored separately on two held-out groups, and both must pass: the held-out
  trajectories (`exc_08` and `exc_09`) and the held-out static poses (`static_00` to
  `static_02`, in `plan_heldout.json` with the two trajectories). The margin is 0: strictly
  lower. A joint whose two models' predictions differ by at most `TIE_NM` = 0.01 N m
  (`INITIAL`, session 3 reads the motors' torque resolution) on every sample of a group is
  reported as tied and not judged in that group: at a static pose j1 carries no gravity torque
  under any model, and the wrist joints very little, so a strict rule there reads only the
  torque noise. `scripts/real_ident.py` reports both groups, their tied joints and the
  verdict.
- `COND_MAX` is 200. A base-parameter regression whose condition number is above it is not
  accepted, and the excitation set is redesigned.
- The numbers, the margin (0), the seed and the held-out file indices go into the ledger before
  the fit.
- Reported, with no bar: the static error at the desk pose under each model, and the largest
  resistance torque while a person drags the arm with the admittance on.

## 4. The deadband rule, session 5 (pre-registered 2026-10-16)

Source: MS5 revision section 8.4.

- The deadband `tau_th` per joint is `K_DEADBAND` times the RMS of the observer's residual in
  free motion. `K_DEADBAND` is 4.
- Free-motion bar: over the fixed trajectory set, `yield_mm` is at most 0.5 mm.
- Push bar: a 5 N push at the tip, read on a scale, gives `yield_mm` of at least 2 mm.
- The push bar must still hold with the observer compensation on.

## 5. The jitter targets (to be written after session 3, before session 6)

Amended 2026-10-16 before session 1.

Source: MS5 revision section 8.5. The baseline runs on `Tracking` as it is on main, with the URDF
model. Session 5 and session 6 fill the baseline column. The target column is written after
session 3 and before session 6. No compensation number is read before the targets exist.

| Side | Metric | Unit | Baseline | Target |
|---|---|---|---|---|
| Motor | `tracking_rms` (q_r minus q during transport) | rad |  |  |
| Motor | `stop_sway` (peak to peak of q in the 0.5 s after a transport, trend removed) | rad |  |  |
| Motor | `hold_jitter` (std of q over a hold, after a 1 s settle) | rad |  |  |
| Motor | `free_residual_rms` (observer r in free motion) | N m |  |  |
| Tip | `tip_sway` (peak to peak of the gripper tag over the stop window) | mm |  |  |
| Tip | `flex_share` (variance share of tag motion that FK of measured q does not explain) | fraction |  |  |
| Tip | tip path error (the gripper tag against FK of q_r during a transport; read added before session 5) | mm |  |  |
| Contact | `yield_mm` (the admittance's shift of the reference at the tip: FK of q_r against FK of q_r less the offset; not the measured tip) | mm |  |  |
| Contact | `drag_resistance` (largest resistance torque while hand-guided) | N m |  |  |

A motor-side target is met on the worst joint: the largest value over the six joints must be at
or below the target (`free_residual_rms` included). A tip or contact target is met on the
pooled tip value. The tip path error needs the tag's offset registered on the gripper, so its
read is added before session 5, which is the first session that reads it.

## 6. The A/B rule and the timing bar (pre-registered 2026-10-16)

Amended 2026-10-16 before session 1.

Source: MS5 revision section 8.6.

Three arms run on the same held-out set and the same fixed trajectory set:

1. The identified model alone.
2. Arm 1 plus the observer compensation.
3. Arm 2 plus the learned residual (MLP).

Adoption rule, for each layer against the arm before it. All must hold:

- The layer meets the targets of section 5.
- No worse tracking (`tracking_rms`) and no worse tip sway (`tip_sway`).
- The 5 N push of section 4 still gives `yield_mm` of at least 2 mm.
- For the MLP only: `free_residual_rms` is also lower, compared as the RMS over the six joints
  (the section 5 target is still met on the worst joint).

A layer that fails is not adopted. The null result is recorded in the Results section.

Exposure for each layer: the bench plant first, with an injected unmodeled friction, then one
joint at low caps, then the arm under reduced caps, then nominal caps. Section 1 states the
promotion rule.

Timing bar: with both layers on, `update()` p99 is at most 100 us on the Thor. It is read with
`scripts/track_timing.py --params PARAMS.json --residual MODEL.json`, with the session's
observer parameters and the trained residual; its `residual_active_share` must be above 0, or
the read timed a residual that skipped every tick. A layer that breaks the bar is not
adopted, whatever its jitter gain.

## Results

Results land below this line after the sessions. Nothing above this heading changes after
session 1, except the target column of section 5, which is written once, before session 6.
