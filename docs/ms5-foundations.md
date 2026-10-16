# MS5: real foundations

Date: 2026-10-16. This document pre-registers the arm sessions of the MS5 revision before any
powered session runs. Every rule below is fixed before its number exists. After a session, only
the Results section changes, and it sits below the pre-registrations. The design is the MS5
real foundations spec (claude-memory, `specs/2026-10-15-ms5-real-foundations-design.md`). "MS5
revision section N" below means a section of that spec.

Values marked `INITIAL` are placeholders. The owner can change them before session 1. After
session 1 starts, they are fixed.

## 1. Exposure steps (pre-registered 2026-10-16)

Source: MS5 revision section 8.7. The twin step is gone. The steps run after the mock and
`SimArm` runs, which need no arm.

| Step | Arm state | Torque caps (`caps` in the rig) |
|---|---|---|
| 1 | One joint moves. The other joints hold. | 0.25 of the profile on all joints (`INITIAL`) |
| 2 | Full arm | 0.5 of the profile (`INITIAL`) |
| 3 | Full arm | Nominal profile caps |

A reduced cap can only lower the profile cap, never raise it.

Promotion rule from one step to the next (`INITIAL`). All three must hold:

- No fault bit on any motor during the step.
- Clamped ticks are under 1 % of the ticks.
- The 60 s hold drifts under 2 mrad per joint.

A step that fails the rule repeats at the same caps after the cause is found. It never promotes.

Standing conditions for every powered session (MS5 revision section 8.7):

- A person stays at the e-stop. The Thor runs no Kachaka or Autoware stack.
- The first gravity-compensation session starts with the arm physically supported.
- The observer compensation and the learned term pass through the clamp of `Tracking`. Each has
  its own per-joint output clamp and its own switch, off by default.
- The hold must survive the loop's worst tick, because motors disable themselves after 200 ms
  without frames (`CAN_TIMEOUT`).

## 2. Bring-up rules, session 2 (pre-registered 2026-10-16)

Source: MS5 revision section 8.2.

- A 10-minute telemetry capture through the Rust stack has no fault and no dropped tick.
- A 60 s hold drifts under 2 mrad per joint.
- The base registration is set so that FK agrees with the overhead camera's view of the gripper
  tag. Its RMS disagreement is reported in mm. There is no pass bar on it. It is the baseline.

## 3. The identification rule, sessions 3 and 4 (pre-registered 2026-10-16)

Source: MS5 revision section 8.3.

- Data: ten excitation trajectories under reduced caps, logged by the telemetry ring. One run
  seed S = 20261016 (`INITIAL`, the owner can change it before session 3) writes them as
  `exc_00` to `exc_09`.
- Held-out set: `exc_08` and `exc_09` of that seed (`INITIAL`). The fit never sees them.
- Rule: on the held-out set, the identified model's residual torque RMS is lower than the URDF
  model's on every joint. Held-out static poses are scored the same way.
- `COND_MAX` is 200. A base-parameter regression whose condition number is above it is not
  accepted, and the excitation set is redesigned.
- The numbers, the margin, the seed and the held-out file indices go into the ledger before
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
| Contact | `yield_mm` (tip yield to a push) | mm |  |  |
| Contact | `drag_resistance` (largest resistance torque while hand-guided) | N m |  |  |

## 6. The A/B rule and the timing bar (pre-registered 2026-10-16)

Source: MS5 revision section 8.6.

Three arms run on the same held-out set and the same fixed trajectory set:

1. The identified model alone.
2. Arm 1 plus the observer compensation.
3. Arm 2 plus the learned residual (MLP).

Adoption rule, for each layer against the arm before it. All must hold:

- The layer meets the targets of section 5.
- No worse tracking (`tracking_rms`) and no worse tip sway (`tip_sway`).
- The 5 N push of section 4 still gives `yield_mm` of at least 2 mm.
- For the MLP only: `free_residual_rms` is also lower.

A layer that fails is not adopted. The null result is recorded in the Results section.

Exposure for each layer: the bench plant first, with an injected unmodeled friction, then one
joint at low caps, then the arm under reduced caps, then nominal caps. Section 1 states the
promotion rule.

Timing bar: with both layers on, `update()` p99 is at most 100 us on the Thor. It is read with
`scripts/track_timing.py`. A layer that breaks the bar is not adopted, whatever its jitter gain.

## Results

Results land below this line after the sessions. Nothing above this heading changes after
session 1, except the target column of section 5, which is written once, before session 6.
