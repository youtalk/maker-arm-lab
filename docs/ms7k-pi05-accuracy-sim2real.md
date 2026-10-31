# MS7k: π0.5 accuracy and sim2real

Date: 2026-10-31. Stage 3 ended at 02:52.

MS7k gave π0.5 one more week before the real Maker Arm comes. The owner set the design of
2026-10-28 as the goal and allowed deviations. This report cites it as "spec section N". It
lives outside the repository. The campaign ledger is `~/data/maker-arm-ms7/k/README.md`, and
`STATE.md` beside it is the resume pointer. The ledger names its rules K1 to K14. Every rule went
into the ledger before its number. Each read has a TAG, its name. The payloads of a read are under
`~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7k TAG starts with `k_`.

The names:

- S is the final MS7g policy: ACT `f/train_F5` 200k, as in the MS7j report.
- The LoRA 30k is the π0.5 LoRA of MS7j Track B at step 30000 (`j/lora/30000_n25`).
- C, L1, L2 and L3 are the stage 1 probes (section 4). Each continues the LoRA 30k for 5000 steps.
- The main run `k_main` continues C's 5k save on Colab (section 6). Its saves take the name M plus
  the run's own step, for example M5000.
- P-base is the perception baseline, and R-base is the real-condition baseline (section 2.3).
- Warp X plays a chunk X times faster. RTC is real-time chunking (MS7j report section 2).

## 1. Summary

The spec has four goals:

1. A π0.5 candidate that places more parts in the correct box than S. Each policy reads under the
   serving that it will use on the real arm.
2. A read that shows the candidate loses less than S on the sim2real rows.
3. A working Colab LoRA path on LeRobot 0.6.1, used for the main run.
4. The first π0.5 tolerance table for camera pose, image and light, beside S's. Both read on
   Machine A.

MS7k meets all four goals. The candidate is C, the plain 5000-step continuation of the LoRA 30k.

- Goal 1, K8 accuracy (section 7.1): C places 2571 against S's 2537 on screen A (D +34, SE
  33.72), and 2555 against 2536 on screen B (D +19, SE 29.29). The pooled D is +53 and the pooled
  SE 44.15, so K8 passes by 1.2 pooled SE.
- Goal 2, K9 sim2real (section 7.2): (a) C passes all 22 expected rows. (b) C has the smaller
  drop on 24 rows and S on 7. (c) C passes five report rows that S fails: overhead tilt -3, +6
  and -6 degrees, and the +5 cm forward and lateral shifts. K9 passes.
- Goal 3: C, L2, L3 and the main run trained on the Colab A100 with LeRobot 0.6.1 (section 5).
- Goal 4, the tolerance table (section 8): C passes every uncorrected overhead pose that the
  suite reads, up to 6 degrees of tilt and a +5 cm shift. S passes 1 degree and 1 cm. Both fail
  the 5 cm coupled shift, where the perceived boxes move too. On every other axis, both pass every
  level read.
- Speed decides nothing: C places 208.0 and 225.1 correct an hour on the screens, S 332.2 and
  343.9. C places about 36 % fewer correct parts an hour.

Caveats:

- K8's margin is small. C wins 24 of the 40 draws, ties 2 and loses 14.
- C's P-base read low, so its drops lean low. After a 1.20-point correction, K9 (b) still holds
  at 18 rows against 13. A shift above 1.5 points turns it over (section 7.2). (a) and (c) do not
  depend on it.
- The camera, image and light rows read each policy at its P-base serving, with no delay. No read
  combines a camera error with RTC at a delay of 6 (section 9).
- Stage 1 found no lever that beats C (section 4). The main run gave no gain: 5000 and 10000 more
  steps read 890 and 889 against C's 908 on draws 0 to 6 (section 6).

## 2. The rules

### 2.1 Rules K1 to K14

- K1: every read runs on Machine A (RTX 5090). Colab only trains. No job runs on Machine B, Machine C
  or Machine D.
- K2: selection reads use the screen scene, seed 20307000, draws 0 to 6 (1050 parts).
- K2a: K8 screen A is seed 20307000, draws 20 to 39. Screen B is seed 20307600, draws 0 to 19.
  Each screen holds 3000 parts.
- K2b: the sim2real rows use the MS7j suite scene, seed 20390000, draws 0 to 11 (600 parts).
- K3: the paired score. D is the sum over draws of the candidate's correct placements minus the
  other's. SE is sqrt(n) times the stdev of the per-draw differences.
- K4: Machine A runs one π0.5 load or training at a time, through `heavy.sh`.
- K5: the π0.5 serving of HD63, at warp 1.0 chunked for stage 0. S serves as in MS7j: 50 actions a
  forward at warp 1.5.
- K6: if a serving probe beats P0 by one paired SE, it replaces K5.
- K7: the real-condition serving. Both policies get an observation delay of 3 control steps. S adds
  an inference delay of 1 step. π0.5 adds RTC at warp 1.0 with the delay of K7a.
- K8: the accuracy rule. On each screen, the candidate's D against S is above 0. The pooled D over
  the 40 draws is at least one pooled paired SE.
- K9: the sim2real rule, the three parts of spec section 4. The pass rule of a row is MS7j J5: the
  drop plus one paired SE is 5 points or less.
- K10: the 31 sim2real rows (section 2.3).
- K11: the stage 1 probes. Each continues the LoRA 30k for 5000 steps at batch 32 with a fresh
  optimizer. The learning rate warms up for 200 steps to 5e-5, then falls by cosine to 1e-5.
- K12: the selection rule of spec section 6.3, written before the first probe read.
- K13: the main run stops, from spec section 6.4. The cap is 230 CU.
- K14: every new knob is off by default. A pinned draw with every knob off reads byte-identical to
  main.

### 2.2 Rulings that changed the plan

- K7a, amended 2026-10-28 17:48: the π0.5 delay comes from a timing run of the final serving on an
  idle 5090. It is 200 single-row forwards, and the delay is max(6, ceil(p90 / 33.33 ms)).
- K6b: D1 replaced the flow-step probe N20 by a jaw-noise probe J05. MS7h HD54 read 20 steps at
  -2.4 SE, and more steps lengthen the RTC delay.
- K6c: J05 needs a second read under RTC, because the jaw noise acts mostly through the RTC forward
  rate.
- K10a: a coupled row moves the image, and the gripper marker takes the shift back off the parts.
  So a coupled row measures the moved image plus boxes off by the shift.
- K11a: L2 is the spec's fallback, a flow time sampler toward low noise (alpha 1.0, beta 1.5).
  P0's p50 xy offset at grasp height is 1.7 mm, against S's 2.3 mm, so finer vision has little
  room.
- K11b: L3 and L2 trained on the Colab A100, not on Machine A.
- K13a: the main run schedule. It runs 20000 steps from C's 5k save, with a kept save every 5000.
- K13b: the main run started from C's 5k save before K12. K12 came about 9.5 h after L2 ended,
  and an idle session bills too. K12 later picked C, so no CU went to waste.
- K13c: the stage 3 candidate is the best by draws 0 to 6 among C's 5k save and the kept saves.
- K13d, 2026-10-30 12:01, before kept save 2's number: with 39.80 CU, a restart reaches only
  about step 13800. So the run restarts only if kept save 10000 beats the start (908) on draws 0
  to 6. Otherwise the K13 plateau stop under K13c ends the run at 10000, and the CU stay unspent.
- On 2026-10-29 at 21:48, L2's four probe rows stopped after its screen read. L2 lost to C on
  correct placements, so its rows did not affect K12. Stage 3 started about 2 h sooner.
- Stage 3 for C started at that time, before the main run ended. Under K13c, a later kept save
  that beat C was to get its own stage 3 reads. C's K8 passed on 10-30 at 12:22, before kept
  save 2 was read at 13:53.
- Spec section 6.9 has the owner run every `colab new`. On 10-30 the A100 pool returned 503 to
  it, and at the owner's request the agent ran it in a retry loop (`k/bin/colab_new.sh`). The
  owner still did the login and the Drive mount.

### 2.3 The sim2real rows

Each row adds one condition to one baseline. P-base is S as today, and π0.5 chunked at warp 1.0
with the K5 serving. R-base is the K7 serving.

| Rows | Baseline | Level | Count |
|---|---|---|---|
| R-base itself, overhead tilt ±1 degree, forward ±1 cm, lateral ±1 cm | P-base | expected | 7 |
| wrist pitch and yaw 2 degrees, field of view ±2 degrees on each camera | P-base | expected | 6 |
| JPEG 85, blur 1.0, exposure ±1 stop, hard light, coupled shift 1 cm forward and lateral | P-base | expected | 7 |
| varying delay, tracking lag 33 ms | R-base | expected | 2 |
| overhead tilt ±3 degrees, forward and lateral +5 cm (the four probe rows) | P-base | report | 4 |
| overhead tilt ±6 degrees, coupled shift 5 cm forward and lateral | P-base | report | 4 |
| tracking lag 100 ms | R-base | report | 1 |

The varying delay draws each chunk's delay from 1 ± 1 steps for S and 6 ± 3 for π0.5. The three
new knobs are in PR #224.

## 3. Stage 0: diagnostics and baselines

### 3.1 D1: the off-target first closes

D1 read existing traces of the LoRA 30k (`k/d1/report.md`). The definition is row (e) of the jaw
table. It counts a stowed cycle whose first close is more than 20 mm off in height or in xy.

- The serving noise explains almost none of them. Retry noise comes before 1 of the 192 real
  off-target first closes, and still resamples explain about 2.
- The arm is calm in the rest. On the screen, the LoRA minus S closes in the air +85 more times
  (SE 22.0), and at the part +14 more times (SE 4.6).
- 170 of the LoRA's 275 row (e) closes come at cycle step 0. These are not grasps.
- In-air closes come at transport height. Under RTC at a delay of 6, they triple against chunked
  warp 1.0 (288 against 93 on the suite). RTC also forwards 3.8 times as often, but no read
  separated the forward rate from the RTC guidance.
- Within 3 mm, the LoRA converts a close as well as S (0.753 against 0.754). Only 0.45 of its real
  closes land within 3 mm (S 0.58).

So D1 ranked L1 above L2 and named a jaw-noise probe (ruling K6b).

### 3.2 D2: the serving probes

P0 is the LoRA 30k at the K5 serving, chunked at warp 1.0, on draws 0 to 6. It places 896 of 1050
at 251.1 an hour. Against the same save at warp 1.5 (864), D is +32 (SE 14.2). Against S on the
same draws (890), D is +6 (SE 20.3). At warp 1.0, the p50 xy offset at grasp height is
1.7 mm. It is 2.7 mm at warp 1.5 and 2.3 mm for S.

J05 halves the jaw noise (`--pi05_jaw_noise_abs 0.5`). It places 889 (D -7 against P0, SE 20.6),
so K6c (i) holds. Under RTC at a delay of 6 on the suite, 40 draws, it places 1863 of 2000. The
LoRA 30k with full jaw noise places 1866, so D is -3 (SE 13.8), and K6c (ii) fails. No serving
probe qualifies under K6. The π0.5 serving stays K5 at warp 1.0.

The still probe S05 of ruling K6b never ran. D1 gives it little room: still resamples fire on 255 of
58,499 forward rows.

### 3.3 D3: the LoRA 30k's baselines and read noise

The LoRA 30k read draws 0 to 6 again as `k_B30_s07`: 889 of 1050. Against P0, the same policy and
serving, D is -7 (SE 24.4). This is the read-to-read noise of one π0.5 policy. The server batches
rows, so the bf16 numerics change between reads. A probe thus needs about 24 parts to win by one
SE on draws 0 to 6.

Its P-base `k_B30_pb` places 572 of 600 on the suite. Its R-base `k_B30_rb` places 565, a drop of
1.17 (SE 1.49), a pass. The four probe rows (`k/table_B30_probe.md`):

| Row | LoRA 30k drop | SE | Rule | S drop | SE | Rule |
|---|---|---|---|---|---|---|
| overhead tilt +3 degrees | 0.50 | 1.35 | pass | 3.00 | 1.99 | pass |
| overhead tilt -3 degrees | -0.17 | 1.19 | pass | 19.33 | 3.20 | FAIL |
| overhead forward +5 cm | 0.67 | 1.21 | pass | 42.50 | 2.87 | FAIL |
| overhead lateral +5 cm | -0.50 | 1.31 | pass | 76.50 | 2.19 | FAIL |

The uncorrected overhead pose that limits S does not limit π0.5 at these levels. This left L3
little room on the probe rows.

### 3.4 D4: S on Machine A

S's P-base `k_S_base` places 552 of 600 on the suite. Machine C read 551 on the same draws in MS7j.
S's R-base `k_S_real_base_act` places 546, a drop of 1.00 (SE 1.55), a pass. Every expected row
passes for S (section 7.2). S fails 7 of the 9 report rows.

S's K8 screens read at the K7 serving on Machine A:

| Screen | Read | Correct of 3000 | An hour | Held-out |
|---|---|---|---|---|
| A, seed 20307000, draws 20 to 39 | `k_S_A` | 2537 | 332.2 | 0.800 |
| B, seed 20307600, draws 0 to 19 | `k_S_B` | 2536 | 343.9 | 0.781 |

Against S's delay-free reads of the same draws on Machine B, the two screens read D +11 (SE 28.1) and
D +1 (SE 29.2). These cross hosts, so they are a record only.

### 3.5 D5: the Colab smoke

The smoke ran on the Colab A100 after the port of section 5. It read 3.273 s a step, a 2.34 GiB
checkpoint and a step-200 loss of 0.029. That loss sits inside the LoRA 30k's band near step 30000
(0.021 to 0.034), so `--lora_from` loaded the adapter. A resume check stopped C at its rolling save
1000 and ran the same command again. The run resumed from 1000 at 3.30 s a step.

## 4. Stage 1: the lever probes

Each probe continued the LoRA 30k for 5000 steps (K11). C, L3 and L2 trained on the Colab A100,
and L1 on Machine A.

| Probe | Change | Trained | Step-200 loss |
|---|---|---|---|
| C | none | Colab | 0.026 |
| L1 | frames from 2 s before to 1 s after a close drawn 3 times as often | Machine A, 3.67 s a step | 0.024 |
| L2 | flow time sampler toward low noise (K11a) | Colab, as `k_L2b` | 0.028 |
| L3 | random affine on the overhead image, half of the frames | Colab | 0.026 |

Each probe read draws 0 to 6 at the K5 serving. C, L1 and L3 also read their P-base and the four
probe rows. L2's stopped (section 2.2):

| Read | Correct of 1050 | An hour | D against C | SE | P-base of 600 | Mean probe-row drop | K12 |
|---|---|---|---|---|---|---|---|
| C, `k_C_s07` | 908 | 252.8 | | | 559 | -1.13 | reference |
| L1, `k_L1_s07` | 890 | 247.2 | -18 | 15.0 | 574 | 0.58 | not eligible |
| L2, `k_L2_s07` | 898 | 264.5 | -10 | 22.4 | not read | not read | cannot win |
| L3, `k_L3_s07` | 907 | 255.0 | -1 | 17.2 | 558 | -1.67 | eligible, no win, does not join |
| LoRA 30k, `k_B30_s07` | 889 | 247.4 | -19 | 11.8 | 572 | 0.13 | start point |
| S, `h_S_scr` draws 0 to 6 | 890 | | -18 | 16.7 | 552 | | report only |

L1's mean drop is 1.71 worse than C's (SE 1.09), so L1 is not eligible under K12 (1). C's own P-base
read low, so C's drops read low, and L1's mean drop looks worse beside them. L3's mean drop is 0.54
better than C's (SE 1.51), under one SE, so L3 does not join under K12 (4). No probe beats C on
correct placements by one SE. K12 (3) thus continues C.

C beats the LoRA 30k on the screen (D +19, SE 11.8) but loses on the suite P-base (559 against
572, D -13, SE 8.8). C, L1 and L3 pass every probe row (`k/table_C_probe.md`, `k/table_L1_probe.md`,
`k/table_L3_probe.md`). L2 read none.

## 5. The Colab LoRA path on LeRobot 0.6.1

`lerobot_stack/colab_train_pi05.sh` gained a LoRA mode on LeRobot 0.6.1 (PR #226):

- The install pins torch 2.11.0+cu126, LeRobot 0.6.1, transformers 5.5.4 and peft 0.21.2.
- The A100 image ships torchao 0.10.0, and peft 0.21.2 refuses it at the build of the LoRA layers.
  The install removes it (commits 23535fc and 035dbaa).
- `--prepare` copies the 62 GB corpus from the Shared drive to the local scratch disk, downloads
  `pi05_base` and reads the tokenizer offline from Drive.
- A run keeps a rolling save every 1000 steps on Drive and a kept save every 5000. A rerun of the
  same command resumes from the last rolling save.
- `k/bin/cfetch.sh` fetches a save in 400 MB parts with sha256. The 818 MB adapter took about one
  minute. Machine A merges it in float32 (max abs diff 2.4e-07).

The A100 trains LoRA r128 at batch 32 at 3.27 to 3.30 s a step. The 5090 runs 3.67 s a step (L1).

Lessons from the CLI (version 0.7.4, then 0.7.5 from 10-29 at 05:47):

- A `colab exec` that starts a background job leaves the Jupyter kernel busy, and later calls
  return nothing. The CLI's SSH proxy (`colab ssh --proxy-mode`) reaches the VM without the kernel.
  The campaign used it for every VM command (`k/bin/cx.sh`).
- The proxy allows one connection per runtime. A second one gets HTTP 429.
- The proxy creates a missing session as a CPU runtime with no Drive. `cx.sh` refuses a missing
  session first.
- From systemd user units, ssh needs a key with no passphrase, because the units see only the GPG
  agent.
- The CLI login lapsed about 12.5 h after one owner login, and the runtime died with it. L2 lost its
  first attempt after about 470 steps, before its first rolling save.
- A later session ended after about 12 h with a valid login. The main run lost about 1000 steps
  past its last rolling save.
- The A100 billed 8.88 CU an hour, not the 6.77 of MS7h.

CU: 277.47 on 2026-10-28, 92.09 on 10-29 at 21:05, so 185.38 spent by then. The final balance is
39.80, read on 10-30 at 12:01 with no runtime after it. The campaign spent 237.67 CU. By the
balances and session times in the ledger, the main run billed about 107 CU, under half of its
230 CU cap (K13). The
rest stays unspent, because the spec forbids CU spent only to use them up.

## 6. Stage 2: the main run

The main run `k_main` continued C's 5k save (35k in all) for 20000 steps (K13a). The schedule warms
up for 200 steps to 5e-5 and falls by cosine to 1e-5 at step 20000. Seed 1001. The step-200 loss
was 0.021, inside the band. The run ended at step 10000 (45k in all). Its runtime ended after
kept save 10000, and the K13 plateau stop then ruled out a restart (K13d).

| Save | Steps in all | Correct of 1050 | An hour | D against C (908) | SE | K13 |
|---|---|---|---|---|---|---|
| M5000, `k_M005000_s07` | 40k | 890 | 231.5 | -18 | 15.64 | no stop |
| M10000, `k_M010000_s07` | 45k | 889 | 256.1 | -19 | 16.78 | plateau stop |

Session breaks: three. The session made on 10-29 at about 08:30 ended near step 7000, and the
run resumed from its rolling save 006000. The CLI login lapsed between 00:22 and 00:36 on 10-30,
near step 9000, and the run resumed from 009000 at 07:24. That runtime ended between 09:16 and
11:59 after kept save 10000, cause not known.

Why the run ended: kept saves 5000 and 10000 both read below the start (908). So two saves in a
row did not beat the best (the K13 plateau under K13c). Ruling K13d came before kept save 2's
number: restart the run only if 10000 beats 908. With 39.80 CU, a restart reaches only about step
13800. The two kept saves read 890 and 889, so 5000 more steps gave no gain on draws 0 to 6.
Under K13c, the stage 3 candidate is C, the run's start (908).

## 7. Stage 3: the K8 and K9 reads

The candidate is C. C's K7a timing on an idle 5090 read p50 124.5 ms, p90 146.5 ms and max
166.3 ms over 200 forwards (`k/fwd_timing/k7a_C.result`). The p90 rounds up to 5 steps, so the
delay is 6, the floor of K7a.

### 7.1 K8: accuracy

| Screen | S | Candidate | D | SE | Candidate an hour | S an hour |
|---|---|---|---|---|---|---|
| A, `k_S_A` and `k_C_A` | 2537 | 2571 | +34 | 33.72 | 208.0 | 332.2 |
| B, `k_S_B` and `k_C_B` | 2536 | 2555 | +19 | 29.29 | 225.1 | 343.9 |
| pooled | 5073 | 5126 | +53 | 44.15 | | |

Verdict: K8 passes (`k/bin/k8.py`, `k/k8_C.result`). D is above zero on each screen, and the pooled
D of 53 is above the pooled SE of 44.15. The margin is small: 1.2 pooled SE. C's per-draw
differences run from -11 to +19, and C wins 24 of 40 draws, ties 2 and loses 14.

### 7.2 K9: sim2real

Every row is a Machine A read at 12 draws on the suite scene (`k/k9_C.md`). S's rows are in
`k/table_S_p.md`, `k/table_S_r.md` and `k/table_S_rbase.md`. C's four probe rows are its stage 1
reads (`k/table_C_probe.md`). A drop is in points: 100 times the parts lost, over the 600 parts.
C's P-base `k_C_pb` places 559 of 600. Its R-base `k_C_rb` places 560, a drop of -0.17 (SE
1.19).

| Row | Level | S drop | SE | Rule | C drop | SE | Rule | Smaller drop |
|---|---|---|---|---|---|---|---|---|
| R-base itself | expected | 1.00 | 1.55 | pass | -0.17 | 1.19 | pass | C |
| overhead tilt +1 | expected | -1.50 | 1.28 | pass | -1.00 | 1.71 | pass | S |
| overhead tilt -1 | expected | 0.33 | 1.15 | pass | -1.83 | 1.19 | pass | C |
| overhead forward +1 cm | expected | -1.50 | 1.40 | pass | -2.67 | 0.93 | pass | C |
| overhead forward -1 cm | expected | 0.67 | 1.62 | pass | -1.33 | 1.11 | pass | C |
| overhead lateral +1 cm | expected | 0.17 | 1.19 | pass | -2.67 | 1.42 | pass | C |
| overhead lateral -1 cm | expected | 0.83 | 1.71 | pass | -1.00 | 1.62 | pass | C |
| wrist pitch 2 | expected | -1.00 | 1.11 | pass | -0.50 | 1.23 | pass | S |
| wrist yaw 2 | expected | -0.83 | 0.90 | pass | -2.33 | 1.01 | pass | C |
| field of view overhead +2 | expected | -1.17 | 1.71 | pass | -2.67 | 1.08 | pass | C |
| field of view overhead -2 | expected | -1.50 | 1.08 | pass | -0.83 | 1.14 | pass | S |
| field of view wrist +2 | expected | 0.33 | 1.12 | pass | -2.00 | 0.92 | pass | C |
| field of view wrist -2 | expected | 3.17 | 1.57 | pass | -2.83 | 0.97 | pass | C |
| JPEG 85 | expected | -1.67 | 1.28 | pass | 0.33 | 1.04 | pass | S |
| blur 1.0 | expected | -0.67 | 1.16 | pass | -2.67 | 0.93 | pass | C |
| exposure +1 | expected | 1.17 | 0.83 | pass | -0.33 | 1.15 | pass | C |
| exposure -1 | expected | -1.83 | 1.14 | pass | -2.50 | 1.28 | pass | C |
| hard light | expected | -1.50 | 1.23 | pass | -1.67 | 0.88 | pass | C |
| coupled forward 1 cm | expected | -2.50 | 1.31 | pass | -3.00 | 1.40 | pass | C |
| coupled lateral 1 cm | expected | 0.50 | 1.44 | pass | 0.17 | 1.51 | pass | C |
| varying delay (S 1 ± 1, π0.5 6 ± 3) | expected | -3.17 | 1.64 | pass | 1.33 | 1.19 | pass | S |
| tracking lag 33 ms | expected | -0.50 | 1.67 | pass | 1.00 | 1.57 | pass | S |
| overhead tilt +3 | report | 3.00 | 1.99 | pass | -2.17 | 1.14 | pass | C |
| overhead tilt -3 | report | 19.33 | 3.20 | FAIL | -1.50 | 1.42 | pass | C |
| overhead forward +5 cm | report | 42.50 | 2.87 | FAIL | -0.17 | 1.47 | pass | C |
| overhead lateral +5 cm | report | 76.50 | 2.19 | FAIL | -0.67 | 1.31 | pass | C |
| overhead tilt +6 | report | 74.00 | 2.51 | FAIL | -1.67 | 0.92 | pass | C |
| overhead tilt -6 | report | 85.33 | 1.21 | FAIL | -2.17 | 1.24 | pass | C |
| coupled forward 5 cm | report | 75.50 | 1.67 | FAIL | 58.00 | 2.19 | FAIL | C |
| coupled lateral 5 cm | report | 85.33 | 1.91 | FAIL | 44.67 | 3.18 | FAIL | C |
| tracking lag 100 ms | report | -2.50 | 1.69 | pass | -2.00 | 1.33 | pass | S |

The varying-delay and lag rows pair with R-base. S's pair with S's R-base (546), and C's with its
own R-base (560).

Verdict: K9 passes (`k/bin/k9.py C`, `k/k9_C.result`).

- (a) C passes all 22 expected rows. Its largest drop plus SE is 2.57, on the 33 ms tracking lag.
- (b) C has the smaller drop on 24 rows and S on 7.
- (c) C passes five report rows that S fails: overhead tilt -3, +6 and -6 degrees, and the +5 cm
  forward and lateral shifts. The two coupled 5 cm rows fail for both policies.

Before the score, three checks showed that the camera rows reach C. C's payloads record the same
`overhead_offset` as S's. `MjLegoSortEnv._stress_cameras` moves the rendered camera for both
policies. C's config takes both camera images. In a coupled row, the perceived boxes are off by the
shift (ruling K10a), so no policy can place by them. That is why C fails those rows too.

C's drops lean low. Its P-base read below its own rows: on the 19 small rows against P-base, C's
mean drop is -1.65 points and S's is -0.45. A low base makes every drop against it smaller. Suppose
that C's drops on the P-base rows move up by the 1.20-point gap. Then C still has the smaller drop
on 18 rows and S on 13, so (b) holds. At a 1.5-point shift, the vote is 15 against 13, with 3 ties.
Above that, (b) turns over: 15 against 16 at 1.6. (a) and (c) do not depend on this. Each of the
five (c) rows still passes under any shift below 3.7 points. This check is a record, not a rule.

## 8. The tolerance table

The table gives the largest passing level of each axis, among the levels that the K10 rows read.
Both columns are Machine A reads at 12 draws on the suite scene. The camera, image and light rows read
each policy at its P-base serving with no delay: S chunked at warp 1.5, C chunked at warp 1.0. Only
the last three rows read the real-condition serving. "The largest level read" means that the suite
read no larger level, so the limit is at least that level.

| Axis | S | C |
|---|---|---|
| Overhead tilt, uncorrected | 1 degree (+3 passes by 0.01, -3 fails) | 6 degrees, the largest level read |
| Overhead forward and lateral, uncorrected | 1 cm (+5 cm fails) | +5 cm, the largest level read (-5 cm not read) |
| Coupled shift, forward and lateral | 1 cm (5 cm fails) | 1 cm (5 cm fails: drops 58.00 and 44.67) |
| Wrist pitch and yaw | 2 degrees | 2 degrees |
| Field of view, both cameras | ±2 degrees | ±2 degrees |
| Image | JPEG 85, blur 1.0, exposure ±1 stop | JPEG 85, blur 1.0, exposure ±1 stop |
| Light | hard light on | hard light on |
| Varying delay | 1 ± 1 steps | 6 ± 3 steps |
| Tracking lag | 100 ms | 100 ms |
| Real-condition serving against P-base | pass (drop 1.00, SE 1.55) | pass (drop -0.17, SE 1.19) |

The MS7j suite read S on Machine C at 40 draws. On some axes it read larger levels, for example
wrist 5 degrees and field of view 10 degrees. MS7k did not read those levels for C.

`~/data/maker-arm-ms5/ms7j-tolerance.md` now holds both MS7k columns beside the MS7j table.

## 9. What this means for the real arm

C meets both rules that the owner set, accuracy and sim2real. By those rules, C is the candidate
for the first real test, and S stays ready as the fallback. The owner makes the choice. The facts
that bear on it:

- In the suite, S needs the overhead camera within 1 degree and 1 cm of the simulator's camera,
  unless the real warp corrects the image (MS7j report). C passes the uncorrected camera up to 6
  degrees of tilt and a +5 cm forward or lateral shift, the largest levels read. No real-arm read
  tests either limit.
- The perception calibration still binds for both. A 5 cm camera error that also moves the
  perceived boxes fails for both policies. The suite read no level between 1 cm and 5 cm.
- C's delay rows (R-base, the varying delay 6 ± 3 and both lags) read at an inference delay of 6
  control steps under RTC. Its camera, image and light rows read on P-base, chunked with no delay.
  So no row reads a camera error and the delay together.
- The delay of 6 is K7a's floor. The forward time on an idle 5090 (p90 146.5 ms) rounds up to 5
  steps. A slower policy host makes the delay longer. In MS7j, the LoRA 30k under RTC failed a
  delay of 12 steps by 0.42 points.
- C places about 36 % fewer correct parts an hour than S. A real test with C takes longer for the
  same number of parts.
- The accuracy margin in the simulator is small (1.2 pooled SE). No read predicts the gap on the
  real arm. In the suite, the camera tolerance is the larger difference.

Open leads:

- The RTC cost itself. The read-only diagnostic D1b (`k/d1b/`) found that RTC at a delay of 6
  loses 27 correct against chunked warp 1.0 on the suite (SE 17.2). Most of the loss comes from
  in-air closes that come down onto the part. J05 did not reduce it. The campaign did not separate
  the guidance from the forward rate.
- The read-to-read noise of one π0.5 policy is about 24 parts on draws 0 to 6. A lever smaller
  than that needs more draws to show.
- More LoRA steps on the V corpus of MS7h gave nothing past C's 35k on draws 0 to 6 (two kept
  saves), and the levers L1 to L3 gave no win. DAgger already failed to lift π0.5 in MS7h and
  MS7i. RECAP, outside this week's spec, stays a possible later lever (spec section 2.5).
- MS7k did not read C at a delay of 12 steps, at a -5 cm shift, or with a camera error under RTC.
  No read covers a coupled shift between 1 cm and 5 cm.

## Appendix A. Commands

```
python3 ~/data/maker-arm-ms7/k/bin/pair.py TAG_A TAG_B [--draws 0-6]   # K3 paired score
python3 ~/data/maker-arm-ms7/k/bin/k12.py                               # K12 selection
python3 ~/data/maker-arm-ms7/k/bin/k8.py k_C                            # K8 against k_S_A, k_S_B
python3 ~/data/maker-arm-ms7/k/bin/k9.py C --out k9_C.md                 # K9 against S's rows
P=C CK=~/data/maker-arm-ms7/k/lora/C_5000_n25 bash ~/data/maker-arm-ms7/k/bin/final_chain.sh
bash ~/data/maker-arm-ms7/k/bin/s_table.sh k_S                          # a suite table
```

## Appendix B. Tags

- Stage 0: `k_P0`, `k_P0_j05`, `k_J05_rtc_d6`, `k_B30_s07`, `k_B30_pb`, `k_B30_rb`, `k_B30_t3`,
  `k_B30_tm3`, `k_B30_f5`, `k_B30_l5`, `k_S_base`, `k_S_real_base_act`, `k_S_<row>`, `k_S_A`,
  `k_S_B`.
- Stage 1: `k_<P>_s07`, `k_<P>_pb`, `k_<P>_t3`, `k_<P>_tm3`, `k_<P>_f5`, `k_<P>_l5` for P in C, L1,
  L3, and `k_L2_s07`. The stopped read is `k_L2_pb.stopped2148`.
- Stage 2: `k_M<step>_s07`.
- Stage 3: `k_<P>_A`, `k_<P>_B`, `k_<P>_rb` and `k_<P>_<row>`.

## Appendix C. Pull requests

| PR | What it adds |
|---|---|
| #224 | The sim2real knobs: `--infer_delay_jitter`, the tracking lag, the coupled overhead shift, and the new suite rows. |
| #225 | The π0.5 levers: `--lora_from`, `--close_weight`, `--siglip_lora`, `--overhead_aug`. |
| #226 | The Colab LoRA mode of `colab_train_pi05.sh` on LeRobot 0.6.1, on top of #225. |

PRs #225 and #226 form GitHub stack #227. Every new knob is off by default (K14).
