# MS7j: real-arm readiness

Date: 2026-10-23.

MS7j used the idle GPU hosts of the week of 2026-10-20 to prepare for the real Maker Arm. The owner
approved the design on 2026-10-20. This report cites it as "spec section N". The plan of the same
day is "plan Task N". Both live outside the repository. The campaign ledger is
`~/data/maker-arm-ms7/j/README.md`, and `STATE.md` beside it is the resume pointer. The ledger
names its rules J1 to J11 and its rulings R1 to R49. Every rule went into the ledger before its
number. Each read has a TAG, its name. The payloads of a read are under
`~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7j TAG starts with `j_`.

The names:

- S is the final MS7g policy: ACT `f/train_F5` 200k with `--carry_guard --overhang_guard`.
  The suite serves it without the image classifier and the vote (spec section 6.1).
- The fresh 30k is the π0.5 full fine-tune of MS7h, `h/vla/fresh/30000_n25`, served at HD63's
  configuration (MS7h ledger HD63): 25 actions a forward, warp 1.5, and the a0j1slR25gm20
  switches.
- The LoRA is the π0.5 run of Track B (section 5.2). "The 30k save" is its save at step 30000.
- Warp X plays a chunk X times faster. The served chunk is resampled at times 0, X, 2X and so on.

## 1. Summary

S is the candidate for the first real test. On the readiness suite, every row at the level that
the rig expects passes the tolerance rule, so no retrain ran (section 4).

The suite gives the calibration and mount limits for MS5:

- If the overhead camera is not corrected, its tilt must stay within 1 degree, and its forward
  and lateral shift within 1 cm. A tilt of 3 degrees fails, and a 5 cm shift loses 46 to 75 points.
- If the real warp corrects the overhead camera, every mount row passes, up to 6 degrees of tilt,
  4 cm of height and 5 cm of forward and lateral shift.
- The wrist camera passes up to 5 degrees and 1 cm uncorrected. The field of view passes up to
  10 degrees on both cameras. An observation delay of 6 steps, an inference delay of 1 step, and
  every image and light row pass.

Track B trained a π0.5 LoRA on the RTX 5090 with LeRobot 0.6.1 in 30 h 46 min. The 30k save equals
the full fine-tune on correct placements and is faster. On the 20-heap screen, it places 2500 of
3000 against 2495 for the fresh 30k, at 280.0 against 230.8 an hour. S places 2533 at 369.5 an
hour, so S stays the faster policy by about 90 correct placements an hour.

Chunked execution does not survive the 5090's inference delay, but real-time chunking (RTC) does. A
control step is 1/30 s. So 6 steps is about one π0.5 forward on the 5090 (172 to 207 ms). 12 steps
is a guess for the Jetson Thor. On the suite, the fresh 30k places 1752 of 2000 chunked with no
delay. It places 544 at a delay of 6 steps and 903 at 12 steps. The LoRA loses less, but it also
fails at both delays (section 5.7). With RTC, the fresh 30k places 1822 at a delay of 6 and 1853 at
12, against 1849 chunked with no delay. Both RTC rows pass (section 5.6). The LoRA's RTC rows pass
at a delay of 6 and fail at 12 by 0.42 points (section 5.7). A real test of π0.5 thus needs RTC.

No Colab job ran. Neither trigger of spec section 6.3 held.

## 2. What was built

Six draft PRs form GitHub stack #222. Every new knob is off by default.

| PR | What it adds |
|---|---|
| #216 | `SERVED_SWITCHES` in `scripts/real_eval_heap.py`: S's loop rules on the real rig, without the image classifier. A test pins the list to the suite's switch list. |
| #217 | The π0.5 scripts on LeRobot 0.6.1 (`lerobot_stack/.venv-pi`), the LoRA recipe and `merge_lora.py`. |
| #218 | The stress knobs of the MuJoCo heap. Camera offsets, the warp back through `real/camera_model.py` and field-of-view deltas. JPEG, blur, exposure and a hard key light. An observation delay, and the sparse scene (`--sparse_gap`). |
| #219 | An emulated inference delay of D control steps in both policy servers (`DelayLine`). |
| #220 | `scripts/readiness_suite.py`: the suite rows and the tolerance table. |
| #221 | Real-time chunking (RTC) in the π0.5 server under the delay (`--pi05_rtc`). |

The inference delay works as spec section 6.1 says. If the queue of a row holds D actions or fewer,
the row asks for its next chunk. The arm plays the rest of the old chunk while it waits. The new
chunk arrives D steps later, without its first D actions. A row with an empty queue holds its last
target. The delay counts control steps, so a read stays deterministic.

Under `--pi05_rtc`, real-time chunking (RTC) guides each new chunk. Its target is the old chunk's
actions that the arm still plays while the new chunk is on its way. If the queue of a row holds 25
actions or fewer, the row asks for its next chunk. At the campaign's 25 actions a forward, that is
at every chunk arrival. So an RTC row plays only D actions of each chunk, and it forwards every D
steps. A chunked row at warp 1.0 with no delay forwards every 25 steps. An RTC row thus differs from
its chunked pair in two ways: the guidance, and a forward three to four times as often.

The final review found a defect in the first RTC code (77c768f). It filled the guidance target
past the old chunk's real actions with copies of its last action, and it guided 25 rows. At a
delay of 12, the rows that play were pulled toward those copies. The fix (ecfbbff) guides only
the real rows: 19 at a delay of 6 and 13 at a delay of 12. Every RTC row in the tables of
sections 5.6 and 5.7 reads on the fix.

## 3. The rules

- J1: the suite reads seed 20390000, the second scene 20391000, and the screen 20307000 (draws 0
  to 19).
- J3: S's baseline must place at least 0.80 of its parts in the correct box.
- J4: N draws a row, fixed from a probe, between 12 and 40. N is 40.
- J5: the drop is the baseline's correct-placement rate minus the row's, paired by draw, in
  points. If the drop plus one paired SE is 5 points or less, the row passes.
- J6: A2 trains on every expected-level row that fails J5 for S.
- J8: parity of LeRobot 0.5.1 and 0.6.1 on the fresh 30k (section 5.1).
- J9: the best LoRA save has the most correct placements on draws 0 to 6.
- J10: the π0.5 delay rows, chunked first, then RTC.
- J11: Track B pauses while the owner runs the real arm with a policy on the 5090. No pause came.

## 4. Track A: S on the suite

### 4.1 The scene and the baseline

The suite scene is the sparse layout of the MS5 sensing plan. Ten parts lie in the tray at least
35 mm apart, with four desk boxes and the four base classes. Each read runs 40 draws of 5 heaps on
Machine C. Rows with the warp back run 8 lanes, because 12 lanes ran the A2000 out of memory
(ledger 2026-10-21 02:54).

The baseline `j_S_base_n40` places 1826 of 2000 parts in the correct box (0.913). It lifts 1960,
puts 27 in a wrong box and never lifts 40. It takes 7.44 sim seconds a correct placement. J3
holds: the first 12 draws alone read 0.918.

### 4.2 The table

The full table is `~/data/maker-arm-ms7/j/table_S.md`: 57 rows, 40 draws each. The paired SE of a
row is 0.65 to 2.0 points. Eight rows fail J5, and all of them are uncorrected overhead poses:

| Row | Correct of 2000 | Drop | SE | Never lifted |
|---|---|---|---|---|
| overhead tilt 3 degrees | 1725 | 5.05 | 1.01 | 163 |
| overhead tilt -3 degrees | 1493 | 16.65 | 1.68 | 396 |
| overhead tilt 6 degrees | 374 | 72.60 | 0.98 | 1602 |
| overhead tilt -6 degrees | 141 | 84.25 | 0.92 | 1845 |
| overhead forward 5 cm | 907 | 45.95 | 1.97 | 1038 |
| overhead forward -5 cm | 530 | 64.80 | 1.53 | 1441 |
| overhead lateral 5 cm | 332 | 74.70 | 1.21 | 1640 |
| overhead lateral -5 cm | 352 | 73.70 | 1.26 | 1613 |

A failed pose row loses its parts before the lift. The wrong-box counts stay at 11 to 33, as in
the baseline. A height change of 5 cm passes in both directions.

Every other row passes. The largest passing level of each axis:

| Axis | Largest passing level |
|---|---|
| Observation delay | 6 steps |
| Inference delay (ACT) | 1 step |
| Overhead pose, uncorrected | tilt 1 degree, forward and lateral 1 cm, height 5 cm |
| Wrist pose, uncorrected | pitch and yaw 5 degrees, forward and lateral 1 cm |
| Mount offset, corrected | height 4 cm, forward and lateral 5 cm, tilt 6 degrees, wrist 1 cm |
| Field of view | 10 degrees narrower or wider, both cameras |
| Image | JPEG 85, blur 1.0, exposure ±1 stop |
| Light | hard light on |

Many rows read a little better than the baseline, for example the wrist mount row at -2.95 (SE
0.75). Ruling R41 tested whether the warp itself helps. Two control rows run the warp back with a
0.1 mm offset. They read -0.20 (SE 0.77) for the overhead camera and -0.15 (SE 1.07) for the
wrist camera, so the warp is not the cause. Every row pairs with the same baseline, so the
baseline's own draw noise moves all rows together. Against the overhead control, the 13 mount
rows read -2.00 to +0.05, each inside 2.6 SE.

### 4.3 A1, A2 and the candidate

A1 found no failed row at the expected levels of spec section 6.2 (`j/table_S_A1.md`, 27 rows).
The largest drop plus SE among them is the wrist field of view at 2 degrees: 1.61 (drop 0.65, SE
0.96). So A2 trained nothing (J6), plan Tasks 19 and 20 did not run, and S is the first real-test
candidate (J7).

The second scene has 5 parts with no gap rule (J6, report only). `j_S2_base` places 884 of 1000
(0.884) at 1.188 cycles a stowed part and 471.3 correct placements an hour.

## 5. Track B: π0.5 with LoRA

### 5.1 Parity of LeRobot 0.5.1 and 0.6.1

J8 failed narrowly in both parts. The test fed the fresh 30k the same noise and observations on
0.5.1 and on 0.6.1. The chunks agree within the bar on five joint columns and the jaw. Column 5
reads a p99 of 0.032 rad against the bar of 0.01 rad. A second 0.6.1 run with the start noise scaled
by one bf16 step reads a column-5 p99 of 0.0063 rad. The version gap is thus 1.8 to 5.0 times one
bf16 step of rounding, per column. The embedding-scale trap of spec section 3.3 moves whole chunks,
and this gap does not.

The 0.6.1 read `j_Vf30k_061` on draws 0 to 6 places 867 of 1050, against HD63's 885 on the same
host and argv. D is -18 with a paired SE of 17.5, 1.03 SE. The owner then ruled that 0.6.1 counts
as equal to 0.5.1 and started the LoRA run (2026-10-21 03:55).

### 5.2 The LoRA run

The recipe of spec section 6.3, ledger 2026-10-21 03:55:

- Base `pi05_base` at revision a538eb27 with the tokenizer patch to 128 tokens. Corpus: the V
  corpus of MS7h.
- LoRA rank 128 and alpha 128 on 252 linears, 126 in the language model and 126 in the action
  expert. No SigLIP linear is a target, and SigLIP stays frozen.
- The action and time projections train in full. Masked loss, bf16, gradient checkpointing.
- Batch 32, peak learning rate 1e-4, warmup 1000, cosine to 1e-5 at step 30000, seed 1000.

The memory probe peaked at 22.1 GB, so batch 32 stood. The run took 30 h 46 min at 3.69 s a step.
A save merges into a full checkpoint for serving. The merge runs in float32, because a bf16 merge
misses the 1e-3 bar at 0.0082 (ruling R37).

### 5.3 The saves on draws 0 to 6

Each save read HD63's argv on Machine B, seed 20307000, draws 0 to 6, 1050 parts. The pair is the
fresh 30k on the same host and version, `j_Vf30k_061`, 867 correct (`j/b4/`).

| Save | Correct | An hour | Held-out | D | SE |
|---|---|---|---|---|---|
| 5k | 629 | 95.1 | 0.546 | -238 | 21.2 |
| 10k | 635 | 113.1 | 0.623 | -232 | 28.3 |
| 15k | 824 | 185.8 | 0.763 | -43 | 19.7 |
| 20k | 862 | 249.9 | 0.775 | -5 | 32.7 |
| 25k | 868 | 250.8 | 0.843 | +1 | 15.8 |
| 30k | 886 | 299.5 | 0.787 | +19 | 17.9 |

The 30k save is the best (J9). Its rate is 55.7 an hour above the fresh 30k (4.38 SE). The LoRA
passed the full fine-tune between 20k and 25k.

Colab trigger 1 did not hold. The 30k save beats the 20k save by D +24 with a paired SE of 25.5,
under one SE. The 5090 also trains 10k more steps in about 10.3 h, well before the due date of
10-30. So the run ended at 30k, as the spec says.

### 5.4 The 20-heap screen

The screen reads 150-part heaps, seed 20307000, draws 0 to 19, 3000 parts, on Machine A.

| Read | Correct | An hour | Held-out | Re-open after a close | Off-target first close |
|---|---|---|---|---|---|
| LoRA 30k, `j_B3_30000_scr` | 2500 | 280.0 | 0.784 | 0.049 | 0.122 |
| LoRA 25k, `j_B3_25000_scr` | 2435 | 254.9 | 0.768 | 0.049 | 0.123 |
| fresh 30k, `h_Vf30k_g20_scr` | 2495 | 230.8 | 0.780 | 0.114 | 0.126 |
| S, `h_S_scr` | 2533 | 369.5 | 0.781 | 0.022 | 0.068 |

Paired by draw, the 30k save against the fresh 30k is D +5 (SE 34.5) and +49.1 an hour (3.52 SE).
Against S, it is D -33 (SE 33.3, 0.99 SE) and -89.5 an hour (8.62 SE). The LoRA re-opens the jaw
after its own close less than half as often as the full fine-tune. Its first close still misses
the target at almost twice S's share. This is the gap that MS7h found.

The fresh 30k's screen ran on LeRobot 0.5.1 in MS7h, and the LoRA's on 0.6.1. Section 5.1 sizes
that gap at about one SE on draws 0 to 6.

### 5.5 The delay rows, chunked

Each candidate reads its delay rows on one host: the fresh 30k on Machine B (ruling R43), and the
LoRA on Machine A (ruling R46). The rows use the suite scene, seed 20390000 and 40 draws, with
HD63's serving. The table tool pairs each delay row with its candidate's own baseline.

The fresh 30k at warp 1.5 (`j/table_B5_Vf30k_chunked.md`):

| Row | Correct of 2000 | Drop | SE | Never lifted | Sim s a correct |
|---|---|---|---|---|---|
| no delay, `j_B5_Vf30k_base` | 1752 | | | 149 | 16.93 |
| delay 6, `j_B5_Vf30k_d6` | 544 | 60.40 | 1.37 | 1410 | 110.44 |
| delay 12, `j_B5_Vf30k_d12` | 903 | 42.45 | 1.33 | 923 | 55.86 |

Both rows fail on all 40 draws. A delay of 12 steps reads better than a delay of 6. The cycle
census (`j/b5_Vf30k_census.txt`) shows two different failures:

- At delay 6, 1286 of 2600 cycles never close the jaw within their 800-step budget (no delay: 34
  of 2335). The jaw is open in 94 % of the steps (no delay: 72 %) and changes state 4.2 times a
  1000 steps (no delay: 13.6).
- At delay 12, only 43 cycles never close. But 1120 cycles close and miss (no delay: 248), and the
  jaw changes state 22.6 times a 1000 steps.

The delay line explains part of this. At warp 1.5, a chunk of 25 actions becomes 17 served
actions. At delay 6, the arm plays actions 6 to 16 of each chunk without a pause. It thus never
plays the first 9 policy steps of any chunk. At delay 12, the arm plays actions 12 to 16 and then
holds the last one for 7 steps. It stops 7 of every 12 steps and plays only the far end of each
chunk. A hypothesis, not yet tested: the policy plans its close inside the first 9 steps of a
chunk. Delay 6 always drops that part, and delay 12 serves the far end, where the jaw is already
closed.

The chunked baseline at warp 1.0 is `j_B5_Vf30k_w1_base`. It places 1849 of 2000 at 1.233 cycles
a stowed part, at the same 212.7 an hour as warp 1.5. The RTC rows need warp 1.0, so they pair with
this row.

The fresh 30k's baseline is not comparable with S's. S read on Machine C, and the π0.5 rows on
Machine B, so the physics differ (spec section 5, anti-goals).

### 5.6 The RTC rows

The fresh 30k with RTC at warp 1.0, paired with the chunked `j_B5_Vf30k_w1_base`
(`j/table_B5_Vf30k_rtc.md`):

| Row | Correct of 2000 | Drop | SE | Never lifted | Sim s a correct |
|---|---|---|---|---|---|
| no delay, chunked, `j_B5_Vf30k_w1_base` | 1849 | | | 106 | 16.93 |
| delay 6, RTC, `j_B5_Vf30k_rtc_d6` | 1822 | 1.35 | 0.99 | 65 | 14.65 |
| delay 12, RTC, `j_B5_Vf30k_rtc_d12` | 1853 | -0.20 | 0.75 | 80 | 14.35 |

Both RTC rows pass, so the largest passing RTC delay is 12 steps. Each places about as many parts
as its chunked pair with no delay, and each is faster. The census (`j/b5_Vf30k_rtc_census.txt`)
shows the jaw at work. At a delay of 12, 3 cycles never close (no delay: 53) and 126 close and
miss (no delay: 159). The jaw changes state 18.6 times a 1000 steps (no delay: 9.4). At a delay
of 6, 78 parts land in a wrong box (no delay: 30). There, 85 cycles end with a part still in the
jaw (no delay: 10).

The row on the code before the fix is a record only (ruling R49). At a delay of 6, it placed 1811
(`j/table_B5_Vf30k_rtc_prefix.md`). The fixed row places 11 more, with a paired SE of 20.9 parts.
The fix thus did not move the read at a delay of 6 by a measurable amount.

An RTC row forwards every D steps (section 2). A gain over the chunked rows can thus come from
the guidance or from the more frequent forwards. The suite does not separate the two.

### 5.7 The LoRA's delay rows

The 30k save at warp 1.5 on Machine A (`j/table_B5_B3_30000_chunked.md`):

| Row | Correct of 2000 | Drop | SE | Never lifted | Sim s a correct |
|---|---|---|---|---|---|
| no delay, `j_B5_B3_30000_base` | 1821 | | | 89 | 12.39 |
| delay 6, `j_B5_B3_30000_d6` | 1426 | 19.75 | 1.08 | 496 | 28.39 |
| delay 12, `j_B5_B3_30000_d12` | 1180 | 32.05 | 1.35 | 583 | 34.46 |

Both rows fail on all 40 draws. The LoRA loses less than the fresh 30k at each delay, and its
loss grows with the delay. The census (`j/b5_B3_30000_census.txt`) shows that the LoRA keeps
closing its jaw under the delay. At delay 6, 265 of 2567 cycles never close, against 1286 of 2600
for the fresh 30k. At delay 12, 798 cycles close and miss (no delay: 201). The two candidates
read on two hosts, so this compares their failure modes, not their numbers.

The LoRA's RTC rows pair with its chunked row at warp 1.0, `j_B5_B3_30000_w1_base`. That row
places 1893 of 2000 at 295.5 an hour (`j/table_B5_B3_30000_rtc.md`):

| Row | Correct of 2000 | Drop | SE | Never lifted | Sim s a correct |
|---|---|---|---|---|---|
| no delay, chunked, `j_B5_B3_30000_w1_base` | 1893 | | | 49 | 12.18 |
| delay 6, RTC, `j_B5_B3_30000_rtc_d6` | 1866 | 1.35 | 0.86 | 89 | 14.28 |
| delay 12, RTC, `j_B5_B3_30000_rtc_d12` | 1802 | 4.55 | 0.87 | 113 | 15.33 |

The LoRA's RTC row at a delay of 6 passes, and the row at 12 fails by 0.42 points. So the
largest passing RTC delay of the LoRA is 6 steps. Both RTC rows are slower than their chunked
pair, unlike the fresh 30k's. The census (`j/b5_B3_30000_rtc_census.txt`) shows more cycles that
hold a part and do not lift it: 272 at a delay of 6 and 354 at 12 (no delay: 142).

## 6. The limits

- The pose rows move only the camera images. The perceived state, which the loop builds from the
  truth with noise, does not move with them. On the real rig, a calibration error also moves the
  perceived target, so the real loss can be larger.
- The first chunk after a reset is not delayed. The delay line takes it at once and whole.
- The delay is a fixed count of control steps. A real delay varies from chunk to chunk.
- The simulator has no tracking controller. The real arm lags its command under admittance.
- The LoRA's pairs with the fresh 30k cross LeRobot versions on the screen (section 5.4).
- Each π0.5 table stands on its own host. No π0.5 row compares with an S row.

## 7. The open leads

- Test where the close sits in a π0.5 chunk near a grasp, offline. That tests the hypothesis of
  section 5.5.
- Separate the RTC guidance from the forward rate. A chunked row that forwards every D steps,
  with no guidance, gives that split (section 5.6).
- The real loop needs asynchronous execution for RTC on the arm (a non-goal of MS7j, spec
  section 5).
- Measure one π0.5 forward on the Jetson Thor. The delay of 12 steps is a guess.
- The LoRA's first close misses at 0.122 against S's 0.068 on the screen. A LoRA run past 30k
  is cheap on the 5090 (about 10.3 h for 10k steps), but trigger 1 found no gain to buy.

## 8. Rulings that changed a read

- R27: the suite reads with no frame jitter (`--no-frame_jitter`), because the overhead camera
  knobs refuse it.
- R37: the LoRA merges in float32.
- R41: two warp-only control rows (section 4.2).
- R42: after the overhead tilt rows, the suite read the expected-level rows first.
- R43: the fresh 30k's delay rows ran on Machine B, so Machine B was not idle during B3.
- R44 and R45: Machine A read the 25k and 30k screens before J9 picked the save.
- R46: Machine A chain runs as a systemd user unit, after a session end killed the first 30k
  screen at 15:11 on 10-22.
- R47: one fix wave after the final review, before the RTC rows at a delay of 12.
- R49: the first RTC row at a delay of 6 read on the code before the fix. It is kept as
  `j_B5_Vf30k_rtc_d6.prefix`, and the row reads again on the fix.
