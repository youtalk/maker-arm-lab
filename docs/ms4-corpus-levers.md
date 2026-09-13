# MS4 — corpus and schema levers 1 to 3

Milestone: MS4, levers 1 to 3 of the post-gate roadmap. Spec: `2026-09-11-post-gate-roadmap-design.md`, section 6. Plan: `2026-09-12-ms4-corpus-levers-plan.md`, Task 18. Executed 2026-09-12 to 2026-09-13 on the RTX 5090 host (MuJoCo 3.13, LeRobot 0.5.1), from maker-arm-lab `main` at 5cccc7d. Every number below is read off a file under `~/data/maker-arm-ms4/` on that host. The running record is `~/data/maker-arm-ms4/README.md`, and this report was assembled from it.

## 1. Summary

**Lever 1, the camera-pose context, is adopted. Lever 2, delta joint actions, is rejected. Lever 3, the privileged-state ceiling, did not act as a ceiling.** One corpus of 1039 success-only source episodes fed four datasets, and each dataset trained one plain ACT policy for 600k steps. Every run went through rule L3, the check step, and (except the ceiling) one official gate on 120 episodes and one generalization read on 60 episodes at `--spatial_scale 3.0`. The read rules were written into the ledger before the first lane.

| run | gated rung | official gate, seed 20261027 | generalization, scale 3.0, seed 20261077 | read |
| --- | --- | --- | --- | --- |
| base | 375k | 585/720 = **0.8125** [0.782, 0.839] | 216/360 = **0.600** [0.549, 0.649] | reference |
| ctx | 425k | 600/720 = **0.833** [0.804, 0.859] | 260/360 = **0.722** [0.674, 0.766] | **lever 1 adopted** |
| delta | 375k | 58/720 = **0.081** [0.063, 0.103] | 31/360 = **0.086** [0.061, 0.120] | **lever 2 rejected** |
| ceiling | 450k | never gated; check step 8/360 = **0.022** on the selection seed | not read | **not a valid ceiling** |

Brackets are Wilson 95 % intervals. The gate criterion is 0.90 per brick. **Every gate read is under 0.90**, so R-GATE leaves each of them to the owner.

The camera context passes both conditions of its rule. Its gate is at least the base, and its generalization read gains 0.122 over the base, where the rule asks for 0.10. The gate difference alone sits inside both intervals. The generalization read carries the decision, as the rule intends. At three times the training spread, the context policy keeps its close accuracy (69.5 % of closes within 2 cm, base 60.6 %) and sorts all six bricks in 18 of 60 episodes (base 11).

The delta policy collapses. The wiring was checked end to end and is correct. A traced comparison shows that its command stays within 5 mrad of the measured pose on 82 % of steps (base 34 %) and that the arm moves at about a quarter of the base speed (section 6.3). The collapse belongs to the delta definition of plan Task 14, target minus the measured joint position of the same frame.

The ceiling policy, trained on the six brick poses without images, stows 0.000 to 0.028 on every rung. The pre-registered rule reads a ceiling under 0.90 as a limit of the demonstrations or the control law. The image runs reach 0.875 and 0.881 with the same demonstrations and control law, and the ceiling's training loss is 2.6 times the base's, so the rule's premise fails (section 7).

The base reads 0.8125 against MS3 t4's 0.840. The two corpora differ (per-episode draws, and t4 merged t3's lanes), and the gates ran on different seeds, so this comparison is reported only, as pre-registered.

## 2. What was built

Six PRs landed as GitHub stack #39 before the campaign: #33 per-env bins in the grader and the eval loop, #34 per-episode scene draws and per-frame camera jitter in the MuJoCo env, #35 context channels in the shard, the converter, the wire and the server, #36 delta actions in the converter, the trainer and the server, #37 the Isaac freeze documented, and #38 `--keep_shards` in the campaign supervisor. The Isaac backend stays at its MS3 feature set.

## 3. The campaign

### 3.1 Isolation and preflight

Every stage ran from a detached worktree `~/src/maker-arm-lab-ms4` at 5cccc7d, with symlinked venvs and `PYTHONPATH` on the worktree's `source`. One chain driver, `run_chain.sh`, ran every stage in order and skips a finished stage on a relaunch.

A preflight ran every stage at toy size first: two lanes of target 3, the three extra conversions, four merges, the video checks, four 100-step training smokes, and one eval draw per run plus one scale-3.0 draw. It passed in 24 minutes. It measured two numbers that the plan had not: an extra conversion takes about 7 minutes per shard, and a shard takes 1.5 to 1.9 GB per 12-attempt run.

### 3.2 Corpus (`c1/`)

Ten lanes ran the t4 lane recipe plus `--episode_draws --frame_jitter --keep_shards`, target 100 each, lane seed `20261067 + 100 × lane`, 12 envs × 12 attempts per run, from 19:08 to 20:14 PDT.

| lane | seed | runs | exported/attempts | rate | time_out | watchdog | abandons | clamp | hover_lowered | s/attempt |
|---|---|---|---|---|---|---|---|---|---|---|
| lane_0 | 3081197816 | 10 | 110/120 | 0.917 | 0 | 0 | 10 | 2.0e-04 | 180 | 17 |
| lane_1 | 3081297816 | 9 | 101/108 | 0.935 | 0 | 0 | 7 | 5.3e-05 | 183 | 18 |
| lane_2 | 3081397816 | 9 | 106/108 | 0.981 | 0 | 0 | 2 | 1.1e-04 | 124 | 17 |
| lane_3 | 3081497816 | 9 | 100/108 | 0.926 | 0 | 0 | 8 | 2.7e-05 | 142 | 17 |
| lane_4 | 3081597816 | 9 | 105/108 | 0.972 | 0 | 0 | 3 | 5.5e-05 | 142 | 17 |
| lane_5 | 3081697816 | 10 | 110/120 | 0.917 | 0 | 0 | 10 | 0.0e+00 | 189 | 16 |
| lane_6 | 3081797816 | 9 | 102/108 | 0.944 | 0 | 0 | 6 | 5.4e-05 | 137 | 17 |
| lane_7 | 3081897816 | 9 | 103/108 | 0.954 | 0 | 0 | 5 | 5.3e-05 | 169 | 17 |
| lane_8 | 3081997816 | 9 | 100/108 | 0.926 | 0 | 0 | 8 | 2.7e-04 | 138 | 17 |
| lane_9 | 3082097816 | 9 | 102/108 | 0.944 | 0 | 0 | 6 | 2.7e-05 | 126 | 17 |
| **all** | | 92 | **1039/1104** | **0.941** [0.926, 0.954] | 0 | 0 | 65 | 8.4e-05 | 1530 | 17 |

The yield is under t4's 0.977. The abandons are 41 `no_reachable_brick` and 24 `grasp_retries_exhausted`. t4's abandons were all retries. The unreachable abandons are new under per-episode draws and are recorded, not explained. The median commanded-acceleration p90 over the 92 runs is 2.45 rad/s², and the median near-static fraction is 0.308.

### 3.3 Four datasets from the same shards

The supervisor converted each run into `dataset` (images, absolute actions) as the lane ran. After the lanes, `convert_extra.sh` converted every lane's kept shards three more times, five lanes at a time: `dataset_ctx` (`--context camera`), `dataset_delta` (`--action_space delta`), and `dataset_ceiling` (`--context privileged --no_images`). That took 3 h 8 min, against the plan's 1 h. In every lane the four families hold identical episode and frame counts.

| family | images | `observation.environment_state` | action space | videos |
| --- | --- | --- | --- | --- |
| dataset | wrist, overhead | none | absolute | 184 |
| dataset_ctx | wrist, overhead | (6,) overhead camera pose | absolute | 55 |
| dataset_delta | wrist, overhead | none | delta | 55 |
| dataset_ceiling | none | (42,) six brick poses | absolute | 0 |

Each merge reads `MERGE_OK lanes=10 episodes=6234 frames=3010848 runs=92`. With per-episode draws, every source episode has its own bin, light and camera layout, so the corpus is no longer counted in run layouts as t4 was. The merges pass `video_files_size_in_mb=1`, and the delta root gets its `meta/action_space.json` from the merge script. Every video of the three image families decodes with `ffmpeg -v error`. The extra conversions write each lane in one pass and pack more frames per file, so their 55 files hold the same 9.2 GB as the base's 184. `/home` held 282 GB free after the merges, so the pre-registered disk guard kept the shards.

## 4. Training and selection

Each run trained alone on the GPU, 600k steps, checkpoints every 25k, the MS3 protocol unchanged: base 3 h 41 min, ctx 3 h 44 min, delta 3 h 38 min, ceiling 2 h 8 min without images. Every run printed TRAIN_OK. Final training loss: base 0.068, ctx 0.071, delta 0.142, ceiling 0.176.

Rule L3 read rungs 250k to 600k on draws 0 to 2 of seed 20261017, and the check step read the top two rungs on draws 0 to 4. Every eval used `--temporal_ensemble_coeff 0.01 --n_action_steps 1 --episode_draws --frame_jitter`, ctx added `--context camera`, and ceiling added `--context privileged`.

| rung | base | ctx | delta | ceiling |
| --- | --- | --- | --- | --- |
| 250k | 0.759 | 0.718 | 0.056 | 0.023 |
| 275k | 0.745 | 0.759 | 0.019 | 0.023 |
| 300k | 0.806 | 0.579 | 0.060 | 0.019 |
| 325k | 0.542 | **0.903** L3 | 0.023 | 0.000 |
| 350k | 0.407 | 0.569 | 0.019 | 0.014 |
| 375k | **0.903** L3 | 0.796 | **0.134** L3, isolated spike | 0.019 |
| 400k | 0.866 | 0.759 | 0.019 | 0.009 |
| 425k | 0.833 | 0.884 | 0.083 | 0.009 |
| 450k | 0.843 | 0.847 | 0.056 | **0.028** L3 |
| 475k | 0.838 | 0.778 | 0.069 | 0.014 |
| 500k | 0.722 | 0.880 | 0.056 | 0.023 |
| 525k | 0.847 | 0.847 | 0.056 | 0.005 |
| 550k | 0.852 | 0.880 | 0.074 | 0.005 |
| 575k | 0.866 | 0.838 | 0.028 | 0.009 |
| 600k | 0.856 | 0.838 | 0.023 | 0.009 |

| check step, draws 0 to 4 | gated | other |
| --- | --- | --- |
| base | 375k 315/360 = 0.875 | 575k 307/360 = 0.853 |
| ctx | 425k 317/360 = 0.881 | 325k 311/360 = 0.864 |
| delta | 375k 46/360 = 0.128 | 425k 22/360 = 0.061 |
| ceiling | 450k 8/360 = 0.022 | 500k 7/360 = 0.019 |

Both image runs dip hard between 300k and 350k (base 0.542 and 0.407, ctx 0.579 and 0.569). Two runs on the same corpus dipping at the same place point at the data or the schedule rather than noise. This is recorded, not explained.

## 5. Lever 1: the camera-pose context

### 5.1 Official gate and generalization read

| | base | ctx | delta | t4 (seed 20261047, reference) |
| --- | --- | --- | --- | --- |
| per-brick | 585/720 = 0.8125 | 600/720 = 0.833 | 58/720 = 0.081 | 605/720 = 0.840 |
| Wilson 95 % | [0.782, 0.839] | [0.804, 0.859] | [0.063, 0.103] | [0.812, 0.865] |
| histogram, 0 to 6 bricks | 0, 5, 10, 9, 11, 21, 64 | 2, 3, 6, 4, 16, 25, 64 | 74, 34, 12, 0, 0, 0, 0 | 3, 2, 3, 11, 12, 18, 71 |
| all six bricks | 64/120 | 64/120 | 0/120 | 71/120 |
| episodes with any brick | 120/120 | 118/120 | 46/120 | 117/120 |
| never closed the jaws | 0/120 | 0/120 | 0/120 | 0/120 |
| first close, median (p10, p90) | 196 (155, 285) | 207.5 (170, 294) | 342.5 (257, 548) | 252 (189, 317) |
| graded grasp attempts | 1110 | 1264 | 2005 | 1196 |
| wrong bin | 4 of 699 releases | 0 of 662 | 48 of 275 | 9 of 764 |
| closes within 2 cm | 775 of 1113 (69.6 %) | 869 of 1267 (68.6 %) | 1220 of 2016 (60.5 %) | 824 of 1191 (69.2 %) |
| clamp fraction | 0.019 | 0.014 | 0.006 | 0.029 |

| generalization, scale 3.0 | base | ctx | delta |
| --- | --- | --- | --- |
| per-brick | 216/360 = 0.600 | 260/360 = 0.722 | 31/360 = 0.086 |
| Wilson 95 % | [0.549, 0.649] | [0.674, 0.766] | [0.061, 0.120] |
| histogram, 0 to 6 bricks | 4, 6, 8, 8, 11, 12, 11 | 1, 3, 6, 7, 9, 16, 18 | 35, 19, 6, 0, 0, 0, 0 |
| all six bricks | 11/60 | 18/60 | 0/60 |
| episodes with any brick | 56/60 | 59/60 | 25/60 |
| first close, median | 211.5 | 220.5 | 321 |
| wrong bin | 5 of 311 releases | 2 of 364 | 17 of 119 |
| closes within 2 cm | 359 of 592 (60.6 %) | 488 of 702 (69.5 %) | 537 of 1005 (53.4 %) |
| clamp fraction | 0.008 | 0.012 | 0.008 |

### 5.2 Read

**Adopt.** The rule: the ctx gate at least the base, and a generalization gain of 0.10 or more. The gate reads 0.833 against 0.8125, and the generalization read gains 0.122. At the training spread the two policies are indistinguishable on the gate. At three times the spread, the base loses close accuracy (69.6 % to 60.6 % within 2 cm) and the context policy keeps it (68.6 % to 69.5 %). A likely reading, not tested here: the camera pose lets the policy correct for a moved camera, and without it the base policy has to infer the camera from the images alone.

## 6. Lever 2: delta joint actions

### 6.1 Read

**Reject.** The rule: the delta gate at least the base minus 0.02 (0.7925), and the commanded acceleration quantiles do not rise. The gate reads 0.081, so the acceleration condition is moot. Action standard deviation from `meta/stats.json`: the six absolute joint columns 0.31 to 1.18 rad, the six delta columns 0.015 to 0.032 rad, the gripper 0.989 in both.

### 6.2 The wiring is correct

The checkpoint directory carries `action_space.json` = delta, and every eval payload's policy id ends in `+delta`. `integrate_delta` adds the measured state to the six joint columns only, and a test pins that the gripper column is untouched. The converter subtracts the measured joints from the six joint targets only. Nothing in `core/eval_loop.py`, `mj_eval_policy.py`, the client or the protocol integrates a second time.

### 6.3 Traced diagnostic

Not a gate read. One 4-env rollout per policy, full 5400-step episodes, the seeds of the delta 250k rung's draw 0, delta 250k against base 250k, with `mj_eval_policy.py --trace_dir`.

| | delta | base |
| --- | --- | --- |
| steps with the command within 5 mrad of the measured pose | 82 % | 34 % |
| command minus measured while the arm moves, median | 7 mrad | 12 mrad |
| joint speed, median | 0.001 rad/step | 0.004 rad/step |
| fraction of steps the arm moves | 0.26 | 0.69 |
| joint travel 30 to 150 steps after a close, median | 0.127 rad | 0.399 rad |
| closes that reopen within 30 steps | 9 of 88 | 0 of 50 |

The delta arm does not park: it moves after every close. It moves slowly, on a command that hugs its own pose. On the ladder episodes it closes 15 to 22 times on one brick at a near-constant distance and rarely holds (4 grasp successes in 205 closes at 250k draw 0, base 16 in 113).

Hypothesis, not verified: under absolute targets, a lagging arm widens the gap between command and measured pose, and the PD law pushes harder. Under target-minus-measured deltas the policy predicts a small, nearly constant gap however far the arm lags, so the controller never catches up, least of all under a brick's load. A delta against the previous target, or against the chunk's first state, keeps that feedback and is the candidate for any retry. That changes plan Task 14 and is an owner decision.

## 7. Lever 3: the privileged-state ceiling

### 7.1 Read

**Not a valid ceiling.** The ceiling run trained on `dataset_ceiling`: no images, and the six brick poses (position x, y, z and the MuJoCo quaternion w, x, y, z) as a 42-wide `observation.environment_state`. It read on the selection seed only and was never gated. Its ladder stows 0.000 to 0.028 on every rung, and the check step reads 450k 8/360 = 0.022 and 500k 7/360 = 0.019. At 450k it is under the delta run.

The rule: under 0.90 on the selection seed, the demonstrations or the control law limit the task. At 0.90 or more while the vision runs read far under it, vision is the limit. The rule fires on the number. Its reading assumes a state policy that learned the task, and this one did not. The image runs reach 0.875 and 0.881 on the same five check draws with the same demonstrations and the same control law. So the read says nothing about the demonstrations or the control law. It says this state encoding did not train. The owner decides what the lever 3 read means for the DAgger plan.

### 7.2 The wiring is correct

Every rung passes `--context privileged`, and the policy id carries `+ctx42`. The env writes the context in `_refresh` from the same `qpos` that the recorder saves during generation. An empirical check built the eval env as `mj_eval_policy.py` does (2 envs, per-episode draws, privileged context) and stepped it 300 times with random joint actions. The served context changed on 299 of 299 steps, and its positions matched the true brick positions within 1.4 mm.

### 7.3 What the policy does

On the 250k rung the policy closes the jaws often and far from the bricks: 103 closes at a median 5.65 cm from the nearest brick on the 250k rung's draw 0, against 1.73 cm for the base, with 5 grasp successes. Its final training loss is 0.176, against 0.068 for the base. The recorded quaternion columns span -1 to 1 with standard deviations of 0.37 to 0.59.

Hypotheses, not tested: the quaternion double cover (q and -q describe one pose), and the fixed brick index order, from which the policy must infer the brick the oracle picks next (nearest first). A retry candidate is a continuous rotation encoding (yaw sine and cosine, or a 6D rotation) and the target brick marked in the state.

## 8. Open questions

- The ladder dip between 300k and 350k in both image runs (section 4).
- The 41 `no_reachable_brick` abandons under per-episode draws (section 3.2).
- A delta formulation that keeps the tracking feedback (section 6.3).
- The ceiling's state encoding: the quaternion double cover and the unmarked target brick (section 7.3).
- R-GATE: every gate read is under 0.90, and the owner decides whether the adopted ctx policy is the starting point for the DAgger plan.

## 9. Reproduction

```
# Code: maker-arm-lab PRs #33 to #38 (stack #39). Drivers and ledger under ~/data/maker-arm-ms4/.
git -C ~/src/maker-arm-lab worktree add --detach ~/src/maker-arm-lab-ms4 5cccc7d
ln -s ~/src/maker-arm-lab/.venv ~/src/maker-arm-lab-ms4/.venv
ln -s ~/src/maker-arm-lab/lerobot_stack/.venv ~/src/maker-arm-lab-ms4/lerobot_stack/.venv

# Preflight: every stage at toy size.
bash ~/data/maker-arm-ms4/run_lane.sh 90 3 preflight & bash ~/data/maker-arm-ms4/run_lane.sh 91 3 preflight &
setsid bash -c 'exec bash ~/data/maker-arm-ms4/preflight/run_preflight.sh' < /dev/null > /dev/null 2>&1 &

# The campaign: lanes, extra conversions, merges, video checks, disk guard, smokes, then
# base, ctx, delta and ceiling each trained and laddered in turn.
setsid bash -c 'exec bash ~/data/maker-arm-ms4/run_chain.sh' < /dev/null > /dev/null 2>&1 &
```
