# MS4 c2: the delta and ceiling retries, and DAgger

Milestone: MS4, campaign c2. It runs the two retries c1 left open and the DAgger rounds the roadmap pre-registered. The spec and the plan are design documents outside this repository. Executed 2026-09-14 to 2026-09-15 on the RTX 5090 host (MuJoCo 3.13, LeRobot 0.5.1), from maker-arm-lab `main` at 654332e. Every number below is read off a file under `~/data/maker-arm-ms4/` on that host. The running record is `~/data/maker-arm-ms4/README.md`, and this report comes from it.

## 1. Summary

**DAgger round 1 gives MS4 its first `GATE_OK`, at 0.9167.** Round 2 improves on it to 0.9444.

Round 2 then misses its own adoption bar of 0.9467 by 2 bricks out of 720, so the rounds stop. Both retries fail their own rules. The privileged-state ceiling drops from MS4, and delta joint actions anchored on the previous target stay rejected. Every read rule went into the ledger before its number arrived.

| run | gated rung | official gate, seed 20261027 | generalization, scale 3.0, seed 20261077 | read |
| --- | --- | --- | --- | --- |
| ceiling retry, 66-wide | 275k, selected | never gated, check step 8/360 = **0.022** on the selection seed | not read | **dropped from MS4** |
| delta_prev | 550k | 93/720 = **0.129** [0.107, 0.156] | 29/360 = **0.081** [0.057, 0.113] | **rejected** |
| DAgger round 1 | 475k | 660/720 = **0.9167** [0.8942, 0.9347] | 266/360 = **0.7389** [0.6912, 0.7816] | **adopted, the first `GATE_OK`** |
| DAgger round 2 | 450k | 680/720 = **0.9444** [0.9252, 0.9589] | **0.7722** | **not adopted, the rounds stop** |

Brackets are Wilson 95 % intervals. The gate criterion is 0.90 per brick. Both DAgger rounds clear it, and R-GATE passes on both.

The round 2 stop needs a plain statement. The DAgger rule adopts a round at 0.03 or more above the previous read. Round 1 read 0.9167, so round 2 needed 0.9467. Round 2 stowed 680 bricks and needed 682. The shortfall is 0.0022, which is 2 bricks out of 720. The Wilson interval [0.9252, 0.9589] contains the bar. The seed noise this campaign measured, 0.047 between the selection and official seeds on one checkpoint, is 21 times the shortfall. The rule fired on a point estimate, exactly as pre-registered. This record states both facts and leaves the judgment to the reader.

The loop did not flatten. Round 2 gained 0.0278 over round 1, against the 0.03 the rule required. Its generalization read rose from 0.7389 to 0.7722, the best of MS4. Its takeover count fell 39 percent, from 4841 to 2970.

The ceiling retry fixed both hypotheses c1 named and moved its rollout rate not at all. The 66-wide channel replaced the MuJoCo quaternion with two rotation columns, and it added a target one-hot and a stowed mask. The final training loss fell 29 percent, from 0.176 to 0.125. The rollout rate read 0.022 in c1 and 0.022 again here. A close distance diagnostic shows both state-only policies close the gripper about 4 to 5 cm from the nearest brick. The image policies close at about 1 cm. The state encoding is not the barrier c1 proposed (section 6).

delta_prev lifted its gate from c1 delta's 0.081 to 0.129 and stayed far under the base run's 0.8125. Its trace refutes the premise the design rests on. The delta_prev command tracks the measured pose more closely than the base run's command does, not less (section 5).

## 2. What was built

Four branches landed before the campaign, as pull requests #41 and #44. #41 carried the delta_prev action space through the converter, protocol version 4 with its optional anchor array, the policy server and the eval client. #44 carried three more branches as one stack: the target-brick rule and the 66-wide privileged channel, the campaign forwarding of `--no_cameras`, `--context` and `--no_images`, and the DAgger rollout loop. That last branch holds the oracle takeovers, the `--dagger` conversion cut, the thin client `scripts/mj_dagger_batch.py` and the held-or-not probe.

The plan expects one commit per stage. All four branches merged together instead, so `654332e` serves both stages. That commit carries the final review's whole fix wave.

## 3. The campaign

### 3.1 Isolation and preflight

Every stage ran from the detached worktree `~/src/maker-arm-lab-ms4` at 654332e, with symlinked venvs and `PYTHONPATH` on the worktree's `source`. One chain driver, `c2/run_chain.sh`, took a stage letter and ran that stage's steps in order. Every finished step skips itself on a relaunch. Stage A converted the delta_prev corpus, generated the ceiling lanes, and trained and laddered both runs. Stage B ran the realizability probe and then the DAgger rounds.

Each stage ran its own preflight at toy size first, and both passed on four toy jobs. Stage A's preflight measured the one result that stops the campaign before a 600k run. The toy ceiling dataset carries an empty image list and an `observation.environment_state` of `[66]`, where c1 measured 42. Stage B's preflight ran one 2-env DAgger rollout on the c1 ctx checkpoint, then its conversion, merge and smoke. That rollout took over 11 times. All 11 takeovers stowed their brick, so the takeover mechanism does real work.

Two rulings kept the preflights out of the campaign's own results. The preflight smokes never ran through `run_train.sh <run> smoke`. That script writes the marker the chain reads. The marker decides whether stage A still needs its own smoke on the merged corpus. The toy ceiling lane 90 went before stage A launched. The chain waits on a count of ten `done` files, and a leftover toy lane makes that count reach ten early.

### 3.2 Stage A corpora

Stage A built two corpora, one from the existing c1 shards and one from ten new lanes.

| corpus | source | images | `observation.environment_state` | action space | episodes | frames |
| --- | --- | --- | --- | --- | --- | --- |
| `dataset_delta_prev` | the ten c1 lanes, reconverted | wrist, overhead | none | delta_prev | 6234 | 3010848 |
| ceiling, 66-wide | ten new lanes, seed `20262067 + 100 * lane` | none | (66,) | absolute | 6228 | 2976134 |

The nine remaining delta_prev lanes converted in 86 minutes, in two batches of five. The stage A preflight already wrote lane 9, so the chain skipped it. Both merges read `MERGE_OK`, and the delta_prev merge passed its video decode at `CHECK_OK videos=55`. The ten ceiling lanes finished in 11 minutes against a 180 minute deadline.

The campaign measured its disk cost rather than trusting the design document's prediction. One delta_prev lane costs 970 MB and a merged image family costs 9.3 GB, so the stage A peak is about 19 GB. The 92 kept c1 shards measured 142 GB, against a predicted 152 GB. Their deletion, gated on the delta_prev video decode, took free space from 205 GB to 347 GB. If a later round needs those shards, they regenerate from their seeds.

### 3.3 Stage B, the probe and a defective seed formula

The probe gates the whole stage. It trains a held-or-not classifier on lanes 0 to 7 and reads the held-out lanes 8 and 9. Round 1 starts only at an accuracy of 0.90 or more. The probe read `PROBE_OK accuracy=0.9882 held=1684 free=4316`.

Round 1 then stopped 18 seconds later, on a defect in the plan's seed formula. All ten policy servers reached `SERVER_READY`. All 84 rollout clients then failed with one identical line: `--run_seed is outside numpy's seed range (0 <= seed < 2**32)`. The plan sets the run seed of round k and process p to `(20261107 + 10 * (k - 1)) * 1000 + p`. For round 1 that reaches 20261107000, about 4.7 times numpy's limit of 4294967296. The script rejects that seed correctly. The formula was wrong, not the guard.

The correction keeps the plan's base and its layout offset. The round base becomes `20261107 + 10000 * (k - 1)`. The run seed is that base plus p, and the layout seed is the base plus 500 plus p. An added round stride of 10000 replaces the multiply by 1000. The three round blocks stay inside numpy's range and collide with no seed this campaign already uses.

The stage B preflight had no way to catch this. Plan step 3 asks for one rollout process on the ctx checkpoint. It never says to use the round's own seed formula, so the preflight chose its own seeds and passed. A preflight that tests a formula has to be told to use that formula.

### 3.4 The shape of a round

Each round runs five steps in order. First come 84 rollout processes against ten policy servers. Then a conversion takes one lane at a time. Then a merge runs with a video decode. MS3 lost two 600k runs to two bad AV1 frames, so that decode gates the deletion of the raw shards. A 600k training and the selection ladder close the round.

| step | round 1 | round 2 |
| --- | --- | --- |
| rollouts, 84 processes | 68 min | 41 min for the 41 resumed processes |
| conversion | 3 h 6 min | 1 h 41 min |
| 600k training | 3 h 47 min | 3 h 47 min |
| ladder | 70 min | 55 min |

The ledger states the round 1 ladder time as a duration. The round 2 time comes from its timestamps, a start at 19 14 PDT and a gated rung at 20 09 PDT.

Round 2 paused mid-rollout at 43 of 84 shards, to free the machine for other work. Seven half-written shards went, each an hdf5 file with no matching json. The resume kept the 43 finished shards and ran only the 41 missing processes. A run seed is a function of the process index alone, so the resumed shards match the ones already on disk. The resume cost no completed work.

## 4. Training and selection

Each run trained alone on the GPU, 600k steps, checkpoints every 25k, the MS3 protocol unchanged. Every run printed `TRAIN_OK`.

| run | training time | final loss | gated rung | check step, draws 0 to 4 |
| --- | --- | --- | --- | --- |
| ceiling retry | 2 h 9 min | 0.125 | 275k | 8/360 = 0.022, against 525k at 5/360 = 0.014 |
| delta_prev | 3 h 38 min | 0.159 | 550k | 32/360 = 0.089, against 250k at 20/360 = 0.056 |
| DAgger round 1 | 3 h 47 min | 0.074 | 475k | 347/360 = 0.9639 |
| DAgger round 2 | 3 h 47 min | about 0.080 | 450k | not in the record |

c1 measured base 0.068, ctx 0.071, delta 0.142 and ceiling 0.176. Rule L3 read rungs 250k to 600k on draws 0 to 2 of seed 20261017. The check step read the top two rungs on draws 0 to 4. Every eval used `--temporal_ensemble_coeff 0.01 --n_action_steps 1 --episode_draws --frame_jitter`. The ceiling run added `--context privileged`, and the DAgger runs added `--context camera`.

The two DAgger runs fit their corpora about as well as c1's ctx run fits the plain one. Round 1 read 0.074 on a corpus 73 percent larger than c1 ctx's. Round 2 read about 0.080 on a corpus 26 percent larger again, across a wider spread of 0.075 to 0.085. The added episodes are takeover segments that start from states the policy itself reached. A reading, marked as one. That distribution is harder than scripted demonstrations, so a fit level with ctx is the good case and not a neutral one.

The ceiling ladder read every rung between 0.000 and 0.032, and its L3 candidate rung275k came back flagged as an isolated spike. The delta_prev ladder gated rung550k at 17/216 = 0.079 on draws 0 to 2, which is not an isolated spike. The 600k step budget is fixed, so a training duration tracks steps and not corpus size.

## 5. Lever 2 retry: delta_prev

### 5.1 What changed

c1's delta action was the joint target minus the measured joint position of the same frame. Its gate read 0.081. The c1 trace diagnosed a command that hugs the measured pose, and it named the anchor as the cause. delta_prev anchors the recorded action on the previous commanded target instead. Protocol version 4 carries an optional anchor array, and the server integrates on the client's anchor. The run trains on the same c1 shards, with no context.

### 5.2 Official gate and generalization read

| | delta_prev | c1 delta | c1 base |
| --- | --- | --- | --- |
| per-brick gate | 93/720 = 0.129 | 58/720 = 0.081 | 585/720 = 0.8125 |
| Wilson 95 % | [0.107, 0.156] | [0.063, 0.103] | [0.782, 0.839] |
| generalization, scale 3.0 | 29/360 = 0.081 | 31/360 = 0.086 | 216/360 = 0.600 |
| final training loss | 0.159 | 0.142 | 0.068 |
| action std per joint | 0.012 | 0.015 to 0.032 | 0.31 to 1.18 |

The action standard deviations come from each corpus `meta/stats.json`, in radians. The gripper column reads 0.989 in every family.

### 5.3 Read

**Reject.** The rule adopts delta_prev at an official gate of 0.7925 or better.

The second condition asks for commanded acceleration quantiles at or below the base run's. The gate reads 0.129, so the rule fires on its first condition alone and the acceleration condition never arbitrates. The new anchor lifts the gate by about half again over c1's delta, and it leaves the run far under the base at 0.8125. Both delta forms stay rejected.

The acceleration measure did not exist before this campaign. The eval JSON carries no acceleration field, and the chain never passes `--trace_dir`. c1's own trace scripts measure close duration, joint travel, command gap and speed. The campaign defined the measure before it computed any number. Take `action_clamped` from a `--trace_dir` rollout and keep the six arm joint columns. Take the second difference along time, take the absolute value, then take the maximum over the six joints. Report the median, p90 and p99, in radians per control step squared.

### 5.4 Traced diagnostic, and what it established

Not a gate read. Two rollouts, one per policy, each on its own gated rung, 4 envs, the full 5400-step budget, and the seeds c1 used for its traces. The delta_prev server logged the action space `delta_prev` and a policy id ending `+dprev`, so the wire ran the new space. A wiring defect does not explain the result.

| | base 375k | delta_prev 550k |
| --- | --- | --- |
| commanded acceleration, median | 0.00072 | 0.00009 |
| commanded acceleration, p90 | 0.00293 | 0.00056 |
| command gap while the arm moves, median | 0.01516 rad | 0.00861 rad |
| steps within 5 mrad of the measured pose | 24.7 % | 51.4 % |
| total steps over 4 envs | 18238 | 21600 |
| bricks stowed | 23 | 3 |

**What the trace measured.** The delta_prev command sits closer to the measured pose than base's does, not further.

More than half of its steps command a target within 5 mrad of the current pose, against a quarter for base. Its commanded acceleration is about eight times smaller at the median. That is the direction c1's delta pointed as well. The anchor change did not break the coupling the design set out to break. For reference, the c1 traces read a base acceleration median of 0.00052 and a c1 delta median of 0.00005.

An interpretation, marked as one. The collapse looks like the size of the predicted delta, not the choice of anchor. The corpus action std reads 0.012 per joint for delta_prev, against 0.31 to 1.18 for the absolute runs. A policy that predicts a near-zero delta produces a target that barely advances, on either anchor. The arm tracks its target closely, so target and pose converge either way. On this reading the anchor was never the problem, and lever 2 fails for one cause in both of its forms.

One more number is worth keeping. delta_prev ran its full 21600 step budget with no episode ending early, and it stowed 3 bricks against base's 23. The run does not fail fast. It runs out the clock.

## 6. Lever 3 retry: the privileged-state ceiling

### 6.1 What changed

c1's ceiling carried a 42-wide `observation.environment_state`. That channel held six brick poses as position x, y, z plus the MuJoCo quaternion w, x, y, z. It stowed 0.022 and left two untested hypotheses for the failure. The first is the quaternion double cover, where q and -q describe one pose. The second is the fixed brick index order, which hides the brick the oracle picks next. The retry widened the channel to 66. Two rotation columns replace the quaternion, and a target one-hot plus a stowed mask mark the oracle's nearest-first target.

### 6.2 Read

**The ceiling drops from MS4.** The pre-registered rule fires as written.

The rule reads a rate under 0.90 with a final loss above 0.102 as an encoding that still fails. The measured rate is 0.022 and the measured loss is 0.125. The bar of 0.102 is 1.5 times the base run's 0.068. That reading went into the ledger before the ladder returned, so it is not fitted to the answer. The 66-wide channel also cost no measurable training time, at 2 hours 9 minutes against c1's 2 hours 8 minutes.

### 6.3 What the retry established

The new encoding did what it was designed to do on the fit. The final loss fell from 0.176 to 0.125, a cut of 29 percent. The rollout rate did not move at all. c1's ceiling read 8/360 = 0.022 on its check step, and this one reads 8/360 = 0.022. Both hypotheses c1 named are now tested, and neither explains the failure.

The close distance diagnostic says more than the rule does. Each row is one selected checkpoint on its own read.

| run | per-brick rate | median close distance | closes within 2 cm | grasp successes |
| --- | --- | --- | --- | --- |
| c2 ceiling rung275k, 66-wide | 0.0222 | 5.20 cm | 41/540 = 7.6 % | 16/536 |
| c1 ceiling rung450k, 42-wide | 0.0222 | 4.20 cm | 63/1073 = 5.9 % | 32/1067 |
| c1 base official, images | 0.8125 | 1.06 cm | 775/1113 = 69.6 % | 188/1110 |
| c1 ctx official, images and camera | 0.8333 | 1.01 cm | 869/1267 = 68.6 % | 227/1264 |

Both state-only runs close the gripper about 4 to 5 cm from the nearest brick. The image runs close at about 1 cm. Only 6 to 8 percent of state-only closes land within 2 cm, against about 69 percent for the image runs. The 66-wide retry did not shorten that distance. It read 5.20 cm against the 42-wide run's 4.20 cm.

So the state encoding is not the barrier c1 proposed. A policy that reads exact brick positions as 66 floats still closes 5 cm away. A policy that holds no such coordinates closes at 1 cm. The barrier sits somewhere else in the state-only path. Naming it needs work this round did not do, and the ceiling leaves MS4 without that name.

### 6.4 The wiring

The stage A preflight measured the channel on a real dataset before any 600k run. The toy ceiling dataset carried an empty image list and an `observation.environment_state` of `[66]`, where c1 measured 42. The source agrees, because `CONTEXT_DIMS` records camera 6 against privileged 66.

## 7. The DAgger rounds

### 7.1 How a round works

The policy drives. The oracle rig takes over on a trigger for one brick cycle, and the round records only takeover frames. Four triggers fire a takeover. `random` fires on a timed interval, and `close_far` on jaws that shut too far from the brick. `no_close` fires on a cycle with no close at all, and `closed_no_pinch` on a close with no grip. The `--dagger` conversion cut keeps complete takeover cycles only. Each round merges its lanes onto the corpus of the round before it, then trains a fresh 600k policy on the whole mixture.

Round 1 started from the c1 ctx checkpoint, which is the policy lever 1 adopted. Round 2 started from round 1's gated rung475k.

### 7.2 Round 1

The rollouts read 84 of 84 `DAGGER_OK` in 68 minutes, over 1008 episodes. The mixed per-brick rate reads 0.991, which is the policy plus the oracle takeovers and not the policy alone. There were 4841 takeovers. Of those, 4725 stowed their brick and 108 failed, a takeover success rate of 97.6 percent. Those two outcomes come to 4833, eight short of the total. The loop counts a takeover at its start, and counts stowed or failed only at the moment a result resolves. Eight takeovers were still running at the end of the rollout, which is a reading of that gap and not an observation. Declined takeovers are counted in a separate field and sit outside the total, at 1241 for this round. No read depends on these counters.

The trigger mix says where the policy gets stuck: `random` 4069, `close_far` 529, `no_close` 129 and `closed_no_pinch` 114. So 772 takeovers fired on a real failure signal rather than on the timed interval. `close_far` leads that group, which agrees with the c1 finding that this policy closes its jaws too far from the brick.

The conversion kept 4571 episodes and 1458733 frames from 4841 takeovers, a ratio of 0.944. Every lane reported both image keys and an `observation.environment_state` of `[6]`, so no lane took the wrong context. The merge read `MERGE_OK roots=85 episodes=10805 frames=4469581`, and the prediction written before it matched to the digit. The corpus grew 73 percent over c1 ctx's 6234 episodes. The merge passed its video decode at `CHECK_OK videos=223`, and the round then deleted its 55 GB of raw shards.

A reading of the 270 missing cycles, marked as inference. 116 takeovers did not stow. A takeover that starts near the end of an episode has no room to finish its cycle, and `--split_cycles` keeps complete cycles only. Across 1008 episodes that is about 0.27 dropped cycles per episode. The split between those two causes is not measured.

**Round 1 read: adopt.** The official gate returned `GATE_OK` at 660/720 = 0.9167.

The full read is Wilson [0.8942, 0.9347], seed 20261027, 120 episodes, checkpoint 475000. This is the first `GATE_OK` of MS4. The rate clears the R-GATE bar of 0.90 and the DAgger bar of 0.863, which is c1 ctx's 0.833 plus 0.03. Round 1 gains 0.084 over ctx.

One honest note on the interval. The Wilson lower bound reads 0.8942, just under the 0.90 threshold. The gate rule fires on the point estimate, which is how the campaign pre-registered it. The interval does not exclude a true rate near 0.90.

A band flagged before the gate did not occur. A rate between 0.863 and 0.899 meets the stated DAgger bar, writes no `gated` file and stops the rounds for an owner decision. 0.9167 sits above that band, so the chain continued on its own.

The generalization read at spatial scale 3.0 returned `GATE_FAIL` at 266/360 = 0.7389, Wilson [0.6912, 0.7816]. That is the highest generalization read of MS4 to that point. The pair of numbers matters more than either one alone.

| run | official gate | generalization at scale 3.0 | drop |
| --- | --- | --- | --- |
| c1 base | 0.8125 | 0.600 | 0.213 |
| c1 ctx | 0.8333 | 0.7222 | 0.111 |
| c2 delta_prev | 0.1292 | 0.0806 | 0.049 |
| c2 DAgger round 1 | 0.9167 | 0.7389 | 0.178 |

A reading, marked as one. DAgger lifted the official gate by 0.084 over ctx and the generalization read by only 0.017. The drop from gate to generalization widened from ctx's 0.111 to 0.178. DAgger corrects a policy on the states that policy reaches inside the training layout distribution. A 3x spatial scale sits outside that distribution, so the correction does not carry there. On this evidence the rounds buy in-distribution reliability and almost no generalization.

The two eval seeds disagree by a useful margin. The check step on selection seed 20261017 reads rung475k at 347/360 = 0.9639. The official gate on seed 20261027 reads 660/720 = 0.9167 for the same checkpoint. The layout draws differ, and the 0.047 gap sets the scale of seed noise in these reads.

### 7.3 Round 2

The rollouts again read 84 of 84 `DAGGER_OK`, over 1008 episodes, at a mixed per-brick rate of 0.990. There were 2970 takeovers, of which 2925 stowed and 44 failed. One takeover was still running at the end, by the same reading of that gap. This round declined 1018. No read depends on these counters.

Takeovers fell by 39 percent, which is what an improving loop looks like. Round 1 needed 4841 takeovers on the ctx policy. Round 2 needed 2970 on the round 1 policy. The rig rescued the newer policy far less often.

| trigger | round 1 | round 2 | change |
| --- | --- | --- | --- |
| `random` | 4069 | 2448 | -1621 |
| `close_far` | 529 | 108 | -421 |
| `no_close` | 129 | 243 | +114 |
| `closed_no_pinch` | 114 | 171 | +57 |

The failure mix moved, and that is the useful part. `close_far` fell by 80 percent. That trigger fires on a policy that shuts its jaws too far from the brick, which is the failure the close distance diagnostic named. A reading, marked as one. Round 1's data fixed most of it. The two rounds differ in more than one way, because the checkpoint, the corpus size and the driving policy all changed. The residual failures moved to `no_close`, up 88 percent, and to `closed_no_pinch`, up 50 percent.

The conversion kept 2797 episodes and 875801 frames, a ratio of 0.942 against this round's takeovers. Round 1 kept 0.944. That stability makes the ratio a planning number rather than a single observation. Every lane again reported an `observation.environment_state` of `[6]` with both image keys. The merge read `MERGE_OK roots=169 episodes=13602 frames=5345382`, and the narrowed prediction written before it was 13602. The 169 roots are the c1 ctx corpus plus 84 lanes from each round. This merge passed its video decode at `CHECK_OK videos=391`, and the round then deleted its 33 GB of raw shards.

**Round 2 read: not adopted. The rounds stop.** The official gate returned `GATE_OK` at 680/720 = 0.9444.

The full read is Wilson [0.9252, 0.9589], seed 20261027, checkpoint 450000. The rate passes R-GATE at 0.90 and misses the DAgger bar of 0.9467. That bar is round 1's 0.9167 plus 0.03, and it binds above R-GATE. `read.json` records adopt false, no `gated` file exists, and the chain logged `round 2 not adopted; rounds stop`.

The miss is 2 bricks out of 720. Round 2 needed 682 bricks to clear the bar and stowed 680. One brick is worth 0.001389 of rate, and the shortfall is 0.002222.

The miss sits inside the noise this campaign already measured. The Wilson interval [0.9252, 0.9589] contains the bar of 0.9467. The seed noise measured on round 1, 0.047 between the selection and official seeds on one checkpoint, is 21 times the shortfall. The rule fired on a point estimate, exactly as pre-registered. This record states both facts and takes no position on them.

The loop did not flatten. Round 2 gained 0.0278 over round 1, against the 0.03 the rule required. Its generalization read rose from round 1's 0.7389 to 0.7722, the best figure of MS4. Round 2 fell 2 bricks short of a threshold on the rate of improvement, not on the rate itself.

Round 3 never started. The campaign's own rule stopped the rounds, and that rule assigns the call to the owner. No `dagger_r3` directory exists, and every input a third round needs stays on disk.

## 8. Open questions

- Round 3. The campaign stopped by its own pre-registered rule, so the decision to run a third round belongs to the owner. The plan pre-registered a maximum of three rounds, so a third round was always in scope. The margin is 2 bricks out of 720, a shortfall of 0.0022. The Wilson interval [0.9252, 0.9589] contains the bar of 0.9467, and the measured seed noise of 0.047 is 21 times the shortfall. Round 2 gained 0.0278 over round 1 and lifted its generalization read, so the loop did not flatten.
- The barrier in the state-only path. It is not the state encoding, and this round measured no evidence that names it (section 6.3).
- The size of the predicted delta, which the trace names as the one cause behind both lever 2 collapses (section 5.4).
- Generalization. DAgger buys in-distribution reliability and almost no transfer to a 3x spatial scale (section 7.2). A lever for that read is still open.
- The defective DAgger seed formula. The stage B preflight had no way to catch it, because plan step 3 never says to use the round's own seed formula (section 3.3). A preflight that tests a formula has to be told to use that formula.
- R-GATE. Both DAgger rounds read above 0.90. MS4 now holds two policies over the gate criterion, and the owner decides what the next milestone starts from.

## 9. Reproduction

```
# Code: maker-arm-lab pull requests #41 and #44, main at 654332e.
# Drivers and ledger under ~/data/maker-arm-ms4/c2/.
git -C ~/src/maker-arm-lab worktree add --detach ~/src/maker-arm-lab-ms4 654332e
ln -s ~/src/maker-arm-lab/.venv ~/src/maker-arm-lab-ms4/.venv
ln -s ~/src/maker-arm-lab/lerobot_stack/.venv ~/src/maker-arm-lab-ms4/lerobot_stack/.venv

# Stage A: the delta_prev conversions and merge, the ten ceiling lanes and merge,
# the c1 shard deletion, then both 600k trainings and their ladders.
setsid bash -c 'exec bash ~/data/maker-arm-ms4/c2/run_chain.sh A' < /dev/null \
    >> ~/data/maker-arm-ms4/c2/chain.out 2>&1 &

# Stage B: the realizability probe, then DAgger rounds 1 to 3.
setsid bash -c 'exec bash ~/data/maker-arm-ms4/c2/run_chain.sh B' < /dev/null \
    >> ~/data/maker-arm-ms4/c2/chain_b.out 2>&1 &
```
