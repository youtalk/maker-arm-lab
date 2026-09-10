# MS2 — ACT training and the first rollout gate on the Maker Arm sim

Milestone: MS2. Spec: `2026-09-07-ms2-act-gate-design.md`. Executed 2026-09-07 to
2026-09-10 on the RTX 5090 host (Isaac Lab v2.3.2 / Isaac Sim 5.1, LeRobot 0.5.1). Every
number below is read off a file named beside it; the running record is
`~/data/maker-arm-ms2/README.md` on that host, from which this report was assembled.

## 1. Summary

**The pre-registered rollout gate of spec section 5.4 FAILED.** The selected checkpoint
(600k, chosen by the measured ladder of section 5.2) scored **52/720 = 0.072 per-brick**
on the 120-episode official set, Wilson 95% [0.055, 0.093], against a **≥ 0.15** criterion;
the interval excludes the bar, so this is not a near miss. No episode sorted all six bricks;
78 of 120 sorted none. The gate was evaluated once, on the checkpoint the rule named, with
the inference setting the server reported (temporal ensembling 0.01, one action step), and
nothing was re-rolled.

Everything the gate stands on was delivered and measured: a 1034-episode success-only
corpus with the pick-order shortcut removed (pooled oracle 0.899 over 1150 attempts, the
re-pick fix adopted by its pre-registered probe), a 600k-step plain ACT run, a fresh eval
harness that grades with the checker and records what the server actually served, and the
selection ladder, which was load-bearing: the four rungs read 0.036 / 0.017 / 0.011 /
0.081, so three of the four checkpoints would have scored worse. The number the next
milestone inherits is a first Maker Arm ACT policy at **0.072**, between openyam-lab's
first-corpus level (0.023) and its 1000-episode level (0.130), with the diagnostic
hand-off of section 5.3: the policy attempts a grasp about six times per episode at the
demonstrated cadence, but only 13% of graded attempts hold a brick, and of the grasps that
do hold, 58% reach the right bin.

## 2. Phase 0 — housekeeping

Everything the MS1 report left open was cleared before GPU time was spent, with one
mechanical exception recorded plainly.

- **The MS1 stack was landed through pull requests #10 (`feat/ms1-lerobot → main`,
  carrying the reviewed #5, #6 and #7) and #11 (`fix/ms1-oracle-gate`, carrying #8), merge
  commits, no squash.** `origin/main` had stopped at #4 because #5–#8 merged into stack
  branches rather than `main`.
- **Citation sweep (PR #13).** The inventory was re-derived by grep rather than taken from
  `task-E2-report.md`: **22 sites in 13 files**, against the report's estimate of 19–20.
  Each `section 5.x` was rewritten to the MS1 spec's real section (2.2, 2.3, 4.2, 4.4, or
  5 item 4); the two references to OpenYAM's contact-escalation ladder, which the MS1 spec
  never had, now name openyam-lab explicitly. `tests/test_spec_citations.py` pins that no
  `section 5.<digit>` string remains under `source/`, `scripts/`, `lerobot_stack/` or
  `tests/`.
- **openyam-lab closeout (openyam-lab PR #27).** The spec's ancestry claim was inverted:
  `integration/m6-w1` had already merged `docs/m7-closeout-reports`, so `m6-w1` is the
  superset. Both branches were pushed as-is first (3b92389 and 675941a), so the record —
  the M6 final report, the W1–W4 gate records, the randomized-pick-order implementation,
  the M5 report and the transferable-lessons document, 42 commits over `main` that had
  existed on the 5090 host only — was safe before anything was merged. `main` (PRs #25,
  #26) merged in with no conflicts; 548 CPU tests pass on the result. The repository is
  archived after that PR merges.
- **CI.** The fix is to make `youtalk/maker-arm-rs` public. Its 39 tracked files, full
  history and Apache-2.0 LICENSE/NOTICE were scanned for secret-shaped strings (none), and
  the visibility change itself is the owner's action (spec section 2 step 1). The owner made the repository public on 2026-09-09 (22:18 PDT); the first
green run in this repository's history followed — `main` run 34310058061, `uv run pytest`
collected 406 items, **406 passed** — and every MS2 branch was re-run green after it
(feat/ms2-pick-order 34194607607, feat/ms2-eval-harness 34420176847). Every earlier red
check was uv failing to clone the private dependency, as the MS1 report said.
- **`wip/ms1-repick` was not applied in Phase 0**; it was paid for in probe 2 (section 3.2).

**Delivery note.** The harness this milestone ran under refuses `gh pr merge` and the
visibility change, so every pull request was opened and left for the owner to merge; the
GPU work ran on local branches whose trees are those of the open PRs (the eval harness was
re-based onto the pick-order branch after the fact, tree-identical to the local
integration commit `82e27f1` every GPU run cites).

## 3. Corpus

### 3.1 Randomized pick order (PR #14)

Ported from openyam-lab's `b82e271` / `d82da09` / `a00cfe9` with Maker Arm names.
`OracleNoise.randomize_pick_order` draws each pick uniformly among the reachable, unsorted,
unstowed bricks with the episode's rng, over exactly the candidate list the lowest-index
path built; off by default and byte-identical in both order and rng stream (pinned). Every
episode's sidecar carries `pick_order`; every run's `RUN_STATS` carries
`off_schedule_episodes`, the audit that the knob changed the order. The supervisor forwards
the flag, records each run's `RUN_META` / `RUN_STATS` in its ledger, records the campaign
`target`, and gains the one rule the spec asked for: a circuit breaker that stops the
campaign once four runs have completed with a pooled success rate below 0.70 (it never
fired; the pool sat at 0.90–0.93 throughout).

### 3.2 Probes, pre-registered (spec section 3.2)

Both on the 0.840 re-run's exact command (run_seed 3081035816, layout_seed 3081036316,
permutation 0,3,1,2, 12 envs, 50 attempts, cameras on).

| probe | recipe | result | rule | read |
| --- | --- | --- | --- | --- |
| 1 | re-run + `--randomize_pick_order` (433839e) | **42/50 = 0.840**, Wilson [0.715, 0.917]; clamp 2.2e-4; time_out 3, watchdog 5; off-schedule 49/50; 127.7 s/episode | GO if ≥ 0.70 and clamp ≤ 0.01 | **GO** |
| 2 | probe 1 + `wip/ms1-repick` (9f01614) | **46/50 = 0.920**, Wilson [0.811, 0.967]; clamp 1.7e-5; **time_out 0**, watchdog 4; off-schedule 50/50; 139.2 s/episode | ADOPT if ≥ probe 1 − 0.05 and time_outs ≤ probe 1's | **ADOPT** |

Probe 1 reproduced the lowest-index re-run's 42/50 exactly, so the randomized order costs
the oracle nothing measurable at n = 50; its three time-outs all delivered 5/6, the
lost-brick shape of MS1 section 9.5. Probe 2 removed that mode. The caveat belongs
beside the read: GPU physics is not bit-reproducible, 42 → 46 at n = 50 is inside
run-to-run noise (two-proportion z ≈ 1.2), and the pre-registered rule is what decided.
The re-pick change (RETREAT marks a brick sorted only when the scene shows it stowed) is
folded into PR #14.

### 3.3 Campaign

```
generate_dataset.py --target 1000 --attempts_per_run 50 --num_envs 12 \
  --seed 20260907 --randomize_pick_order
```

Launched 2026-09-07 23:25 PDT at 9f01614, one detached process, never restarted;
`CAMPAIGN_OK total=1034 runs=23` at 2026-09-09 16:56 PDT, **41.0 GPU hours**. The ledger
(`campaign/work/ledger.json`) is the source of this table.

| run | perm | exported/attempts | rate | time_out | watchdog | clamp | raise_vetoed | hover_lowered | off-schedule | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2,3,1,0 | 47/50 | 0.94 | 1 | 2 | 5.0e-05 | 8 | 5 | 50 | 126 |
| 1 | 0,2,1,3 | 48/50 | 0.96 | 1 | 1 | 3.3e-05 | 8 | 5 | 50 | 139 |
| 2 | 2,1,3,0 | 45/50 | 0.90 | 0 | 2 | 8.4e-06 | 7 | 7 | 49 | 135 |
| 3 | 1,3,0,2 | 45/50 | 0.90 | 2 | 3 | 1.8e-04 | 2 | 7 | 49 | 125 |
| 4 | 1,2,0,3 | 46/50 | 0.92 | 0 | 4 | 8.8e-06 | 3 | 3 | 49 | 123 |
| 5 | 2,0,3,1 | 47/50 | 0.94 | 0 | 3 | 6.5e-05 | 6 | 7 | 48 | 118 |
| 6 | 1,0,3,2 | 46/50 | 0.92 | 1 | 1 | 8.2e-06 | 4 | 2 | 50 | 127 |
| 7 | 3,2,0,1 | 45/50 | 0.90 | 1 | 2 | 8.3e-05 | 7 | 9 | 50 | 119 |
| 8 | 0,2,3,1 | 41/50 | 0.82 | 0 | 4 | 1.4e-05 | 4 | 5 | 48 | 110 |
| 9 | 3,0,1,2 | 44/50 | 0.88 | 1 | 4 | 8.0e-05 | 3 | 15 | 49 | 131 |
| 10 | 0,3,2,1 | 44/50 | 0.88 | 0 | 4 | 1.8e-05 | 8 | 7 | 50 | 130 |
| 11 | 2,0,1,3 | 48/50 | 0.96 | 0 | 1 | 4.0e-05 | 3 | 9 | 49 | 141 |
| 12 | 3,0,2,1 | 47/50 | 0.94 | 1 | 2 | 9.2e-05 | 4 | 1 | 50 | 134 |
| 13 | 1,2,3,0 | 44/50 | 0.88 | 1 | 5 | 1.5e-04 | 2 | 6 | 49 | 123 |
| 14 | 1,3,2,0 | 47/50 | 0.94 | 0 | 0 | 6.1e-05 | 7 | 9 | 49 | 140 |
| 15 | 2,3,0,1 | 46/50 | 0.92 | 3 | 1 | 4.1e-05 | 3 | 10 | 50 | 130 |
| 16 | 3,1,0,2 | 49/50 | 0.98 | 0 | 0 | 6.8e-05 | 5 | 10 | 50 | 141 |
| 17 | 3,2,1,0 | 38/50 | 0.76 | 1 | 2 | 8.5e-06 | 8 | 4 | 49 | 127 |
| 18 | 0,1,2,3 | 45/50 | 0.90 | 2 | 2 | 1.0e-04 | 4 | 9 | 50 | 128 |
| 19 | 1,0,2,3 | 40/50 | 0.80 | 2 | 4 | 1.1e-04 | 4 | 7 | 47 | 132 |
| 20 | 2,1,0,3 | 41/50 | 0.82 | 0 | 4 | 2.8e-05 | 5 | 5 | 47 | 115 |
| 21 | 3,1,2,0 | 44/50 | 0.88 | 1 | 3 | 1.9e-04 | 10 | 5 | 48 | 132 |
| 22 | 0,3,1,2 | 47/50 | 0.94 | 0 | 2 | 4.3e-05 | 0 | 6 | 50 | 129 |
| **all** | | **1034/1150** | **0.899** [0.880, 0.915] | 18 | 56 | 6.4e-05 | 115 | 153 | 1130 | 128 |

The corpus is **1034 success-only episodes, 4,406,024 frames, 9.2 GB** in LeRobot v3.0
(`meta/info.json` total_episodes equals the ledger's converted_total). Pooled oracle rate
**0.899** [0.880, 0.915], read from the ledger as the spec requires and not from the
probes — and below both probes (0.840 and 0.920), which is MS1's process lesson repeating
at a smaller amplitude: 50-attempt probes still over-read. The failure modes over 1150
attempts: watchdog 56, grasp_retries_exhausted 26, time_out 18, no_reachable_brick 16
(the re-pick rule abandoning a lost brick at once instead of idling to time-out). The
later runs were weaker than the first ten (runs 17 and 19 at 38/50 and 40/50, where the
grasp-retry mode clustered); no run approached the breaker. `off_schedule_episodes` is
1130/1150: the fixed R,B,Y,G,R,B schedule is gone from this corpus. The lookahead
counters fired in a minority of legs (raise_vetoed 115, hover_lowered 153 over 1150
episodes) and the clamp fraction stayed at 6.4e-5.

Not in the corpus, by decision: widened noise or start-pose offset, failure induction,
per-brick re-split, dwell compression (section 7).

The optional 24-env throughput probe (spec section 3.4) was not run: the campaign fit the
budget at 12 envs and the GPU had no idle window under depth-1 pipelining.

## 4. Training

Plain LeRobot ACT through `lerobot_stack/train_act.sh`, LeRobot defaults (chunk 100,
batch 8), **600k steps, checkpoint every 50k**, one seed, no BCE gripper head, no sweep;
started only after the campaign finished. The hardened script (PR #15) refused to start
until the dataset's episode count (1034) cleared the ledger's target (1000).
wandb: `youtalk-openyam-lab/maker-arm-lab`, run `ms2-act-pick-order-600k`,
https://wandb.ai/youtalk-openyam-lab/maker-arm-lab/runs/afh8r7rf.

`TRAIN_OK steps=600000` after 4 h 24 min (47 step/s at the start, 32 step/s by the end).
Logged loss 0.202 at 12k, 0.075 at 180k, 0.060 at 300k, 0.052 at 420k, 0.045 at 480k and
0.044 from 540k to 600k: 1.09 epochs of the 4.4M-frame corpus at batch 8. Twelve
checkpoints (050000–600000), 7.0 GB. As openyam-lab's M3 warned and this ladder confirms
(section 5.2), the loss curve is a liveness instrument, not a selection one: loss fell
monotonically while the per-brick rate went 0.036 → 0.017 → 0.011 → 0.081.

## 5. Eval harness and gate

### 5.1 Harness (PR #15)

`scripts/eval_policy.py` (Isaac venv) runs `MakerArm-LegoSort-Eval-v0` at N envs against
`lerobot_stack/policy_server.py`: per step it sends the (7,) `observation.state` built by
`core.gate.policy_state` (the converter's own formula) with both 224×224 frames, receives
the (7,) action, clamps the six arm targets through `core.gate.clamp_arm_actions`, hands
the gripper sign to the binary term, and grades with `RolloutTracker` off
`per_brick_stowed` and the env's own `success` termination. `scripts/run_gate_eval.py`
(stdlib) draws the run-level seeds with the campaign's formula, asserts them disjoint from
the campaign ledger, runs one server and one eval process per draw, aggregates the
per-brick rate with a Wilson 95% interval, and reuses finished draws on relaunch. The
server echoes the inference config it actually serves with, and the payload records it.
Not ported: unstick, replay bundles, preview video, wrist fold, close diagnostics.
`core/trace.py`'s empty-env state width is (7,) now, deliberately.

Three disjoint seed sets, fixed before training: corpus 20260907 (ledger), selection
20260917, official 20260927. They are ten apart because the `seed*1000` formula spreads a
campaign of R runs over `[0, R)` and `[500, 500+R)`, so seeds one apart collide once R >
500 — the pairwise-disjointness test found this before any GPU ran.

GPU smoke (2026-09-09, MS1's 100-step checkpoint, 1 process × 2 envs, te 0.01): `EVAL_OK`,
`GATE_FAIL` 0/12 as it should, 5400 steps in 526.7 s. The ladder and gate then measured **25 s per episode at 12 envs**
(about 7 minutes per full-length process, 5400 steps), five times faster than datagen's
128 s, so a 60-episode rung costs 26 minutes and the 120-episode gate 52.

### 5.2 Checkpoint selection (rule L3)

Rungs 300k, 400k, 500k, 600k, each read on the same 60-episode selection set (5 processes
× 12 envs, seed 20260917, te 0.01 with one action step); best per-brick rung is the
candidate, a tie goes to the later rung.

| run | checkpoint | episodes | bricks | per-brick | Wilson 95% | all-six | grasp rate (strict) | wrong-bin / releases | clamp frac | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| rung300k | 300000 | 60 | 13/360 | **0.036** | [0.021, 0.061] | 0.000 | 0.10 (0.10) | 6 / 24 | 1.8e-02 | 25 |
| rung400k | 400000 | 60 | 6/360 | **0.017** | [0.008, 0.036] | 0.000 | 0.11 (0.11) | 0 / 6 | 2.9e-03 | 25 |
| rung500k | 500000 | 60 | 4/360 | **0.011** | [0.004, 0.028] | 0.000 | 0.07 (0.07) | 1 / 6 | 6.5e-04 | 24 |
| rung600k | 600000 | 60 | 29/360 | **0.081** | [0.057, 0.113] | 0.000 | 0.13 (0.13) | 5 / 40 | 2.1e-04 | 25 |

**Selected: 600000** (`eval/selection`, 2026-09-09 23:15 PDT). The ladder is not
monotone: 300k → 500k falls from 0.036 to 0.011 while the loss keeps falling, and 600k
recovers to 0.081. openyam-lab's W4 read 0.300/0.389/0.289/0.322 across the same rungs;
here blind-`last` would by coincidence have picked the same checkpoint, but the rule is what
made that a measurement rather than luck, and the 0.011 rung says a blind 500k would have
cost 7 pp of an 8 pp result. The selection read (0.081 on 60 episodes) and the official
read (0.072 on 120 disjoint episodes) agree within their intervals.

### 5.3 Official gate, evaluated once

| run | checkpoint | episodes | bricks | per-brick | Wilson 95% | all-six | grasp rate (strict) | wrong-bin / releases | clamp frac | s/episode |
|---|---|---|---|---|---|---|---|---|---|---|
| official | 600000 | 120 | 52/720 | **0.072** | [0.055, 0.093] | 0.000 | 0.13 (0.13) | 13 / 80 | 7.4e-04 | 25 |

`GATE_FAIL`: **per-brick 0.072 against ≥ 0.15**, Wilson [0.055, 0.093], 120 episodes over
ten run-level draws (seed 20260927; the permutations are not held out, since the campaign
covered all 24, and the seeds are asserted disjoint from the ledger). The server served
`{n_action_steps: 1, temporal_ensemble_coeff: 0.01, chunk_size: 100}`, echoed into every
payload rather than assumed. Reported, not gated:

| instrument | value | read |
| --- | --- | --- |
| all-six episode rate | 0/120 | no episode finished; every one ran to the 5400-step time-out |
| bricks per episode | {0: 78, 1: 32, 2: 10} | 42 episodes sorted something; per-process stows 7,5,7,5,8,8,4,2,5,1 of 72 |
| grasp attempts, graded | 693 (median 6 per episode; 2 unresolved) | the policy cycles the gripper at about the demonstrated cadence of six picks |
| grasp success rate (strict) | 90/693 = 0.13 (0.13) | **the binding factor**: 87% of closes grip nothing; the loose and strict thresholds agree, so it is not a marginal-pinch artefact |
| episodes that ever held a brick | 66/120 | half the episodes never acquire anything |
| releases detected / wrong-bin | 80 / 13 = 0.16 | of 90 held bricks, 52 reached the right bin (58%), 13 the wrong one, the rest were dropped |
| never closed | 3/120 | the gripper head is live; this is not a jammed-shut policy |
| clamp | 477 steps, all on joint 6, 5 episodes, max overshoot 0.19 rad; fraction 7.4e-4 | the wrist grazes its stop; negligible against the datagen 6.4e-5 and nothing like openyam-lab's joint-6 jam |
| wall | 25 s/episode | |

**Read rules carried from openyam-lab's M6:** wrong-bin is biased low (a release inside
the last 30 steps of a rollout is never graded) and the grasp rate biased high (a loose
attribution threshold, a settle window that grades a momentary pinch, and a chatter guard
that drops the closes most likely to have failed), so both are relative instruments; the
Wilson interval is over bricks and therefore optimistic, since the six bricks of one
episode are not independent.

## 6. Sim/real divergences, carried forward

Nothing in this milestone touched the dataset schema, the Rust clamp path, the
joint-coordinate contract, or the list in the MS1 report's section 4.3. That list is the
one the October bring-up reads next to the gate number, so its items are restated here
without re-argument (the derivations are in MS1 §4.3, and
`tests/test_report_unverified_constants.py` still couples the `*_UNVERIFIED` flags to it):

- `ARM_ARMATURE = 0.02` is a bracketed stability value, not a measured rotor inertia.
- The wrist-camera pose is bounded by a hardware fact (in front of the wrist motor) and
  otherwise unmeasured; **the policy gated here looks through that lens**, so re-measuring
  the mount is a prerequisite for transferring anything learned from the wrist stream.
- Gripper `stiffness = 1000 N/m` is what sets the grip in sim and has no real-arm
  counterpart; `GRIPPER_EFFORT_LIMIT_N = 20` is inert.
- `maker-arm-rs` `tau_max` on j2/j3 was raised upstream (12 / 10 N·m); gains are the
  profile's, unmodified (the kp raise was refuted by measurement).
- The oracle plans joint targets with `core/global_ik.py` and drives the joint-position
  action space; the recorded action stream was always the post-IK joint target.

## 7. Hand-off: levers not pulled

Each lever is named with the openyam-lab number it earned, on that arm, so the next
milestone can rank them without re-deriving.

| lever | openyam-lab evidence | status here |
| --- | --- | --- |
| Widened noise / start-pose offset in the corpus | M3: gate 2.33% → 13.0% from widened data (p = 4e-5) | not in the corpus |
| Failure induction (recovery data) | M5: 13.0% → 19.0% (p = .039), sub-linear | not in the corpus |
| Corpus scale beyond ~1000 | M3 headroom note; W4's 729-episode append moved per-brick 0.372 → 0.626 together with pick order | 1034 episodes |
| Checkpoint selection (L3) | W4 ladder 0.300/0.389/0.289/0.322; blind-`last` would have cost ~7 pp | applied (section 5.2) |
| Temporal-ensemble coefficient (te −0.02) + unstick | B3: 0.356 → 0.411 paired | te 0.01, no unstick |
| Unstick escape | 0.250 → 0.300, dead tail 42% → 17% | not ported |
| BCE gripper head | ~1 pp at the 2% level | not ported |
| Per-brick re-split, dwell compression | W2/W3 corpus re-shaping | not ported |

**Diagnostic hand-off, from the gate's own instruments.** The factor structure is the
one openyam-lab's M6 found, at an earlier stage: (1) *acquisition* — 87% of grasp attempts
close on nothing, and this is where the 0.072 is decided; (2) *delivery* — once a brick is
held, 58% of them are sorted correctly and 16% of detected releases are wrong-bin, which is
the randomized corpus doing its job on the first try (openyam-lab needed W4 to bring
off-schedule wrong-bin from 0.415 to 0.058); (3) *cadence* — six attempts per episode over
5400 steps means the policy is not stalled, it is missing. The levers that address (1) on
openyam-lab were widened approach noise and failure induction (recovery from a missed close),
and the corpus here has neither, by decision. The first-attempt precision the oracle
demonstrates (hover noise ±0.02 m, lateral ±3 mm) is what the policy is failing to imitate,
so widened noise (the M3 lever, 2.33% → 13.0%) is the ranked first lever for MS3, with
induction (M5) second and a 24-env throughput probe to make either affordable. The wrist
camera's unmeasured pose (section 6) sits behind all of this: the acquisition failure is a
visual-servoing failure on a lens no one has measured.

## 8. Reproduction

```bash
# Phase 0 and the code are the pull requests #10, #11, #13, #14, #15 (stack #12) and openyam-lab #27.
# GPU commands run from the repo root with the Isaac venv activated (MS1 report section 8).

# Probes (spec section 3.2): the 0.840 re-run's command + the knob; probe 2 on the re-pick commit.
"$ISAACLAB_PATH/isaaclab.sh" -p scripts/generate_batch.py --headless --enable_cameras \
  --out_dir <dir> --shard_name shard_000 --attempts 50 --num_envs 12 --probe \
  --run_seed 3081035816 --layout_seed 3081036316 --permutation 0,3,1,2 --max_wall_minutes 240 \
  --randomize_pick_order

# Campaign (detached; the ledger resumes with --resume).
setsid bash -c 'python3 scripts/generate_dataset.py \
  --out_dir ~/data/maker-arm-ms2/campaign/work --dataset_root ~/data/maker-arm-ms2/campaign/dataset \
  --target 1000 --attempts_per_run 50 --num_envs 12 --seed 20260907 --randomize_pick_order \
  > ~/data/maker-arm-ms2/campaign/supervisor.log 2>&1' < /dev/null &

# Training.
STEPS=600000 SAVE_FREQ=50000 OUT=~/data/maker-arm-ms2/train/act_pick_order_600k \
  JOB_NAME=ms2-act-pick-order-600k DATASET_ROOT=~/data/maker-arm-ms2/campaign/dataset \
  LEDGER=~/data/maker-arm-ms2/campaign/work/ledger.json bash lerobot_stack/train_act.sh

# Selection ladder (one rung shown) and the official gate.
python3 scripts/run_gate_eval.py --checkpoint <OUT>/checkpoints/300000/pretrained_model \
  --out_dir ~/data/maker-arm-ms2/eval/rung300k --ledger ~/data/maker-arm-ms2/campaign/work/ledger.json \
  --threshold 0.15 --processes 5 --num_envs 12 --seed 20260917 --n_action_steps 1 --temporal_ensemble_coeff 0.01
python3 scripts/run_gate_eval.py --checkpoint <OUT>/checkpoints/600000/pretrained_model \
  --out_dir ~/data/maker-arm-ms2/eval/official --ledger ~/data/maker-arm-ms2/campaign/work/ledger.json \
  --threshold 0.15 --processes 10 --num_envs 12 --seed 20260927 --n_action_steps 1 --temporal_ensemble_coeff 0.01
```
