# MS7h: beyond the oracle

Date: 2026-10-06. Status: draft. This draft uses the ledger through its entry of 15:38 on
2026-10-06. The campaign still runs. A placeholder of the form `<<NAME>>` marks each result that is
not in yet. The controller fills them in.

The design of MS7h is the document of 2026-10-05 whose questions the owner answered in chat. This
report cites it as "MS7h design section N" and restates every rule that it uses. The MS7g report,
`docs/ms7g-close-the-ms7f-gate.md`, is cited as "MS7g report section N". The MS7f report,
`docs/ms7f-real-scale-sorting.md`, is cited as "MS7f report section N". The campaign ledger is
`~/data/maker-arm-ms7/h/README.md`, and `STATE.md` beside it is the resume pointer. Every rule went
into the ledger before its number. The mechanism tables are
`~/data/maker-arm-ms7/h/tables/<TAG>.mech.txt`. The track tables are under
`~/data/maker-arm-ms7/h/tables/<track>/`. Each read has a TAG, its name. The payloads of a read are
under `~/data/maker-arm-ms7/reads/<TAG>/`. Every MS7h TAG starts with `h_`.

The names. The design names its decisions H1 to H7. The ledger names its rules HR1 to HR11 and its
deviations HD1 to HD13. The owner decision of the run is D1. The policies and drivers:

- B is the base of the design: F9n 800k, the MS7g policy from random init with the context fix (MS7g
  report section 9.1). It serves with the switches and the classifier of S.
- S is the final MS7g stack: F5 200k with `--carry_guard --overhang_guard`, the classifier v10e3
  with `--cls_flip` and the final-only vote. From HD3 on, S is the base of Track R.
- O is the oracle: the scripted corpus teacher inside the perceived loop of the gate (section 4.2).
  O_truth is the same teacher with the truth. It is report-only.
- R is a residual actor on a frozen base. The residual is a small correction that the actor adds to
  the base's action (section 5).
- V is π0.5, a pretrained VLA, fine-tuned on the MS7g corpus (section 7). V+R is V with a residual.
- SP is S with `--rig_place`, a hybrid of a learned pick and a scripted carry. The policy drives
  from the selection to the lift. Then the rig carries the part to the voted box and places it
  (section 6).

The bar. D is the sum over the heaps of the candidate's correct stows minus O's, on the same seed
and draws. The paired SE is the standard error (SE, the spread of a measured value) of D. It comes
from the differences heap by heap (`h/bin/pair.py`).

The times. All times are PDT. Each time is the stamp of the ledger entry that records the event. A
time without a date from 13:10 to 23:59 is on 2026-10-05. A time without a date from 00:00 to 15:38
is on 2026-10-06. Some ledger entries stand out of time order. 18:50 stands before "about 17:35",
and the ablation result of 01:46 stands after 01:53. The ledger lists HD10 before HD9, and HD11 to
HD13 stand in its log. The "[V]" entries of the Track V session interleave with the others. This
report keeps each stamp as the ledger writes it.

The numbers. Every number comes from the ledger or from a file that the ledger names. This report
computes no new number.

## 1. Summary

MS7g closed as a near miss (MS7g report section 9.6). On the checked seed its final stack S stows
2545 of 3000 parts in a box of their category. Five policies and servings read 2519 to 2554 on the
screen seed, a plateau. The owner then asked for reinforcement learning (RL) or a
vision-language-action model (VLA), and for a performance beyond the oracle.

The goal (MS7h design section 4.2). A learned policy beats the oracle O head-to-head on paired
heaps. The candidate stows at least 61 more parts correctly than O on 20 heaps, or 86 more on 40
heaps. Its rate and its held-out row must be at least O's on the same read. A correct stow, the rate
and the held-out row keep their MS7f definitions (MS7f report section 1.3):

- A correct stow is a part at rest in a box of its category.
- The rate is the correct stows per simulated hour.
- The held-out row counts the parts of the kinds that no policy and no classifier saw in training.

The verdict. `<<VERDICT>>`

What Phase 0 found. O is far stronger than the design expected. Design section 3.3 read the policy
as faster than its teacher, from teacher reads in other scenes. In the gate scene and the perceived
loop, O stows more parts than B and S, and it stows them faster:

| read | who | seed | correct of 3000 | correct stows an hour | held-out row |
| --- | --- | --- | --- | --- | --- |
| h_O_scr | O | screen | 2670, 0.8900 | 388.9 | 640 of 744, 0.8602 |
| h_O_chk | O | checked | 2685, 0.8950 | 390.2 | 653 of 785, 0.8318 |
| h_B_scr | B | screen | 2480, 0.8267 | 331.0 | 588 of 744, 0.7903 |
| h_B_chk | B | checked | 2490, 0.8300 | 330.7 | 612 of 785, 0.7796 |
| h_S_scr | S | screen | 2533, 0.8443 | 369.5 | 581 of 744, 0.7809 |
| h_S_chk | S | checked | 2525, 0.8417 | 366.0 | 621 of 785, 0.7911 |
| h_Ot_scr | O_truth, report-only | screen | 2901, 0.9670 | 381.1 | 715 of 744, 0.9610 |

So the bar on the checked seed is 2746 correct stows at 390.2 an hour, with a held-out row of 0.8318
(17:37). S stands 160 correct stows under O on that seed. A candidate needs about 221 more correct
stows than S (02:48). The perceived loop and the vote cost O 231 correct stows against O_truth
(18:10).

Where S loses to O (section 4.7). On the screen seed S puts 311 parts in a wrong box, and O puts 163
there. The difference of 148 parts is more than the whole gap of 137 correct stows. Riders give 84
of the 148 (129 against 45), and passengers give 50 of those (55 against 5). S also has 90 wrong-box
cycles whose views agreed on a class, against 46 for O. A part-by-part breakdown puts most of the
gap in the pick (section 4.8).

What execution showed (section 2). Three findings changed the plan. O is much stronger than the
design assumed. The critic of runs 1 to 3 memorized the return of each replay row and gave the actor
no action signal. S's losses against O sit mostly in the pick. On 2026-10-06 at 09:32 the owner
asked for fixes. HD12 (09:38) is the revised plan.

Track R so far (section 5). No run has given a candidate yet. Run 1 (19:36 to 20:28) stopped after
its critic diverged. Run 2 (20:29 to 01:47) read its screen at 3M collected steps. It stowed 2490,
43 under S (paired SE 29.4), at 55.1 fewer correct stows an hour. An ablation put the share loss on
the jaw-gap channel. Run 3 (02:16 to 11:41) held the residual for 10 steps and kept the base's jaw
gap. Its 1M and 3M screens stowed 2536 and 2539, level with S (D = +3 and +6). Run 4 (from 11:41)
changed the critic, the exploration noise and the reward (HD12 P2). Its 3M screen stowed 2432, 101
under S, because its actor loss divided the critic's value by a mean near 0 (HD13). Section 5.10
gives the screens that are not in yet.

The hybrid (section 6). SP stowed 2527 on the screen seed, 6 under S (paired SE 29.2), at 351.8 an
hour. It is not a candidate. The rig's carry removed the air stalls, but the losses moved from the
carry to the hold.

Track V so far (section 7). π0.5 trains on Colab Pro+, a full fine-tune on the full derived dataset
(D1). The 30k-step run started at 20:19. It lands about 04:00 on 2026-10-07, plus the gap of one
session resume ([V] 21:18). The report-only screen of step 10k finished one heap of 20 before its
cut at 09:30. On that heap V stowed 28 of 150, and S stowed 133. The 30k screen decides V's week-2
gate: at least h_S_scr - 30 = 2503 (HD9, and HD11 as HD12 P4 amends it). Section 7.9 gives its
results.

## 2. What execution showed

The design rested on three expectations that execution did not hold. On 2026-10-06 at 09:32 the
owner asked the controller to devise fixes for what execution showed. HD12 (09:38) is the revised
plan (section 2.4). This section states each finding and what each change did.

### 2.1 O is much stronger than the design assumed

The design had no teacher read in the refill scene. From teacher reads in other scenes, it put the
policy above its teacher on the rate (MS7h design section 3.3). In the gate scene and the perceived
loop, O stows more parts than S, and it stows them faster:

- On the screen seed O stows 2670 at 388.9 an hour, and S stows 2533 at 369.5. For S, D = -137 at
  a paired SE of 30.5.
- On the checked seed O stows 2685 at 390.2 an hour, and S stows 2525 at 366.0. For S, D = -160 at
  a paired SE of 29.5.
- The bar on the checked seed is D of at least 61. So a candidate needs about 221 more correct
  stows than S (02:48).

What the changes did:

- HD3 made S the base of Track R in place of B (section 4.6). On the screen seed S stows 53 more
  than B (paired SE 32.8) and 38.4 more an hour. So the gap to O fell from 190 to 137.
- HD9 made V's week-2 gate follow the incumbent: h_S_scr - 30 = 2503 (section 7.7).
- The bar stays (HD12 P5). HD12 F1 states that O's lead is far over what a residual near S can
  add. The data-only slopes of 10:44 support it: a fitted sustained shift around S is worth about
  +25 correct stows a screen, inside one screen SE (section 5.8).

### 2.2 The critic memorized and gave the actor no action signal

The design expected a critic with privileged state to tell the actor which correction helps (MS7h
design section 6.2). Runs 1 to 3 changed the TD target, the weight on the critic and the time scale
of an action. Each change removed a defect that the logs or the screens showed. No screen gained:

- HD5, the mean target (run 2). Run 1's critic diverged to a q_mean of -10.6 for rewards near -1
  to +1. Under the mean target, q_mean tracked the MC return within 0.01 at alpha 2.5.
- HD7, alpha 25 (run 2). At alpha 2.5 the prior held the actor at S, at |r|/b 0.014 to 0.017. At
  alpha 25 the actor moved to |r|/b 0.12 to 0.14. Its 3M screen lost 43 correct stows and 55.1 an
  hour against S, and the ablation put the share loss on the jaw-gap channel.
- HD8, the residual held for 10 steps, the gap channel off and a time cost of 0.002 (run 3). The
  1M and 3M screens read D = +3 and +6 against S, at 224.6 and 224.4 steps a cycle against S's
  228.7. So the harm of run 2 was gone, and no gain came.
- HD10, alpha 10 (run 3). At 3M collected steps the actor sat at |r|/b 0.0184. At alpha 10 it moved
  to 0.065 and passed its do-no-harm check by +2.

The read-only diagnostic of 09:38 found the cause (section 5.7). The critic memorizes the realized
return of each replay row through the privileged vector. On its own rows it predicts the return
with a correlation of 1.000 (RMSE 0.018). On run 2's rows, which it never saw, the correlation is
0.51 and the RMSE 0.615, against 0.655 for a constant. Each state in the replay has one action, so
nothing shapes Q along the action. Over the whole action box Q moves 0.0195 (median), the same size
as the spread of the four critics at one action, 0.0194. The four critics agree on the sign of
dQ/du on 12.6 to 14.8 % of states, where chance is 12.5 %. So runs 1 to 3 had no learning signal,
and screens level with S are what that predicts.

What the changes did:

- An offline test on run 3's replay (10:44, section 5.8) found that every critic variant memorizes
  at 2 or more updates a row. Only the update budget matters. At 0.05 updates a row the held-out
  RMSE falls to 0.540, against 0.652 for a constant. No variant's slope along the action matches
  the data.
- HD12 P2 started run 4 with a 2 x 256 critic, about 0.05 updates a row, held-out cycles that pause
  the critic, exploration noise of 0.5 x the bound and dense pick terms in the reward (section
  5.9).
- Run 4's 3M screen stowed 2432, D = -101 against S. The ledger puts the cause on the scale of the
  actor loss. The loss divides the critic's value by mean|Q|, and run 4's returns sit near 0
  (q_mean 0.002 at 5.9M against run 3's 0.209). So alpha 10 acted about 100 times run 3's. HD13
  floors the scale at 0.6 and sets alpha 4. `<<R4_10M_SCR>>` `<<R4_30M_SCR>>`

### 2.3 S's losses sit in the pick

At 09:38 HD12 F1 read S's excess losses as mostly after the lift. Its evidence on the screen seed,
S against O:

- wrong-box parts 311 against 163
- releases over the desk 34 against 0, and over a rim 11 against 0
- about 50 bounce-outs at the right box
- 83 air stalls against 0

So HD12 P1 built a hybrid in which the rig carries and places each lifted part. At 09:49 a
part-by-part breakdown followed every part of five reads to its final class (section 4.8). On the
screen seed S has 135 fewer correct parts than O. The five largest losses of S, on the screen seed
and on the checked seed:

| class | phase | S - O, screen | S - O, checked |
| --- | --- | --- | --- |
| wrong-box passengers | pick | +50 | +76 |
| touched riders in a wrong box | pick | +34 | +27 |
| on the desk beside a box | release | +30 | +39 |
| in a wrong box after a cycle that held a non-target | pick | +22 | +25 |
| class right, box wrong | carry and release | +21 | +13 |

A match with O on every class that S loses gives S 205 more parts and puts it at O + 70. So the
pick, not the carry, holds most of the gap. The hybrid can recover at most the release and carry
classes.

What the changes did:

- The hybrid's screen h_SP_scr stowed 2527, D = -6 against S (paired SE 29.2), at 351.8 an hour
  (section 6). The rig's carry removed the air stalls, 16 against 83. But riders rose to 157 against
  129 and passengers to 66 against 55. 151 carries ended `lifted_lost`, which the ledger puts on
  false ends of the hold detector. So the hybrid moves losses from the carry to the hold and does
  not lift correct stows. HD12 P1 ended, and run 4 stays on S.
- HD12 P2 added three dense pick terms to run 4's reward: passengers, a held non-target and the
  sweep of the neighbours (section 5.9). They read the truth and enter the reward only.

### 2.4 The revised plan: HD12

HD12 (09:38) went into the ledger after the owner's request of 09:32 and before any number of what
it names. Its findings:

- F1. O stows 137 more than S on the screen seed and 160 more on the checked seed, far over what a
  residual near S can add.
- F2. The run 3 critic diagnostic (section 5.7).
- F3. V at step 10k stows 28 of 150 on one heap, and one V heap takes 71 minutes (section 7.8).

Its plan:

- P1. The hybrid serving `--rig_place` (section 6). Its screen h_SP_scr is a candidate at 2563 or
  more (h_S_scr + 30), paired with h_S_scr and h_O_scr. SP joins the candidate list of HR9, and
  this report names it a hybrid of a learned pick and a scripted carry.
- P2. Run 4 replaces the revision of HR7, because the revisions (a) to (c) do not touch F2. It runs
  on the better of S and SP by screen. An offline test on run 3's replay chooses its critic before
  run 4 starts. Its reward can add dense privileged pick terms. Each choice enters the ledger before
  run 4's first number (section 5.9).
- P3. Run 3 collects until run 4 starts. Its 10M screen runs as scheduled, as the record of F2.
- P4. HD11 amended. V's 30k screen reads draws 0 to 6 first, on 7 lanes. If their correct stows
  plus 150 for each of the 13 other draws stay under 2503, V fails HD9 and stops at its report.
  Otherwise draws 7 to 19 follow (section 7.7).
- P5. The bar of HR2, the checked read of HR9 by 2026-10-17 and the rule of the official read stay.

## 3. The decisions and the deviations

### 3.1 The decisions of the design

The owner took these decisions on 2026-10-05, in chat (MS7h design section 2):

| id | question | decision |
| --- | --- | --- |
| H1 | the purpose | Prove the method in simulation first, then carry it to the real arm. The policy reads only what the real arm has. Privileged state enters only a critic and the reward. |
| H2 | the bar | A head-to-head against O under the same conditions, on paired seeds. The candidate needs about two pair SEs more correct stows. Its rate and held-out row must be at least O's. |
| H3 | the budget | Two weeks, 2026-10-06 to 2026-10-19. If the arm arrives, MS5 takes priority. |
| H4 | the compute | The local hosts, plus Google Colab Pro+ for π0.5 training. The owner subscribes after the data is in Drive and a local dry run passes. |
| H5 | the context fix | `--context_norm=roll_wrapped_norm` and the MEAN_STD normalization of the context are adopted. The fix gets three safeguards. |
| H6 | the approach | RL as the main track, with a π0.5 track in parallel. |
| H7 | Colab for RL | No. Colab trains π0.5 only. Residual RL runs on the local hosts. |

### 3.2 The decisions of the owner during the run

| time | id | decision |
| --- | --- | --- |
| 14:35 | none | "Try again": six new subagents continue the six tracks after the controller's first session ended at 14:10. |
| 15:56 | owner gate 1 | The owner received access to the gated tokenizer repository `google/paligemma-3b-pt-224`. |
| 16:35 | D1 | Track V trains on Colab Pro+ with a full fine-tune on the FULL dataset: 23,957 episodes, 66.3 GB, a Drive peak of about 116 GB. The owner uploads it to Drive. |
| [V] 19:58 and 20:19 | none | The owner ran the Colab smoke and then started the full run. |
| [V] 09:25 | none | The owner closed the Track V session. The first session owns Track V from then on (section 3.5). |
| 09:32 | none | The owner asked the controller to devise fixes for what execution showed. HD12 is the answer (section 2). |

### 3.3 The rules of the ledger

The ledger holds rules HR1 to HR11. Each rule entered the ledger before its number:

| rule | subject | section |
| --- | --- | --- |
| HR1 | the seeds | 4.4 |
| HR2 | the bars, and the MS7f staged and stretch rows as report-only rows | 1, 8 |
| HR3 | the read protocol | 4.4 |
| HR4 | the oracle O, O_truth and the preflight | 4.2 |
| HR5 | equality with main c6d4981 with every new switch off | 4.1 to 4.3 |
| HR6 | the incumbent, h_B_scr, and h_O_scr as the reference of every screen | 4.6 |
| HR7 | Track R adoption, the one revision and the stop | 5 |
| HR8 | Track V and the week-2 rule | 7.7 |
| HR9 | the verdict: the candidate, its checked read and the official read | 8 |
| HR10 | the names of the reads and their tables | 11 |
| HR11 | the hosts | 4.4, 11 |

The protocol and the rules of MS7g carry over where the MS7h design does not replace them.

### 3.4 The deviations

| time | id | what changed | why |
| --- | --- | --- | --- |
| 16:00 | HD1 | O keeps its declines. A decline of a perceived target books the target failed, as any failed cycle does. | On the O preflight, 44 of 190 cycles were rig declines. The wave takeovers fell back to the part that the teacher ranks first from the truth (`any_part`, 477 of 1387 smoke takeovers). Design section 4.1 picks each target from the perceived scene. |
| 16:00 | HD2 | Pre-registered before any O read. If any cycle of an O read runs over 1600 steps, that read runs again at a commit that caps a takeover at 1600 steps. | On an O_truth smoke heap one takeover ran 15,971 steps. No O cycle of the preflight passed 538 steps. |
| 18:47 | HD3 | Pre-registered before h_S_scr's number. S replaces B as the base of Track R with at least h_B_scr's 2480 correct stows and a rate above 331.0. The chosen base's screen is the incumbent of HR6 and HR7. | The O reads put B 190 and 195 correct stows under O and 58 an hour slower. So the rate row binds as hard as the share. |
| 19:40 | HD4 | R screens run on 20 Machine A lanes, and the Machine B collection lanes keep running. The checked and official reads keep 10 + 10 lanes with the collection paused. | The 5080 cannot hold 15 collection servers and 10 read servers together. A collection lane stops only after its heap, about 40 minutes. |
| 20:25 | HD5 | The TD target is the mean of the 4 target critics, n = 10, gamma 0.998. The servers explore with AR(1) noise. Run 2 starts from scratch. | Run 1's critic diverged (section 5.2). |
| 21:08 | HD6 | Run 2's learner resumes on 8d66722 (CUDA graphs) with `--utd 2` (INITIAL). The restart drops the RAM replay once. | The learner ran about 192 updates a second against about 350 agent steps a second. |
| 21:28 | HD7 | Run 2's learner resumes at ee03c38 with `--actor_alpha 25`, ten times the old weight. A do-no-harm check follows after 500k collected steps. | At alpha 2.5 the prior held the actor at S (section 5.3). |
| 01:48 | HD8 | Run 2 stops, and run 3 replaces it: the chunked residual, the gap channel off, a time cost of 0.002 (INITIAL). | The 3M screen and the 3M ablation (section 5.4). |
| 03:05 | HD9 | Pre-registered before any V screen number. V's 30k screen must read at least h_S_scr - 30 = 2503 for residual RL on V in week 2. h_V10k_scr decides nothing. | HR8 names h_B_scr - 30. It was written before HD3 made S the base. The Track V session used 2503 since 20:31. |
| 03:27 | HD10 | Pre-registered before any run 3 screen number. The check reads the mean residual size of the last three log lines at 3M collected steps. If it is under 0.05 of the bound, run 3's learner resumes with `--actor_alpha 10`. A do-no-harm check follows after 500k collected steps at alpha 10. On a fail, the learner resumes with alpha 2.5. | At alpha 2.5 run 3's actor stays at 0.023 of the bound. In run 2 the harm of alpha 25 came from two causes. One is the gap channel, which is off in run 3. The other is slower cycles, which run 3's time cost of 0.002 now prices. |
| 09:31 | HD11 | Pre-registered before any V number at step 30k. V's 30k screen h_V30k_scr reads checkpoint 030000 with S's serving and `--pi05`, on the screen seed, draws 0 to 19. It runs on Machine A with two π0.5 servers of 10 lanes each, only while no R screen reads. It is cut at 4 h. HD9 decides on 20 complete heaps. When heaps are missing at the cut and the finished heaps' correct stows plus 150 for each missing heap stay under 2503, V fails HD9. Otherwise the missing draws run again before the decision. HD12 P4 amends it (section 7.7). | One V heap took 71 minutes on a shared server ([V] 09:31). |
| 09:38 | HD12 | The revised plan, P1 to P5 (section 2.4). | The owner's request of 09:32 and the findings F1 to F3. |
| 15:35 | HD13 | Before any number of it. Run 4's actor loss gets a scale floor, `--actor_scale_floor 0.6`, and `--actor_alpha 4`. The learner resumes from its state, and the lanes keep running. | h_R4_3M_scr read 101 under S (section 5.9). |

The tracks adopted five more deviations from the design. The ledger records each with its evidence:

| time | track | what changed | why |
| --- | --- | --- | --- |
| about 15:20 | learn | The critic reads `(action - B) / bound` instead of the raw action. The actor loss is the ensemble mean. | The same information at a scale that the network resolves. |
| about 15:35 | vla | The action quantiles of the π0.5 dataset are recomputed over all frames. | The corpus statistics average the quantiles of each episode (section 7.1). |
| about 16:30 | rl | A lost part needs z at most 0.05 m. A part stays at rest until it moves faster than 0.05 m/s. The end-of-episode regrade is unassigned, so Phi at the end equals the metric. | 108 of 111 first lost reads were parts in opening jaws. One jittering part made 444 of 777 status changes. |
| 19:00 | res | A conservative, regularized actor loss with the zero residual as the prior, and a critic warmup (f10ea78). | Smoke 3's actor degraded the base (section 4.3). |
| 21:30 | res | The actor's context uses the wrapped corpus statistics (`residual_context_stats.json`). | F5's own statistics hold the raw roll, with a standard deviation of 3.12. |

### 3.5 Two controller sessions

The owner branched the controller conversation. From about 17:40 a second session ran beside the
first one and wrote duplicate ledger lines. The first session removed them at 19:15 and kept a
backup, `h/tables/README.bak-1915`. The final split came at 19:25:

- The second session, the Track V session, owns Track V only: the owner's Colab run and the
  worktrees 7h-vla, 7h-shared, 7h-vread and 7h-vres. This session schedules V's screen.
- The Track V session uses at most 4 Machine A CPU processes and 10 GB of the 5090. It runs no Machine B or
  aarch64 jobs, uses ports 33500 to 33999 and appends only "[V]" lines to the ledger.
- The first session owns the rest of MS7h.

During the 1M screen of run 2, a π0.5 server of the Track V session held about 9.9 GB of the 5090.
14 of the 20 read servers failed on CUDA out-of-memory (22:01, section 5.3). From then on, the Track
V session starts no Machine A GPU job without a window from the first session (22:05).

The Track V session had the name maker-arm-lab-81 until 20:35 on 2026-10-05 and maker-arm-lab-92
after its restart. It is one session (03:05). At 09:25 the owner closed it ([V] 09:25). The first
session took over its work:

- the unit of h_V10k_scr, the Colab run and the fetch of the 30k checkpoint
- the branches `feat/ms7h-vla-*`
- the worktrees 7h-vla, 7h-shared, 7h-vread and 7h-vres

At 09:56 six lanes of h_R3_10M_scr died on CUDA out-of-memory in the policy server while agents
ran GPU tests (12:04). From then on, the agents run their GPU tests off the shared 5090 unless a
test needs the GPU.

## 4. Phase 0

Phase 0 builds what both tracks need (MS7h design section 5). It ran from 13:10 on 2026-10-05. The
design's schedule put it on 2026-10-06 to 2026-10-08 (MS7h design section 8). Six tracks started
from main c6d4981 at 13:25 in their own worktrees. The ledger names them guard, oracle, rl, learn,
res and vla.

### 4.1 The context-fix safeguards

MS7h design section 9.1 asks for three safeguards of the context fix. The guard track pushed
`feat/ms7h-context-guard` at about 15:10:

- 5d177c8: the policy server refuses a checkpoint whose configuration maps ENV to MEAN_STD while its
  marker has no `"context"` key.
- 89a8813: `train_masked.py --context_norm` writes the marker key `"context": "roll_wrapped_norm"`
  and merges it into the run's `action_space.json`. A resume of an ENV-mapped configuration needs
  `--context_norm`.

The lerobot_stack tests (170) and the repo tests (1679, 84 skipped) pass, and ruff is clean.

### 4.2 The oracle O in the perceived loop

Rule HR4 (MS7h design section 4.1). O runs the gate loop of HR3 with `--perceived` and the
classifier stack of S. The selector picks each target from the perceived scene. The rig drives every
cycle from the truth plan through the takeover entry of the DAgger waves. It places each part into
the box that the vote serves. O uses the teacher flags of wave 7, copied from
`g/tables/w7/bin/wave7_host.sh`: `--descend_speed 3.0 --lift_hold_steps 10 --release_steps 10
--landing_steps 15 --close_steps 25 --close_guard`. O_truth is the rig with the truth: no
`--perceived`, the true class, the refill scene and the same teacher flags.

The oracle track pushed `feat/ms7h-oracle-loop` 31298e8 at about 15:55:

- 417a6b1: the rig drives every perceived cycle and places into the voted box. The place leg follows
  the cycle's box on every step. While a vote is pending, the rig holds LIFT and EXTEND. The rig
  hands back at the end of RELEASE. A look-away moves an idle tool off the tray.
- d6f4d68: the O read mode with no policy server, in `run_gate_eval.py` and `fleet.py read
  --oracle_driver`.
- 31298e8: the docstrings of the waits and the declines.

The pin of rule HR5 passes: three policy draws give main's payload bytes. 1699 tests pass. The
smokes at K = 50 (50 parts a heap): O stows 42 of 50 at 386.8 an hour, and O_truth 50 of 50 at
433.7.

The preflight (rule HR4, run 20360000, layout 20360100, K = 150, Machine A):

| row | value |
| --- | --- |
| correct | 126 of 150, 0.840 |
| correct stows an hour | 380.5 |
| held-out row | 39 of 48, 0.8125 |
| cycles | 190, 1.508 a correct stow, against a budget of 195 |
| takeovers, stowed | 146, of which 114 stowed (0.781) |
| declines | 44 |
| taken class right | 110 of 117, 0.940 |

Wave 7 stowed 2122 of its 2313 takeovers, 0.917. The ledger puts the gap on three causes: the
`any_part` fallback of the waves, the perceived selection and the vote. On a K = 150 heap O_truth
stowed 146 of 147. The track found no defect. HD1 and HD2 went into the ledger at 16:00 (section
2.4).

The takeover cap (HD2). 7a1344f abandons a takeover with no policy at two cycle step budgets, 1600
steps. Such a cycle ends `takeover_capped`, booked like a decline. The O_truth smoke heap ran again
(run 20360012): 147 of 150 both ways, with 3 capped takeovers, at 386.0 an hour against 304.8. The
stuck states stay: the force cap held the arm about 4,800 steps. The O reads ran at 31298e8 without
the cap. Their longest cycles were 630 steps (h_O_scr) and 652 steps (h_O_chk), so HD2 did not fire.
h_Ot_scr ran at 7a1344f with the cap.

### 4.3 The RL plumbing and its smokes

The RL stream (rl track, `feat/ms7h-rl-stream` d246282, about 16:30). `--rl_stream` is observation
only. The loop computes Phi and its credit and sends them with a privileged vector of 90 values
(dbdae0d, `PRIV_DIM` 90). Phi is `correct - wrong - lost` (MS7h design section 6.3). A cycle's
reward is the change of Phi that the credit rule gives to the cycle, minus a time cost a step. If a
change comes within 300 steps after the release of the cycle that last held the part, it goes to
that cycle. Other changes go to the cycle in progress. An agent step is a step whose served action
came from the policy. A new truth read, `rl.critic_reads`, holds the finger contacts and the parked
parts. It joins `truth_reads`, and a test pins that no decision reads it.

The smoke of the stream (K = 50, run 20360010):

- Phi identity: 27 = 27 - 0 - 0. The end counts equal the metric: 37 correct, 10 wrong.
- 49 cycles: +1 on 36, 0 on 5, -1 on 7 and -2 on 1.
- Agent steps are 0.918 of all steps. The hooks cost 0.53 ms a step, 1.3 %.
- 24 new tests, and the full suite of 1706 passes.

The learner (learn track, `feat/ms7h-learner`, about 15:20). 8d1a185 adds `residual.py`: the actor,
the 4-critic ensemble and the actor file, with a header and the base's sha256 id. a1ed165 adds
`residual_learner.py`. On the 5090 with a full 2M replay, it runs about 300 updates a second at
batch 256. Beside 10 GB of other GPU jobs it runs 205. It uses 0.15 GB of GPU memory and 4.4 GB of
RAM. A synthetic feed of 360 steps a second for 240 s gave 270 to 325 updates a second at `--utd 1`.
The update-to-data ratio (UTD) is the number of critic updates per collected step.

The residual server (res track, `feat/ms7h-residual` 0247975, about 17:35). It sits on the rl branch
and cherry-picks the learner. cb680ce forwards `--residual` and ships the actor file with fleet.
7e818f1 is the server side. 0247975 flushes queued cycles at exit. 1713 repo tests and 201
lerobot_stack tests pass. The features and the actor cost 2.4 to 3.6 ms a step on the shared 5090.
They cost 1.7 ms on the 5080 and 7.3 ms on the A2000.

The three smokes of MS7h design section 5.3:

| smoke | what it shows | result |
| --- | --- | --- |
| 1, zero residual | A zero residual serves the base's actions. | Two separate runs diverge from cycle 0 (GPU and CPU float order). So a proxy fed every request of one live heap to plain B and to B with `--residual zero`. 14,069 of 14,069 replies are byte-equal, and 11,454 of 11,454 with `--rl_stream`. |
| 2, one lane | Transitions reach the learner with their rewards. | Run 20360011 with the real learner. The Phi identity 17 = 17 - 0 - 0 holds. Both sides count 65 cycles and 32,676 steps. |
| 3, two hosts | Machine B and aarch64 lanes stream to the learner for one hour. | 16:32 to 17:32, 5 Machine B and 5 aarch64 lanes, K = 150. See below. |

Smoke 3 in numbers:

- Machine B ran 143 env steps a second in all (26 to 33 a lane), and Machine C about 68 (12 to
  15 a lane).
- The learner received 203.5 agent steps a second, at 2,344 bytes a step and a mean of 477 KB/s.
- The learner made 712k updates (198 a second). The losses stayed finite. Every server loaded the
  weight versions 1 to 12.

The finding of smoke 3: the learned actor degraded B. Its actor v13 (745k steps) with no noise
stowed 15 of 50 on smoke seed i = 0. Plain B stowed 36 to 43 there, and B with noise 0.2 only
stowed 43. Collection heaps under the learning actor stowed 60 to 72 of 150. v13 was bang-bang: the
mean size of the residual as a fraction of its bound, |r|/b, read 0.895 (21:30). 78 % of its outputs
lay over 0.9 of the bound. So the res track changed the learner before the campaign: a conservative,
regularized actor loss with the zero residual as the prior, and a critic warmup (f10ea78, 19:00). A
live K = 50 smoke followed (`h/tables/res/bc_live`, actor start 500). The regularized actor stayed
at |r|/b 0.11 to 0.14, with 0.2 % of its outputs saturated.

### 4.4 The reads and the baseline

Rule HR1 fixes every seed before its use:

| use | seed |
| --- | --- |
| RL collection heaps, train split | run 20340000 + p, layout 20350000 + p, p = 0 to 9999 |
| the O preflight | run 20360000, layout 20360100 |
| smokes | run 20360010 + i, layout 20360110 + i, i = 0 to 9 |
| screens | 20307000, draws 0 to 19 |
| checked reads | 20307100, draws 0 to 19 |
| the official read | 20307200, draws 0 to 39, one shot, only after a checked pass |

Rule HR3, the read protocol. Every deciding read runs on x86 lanes only, 10 Machine A and 10 Machine B,
through `h/bin/sync_read_h.sh`. Its worktree is 7h-r<N>, and its aliases are `a_h<N>l` and
`b_h<N>`. The serving is S's: the `g/bin/read_g.sh` flags (the MS7f gate scene, `--action_warp
1.5 --missed_lift --cell_failure_cap 5 --tray_refill`) plus `--carry_guard --overhang_guard
--cls_flip --vote_weights g/tables/vote/vote_v10e3_flip_final.json`. The classifier is
`f/tables/cls/model_v10e3`. A read pauses the Machine B collection lanes while it runs.
Machine C collects RL data only. Every file that a switch names is copied to Machine B before the read. The
controller checks every read's `policy_id` set before its number enters the ledger. HD4 moves the R
screens to 20 Machine A lanes (section 3.4).

The baseline reads:

| time | TAG | who | SHA | correct of 3000 | correct stows an hour | held-out row | D against O, paired SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14:05 | h_B_scr | B, screen seed | c6d4981 | 2480, 0.8267 | 331.0 | 588 of 744, 0.7903 | -190, 31.2 |
| 14:46 | h_B_chk | B, checked seed, report-only | c6d4981 | 2490, 0.8300 | 330.7 | 612 of 785, 0.7796 | -195, 31.6 |
| about 16:55 | h_O_scr | O, screen seed | 31298e8 | 2670, 0.8900 | 388.9 | 640 of 744, 0.8602 | |
| 17:37 | h_O_chk | O, checked seed | 31298e8 | 2685, 0.8950 | 390.2 | 653 of 785, 0.8318 | |
| 18:10 | h_Ot_scr | O_truth, screen seed, report-only | 7a1344f | 2901, 0.9670 | 381.1 | 715 of 744, 0.9610 | |

The ledger gives the `policy_id` of every payload: `act:F9n 800k+nas50+warp1.5+ctx16+ctxnorm+gap`
for h_B_scr, `oracle:perceived:<wave 7 flags>` for h_O_scr and `oracle:truth:<wave 7 flags>` for
h_Ot_scr. O used 3516 cycles on the screen seed and 3452 on the checked seed. O_truth used 1.029
cycles a correct stow, and 23 of its takeovers hit the cap.

h_B_scr was the incumbent of HR6 at first, so the R candidate bar of HR7 was 2510 (14:05). Against
g_S_chk, S on the same checked draws in MS7g, h_B_chk reads D = -55 at a paired SE of 29.9.

B against O by mechanism (18:50, h_B_scr against h_O_scr, `h_O_scr_vs_B.mech.txt`):

| row | B | O |
| --- | --- | --- |
| wrong-box parts | 316 | 163 |
| riders | 151 | 45 |
| passengers | 84 | 5 |
| `wrong_box` cycles | 165 | 118 |
| unboxed parts | 204 | 167 |
| `close_miss` cycles | 445 | 311 |
| M_stall, steps | 170 | 0 |
| steps a correct stow | 326 | 277 |

A rider is a part in a wrong box without a `wrong_box` cycle of its own. A passenger is a part in a
wrong box that was never a target and never held (MS7g report section 4.1). M_stall is the summed
steps of air stalls while the hold detector holds a part.

### 4.5 The B drift check and the noise of a repeat

h_B_scr reads 2480, but g_F9n_800000_scr read 2554 in MS7g. That read had the same policy, the same
serving and the same draws. D = -74 at a paired SE of 29.0 (14:05). The controller looked for the
cause at 14:55:

- The client code is the same between the two reads. The server's ensemble refactor (c3c3ff2) keeps
  the one-model path. No client decision reads the clock.
- h_B_scr and g_F9n_800000_scr follow the same path on heap 0 for the first cycle, 115 steps in
  both. They part at step 275 of cycle 2 (276 against 275 steps). So the gap is the chaos of the
  scene after a float difference, not a changed rule.
- Both hosts moved the same way: Machine A 1233 against 1287, Machine B 1247 against 1267.
- The payload blocks (inference, context, scene, budgets) are equal. The checkpoint hash is equal on
  both hosts. The venvs did not change since 2026-10-03.
- The MS7g F9n screens around 800k read 2513 (600k) and 2490 (1M).

So 2554 was a high draw of a checkpoint that MS7g selected on this seed. The rule stays: h_B_scr is
the incumbent. A same-policy repeat on one seed differs by about one paired SE, 29. So the +30
screen margin of HR7 is a one-SE filter, and only the checked read against O decides.

### 4.6 The base of Track R: HD3

The O reads put B 190 and 195 correct stows under O and 58 an hour slower. S read 2545 to 2548 at
372 to 379 an hour in MS7g. HD3 (18:47) went into the ledger before h_S_scr's number. If h_S_scr
stows at least 2480 at a rate above 331.0, S replaces B as the base of Track R.

h_S_scr ran at c6d4981 from 18:48 to 19:26 (`h_S_scr.chain.log`). Its `policy_id` is
`act:F5 200k+nas50+warp1.5+ctx16+gap` on every payload. A rewrite of the ledger between 19:26 and
19:36 dropped its entry and the next one. The controller restored both at 03:05 from the backup
`h/tables/README.bak-0305`:

| read | correct of 3000 | correct stows an hour | held-out row | cycles | steps |
| --- | --- | --- | --- | --- | --- |
| h_S_scr | 2533, 0.8443 | 369.5 | 581 of 744, 0.7809 | 3238 | 740,435 |

Both conditions of HD3 hold, so S is the base of Track R and its screen is the incumbent. Against
h_B_scr, S reads D = +53 at a paired SE of 32.8, and 38.4 more correct stows an hour
(`h_S_scr.mech.txt`). Against h_O_scr, S reads D = -137 at a paired SE of 30.5. The R candidate bar
of HR7 is 2563 on the screen seed. For the V+R rule, see section 7.7.

h_S_chk (report-only) reads S on the checked seed. Its first launch at 01:50 failed at the ship,
because Machine B's F5 `pretrained_model` held a stray `action_space.json` from 18:58 on 2026-10-05. The
controller moved it aside, and the read ran again from 02:15 (unit ms7h-s-chk2). At 02:48:

| read | correct of 3000 | correct stows an hour | held-out row | D against h_O_chk, paired SE |
| --- | --- | --- | --- | --- |
| h_S_chk | 2525, 0.8417 | 366.0 | 621 of 785, 0.7911 | -160, 29.5 |

Its `policy_id` set is `act:F5 200k+nas50+warp1.5+ctx16+gap` on every payload. So the base stands
160 correct stows and 24 an hour under O on the verdict's seed.

### 4.7 Where S loses to O

The mechanism table of h_S_scr against h_O_scr (`h_S_scr_vs_O.mech.txt`), with the checked seed
beside it (`h_S_chk.mech.txt`):

| row | S, screen | O, screen | S, checked | O, checked |
| --- | --- | --- | --- | --- |
| correct stows | 2533 | 2670 | 2525 | 2685 |
| wrong-box parts | 311 | 163 | 305 | 176 |
| riders | 129 | 45 | 146 | 43 |
| passengers | 55 | 5 | 79 | 3 |
| `wrong_box` cycles | 182 | 118 | 159 | 133 |
| `wrong_box` cycles, views agreed (cycles / class wrong) | 90 / 79 | 46 / 46 | 74 / 68 | 55 / 55 |
| `wrong_box` cycles, split and voted (cycles / class wrong) | 87 / 79 | 57 / 57 | 76 / 69 | 66 / 66 |
| `wrong_box` cycles, no lift read | 3 / 2 | 12 / 12 | 1 / 1 | 11 / 11 |
| `close_miss` cycles | 457 | 311 | 443 | 268 |
| M_stall, steps | 752 | 0 | 496 | 0 |
| release over the tray | 101 | 135 | 111 | 117 |
| release over the right box | 2628 | 2364 | 2634 | 2402 |
| release over a wrong box | 236 | 111 | 211 | 122 |
| release over the desk | 34 | 0 | 34 | 1 |
| lifted, never opened | 30 | 4 | 20 | 3 |

The reading. The wrong box carries the gap. On the screen seed S puts 148 more parts in a wrong box
than O. That is more than the 137 correct stows between them. Riders give 84 of the 148, and
passengers give 50 of the riders. O carries 5 passengers on the screen seed and 3 on the checked
seed. S also has more wrong-box cycles whose views agreed on a wrong class (79 against 46). Every
wrong-box cycle of O has a wrong class. Of S's 90 agreed wrong-box cycles, 79 have a wrong class. S
releases over the right box more often than O (2628 against 2364), and over a wrong box more often
(236 against 111). O releases over the tray more often (135 against 101).

### 4.8 Where the parts go: the loss breakdown

At 09:49 the controller followed every part of five reads to its final class
(`h/tables/loss/tables.md` and `loss.py`). The reads are h_O_scr, h_S_scr, h_R3_3M_scr, h_O_chk and
h_S_chk. Every read sums to 3000 parts. The wrong-box ids match `wrong_box_parts` in all five reads.
In each read 11 to 20 parts sit in their own box but still move at the last step. The grader does
not count them. So the breakdown counts 2681 correct parts for h_O_scr and 2546 for h_S_scr,
where the grader counts 2670 and 2533.

The rollups of `tables.md`:

| final class | h_O_scr | h_S_scr | S - O, screen | h_O_chk | h_S_chk | S - O, checked |
| --- | --- | --- | --- | --- | --- | --- |
| correct | 2681 | 2546 | -135 | 2697 | 2545 | -152 |
| wrong box | 163 | 311 | +148 | 176 | 305 | +129 |
| wrong box, passenger | 5 | 55 | +50 | 3 | 79 | +76 |
| wrong box, touched rider | 40 | 74 | +34 | 40 | 67 | +27 |
| wrong box, own cycle held a non-target | 2 | 24 | +22 | 4 | 29 | +25 |
| wrong box, own cycle, class right | 0 | 21 | +21 | 0 | 13 | +13 |
| unboxed | 156 | 143 | -13 | 127 | 150 | +23 |
| unboxed, desk beside a box | 30 | 60 | +30 | 13 | 52 | +39 |
| unboxed, tray interior | 113 | 43 | -70 | 98 | 48 | -50 |

The reading (09:49):

- On the screen seed S has 135 fewer correct parts than O. The wrong box gives +148, and the
  unboxed parts give -13. The tray interior gives -70, because O hits the failure cap on 58 tray
  parts that S picks.
- Section 2.3 lists the five largest losses of S. Three of them come from the pick. 59 of the 74
  touched riders in a wrong box were an earlier target that the policy did not keep.
- The parts on the desk beside a box come from the release. O has 39 bounce-outs and S 56
  (`M_bounce`).
- A match with O on every class that S loses gives S 205 more parts and puts it at O + 70.
- So the pick, not the carry, holds most of the gap. The hybrid of HD12 P1 can recover at most the
  release and carry classes.

The same table puts h_R3_3M_scr 7 correct parts over S. It has 23 more wrong-box passengers (78
against 55) and 23 fewer wrong-box parts from cycles whose views agreed (67 against 90).

## 5. Track R: residual RL on the frozen base

### 5.1 The residual as built

MS7h design section 6 and the deviations of section 3.4 define the method:

- The actor reads only what the real arm has: the features of both cameras from the base's frozen
  ResNet18 (512 each), the state (7), the context (16) and the base's action (7). That is 1054
  values. Its output corrects the six joint targets within 0.05 rad and the jaw gap within 0.010 m.
  Its last layer starts at zero, so the actor starts as the base.
- The critic is an ensemble of four Q networks with LayerNorm. Each reads the actor's input, the
  served action as `(action - B) / bound` and the privileged vector of 90 values.
- The learner follows ResFiT: TD3 with the ensemble and n-step returns. A TD target is the value
  that a critic learns to predict: the next n rewards plus the discounted value after them. The
  learner runs on Machine A 5090 and publishes the actor weights. The servers load new weights
  between cycles.
- The actor loss is conservative and regularized (f10ea78). It weighs the critics' value by alpha
  and keeps a pessimism term (HD5). A prior pulls the residual to zero. Alpha is the weight of the
  critics' value against that prior.
- One cycle is one episode. The reward of a cycle is the credited change of Phi minus c a step
  (section 4.3). The design's c is 0.0005.
- Collection runs on 15 Machine B lanes and 10 aarch64 lanes on the RL collection heaps, with exploration
  noise. A do-no-harm check reads the actor with no noise against the plain base. It uses the smoke
  seeds i = 0 to 3 at K = 50. The bar is -20 correct stows.

Rule HR7 (MS7h design section 6.5). Screens of the deterministic actor run at 1M, 3M, 10M and 30M
collected steps, then every 30M. A checkpoint with a screen of at least the incumbent + 30 is a
candidate. With S as the base, the bar is 2563. If three days of collection give no candidate, one
of three revisions runs. (a) sets the wrong and lost weights to 0.5 and c to 0. (b) halves or
doubles the residual bounds. (c) takes half of each batch from the corpus takeover frames. The
controller picks it by the mechanism numbers and writes the reason into the ledger before its first
number. If the revised run gives no candidate by 2026-10-16, Track R stops. HD8 counts the three
days from run 2's start, so a candidate is due by 2026-10-08 20:29. HD12 P2 replaced the revision
with run 4, because (a) to (c) do not touch the finding of section 5.7.

### 5.2 Run 1: the critic diverges

Run 1 started at 19:36 as the pilot. It used the base S, f10ea78 and the learner on Machine A
(`h/col/learner`, ports 32900 and 32901). The actor updates started at 300k collected steps. The
collection ran on `b_hc` (15 lanes) and `c_hc` (10 lanes) into `h/col/heaps`, from p = 0, with
noise 0.2. At 4 minutes the learner received 430 agent steps a second and made 407 updates a second.
The mean cycle reward was 0.63.

The critic diverged (20:25, `h/col/learner/learner.jsonl`):

| collected steps | q_mean |
| --- | --- |
| 89k | -0.18 |
| 240k, before the first actor update | -1.5 |
| 614k | -10.6 |
| 942k | -7.2 |

The critic loss rose from 0.004 to 1.3. A cycle's reward lies near -1 to +1, so the true Q cannot
reach -10. The cause: the TD target is the minimum over a random pair of the 4 target critics (MS7h
design section 6.2). Each bootstrap adds a negative bias. At n = 3 and gamma 0.998, the bias
compounds over about 1 / (1 - 0.998^3) = 167 bootstraps. The actor stayed at |r|/b 0.014, because
the prior dominates. So it did no harm, and it learned nothing. Its do-no-harm check passed before
the stop: actor v5 164 against S 174 (bar -20). Run 1 stopped at 20:28 with LEARNER_OK at 1,128,130
collected steps. The screen chain stopped before the 1M actor landed.

HD5 (20:25, before run 2's numbers). The TD target is the mean of the 4 target critics, and the
pessimism stays in the actor loss only. n = 10 and gamma 0.998. The servers explore with AR(1) noise
(rho 0.9, sigma 0.2 x the bound, reset at each cycle). AR(1) noise is correlated in time: each step
keeps 0.9 of the last step's value and adds a fresh draw. The reason: the arm's tracking filters
white noise out. The learner logs the batch's Monte Carlo (MC) return beside q_mean. The MC return
is the discounted reward that the batch's steps actually received.

### 5.3 Run 2: the per-step residual

Run 2 started at 20:29 from scratch (unit ms7h-learner-2, `h/col/learner2`). It used the base S, 15
Machine B lanes and 10 aarch64 lanes. The branch tip was 8841f87: d9337cb the mean target with n = 10,
and 8841f87 the AR(1) noise.

The learner speed (HD6). The learner made about 192 updates a second against about 350 agent steps a
second, a UTD of about 0.55 (20:40). The speed track measured on the shared 5090 (21:00):

| batch | updates a second before | after, CUDA graphs |
| --- | --- | --- |
| 256 | 265 | 632 |
| 512 | 244 | 574 |
| 1024 | 225 | 351 |

The replay is saved at a clean stop and loaded on `--resume`: 4.67 GB for 2M transitions, 2.8 s to
write and 2 to 3 s to load. At 21:05 the learner unit got a stop from outside this controller.
SIGTERM did not end it in 90 s, and systemd sent SIGKILL at 21:07:01. So no state was written at the
stop. The last periodic save (21:04:29) held 754,742 collected steps. The learner resumed at 21:10
as unit ms7h-learner-2b on 8d66722, with `--resume --utd 2` and an empty replay. At 21:04 run 2 read
q_mean 0.374 against an MC return of 0.371 and a critic loss of 0.0003. |r|/b read 0.017, and the
mean cycle reward 0.57.

The res track's report on run 2 (21:30):

| collected steps | q_mean | MC return |
| --- | --- | --- |
| 305,947 | 0.389 | 0.392 |
| 601,072 | 0.382 | 0.375 |
| 1,139,587, under 2b | 0.344 | 0.337 |

The critic loss stayed at 0.0002 to 0.0003, and |r|/b at 0.016, with no saturated outputs. Over 24
minutes before 2b, Machine B sent 284 agent steps a second (19 a lane) and Machine C 92 (9 a
lane): 376 in all, at 0.88 MB/s. The do-no-harm check passed. Actor v6 (882,976 collected steps)
stowed 39 + 42 + 44 + 46 = 171. Plain S stowed 41 + 40 + 43 + 43 = 167. Under 2b the learner ran
about 720 updates a second at first and 395 at 21:25, against 362 agent steps a second. The R screen
shares the GPU, so `--utd 2` was not met, and the owed updates grew to 193,507.

Alpha holds the actor at S (HD7, 21:28). At alpha 2.5 run 2's actor stayed at |r|/b 0.014 to 0.017
from 0.3M to 1.3M collected steps. At that point the critics' gradient is about 0.0006 of Q per full
bound, and the prior holds the actor at S. Learner 2b stopped cleanly with LEARNER_OK: a replay of
455,954 transitions saved in 0.8 s, version 11, 1,210,696 collected steps. ee03c38 adds
`--actor_alpha`. Unit ms7h-learner-2c resumed with `--utd 2 --actor_alpha 25`. The aim was |r|/b
near the exploration noise of 0.2.

The do-no-harm check of HD7 (22:13). The actor `dnh_c.pt` is ee03c38 at alpha 25 and about 1.71M
collected steps (sha256 229dd864fd53). It stowed 43 + 34 + 38 + 42 = 157. Plain S stowed 42 + 40 +
45 + 39 = 166. The margin is -9, inside the bar of -20, so it passed. The actor moved to |r|/b 0.12
to 0.14. Joint 4 moved most, at 0.14 to 0.17, and 0.1 to 1.5 % of the outputs saturated. Two watch
items followed:

- q_mean now exceeded the batch's MC return: 0.344 against 0.299 at 2.10M. At alpha 2.5 the gap was
  under 0.01. The ledger set a rule: if that gap passes 0.10 on three log lines in a row, the
  learner resumes with alpha 2.5.
- The mean reward per collected cycle drifted 0.561 (0 to 0.3M), 0.545, 0.500, 0.497, 0.434 and
  0.481 (1.8M to 2.2M). All lanes started their heaps together, so this mixes in the heap phase.

The 1M screen (22:01). h_R2_1M_scr ended READ_FAIL. 14 of its 20 Machine A lanes died at server start
on CUDA out-of-memory. The 5090 held the learner (about 1 GB) and the Track V session's
`pi05_server.py` (about 9.9 GB) beside the read servers (about 0.9 GB each). 6 heaps ran. The 1M
actor (alpha 2.5, |r|/b 0.016) is close to S, so the 1M screen was not read again. A 20-lane Machine A
screen needs about 19 GB of the 5090 beside the learner.

### 5.4 The 3M screen and the ablation

h_R2_3M_scr (01:27) reads run 2's actor at 3,000,156 collected steps, with alpha 25 since 1.21M
(file sha256 e88c7e3a1a57). It ran at 8841f87 on 7h-r3 with 20 Machine A lanes. Every one of the 20
payloads carries `policy_id` `act:F5 200k+...+res:e88c7e3a1a57`.

| read | correct of 3000 | correct stows an hour | held-out row | D against S, paired SE | D against O, paired SE |
| --- | --- | --- | --- | --- | --- |
| h_R2_3M_scr | 2490, 0.8300 | 314.4 | 588 of 744, 0.7903 | -43, 29.4 | -180, 33.1 |
| h_S_scr | 2533, 0.8443 | 369.5 | 581 of 744, 0.7809 | | -137 |

It is not a candidate (HR7 bar 2563). Its rate is 55.1 an hour under S. The mechanisms against S
(`h_R2_3M_vs_S.mech.txt`):

| row | 3M actor | S |
| --- | --- | --- |
| steps a cycle | 260.6 | 228.7 |
| releases over the tray | 140 | 101 |
| passengers | 83 | 55 |
| riders | 154 | 129 |
| guard steps | 9435 | 5744 |
| M_stall, steps | 60 | 752 |

The screen took 2.5 h, because the per-step features of 20 servers and the learner share the 5090.

The ablation (01:31, report-only). The gap residual loosens a hold. That is a likely cause of the
slower cycles and of the drops over the tray. The controller zeroed the gap output of the 3M actor
(`h/col/screens/R2_actor_3000000_nogap.pt`, row 6 of the last layer set to 0). Three actors ran on
six paired smoke heaps, i = 0 to 5, at K = 50. The 18 runs ran at once on Machine A
(`h/bin/abl.sh`, `h/tables/res/abl_3m`). The result (01:46):

| actor | correct stows on heaps 0 to 5 | sum | steps | steps a correct stow |
| --- | --- | --- | --- | --- |
| plain S | 41, 44, 44, 45, 42, 41 | 257 | 74,367 | 289.4 |
| the full 3M actor | 41, 36, 42, 37, 42, 38 | 236 | 87,962 | 372.7 |
| the 3M actor, gap output zeroed | 42, 41, 45, 44, 43, 46 | 261 | 82,423 | 315.8 |

So the jaw-gap channel causes the share loss, and the joint residual still slows the cycles (HD8).

### 5.5 HD8 and run 3: the chunked residual

Why a chunk (01:36). One step's action barely moves a cycle of about 230 steps. So the per-step
critic has almost no action signal: 0.0006 of Q per full bound at alpha 2.5. A larger alpha follows
the critic's errors, as the 3M screen shows. If the residual holds for K = 10 steps, a cycle has
about 23 decisions.

Why a larger time cost (HD8). A cycle stows about 0.85 correct parts in about 230 steps, so the
long-run worth of a step is near 0.0037. At c = 0.0005 the critic values a 10 % slower cycle at 1.2
% of a stow. The actor slowed the policy at no visible cost. At c = 0.002 a 10 % slower cycle costs
about 5 % of a stow.

The chunk track pushed `feat/ms7h-residual-chunk` 6b10b74 at about 02:00, from ee03c38:

- 348d5b8: the actor file records the corrected columns and the hold.
- 4d1934c: the server's `--residual_hold` and `--residual_dims`. The server holds the residual and
  its noise for K steps and pushes one row a chunk. `--residual_dims 6` turns the jaw-gap channel
  off.
- afc03f5: the learner trains on one row per chunk. The reward of a row is -c x steps, plus the
  change of Phi on the last row. The bootstrap uses gamma ** steps.
- 86af792: the supervisor forwards both flags.
- 6b10b74: the fix of the SIGTERM deadlock. A re-entered handler waited on the lock of
  `threading.Event.set`. The handler now takes no lock. On the 5090 the fixed learner ended in 4.8 s
  with its state and a replay of 4.67 GB.

225 lerobot_stack tests and 1714 repo tests pass. The smoke (K = 50, run 20360013):

- The Phi identity 35 = 35 - 0 - 0 holds.
- Both sides count 1,424 rows for 14,048 agent steps.
- The served gap equals the base's on all 14,770 steps.
- The residual is constant inside each of 1,499 chunks.

Run 2 stopped at 01:47. The screen chain stopped with no 10M screen. The collection lanes stopped
(MODE=hard) at about 6.5M collected steps. The learner unit got SIGTERM and hung again, as at 21:05.
At 01:53 its process still waited in `futex_do_wait` with 9 threads and held the ports 32900
and 32901. A SIGKILL by hand freed them.

Run 3 started at 02:16 (HD8), from scratch in `h/col/learner3`. The worktree 7h-col is at 6b10b74 on
all three hosts.

| item | value |
| --- | --- |
| base | S |
| learner flags | `--residual_hold 10 --residual_dims 6 --n_step 3 --time_cost 0.002 --utd 8 --actor_alpha 2.5`, the mean target |
| n and gamma | 3 rows, gamma 0.998 a step |
| UTD | `--utd` counts rows, so 8 is about 0.8 updates an agent step |
| lanes | 10 aarch64 lanes from 02:16, 15 Machine B lanes from 02:49, after h_S_chk |
| screens | unit ms7h-screen-r3, tags h_R3_<M>M_scr, 7h-r3 at 6b10b74, 20 Machine A lanes |
| do-no-harm check | the actor at 800k collected steps, about 500k after the actor start, bar -20 |

At 02:22 run 3 had 20,048 collected steps and 2,036 rows. The aarch64 lanes alone sent 108.7 agent
steps a second. At 02:55 all 25 lanes sent 494 to 557 agent steps a second (`h/col/learner3`). The
actor runs only at chunk starts, so a lane runs faster than in run 2. The learner made 400 to 450
updates a second at `--utd 8`. q_mean read 0.192 against an MC return of 0.195 at 296,579 collected
steps, and 0.220 against 0.213 at 355,868. The critic loss read 0.0005. After 22,619 actor updates
|r|/b read 0.038: 0.040 to 0.050 on the joints and 0 on the gap.

### 5.6 The results of run 3

The do-no-harm check (03:26, `h/tables/res/dnh/r3a`). The actor `dnh_r3a.pt` (6b10b74, sha256
66a4c9b98042) at about 800k collected steps served with the chunk flags and no noise. On the smoke
seeds i = 0 to 3 at K = 50 it stowed 44 + 41 + 43 + 37 = 165. Plain S stowed 40 + 40 + 40 + 38 =
158. The margin is +7, inside the bar of -20, so it passed. At 1.12M collected steps q_mean read
0.197 against an MC return of 0.195, and |r|/b read 0.023. The lanes sent 407 to 470 agent steps a
second, and the learner made 334 to 381 updates a second.

HD10 (section 3.4). Unit ms7h-hd10 (`h/bin/hd10.sh`) ran the rule from 03:29. It fired at 04:49:
run 3's mean |r|/b at 3M was 0.0184. The learner stopped cleanly (LEARNER_OK, a replay of 309,997
rows saved, 3,043,059 collected steps). It resumed as unit ms7h-learner-3b with `--actor_alpha 10`.
Its do-no-harm check passed at 06:45 (`h/tables/res/dnh/r3b`). The actor `dnh_r3b.pt` (sha256
31640b0ecb0f, alpha 10) stowed 38 + 46 + 43 + 46 = 173. Plain S stowed 42 + 41 + 45 + 43 = 171.
The margin is +2. At 5.67M collected steps q_mean read 0.215 against an MC return of 0.211. |r|/b
read 0.065: 0.07 to 0.09 on the joints and 0 on the gap. Alpha 10 stayed.

The screens. Each ran at 6b10b74 on 7h-r3 with 20 Machine A lanes. All 20 payloads of each read
carry the `policy_id` `act:F5 200k+...+res:<sha256>+hold10+dims6`.

| read | actor | correct of 3000 | correct stows an hour | held-out row | steps a cycle | D against S, paired SE | D against O, paired SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h_R3_1M_scr, 04:48 | 1M, alpha 2.5, abe34f9370f1 | 2536, 0.8453 | 389.2 | 599 of 744, 0.8051 | 224.6 | +3, 24.7 | -134, 22.8 |
| h_R3_3M_scr, 06:35 | 3M, alpha 2.5, 5ffed88f2698 | 2539, 0.8463 | 378.2 | 597 of 744, 0.8024 | 224.4 | +6, 19.6 | -131, 29.4 |
| h_S_scr | S | 2533, 0.8443 | 369.5 | 581 of 744, 0.7809 | 228.7 | | -137, 30.5 |

Neither screen is a candidate (HR7 bar 2563). Both read a rate and a held-out row above S's. The
mechanisms against S (`h_R3_1M_scr.mech.txt`, `h_R3_3M_scr.mech.txt`):

| row | 1M actor | 3M actor | S |
| --- | --- | --- | --- |
| passengers | 85 | 78 | 55 |
| riders | 144 | 142 | 129 |
| wrong-box parts | 328 | 306 | 311 |
| `wrong_box` cycles | 184 | 164 | 182 |
| `close_miss` cycles | 417 | 462 | 457 |
| air stalls | 63 | 108 | 83 |
| releases over the tray | 75 | 97 | 101 |

So the share loss and the slow cycles of run 2 are gone. The passengers rose in both screens.

h_R3_10M_scr. It started at 09:54 with the actor at 10M collected steps (sha256 dea8e329845e, alpha
10). At 09:56 six lanes (draws 4, 6, 10, 13, 16 and 19) died on CUDA out-of-memory in the policy
server while agents ran GPU tests. 14 heaps finished, so the pairing failed (12:04). The 14 payloads
are backed up in `h/tables/h_R3_10M_scr.partial14.bak`. The first fill (unit ms7h-r3-fill) failed at
15:15, because `fleet.py` reads a TAG once and counts no cached payload as a read. Unit
ms7h-r3-fill2 reads the six dead draws as h_R3_10M_fill with the same actor and serving on 6 Machine A
lanes from 15:36. Then h_R3_10M_all (report-only) holds the 14 + 6 payloads for the pairing and the
mechanism table (15:38). `<<R3_10M_SCR>>`

The stop (HD12 P3). Run 3 collected until run 4 started. At 11:41 its lanes stopped hard. The
learner ended with LEARNER_OK at 12,452,976 collected steps, 54,344 cycles and version 110. Its
state stays in `h/col/learner3`.

### 5.7 The critic diagnostic of run 3

At 09:38 a read-only diagnostic asked whether run 3's critic gives the actor an action signal
(`h/tables/res/qdiag/report.md`). It loaded run 3's learner state at 9,437,401 collected steps and
its saved replay of 309,997 chunk rows, from 0 to 3,043,059 collected steps. The critic trained on
every one of these rows. On 4096 sampled rows it read Q at 59 actions each: zero, the actor's
output, the stored action, probes of each joint and 32 uniform residuals.

The answer is no:

| measure | value | reference |
| --- | --- | --- |
| change of Q over the whole action box, median | 0.0195 | |
| spread of the four critics at one action (ensemble std), median | 0.0194 | |
| `Q(actor) - Q(0)`, median | 2.5e-4 | |
| the critics' disagreement on that gap (ensemble std), median | 8.9e-4 | |
| size of dQ/du at the actor, per full bound, median by joint | 0.0023 to 0.0026 | |
| the critics' spread of dQ/du, median by joint | 0.0061 to 0.0065 | |
| size of dQ/du over its spread, median by joint | 0.40 to 0.43 | 0.38 for a true zero slope |
| share of states where all four critics agree on the sign of dQ/du, by joint | 12.6 to 14.8 % | 12.5 % by chance |
| share of the action dependence that the critics have in common (ICC), median over states | 0.003 | 0 for independent critics, 1 for identical ones |

The root cause is memorization. On the rows it trained on, the critic predicts the MC return with a
correlation of 1.000 at an RMSE of 0.018. On run 2's rows, which it never saw, the correlation is
0.51 and the RMSE 0.615, against 0.655 for a constant. At the first row of a cycle that ends with
Phi = -1, about 170 steps before the end, the critic says -1.004. On held-out first steps of such
cycles it says 0.118. Each replay row holds exactly one action. So a critic that fits every row's
return does not need the action input. The data then does not constrain the slope of Q along the
action.

At alpha 10 the actor's Q term is about 5 times the pull of the prior: 0.055 to 0.062 against 0.009
to 0.013 a sample (median). The Q term wins on 85 to 88 % of states. So the prior does not hold the
actor at the base, and the actor follows a gradient that is mostly the noise of the members. The
ledger concludes that runs 1 to 3 had no learning signal, and that screens level with S are what
that predicts (HD12 F2).

### 5.8 The offline critic test and the data-only slopes

At 10:44 an offline test asked which critic change stops the memorization
(`h/tables/res/ctest/tables.md`). It split run 3's replay by cycle and held a third out: 4,413
cycles and 101,720 rows. A constant scores a held-out RMSE of 0.652.

- At 2 or more updates a row every variant memorizes: the control, no neighbour block, no privileged
  vector, weight decay, dropout, noise on the privileged vector and a 2 x 256 critic. The train RMSE
  reads 0.010 to 0.081, and the held-out RMSE 0.652 to 0.678.
- Only the update budget matters. At 0.05 updates a row the 3 x 512 critic reads a held-out RMSE of
  0.540 (correlation 0.58), and the 2 x 256 critic 0.538. Run 3's live critic reads 0.615 on run 2's
  rows. The critics of this test read 0.570 to 0.576 there.
- No variant's dQ/du matches the data. The within-state calibration reads -0.43 to +0.28. A right
  critic reads 1.

The data alone (`dataslope.md`: no critic, a cluster SE by cycle, the confounders partialled out):

- Before the hold, the return rises with the residual on j2 (+0.040 +- 0.008 per bound unit), j3
  (+0.030 +- 0.007) and j1 (+0.019 +- 0.008).
- Noise costs return: -0.024 +- 0.008 per unit of the sum of u^2.
- So the local gain around S is small. A fitted sustained shift is worth about +0.03 return a
  cycle, about +25 correct stows a screen. That is inside one screen SE.
- A slope of 0.0025 needs about 8M rows at noise 0.2 to resolve at 2 SE, or 1.2 to 1.5M rows at
  noise 0.5.

### 5.9 Run 4: the critic, the noise and the pick terms (HD12 P2)

The choices (10:44, before any run 4 number):

- a critic ensemble of 4 members, 2 x 256 with LayerNorm
- about 0.05 critic updates a row
- 10 % of the cycles held out, with their MC RMSE logged. A rise of that RMSE pauses the critic.
- exploration noise of 0.5 x the bound
- each cycle pushes its per-row noise and its actor version

The actor's alpha and the weights of the pick terms waited for their code.

The pick terms (10:58, `feat/ms7h-pick-reward` 7984643 on 6b10b74: bdfb24b the protocol, 14e1818
the loop, 7984643 the server and the learner). They read the truth and enter the reward only:

- passenger: +1 for each part other than the target and the held parts that rises 0.05 m above its
  selection height within 30 steps of the first lift
- nontarget: 1 when the census's lift part is not the target
- sweep: the summed displacement in m of the 8 live parts nearest the target since the last agent
  step. It counts only while the pick point is in the tray region, and it is capped at 0.02 m a
  step.

With the flags off, the stream, the pushed frames and the learner's training equal 6b10b74's bit for
bit. A pin holds this (`tests/data/ms7h_rl_pick_pin_6b10b74.json`). 1715 repo tests and 235
lerobot_stack tests pass. The smoke (2 heaps at K = 50, `h/tables/pick/smoke/table.md`) pushed 106
cycles, and the learner received them. The passenger term fired on 8 cycles. The nontarget term
fired on 7 of the 14 census non-target cycles and on 0 of the other 92. The sweep a cycle had a
median of 0.023 m and a p90 of 0.18 m. The weights are 0.5 for passengers, 0.5 for a non-target
and 1.0 a metre for the sweep. So a p90 cycle gets about -0.18, under the eventual -1 of a
wrong-box part.

The launch (11:41). `feat/ms7h-run4` 7fd4f0e merges `feat/ms7h-learner4` 741ca1a with
`feat/ms7h-pick-reward` 7984643. 1716 repo tests and 250 lerobot_stack tests pass, and ruff is
clean. The controller synced 7h-col on Machine A, Machine B and Machine C one host at a time.
Two bundle hosts in one fleet sync race on `sync.bundle`.

| item | value |
| --- | --- |
| base | S (F5 200k) |
| learner | fresh, `h/col/learner4`, unit ms7h-learner-4 |
| learner flags | `--residual_hold 10 --residual_dims 6 --n_step 3 --time_cost 0.002 --actor_alpha 10 --utd 0.05 --critic_hidden 256 --critic_depth 2 --holdout 0.1 --store_noise --pause_window 10 --pause_margin 0.05 --w_passenger 0.5 --w_nontarget 0.5 --w_sweep 1.0` |
| lanes | 15 Machine B and 10 aarch64 lanes with `HOLD=10 DIMS=6 NOISE=0.5 PUSH_NOISE=1 EXTRA=--rl_pick` (`h/bin/col_start4.sh`, `collect_host4.sh`) |
| screens | the deterministic actor at 3M, 10M and 30M collected steps (unit ms7h-screen-r4, `h/bin/screen_r4.sh`) |

The learner writes actors at 1M, 3M, 10M and 30M. So the screens run at 3M, 10M and 30M, not at 20M
as the launch line first said. The 10M screen is the first with about 1M chunk rows, the size that
the offline test asked for at noise 0.5.

h_R4_3M_scr (15:35). It read run 4's actor at 3M collected steps (sha256 a258f0dfdc99) at 7fd4f0e
on 7h-r3, with 20 Machine A lanes, from 13:37 to 15:14:

| read | correct of 3000 | correct stows an hour | held-out row | D against S, paired SE | D against O, paired SE |
| --- | --- | --- | --- | --- | --- |
| h_R4_3M_scr | 2432, 0.8107 | 312.3 | 567 of 744, 0.7621 | -101, 28.0 | -238, 27.9 |

The mechanisms against S:

- passengers 119 against 55, and riders 192 against 129
- wrong-box parts 377 against 311
- releases over the desk 48 against 34
- air stalls 1 against 83

The cause (15:35). TD3+BC divides the Q term of the actor loss by mean|Q|. Run 4's returns sit near
0: q_mean read 0.002 at 5.9M collected steps, against run 3's 0.209. So alpha 10 acted about 100
times run 3's. actor_prior read 0.176, so the deterministic actor served about 0.45 of the bound
RMS. Run 3's 0.011 means about 0.11.

HD13 (15:35, before any number of it). `--actor_scale_floor 0.6` puts a floor under that scale
(396a206 on `feat/ms7h-run4`, off by default). Run 3's measured scale was 0.62. `--actor_alpha 4`
aims at about 0.15 to 0.2 of the bound RMS. That lies inside the range where run 3's data shows a
gain: its fitted shift on j2 is 0.4 to 0.8 of the bound. The learner resumes from its state, with
the critic, the replay and the counters kept. The lanes keep running. The 10M and 30M screens follow
as planned.

`<<R4_10M_SCR>>`

`<<R4_30M_SCR>>`

### 5.10 The Track R screens

All screens on seed 20307000, draws 0 to 19, against h_S_scr (the incumbent) and h_O_scr:

| TAG | run, collected steps | actor sha256 | correct of 3000 | correct stows an hour | held-out row | D against S | D against O | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h_R2_1M_scr | run 2, 1M | faeb6f53e654 | READ_FAIL | | | | | 14 of 20 lanes out of GPU memory |
| h_R2_3M_scr | run 2, 3M | e88c7e3a1a57 | 2490, 0.8300 | 314.4 | 588 of 744, 0.7903 | -43 | -180 | not a candidate |
| h_R3_1M_scr | run 3, 1M, alpha 2.5 | abe34f9370f1 | 2536, 0.8453 | 389.2 | 599 of 744, 0.8051 | +3 | -134 | not a candidate |
| h_R3_3M_scr | run 3, 3M, alpha 2.5 | 5ffed88f2698 | 2539, 0.8463 | 378.2 | 597 of 744, 0.8024 | +6 | -131 | not a candidate |
| h_R3_10M_scr | run 3, 10M, alpha 10 | dea8e329845e | `<<R3_10M_SCR>>` | | | | | |
| h_R4_3M_scr | run 4, 3M, alpha 10 | a258f0dfdc99 | 2432, 0.8107 | 312.3 | 567 of 744, 0.7621 | -101 | -238 | not a candidate |
| h_R4_10M_scr | run 4, 10M, HD13 | | `<<R4_10M_SCR>>` | | | | | |
| h_R4_30M_scr | run 4, 30M, HD13 | | `<<R4_30M_SCR>>` | | | | | |

## 6. The hybrid: a learned pick and a scripted carry

### 6.1 The switch

HD12 P1 (09:38). Under `--rig_place` the policy drives from the selection to the lift as S does.
Then the rig's perceived carry carries the part to the voted box and places it. That carry is the
carry guard's handoff path with O's teacher motion. The rig reads only the perceived state, so the
hybrid stays within H1.

`feat/ms7h-rig-place` 0b91d7b (10:54) holds two commits: 8880381 the switch, and 0b91d7b the rig
hands back at the end of RELEASE.

- The handoff fires on every lift once the lift's class read ran, no vote is pending and the
  step's request is not sent. With the final-only vote the policy drives the 15 carry-view steps.
  Then the rig plans EXTEND to the voted box.
- The census gets the field `rig_place`. The switch needs `--perceived` and `--carry_guard`, and it
  refuses `--passenger_drop`.
- A read adds `--rig_place` and the wave 7 teacher flags. Only `--release_steps` acts on the rig's
  carry.
- The pin of HR5 passes. 1721 repo tests and 225 lerobot_stack tests pass.

### 6.2 The smoke

The smoke is report-only (10:54, `h/tables/place/`): 4 heaps at K = 50, the smoke seeds i = 0 to 3,
F5 200k.

| row | S | S with `--rig_place` |
| --- | --- | --- |
| correct | 160 | 156 |
| correct stows an hour | 350.4 | 315.0 |
| held-out row | 27 of 40 | 30 of 40 |
| `wrong_box` cycles | 16 | 9 |
| riders | 8 | 16 |
| `close_miss` cycles | 18 | 34 |

The rig lost 19 of 173 carries. About half were false ends of the hold detector: the carry grip sits
2 mm over the detector's exit margin.

### 6.3 The screen h_SP_scr

h_SP_scr (15:32) read S with `--rig_place` and the wave 7 teacher pace, at 0b91d7b on 7h-r1, with
20 Machine A lanes, from 12:02 to 13:36:

| read | correct of 3000 | correct stows an hour | held-out row | cycles | steps | D against S, paired SE | D against O, paired SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h_SP_scr | 2527, 0.8423 | 351.8 | 589 of 744, 0.7917 | 3330 | 775,854 | -6, 29.2 | -143, 24.8 |

It is not a candidate (bar 2563), and its rate is under S's. The mechanisms against S
(`h/tables/h_SP_scr.mech.txt`):

| row | SP | S |
| --- | --- | --- |
| rig handoffs (for S, the carry guard's only) | 2627 | 210 |
| of them placed | 2468 | 188 |
| of them `lifted_lost` | 151 | 21 |
| air stalls | 16 | 83 |
| M_stall, steps | 119 | 752 |
| `wrong_box` cycles | 158 | 182 |
| `wrong_box` cycles whose views agreed | 62 | 90 |
| wrong-box parts | 315 | 311 |
| riders | 157 | 129 |
| passengers | 66 | 55 |
| `release_none_lifted` | 172 | 30 |

The reading (15:32). The carry stalls are gone. The wrong-box cycles fell, but the wrong-box parts
did not, because riders and passengers rose. `release_none_lifted` reads 172 against 30, and 151
carries end `lifted_lost`. The ledger puts these on the detector's false ends on the carry grip. So
the hybrid moves losses from the carry to the hold and does not lift correct stows. HD12 P1 ends,
and run 4 stays on S.

## 7. Track V: π0.5

The Track V session owns this track (section 3.5). The ledger lines marked "[V]" and the vla, vread
and vres track reports hold its facts.

### 7.1 The data

`lerobot_stack/derive_vla_dataset.py` builds the π0.5 dataset from the MS7g corpus
`g/w1234567/dataset` (LeRobot v3.0, waves 1 to 7). It holds 23,957 episodes and 9,676,460 frames at
30 fps. Its videos are hard links to the corpus files, which passed the decode check on 2026-10-03.
The dataset is `h/vla/dataset`, 66.3 GB with hard links. One third is `h/vla/dataset_third`, 7,988
episodes.

- `observation.state` is 23 wide. It holds the 7 joint and gripper values, then the 16 context
  values with the camera roll wrapped as the ACT context fix wraps it. The wrapped roll spans
  -0.0348 to 0.0348 rad (q01 -0.0341, q99 0.0344).
- `observation.environment_state` leaves the dataset.
- Every episode has one task string: "pick up the marked part and place it in its box".
- The state and action statistics are recomputed over all frames, quantiles too. This is a deviation
  (section 3.4). The corpus statistics average the quantiles of each episode: joint 1 q01 reads
  -0.601 there, against -1.779 over all frames. So normalized actions spanned -4.18 to 3.89. With
  the recomputed quantiles they span -2.00 to 1.93.

### 7.2 The base model and the tokenizer

`lerobot/pi05_base` is pinned at revision a538eb27. Its main branch names a processor step that
LeRobot 0.5.1 cannot load. The tokenizer repository `google/paligemma-3b-pt-224` is gated: it gave
HTTP 401 without a login (about 15:35). The owner received access at 15:56, and the controller
downloaded the five tokenizer files to the HF cache (snapshot 35e4f464). Offline hosts load the
cached files with `GemmaTokenizer`, because `AutoTokenizer` also reads the repository's
`config.json`.

### 7.3 The local dry run (MS7h design section 5.4 step 3)

The dry run ran on the RTX 5090 with `HF_HUB_OFFLINE=1`, the real tokenizer and expert-only training
(about 16:20):

| run | loss | GPU memory | seconds a step |
| --- | --- | --- | --- |
| batch 2, 50 steps | 0.122 at step 1, 0.137 at step 10, 0.223 at step 25, 0.192 at step 50 (flat in the noise) | 15.2 GiB | 0.159 |
| batch 32, gradient checkpointing, 10 steps | 0.213 to 0.169 | 18.5 GiB | 3.34 |

The prompt and the 23-wide state print right, so the local dry run passes. A full fine-tune does not
fit the 5090. An expert-only run of 30k steps at batch 32 takes about 25 h on it (about 15:35).

### 7.4 Training on Colab (MS7h design section 7.2)

The owner chose Colab Pro+ and a full fine-tune on the full dataset (D1, 16:35). The upload folder
`h/vla/upload/maker-arm-ms7h-pi05/` (16:48) holds `dataset/` (492 files, 62 GiB), the script
`colab_train_pi05.sh` and the notebook `train_pi05.ipynb`. Its Drive target is
`MyDrive/maker-arm-ms7h-pi05`.

- The first smoke failed on a 401 "Please log in" for the gated tokenizer with HF_TOKEN set ([V]
  19:11). The fix 61737f9 strips the token and names the cause through `whoami` and `auth_check`.
- The smoke ran 200 steps on an A100 80GB at batch 32 ([V] 19:58): 3.791 s a step, a loss of 0.171
  to 0.092, a checkpoint of 22.80 GiB and 46 GB of GPU memory. The system disk held 165.8 of 235.7
  GB. 8f348ca moves the local checkpoints off it with `OUT_ROOT`.
- The full run started at 20:19: 30k steps (the design's INITIAL), batch 32, a full fine-tune on an
  A100 80GB. Its local checkpoints go to `/mnt/local-scratch` (350 GB free). The script md5 is
  4e36ca39 (8f348ca).
- The runtime bills 6.77 CU an hour. So 30k steps take about 32 h and 220 CU of 592. The design's
  cap is 400 CU.
- At step 800 the loss read 0.083 (0.185 at step 100), at 3.79 s a step ([V] 21:18). The learning
  rate was still in its 1000-step warmup.
- A checkpoint reaches Drive every 2k steps. The 24 h session limit falls near step 22,000, and the
  run resumes from the last Drive checkpoint. The end is about 04:00 on 2026-10-07, plus the resume
  gap.
- `colab download` from a CPU runtime moves 15.3 MB/s, so the 9.4 GB model reaches Machine A in
  about 10 minutes ([V] 23:07). `colab drivemount` needs a person at a terminal. The fetch test on
  the step-2000 checkpoint passed at 23:51: 7 files with equal sha256, in 17 minutes.

### 7.5 Serving and reads (MS7h design section 7.3)

`lerobot_stack/pi05_server.py` holds one π0.5 copy for many lanes on a ZMQ ROUTER socket. It builds
the 23-wide state from each request's state and context, so the clients do not change. It serves 50
actions a forward with `--action_warp 1.5`, open loop, 10 flow steps. A forward of 1 row takes 172
to 207 ms, and of 6 rows 346 ms, in 9.4 GiB (about 15:35).

`feat/ms7h-vla-read` b7c3ebd ([V] 20:03) adds `run_gate_eval.py --pi05 --lanes N` (one server for N
clients, cache key `pi05:<ckpt>`) and `fleet.py read --pi05` (one supervisor and one server a host).
With the mode off, the supervisor builds the same commands as before, and tests pin them. 1698 repo
tests pass.

- A two-lane smoke with the 50-step dry-run checkpoint ended EVAL_OK on both lanes. It stowed 0 of
  100, as expected of that checkpoint.
- It ran 40 steps a second a lane and made 3,204 forwards of one row at a median of 138.6 ms.
- The server took 9.8 GB of GPU memory.
- Machine B needs `.venv-pi`, the tokenizer cache and a shipped checkpoint (9.4 GB, about 56 minutes).
  `.venv-pi` and the tokenizer cache were ready on Machine B at 20:19.

### 7.6 V plus the residual (MS7h design section 7.4)

The shared server also serves V plus a residual actor (`feat/ms7h-vla-residual` 73e6f1b, [V] 22:19).
The actor keeps the frozen ResNet18 features of the Track R base S (F5 200k). Its base action input
is V's action. A V+R actor's base id names V and the feature checkpoint, so the server refuses an
actor that was trained on B alone (1c637a5). 1724 repo tests pass.

- With a zero residual, one V copy answered every request plain and with the residual: 51,615 of
  51,615 replies were equal. Two separate servers cannot agree, because π0.5 draws its flow noise
  from an unseeded generator.
- A two-lane push smoke pushed 24 cycles and 18,798 steps. Each lane matched one heap's RL census,
  the steps came in order and the Phi identities held. The lag-1 autocorrelation of the noise was
  0.897. The server took 9,886 MiB.
- The residual adds 5.46 ms a step at the median on the loaded 5090. V alone takes about 4.7 ms a
  step.
- The cost probe ([V] 22:20, `h/tables/vres/cost_probe.out`): the residual costs 5.1, 6.7, 10.7,
  17.8 and 33.8 ms a loop iteration at 1, 2, 4, 8 and 15 lanes. That is about 2.2 ms a lane step
  from 8 lanes on.

### 7.7 The week-2 rule: HR8, HD9, HD11 and HD12 P4

Rule HR8 (MS7h design section 7.4). If V's screen reads at least h_B_scr - 30, residual RL on V runs
in week 2. Otherwise V stops at its report. The report gives V's held-out row in both cases. The
held-out row is the reason for Track V.

HR8 names h_B_scr - 30, which is 2450. It was written before HD3 made S the base. The Track V
session used h_S_scr - 30 = 2503 from 20:31. HD9 (03:05) rules between the two numbers: the gate
follows the incumbent, so V's 30k screen must read at least 2503. The report-only h_V10k_scr decides
nothing.

HD11 (09:31) fixes the 30k screen h_V30k_scr before any V number at step 30k:

- The checkpoint is 030000 from Drive. If the Colab run stops early, it is the run's last
  checkpoint, named in the ledger before the read.
- The serving is S's with `--pi05`, on the screen seed 20307000, draws 0 to 19.
- The read runs on Machine A with two π0.5 servers of 10 lanes each, only while no R screen
  reads. It is cut at 4 h.
- HD9 decides on 20 complete heaps. When heaps are missing at the cut and the finished heaps'
  correct stows plus 150 for each missing heap stay under 2503, V fails HD9. That sum is a bound,
  not an estimate. Otherwise the missing draws run again before the decision.

HD12 P4 (09:38) amends HD11. V's 30k screen reads draws 0 to 6 first, on 7 lanes. If their correct
stows plus 150 for each of the 13 other draws stay under 2503, V fails HD9 and stops at its report.
Otherwise draws 7 to 19 follow.

### 7.8 The early screen h_V10k_scr

The pre-registration ([V] 23:55, report-only). The read takes V at step 10k of the Colab run, on
seed 20307000, draws 0 to 19. It uses `read_h.sh`'s serving with `ADD=--pi05` and
`HOSTS=a_h2l:20`, one π0.5 server for 20 lanes. The worktree is 7h-r2 at `feat/ms7h-vla-read`
b7c3ebd, PORT_BASE 33600. It decides nothing. It is placed against h_S_scr (2533) and h_O_scr
(2670), with the held-out row. After HD8 the slot changed ([V] 02:18): the read starts at 06:45 or
later, only while no h_R3 screen reads, and it is cut at 09:30. V's deciding screen stays the
step-30k checkpoint.

The run ([V] 07:22 and 09:31). The step-10k checkpoint reached Drive at 07:04. The fetch gave equal
sha256 on 7 files by 07:22, and the read started then. The 5090 held 1.0 GB at the start. At 09:30
the read was cut as pre-registered (V10K_FAIL, rc 124). One heap of 20 finished:

| row | V at step 10k, draw 1 |
| --- | --- |
| correct | 28 of 150. On the same draw S stowed 133 and B 116. |
| correct stows an hour | 82.6 |
| held-out row | 8 of 42 |
| cycles | 72: placed 37, budget 21, dropped 14 |
| wrong-box parts | 6 |
| end | no_candidate |

The 20 lanes on one π0.5 server ran about 8.6 steps a second a lane (4.6 forwards a second at 168
ms, 1.12 rows). So a V heap took about 71 minutes of wall time, and most heaps did not finish in 2 h
8 min. The report-only reading: V at step 10k is far under S, and one heap is not a screen. For the
30k screen the ledger plans about 3 h of wall time on one server, or two servers. HD11 takes two
servers.

### 7.9 The results of Track V

`<<V_RESULTS>>`

The rows to fill: V's 30k screen h_V30k_scr against h_S_scr (2533, held-out 0.7809), h_B_scr
(2480, held-out 0.7903) and h_O_scr (2670, held-out 0.8602). Then the decision of HD9 under HD12
P4.

## 8. The verdict reads

### 8.1 The candidate

Rule HR9. The candidate is the best screen among B, R, V and V+R. HD12 P1 adds SP to that list.
Its checked read on seed 20307100 runs by 2026-10-17, with or without a screen margin, paired with
h_O_chk. The official read (seed 20307200, 40 heaps, the candidate and O) runs only on a checked
pass of HR2. Seed 20307200 is still unburned from MS7f and MS7g. HD12 P5 keeps these rules and the
bar of HR2.

### 8.2 The checked read

The bars of HR2 on the checked seed, against h_O_chk:

| bar | rule | value on the checked seed |
| --- | --- | --- |
| correct stows | D at least 61 against h_O_chk (2685) | at least 2746 correct of 3000 |
| correct stows an hour | at least O's on the same read | at least 390.2 |
| held-out row | at least O's on the same read | at least 0.8318 (653 of 785) |
| MS7f staged row, report-only | 0.85, 200 an hour, held-out 0.80 | |
| MS7f stretch row, report-only | 0.92, 300 an hour, held-out 0.85 | |

The report also gives the measured paired SE of D.

`<<CHECKED_READ>>`

### 8.3 The official read

The bars of HR2 on 40 heaps: D at least 86 against O's official read on the same draws. The rate and
the held-out row must be at least O's on that read. The official read is one shot. Both the
candidate and O read seed 20307200, draws 0 to 39.

`<<OFFICIAL_READ>>`

## 9. Lessons that carry over

1. The credit of one step. One step's action barely moves a cycle of about 230 steps. At alpha 2.5
   the per-step critic gave about 0.0006 of Q per full bound, and the prior held the actor at the
   base. At alpha 25 the actor followed the critic's errors. The 3M screen lost 43 correct stows and
   55.1 an hour against S. A residual held over a chunk gives the critic fewer, larger decisions.
   Run 3 showed that a chunk alone does not give the critic an action signal (lesson 12).
2. The price of time. Under HR2 the rate binds as hard as the share. A step is worth about 0.0037 of
   a correct stow. At the design's c = 0.0005 the critic valued a 10 % slower cycle at 1.2 % of a
   stow. The actor of run 2 slowed the cycles from 228.7 to 260.6 steps.
3. The jaw-gap channel. A gap residual loosens a hold. The 3M actor with its gap output zeroed
   stowed 261 against S's 257 on six smoke heaps, while the full actor stowed 236.
4. The pessimistic target. The minimum over a random pair of target critics compounds its bias over
   about 167 bootstraps at n = 3 and gamma 0.998. q_mean reached -10.6 for rewards near -1 to +1.
   The mean target tracked the MC return within 0.01 at alpha 2.5.
5. A learning actor degrades its own collection. Smoke 3's unregularized actor turned bang-bang
   (|r|/b 0.895) and stowed 15 of 50 against 36 to 43 for plain B. With the prior toward the zero
   residual, the actor stayed at |r|/b 0.11 to 0.14 in a live smoke. Every later do-no-harm check
   passed.
6. The GPU memory budget. A 20-lane Machine A screen needs about 19 GB of the 5090 beside the learner,
   about 0.9 GB a read server. A π0.5 server of about 9.9 GB beside them made 14 of 20 servers fail.
   The per-step features of 20 servers and the learner made a screen take 2.5 h. GPU tests of the
   agents on the shared 5090 killed six lanes of h_R3_10M_scr (09:56).
7. The SIGTERM deadlock. Twice the learner outlived its unit's stop (21:05 and 01:47). At 01:53 its
   process waited in `futex_do_wait` and held its ports until a SIGKILL by hand. A re-entered signal
   handler waited on the lock of `threading.Event.set`. A handler that takes no lock fixes it
   (6b10b74). The fixed learner ends in 4.8 s with its state and its replay.
8. The noise of a same-seed repeat. The same policy, serving and draws read 2480 and 2554 (D = -74,
   paired SE 29.0). The two runs part at step 275 of cycle 2 on a float difference. A screen of a
   checkpoint selected on that seed is a high draw. Two separate runs diverge from cycle 0, so an
   equality test needs one live heap fed to both sides (smoke 1).
9. The truth gap of the oracle. The perceived loop and the vote cost O 231 correct stows against
   O_truth. Still, O stands 137 to 160 correct stows over S, mostly in the wrong box.
10. A stray file in a checkpoint directory. An extra `action_space.json` in Machine B's F5
    `pretrained_model` stopped the ship of h_S_chk (02:15).
11. Corpus statistics. The corpus statistics average the quantiles of each episode. A quantile
    normalization needs them recomputed over all frames (section 7.1).
12. A critic that sees privileged state memorizes. Run 3's critic fit the return of each replay row
    with a correlation of 1.000. On unseen first steps it did no better than a constant (RMSE 0.556
    against 0.554). Each state has one action, so nothing shapes Q along the action. A held-out RMSE
    against a constant and the critics' agreement on dQ/du test for this before a screen. In the
    offline test only the update budget changed it (section 5.8).
13. Bound the gain before a long run. The data-only slopes of run 3's replay put a sustained shift
    around S at about +25 correct stows a screen, inside one screen SE. The bar asked for about 221
    over S on the checked seed.
14. A loss scaled by mean|Q| needs a floor. Run 4's returns sat near 0 (q_mean 0.002). So alpha 10
    acted about 100 times run 3's, and the 3M screen lost 101 correct stows against S (HD13).
15. Find where the losses sit before choosing the lever. HD12 F1 read S's excess losses as after the
    lift. A part-by-part breakdown put most of them in the pick. The hybrid that takes over the
    carry moved losses from the carry to the hold (D = -6).
16. A fill needs its own TAG. `fleet.py` reads a TAG once and counts no cached payload as a read. So
    the fill of h_R3_10M_scr reads the six dead draws as h_R3_10M_fill (15:38).

## 10. The code

### 10.1 The branches

The branches start from main c6d4981. The SHAs are the pushed tips at this draft. The interface
names the restack order: guard, oracle, rl, res, vla, shared, docs.

| branch | tip | what it adds |
| --- | --- | --- |
| `feat/ms7h-context-guard` | 89a8813 | 5d177c8 the server refuses an ENV MEAN_STD checkpoint without a context marker. 89a8813 `train_masked.py --context_norm` writes and merges the marker, and an ENV-mapped resume needs the flag. |
| `feat/ms7h-oracle-loop` | 7a1344f | 417a6b1 the rig drives every perceived cycle and places into the voted box. d6f4d68 the O read mode with no server. 31298e8 docs. 7a1344f the takeover cap of HD2. |
| `feat/ms7h-rl-stream` | d246282 | dbdae0d the optional protocol fields (`PRIV_DIM` 90). 03f9fce `core/rl.py` and `--rl_stream`: Phi, its credit and the privileged vector. 0151512 Phi's statuses with a rest hysteresis. d246282 the end regrade is unassigned. |
| `feat/ms7h-learner` | a1ed165 | 8d1a185 `residual.py`: the actor, the 4-critic ensemble and the actor file. a1ed165 `residual_learner.py`. |
| `feat/ms7h-learner-speed` | 8d66722 | ceffae4 a test that pins the batched critics to four MLPs. 81d0303 the replay saved at a clean stop and loaded on `--resume`. 8d66722 the update as CUDA graphs. |
| `feat/ms7h-residual` | ee03c38 | On the rl branch: the learner's two commits (06d765a, 4de9bb1), cb680ce `--residual` forwarding and fleet shipping, 7e818f1 the server side, 0247975 a flush of queued cycles at exit. Then a7af1c1 a base without the context fix, f10ea78 the conservative actor, d9337cb the mean target and 8841f87 the AR(1) noise. Then the speed commits (fast-forward at 21:08) and ee03c38 `--actor_alpha`. |
| `feat/ms7h-residual-chunk` | 6b10b74 | 348d5b8, 4d1934c, afc03f5, 86af792 and 6b10b74 (section 5.5). |
| `feat/ms7h-vla-data` | 8f348ca | `derive_vla_dataset.py`, `setup_pi_env.sh`, `colab_train_pi05.sh`, the recomputed quantiles, the token check (61737f9) and `OUT_ROOT` (8f348ca). |
| `feat/ms7h-shared-server` | 6c45b3b | 7995927 the message framing split from the socket calls, 6c45b3b the shared π0.5 server. Rebased onto `feat/ms7h-vla-data` 61737f9 at 19:12. |
| `feat/ms7h-vla-read` | b7c3ebd | b2cc4d4 the cached tokenizer offline, 0dc6327 the forward logs, e3d3646 `run_gate_eval.py --pi05 --lanes N`, b7c3ebd `fleet.py read --pi05`. |
| `feat/ms7h-vla-residual` | 73e6f1b | The V branches cherry-picked onto `feat/ms7h-residual` 8841f87 (1bedfe2). a899453 the residual setup split, 1c637a5 a V+R actor's base id, 1d71b51 the residual in `pi05_server.py`, 73e6f1b `--pi05 --residual` forwarding. |
| `feat/ms7h-rig-place` | 0b91d7b | 8880381 the switch `--rig_place`, 0b91d7b the rig hands back at the end of RELEASE (section 6.1). |
| `feat/ms7h-pick-reward` | 7984643 | On 6b10b74: bdfb24b the protocol, 14e1818 the loop's pick terms under `--rl_pick`, 7984643 the server and the learner (section 5.9). |
| `feat/ms7h-learner4` | 741ca1a | On 6b10b74: 14404ac the critic sizes and the held-out cycles, 741ca1a each row's exploration noise pushed and kept in the replay. |
| `feat/ms7h-run4` | 396a206 | 7fd4f0e merges `feat/ms7h-learner4` and `feat/ms7h-pick-reward`. 396a206 the scale floor of HD13. |
| `stack/ms7h-docs` | this report | `docs/ms7h-beyond-the-oracle.md` and the README line |

The restack note of the Track V session (22:20): 73e6f1b supersedes `feat/ms7h-vla-data`,
`feat/ms7h-shared-server` and `feat/ms7h-vla-read`, and 1c637a5 touches `residual_learner.py`.

The test counts in the ledger:

| commit | repo tests | lerobot_stack tests |
| --- | --- | --- |
| 89a8813, context guard | 1679, 84 skipped | 170 |
| a1ed165, learner | 1683 | 186 |
| 31298e8, oracle loop | 1699 | |
| d246282, RL stream | 1706 | |
| 0247975, residual server | 1713 | 201 |
| 8d66722, learner speed | 1713 | 212 |
| b7c3ebd, π0.5 read | 1698 | |
| 73e6f1b, V+R serving | 1724 | |
| 6b10b74, chunked residual | 1714 | 225 |
| 0b91d7b, rig place | 1721 | 225 |
| 7984643, pick terms | 1715 | 235 |
| 7fd4f0e, run 4 | 1716 | 250 |

### 10.2 The stack

The restack of 11:33 changed no `feat/ms7h-*` branch and opened no PR. It built six layers:

| layer | tip | commits | note |
| --- | --- | --- | --- |
| `stack/ms7h-guard` | 89a8813 | 2 | |
| `stack/ms7h-oracle` | 04c81ec | 4 | |
| `stack/ms7h-rl` | ae740b1 | 4 | |
| `stack/ms7h-residual` | 2a3f36b | 19 | the chunk code. 2a3f36b refuses `--residual` on an oracle read. |
| `stack/ms7h-vla` | 55e9258 | 20 | 193cdd2 holds the residual chunk in the π0.5 server. 55e9258 refuses `--pi05` on an oracle read. |
| `stack/ms7h-docs` | 588c716 | 1 | this report |

Each layer's tests and ruff pass: 1671 to 1750 repo tests and 170 to 255 lerobot_stack tests. The
residual layer differs from `feat/ms7h-residual-chunk` only in the guard and oracle code. Four
branches are not stacked yet: `feat/ms7h-rig-place`, `feat/ms7h-pick-reward`, `feat/ms7h-learner4`
and `feat/ms7h-run4`.

`<<STACK>>`

## 11. Records

### 11.1 The files

| what | where |
| --- | --- |
| the ledger, the resume pointer, the shared names, the subagent rules | `~/data/maker-arm-ms7/h/README.md`, `STATE.md`, `interface.txt`, `agent_rules.txt` |
| the read payloads | `~/data/maker-arm-ms7/reads/h_*/`, and the supervisor log `reads/h_*.log` |
| the mechanism tables and the pair tables | `h/tables/<TAG>.mech.txt`, `h/tables/<TAG>.pair_O.txt` and `.pair_S.txt` |
| the chain logs | `h/tables/o_chain.log`, `ot_chain.log`, `h_*.chain.log`, `screen_r2.log`, `screen_r3.log`, `screen_r4.log`, `place/sp_chain.log`, `r3_10m_fill.log`, `r3_10m_fill2.log` |
| the track tables | `h/tables/<track>/`: oracle, rl, learn, res, speed, chunk, vla, vread, vres, place, pick, loss |
| the critic diagnostic, the offline critic test and the data-only slopes | `h/tables/res/qdiag/report.md`, `h/tables/res/ctest/tables.md`, `dataslope.md` |
| the loss breakdown | `h/tables/loss/tables.md`, `loss.py` |
| the learners | `h/col/learner` (run 1), `learner2` (run 2), `learner3` (run 3), `learner4` (run 4) |
| the actor files of the screens | `h/col/screens/` |
| the π0.5 dataset and the upload folder | `h/vla/dataset`, `h/vla/upload/maker-arm-ms7h-pi05/` |
| the read scripts | `h/bin/`: `sync_read_h.sh`, `read_h.sh`, `read_o.sh`, `screen_r.sh`, `screen_r4.sh`, `pair.py` |
| the run scripts | `h/bin/`: `abl.sh`, `dnh.sh`, `hd10.sh`, `col_start.sh`, `col_start4.sh`, `collect_host4.sh`, `col_stop.sh`, `unit.sh` |

### 11.2 The reads

Each read uses seed 20307000 (screen) or 20307100 (checked), draws 0 to 19, 3000 parts. The rate is
the correct stows an hour.

| time | TAG | SHA | worktree | policy or driver | seed | correct of 3000 | rate | held-out row | cycles | steps | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14:05 | h_B_scr | c6d4981 | 7h-r0 | B | screen | 2480 | 331.0 | 588 of 744, 0.7903 | 3186 | 809,083 | the first incumbent |
| 14:46 | h_B_chk | c6d4981 | 7h-r0 | B | checked | 2490 | 330.7 | 612 of 785, 0.7796 | 3241 | 813,195 | report-only |
| about 16:55 | h_O_scr | 31298e8 | 7h-r1 | O | screen | 2670 | 388.9 | 640 of 744, 0.8602 | 3516 | 741,451 | the reference of every screen |
| 17:37 | h_O_chk | 31298e8 | 7h-r1 | O | checked | 2685 | 390.2 | 653 of 785, 0.8318 | 3452 | 743,180 | the pair of the checked read |
| 18:10 | h_Ot_scr | 7a1344f | 7h-r1 | O_truth | screen | 2901 | 381.1 | 715 of 744, 0.9610 | 2985 | 822,142 | report-only |
| 19:26 | h_S_scr | c6d4981 | 7h-r2 | S | screen | 2533 | 369.5 | 581 of 744, 0.7809 | 3238 | 740,435 | the incumbent (HD3) |
| 22:01 | h_R2_1M_scr | 8841f87 | 7h-r3 | S + run 2 actor at 1M | screen | READ_FAIL | | | | | 6 heaps ran |
| 01:27 | h_R2_3M_scr | 8841f87 | 7h-r3 | S + run 2 actor at 3M | screen | 2490 | 314.4 | 588 of 744, 0.7903 | 3282 | 855,306 | not a candidate |
| 02:48 | h_S_chk | c6d4981 | 7h-r0 | S | checked | 2525 | 366.0 | 621 of 785, 0.7911 | 3220 | 745,148 | report-only |
| 04:48 | h_R3_1M_scr | 6b10b74 | 7h-r3 | S + run 3 actor at 1M | screen | 2536 | 389.2 | 599 of 744, 0.8051 | 3134 | 703,799 | not a candidate |
| 06:35 | h_R3_3M_scr | 6b10b74 | 7h-r3 | S + run 3 actor at 3M | screen | 2539 | 378.2 | 597 of 744, 0.8024 | 3231 | 725,132 | not a candidate |
| [V] 09:31 | h_V10k_scr | b7c3ebd | 7h-r2 | V at step 10k | screen | 28 of 150, one heap | 82.6 | 8 of 42 | 72 | | report-only, cut at 09:30 |
| 12:04 | h_R3_10M_scr | 6b10b74 | 7h-r3 | S + run 3 actor at 10M | screen | `<<R3_10M_SCR>>` | | | | | 14 of 20 heaps, then a fill |
| 15:32 | h_SP_scr | 0b91d7b | 7h-r1 | SP | screen | 2527 | 351.8 | 589 of 744, 0.7917 | 3330 | 775,854 | not a candidate |
| 15:35 | h_R4_3M_scr | 7fd4f0e | 7h-r3 | S + run 4 actor at 3M | screen | 2432 | 312.3 | 567 of 744, 0.7621 | 3123 | 840,994 | not a candidate |

The cycles and steps come from the chain logs and the supervisor logs. `<<READS_AFTER_1538>>`

### 11.3 The commands

One read, as the ledger gives its form for a V read ([V] 20:03):

```bash
N=<n> SHA=<40-hex SHA> TAG=<TAG> SEED=20307000 PORT_BASE=<p> CKPT=<checkpoint> ADD='<switches>' \
  bash ~/data/maker-arm-ms7/h/bin/sync_read_h.sh
```

`sync_read_h.sh` puts the read worktree `~/src/maker-arm-lab-7h-r$N` at the SHA on Machine A and
Machine B with `scripts/fleet.py sync`. Then it runs `read_h.sh` with 10 lanes on each host, or with
`HOSTS`. It writes `h/tables/$TAG.mech.txt` with `g/bin/mech.py` against `BASE` and prints
`SYNC_READ_DONE`. `h/bin/screen_r.sh` runs the R screens on `a_h3l:20` and pairs each with
h_S_scr and h_O_scr through `h/bin/pair.py`. `h/bin/screen_r4.sh` does the same for run 4, and it
waits for any `h_` read to end first. `h/bin/col_start.sh` (BASE, TAG, WHAT, LOUT) and
`h/bin/col_stop.sh` (a STOP file, or MODE=hard) start and stop the collection lanes.
`h/bin/col_start4.sh` and `collect_host4.sh` start run 4's lanes.

### 11.4 Video

`<<VIDEO>>`
