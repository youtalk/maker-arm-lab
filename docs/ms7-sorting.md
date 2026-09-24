# MS7a: a tray heap sorted into desk boxes (draft)

Date: 2026-09-24. Status: draft. The campaign still runs. Each `TBD` marks a result that did
not exist at the time of this draft.

The design is `2026-09-23-ms7a-sorting-design.md` in the claude-memory repository. The owner
approved it section by section. The campaign ledger is `~/data/maker-arm-ms7/README.md`, and
`STATE.md` beside it is the resume pointer. Every rule below went into that ledger before its
number. The read payloads are under `~/data/maker-arm-ms7/reads/<TAG>/`. The cycle, rake and
per-draw counts of the policy reads come from those payloads.

All times are PDT, and each is the time of the ledger entry that records the event. Times after
midnight are on 2026-09-24. All other times are on 2026-09-23.

## 0. Read this first

MS7a moves the MS6 heap policy toward the sorting task of roadmap decision R1. It has two
rungs. In rung 1, a heap of 50 parts lies in a walled tray, and every part goes into one desk
box. In rung 2, the same tray heap goes into four desk boxes by four part classes.

Neither rung passes its checked read so far. The rung 2 checked read stows 2482 of 3000 parts
in their own class's box, 0.827 (Wilson 95 % 0.813 to 0.840). It uses 1.192 cycles per correct
stow. The gate needs at least 2550 parts and at most 1.3 cycles. The cycle bar passes, and the
read is 68 parts short.

The rung 1 checked read stows 2466 of 3000, 0.822 (0.808 to 0.835), at 1.352 cycles per stowed
part. That read ran before the last two serving rules entered the ledger. A re-read under the
full protocol runs now: TBD. The official read of rung 2 is the milestone's number: TBD.

Slopes are the largest loss. On the rung 2 checked read, 609 of 910 slopes stow, 0.67. Slopes
are 301 of the 518 lost parts. A slope one stud wide that lies flush on a vertical tray wall has
no grasp, and the scripted rake cannot get behind it. The teacher loses slopes the same way. It
stows 0.81 of the slopes on its rung 2 read.

The reads come from a learned policy plus serving rules. A serving rule changes the policy's
command, or gives the env to the teacher for a few steps, before the env receives the command.
The rung 2 checked read serves nine switches. Four come from MS6: `--select_at_release
--replan_on_switch --squeeze_floor --square_wrist`. Five are new in MS7a: `--scripted_rake
--part_failures --pick_snap --dwell_close --adopt_held`. Section 7 gives each rule and the
forensic that found it.

Some rules read simulator state. `--square_wrist`, `--dwell_close` and `--adopt_held` read the
held part and the jaw state from the simulator. `--pick_snap` uses the stop flag of
`--square_wrist`. The real arm needs a hold signal and a held-part detector for these rules.
`--scripted_rake` runs the teacher's rake, and the teacher plans on the simulator's part poses.

The teachers pass every bar. On 600 held-out parts each, the rung 1 teacher stows 0.920, the
sorting teacher on the bare desk 0.963 and the rung 2 teacher 0.905. The base refresh failed. T5
read 777 and T6 853 of 1000 against T0's 900, so T0 400k stayed the base.

## 1. The goal, the rungs and the gates

The ledger fixed these rules at 10:30, before any MS7 read. Both rungs use held-out parts
(`--parts_split heldout --parts_seed 20260915`) and heaps of K = 50.

| rung | scene | pass on the checked read |
| --- | --- | --- |
| 1, the tray | the tray heap, one desk box at a drawn pose | at least 2550 of 3000 parts at rest in the box, at most 1.3 cycles per stowed part |
| 2, sorting | the tray heap, four desk boxes at drawn poses, four classes, a class-to-box map drawn per episode | at least 2550 of 3000 parts at rest in their own class's box, at most 1.3 cycles per correct stow |

A correct stow is a part at rest in its own class's box. The supervisor runs each read with
`--threshold 0.85` and `--max_cycles_per_stowed 1.3`, which is the bar of 2550 of 3000. The
checked winner of rung 2 gets one official read of 120 episodes: seed 20261800, draws 0 to 23,
6000 parts. That read is the milestone's number, and it passes at 5100 of 6000.

Each teacher read drives with `--oracle_driver`, so it needs no GPU and no policy server:

| teacher | first read | repeat read |
| --- | --- | --- |
| rung 1: tray, one box | 150 held-out parts, at least 0.90 stowed, at most 1.3 cycles per stowed part | 600 held-out parts, the same bar |
| sorting on the bare desk, four boxes | 150 parts, at least 0.90 in the correct box | 600 parts, the same bar |
| rung 2: tray, four boxes | | 600 parts, at least 0.90 in the correct box |

The base refresh rule: the better of T5 and T6 reads 1000 parts on the MS6 seed 20261300,
draws 0 to 3, with the four MS6 switches. At least 932 of 1000 replaces T0 400k as the base. T0
read 902 there in MS6.

The stop rules:

1. If the tray teacher reads under 0.90 at noon on 2026-09-26, rung 1 is dropped, and rung 2
   runs on the bare desk.
2. Rung 2 starts on 2026-09-27 whatever rung 1 reads. It warm-starts from rung 1's best
   checkpoint.
3. Rung 2's official read on 2026-09-30 is the milestone's number. There is no extension.

The scripted-place rule: the gate reads the policy's own place first. If that read fails and
`wrong_box` is its largest failure class, the same checkpoint reads again on the same draws with
`--scripted_place`. That switch hands the cycle to the teacher at the first hold plus a 5 cm
lift. The teacher then carries the part to its box. A pass there is `PASS (scripted place)`, a
separate outcome from `PASS`. The rule never fired, because `wrong_box` was never the largest
failure class (section 8).

The seeds:

| use | seed | draws |
| --- | --- | --- |
| screen | 20261700 | draw 0 (250 parts) |
| selection | 20261700 | draws 0 to 3 (1000 parts) |
| checked | 20261700 | draws 0 to 11 (3000 parts), fresh TAG |
| official, rung 2 only | 20261800 | draws 0 to 23 (6000 parts, 120 episodes) |
| base refresh | 20261300 (MS6) | draws 0 to 3 |
| teacher read N, process i | run seed 20267500 + 20 N + i, layout seed 20268500 + 20 N + i | one heap per process |
| corpus process p | run seed 20270000 + p, layout seed 20271000 + p | never a read seed |

`ledger.json` in the campaign directory lists every corpus seed. The gate supervisor refuses a
read seed that a corpus used.

The rules of campaign n carry over. A payload from the supervisor cache is not a read. Every
read records the SHA, the `policy_id`, the split and the serving switches. A mid-training
checkpoint gets its `action_space.json` marker before a read. Every video in a corpus is
decode-checked before training. The MS6 negatives stay closed: no noise-scaled corpus, no
round-1 DAgger frames, no 25-action horizon and no close gate.

The design gave MS7a one week, from 2026-09-24 to 2026-09-30. The campaign started on
2026-09-23 at 09:58 with the launch of T5. Stop rule 1 cleared at 13:27 the same day, and rung 2
training started at 19:45.

## 2. The hosts and fleet.py

### 2.1 The roles

| host | GPU | role | read lanes |
| --- | --- | --- | --- |
| Machine A | RTX 5090 32 GB | the controller session and the ledger, T6, arm A of each round, screens, selection and checked reads | 8 |
| Machine B | RTX 5080 16 GB | T5, arm B of each round, screens, selection and checked reads | 6 in `hosts.json`, 4 on the checked reads |
| Machine C | RTX A2000 12 GB, 128 cores | teacher reads, corpus recording and conversion | 12, on CPU policy servers |

The trainers never recorded a corpus. Campaign n showed that recording, reads and training on
one host drove training down to 5 step/s.

### 2.2 fleet.py

`scripts/fleet.py` uses only the standard library, like the gate supervisor. It reads the host
table `hosts.json` in the campaign directory. For each host, the table holds six values. They
are the ssh target, the main clone, the campaign worktree, a setup line, the read lane cap and
the policy server device. Each subcommand prints one terminal token:

| subcommand | what it does | token |
| --- | --- | --- |
| `sync --sha SHA` | puts the campaign worktree of each host on SHA and runs a fast test subset | `SYNC_OK` or `SYNC_FAIL` |
| `ship SRC DST` | copies with `rsync --partial` and compares the file lists of both ends | `SHIP_OK` or `SHIP_FAIL` |
| `run HOST NAME -- CMD` | runs a logged job in the worktree of the host. A remote job detaches. | `RUN_OK` or `RUN_FAIL` |
| `read` | splits one gate read across hosts by lane caps, pulls the payloads into a fresh TAG directory and runs the aggregate pass | `GATE_OK` or `GATE_FAIL` |
| `status` | prints the last token line of every job log on each host | `STATUS_OK` |

`read` refuses a payload older than the read. Every path is the same absolute path on every
host, because a payload records its checkpoint path as its policy id. `feat/fleet` landed at
10:32 with 17 tests. The first live sync passed on all three hosts in 3 s, 3 s and 21 s.

A control read tested the pipeline. T0 400k read 900 of 1000 through `fleet.py read` on Machine B
(TAG t0_400000_ctrl), against 902 in MS6. So a regression under this pipeline is a real
regression.

### 2.3 The LAN change

The design measured 2.8 MB/s on every host pair through the Tailscale DERP relay. At 13:40, the
owner wired Machine B and Machine C into one LAN. Machine B then
read 117.6 MB/s from Machine C (300 MB in 2.6 s), and Tailscale went direct between
them in 1 ms. Machine A still reaches both hosts at 2.8 MB/s.

The data flow changed with it. Corpora ship from Machine C to Machine B over Tailscale SSH
at LAN speed. Machine A gets each dataset over the slow path. That takes about 33 minutes per
5.4 GB and about 95 minutes for the 16 GB rung 2 dataset.

At 12:10, both remote hosts asked for a Tailscale SSH check again, and the owner approved both
at 12:11. At 14:00, the owner set the Tailscale SSH action to "accept". After that, remote ssh
needed no approval.

### 2.4 Host lessons

- Machine C did not resolve github.com at the start. `sync` then shipped it a git
  bundle (`"fetch": "bundle"` in `hosts.json`). The owner fixed its DNS at 14:12, and the bundle
  entry left `hosts.json` at 16:31.
- A subagent cannot launch a detached remote job, because the permission classifier blocks it.
  The controller session launches every remote job.
- An ssh without `-n` stayed open on its detached child. Arm A of rung 1 started 22 minutes late,
  until the controller ended that ssh client. Remote launches now use `timeout 60 ssh -n ...
  setsid nohup ... &`.
- `fleet.py sync` gained `--hosts` and a guard that refuses to move a worktree under a running
  job (b843d3d). Reads run `fleet.py` from the synced campaign worktree, because the aggregate
  pass imports the `run_gate_eval.py` of that worktree.
- The first rung 1 pipeline opened the Machine B ship log before its local directory existed, so
  that ship never started. Machine B also had no `ffmpeg` on a plain ssh PATH. The controller ran
  the ship and the decode check by hand. `bin/rung_pipeline2.sh` is the fixed copy, and the
  Machine B copy of `check_videos.sh` falls back to campaign n's `n/bin/ffmpeg`.
- 19 waiters of two finished agents waited on `pgrep -f`. That pattern matched their own
  command line, so they never ended. The controller stopped them at 13:45.
- CPU read lanes are slow. The first three rung 1 selection reads ran on Machine C at
  5.8 steps/s per lane. Draws 1 to 3 hit the eval client's 130-minute wall cap at 41k to 45k
  steps, and no read counted. `bin/read.sh` now passes `--max_wall_minutes 480` to the client
  and `--eval_timeout_s 36000` to the supervisor.
- EGL oversubscription lost the r2s wave. About nine lanes of the failed selection reads kept
  running from 20:34 to 03:05. A 16-lane conversion ran from 02:26 to 03:25. Both ran beside 40
  corpus processes. The 12 GB GPU filled, and 16 processes likely started their EGL context on
  Mesa (CPU rendering). They crawled at 580 to 1884 steps, and the wave wrote 0 episodes in its
  300-minute cap. The ledger first blamed the smooth carry at 04:45 and corrected that at 05:30
  (section 3.4).
- The EGL rule: Machine C runs at most 40 camera (EGL) processes in total, read lanes
  included. Before a wave, read `nvidia-smi` and make sure that no stray `policy_server` or
  `mj_eval_heap` process runs.
- A full disk lost wave 3. The aarch64 disk (937 GB) filled at 05:59. All 40 wave 3 processes
  crashed in the image flush, with no finished episode in any shard. A finished rung 2 shard is
  4.4 GB, so 40 processes needed about 176 GB, but about 134 GB was free.
- The disk rule: before a wave, free space must exceed 4.5 GB per process plus 40 GB for its
  conversion.

## 3. The scene

### 3.1 The tray

`--tray` drops the heap into the H1 tray of MS6 phase A. The interior is 0.13 x 0.20 m, at x
0.13 to 0.26 and y -0.10 to 0.10. Four static walls stand 20 mm high and 5 mm thick. The drop
insets each part by its whole half diagonal plus 5 mm, so no part starts over a wall.
`tests/test_layout.py` pins the tray interior inside `SCATTER_MAX_RADIUS`. Without the switch,
the heap drops on the bare desk as in MS6.

The teacher gained these tray mechanisms in `core/grasp.py` and `core/heap_oracle.py`:

- The wall term. Near a wall, a closing axis must run within 45 degrees of the wall's direction
  (`WALL_BAND_RAD`). The band applies to a grasp whose part centre is closer to the wall than
  the grasp's own finger reach plus 5 mm.
- End-to-end grasps. Along a wall, the jaws can close end to end on a part up to 60 mm long
  (`WALL_MAX_GRASP_WIDTH`). A brick_1x6 or a brick_2x7 flush along a wall has no other grasp.
- The rake. A candidate with no grasp within 30 mm of a wall (`RAKE_BAND_M`) is swept off the
  wall by the closed jaws. The rake plans around the finger hull and the jaw heading.
- Swings and the wedge. With no room behind the part, the sweep swings by up to 1 rad off the
  wall's normal. As a last resort, the tips wedge in 0.5 mm from the wall face.
- The rake caps. An episode spends at most 20 rakes (`MAX_RAKES`), and a part takes at most 2
  (`RAKE_PART_CAP`).
- Jam recovery. If the watchdog finds the arm jammed in a tray pick or a rake, the teacher gives
  up that leg, not the episode.
- The rake runs as its own oracle-driver takeover. The converter drops the rake frames (phases
  PUSH and TIP) from every corpus, so the policy never sees a rake.

The tray ladder ran at 13:15 on the MS6 shelf bin with `--tray` and the MS6 corpus recipe. It
used held-out heaps 0 to 5 (300 parts):

| step | change | stowed fraction |
| --- | --- | --- |
| r0 | the tray alone | 0.673 |
| r1 | the wall term | 0.767 |
| r2 | the rake | 0.833 |
| r3 | end-to-end grasps along a wall | 0.850 |
| r4 | the wedge rake | 0.860 |
| | the wall band at 45 degrees | 0.903 |
| r5 | the hull-aware rake | 0.910 |
| r6 | r5 plus the wall band at 45 degrees | 0.940 |
| r7 | jam recovery | 0.940 |

The first read on heaps 0 to 2 stowed 144 of 150, 0.960. The 600-part repeat read of r6, on
fresh heaps 10 to 21, failed at 525 of 600, 0.875, at 1.046 cycles per stowed part. The
watchdog abandoned two of its episodes. r7 on the same heaps read 538 of 600, 0.897, but those
heaps were no longer fresh.

The ledger lists the losses: 46 parts at a wall with no grasp, of which 33 are 1-wide slopes
flush on a wall. 11 double picks dropped past the far wall, and 7 parts were left at the cycle or
time limit. The tray ladder used the shelf bin. The decisive read is the rung 1 teacher with a
desk box (section 4).

### 3.2 The desk boxes

`--desk_boxes N` replaces the shelf and its heap bin with N open boxes on the desk
(`core/desk_boxes.py`). The boxes stand to the left (+y) and right (-y) of the arm. No box
stands behind the arm or in front of the tray. The boxes are near black mocap bodies, so they
read against every desk color draw. Each episode draws each box's centre in a side band and its
yaw uniform in plus or minus 45 degrees.

- The side band: box centre x 0.05 to 0.21 m and |y| 0.18 to 0.22 m. The one box of rung 1
  starts at x 0.08. At x 0.05, its outer corner left the overhead frame by 6.1 px at the extremes
  of the camera draw.
- The certification: `scripts/mj_release_envelope_check.py` released brick_2x4, brick_1x2x5 and
  brick_2x7 over the corners and the centre of the band. It used both sides and box yaws of plus
  and minus 45 degrees. Each part landed 80 of 80.
- The release: the pick point is at 0.10 m (`DESK_BOX_RELEASE_Z`), at the pick tilt.
- The clearance: every footprint keeps 15 mm from the pick corridor, the arm base and the
  tray's outer walls. Two boxes keep 20 mm apart.
- The interior: the one box of rung 1 is 0.11 x 0.11 x 0.06 m. Each of the four boxes of rung 2
  is 0.09 x 0.09 x 0.04 m, the six-brick `BIN_INTERIOR`.
- The grader counts a part at rest in its own box up to 0.08 m above the box floor. The 11 cm
  box keeps that 0.08 m. A band of twice its depth reached above the release height and counted
  a part that was still in the jaws.
- RETREAT ends at (0.15, 0.0, 0.12) m, over the near half of the tray, at the pick tilt.

The interior came from a teacher smoke at 11:33. It used the MS6 recipe, held-out parts, K =
50 and 2 heaps dropped on the tray footprint without walls. The MS6 bin stowed 94 of 100 at 1.05
cycles per stowed part. One 9 cm box stowed 93 of 100 at 1.02, and four 9 cm boxes 97 of 100 at
1.02. The one box lost parts to a pile over its 9 cm rim. The design allowed the larger interior
for that case, so one box uses 11 cm and four boxes keep 9 cm.

Each part has its own destination box. The box part of the served context row (`box_x`,
`box_y`, `box_yaw`) carries the current target's box. The loop writes it at each target
selection.

### 3.3 The classes and the class map

`part_class` in `core/parts.py` maps each part to one of four classes:

| class | rule | catalog parts | held out |
| --- | --- | --- | --- |
| `slope` | a slope | 6 | slope_1x2, slope_1x3 |
| `tall` | a brick taller than one brick height | 5 | brick_1x2x2, brick_1x2x5 |
| `narrow` | a one-height brick one stud wide | 8 | brick_1x6 |
| `wide` | a one-height brick two or more studs wide | 10 | brick_2x3, brick_2x7 |

Every class holds a held-out part, so a held-out heap fills all four boxes. Each episode draws
a class-to-box map, a random permutation of the four classes over the four boxes. So no link
from part appearance to box position survives across episodes. The ground-truth class stands in
for the classifier, which is MS7b.

The teacher releases each part above the centre of its class's box. The cycle census gains the
class `wrong_box`, a lifted part that comes to rest in a box that is not its own. The census and
the gate payload also report the parts stowed per class and per box.

### 3.4 The release roll along the carry

At 21:30, the owner reviewed a rung 2 teaching episode and asked two questions. Is the tray in
the scene? Does the motion stop at each waypoint? The corpus `run_meta` says `tray: true,
desk_boxes: 4`, and an overhead frame shows the tray. But the heap spans only about 30 x 20 px
of the 224 px frame, because the overhead camera still frames the old shelf region.

A motion trace on held-out heaps of K = 20 at be86bc8 found no stop inside any leg in 232
cycles.
Every rest is at a leg end, as the MS3 minimum-jerk walk intends. The owner saw the level-release
roll of MS6. The tool stands still over the box for 37 to 50 steps per cycle while only the
wrist turns. In MS6 and in both rungs, the tool is at rest about 46 % of each cycle.

The prototype 25f1dcf, 48bc50c on the stack, spreads the roll along the carry and the unroll
along RETREAT. On rung 2 over 12 seeds, the teacher stowed 224 of 240 with and without it. The
median steps per cycle fell from 392 to 373. Wrist-only frames fell from 37 to 5, and the p99
acceleration from 13.8 to 8.1.

A later timing of 48bc50c against be86bc8 read equal steps per second on both hosts. For
example, it read 11.8 against 12.1 with cameras, and 10.9 against 11.1 on Machine C. No MS7a corpus uses the
roll yet. The r2s re-record with it failed on host load, and wave 3 recorded at 9bcb2ed without
it.

## 4. The teacher reads

| teacher | commit | heaps | correct stows | fraction | cycles per stow | rakes | bar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1, first read | bcaa0ba | 3 (seeds 20267540 to 20267542) | 139 of 150 | 0.927 | 0.957 | 11 to 15 per heap | pass |
| rung 1, repeat read | bcaa0ba | 12 fresh (seeds 20267560 to 20267571) | 552 of 600 | 0.920 | 0.984 | 111 | pass |
| sorting, bare desk, smoke | 3c80aff | 2 (seeds 20267510, 20267511) | 96 of 100 | 0.96 | 1.04 | | |
| sorting, bare desk, repeat read | 3c80aff | 12 (read N 1) | 578 of 600 | 0.963 | 1.017 | | pass |
| rung 2, repeat read | 97c2ae1 | 12 (read N 2) | 543 of 600 | 0.905 | 1.009 | | pass |

The sorting reads drop the heap over the tray footprint with no walls. The oracle driver read 99
of 100 on the seeds of the sorting smoke.

Rung 1, repeat read, per heap: 45, 45, 47, 44, 43, 46, 48, 46, 45, 47, 49 and 47. The parts
left are mostly wall-stuck 1-wide slopes that no rake frees. This read cleared stop rule 1 at
13:27.

The sorting repeat read put 7 parts in a wrong box. Rung 2 put 12 parts in a wrong box, and its
stowed cycles took a median 336 steps. The per-class rows:

| class | sorting, bare desk | rung 2 |
| --- | --- | --- |
| slope | 184 of 191 | 133 of 165 |
| tall | 147 of 154 | 167 of 175 |
| narrow | 81 of 84 | 74 of 77 |
| wide | 166 of 171 | 169 of 183 |

Rung 2 per box: 154 of 164, 119 of 145, 142 of 155 and 128 of 136. The rung 2 loss is again the
wall-stuck slopes.

## 5. The base refresh

On days 1 and 2 of the design, the GPUs had no MS7 corpus yet. So they refreshed the bare-desk
base on corpora that already existed. Both runs warm-start from T0 400k.

| run | host | data | steps |
| --- | --- | --- | --- |
| T5 | Machine B | dataset_v5 = v2 + v3a, 8369 segments | 300k, 38 step/s |
| T6 | Machine A | dataset_v6 = v5 + v3b + v3c, 16893 segments, 7,778,532 frames | 300k |

dataset_v3b (5.6 GB) and dataset_v3c (12 GB) moved from Machine C to Machine A from
10:05, over five rsync streams each. dataset_v6 merged at 12:00 and passed the decode check on
239 videos.

Each read uses seed 20261300, draws 0 to 3, the four MS6 switches and commit bf9333a. T5 and T6
read on 4 lanes each:

| read | stowed of 1000 | Wilson 95 % | cycles per stowed | per draw |
| --- | --- | --- | --- | --- |
| T0 400k in MS6 | 902 | | | |
| T0 400k, control (t0_400000_ctrl) | 900 | 0.880 to 0.917 | 1.246 | 225, 232, 219, 224 |
| T5 300k (t5_300000_base) | 777 | 0.750 to 0.802 | 1.533 | 185, 222, 212, 158 |
| T6 300k (t6_300000_base) | 853 | 0.830 to 0.874 | 1.366 | 216, 225, 221, 191 |

T5 was rejected at 12:55. The base decision came at 14:55, on three of the T6 draws. After 216,
221 and 191 on draws 0, 2 and 3, the last draw needed 304 of 250 to reach 932. T0 400k stayed
the base (`base/BASE`). Both refresh runs read worse than their start. More data on the v2
recipe does not help a warm start of T0.

## 6. The corpora and the training arms

### 6.1 The corpora

Every corpus records train-split heaps on Machine C, with the teacher driving.

| corpus | commit | processes | heaps | stowed | cycles per stow | rakes | out of reach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1, wave 1 | 97c2ae1 | 0 to 29 | 60 | 2694 of 3000, 0.898 | 1.059 | 377 | 44 |
| rung 1, wave 1 and wave 2 part A | 97c2ae1, a6b631e | 0 to 39 | 90 | 4046 of 4500, 0.899 | 1.059 | 549 | |
| rung 2, waves 1 and 2 | a6b631e | 100 to 139 | 120 | 5217 of 6000 in the correct box, 0.870 | 1.098 | 803 | 107 |
| rung 2, re-record r2s | 25f1dcf | 200 to 239 | 120 planned | 0 episodes in the 300-minute cap | | | |
| rung 2, wave 3 | 9bcb2ed | 240 to 279 | 120 planned | no finished episode | | | |

The r2s wave failed on host load (section 2.4). Wave 3 was lost to a full disk. The wave 3
shards (134 GB) and four merged lane work directories (41 GB) were deleted.

Two converter fixes landed during the campaign:

- a8fc688 drops the tray rake frames (phases PUSH and TIP). A corpus recorded at 97c2ae1 then
  converts correctly with no re-record.
- 944cdc1 drops the REOPEN that a failed teacher cycle leaves positive. The pick forensic found
  that 7.3 % of the segments were 20-frame REOPEN fragments of failed takeovers.

The datasets:

| dataset | segments | makeup | decode check |
| --- | --- | --- | --- |
| r1/dataset_w1 | 2553 (2366 with the REOPEN fix) | rung 1 wave 1 | CHECK_OK, 49 videos on Machine B |
| r1/dataset_w1_mix | 5296 | r1/dataset_w1 (2553) + dataset_v2 (2743) | |
| r1/dataset_w2a | 1280 | rung 1 processes 30 to 39 | |
| r2/dataset_all | 5540 | rung 2 waves 1 and 2, 16 GB | CHECK_OK, 95 videos on each trainer |
| r2/dataset_mixB | 9373 | r2/dataset_all (5540) + r1 w1 (2553) + r1 w2a (1280) | CHECK_OK, 165 videos |
| r2/dataset_v2c | 5193 | rung 2 waves 1 and 2, converted again with the REOPEN fix | |
| r2/dataset_mixB2 | 9026 | r2/dataset_v2c (5193) + r1 w1 + r1 w2a | |

### 6.2 The training runs

Every run uses LeRobot ACT with the 16-wide `camera_box_target` context and the continuous
gripper. It saves a checkpoint every 50k steps and runs 400k steps. It serves open loop at 50
actions per forward.

| run | host | start | data | ledger entry |
| --- | --- | --- | --- | --- |
| r1 A | Machine A | T0 400k | r1/dataset_w1 (2553) | 16:22 |
| r1 B | Machine B | T0 400k | r1/dataset_w1_mix (5296) | 16:00, 38 step/s |
| r2 B | Machine B | r1 B 300k | r2/dataset_mixB (9373) | 19:45, 35 step/s |
| r2 A | Machine A | r1 B 300k | r2/dataset_all (5540) | 20:30 |
| r2 B2, round 2 | Machine B | r2 B 300k | r2/dataset_mixB2 (9026) | 05:35, training at 37.8 step/s |
| r2 A2, round 2 | Machine A | r2 A 350k | r2/dataset_all (5540) | 05:15 |
| round 3 | TBD | TBD | TBD | TBD |

At the launch of rung 2, r1 B 300k was the best rung 1 screen (173 of 250). Round 2 was planned on
the r2s re-record. That wave failed, so round 2 trains on the data already in place. The round
2 results are TBD.

## 7. The serving rules

Every rule is a loop switch of `scripts/mj_eval_heap.py`, off by default. The payload's
`inference.loop` block records each switch. The read tables use these protocol names:

| protocol | switches |
| --- | --- |
| MS6 | `--select_at_release --replan_on_switch --squeeze_floor --square_wrist` |
| cell | MS6 + `--scripted_rake` |
| part | cell + `--part_failures` |
| snap | part + `--pick_snap` |
| full | snap + `--dwell_close --adopt_held` |

Every MS7a policy read also passes `--tray --desk_boxes 1` for rung 1 or `--tray --desk_boxes 4`
for rung 2. It serves 50 actions per forward on held-out parts.

### 7.1 The scripted rake

The rule entered the ledger at 13:30, before any MS7 policy read. The teacher frees wall-stuck
parts with rakes, but the converter drops the rake frames, so the policy never learned to rake.
Without a rake, a read ends at the first wall-stuck state.

If the loop finds no target and the teacher plans a rake, `--scripted_rake` acts. The rig then
runs that rake as a takeover and hands the env back to the policy. The design left wall-stuck
parts to the policy, so this rule is a deviation. The rake is scripted, and every read table
below gives its rake count.

### 7.2 The failure memory per part

The first two rung 1 screens served the cell protocol. Arm A 50k stowed 62 of 250, and arm B
50k 102 of 250. Every screen episode ended `no_candidate` after about 50 cycles, with 16 to 27
of 50 parts stowed.

The loop booked each failed policy cycle in the teacher's 5 cm cell memory, with a cap of 3
failures per cell. The tray holds about a dozen cells. A policy that fails a third of its picks
closes every cell before the 65-cycle budget. The teacher rarely fails, so the cell rule suits
the teacher.

`--part_failures` books a failed cycle against its part, with the same cap of 3, so the
neighbours stay candidates. The 65-cycle budget and the bar of 1.3 cycles still bound the
retries. The rule entered the ledger at 17:35. A census at 17:50 found that all 5 episodes of
each arm ended `no_candidate` with 26 to 45 parts in reach. 15 to 35 of those parts were never
tried. The cell-protocol screens from 50k to 150k are void.

### 7.3 The pick snap

A forensic at 22:40 traced rung 1 arm B 100k on screen draw 0. The served row, the marker and
the box row matched the truth on every step of 325 cycles. The corpus was consistent: the
teacher closes a median 2.5 mm from the row. The forensic found three causes:

1. The final approach is imprecise. The command sits a median 0.9 cm and a p90 4.5 cm from the
   row on fresh cycles.
2. After a failed cycle, the next row arrives with the jaws closed at the heap. 79 of the 95
   closes 3 cm or more off follow a failed cycle.
3. A height fault depends on the checkpoint and the heap. In some heaps, the policy hovers 15 to
   25 mm above the part centre and closes on air.

The height fault collapsed arm B 350k to 78 of 250 (episodes 31, 2, 30, 11 and 4). At 300k the
same heaps sit 4 to 8 mm high, and at 100k about 2 mm off. This fault drives the screen swings
from 78 to 173 on one draw. A swap of box side, lighting, cameras, colors or physics does not
rescue those heaps.

`--pick_snap` edits the commanded pick point in the window of `--square_wrist`, within 40 mm of
the row centre in xy. It moves the xy toward the row centre by at most 3 mm per step. The height
comes down only within 1 cm of the centre, with the jaws open wider than the row's width. There
the snap lowers the point by at most 2 mm per step. The floor is the row centre plus 3 mm, and the snap never goes lower.

While the snap lowers the point, the jaws stay at the landing gap. An earlier hold of the jaws
without the lowering turned 40 of 44 `close_miss` cycles into `no_close`. The policy commands
the close that it believes is at grasp height. The snap stops at the stop flag of
`--square_wrist`.

The snap measured on draw 0 of seed 20261700 at 9bcb2ed, one rollout each:

| checkpoint | protocol without the snap | without | with the snap | cycles per stow with the snap |
| --- | --- | --- | --- | --- |
| r1 B 100k | part | 121 | 128 | |
| r1 B 300k | part | 173 | 203 | 1.33 |
| r1 B 350k | part | 78 | 205 | 1.33 |
| r1 A 350k | part | 161 | 199 | 1.37 |
| r2 B 300k | part | 45 | 205 (11 wrong-box parts) | 1.38 |

The 350k collapse recovers. Its median close height moves to -1.1 to +1.0 mm, against +15 to +25
mm without the snap. The rung 2 run stowed 42, 45, 45, 37 and 36 per episode.

The rule entered the ledger at 01:20: every later selection, checked and official read serves
`--pick_snap`. Draw 0 also served the development of the snap, so only the checked read
decides. A selection rule from 22:40 also stands. The ledger flags any checkpoint whose screen
shows an episode with a median close above +9 mm, and selection uses more than one draw.

### 7.4 The dwell close and the adopted part

A second forensic at 05:05 traced rung 2 arm B 300k on draws 1 and 3 under the snap protocol.
It measured levers on draw 2 only. It found these failures:

- `no_close`: the policy freezes at grasp height with the jaws open until the 800-step budget
  ends. Such a cycle spends a median 658 steps there. A stowed cycle waits a median 38 steps
  there (p95 134) before its own close. The snap hold is not the cause.
- The wrong box: the policy carries a part that is not the target to the target's box. The
  traces hold 10 double picks and 9 held non-targets. The box row is right on every step.
- The slopes: most are lost as wall-stuck leftovers after their 2 rakes.
- The leftovers: of 110 parts left at the episode ends, 66 lay at a wall with no grasp. 23 lay
  in a wrong box, 15 at the 3-failure cap and 5 out of reach.

`--dwell_close` counts the steps at grasp height with the jaws open. After 150 steps
(`DWELL_STEPS`), it commands the teacher's close gap and replans. It fires at most once per
cycle, and it holds the close for at most 150 steps.

If a lifted part that is not the target rises 5 cm, `--adopt_held` acts. The loop then serves
the box and the marker of that part, as the teacher's `adopt_held_part` does.

Both measured on draw 2, rung 2 arm B 300k, one rollout each:

| serving | stowed of 250 | cycles per correct stow | note |
| --- | --- | --- | --- |
| snap | 192 | 1.484 | the baseline |
| snap + `--dwell_close` | 208 | 1.173 | |
| snap + `--dwell_close`, second rollout | 208 | 1.293 | |
| snap + `--adopt_held` | 197 | 1.513 | wrong-box parts 13 to 3 |
| full | 210 | 1.181 | |

The rule entered the ledger at 05:05: every later MS7 read also serves `--dwell_close` and
`--adopt_held`. The rung 1 checked read ran before this rule, and it stays the rung 1 number
under the snap protocol. A re-read of r1 A 350k under the full protocol runs in a fresh TAG. The
report gives it beside the first read, not in its place.

### 7.5 What the rules read

| rule | inputs | simulator state |
| --- | --- | --- |
| `--square_wrist` | the served row, the policy's command, the measured tool pose | the stop flag: a held part from the simulator's contacts, or a 5 cm rise of the held part |
| `--pick_snap` | the served row, the FK of the command, the measured pick point, the measured jaw gap | the stop flag of `--square_wrist` |
| `--dwell_close` | the measured pick point, the served row, the measured jaw gap, the gap command | the stop flag of `--square_wrist` |
| `--adopt_held` | the lifted part and its box | the identity of the held part |
| `--scripted_rake` | the teacher's rake plan | the part poses |

`--dwell_close`, `--adopt_held` and `--square_wrist` read the held part and the jaw state from
the simulator. The real arm needs a hold signal and a held-part detector for them. The served
row stands in for the detector output of the real stack.

## 8. The reads

### 8.1 The screens

Each screen reads seed 20261700, draw 0, 250 held-out parts, on one lane of the arm's own host.
Screens rank checkpoints and decide nothing. The columns give the stowed parts, the cycles per
stow and the rakes.

The two void screens of rung 1 under the cell protocol: arm A 50k stowed 62 at 3.452 cycles
per stow with 31 rakes. Arm B 50k stowed 102 at 2.471 with 44 rakes. Arm B 50k ran 252 cycles,
and 154 of them ran out their 800-step budget.

Rung 1, part protocol:

| step | A stowed | A cycles per stow | A rakes | B stowed | B cycles per stow | B rakes |
| --- | --- | --- | --- | --- | --- | --- |
| 50k | 107 | 3.028 | 17 | 115 | 2.826 | 15 |
| 100k | 111 | 2.883 | 23 | 121 | 2.686 | 1 |
| 150k | 143 | 2.259 | 34 | 164 | 1.951 | 36 |
| 200k | 134 | 2.425 | 2 | 126 | 2.571 | 15 |
| 250k | 150 | 2.153 | 13 | 102 | 3.186 | 0 |
| 300k | 116 | 2.776 | 13 | 173 | 1.861 | 31 |
| 350k | 161 | 1.988 | 25 | 78 | 4.077 | 14 |
| 400k | 127 | 2.559 | 8 | 152 | 2.125 | 29 |

Rung 2, part protocol, stowed in the correct box:

| step | A stowed | A cycles per stow | A rakes | A wrong-box parts | B stowed | B cycles per stow | B rakes | B wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50k | 53 | 6.132 | 0 | 10 | 117 | 2.769 | 35 | 27 |
| 100k | 123 | 2.642 | 15 | 15 | 152 | 2.066 | 48 | 21 |
| 150k | 87 | 3.736 | 0 | 12 | 117 | 2.744 | 29 | 22 |
| 200k | 137 | 2.343 | 51 | 10 | 108 | 2.963 | 19 | 20 |
| 250k | 136 | 2.390 | 29 | 15 | 81 | 3.840 | 27 | 4 |
| 300k | 142 | 2.282 | 21 | 13 | 45 | 7.222 | 10 | 4 |
| 350k | 165 | 1.836 | 37 | 13 | 65 | 5.000 | 4 | 6 |
| 400k | 137 | 2.328 | 28 | 11 | 138 | 2.225 | 43 | 9 |

The census of rung 2 arm B 50k: `close_miss` 92, `held_no_lift` 46, `no_close` 31, `carrying`
17 and `lifted_lost` 9, with 13 `wrong_box` cycles of 324. The box row is mostly followed. The
pick is the bottleneck in both rungs.

Round 2, full protocol, stowed in the correct box:

| run | step | stowed | cycles per stow | rakes | wrong-box parts |
| --- | --- | --- | --- | --- | --- |
| r2 B (start of B2) | 300k | 204 | | | |
| r2 A2 | 50k | 202 | 1.139 | 74 | 9 |
| r2 B2 | 50k | 212 | 1.264 | 69 | 8 |
| r2 B2 | 100k | 200 | 1.155 | 70 | 7 |
| r2 A2 | 100k to 400k | TBD | TBD | TBD | TBD |
| r2 B2 | 150k to 400k | TBD | TBD | TBD | TBD |

The start row is draw 0 of the rung 2 checked read. The two r2 B2 screens and their start lie
within about 2 standard deviations of one another.

### 8.2 The selection reads

The first rung 1 selection ran at 20:30 on aarch64 CPU lanes, 4 each, under the part protocol.
It read r1 B 300k (screen 173), r1 B 150k (164) and r1 A 350k (161). A 350k joined because its
screen was within the noise of B 150k. Draws 1 to 3 hit the client's wall cap, and no read
counted.

Every counted selection read uses seed 20261700, draws 0 to 3, 1000 parts, the snap protocol and
commit 944cdc1:

| checkpoint | host | stowed of 1000 | Wilson 95 % | cycles per stow | per draw | rakes | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 A 350k | Machine A, 4 lanes | 820 | 0.795 to 0.843 | 1.338 | 201, 210, 202, 207 | 267 | 0 |
| r1 B 300k | Machine B, 4 lanes | 804 | | 1.443 | 192, 202, 204, 206 | 251 | 0 |
| r1 B 350k | Machine B, 4 lanes | 802 | | 1.463 | 197, 207, 193, 205 | 247 | 0 |
| r2 B 300k | Machine A, 4 lanes | 765 | 0.738 to 0.790 | 1.387 | 189, 185, 195, 196 | 261 | 50 |
| r2 A 350k | Machine A, 4 lanes | 765 | | 1.416 | 196, 190, 186, 193 | 274 | 42 |
| r2 B 400k | Machine B, 4 lanes | 704 | | 1.639 | 168, 171, 191, 174 | 248 | 56 |
| round 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The rung 1 rerun at 02:30 read r1 B 350k in place of r1 B 150k. The ledger gives no reason. The
snap run of r1 B 350k read 205 on draw 0. r1 A 350k won rung 1 selection.

Rung 2 selection read three checkpoints at 01:30. The part-protocol screens of r2 B 300k and
r2 B 400k read 45 and 138. r2 A 350k had the best rung 2 screen (165). The snap run of
r2 B 300k read 205 on draw 0 at 00:15. The ledger gives no other reason for the picks. r2 B 300k
and r2 A 350k tie at 765, and r2 B 300k took the checked read.

The census of r2 B 300k over 1061 cycles: 754 stowed, `no_close` 123, `close_miss` 81,
`held_no_lift` 49, `wrong_box` 22, `carrying` 19, `lifted_lost` 10 and `no_open` 3. `no_open`
means that the gripper command never read open. Per class: slope 197 of 320, tall 193 of 250,
narrow 132 of 155, wide 243 of 275. Episodes ended `no_candidate` with 31 to 45 parts stowed.
`wrong_box` is not the largest failure class, so the scripted-place rule does not fire.

### 8.3 The checked reads

Each checked read uses seed 20261700, draws 0 to 11, 3000 held-out parts and a fresh TAG.
Machine A runs draws 0 to 7 on 8 lanes, and Machine B runs draws 8 to 11 on 4 lanes.

| read | TAG | commit | protocol | stowed of 3000 | fraction | Wilson 95 % | cycles | cycles per stow | rakes | wrong-box parts | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1, r1 A 350k | r1A_350000_chk | 944cdc1 | snap | 2466 | 0.822 | 0.808 to 0.835 | 3334 | 1.352 | 802 | 0 | GATE_FAIL |
| rung 1 re-read, r1 A 350k | r1A_350000_chk2 | TBD | full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| rung 2, r2 B 300k | r2B_300000_chk | 2b87de2 | full | 2482 | 0.827 | 0.813 to 0.840 | 2959 | 1.192 | 758 | 66 | GATE_FAIL |
| rung 2, round 2 winner | TBD | TBD | full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The policy ids are
`act:~/data/maker-arm-ms7/r1/train_A/checkpoints/350000/pretrained_model+nas50+ctx16+gap`
and
`act:~/data/maker-arm-ms7/r2/train_B/checkpoints/300000/pretrained_model+nas50+ctx16+gap`.

Per draw, 250 parts each:

| read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1 | 202 | 218 | 194 | 208 | 215 | 207 | 188 | 203 | 211 | 212 | 198 | 210 |
| rung 1 re-read | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| rung 2 | 204 | 187 | 208 | 215 | 218 | 198 | 209 | 217 | 206 | 209 | 209 | 202 |

On rung 1, Machine A draws read 1635 of 2000 and the Machine B draws 831 of 1000.

The census of the two reads, as counts and as a share of all cycles:

| read | cycles | stowed | close_miss | held_no_lift | no_close | carrying | lifted_lost | no_open | wrong_box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1 | 3334 | 2364 (0.709) | 354 (0.106) | 246 (0.074) | 229 (0.069) | 74 (0.022) | 55 (0.016) | 12 (0.004) | 0 |
| rung 2 | 2959 | 2451 (0.828) | 278 (0.094) | 121 (0.041) | 41 (0.014) | 29 (0.010) | 21 (0.007) | 17 (0.006) | 1 |

On rung 1, 925 cycles ran to their 800-step budget. On rung 2, 481 did. The rung 1 read ran
without `--dwell_close`. The census counts cycles, and the grader counts parts at rest at the
episode end, so the two stowed counts differ.

The census books a `wrong_box` cycle only for a part that a `lifted_lost` cycle held last. On
rung 2, one cycle is `wrong_box`, but the grader finds 66 parts at rest in a wrong box.

Per class, over the same heaps:

| class | parts | rung 1 stowed | fraction | rung 2 correct | fraction | rung 2 teacher |
| --- | --- | --- | --- | --- | --- | --- |
| slope | 910 | 602 | 0.66 | 609 | 0.67 | 133 of 165, 0.81 |
| tall | 900 | 776 | 0.86 | 787 | 0.87 | 167 of 175, 0.95 |
| narrow | 415 | 370 | 0.89 | 363 | 0.87 | 74 of 77, 0.96 |
| wide | 775 | 718 | 0.93 | 723 | 0.93 | 169 of 183, 0.92 |

Slopes are 301 of the 518 parts that rung 2 loses and 308 of the 534 that rung 1 loses. A slope
one stud wide that lies flush on a vertical tray wall has no grasp. The rake cannot get behind
it. The teacher also stows slopes at only 0.81 on rung 2.

Rung 2 per box index: 588 of 740, 625 of 747, 605 of 727 and 664 of 786. The class map is drawn
per episode, so a box index holds a different class in each episode.

### 8.4 The official read

The official read runs the rung 2 checked winner on seed 20261800, draws 0 to 23, 6000 parts.
The design splits it with `fleet.py read` across Machine C (12 lanes), Machine A (8
lanes) and Machine B (4 lanes). The actual split is TBD. It passes at 5100 of 6000 and at most 1.3 cycles per correct stow.

| item | value |
| --- | --- |
| checkpoint | TBD |
| TAG, commit, protocol | TBD |
| stowed of 6000 | TBD |
| fraction, Wilson 95 % | TBD |
| cycles per correct stow | TBD |
| rakes, wrong-box parts | TBD |
| per draw | TBD |
| outcome (`PASS`, `PASS (scripted place)` or `FAIL`) | TBD |

## 9. Rejected levers

### 9.1 More rakes and the crowd start

The rake prototype `feat/ms7-rake-more` (3cda4c4) adds `--rake_cap N`, a larger rake cap per
part. It also adds `--rake_crowd`, a rake start between neighbours, and the census class
`left_at_wall`.

On the teacher (oracle driver, rung 2, three heaps of draw 2): the base read 141 of 150, the
crowd start 141, a cap of 4 143, and both together 143.

On the policy (r2 B 300k, draw 2, full protocol), against the reference of 210 of 250 at 1.181:

| serving | stowed of 250 | cycles per stow | note |
| --- | --- | --- | --- |
| `--rake_cap 4` | 209 | 1.244 | 108 rakes, 30 parts left at a wall, 20 of them after 4 rakes |
| `--rake_cap 4 --rake_crowd` | 202 | 1.371 | the cycle bar fails. 25 parts left at a wall. 4 parts that were never raked before were reached, and none was freed. |

The ledger records no policy number for the crowd start alone. Neither lever beats the
reference, so both were dropped at 06:20. The branch stays unpushed.

### 9.2 The base refresh

T5 (777) and T6 (853) both read under T0's 900 (section 5). Neither replaced the base.

## 10. Deviations from the design

| deviation | reason |
| --- | --- |
| Corpus seeds are run seed 20270000 + p and layout seed 20271000 + p, not 20267000 + p. | Corpus v3 of campaign n already used 20267000 + p. |
| Every policy read serves `--scripted_rake`. | The policy never learned to rake, because the converter drops the rake frames. The design left wall-stuck parts to the policy. |
| Selection, checked and official reads serve `--part_failures`. | The cell memory closed every tray cell before the cycle budget (section 7.2). |
| Every read from 01:20 serves `--pick_snap`. | The pick forensic found an imprecise approach and a height fault (section 7.3). |
| Every read from 05:05 serves `--dwell_close` and `--adopt_held`. | The second forensic found the freeze at grasp height and the carry of a non-target part (section 7.4). |
| The rung 1 checked read is under the snap protocol. A full-protocol re-read is reported beside it. | The read ran before the 05:05 rule. |
| The owner wired Machine B and Machine C into one LAN. | The design allowed no network change except the end of the Tailscale SSH check. The owner changed the network at 13:40, and corpora then moved to Machine B at LAN speed. |
| Rung 1 arm B mixes the wave with dataset_v2, not dataset_v5. | The ledger gives no reason. T5, the warm start on dataset_v5, read 777 at 12:55, before the pipeline was armed. |
| The first rung 2 round trains on both waves, 120 heaps, not on wave 1 alone. | Both waves finished before rung 1's best checkpoint existed. |
| Rung 2 starts on 2026-09-23 from r1 B 300k, the best screen at launch, before any rung 1 selection read. Arm A takes the same start. | The RTX 5080 does not idle. The rung 2 corpus is itself a tray-pick corpus, so rung 2 training keeps training the pick. |
| Rung 1 selection read three checkpoints, not two. | r1 A 350k's screen (161) was within the noise of r1 B 150k (164). |
| Rung 2 selection did not read the two best screens. | The ledger gives no reason (section 8.2). |
| Round 2 trains on data already in place. | The r2s re-record failed on host load, and wave 3 was lost to a full disk. |
| Selection and checked reads run on Machine A and Machine B, not on aarch64 CPU policy servers. | The first selection reads on aarch64 CPU lanes hit the 130-minute wall cap at 5.8 steps/s per lane. |
| The sorting teacher had a 100-part smoke before its 600-part read, not a 150-part first read. | The ledger gives no reason. |
| The wall band reaches the grasp's own finger reach plus 5 mm, not `OPEN_HALF_GAP`. | On the 0.13 x 0.20 m tray, `OPEN_HALF_GAP` bands every part within 57 mm of two walls, half the tray, where no heading survives. |
| The teacher turns the release roll along the carry (48bc50c). | The owner's review of a teaching episode. The tool stood still over the box for 37 to 50 steps per cycle. |

## 11. The code

The code is one stack of one-concern branches on main, all pushed:

| order | branch | tip | what it adds |
| --- | --- | --- | --- |
| 1 | `feat/fleet` | b843d3d | `scripts/fleet.py` and its tests, the bundle fetch, `--hosts` and the sync guard |
| 2 | `feat/ms7-desk-boxes` | 0602635 | the H1 tray footprint constants and the desk boxes in the grader and the scene. The release over each part's own box, `--desk_boxes`, the certified side band and the box sizes. |
| 3 | `feat/ms7-tray` | 13bd798 | `--tray`, the wall-ranked grasps, the rake, the end-to-end grasps and the wedge. The 45 degree band, the jam recovery and the rake as its own takeover. |
| 4 | `feat/ms7-classes` | 22153d5 | `part_class`, the class-to-box map, the target's box on the context row, the `wrong_box` census, the rows per class and per box, `--scripted_place` |
| 5 | `feat/ms7-rake-serving` | a6b631e | the converter without the rake frames, `--scripted_rake` |
| 6 | `feat/ms7-part-failures` | be86bc8 | `--part_failures` |
| 7 | `feat/ms7-pick-snap` | 9bcb2ed | `--pick_snap` |
| 8 | `feat/ms7-smooth-carry` | 48bc50c | the release roll along the carry |
| 9 | `fix/convert-reopen-fragments` | 944cdc1 | the converter without the REOPEN fragments |
| 10 | `feat/ms7-serving-2` | 2b87de2 | `--dwell_close` and `--adopt_held` |
| 11 | `docs/ms7-sorting` | | this report |

Every new switch is off by default, so the MS6 protocol stays reproducible. At 2b87de2 the CPU
suite read 1073 tests. Two prototypes stay off the stack: `feat/ms7-rake-more` (3cda4c4,
rejected) and `feat/ms7-tray-lean` (59ccf5f, TBD). No PR is open yet. The stack goes up with
`GH_REPO=youtalk/maker-arm-lab gh stack`. The PR numbers are TBD.

## 12. What is still open

- The rung 1 re-read under the full protocol, TAG r1A_350000_chk2: TBD.
- Round 2: the remaining screens, the selection reads and a checked read: TBD.
- Round 3 and later rounds: TBD. Wave 3 was lost, so round 3 needs a new corpus wave.
- The tray-wall lever: TBD. `--tray_lean DEG` (59ccf5f, off by default) leans the H1 walls out
  about the interior edge, with a 20 mm rim. Teacher reads at 30, 45 and 60 degrees on the three
  heaps of draw 2 run now. The base reads 141 of 150 there, with slopes at 33 of 39.
- The bar for a re-record with the new tray: the teacher at 0.95 or more on 600 parts, with
  slopes near the other classes. The re-record and its round: TBD.
- The official read of rung 2 and the milestone outcome: TBD.
- The real arm. `--square_wrist`, `--dwell_close` and `--adopt_held` read the held part and the
  jaw state from the simulator. The real arm needs a hold signal and a held-part detector.
- The scripted rake. It plans on the simulator's part poses, and the policy does not rake.
- The overhead camera still frames the old shelf region. The tray heap spans about 30 x 20 px of
  the 224 px frame.
- The release roll along the carry is on the stack, but no MS7a corpus uses it.
- The MS6 debts move on: the learned heading and a noise model on the target row.
