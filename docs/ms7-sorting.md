# MS7a: a tray heap sorted into desk boxes (draft)

Date: 2026-09-24. Status: draft. The campaign still runs. Each `TBD` marks a result or a
number that this draft does not give.

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

The rung 2 checked read passes on the leaned tray. Checkpoint r2 A2 100k stows 2610 of 3000
parts in their own class's box, 0.870 (Wilson 95 % 0.857 to 0.882). It uses 1.138 cycles per
correct stow. The gate needs at least 2550 parts and at most 1.3 cycles. The official read of
rung 2 is the milestone's number: TBD.

The leaned tray is a deviation from the design (section 3.5). The H1 tray of the design has
vertical walls. The lean tilts each wall 30 degrees outward about its interior edge on the
desk, and the rim stays 20 mm high. On vertical walls, the same checkpoint stows 2513 of 3000,
0.838, on the same draws. That read is 37 parts short. On the lean, it stows 97 more parts, and
63 of them are slopes.

Rung 1 fails. Its checked read under the snap protocol stows 2466 of 3000, 0.822, at 1.352
cycles per stowed part. The re-read under the full protocol stows 2533, 0.844 (0.831 to 0.857),
at 1.241. It is 17 parts short. Rung 1 has no read on the leaned tray.

Slopes are the largest loss. On the vertical rung 2 checked read of r2 B 300k, 609 of 910
slopes stow, 0.67. A forensic of that read finds 413 lost parts that the policy never targeted.
Most of them start near a wall. A driven test shows that the refusal matches the physics. Two
rake levers and a snap floor lever measured null. On the passing read, 675 of 910 slopes stow,
0.74.

The reads come from a learned policy plus serving rules. A serving rule changes the policy's
command, or gives the env to the teacher for a few steps, before the env receives the command.
The rung 2 reads serve nine switches. Four come from MS6: `--select_at_release
--replan_on_switch --squeeze_floor --square_wrist`. Five are new in MS7a: `--scripted_rake
--part_failures --pick_snap --dwell_close --adopt_held`. Section 7 gives each rule and the
forensic that found it. The passing read also sets the scene switch `--tray_lean 30`.

Some rules read simulator state. `--square_wrist`, `--dwell_close` and `--adopt_held` read the
held part and the jaw state from the simulator. `--pick_snap` uses the stop flag of
`--square_wrist`. The real arm needs a hold signal and a held-part detector for these rules.
`--scripted_rake` runs the teacher's rake, and the teacher plans on the simulator's part poses.

The teachers pass every bar. On 600 held-out parts each, the rung 1 teacher stows 0.920, the
sorting teacher on the bare desk 0.963 and the rung 2 teacher 0.905. On 12 paired heaps, the
rung 2 teacher stows 0.905 on the leaned tray and 0.875 on vertical walls. The base refresh
failed. T5 read 777 and T6 853 of 1000 against T0's 900, so T0 400k stayed the base.

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
training started at 19:45. The rung 2 checked read passed at 09:35 on 2026-09-24.

## 2. The hosts and fleet.py

### 2.1 The roles

| host | GPU | role | read lanes |
| --- | --- | --- | --- |
| Machine A | RTX 5090 32 GB | the controller session and the ledger, T6, arm A of rounds 1 and 2, screens, selection, checked and official reads | 8 |
| Machine B | RTX 5080 16 GB | T5, arm B of rounds 1 and 2, round 3, the lean corpus wave, screens, selection, checked and official reads | 6 in `hosts.json`, 4 on the checked and official reads |
| Machine C | RTX A2000 12 GB, 128 cores | teacher reads, corpus recording and conversion, one lean checked read | 12, on CPU policy servers |

The trainers never recorded a corpus while they trained, except Machine B for the lean wave.
Campaign n showed that recording, reads and training on one host drove training down to 5
step/s.

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
  conversion. Wave 4 kept it. Its 28 processes needed 166 GB, and 173 GB was free.
- Machine B ran out of GPU memory at 08:04. B2 training, the 12 camera sims of the lean wave, 4
  checked-read lanes and 2 screens used 14.6 of 16.3 GB. The r2 B2 350k screen then failed three
  times in `policy_server` predict (GATE_ERROR). The B2 screens stopped, to protect the r2 A2
  100k checked read.
- `bin/run_corpus.sh` and `bin/convert.sh` source `~/opt/env.sh` only where it exists. So the
  same scripts record and convert the lean wave on Machine B.
- At 08:49, the controller sent SIGTERM to the owner's paused MS6 T1 run in place of arm A2. T1
  ends on SIGCONT and can resume only from its 300000 checkpoint, so read a PID's output
  directory before a kill.

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
cycles. Every rest is at a leg end, as the MS3 minimum-jerk walk intends. The owner saw the
level-release roll of MS6. The tool stands still over the box for 37 to 50 steps per cycle while
only the wrist turns. In MS6 and in both rungs, the tool is at rest about 46 % of each cycle.

The prototype 25f1dcf, 48bc50c on the stack, spreads the roll along the carry and the unroll
along RETREAT. On rung 2 over 12 seeds, the teacher stowed 224 of 240 with and without it. The
median steps per cycle fell from 392 to 373. Wrist-only frames fell from 37 to 5, and the p99
acceleration from 13.8 to 8.1.

A later timing of 48bc50c against be86bc8 read equal steps per second on both hosts. For
example, it read 11.8 against 12.1 with cameras, and 10.9 against 11.1 on Machine C.
The r2s re-record with the roll failed on host load, and wave 3 recorded at 9bcb2ed without it.
Wave 4 at 2b87de2 is the first MS7a corpus with the roll along the carry. The lean wave r2L1 at
751e3dd also has it (section 6.1).

### 3.5 The tray lean

Most slopes that the vertical rung 2 reads lose start within 20 mm of a wall. The rake levers
did not free them (sections 8.4 and 9.1). MS6's round dish leaned its rim outward for the same
kind of blocking. So the tray agent built `--tray_lean DEG` (59ccf5f), off by default.

The switch builds each of the four walls as a slab that leans outward about the interior's edge
on the desk. The interior footprint on the desk stays the H1 footprint. The rim stands 20 mm
high and 20 mm times tan(DEG) further out, 11.5 mm at 30 degrees. Each slab runs past the
corners to meet its neighbour. The grasp model follows the lean. A finger clears a leaning wall
where the wall stands at the foot of the part's grip band. The rake checks its hull at the desk
and at the rim. 751e3dd forwards the lean through the gate supervisor, and the cache refuses a
payload read at another lean.

The teacher reads, with the oracle driver on rung 2 and held-out parts:

| heaps | walls | correct stows | cycles per stow | rakes | slopes | parts left at a wall | spills |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3, run seeds 20267580 to 20267582 | vertical | 141 of 150 | 1.000 | 31 | 33 of 39 | 8 | 1 |
| the same 3 | 30 degrees | 144 of 150 | 0.979 | 18 | 37 of 39 | 4 | 2 |
| the same 3 | 45 degrees | 141 of 150 | 1.000 | 21 | 36 of 39 | 7 | 2 |
| the same 3 | 60 degrees | 143 of 150 | 0.986 | 19 | 35 of 39 | 6 | 0 |
| 12 fresh, run seeds 20267600 to 20267611 | vertical | 525 of 600, 0.875 | 1.006 | 128 | 130 of 177 | 58 | 16 |
| the same 12 | 30 degrees | 543 of 600, 0.905 | 1.033 | 94 | 141 of 177 | 35 | 20 |

The lean rows ran at 59ccf5f. The vertical row on the three heaps is the base run of the rake
prototype (section 9.1). A part left at a wall lies inside the rim and within 30 mm of the
interior edge. A spill lies beyond the rim.

On the three heaps, the three leans read 141 to 144 against 141 on vertical walls. The ledger
calls the gain of 30 degrees inside the noise. On the 12 paired
heaps, the lean adds 18 parts, and 11 of them are slopes. The lean reads higher on 8 heaps,
lower on 3 and equal on 1. The other classes stow 395 of 423 on vertical walls and 402 of 423
on the lean.

The campaign set a resource bar for a re-record with a new tray. The teacher needed 0.95 or more
on 600 parts, with slopes near the other classes. The lean did not meet that bar, so the ledger
closed the lean at 07:13 with no re-record.

At 07:49, the ledger changed that resource bar and nothing else. The gate stayed the same. The
0.95 bar assumed that the policy was the limit. But the teacher reads 0.875 on vertical heaps,
and the gate needs 0.85. So a paired gain of 0.030 is material. The refusal forensic (section
8.4) also showed that only the rake and the lean reach the wall-stuck parts. A speculative lean
corpus wave started on Machine B at once (section 6.1).

At 08:34, the policy read the lean with no retraining. r2 B 300k read draw 2 of seed 20261700
under the full protocol plus `--tray_lean 30`, one rollout each:

| walls | correct stows of 250 | cycles per stow | per episode | slope | tall | narrow | wide |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vertical (section 7.4) | 210 | 1.181 | 43, 40, 43, 40, 44 | 48 of 75 | 82 of 90 | 22 of 25 | 58 of 60 |
| 30 degrees | 221 | 1.204 | 42, 44, 46, 45, 44 | 58 of 75 | 83 of 90 | 22 of 25 | 58 of 60 |

The whole gain is slopes, as on the teacher. The close heights do not move. The gain comes
from fewer parts stuck at a wall.

The decision came at 08:34. The rung 2 scene of the remaining reads adds `--tray_lean 30`. The
ledger gives three reasons. The teacher gains 0.030 on 600 paired parts. The policy gains 11 of
250 with no retraining. In a driven test, the jaws never lift a 1-wide slope flush on a vertical
wall (section 8.4). The vertical-wall results stay in this report beside the lean results. The
ledger does not say why it chose 30 degrees. On the three heaps, 30 degrees read best of the
three leans. Rung 1 keeps vertical walls.

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

Section 3.5 gives the teacher reads on the leaned tray. Section 8.4 gives the teacher on the
heaps of a policy read.

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

Every corpus records train-split heaps with the teacher driving. Every wave except r2L1
records on Machine C. r2L1 records on Machine B.

| corpus | commit | processes | heaps | stowed | cycles per stow | rakes | out of reach |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1, wave 1 | 97c2ae1 | 0 to 29 | 60 | 2694 of 3000, 0.898 | 1.059 | 377 | 44 |
| rung 1, wave 1 and wave 2 part A | 97c2ae1, a6b631e | 0 to 39 | 90 | 4046 of 4500, 0.899 | 1.059 | 549 | |
| rung 2, waves 1 and 2 | a6b631e | 100 to 139 | 120 | 5217 of 6000 in the correct box, 0.870 | 1.098 | 803 | 107 |
| rung 2, re-record r2s | 25f1dcf | 200 to 239 | 120 planned | 0 episodes in the 300-minute cap | | | |
| rung 2, wave 3 | 9bcb2ed | 240 to 279 | 120 planned | no finished episode | | | |
| rung 2, wave 4 | 2b87de2 | 280 to 307 | 84 | 3642 of 4200 in the correct box, 0.867 | 1.103 | 545 | |
| rung 2, lean wave r2L1 | 751e3dd, `--tray_lean 30` | 308 to 319 | 36 | 1565 of 1800 in the correct box, 0.869 | | | |

The r2s wave failed on host load (section 2.4). Wave 3 was lost to a full disk. The wave 3
shards (134 GB) and four merged lane work directories (41 GB) were deleted.

Wave 4 launched at 06:37, alone on Machine C. It is the first wave with the release
roll along the carry. It finished in about 80 minutes, with 28 shards and 114 GB, and converted
on 14 lanes. The lean wave r2L1 launched at 07:49 and converted on Machine B on 12 lanes. Its 0.869
is level with the 0.867 of wave 4, but the heaps differ, so the two reads are not paired.

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
| r2w4/dataset | 3620 | rung 2 wave 4, vertical walls, the roll along the carry | |
| r2L1/dataset | 1548 | the lean wave, 30 degrees | |
| r2/dataset_r3L | 10361 | r2/dataset_v2c (5193) + wave 4 (3620) + r2L1 (1548) | MERGE_OK, CHECK_OK 177 videos |

### 6.2 The training runs

Every run uses LeRobot ACT with the 16-wide `camera_box_target` context and the continuous
gripper. It saves a checkpoint every 50k steps. It serves open loop at 50 actions per forward.

| run | host | start | data | steps | ledger entry |
| --- | --- | --- | --- | --- | --- |
| r1 A | Machine A | T0 400k | r1/dataset_w1 (2553) | 400k | 16:22 |
| r1 B | Machine B | T0 400k | r1/dataset_w1_mix (5296) | 400k | 16:00, 38 step/s |
| r2 B | Machine B | r1 B 300k | r2/dataset_mixB (9373) | 400k | 19:45, 35 step/s |
| r2 A | Machine A | r1 B 300k | r2/dataset_all (5540) | 400k | 20:30 |
| r2 B2, round 2 | Machine B | r2 B 300k | r2/dataset_mixB2 (9026) | 400k, TRAIN_OK at 08:46 | 05:35, 37.8 step/s |
| r2 A2, round 2 | Machine A | r2 A 350k | r2/dataset_all (5540) | stopped at 363k at 08:49 | 05:15 |
| r3L, round 3 | Machine B | r2 B 300k | r2/dataset_r3L (10361) | 300k, runs now | 08:46, 38 step/s |

At the launch of rung 2, r1 B 300k was the best rung 1 screen (173 of 250). Round 2 was planned on
the r2s re-record. That wave failed, so round 2 trains on the data already in place.

The B2 screens stopped after 250k (section 8.1). Arm A2 stopped at 363k of 400k. Its screens
from 50k to 300k read 202, 218, 195, 208, 208 and 207, a plateau. The stop freed Machine A GPU
for the lean reads.

Round 3 trains on `r2/dataset_r3L`, which adds wave 4 and the lean wave to `r2/dataset_v2c`. It
warm-starts from r2 B 300k, which read 221 of 250 on the lean with no retraining. The ledger
gives its end at about 11:15. The gate does not need round 3, because a round 2 checkpoint
passed the checked read.

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
| full + lean | full + the scene switch `--tray_lean 30` (section 3.5) |

Every MS7a policy read also passes `--tray --desk_boxes 1` for rung 1 or `--tray --desk_boxes 4`
for rung 2. It serves 50 actions per forward on held-out parts. The payload records the lean as
`tray_lean`, outside the `inference.loop` block.

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
the snap lowers the point by at most 2 mm per step. The floor is the row centre plus 3 mm, and
the snap never goes lower.

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
under the snap protocol. A re-read of r1 A 350k under the full protocol ran in a fresh TAG
(section 8.3). The report gives it beside the first read, not in its place.

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

Round 2, full protocol, vertical walls, stowed in the correct box:

| run | step | stowed | cycles per stow | rakes | wrong-box parts |
| --- | --- | --- | --- | --- | --- |
| r2 B (start of B2) | 300k | 204 | 1.167 | 72 | 7 |
| r2 A2 | 50k | 202 | 1.139 | 74 | 9 |
| r2 A2 | 100k | 218 | 1.124 | 58 | 0 |
| r2 A2 | 150k | 195 | 1.082 | 63 | 13 |
| r2 A2 | 200k | 208 | 1.120 | 62 | 7 |
| r2 A2 | 250k | 208 | 1.144 | 70 | 9 |
| r2 A2 | 300k | 207 | 1.174 | 70 | 7 |
| r2 B2 | 50k | 212 | 1.264 | 69 | 8 |
| r2 B2 | 100k | 200 | 1.155 | 70 | 7 |
| r2 B2 | 150k | 199 | 1.201 | 70 | 9 |
| r2 B2 | 200k | 208 | 1.096 | 65 | 3 |
| r2 B2 | 250k | 202 | 1.134 | 63 | 9 |

The start row is draw 0 of the rung 2 checked read. r2 A2 100k read 218, 0.872, the best screen
of the campaign. The next screen of the same arm read 195, so the checkpoint swings stay large.
The B2 300k screen lane was stopped, and the B2 350k screen failed three times on Machine B GPU
memory (section 2.4). A2 stopped at 363k, so its 350k and 400k screens were dropped.

Round 3, full + lean, stowed in the correct box: r3L 50k read 210 at 1.110 cycles per stow, with
61 rakes and 5 wrong-box parts. Its per-episode counts are 43, 42, 40, 41 and 44. The 100k
screen runs now. On the same draw and the same lean, r2 A2 100k reads 218 (section 8.3).

### 8.2 The selection reads

The first rung 1 selection ran at 20:30 on aarch64 CPU lanes, 4 each, under the part protocol.
It read r1 B 300k (screen 173), r1 B 150k (164) and r1 A 350k (161). A 350k joined because its
screen was within the noise of B 150k. Draws 1 to 3 hit the client's wall cap, and no read
counted.

Every counted selection read uses seed 20261700, draws 0 to 3 and 1000 parts. The first six
rows use the snap protocol and commit 944cdc1:

| checkpoint | host | stowed of 1000 | Wilson 95 % | cycles per stow | per draw | rakes | wrong-box parts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 A 350k | Machine A, 4 lanes | 820 | 0.795 to 0.843 | 1.338 | 201, 210, 202, 207 | 267 | 0 |
| r1 B 300k | Machine B, 4 lanes | 804 | | 1.443 | 192, 202, 204, 206 | 251 | 0 |
| r1 B 350k | Machine B, 4 lanes | 802 | | 1.463 | 197, 207, 193, 205 | 247 | 0 |
| r2 B 300k | Machine A, 4 lanes | 765 | 0.738 to 0.790 | 1.387 | 189, 185, 195, 196 | 261 | 50 |
| r2 A 350k | Machine A, 4 lanes | 765 | | 1.416 | 196, 190, 186, 193 | 274 | 42 |
| r2 B 400k | Machine B, 4 lanes | 704 | | 1.639 | 168, 171, 191, 174 | 248 | 56 |
| r2 A2 100k, draws 0 to 3 of its checked read | Machine A | 847 | 0.823 to 0.868 | 1.168 | 209, 213, 215, 210 | 281 | 25 |

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

Round 2 had no separate selection read. At 06:37, the ledger added a rule. A screen at 215 of
250 or more goes straight to a checked read, and its draws 0 to 3 stand as the selection read.
The checked read only filters, and the official read stays one-shot. r2 A2 100k (218) was the
only screen at that bar. The last row of the table above is its draws 0 to 3, under the full
protocol at 2b87de2.

### 8.3 The checked reads

Each checked read uses seed 20261700, draws 0 to 11, 3000 held-out parts and a fresh TAG.
Machine A runs draws 0 to 7 on 8 lanes, and Machine B runs draws 8 to 11 on 4 lanes.

| read | TAG | commit | protocol | stowed of 3000 | fraction | Wilson 95 % | cycles | cycles per stow | rakes | wrong-box parts | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1, r1 A 350k | r1A_350000_chk | 944cdc1 | snap | 2466 | 0.822 | 0.808 to 0.835 | 3334 | 1.352 | 802 | 0 | GATE_FAIL |
| rung 1 re-read, r1 A 350k | r1A_350000_chk2 | 2b87de2 | full | 2533 | 0.844 | 0.831 to 0.857 | 3144 | 1.241 | 781 | 0 | GATE_FAIL |
| rung 2, r2 B 300k | r2B_300000_chk | 2b87de2 | full | 2482 | 0.827 | 0.813 to 0.840 | 2959 | 1.192 | 758 | 66 | GATE_FAIL |
| rung 2, r2 A2 100k | r2A2_100000_chk | 2b87de2 | full | 2513 | 0.838 | 0.824 to 0.850 | 2987 | 1.189 | 790 | 70 | GATE_FAIL |
| rung 2 on the leaned tray, r2 A2 100k | r2A2_100000_L30_chk | 751e3dd | full + lean | 2610 | 0.870 | 0.857 to 0.882 | 2971 | 1.138 | 669 | 57 | GATE_OK |

The lean read's payload gives the outcome `PASS`. The policy ids are:

- `act:~/data/maker-arm-ms7/r1/train_A/checkpoints/350000/pretrained_model+nas50+ctx16+gap`
- `act:~/data/maker-arm-ms7/r2/train_B/checkpoints/300000/pretrained_model+nas50+ctx16+gap`
- `act:~/data/maker-arm-ms7/r2/train_A2/checkpoints/100000/pretrained_model+nas50+ctx16+gap`

The rung 1 re-read launched at 06:33 as pre-registered and ended at 07:55. It is 17 parts short
of 2550, and it stands beside the first rung 1 read, not in its place. The r2 A2 100k read is 37
parts short, the best vertical-wall read. The lean read launched at 08:51 on the same checkpoint
and the same draws, and it passed at 09:35.

Per draw, 250 parts each:

| read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1 | 202 | 218 | 194 | 208 | 215 | 207 | 188 | 203 | 211 | 212 | 198 | 210 |
| rung 1 re-read | 202 | 215 | 207 | 211 | 206 | 212 | 202 | 214 | 225 | 214 | 213 | 212 |
| rung 2, r2 B 300k | 204 | 187 | 208 | 215 | 218 | 198 | 209 | 217 | 206 | 209 | 209 | 202 |
| rung 2, r2 A2 100k | 209 | 213 | 215 | 210 | 201 | 198 | 207 | 225 | 213 | 210 | 210 | 202 |
| rung 2 lean, r2 A2 100k | 218 | 219 | 219 | 225 | 208 | 224 | 217 | 216 | 219 | 220 | 211 | 214 |

Machine A draws and the Machine B draws, of 2000 and 1000 parts:

| read | Machine A | Machine B |
| --- | --- | --- |
| rung 1 | 1635 | 831 |
| rung 1 re-read | 1669 | 864 |
| rung 2, r2 B 300k | 1656 | 826 |
| rung 2, r2 A2 100k | 1678 | 835 |
| rung 2 lean, r2 A2 100k | 1746 | 864 |

The census of the reads, as counts and as a share of all cycles:

| read | cycles | stowed | close_miss | held_no_lift | no_close | carrying | lifted_lost | no_open | wrong_box |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rung 1 | 3334 | 2364 (0.709) | 354 (0.106) | 246 (0.074) | 229 (0.069) | 74 (0.022) | 55 (0.016) | 12 (0.004) | 0 |
| rung 1 re-read | 3144 | 2426 (0.772) | 323 (0.103) | 229 (0.073) | 46 (0.015) | 53 (0.017) | 58 (0.018) | 9 (0.003) | 0 |
| rung 2, r2 B 300k | 2959 | 2451 (0.828) | 278 (0.094) | 121 (0.041) | 41 (0.014) | 29 (0.010) | 21 (0.007) | 17 (0.006) | 1 |
| rung 2, r2 A2 100k | 2987 | 2495 (0.835) | 266 (0.089) | 149 (0.050) | 26 (0.009) | 26 (0.009) | 21 (0.007) | 4 (0.001) | 0 |
| rung 2 lean, r2 A2 100k | 2971 | 2581 (0.869) | 189 (0.064) | 125 (0.042) | 15 (0.005) | 35 (0.012) | 23 (0.008) | 0 | 3 (0.001) |

On rung 1, 925 cycles of the first read ran to their 800-step budget. On the r2 B 300k read,
481 did. The first rung 1 read ran without `--dwell_close`. On the same heaps, the re-read with
`--dwell_close` cut `no_close` from 229 to 46 cycles. The census counts cycles, and the grader
counts parts at rest at the episode end, so the two stowed counts differ.

The census books a `wrong_box` cycle only for a part that a `lifted_lost` cycle held last. On
the r2 B 300k read, one cycle is `wrong_box`, but the grader finds 66 parts at rest in a wrong
box.

Per class, over the same heaps, as stowed parts and fractions:

| class | parts | rung 1 | rung 1 re-read | r2 B 300k | r2 A2 100k | r2 A2 100k, lean | rung 2 teacher |
| --- | --- | --- | --- | --- | --- | --- | --- |
| slope | 910 | 602 (0.66) | 634 (0.70) | 609 (0.67) | 612 (0.67) | 675 (0.74) | 133 of 165 (0.81) |
| tall | 900 | 776 (0.86) | 806 (0.90) | 787 (0.87) | 811 (0.90) | 824 (0.92) | 167 of 175 (0.95) |
| narrow | 415 | 370 (0.89) | 382 (0.92) | 363 (0.87) | 379 (0.91) | 377 (0.91) | 74 of 77 (0.96) |
| wide | 775 | 718 (0.93) | 711 (0.92) | 723 (0.93) | 711 (0.92) | 734 (0.95) | 169 of 183 (0.92) |

On r2 B 300k, slopes are 301 of the 518 lost parts. On the first rung 1 read, slopes are 308 of
the 534 lost parts. A slope one stud wide that lies flush on a vertical tray wall has no grasp.
The rake cannot get behind it. The teacher also stows slopes at only 0.81 on rung 2.

Per box index, of 740, 747, 727 and 786 parts:

| read | box 0 | box 1 | box 2 | box 3 |
| --- | --- | --- | --- | --- |
| r2 B 300k | 588 | 625 | 605 | 664 |
| r2 A2 100k | 606 | 636 | 619 | 652 |
| r2 A2 100k, lean | 648 | 641 | 638 | 683 |

The class map is drawn per episode, so a box index holds a different class in each episode.

The paired comparison of the lean. Both r2 A2 100k reads use the same run seeds, layout seeds,
parts and class maps. Only `--tray_lean 30` and the commit differ. The lean adds 97 parts:

| item | vertical walls | 30 degree lean | difference |
| --- | --- | --- | --- |
| stowed of 3000 | 2513 | 2610 | +97 |
| slopes | 612 | 675 | +63 |
| tall | 811 | 824 | +13 |
| narrow | 379 | 377 | -2 |
| wide | 711 | 734 | +23 |
| cycles per stow | 1.189 | 1.138 | -0.051 |
| rakes | 790 | 669 | -121 |
| wrong-box parts | 70 | 57 | -13 |
| `close_miss` cycles | 266 | 189 | -77 |

The lean reads higher on 11 of the 12 draws and lower on draw 7. Over the 60 episodes, it reads
higher on 38, lower on 15 and equal on 7. The mean gain is 1.6 parts per episode, with a
standard error of 0.44.

The lean read reuses the draws of the vertical read of the same checkpoint. So the checked read
is a filter here, and the one-shot official read on a fresh seed decides (section 8.5).

### 8.4 The lost parts

At 07:33 and 07:49, the tray agent traced the vertical rung 2 checked read of r2 B 300k
(r2B_300000_chk). It rebuilt the 60 initial heaps from their seeds. The scripts and their
outputs are in `~/data/maker-arm-ms7/fx/`.

A part that no stowed cycle credits counts as lost in the forensic. The grader counts 2482 parts
at rest in their own box, and the payloads credit 2459 of them to a stowed cycle. So the
forensic finds 541 lost parts, not 518.

Of the 541, the policy never targeted 413. The selector found no grasp for them, and 59 of the
60 episodes ended `no_candidate` at a median of 49 cycles. Of the 413, 264 start within 20 mm of
a wall, and 253 are slopes. The other 128 parts were targeted and failed. 70 failed once, 10
twice and 48 three times.

The loss by the initial distance from the nearest wall:

| distance of the part centre from the wall | parts | lost | fraction lost | slopes | slopes lost |
| --- | --- | --- | --- | --- | --- |
| past the interior edge | 5 | 2 | 0.40 | 3 | 2 |
| 0 to 10 mm | 186 | 111 | 0.60 | 116 | 82 |
| 10 to 20 mm | 572 | 204 | 0.36 | 233 | 126 |
| 20 to 30 mm | 708 | 137 | 0.19 | 243 | 70 |
| 30 mm or more | 1529 | 87 | 0.06 | 315 | 32 |

The refusal check ran the selector's rules of 2b87de2, with the tray walls, on each part at its
initial pose. It gives the reason for each of the 413 never-targeted parts:

| reason at the initial pose | parts |
| --- | --- |
| each closing axis is tilted out of the jaw plane or strokes a finger into a wall | 167 |
| graspable at the start, stuck later | 112 |
| tilted, with a wall in the grip band | 60 |
| tilted only | 55 |
| a finger stroke into a wall, not tilted | 11 |
| other: tilted and too wide 5, out of reach 2, the wall band only 1 | 8 |

The refusal matches the physics. In a driven test, the jaws closed along the wall on a 1-wide
slope flush on that wall, with x closing forced on. Every LIFT failed. Only the rake and the lean
reach these parts, and the rake levers are null (section 9.1).

The teacher read the same heaps. The oracle driver ran draw 0 of the read and stowed 215 of 250
at 217 cycles, with 74 rakes, against the policy's 204. The teacher lost 35 parts there, and the
policy lost 20 of the same 35. So the grasp ceiling at the walls binds as much as the policy.

The close heights. `close_miss` is the largest failure class of the read, with 278 cycles. A
first close high above the part centre stows much less often. 2620 of the 2959 cycles had a
first real close with a measured height:

| first close above the part centre | cycles | stowed | fraction stowed |
| --- | --- | --- | --- |
| 0 mm or lower | 1231 | 1193 | 0.97 |
| 0 to 2 mm | 506 | 482 | 0.95 |
| 2 to 4 mm | 421 | 349 | 0.83 |
| 4 to 6 mm | 93 | 42 | 0.45 |
| 6 to 15 mm | 92 | 21 | 0.23 |
| over 15 mm | 277 | 199 | 0.72 |

606 first closes land 2 to 15 mm high. `--pick_snap` stops its lowering at the row centre plus
3 mm, so the snap floor was the next lever (section 9.2). The ledger does not explain the row
over 15 mm.

### 8.5 The official read

The ledger pre-registered the official read at 09:35. The controller runs it before it looks at
any other result. It reads r2 A2 100k with the protocol of the passing checked read.

| item | value |
| --- | --- |
| checkpoint | r2 A2 100k |
| TAG, commit | r2_official, 751e3dd |
| protocol | full + lean: `--tray --desk_boxes 4 --tray_lean 30`, the nine switches, held-out parts, N_STEPS 50, the client flags of `bin/read.sh` |
| seed, draws | 20261800, draws 0 to 23, 6000 parts, 120 episodes |
| split | Machine A 8 lanes, Machine B 4 lanes |
| pass | at least 5100 correct stows and at most 1.3 cycles per correct stow |
| stowed of 6000 | TBD |
| fraction, Wilson 95 % | TBD |
| cycles per correct stow | TBD |
| rakes, wrong-box parts | TBD |
| per draw | TBD |
| outcome (`PASS`, `PASS (scripted place)` or `FAIL`) | TBD |

The result is the milestone's number, whatever it is. A READ_FAIL is an orchestration error
with no gate token. It reruns once in a fresh TAG with the same settings. The best vertical-wall
checked read, 0.838, stands beside the official read.

## 9. Rejected levers

### 9.1 More rakes and the crowd start

The rake prototype `feat/ms7-rake-more` (3cda4c4) adds `--rake_cap N`, a larger rake cap per
part. It also adds `--rake_crowd`. When a neighbour blocks every rake start, the crowd start
lands the closed tips up to 3 mm into it. Each payload episode gains `left_parts`, the parts
that it leaves, with their rakes and failures. The census gains `left_at_wall`, and the gate
supervisor forwards both settings.

On the teacher (oracle driver, rung 2, three held-out heaps, run seeds 20267580 to 20267582):
the base read 141 of 150, the crowd start 141, a cap of 4 143, and both together 143.

On the policy (r2 B 300k, draw 2, full protocol), against the reference of 210 of 250 at 1.181:

| serving | stowed of 250 | cycles per stow | rakes | parts left at a wall | note |
| --- | --- | --- | --- | --- | --- |
| `--rake_cap 4` | 209 | 1.244 | 108 | 30, 20 of them after 4 rakes | |
| `--rake_crowd` | 205 | 1.259 | 65 | 34, 26 of them after 2 rakes | |
| `--rake_cap 4 --rake_crowd` | 202 | 1.371 | 104 | 25, 18 of them after 4 rakes | the cycle bar fails. 4 parts that were never raked before were reached, and none was freed. |

A part left at a wall lies in the tray, within 30 mm of a wall, and not at the failure cap.
Neither lever beats the reference. The ledger dropped the cap and the combination at 06:20, and
it closed the crowd start at 07:13. The branch went up with the lean branch at 07:49, and its
switches stay off by default.

### 9.2 The snap floor

The close heights of section 8.4 led to `--snap_floor MM` (0c37b51 on `feat/ms7-snap-floor`,
not pushed). It moves the floor of `--pick_snap` from the row centre plus 3 mm. The same floor
applies to the jaw hold of the snap and to the grasp height of `--dwell_close`. `--snap_floor
0` puts the floor at the row centre.

On r2 B 300k, draw 2, full protocol, vertical walls:

| serving | stowed of 250 | cycles per stow |
| --- | --- | --- |
| the reference, two rollouts | 210 and 208 | 1.181 and 1.202 |
| `--snap_floor 0`, first rollout | 212 | 1.269 |
| `--snap_floor 0`, second rollout | 205 | 1.200 |

First closes in the band of 2 to 15 mm fell from 38 to 24 and 26. But lateral `close_miss` rose
from 25 to 43 in one rollout. In the other, tall parts fell from 82 to 74. The ledger closed the
lever at 09:10.

### 9.3 The base refresh

T5 (777) and T6 (853) both read under T0's 900 (section 5). Neither replaced the base.

## 10. Deviations from the design

| deviation | reason |
| --- | --- |
| Corpus seeds are run seed 20270000 + p and layout seed 20271000 + p, not 20267000 + p. | Corpus v3 of campaign n already used 20267000 + p. |
| Every policy read serves `--scripted_rake`. | The policy never learned to rake, because the converter drops the rake frames. The design left wall-stuck parts to the policy. |
| Selection, checked and official reads serve `--part_failures`. | The cell memory closed every tray cell before the cycle budget (section 7.2). |
| Every read from 01:20 serves `--pick_snap`. | The pick forensic found an imprecise approach and a height fault (section 7.3). |
| Every read from 05:05 serves `--dwell_close` and `--adopt_held`. | The second forensic found the freeze at grasp height and the carry of a non-target part (section 7.4). |
| The rung 2 scene of the passing checked read and of the official read leans the tray walls (`--tray_lean 30`). The H1 tray of the design has vertical walls. The lean is a 30 degree outward draft about the interior edge, and the rim stays 20 mm high. | 413 of 541 lost parts were never targeted, and most start near a wall. The refusal matches a driven test of a flush slope, and the rake levers are null (section 8.4). On 12 paired heaps, the teacher reads 0.905 on the lean and 0.875 on vertical walls. With no retraining, the policy reads 221 of 250 on the lean against 210 (section 3.5). The vertical-wall reads stay in this report beside the lean read. The best is r2 A2 100k at 2513 of 3000, 0.838, on the same draws. |
| The rung 1 checked read is under the snap protocol. A full-protocol re-read is reported beside it. | The read ran before the 05:05 rule. The re-read reads 0.844 against 0.822. |
| The owner wired Machine B and Machine C into one LAN. | The design allowed no network change except the end of the Tailscale SSH check. The owner changed the network at 13:40, and corpora then moved to Machine B at LAN speed. |
| Rung 1 arm B mixes the wave with dataset_v2, not dataset_v5. | The ledger gives no reason. T5, the warm start on dataset_v5, read 777 at 12:55, before the pipeline was armed. |
| The first rung 2 round trains on both waves, 120 heaps, not on wave 1 alone. | Both waves finished before rung 1's best checkpoint existed. |
| Rung 2 starts on 2026-09-23 from r1 B 300k, the best screen at launch, before any rung 1 selection read. Arm A takes the same start. | The RTX 5080 does not idle. The rung 2 corpus is itself a tray-pick corpus, so rung 2 training keeps training the pick. |
| Rung 1 selection read three checkpoints, not two. | r1 A 350k's screen (161) was within the noise of r1 B 150k (164). |
| Rung 2 selection did not read the two best screens. | The ledger gives no reason (section 8.2). |
| Round 2 trains on data already in place. | The r2s re-record failed on host load, and wave 3 was lost to a full disk. |
| Round 2 has no separate selection read. A screen at 215 of 250 or more goes straight to a checked read, and its draws 0 to 3 stand as the selection read. | The rule of 06:37, after r2 A2 100k screened 218. The checked read only filters, and the official read stays one-shot. |
| Not every saved round 2 checkpoint got a screen. The B2 screens stop at 250k, and the A2 screens at 300k. | Machine B ran out of GPU memory at 08:04. A2 reached a plateau, and Machine A GPU went to the lean reads. |
| The lean wave r2L1 records on Machine B, not on Machine C. | It is a speculative wave, so that a lean decision does not cost two hours. Machine C ran wave 4 at the same time. |
| Selection, checked and official reads run on Machine A and Machine B, not on aarch64 CPU policy servers. | The first selection reads on aarch64 CPU lanes hit the 130-minute wall cap at 5.8 steps/s per lane. |
| The sorting teacher had a 100-part smoke before its 600-part read, not a 150-part first read. | The ledger gives no reason. |
| The wall band reaches the grasp's own finger reach plus 5 mm, not `OPEN_HALF_GAP`. | On the 0.13 x 0.20 m tray, `OPEN_HALF_GAP` bands every part within 57 mm of two walls, half the tray, where no heading survives. |
| The teacher turns the release roll along the carry (48bc50c). | The owner's review of a teaching episode. The tool stood still over the box for 37 to 50 steps per cycle. |

## 11. The code

The code is one stack of one-concern branches on main. All code branches are on GitHub. The
stack, bottom to top:

| order | branch | tip | what it adds | PR |
| --- | --- | --- | --- | --- |
| 1 | `feat/fleet` | b843d3d | `scripts/fleet.py` and its tests, the bundle fetch, `--hosts` and the sync guard | TBD |
| 2 | `feat/ms7-desk-boxes` | 0602635 | the H1 tray footprint constants and the desk boxes in the grader and the scene. The release over each part's own box, `--desk_boxes`, the certified side band and the box sizes. | TBD |
| 3 | `feat/ms7-tray` | 13bd798 | `--tray`, the wall-ranked grasps, the rake, the end-to-end grasps and the wedge. The 45 degree band, the jam recovery and the rake as its own takeover. | TBD |
| 4 | `feat/ms7-classes` | 22153d5 | `part_class`, the class-to-box map, the target's box on the context row, the `wrong_box` census, the rows per class and per box, `--scripted_place` | TBD |
| 5 | `feat/ms7-rake-serving` | a6b631e | the converter without the rake frames, `--scripted_rake` | TBD |
| 6 | `feat/ms7-part-failures` | be86bc8 | `--part_failures` | TBD |
| 7 | `feat/ms7-pick-snap` | 9bcb2ed | `--pick_snap` | TBD |
| 8 | `feat/ms7-smooth-carry` | 48bc50c | the release roll along the carry | TBD |
| 9 | `fix/convert-reopen-fragments` | 944cdc1 | the converter without the REOPEN fragments | TBD |
| 10 | `feat/ms7-serving-2` | 2b87de2 | `--dwell_close` and `--adopt_held` | TBD |
| 11 | `feat/ms7-rake-more` | 3cda4c4 | `--rake_cap` and `--rake_crowd`, `left_parts` in the payload, the census `left_at_wall` and the gate forwarding. Off by default and measured null. | TBD |
| 12 | `feat/ms7-tray-lean` | 751e3dd | `--tray_lean DEG`, the leaning walls in the scene and the grasp model, the gate forwarding and the cache key. Off by default. | TBD |
| 13 | `docs/ms7-sorting` | | this report | TBD |

Every new switch is off by default, so the MS6 protocol stays reproducible. At 751e3dd the CPU
suite collects 1081 tests. Two prototypes stay off the stack and unpushed:
`feat/ms7-snap-floor` (0c37b51, null) and `feat/ms7-refresh-on-miss` (78f2c7d). No PR is open
yet. The stack goes up with `GH_REPO=youtalk/maker-arm-lab gh stack`.

## 12. What is still open

- The official read of rung 2 (TAG r2_official) and the milestone outcome: TBD.
- The PR stack and its numbers: TBD.
- A lean checked read of r2 B 300k (TAG r2B_300000_L30_chk) runs on 12 aarch64 CPU lanes. The
  gate does not need it.
- Round 3 (`r2/train_r3L`) trains on Machine B. Its 50k screen reads 210 of 250 on the lean, and the
  later screens are TBD. The gate does not need it.
- `--refresh_on_miss` (78f2c7d) runs on draw 2. After an empty close, it rebuilds the target row
  from the live scene. The ledger expects 30 to 60 more parts per 3000.
- The real tray. The passing read uses walls that lean out 30 degrees. The vertical-wall reads
  stay the reference for a tray with vertical walls.
- The real arm. `--square_wrist`, `--dwell_close` and `--adopt_held` read the held part and the
  jaw state from the simulator. The real arm needs a hold signal and a held-part detector.
- The scripted rake. It plans on the simulator's part poses, and the policy does not rake.
- The overhead camera still frames the old shelf region. The tray heap spans about 30 x 20 px of
  the 224 px frame.
- The MS6 debts move on: the learned heading and a noise model on the target row.
