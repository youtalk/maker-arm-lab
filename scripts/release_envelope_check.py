"""Release-envelope probe: THE REACH AUTHORITY for `core.layout` on the Maker Arm.

Spec section 5 item 5 permits moving the layout numbers for this arm's reach
("the Maker Arm's reach ... may require moving the bins and the scatter band.
The oracle state machine is not modified; only layout numbers are"). This
script is what decides whether they need moving: it drives a real
pick-and-place into every one of the 4 bins at the nominal layout and at the
four extremes of the run-level bin jitter, and gates on the two things reach
actually costs -- whether EXTEND CONVERGES on the release pose, and whether the
brick then LANDS IN ITS BIN.

IT SWEEPS TRACKING, NOT IK, AND MUST KEEP DOING SO. Task D3 measured that this
arm's IK-feasible envelope is strictly larger than the one it can actually
track: the solver reports `worst_pos_resid=0.00um` on a continuous path at
poses where the arm then stops 12-50 mm short. A probe that asked "can the
solver find a pose for this bin" would answer yes everywhere and re-derive a
bound that was never binding -- the error that cost Task D3 several rounds (see
`core.layout.SCATTER_MAX_RADIUS`'s comment). Every gate below is therefore read
off a driven servo's residual, never off a solver's report.

Two independent bounds coexist and a layout move has to respect BOTH:

* `layout.SCATTER_MAX_RADIUS` -- the MEASURED trackable radius, and an
  unconditional invariant on where bricks may spawn; and
* `layout.top_down_x_limit(bl)` = `bin_front_x(bl) - X_LIMIT_MARGIN` -- a
  BIN-FACE CLEARANCE bound (where the forearm fouls the bins), which moves with
  the bins. Pulling the bins closer lowers it, so it can drop below the radius
  bound and start binding where it did not before.

The jitter corners: `layout.sample_bin_layout` draws `bin_x` in
[_BIN_X-0.01, _BIN_X+0.02] and each slot's y in +-0.02 at reset, so this script
builds the two extremes of that jitter explicitly -- nominal plus 4 corners,
`bin_x in {0.43, 0.46}` crossed with the two OUTER bins' y pushed to their
+-0.02 m bound together (inward together / outward together). If any corner
cannot converge EXTEND or land the brick in its bin, either the layout numbers
or `core.oracle`'s release constants (`RELEASE_Z`/`RELEASE_TILT_RAD`/
`RELEASE_X_SETBACK`) have to move; the oracle STATE MACHINE is never modified.

Which brick lands in which bin is controlled by teleporting (isolate_brick,
ported from scripts/scripted_grasp_check.py), not by relying on
``EnvOracle._next_brick``'s scatter-order scan -- this probe is about the
RELEASE leg's reachable envelope, not the pick leg's brick-selection order
(exercised by scripted_grasp_check.py, and end to end by scripts/oracle_check.py
once Task D5 lands it). ``num_bricks=NUM_BRICKS=4`` (one per color) makes brick
index == color index == target bin, so ``isolate_brick(keep=color)`` pins the
trial to exactly the bin under test. The pick itself still runs for real
through OracleRig's SETTLE..LIFT phases (not faked), so the release leg the
probe measures is fed by the same pinch/attitude state a real episode would
produce, not a hand-placed one.

Run, one layout per process:

    for i in 0 1 2 3 4; do
      $ISAACLAB_PATH/isaaclab.sh -p scripts/release_envelope_check.py --headless \
        --layout_index $i
    done

RUN IT THAT WAY, NOT AS A SINGLE PROCESS. `run()` still supports the all-in-one-process loop
and it is the honest default -- only that path prints the cross-layout ENVELOPE_OK -- but on
Isaac Lab v2.3.2 here it does not survive the SECOND `gym.make`. Observed once, Task D4
2026-09-06: layout 0 finished all 8 trials and stowed all 8, then the process sat at 100 % of
one CPU core with the GPU at 0 %, no disk I/O and flat RSS for 13 minutes inside the layout-1
environment construction, and had to be killed. `--layout_index` sidesteps it by giving each
layout a fresh process; the cost is that each process prints ENVELOPE_PARTIAL and the
cross-layout gate has to be read off the five ENVELOPE_META lines by hand.

Prints ENVELOPE_OK or ENVELOPE_FAIL from every exit path inside `run()`, including the
`--layouts` guard, which is validated there rather than at module scope for that reason.
It does NOT cover the two paths that exit before the guarded region -- argparse usage
errors and AppLauncher's own argument validation. scripts/scripted_grasp_check.py grew a
`_TokenEmittingParser` for exactly those; this script deliberately has not, because nothing
parses its output yet. Give it one before anything does.
"""

import argparse
import json
import math
import os
import random
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Datagen-v0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--layouts",
    type=int,
    default=5,
    help="Must be 5 (nominal + 4 explicit envelope corners) -- not a sampling knob. "
    "The corners are constructed, not drawn: a draw would leave the extremes to luck.",
)
parser.add_argument("--per_bin", type=int, default=2)
parser.add_argument(
    "--layout_index",
    type=int,
    default=None,
    help="Run only this one 0-based layout index (its own OS process) and skip the "
    "cross-layout gate, instead of looping all 5 layouts in one process. THIS IS CURRENTLY "
    "THE WORKING PATH, not just a time-budget fallback -- see the module docstring.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# --layouts is validated inside run(), NOT here: a raise at module scope lands
# outside the try/except that turns a failure into a token, so a supervisor
# parsing this script's output would read it as a hang rather than a failure.
# Task D3 paid for that lesson twice on scripts/scripted_grasp_check.py.

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab (and friends) are only importable once AppLauncher has launched Kit; the
# deferral is deliberate, as in smoke_usd.py.
import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import DatasetExportMode

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.core import layout, oracle
from maker_arm_lab.robots.frames import TOOL_AXIS_BODY
from maker_arm_lab.tasks.manager_based.lego_sort import mdp
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import LegoSortDatagenEnvCfg
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
    GRIPPER_OPEN_ACTION,
    ROT_CONVERGED,
    OracleRig,
)

# One brick per color: brick index == color index == target bin, which is what
# lets isolate_brick(keep=color) pin a trial to a specific bin deterministically
# (see module docstring). LegoSortEnvCfg requires num_bricks in [4, 8].
NUM_BRICKS = 4

# core/oracle.py's EXTEND goal: stop_pos=0.008 (see next_goal's EXTEND branch).
# Restated here, not imported, because core.oracle does not name it -- it is an
# inline literal on the Goal construction, so there is nothing to import.
EXTEND_POS_CONVERGED_M = 0.008

# Raw open-gripper steps between RELEASE completing and the stow being read, so the brick
# has landed and come to rest before `per_brick_stowed` (which demands at-rest) is asked.
SETTLE_AFTER_RELEASE_STEPS = 30

# World pose (identity yaw) every trial's kept brick is teleported to before
# the pick. x=0.20 clears BOTH bounds by a wide margin at every corner, which is
# what makes the pick leg constant across corners so only the release leg
# varies: it is inside the measured trackable radius (0.20 < SCATTER_MAX_RADIUS
# = 0.280, at azimuth 0 -- the wedge Task D3 mapped, so the most demanding
# azimuth) and inside every corner layout's bin-face clearance (worst case
# bin_x=0.43 -> top_down_x_limit 0.28 m). y=0.0 keeps the pick yaw at 0
# (identity), same as scripted_grasp_check.py's x-sweep mode.
PICK_POSITION = (0.20, 0.0, layout.BRICK_SIZE[2] / 2)

# Parking spot for the 3 bricks not under test this trial: off to the side,
# clear of the arm and of every corner layout's bins (bin_x <= 0.46, bin outer
# wall <= 0.46+0.045+0.005=0.51). Ported verbatim from
# scripts/scripted_grasp_check.py's isolate_brick.
PARK_X = 0.66
PARK_Y0 = -0.40
PARK_Y_SPACING = 0.04

# --- Envelope corners, explicit (not sampled) -----------------------------
#
# layout.sample_bin_layout's own jitter ranges: bin_x in [_BIN_X-0.01,
# _BIN_X+0.02] = [0.43, 0.46], per-slot y in [-0.02, +0.02]. Taking the
# outermost slots to their +-0.02 bound pushes BOTH outer bins (the y=-0.21 and
# y=+0.21 slots -- the ones farthest from the center, so the ones that cost the
# most reach and the most base swing) to that bound together, either both
# outward (wider spread) or both inward (narrower spread), crossed with bin_x's
# two extremes: 2 x 2 = 4 corners. The outward/0.46 pair is the worst case, and
# it is the one that decides RELEASE_TILT_RAD: it puts a release target at
# radius 0.497 m.
_OUTER_JITTER = 0.02
_ENVELOPE_BIN_X = (0.43, 0.46)


def _corner(bin_x: float, outer_dy: float) -> layout.BinLayout:
    y0, y1, y2, y3 = layout.DEFAULT_BIN_LAYOUT.bin_ys
    return layout.BinLayout(
        bin_x=bin_x, bin_ys=(y0 - outer_dy, y1, y2, y3 + outer_dy), slot_colors=(0, 1, 2, 3)
    )


ENVELOPE_LAYOUTS = [layout.DEFAULT_BIN_LAYOUT] + [
    _corner(bin_x, outer_dy)
    for bin_x in _ENVELOPE_BIN_X
    for outer_dy in (_OUTER_JITTER, -_OUTER_JITTER)
]


def isolate_brick(env, keep: int, num_bricks: int, position) -> None:
    """Park every brick except ``keep`` off to the side; place ``keep`` at ``position``.

    Ported from scripts/scripted_grasp_check.py's isolate_brick (same parking
    spot, same identity-yaw placement, same single-env world-frame write --
    env_origins[0] is (0,0,0) for num_envs=1 so no origin subtraction is
    needed here, unlike OracleRig's own env-local frame convention).
    """
    env_ids = torch.tensor([0], device=env.device)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    zero_vel = torch.zeros(1, 6, device=env.device)
    for index in range(num_bricks):
        brick = env.scene[f"brick_{index}"]
        if index == keep:
            place = list(position)
        else:
            place = [PARK_X, PARK_Y0 + PARK_Y_SPACING * index, layout.BRICK_SIZE[2] / 2]
        pose = torch.cat([torch.tensor([place], device=env.device), identity], dim=-1)
        brick.write_root_pose_to_sim(pose, env_ids=env_ids)
        brick.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


def _tool_tilt_deg(rig, i: int) -> float:
    """Tool-axis tilt off world -z, deg -- 0 is a level (straight-down) tool.

    Uses ``TOOL_AXIS_BODY`` rather than body +z: on this arm the tool points
    along grasp_center's -z (frames.TOOL_AXIS_BODY), so measuring +z would
    report 180 deg minus the real tilt. Same expression as
    ``scripted_grasp_check.GraspRig.tool_angle_deg``.
    """
    tool_axis = torch.tensor(TOOL_AXIS_BODY, device=rig.device)
    axis = math_utils.quat_apply(rig.tool_quat(i), tool_axis)
    return math.degrees(math.acos(max(-1.0, min(1.0, (-axis[2]).item()))))


def _tightest_joint_margin(rig, i: int) -> tuple[float, str]:
    """Smallest distance from any arm joint to its nearer stop, and that joint's name.

    How much travel the arm has left when it holds the release pose. NOT gated on --
    ``record["ok"]`` is EXTEND convergence plus the stow -- but it is what says whether a
    corner that passes passes COMFORTABLY, and it is the first thing to read if one starts
    failing.

    ``rig.limits`` is the articulation's own ``joint_pos_limits`` row for the six arm
    joints, in ``arm_ids`` order, so there is no hard-coded stop value here and the
    diagnostic follows the asset. That matters on this arm specifically: its stops are
    asymmetric and joint-specific (``link_005_joint`` is [-1.6, 1.73], ``link_006_joint``
    [-1.57, 1.57]), so there is no single "lower stop" to measure a margin against, and
    naming the TIGHTEST joint rather than watching a fixed one means the diagnostic survives
    a change in which joint binds.
    """
    position = rig.robot.data.joint_pos[i, rig.arm_ids]
    margin = torch.minimum(position - rig.limits[:, 0], rig.limits[:, 1] - position)
    tightest = int(margin.argmin().item())
    return margin[tightest].item(), rig.robot.joint_names[rig.arm_ids[tightest]]


def phase_telemetry(rig, i: int, prev_goal, quat_ref_before, steps_used: int) -> dict:
    """Read-only reconstruction of what OracleRig._phase_result computed
    internally for ``prev_goal``'s ending step, from OUTSIDE the class.

    Safe and exact, not approximate: ``OracleRig.step()`` only ever calls
    ``AttitudeHold.advance()`` with a literal zero steering command (never a
    P-servo rotation command), and a zero ``rot_command`` makes
    ``AttitudeHold.advance``'s ``advanced == reference`` every step (see
    ``core.attitude_hold.AttitudeHold.advance``'s anti-windup branch: the
    quaternion multiply by ``quat_from_axis_angle(0)`` is the identity). So the
    hold's reference never moves during a phase, and ``quat_ref_before`` --
    ``rig.holds[i].reference`` captured the instant ``prev_goal`` was entered
    -- is bit-identical to what ``_phase_result`` read at the transition step.
    ``rig.pick_point``/``rig.tool_quat``/``rig._brick_pos``/``rig.pinch_evidence``
    are all read-only (checked against oracle_runtime.py's source). Nothing
    here mutates rig state -- unlike ``rig.desired_quat``, which has a
    documented ``commanded_yaw[i]`` side effect and must never be called from
    a probe like this one.
    """
    info: dict = {"phase": prev_goal.phase.name, "steps": steps_used}
    if prev_goal.target_pos is not None:
        pos_err = (prev_goal.target_pos.to(rig.device) - rig.pick_point(i)).norm().item()
        rot_err = (
            math_utils.axis_angle_from_quat(
                math_utils.quat_mul(quat_ref_before, math_utils.quat_conjugate(rig.tool_quat(i)))
            )
            .norm()
            .item()
        )
        info["pos_err_mm"] = pos_err * 1000.0
        info["rot_err_deg"] = math.degrees(rot_err)
    if prev_goal.phase == oracle.Phase.LIFT:
        brick = rig.oracles[i].target_brick
        brick_end = rig._brick_pos(i, brick)
        rise = (brick_end[2] - rig.brick_start[i][2]).item()
        xy_error = (brick_end[:2] - rig.pick_point(i)[:2]).norm().item()
        pinch = rig.pinch_evidence(i, brick)
        info.update(
            rise_mm=rise * 1000.0,
            xy_error_mm=xy_error * 1000.0,
            grip_left_n=pinch["left"],
            grip_right_n=pinch["right"],
            jaw_stall_mm=pinch["stall"] * 1000.0,
            asymmetry=pinch["asymmetry"],
            pinch_ok=pinch["ok"],
        )
    if prev_goal.phase in (oracle.Phase.EXTEND, oracle.Phase.RELEASE):
        margin_rad, joint_name = _tightest_joint_margin(rig, i)
        info["joint_margin_rad"] = margin_rad
        info["joint_margin_name"] = joint_name
        # The wrist roll is called out separately from the margin above because the
        # jaw-heading fold (`core.oracle.fold_jaw_heading`) spends exactly this joint to hold
        # a heading across EXTEND's base swing, so its value here says how much of that fold
        # budget the release pose consumed.
        info["wrist_roll_rad"] = rig.wrist_roll(i)
        info["tool_tilt_deg"] = _tool_tilt_deg(rig, i)
    return info


def _format_phase_line(label: str, info: dict) -> str:
    parts = [f"PHASE {info['phase']} [{label}]: steps={info['steps']}"]
    if "pos_err_mm" in info:
        parts.append(f"pos_err={info['pos_err_mm']:.2f}mm rot_err={info['rot_err_deg']:.2f}deg")
    if info["phase"] == "LIFT":
        parts.append(
            f"rise={info['rise_mm']:+.2f}mm xy_err={info['xy_error_mm']:.2f}mm "
            f"grip={info['grip_left_n']:.2f}/{info['grip_right_n']:.2f}N "
            f"jaw_stall={info['jaw_stall_mm']:.3f}mm asymmetry={info['asymmetry']:.3f} "
            f"pinch_ok={info['pinch_ok']}"
        )
    if info["phase"] in ("EXTEND", "RELEASE"):
        parts.append(
            f"joint_margin={info['joint_margin_rad']:.4f}rad ({info['joint_margin_name']}) "
            f"wrist_roll={info['wrist_roll_rad']:.4f}rad "
            f"tool_tilt={info['tool_tilt_deg']:.2f}deg"
        )
    return " ".join(parts)


def run_to_release(rig, label: str, i: int = 0, max_steps: int = 3000) -> dict:
    """Step env ``i`` through a real pick (SETTLE..LIFT) and the release leg
    (RAISE..RELEASE), stopping the instant RELEASE's hold completes -- so the
    30-settle-step stow check that follows (outside this function) watches the
    brick fall in a clean, oracle-free window, instead of grading it while the
    oracle's own RETREAT leg is still swinging the arm over the bin. Prints one
    ``PHASE`` line per transition.

    Returns a dict keyed by phase name (``"LIFT"``, ``"EXTEND"``,
    ``"RELEASE"``, ...) -> ``phase_telemetry`` dicts, plus ``"abandoned"``/
    ``"timed_out"`` flags.
    """
    telemetry: dict = {}
    prev_goal = rig.goals[i]
    quat_ref_before = rig.holds[i].reference.clone() if prev_goal is not None else None
    steps_in_phase = 0
    for _ in range(max_steps):
        result = rig.step()
        steps_in_phase += 1
        new_goal = rig.goals[i]
        if new_goal is not prev_goal:
            ended_phase = prev_goal.phase if prev_goal is not None else None
            if prev_goal is not None:
                info = phase_telemetry(rig, i, prev_goal, quat_ref_before, steps_in_phase)
                telemetry[ended_phase.name] = info
                print(_format_phase_line(label, info), flush=True)
            steps_in_phase = 0
            prev_goal = new_goal
            quat_ref_before = rig.holds[i].reference.clone() if new_goal is not None else None
            if ended_phase == oracle.Phase.RELEASE:
                telemetry["abandoned"] = False
                telemetry["timed_out"] = False
                return telemetry
        if bool(result["abandoned"][i]):
            telemetry["abandoned"] = True
            telemetry["timed_out"] = False
            print(f"  [{label}] abandoned: reason={rig.episode_stats(i)['abandon']}", flush=True)
            return telemetry
        if bool(result["dones"][i]):
            # Shouldn't happen here: the other 3 bricks are parked unreachable
            # (all_bricks_stowed can't fire) and this trial is far short of
            # the 180 s time_out -- treat it as an anomaly, not a pass.
            telemetry["abandoned"] = False
            telemetry["timed_out"] = True
            return telemetry
    telemetry["abandoned"] = False
    telemetry["timed_out"] = True
    return telemetry


def settle_and_check_stowed(env, brick: int, bl: layout.BinLayout) -> tuple[bool, float]:
    """30 raw open-gripper steps post-RELEASE, then read ``per_brick_stowed``
    and the landing offset from that bin's interior center (3-D norm, mm)."""
    action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    action[0, -1] = GRIPPER_OPEN_ACTION
    for _ in range(SETTLE_AFTER_RELEASE_STEPS):
        env.step(action)
    stowed = bool(mdp.per_brick_stowed(env)[0, brick].item())
    pos = (env.scene[f"brick_{brick}"].data.root_pos_w[0] - env.scene.env_origins[0]).cpu()
    center = layout.bin_interior_centers(bl)[brick]
    offset_mm = (pos - center).norm().item() * 1000.0
    return stowed, offset_mm


def run_layout(bl: layout.BinLayout, layout_idx: int) -> list[dict]:
    cfg = LegoSortDatagenEnvCfg(
        cameras=False,
        desk_color=False,
        physics_dr=False,
        start_pose_offset_rad=0.0,
        num_bricks=NUM_BRICKS,
        bin_x=bl.bin_x,
        bin_ys=bl.bin_ys,
        slot_colors=bl.slot_colors,
    )
    cfg.sim.device = args.device
    # This probe never needs a dataset -- avoid piling up HDF5 exports (and
    # the write cost) on every one of the ~40 env.reset() calls below.
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset(seed=args.seed)
    noise = oracle.OracleNoise()
    rng = random.Random(args.seed)
    rig = OracleRig(env, bl, noise, rng)

    trials = []
    try:
        for color in range(4):
            for trial in range(args.per_bin):
                seed = args.seed + layout_idx * 1000 + color * 100 + trial
                label = f"layout={layout_idx} color={layout.BRICK_COLOR_ORDER[color]} trial={trial}"
                env.reset(seed=seed)
                isolate_brick(env, keep=color, num_bricks=NUM_BRICKS, position=PICK_POSITION)
                rig.rearm([0])
                release_telemetry = run_to_release(rig, label)

                extend = release_telemetry.get("EXTEND")
                release = release_telemetry.get("RELEASE")
                extend_ok = (
                    extend is not None
                    and extend["pos_err_mm"] / 1000.0 < EXTEND_POS_CONVERGED_M
                    and extend["rot_err_deg"] < math.degrees(ROT_CONVERGED)
                )
                stowed, landing_offset_mm = (False, None)
                if not release_telemetry["abandoned"] and not release_telemetry["timed_out"]:
                    stowed, landing_offset_mm = settle_and_check_stowed(env, color, bl)

                record = {
                    "layout": layout_idx,
                    "bin_x": bl.bin_x,
                    "bin_y": bl.bin_ys[color],
                    "color": layout.BRICK_COLOR_ORDER[color],
                    "trial": trial,
                    "abandoned": release_telemetry["abandoned"],
                    "timed_out": release_telemetry["timed_out"],
                    "extend_pos_err_mm": extend["pos_err_mm"] if extend else None,
                    "extend_rot_err_deg": extend["rot_err_deg"] if extend else None,
                    "extend_ok": extend_ok,
                    "release_tilt_deg": release["tool_tilt_deg"] if release else None,
                    "joint_margin_rad": release["joint_margin_rad"] if release else None,
                    "joint_margin_name": release["joint_margin_name"] if release else None,
                    "wrist_roll_rad": release["wrist_roll_rad"] if release else None,
                    "stowed": stowed,
                    "landing_offset_mm": landing_offset_mm,
                }
                record["ok"] = bool(extend_ok and stowed)
                trials.append(record)

                def _fmt(value, spec):
                    return format(value, spec) if value is not None else "nan"

                print(
                    f"TRIAL layout={layout_idx} bin_x={bl.bin_x:.3f} bin_y={bl.bin_ys[color]:+.3f} "
                    f"color={record['color']} trial={trial}: "
                    f"extend_pos_err={_fmt(record['extend_pos_err_mm'], '.2f')}mm "
                    f"extend_rot_err={_fmt(record['extend_rot_err_deg'], '.2f')}deg "
                    f"extend_ok={extend_ok} "
                    f"release_tilt={_fmt(record['release_tilt_deg'], '.2f')}deg "
                    f"joint_margin={_fmt(record['joint_margin_rad'], '.4f')}rad "
                    f"({record['joint_margin_name'] or 'nan'}) "
                    f"wrist_roll={_fmt(record['wrist_roll_rad'], '.4f')}rad "
                    f"landing_offset={_fmt(landing_offset_mm, '.2f')}mm "
                    f"stowed={stowed} abandoned={record['abandoned']} timed_out={record['timed_out']} "
                    f"-> {'OK' if record['ok'] else 'FAIL'}",
                    flush=True,
                )
    finally:
        env.close()
    return trials


def run() -> bool:
    if args.layouts != len(ENVELOPE_LAYOUTS):
        raise ValueError(
            f"release-envelope probe always uses exactly {len(ENVELOPE_LAYOUTS)} explicit "
            "layouts (nominal + 4 corners); --layouts is not a sampling knob"
        )
    if args.layout_index is not None:
        trials = run_layout(ENVELOPE_LAYOUTS[args.layout_index], args.layout_index)
    else:
        trials = []
        for idx, bl in enumerate(ENVELOPE_LAYOUTS):
            trials.extend(run_layout(bl, idx))

    meta = {
        "n_trials": len(trials),
        "n_ok": sum(1 for t in trials if t["ok"]),
        "n_stowed": sum(1 for t in trials if t["stowed"]),
        "n_extend_ok": sum(1 for t in trials if t["extend_ok"]),
        "trials": trials,
    }
    print("ENVELOPE_META " + json.dumps(meta), flush=True)

    ok = bool(trials) and all(t["ok"] for t in trials)
    if args.layout_index is not None:
        print(
            "ENVELOPE_PARTIAL layout_index mode -- cross-layout gate not evaluated in this "
            "process (this process's own trials ARE gated above); re-run once every layout "
            "index has been captured to confirm all layouts.",
            flush=True,
        )
    return ok


try:
    ok = run()
except Exception:  # noqa: BLE001 -- any failure must become ENVELOPE_FAIL, not a crash.
    traceback.print_exc()
    ok = False

print("ENVELOPE_OK" if ok else "ENVELOPE_FAIL", flush=True)
os._exit(0 if ok else 1)
