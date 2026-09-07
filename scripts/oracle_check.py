"""Full-episode scripted-oracle check: the MS1 milestone run (Task D5).

The first end-to-end exercise of ``OracleRig`` on the Maker Arm driving a complete scripted
sort -- every brick, every phase, SETTLE through the last RETREAT -- recorded to HDF5
through the real ``MakerArm-LegoSort-Datagen-v0`` recorder stack. Everything before this
exercised pieces of the pipeline in isolation: the pick leg in
``scripts/scripted_grasp_check.py`` (Task D3), the release leg in
``scripts/release_envelope_check.py`` (Task D4), the datagen env's recorders and
domain randomisation in ``scripts/smoke_lego_sort.py``. This is the first script that runs a
whole scripted episode and checks it against the recorder's own idea of "succeeded", not a
hand-rolled proxy.

It is also the first thing that executes RETREAT: ``release_envelope_check.py`` stops the
instant RELEASE completes, by design, so the retreat leg has never been driven on this arm.

Run:
  $ISAACLAB_PATH/isaaclab.sh -p scripts/oracle_check.py --headless --trials 2
  $ISAACLAB_PATH/isaaclab.sh -p scripts/oracle_check.py --headless --enable_cameras \\
      --with_cameras --trials 1

Token contract: every exit path prints ORACLE_OK or ORACLE_FAIL. Argparse usage errors go
through ``_TokenEmittingParser`` and ``AppLauncher``'s own argument validation is guarded,
because a script that exits silently reads to the D6/D7 supervisor as a HANG rather than as
a failure -- the defect class Task D3 paid for four times in the grasp check.
"""

import argparse
import math
import os
import random
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher


class _TokenEmittingParser(argparse.ArgumentParser):
    """An ArgumentParser whose USAGE ERRORS still honour the ORACLE_OK / ORACLE_FAIL contract.

    argparse's ``error()`` exits 2 with a usage message and no token, which a supervisor
    parsing for a token cannot distinguish from a hang. Same treatment, and same reasoning,
    as ``scripts/scripted_grasp_check.py``'s parser.

    ``--help`` is deliberately NOT covered: it exits 0 through ``exit()`` rather than
    ``error()``, it is not an execution path, and printing a verdict after a help text would
    be nonsense.
    """

    def error(self, message: str):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr, flush=True)
        print("ORACLE_FAIL", flush=True)
        raise SystemExit(2)


parser = _TokenEmittingParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-Datagen-v0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trials", type=int, default=2)
parser.add_argument("--num_bricks", type=int, default=6)
parser.add_argument("--noise", type=str, choices=("on", "off"), default="on")
parser.add_argument(
    "--with_cameras",
    action="store_true",
    help="cameras=True in the env cfg, plus an HDF5 shard inspection at the end "
    "(needs --enable_cameras too, to actually render).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# AppLauncher validates its own arguments (a bad --device, for one) and raises before the
# `try:` at the bottom of this file can convert anything to a token.
try:
    app_launcher = AppLauncher(args)
except Exception:  # noqa: BLE001 -- a launch failure must be a token, not a bare traceback.
    traceback.print_exc()
    print("ORACLE_FAIL", flush=True)
    raise SystemExit(1) from None
simulation_app = app_launcher.app

# isaaclab (and friends) are only importable once AppLauncher has launched Kit;
# the deferral is deliberate, as in smoke_usd.py.
import gymnasium as gym
import torch
from isaaclab.managers import DatasetExportMode

import maker_arm_lab.tasks  # noqa: F401 -- registers MakerArm-* envs
from maker_arm_lab.core import layout, oracle
from maker_arm_lab.tasks.manager_based.lego_sort import mdp
from maker_arm_lab.tasks.manager_based.lego_sort.datagen import LegoSortDatagenEnvCfg
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

# Safety cap on control steps per episode -- comfortably above the 180 s / 30 Hz = 5400-step
# time_out budget datagen.py sets, so a genuinely stuck env raises here instead of looping
# forever; a normally-terminating env (success or time_out) always returns well before this.
# Carried over unchanged from openyam-lab, where the 580-750 control steps per brick behind
# that budget were measured on OpenYAM's arm. THIS script is what re-measures the per-brick
# cost on the Maker Arm -- the phase lines below print every phase's step count.
MAX_EPISODE_STEPS = 6000

# Control steps between per-brick stow samples (see `_stow_snapshot`).
STOW_SAMPLE_STEPS = 50

# --noise off: every OracleNoise axis pinned to zero (or a degenerate single-value range),
# for a deterministic run with no per-episode draw.
NOISE_OFF = oracle.OracleNoise(
    hover_height=(0.12, 0.12),
    lateral_offset=0.0,
    yaw_offset=0.0,
    gain_scale=(1.0, 1.0),
    action_noise_std=0.0,
    via_jitter=0.0,
)


def _tightest_joint_margin(rig, i: int) -> tuple[float, str]:
    """Smallest distance from any arm joint to its nearer stop, and that joint's name.

    Ported from ``scripts/release_envelope_check.py``, and for the reason recorded there:
    this arm's stops are asymmetric and joint-specific, so there is no single "lower stop" to
    subtract, and naming the TIGHTEST joint rather than watching a fixed index means the
    diagnostic survives a change in which joint binds -- which already happened once, from
    ``link_005_joint`` to ``link_006_joint``, when D4 raised the release tilt.
    """
    position = rig.robot.data.joint_pos[i, rig.arm_ids]
    margin = torch.minimum(position - rig.limits[:, 0], rig.limits[:, 1] - position)
    tightest = int(margin.argmin().item())
    return margin[tightest].item(), rig.robot.joint_names[rig.arm_ids[tightest]]


def phase_telemetry(rig, i: int, prev_goal, attitude_ref, steps_used: int) -> dict:
    """Read-only reconstruction of what ``OracleRig._phase_result`` computed internally for
    ``prev_goal``'s ending step, from OUTSIDE the class.

    Safe and exact, not approximate: ``rig.pick_point`` / ``rig.attitude_error`` /
    ``rig.tool_tilt`` / ``rig._brick_pos`` / ``rig.pinch_evidence`` are all read-only, and
    nothing here mutates rig state.

    ``attitude_ref`` is ``rig.attitude_reference(i)`` captured when ``prev_goal`` was
    ENTERED, and it has to be: a caller watching for a goal transition only sees it after
    ``_enter_phase`` has planned the next leg, which sets the commanded attitude, so grading
    the ended phase against the live value would grade it against its successor's. Measured
    the wrong way round first -- LIFT reported a 60 deg attitude error that was really RAISE's
    freshly commanded release tilt.
    """
    info: dict = {"phase": prev_goal.phase.name, "steps": steps_used}
    if prev_goal.target_pos is not None:
        pos_err = (prev_goal.target_pos.to(rig.device) - rig.pick_point(i)).norm().item()
        info["pos_err_mm"] = pos_err * 1000.0
        info["rot_err_deg"] = math.degrees(rig.attitude_error(i, attitude_ref))
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
    if prev_goal.phase in (oracle.Phase.EXTEND, oracle.Phase.RELEASE, oracle.Phase.RETREAT):
        # RETREAT is in this group and was not in the ported script's: it holds
        # RELEASE_TILT_RAD out to core.oracle.RETREAT_POINT, and D4 predicted from its reach
        # sweep that that point was ALSO out of reach at the old 0.70 rad tilt (ceiling
        # ~0.242 m at radius 0.26 against a target z of 0.28) and should now clear widely.
        # D4 could not test it -- release_envelope_check.py stops at RELEASE -- so this is
        # the first run that drives the leg, and these are the numbers that settle it.
        margin_rad, joint_name = _tightest_joint_margin(rig, i)
        info["joint_margin_rad"] = margin_rad
        info["joint_margin_name"] = joint_name
        info["wrist_roll_rad"] = rig.wrist_roll(i)
        info["tool_tilt_deg"] = math.degrees(rig.tool_tilt(i))
    return info


def _format_phase_line(trial: int, info: dict) -> str:
    parts = [f"PHASE {info['phase']} [trial {trial}]: steps={info['steps']}"]
    if "pos_err_mm" in info:
        parts.append(f"pos_err={info['pos_err_mm']:.2f}mm rot_err={info['rot_err_deg']:.2f}deg")
    if info["phase"] == "LIFT":
        parts.append(
            f"rise={info['rise_mm']:+.2f}mm xy_err={info['xy_error_mm']:.2f}mm "
            f"grip={info['grip_left_n']:.2f}/{info['grip_right_n']:.2f}N "
            f"jaw_stall={info['jaw_stall_mm']:.3f}mm asymmetry={info['asymmetry']:.3f} "
            f"pinch_ok={info['pinch_ok']}"
        )
    if info["phase"] in ("EXTEND", "RELEASE", "RETREAT"):
        parts.append(
            f"joint_margin={info['joint_margin_rad']:.4f}rad ({info['joint_margin_name']}) "
            f"wrist_roll={info['wrist_roll_rad']:.4f}rad "
            f"tool_tilt={info['tool_tilt_deg']:.2f}deg"
        )
    return " ".join(parts)


def _stow_snapshot(env, i: int) -> list[dict]:
    """Per brick: is it stowed, and how far is it from its bin's interior center (mm)?

    The gate is ``mdp.per_brick_stowed``, i.e. the same predicate the ``success``
    termination and the recorder's export filter grade on, so a disagreement between "the
    oracle sorted six bricks" and "the episode did not export" resolves here rather than by
    inference.
    """
    stowed = mdp.per_brick_stowed(env)[i]
    centers = layout.bin_interior_centers(env.cfg.bin_layout)
    half = layout.bin_half_extents()
    out = []
    for brick in range(env.cfg.num_bricks):
        pos = (env.scene[f"brick_{brick}"].data.root_pos_w[i] - env.scene.env_origins[i]).cpu()
        offset = pos - centers[brick % len(centers)]
        out.append(
            {
                "brick": brick,
                "stowed": bool(stowed[brick].item()),
                "offset_mm": tuple(float(v) * 1000.0 for v in offset),
                "outside_mm": tuple(float(v) * 1000.0 for v in (offset.abs() - half)),
            }
        )
    return out


def _report_stow_changes(trial: int, step: int, rig, i: int, before, after) -> None:
    """Print every brick that entered or LEFT its bin since the last sample.

    A brick leaving is the one this exists for. "The oracle sorted six bricks" and "six
    bricks are in bins" are different claims -- ``core.oracle`` marks a brick sorted when its
    RETREAT completes, whether or not the brick is where it was aimed -- and the two ways
    they can disagree (dropped in transit, or knocked out of a bin by a later delivery) are
    different problems. Only a transition tells them apart.
    """
    if before is None:
        return
    phase = rig.goals[i].phase.name if rig.goals[i] is not None else rig.oracles[i].phase.name
    for was, now in zip(before, after, strict=True):
        if was["stowed"] == now["stowed"]:
            continue
        verb = "ENTERED_BIN" if now["stowed"] else "LEFT_BIN"
        print(
            f"STOW {verb} [trial {trial}]: brick={now['brick']} step={step} phase={phase} "
            f"offset_from_bin_center=({now['offset_mm'][0]:+.1f},{now['offset_mm'][1]:+.1f},"
            f"{now['offset_mm'][2]:+.1f})mm",
            flush=True,
        )


def _termination_reason(env, i: int) -> str:
    """Which termination term ended the episode, read after the auto-reset.

    The termination manager's per-term tensors are the terminating step's values and are not
    recomputed by the reset, which is exactly what the recorder's export filter grades on
    (``RecorderManager.record_pre_reset`` reads ``get_term("success")``). So this says why
    an episode did or did not export, from the same source that decided it.
    """
    active = env.termination_manager.active_terms
    fired = [name for name in active if bool(env.termination_manager.get_term(name)[i].item())]
    return ",".join(fired) if fired else "none"


def run_episode(
    rig, trial: int, i: int = 0, max_steps: int = MAX_EPISODE_STEPS
) -> tuple[bool, bool, list[dict] | None]:
    """Step env ``i`` to completion, printing one ``PHASE`` line per phase transition (see
    ``phase_telemetry``'s docstring for why reading rig internals here is safe). Returns
    ``(done, abandoned, last stow snapshot)``."""
    prev_goal = rig.goals[i]
    attitude_ref = rig.attitude_reference(i)
    steps_in_phase = 0
    stow_snapshot = None
    for step in range(max_steps):
        result = rig.step()
        steps_in_phase += 1
        if step % STOW_SAMPLE_STEPS == 0:
            # The episode's own state is gone by the time `dones` is visible: env.step()
            # auto-resets the env before it returns, so a stow read afterwards describes the
            # NEXT episode's scatter. Sampling is what makes the last pre-reset picture
            # available -- and it is cheap, because the arm is standing still for the whole
            # DONE coast that a not-quite-stowed episode spends waiting for its time_out.
            previous, stow_snapshot = stow_snapshot, _stow_snapshot(rig.env, i)
            _report_stow_changes(trial, step, rig, i, previous, stow_snapshot)
        new_goal = rig.goals[i]
        if new_goal is not prev_goal:
            if prev_goal is not None:
                info = phase_telemetry(rig, i, prev_goal, attitude_ref, steps_in_phase)
                print(_format_phase_line(trial, info), flush=True)
            steps_in_phase = 0
            prev_goal = new_goal
            attitude_ref = rig.attitude_reference(i)
        if bool(result["dones"][i]) or bool(result["abandoned"][i]):
            return bool(result["dones"][i]), bool(result["abandoned"][i]), stow_snapshot
    raise RuntimeError(f"env {i} did not finish within {max_steps} control steps")


def _inspect_shard(export_dir: str) -> bool:
    """The recorded shard's HDF5 layout, re-checked against a real run."""
    import h5py

    path = os.path.join(export_dir, "dataset.hdf5")
    if not os.path.exists(path):
        print(f"SHARD_FAIL: {path} does not exist", flush=True)
        return False
    ok = True
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        if not demos:
            print("SHARD_FAIL: no demos exported", flush=True)
            return False
        for name in demos:
            grp = f["data"][name]
            num_samples = int(grp.attrs.get("num_samples", 0))
            success = bool(grp.attrs.get("success", False))
            has_wrist = "obs/images/wrist_rgb" in grp
            has_overhead = "obs/images/overhead_rgb" in grp
            clamped = grp["obs/clamped"][:].mean() if "obs/clamped" in grp else float("nan")
            print(
                f"SHARD {name}: num_samples={num_samples} success={success} "
                f"wrist_rgb={has_wrist} overhead_rgb={has_overhead} "
                f"clamped_mean={clamped:.4f}",
                flush=True,
            )
            ok = ok and num_samples > 0 and success and has_wrist and has_overhead
    return ok


def run() -> bool:
    cfg = LegoSortDatagenEnvCfg(cameras=args.with_cameras, num_bricks=args.num_bricks)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    export_dir = tempfile.mkdtemp(prefix="oracle_check_")
    cfg.recorders.dataset_export_dir_path = export_dir
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    cfg.recorders.dataset_filename = "dataset"

    env = gym.make(args.task, cfg=cfg).unwrapped
    bl = cfg._bin_layout()
    noise = oracle.OracleNoise() if args.noise == "on" else NOISE_OFF
    rng = random.Random(args.seed)

    env.reset(seed=args.seed)
    rig = OracleRig(env, bl, noise, rng)
    print(f"EXPORT_DIR {export_dir}", flush=True)

    outcomes = []
    try:
        for trial in range(args.trials):
            if trial > 0:
                # No explicit env.reset() here: ManagerBasedRLEnv.step() already auto-reset
                # env 0 internally the instant the previous trial's dones[0] fired (and the
                # recorder already exported that episode from inside that auto-reset's own
                # record_pre_reset call). An extra explicit reset() here would call
                # record_pre_reset() a second time on an otherwise-untouched env -- and
                # record_pre_reset grades success from termination_manager's STALE cached
                # "success" tensor (last computed on the step that ended the PREVIOUS
                # episode, not recomputed for the fresh post-auto-reset state) -- so it would
                # spuriously export a second, 0-step "successful" episode every trial
                # (measured on openyam-lab: exported_successful_episode_count jumped by 2 on
                # trial 1, not 1). rig.rearm() alone is correct: it only reads current
                # (already-fresh) scene state.
                rig.rearm([0])
            done, abandoned, stow = run_episode(rig, trial)
            # A property, not a method (isaaclab's RecorderManager decorates it with
            # @property despite taking an env_id kwarg that a property access can never
            # pass) -- calling it raises "'int' object is not callable".
            exported = env.recorder_manager.exported_successful_episode_count
            stats = rig.episode_stats(0)
            success = exported == trial + 1
            outcomes.append(success)
            print(
                f"TRIAL {trial}: done={done} abandoned={abandoned} exported_count={exported} "
                f"sorted={stats['sorted']} retries={stats['retries']} "
                f"unplanned_legs={stats['unplanned_legs']} "
                f"abandon_reason={stats['abandon']} "
                f"terminated_by={_termination_reason(env, 0)} "
                f"-> {'SUCCESS' if success else 'FAIL'}",
                flush=True,
            )
            # Only on a failure: the snapshot is sampled up to STOW_SAMPLE_STEPS before the
            # end, so on a success the last brick is usually still in flight in it and
            # printing that would report a brick as lost that the terminating step graded as
            # stowed.
            for entry in ([] if success else stow) or []:
                if not entry["stowed"]:
                    print(
                        f"  NOT_STOWED brick={entry['brick']} "
                        f"offset_from_bin_center="
                        f"({entry['offset_mm'][0]:+.1f},{entry['offset_mm'][1]:+.1f},"
                        f"{entry['offset_mm'][2]:+.1f})mm outside_half_extent_by="
                        f"({entry['outside_mm'][0]:+.1f},{entry['outside_mm'][1]:+.1f},"
                        f"{entry['outside_mm'][2]:+.1f})mm",
                        flush=True,
                    )
    finally:
        env.close()

    ok = bool(outcomes) and all(outcomes)
    if args.with_cameras:
        ok = _inspect_shard(export_dir) and ok
    return ok


try:
    ok = run()
except Exception:  # noqa: BLE001 -- any failure must become ORACLE_FAIL, not a crash.
    traceback.print_exc()
    ok = False

print("ORACLE_OK" if ok else "ORACLE_FAIL", flush=True)
os._exit(0 if ok else 1)
