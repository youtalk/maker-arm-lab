"""Design the identification excitations for a session (MS5 control plan task 7): ``count``
Fourier trajectories about a center pose (``ident.excitation``), each checked as the session
will check it, and the session plan that runs them.

    .venv/bin/python scripts/real_excitation.py --rig ~/data/maker-arm-ms5/rig.json \\
        --seed S --count 10 --out_dir ~/data/maker-arm-ms5/plans/exc_S

It writes ``exc_00.npz`` .. ``exc_NN.npz`` (``t``, ``q``, ``grip`` for ``real_session.py``,
and ``dq``, ``ddq``, ``a``, ``b``, ``q0``, ``wf_hz``) and ``plan.json``: a hold, then each
trajectory, then a hold (the session's minimum-jerk approach walks the arm between them at
rest). The profile is the rig's reduced one (``real.rig.reduced_profile``). Every trajectory
must pass ``real.session.check_traj``, keep every moving link origin of the bench plant's
arm at least ``MIN_CLEARANCE_M`` above the rig's desk, and the stacked regressor's condition
number (``ident.fit.excitation_condition``) must be at most ``COND_MAX``. Nothing is written
unless every check passes, and no file is ever overwritten. The pre-registered held-out set
is the last two files of a seed (``exc_08`` and ``exc_09`` for ``--count 10``); the plan
records it.

It prints exactly one ``EXCITE_OK {cond, ...}`` or ``EXCITE_FAIL {reason, detail}`` line on
every exit path and exits through ``os._exit``.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

COND_MAX = 200.0  # INITIAL: session 3
MIN_CLEARANCE_M = 0.05  # INITIAL: session 3
"""How far above the desk every moving link origin of the arm must stay."""
HOLD_S = 5.0
HELDOUT = 2
"""The last two excitations of a seed are the held-out set (MS5 revision section 8.3)."""


class Failure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("EXCITE_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True, help="rig JSON (real.rig.RigConfig)")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--center", type=float, nargs=6, default=None,
        help="the pose to swing about, URDF joint coordinates (default EXCITE_CENTER)",
    )  # fmt: skip
    parser.add_argument(
        "--amplitude", type=float, default=None,
        help="the farthest swing from the center, rad (default EXCITE_AMPLITUDE_RAD)",
    )  # fmt: skip
    parser.add_argument("--wf_hz", type=float, default=0.1)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--v_scale", type=float, default=0.5)
    parser.add_argument("--duration_s", type=float, default=20.0)
    return parser


def min_clearance(trajs, desk_z: float, stride: int = 5) -> float:
    """The lowest height above ``desk_z`` of any moving link origin (link_004 to
    grasp_center) of the bench plant's arm over ``trajs``."""
    import mujoco

    from maker_arm_lab.mj import bench

    plant = bench.build_plant()
    data = mujoco.MjData(plant.model)
    first = plant.model.body("link_004").id
    low = float("inf")
    for traj in trajs:
        for q in traj["q"][::stride]:
            data.qpos[plant.qpos] = q
            mujoco.mj_kinematics(plant.model, data)
            low = min(low, float(data.xpos[first:, 2].min()))
    return low - desk_z


def run(args) -> dict:
    import numpy as np

    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.ident import excitation, fit
    from maker_arm_lab.real import session
    from maker_arm_lab.real.rig import RigConfig, reduced_profile

    if args.center is None:
        args.center = list(excitation.EXCITE_CENTER)
    if args.amplitude is None:
        args.amplitude = excitation.EXCITE_AMPLITUDE_RAD
    if args.count < HELDOUT + 1:
        raise Failure("usage", f"--count must be at least {HELDOUT + 1}, got {args.count}")
    out = Path(args.out_dir).expanduser()
    names = [f"exc_{k:02d}.npz" for k in range(args.count)]
    taken = [n for n in [*names, "plan.json"] if (out / n).exists()]
    if taken:
        raise Failure("exists", f"{out} already holds {taken}; nothing is ever overwritten")
    rig_path = Path(args.rig).expanduser()
    try:
        rig = RigConfig.from_json(rig_path)
        profile = reduced_profile(rig)
    except (OSError, ValueError, TypeError) as err:
        raise Failure("rig", f"{rig_path}: {err}") from err
    try:
        trajs = excitation.design_excitation(
            np.random.default_rng(args.seed), profile, args.center, count=args.count,
            wf_hz=args.wf_hz, harmonics=args.harmonics, v_scale=args.v_scale,
            duration_s=args.duration_s, amplitude_rad=args.amplitude,
        )  # fmt: skip
    except ValueError as err:
        raise Failure("design", str(err)) from err
    for name, traj in zip(names, trajs):
        try:
            session.check_traj(traj, profile)
        except ValueError as err:
            raise Failure("traj", f"{name}: {err}") from err
    clearance = min_clearance(trajs, float(rig.desk_z))
    if clearance < MIN_CLEARANCE_M:
        raise Failure(
            "desk", f"a link comes {clearance:.3f} m from the desk, under {MIN_CLEARANCE_M} m"
        )
    cond, rank = fit.excitation_condition(trajs, dm.urdf_bodies())
    if not cond <= COND_MAX:
        raise Failure("cond", f"the stacked regressor's condition number {cond:.1f} > {COND_MAX}")
    out.mkdir(parents=True, exist_ok=True)
    for name, traj in zip(names, trajs):
        np.savez(out / name, **{k: np.asarray(v) for k, v in traj.items()})
    plan = {
        "segments": [{"hold": HOLD_S}, *({"traj": n} for n in names), {"hold": HOLD_S}],
        "excitation": {
            "seed": args.seed,
            "count": args.count,
            "center": list(args.center),
            "amplitude_rad": args.amplitude,
            "wf_hz": args.wf_hz,
            "harmonics": args.harmonics,
            "v_scale": args.v_scale,
            "duration_s": args.duration_s,
            "cond": cond,
            "base_columns": rank,
            "min_clearance_m": clearance,
            "heldout": names[-HELDOUT:],
        },
    }
    with open(out / "plan.json", "x") as f:
        json.dump(plan, f, indent=2)
    return {
        "cond": cond,
        "base_columns": rank,
        "files": len(names),
        "heldout": names[-HELDOUT:],
        "min_clearance_m": clearance,
        "out_dir": str(out),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
        token, code = "EXCITE_OK " + json.dumps(summary), 0
    except Failure as err:
        token = "EXCITE_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token, code = "EXCITE_FAIL " + json.dumps({"reason": "interrupted", "detail": ""}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "EXCITE_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
