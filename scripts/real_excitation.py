"""Design the identification excitations for a session (MS5 control plan task 7): ``count``
Fourier trajectories about a center pose (``ident.excitation``), each checked as the session
will check it, and the two session plans that run them.

    .venv/bin/python scripts/real_excitation.py --rig ~/data/maker-arm-ms5/rig.json \\
        --seed S --count 10 --out_dir ~/data/maker-arm-ms5/plans/exc_S

It writes ``exc_00.npz`` .. ``exc_NN.npz`` (``t``, ``q``, ``grip`` for ``real_session.py``,
and ``dq``, ``ddq``, ``a``, ``b``, ``q0``, ``wf_hz``), ``static_00.npz`` .. ``static_02.npz``
(a pose of the held-out excitations held ``STATIC_S`` at rest) and two session plans, each a
hold, its files, a hold (the session's minimum-jerk approach walks the arm between them at
rest): ``plan_fit.json`` runs every excitation but the last two, ``plan_heldout.json`` runs
the last two (``exc_08`` and ``exc_09`` for ``--count 10``, the pre-registered held-out set,
MS5 revision section 8.3) and the static poses. Each plan is its own session and so its own
log: ``real_ident.py --logs`` takes the fit plan's logs and ``--heldout`` the held-out
plan's, whole files. The profile is the rig's reduced one (``real.rig.reduced_profile``).
Every file must pass ``real.session.check_traj``, the excitations must keep every moving link
origin of the bench plant's arm at least ``MIN_CLEARANCE_M`` above the rig's desk, and the
fit excitations' stacked regressor must have a condition number (``ident.fit
.excitation_condition``) of at most ``COND_MAX``; the token names the columns that number
leaves out (``dropped_columns``, with the residual of each). Nothing is written unless every
check passes, and no file is ever overwritten.

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
STATIC_POSES = 3
STATIC_S = 3.0
"""The held-out plan also holds this many poses for this long, for 8.3's static part."""
PLAN_FIT = "plan_fit.json"
PLAN_HELDOUT = "plan_heldout.json"


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


def static_poses(rng, trajs, count: int, seconds: float, dt: float) -> list[dict]:
    """``count`` held poses for the held-out plan, each a row drawn at random from the held-out
    excitations (so inside their limits and clearance), held ``seconds`` at rest."""
    out = []
    for k in range(count):
        traj = trajs[k % len(trajs)]
        row = traj["q"][int(rng.integers(traj["q"].shape[0]))]
        n = round(seconds / dt) + 1
        t = [i * dt for i in range(n)]
        out.append({"t": t, "q": [list(row)] * n, "grip": [float(traj["grip"][0])] * n})
    return out


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
    statics = [f"static_{k:02d}.npz" for k in range(STATIC_POSES)]
    fit_names, held_names = names[:-HELDOUT], names[-HELDOUT:]
    files = [*names, *statics, PLAN_FIT, PLAN_HELDOUT]
    taken = [n for n in files if (out / n).exists()]
    if taken:
        raise Failure("exists", f"{out} already holds {taken}; nothing is ever overwritten")
    rig_path = Path(args.rig).expanduser()
    try:
        rig = RigConfig.from_json(rig_path)
        profile = reduced_profile(rig)
    except (OSError, ValueError, TypeError) as err:
        raise Failure("rig", f"{rig_path}: {err}") from err
    rng = np.random.default_rng(args.seed)
    try:
        trajs = excitation.design_excitation(
            rng, profile, args.center, count=args.count,
            wf_hz=args.wf_hz, harmonics=args.harmonics, v_scale=args.v_scale,
            duration_s=args.duration_s, amplitude_rad=args.amplitude,
        )  # fmt: skip
    except ValueError as err:
        raise Failure("design", str(err)) from err
    holds = [
        {k: np.asarray(v, dtype=np.float64) for k, v in h.items()}
        for h in static_poses(rng, trajs[-HELDOUT:], STATIC_POSES, STATIC_S, session.DT)
    ]
    for name, traj in zip([*names, *statics], [*trajs, *holds]):
        try:
            session.check_traj(traj, profile)
        except ValueError as err:
            raise Failure("traj", f"{name}: {err}") from err
    clearance = min_clearance(trajs, float(rig.desk_z))
    if clearance < MIN_CLEARANCE_M:
        raise Failure(
            "desk", f"a link comes {clearance:.3f} m from the desk, under {MIN_CLEARANCE_M} m"
        )
    cond, rank, dropped = fit.excitation_condition(trajs[:-HELDOUT], dm.urdf_bodies())
    weak = [[fit.COLUMN_NAMES[i], r] for i, r in dropped]
    if not cond <= COND_MAX:
        raise Failure("cond", f"the stacked regressor's condition number {cond:.1f} > {COND_MAX}")
    out.mkdir(parents=True, exist_ok=True)
    for name, traj in zip([*names, *statics], [*trajs, *holds]):
        np.savez(out / name, **{k: np.asarray(v) for k, v in traj.items()})
    meta = {
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
        "dropped_columns": weak,
        "min_clearance_m": clearance,
        "fit": fit_names,
        "heldout": [*held_names, *statics],
    }
    plans = {
        PLAN_FIT: fit_names,
        PLAN_HELDOUT: [*held_names, *statics],
    }
    for plan_name, segs in plans.items():
        plan = {
            "segments": [{"hold": HOLD_S}, *({"traj": n} for n in segs), {"hold": HOLD_S}],
            "excitation": {**meta, "plan": plan_name},
        }
        with open(out / plan_name, "x") as f:
            json.dump(plan, f, indent=2)
    return {
        "cond": cond,
        "base_columns": rank,
        "dropped": len(weak),
        "dropped_columns": weak,
        "plans": list(plans),
        "fit": fit_names,
        "heldout": [*held_names, *statics],
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
