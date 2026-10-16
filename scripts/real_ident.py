"""Identify the arm's model from excitation session logs (MS5 control plan task 7) and apply
the pre-registered identification rule on held-out logs.

    .venv/bin/python scripts/real_ident.py --hold_steps 0 \\
        --logs ~/data/maker-arm-ms5/logs/s3_fit.h5 \\
        --heldout ~/data/maker-arm-ms5/logs/s3_heldout.h5 \\
        --out ~/data/maker-arm-ms5/ident.json --report ~/data/maker-arm-ms5/ident_report.json

``--logs`` are the session logs of ``real_excitation.py``'s ``plan_fit.json`` and
``--heldout`` those of its ``plan_heldout.json`` (the held-out excitations and the static
poses), each read whole.

The fit is ``ident.fit.fit`` on the free-motion samples of ``--logs``, ridge least squares
toward the URDF bodies. ``--out`` is the model file (``core.dynamics_model.from_ident`` reads
it; ``real_session.py --dynamics IDENT.json`` runs it). The rule (MS5 revision section 8.3,
amended 2026-10-16 before session 1): on the ``--heldout`` logs the identified model's
residual RMS is lower than the URDF model's on every joint, scored and judged apart on the two
held-out groups (``HELDOUT_GROUPS``): the held-out excitations (the ``traj`` segments named
``exc_*``, ``exc_08`` and ``exc_09``) and the static poses (``static_*``). Both must pass, with
no margin (``MARGIN`` 0: strictly lower). A joint on which the two models predict the same
torque to within ``TIE_NM`` on every sample of a group (``ident.fit.prediction_spread``)
cannot be told apart there and is reported under ``tied``, not judged: at the static poses the
joints with little gravity load (j1, and the wrist's j5 and j6) see the two models agree to a
few mN m, so a strict rule would read the torque noise there. A group with no judged joint or
no free-motion sample fails (no judged joint: ``pass`` false; no sample: ``heldout``). ``--hold_steps`` (``ident.fit.HOLD_STEPS``: how the motor's PD law runs
through a held command) has no default: its real-arm value is unverified until session 3, and
it moves the identified viscous term by about ``kp T / 2``. The token and the model use the
chosen value; ``--report`` carries the fit, both RMS vectors and the verdict under ``none``
and under the chosen value, and the fit's report. The report is written first, then the
model; neither file is ever overwritten.

It prints exactly one ``IDENT_OK {hold_steps, groups: {trajectories, static}: {rms_ident,
rms_urdf, samples, pass}, pass}`` (``pass`` false is
still a finished read) or ``IDENT_FAIL {reason, detail}`` line on every exit path and exits
through ``os._exit``.
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


HELDOUT_GROUPS = {"trajectories": "exc_", "static": "static_"}
"""The held-out groups of ``plan_heldout.json`` and the ``traj`` label prefix of each."""
MARGIN = 0.0
"""N m: how far below the URDF model's RMS the identified model's must be. Pre-registered as
none: strictly lower on every joint."""
TIE_NM = 0.01  # INITIAL: session 3 reads the torque resolution of the real motors
"""N m: a joint whose two models' predictions never differ by more than this on a group's
samples is tied there, its verdict a coin toss on the noise: on the bench's static poses the
spread reads 0.9 mN m on j1, 6.4 on j5 and 2.1 on j6, against 0.2 to 0.5 N m on j2 to j4 and
the 0.05 N m torque noise of the synthetic logs."""


class Failure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("IDENT_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def _hold_steps(text: str):
    if text == "none":
        return None
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("hold_steps is 'none' or a count >= 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, nargs="+", help="session logs to fit")
    parser.add_argument("--heldout", required=True, nargs="+", help="session logs to judge on")
    parser.add_argument("--out", required=True, help="the ident JSON to create")
    parser.add_argument("--report", required=True, help="the report JSON to create")
    parser.add_argument("--lam", type=float, default=1e-3, help="the ridge weight")
    parser.add_argument(
        "--hold_steps", type=_hold_steps, required=True,
        help="ident.fit.HOLD_STEPS: 0 continuous, n steps, or 'none' (no default)",
    )  # fmt: skip
    return parser


def lab_revision() -> str:
    """HEAD of this checkout, with ``-dirty`` for uncommitted changes to tracked files."""
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def run(args) -> dict:
    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.ident import fit

    out, report = Path(args.out).expanduser(), Path(args.report).expanduser()
    taken = [str(p) for p in (out, report) if p.exists()]
    if taken:
        raise Failure("exists", f"{taken} exist; an ident file or report is never overwritten")
    logs = [Path(p).expanduser() for p in args.logs]
    heldout = [Path(p).expanduser() for p in args.heldout]
    missing = [str(p) for p in [*logs, *heldout] if not p.is_file()]
    if missing:
        raise Failure("logs", f"no such log: {missing}")
    shared = sorted({str(p.resolve()) for p in logs} & {str(p.resolve()) for p in heldout})
    if shared:
        raise Failure("logs", f"a held-out log is also fitted: {shared}")
    urdf_model = dm.dynamics(dm.urdf_bodies())

    def group_read(dynamics, hold_steps) -> dict:
        groups = {}
        for name, prefix in HELDOUT_GROUPS.items():
            samples = sum(
                fit.load_free_motion(h, hold_steps, segment_prefix=prefix)["samples"]
                for h in heldout
            )
            if samples == 0:
                raise Failure(
                    "heldout",
                    f"hold_steps {hold_steps}: no free-motion sample in the held-out {name} "
                    f"(traj segments {prefix}*) of {[str(h) for h in heldout]}",
                )
            kw = {"hold_steps": hold_steps, "segment_prefix": prefix}
            rms_ident = fit.residual_rms(heldout, dynamics, **kw)
            rms_urdf = fit.residual_rms(heldout, urdf_model, **kw)
            spread = fit.prediction_spread(heldout, dynamics, urdf_model, **kw)
            tied = [j for j, s in enumerate(spread) if s <= TIE_NM]
            judged = [j for j in range(len(rms_ident)) if j not in tied]
            groups[name] = {
                "rms_ident": rms_ident,
                "rms_urdf": rms_urdf,
                "samples": int(samples),
                "tied": tied,
                "pass": bool(judged) and all(rms_ident[j] < rms_urdf[j] - MARGIN for j in judged),
            }
        return groups

    def read(hold_steps):
        try:
            res = fit.fit(logs, dm.urdf_bodies(), lam=args.lam, hold_steps=hold_steps)
            groups = group_read(res.dynamics(), hold_steps)
        except fit.IdentError as err:
            raise Failure("fit", f"hold_steps {hold_steps}: {err}") from err
        verdict = all(g["pass"] for g in groups.values())
        return res, {
            "hold_steps": hold_steps,
            "groups": groups,
            "pass": verdict,
            "armature": res.armature,
            "friction": res.friction,
            "torque_sigma": res.torque_sigma,
            "fit": {k: v for k, v in res.report.items() if k != "provenance"},
        }

    res, chosen = read(args.hold_steps)
    if args.hold_steps is None:
        reads = {"none": chosen}
    else:
        try:
            reads = {"none": read(None)[1]}
        except Failure as err:  # a diagnostic read: its failure must not lose the chosen fit
            reads = {"none": {"hold_steps": None, "failed": err.detail}}
    reads[str(args.hold_steps).lower()] = chosen
    body = {
        **{k: chosen[k] for k in ("hold_steps", "groups", "pass")},
        "rule": "held-out residual RMS of the identified model below the URDF model's on every "
        "joint, on the held-out trajectories and on the static poses apart, both to pass "
        "(MS5 revision section 8.3, amended 2026-10-16 before session 1)",
        "margin": MARGIN,
        "logs": [str(p) for p in logs],
        "heldout": [str(h) for h in heldout],
        "ident": str(out),
        "reads": reads,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "x") as f:
        json.dump(body, f, indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov = {"lab_revision": lab_revision(), "heldout": [str(h) for h in heldout]}
    res.to_json(out, provenance={**prov, "hold_steps": args.hold_steps})
    return {k: chosen[k] for k in ("hold_steps", "groups", "pass")}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = "IDENT_OK " + json.dumps(run(args)), 0
    except Failure as err:
        token = "IDENT_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token, code = "IDENT_FAIL " + json.dumps({"reason": "interrupted", "detail": ""}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "IDENT_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
