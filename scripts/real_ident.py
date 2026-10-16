"""Identify the arm's model from excitation session logs (MS5 control plan task 7) and apply
the pre-registered identification rule on held-out logs.

    .venv/bin/python scripts/real_ident.py \\
        --logs ~/data/maker-arm-ms5/logs/s3_exc_0{0..7}.h5 \\
        --heldout ~/data/maker-arm-ms5/logs/s3_exc_0{8,9}.h5 \\
        --out ~/data/maker-arm-ms5/ident.json --report ~/data/maker-arm-ms5/ident_report.json

The fit is ``ident.fit.fit`` on the free-motion samples of ``--logs``, ridge least squares
toward the URDF bodies. ``--out`` is the model file (``core.dynamics_model.from_ident`` reads
it; ``real_session.py --dynamics IDENT.json`` runs it). The rule (MS5 revision section 8.3):
on the ``--heldout`` logs the identified model's residual RMS is lower than the URDF model's
on every joint. ``--report`` holds both RMS vectors, the verdict and the fit's report. Neither
file is ever overwritten.

It prints exactly one ``IDENT_OK {rms_ident, rms_urdf, pass}`` (``pass`` false is still a
finished read) or ``IDENT_FAIL {reason, detail}`` line on every exit path and exits through
``os._exit``.
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
        "--hold_steps", type=_hold_steps, default="0",
        help="ident.fit.HOLD_STEPS: 0 continuous (default), n steps, or 'none'",
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
    try:
        res = fit.fit(logs, dm.urdf_bodies(), lam=args.lam, hold_steps=args.hold_steps)
        rms_ident = fit.residual_rms(heldout, res.dynamics(), hold_steps=args.hold_steps)
        rms_urdf = fit.residual_rms(
            heldout, dm.dynamics(dm.urdf_bodies()), hold_steps=args.hold_steps
        )
    except fit.IdentError as err:
        raise Failure("fit", str(err)) from err
    verdict = all(i < u for i, u in zip(rms_ident, rms_urdf))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(
        out, provenance={"lab_revision": lab_revision(), "heldout": [str(h) for h in heldout]}
    )
    body = {
        "rms_ident": rms_ident,
        "rms_urdf": rms_urdf,
        "pass": verdict,
        "rule": "held-out residual RMS of the identified model below the URDF model's on every "
        "joint (MS5 revision section 8.3)",
        "logs": [str(p) for p in logs],
        "heldout": [str(h) for h in heldout],
        "ident": str(out),
        "armature": res.armature,
        "friction": res.friction,
        "torque_sigma": res.torque_sigma,
        "fit": {k: v for k, v in res.report.items() if k != "provenance"},
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "x") as f:
        json.dump(body, f, indent=2)
    return {"rms_ident": rms_ident, "rms_urdf": rms_urdf, "pass": verdict}


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
