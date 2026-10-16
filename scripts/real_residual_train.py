"""Train the residual torque model on session logs (MS5 control plan task 10, the MA3 recipe)
and export it for ``maker_arm_rs``.

    .venv/bin/python scripts/real_residual_train.py \\
        --logs ~/data/maker-arm-ms5/logs/s8_a.h5 ~/data/maker-arm-ms5/logs/s8_b.h5 \\
        --ident ~/data/maker-arm-ms5/ident.json --history 10 \\
        --out ~/data/maker-arm-ms5/residual.json --report ~/data/maker-arm-ms5/residual_report.json

The dataset is ``models.residual.build_dataset`` over ``--logs`` (in the order given, which
should be time order: the validation split is the last ``--val_share`` of the rows) against
the model of ``--ident`` (``core.dynamics_model.from_ident``). The hold model is the ident
file's own (its report's ``hold_steps``), so the residual is what the model misses on the rows
it was fitted to; ``--hold_steps`` must agree with it, and is needed only for an ident file
that does not record one. The model is ``models.residual.train``; ``--out`` is the SDK's
residual dict (``real_session.py --residual MODEL.json`` runs it), with ``out_max`` at
``OUT_MAX_SIGMA`` standard deviations of the target. The SDK builds a controller with the
model before anything is written. ``--report`` holds the dataset's counts, the training
report (per joint RMSE of the model and of the zero model on the validation rows, the rows
past the SDK's ``X_BOUND`` after the scale floors), ``model_sha256`` (the sha256 of the model
file) and the provenance, with ``ident_sha256`` (the sha256 of the ``--ident`` file): ruling
R35, ``real_session.py --residual MODEL.json --residual_report REP.json`` runs the model only
over the ident file it was trained against. The report is written
first, then the model; neither file is ever overwritten.

It prints exactly one ``RESID_OK {val_rmse, zero_rmse, rows, beyond_x_bound}`` (a model that
does not beat zero is still a finished read: compare the two) or ``RESID_FAIL {reason,
detail}`` line on every exit path and exits through ``os._exit``.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

MIN_ROWS = 200
"""Fewer dataset rows than this is no training set."""


class Failure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("RESID_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
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
    parser.add_argument("--logs", required=True, nargs="+", help="session logs, in time order")
    parser.add_argument("--ident", required=True, help="the ident JSON (real_ident.py)")
    parser.add_argument("--history", type=int, default=10, help="ticks of q and dq per input")
    parser.add_argument("--out", required=True, help="the residual model JSON to create")
    parser.add_argument("--report", required=True, help="the report JSON to create")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_share", type=float, default=0.2)
    parser.add_argument(
        "--hold_steps", type=_hold_steps, default=argparse.SUPPRESS,
        help="ident.fit.HOLD_STEPS; must match the ident file's, which is the default",
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


def _ident_hold_steps(data: dict):
    """The hold model the ident file was fitted under, or "unset" if it records none."""
    for where in (data.get("report", {}), data.get("provenance", {})):
        if isinstance(where, dict) and "hold_steps" in where:
            return where["hold_steps"]
    return "unset"


def run(args) -> dict:
    import maker_arm_rs

    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.models import residual
    from maker_arm_lab.real.telemetry import sdk_revision

    out, report = Path(args.out).expanduser(), Path(args.report).expanduser()
    taken = [str(p) for p in (out, report) if p.exists()]
    if taken:
        raise Failure("out_exists", f"{taken} exist; a model or report is never overwritten")
    if args.history < 1 or args.epochs < 1 or not 0.0 < args.val_share < 1.0:
        raise Failure("usage", "history and epochs need >= 1 and val_share (0, 1)")
    logs = [Path(p).expanduser() for p in args.logs]
    missing = [str(p) for p in logs if not p.is_file()]
    if missing:
        raise Failure("logs", f"no such log: {missing}")
    ident = Path(args.ident).expanduser()
    try:
        dynamics = dm.from_ident(ident)
        ident_bytes = ident.read_bytes()
        ident_data = json.loads(ident_bytes)
    except (OSError, ValueError, TypeError, KeyError) as err:
        raise Failure("ident", f"{ident}: {err}") from err
    recorded = _ident_hold_steps(ident_data)
    given = getattr(args, "hold_steps", "unset")
    if recorded == "unset" and given == "unset":
        raise Failure("hold_steps", f"{ident} records no hold_steps: pass --hold_steps")
    if recorded != "unset" and given not in ("unset", recorded):
        raise Failure("hold_steps", f"--hold_steps {given} differs from the ident's {recorded}")
    hold_steps = recorded if recorded != "unset" else given
    info: dict = {}
    try:
        X, Y = residual.build_dataset(logs, dynamics, args.history, hold_steps, info=info)
    except (OSError, KeyError, ValueError) as err:
        raise Failure("dataset", str(err)) from err
    if X.shape[0] < MIN_ROWS:
        raise Failure("dataset", f"{X.shape[0]} rows, fewer than {MIN_ROWS}: {info}")
    try:
        model, rep = residual.train(
            X, Y, seed=args.seed, epochs=args.epochs, lr=args.lr, val_share=args.val_share
        )
    except ValueError as err:
        raise Failure("train", str(err)) from err
    try:  # the out_max cap, then the SDK, must take the file before it is written
        spec = residual.residual_spec(model, rep["stats"], args.history)
        maker_arm_rs.Tracking(dynamics, residual=spec)
    except (TypeError, ValueError) as err:
        raise Failure("export", str(err)) from err
    body = {
        "val_rmse": rep["val_rmse"],
        "zero_rmse": rep["zero_rmse"],
        "beats_zero": [v < z for v, z in zip(rep["val_rmse"], rep["zero_rmse"])],
        "hold_steps": hold_steps,
        "history": args.history,
        "out_max": spec["out_max"],
        "out_max_sigma": residual.OUT_MAX_SIGMA,
        "dataset": {k: v for k, v in info.items() if k != "t"},
        "train": rep,
        "model": str(out),
        "model_sha256": hashlib.sha256(residual.spec_json(spec).encode()).hexdigest(),
        "provenance": {
            "logs": [str(p) for p in logs],
            "ident": str(ident),
            "ident_sha256": hashlib.sha256(ident_bytes).hexdigest(),
            "ident_provenance": ident_data.get("provenance", {}),
            "lab_revision": lab_revision(),
            "sdk_revision": sdk_revision(),
            "date": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "x") as f:
        json.dump(body, f, indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    residual.export(model, rep["stats"], args.history, spec["out_max"], out)
    return {
        "val_rmse": rep["val_rmse"],
        "zero_rmse": rep["zero_rmse"],
        "rows": int(X.shape[0]),
        "beyond_x_bound": rep["beyond_x_bound"],
    }


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
        token, code = "RESID_OK " + json.dumps(run(args)), 0
    except Failure as err:
        token = "RESID_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token, code = "RESID_FAIL " + json.dumps({"reason": "interrupted", "detail": ""}), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "RESID_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
