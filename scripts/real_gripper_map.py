"""Measure the real gripper's motor angle to jaw gap line with a caliper (MS5 control plan
task 8; MS5 open point O6).

    source ~/opt/env.sh  # the Thor's venv
    PATH=$HOME/.cargo/bin:$PATH .venv/bin/python scripts/real_gripper_map.py \\
        --rig ~/data/maker-arm-ms5/rig.json --angles 8 --rig_out ~/data/maker-arm-ms5/rig_g.json

The arm is held under the rig's reduced caps (``real.rig.reduced_profile``). The script steps
the gripper motor through ``--angles`` angles, evenly across its soft limits less
``real.session.TRAJ_MARGIN_RAD``, at ``real.session.APPROACH_SPEED_RAD_S``, and at each one
waits ``SETTLE_S`` and asks for the gap between the jaws in mm. It drops anything typed
before the question and accepts only a finite number from 0 up to ``MAX_GAP_MM``; anything else asks again. It
fits ``gap = a * angle + b`` (m) by least squares. The fit fails unless the gap grows toward
the open limit (the profile gripper's ``q_hi``) by at least ``MIN_GAP_SPAN_M`` over the angles
and is positive there. ``full_gap_mm`` is the gap at that limit.

``--rig_out`` is the input rig with ``gripper_map = [a, b]``; it must not exist, and the input
rig is never written. It prints exactly one ``GRIPPER_MAP_OK {a, b, full_gap_mm, rms_mm,
angles, gaps_mm}`` or ``GRIPPER_MAP_FAIL {reason, detail}`` line on every exit path and exits
through ``os._exit``. The arm session is ``scripts/real_session.py``'s: a failure (a fault,
Ctrl-C, an exception, the end of input at a question) holds the arm, and the motors are
disabled only on the operator's typed ``RELEASE`` or at the end of input (``--release
prompt``, the default) or at once (``--release now``, for tests). A person stands at the
e-stop.
"""

import argparse
import dataclasses
import json
import math
import os
import signal
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))

import real_session as rs
from maker_arm_lab.real import session as real_session
from maker_arm_lab.robots.frames import JAW_FULL_GAP

SETTLE_S = 1.0  # INITIAL: session 2
"""Seconds the gripper holds at an angle before the question."""
MIN_GAP_SPAN_M = 0.001  # INITIAL: session 2
"""The fit needs the measured gap to span at least this much, or the caliper line is flat."""


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("GRIPPER_MAP_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True, help="rig JSON (real.rig.RigConfig)")
    parser.add_argument("--angles", type=int, default=8, help="gripper angles to measure (>= 2)")
    parser.add_argument("--rig_out", required=True, help="the rig JSON with gripper_map to create")
    parser.add_argument(
        "--interface", default="", help="'sim' for the SDK simulator; default the rig's CAN"
    )
    parser.add_argument("--release", choices=("prompt", "now"), default="prompt")
    return parser


class NullLog:
    """The session machinery's log surface, with nothing kept: a gripper map is not logged."""

    def __init__(self):
        self.ticks = self.dropped = 0

    def append(self, ticks, tracking, mono_ns=None) -> None:
        if ticks and ticks.get("t"):
            self.ticks += len(ticks["t"])

    def add_segment(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def fit_line(angles, gaps_m, q_hi: float) -> dict:
    """Least squares ``gap = a * angle + b``; raises SessionFailure("fit") when the gap does
    not grow toward ``q_hi`` over at least ``MIN_GAP_SPAN_M`` or is not positive there."""
    x, y = np.asarray(angles, dtype=np.float64), np.asarray(gaps_m, dtype=np.float64)
    a, b = (float(v) for v in np.polyfit(x, y, 1))
    span = a * float(x.max() - x.min())
    full = a * q_hi + b
    if not (span >= MIN_GAP_SPAN_M and full > 0.0):
        raise real_session.SessionFailure(
            "fit",
            f"gap = {a:.6f} * angle + {b:.6f} m spans {1000 * span:.2f} mm over the angles and "
            f"is {1000 * full:.2f} mm at the open limit {q_hi} rad; it must grow toward the "
            f"open limit by at least {1000 * MIN_GAP_SPAN_M} mm",
        )
    rms = float(np.sqrt(np.mean((a * x + b - y) ** 2)))
    return {"a": a, "b": b, "full_gap_mm": 1000.0 * full, "rms_mm": 1000.0 * rms}


MAX_GAP_MM = 2.0 * 1000.0 * float(JAW_FULL_GAP)  # INITIAL: session 2
"""A typed gap above this (twice the provisional full gap) is a typo, not a reading."""


def parse_gap_mm(text: str) -> float | None:
    """The caliper reading in mm, or None unless the line is a finite number in
    (0, ``MAX_GAP_MM``]."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and 0.0 < value <= MAX_GAP_MM else None


def prepare(args) -> dict:
    """Everything short of the arm, checked. Raises SessionFailure."""
    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.real.rig import RigConfig, reduced_profile

    fail = real_session.SessionFailure
    rig_path = Path(args.rig).expanduser()
    try:
        rig = RigConfig.from_json(rig_path)
    except (OSError, ValueError, TypeError) as err:
        raise fail("rig", f"{rig_path}: {err}") from err
    try:
        profile = reduced_profile(rig)
    except (TypeError, ValueError) as err:
        raise fail("profile", str(err)) from err
    if args.angles < 2:
        raise fail("angles", f"--angles needs at least 2, got {args.angles}")
    out_path = Path(args.rig_out).expanduser()
    if out_path.exists() or out_path.resolve() == rig_path.resolve():
        raise fail("rig_out_exists", f"{out_path} exists; a rig JSON is never overwritten")
    g = profile["joints"][6]
    lo, hi = float(g["q_lo"]) + real_session.TRAJ_MARGIN_RAD, float(g["q_hi"])
    hi -= real_session.TRAJ_MARGIN_RAD
    if not lo < hi:
        raise fail("profile", f"the gripper range {lo}..{hi} rad is empty")
    interface = args.interface or rig.can_interface
    rt = None if interface == "sim" or rig.rt is None else (int(rig.rt[0]), int(rig.rt[1]))
    return {
        "rig": rig,
        "profile": profile,
        "dynamics": dm.dynamics(dm.urdf_bodies()),
        "params": dict(rig.tracking_params),
        "interface": interface,
        "rt": rt,
        "out_path": out_path,
        "angles": [lo + (hi - lo) * k / (args.angles - 1) for k in range(args.angles)],
    }


def measure(world: dict, ready: dict, lines) -> list[float]:
    """Step the gripper through the angles and ask for the gap at each: mm per angle."""
    sdk, log = world["arm"], world["log"]
    pose = list(sdk.read().q)  # the arm stays where it is; only the gripper moves
    gaps = []
    for k, angle in enumerate(ready["angles"]):
        traj = {"t": [0.0], "q": [pose], "grip": [angle]}
        traj = {key: np.asarray(v, dtype=np.float64) for key, v in traj.items()}
        real_session.check_traj(traj, ready["profile"])
        label = f"angle {angle:.6f} rad"
        segments = [
            {"kind": "traj", "label": label, "traj": traj},
            {"kind": "hold", "seconds": SETTLE_S},
        ]
        question = (
            f"Gripper {k + 1}/{len(ready['angles'])} at angle {angle:.6f} rad. "
            "Measure the gap between the jaws with the caliper and type it in mm."
        )
        while True:
            segments.append({"kind": "prompt", "text": question, "record": f"gap_{k}"})
            summary = real_session.run_session(
                sdk,
                segments,
                log,
                lines,
                interrupted=lambda: world.get("interrupted", False),
            )
            gap = parse_gap_mm(summary["records"][f"gap_{k}"])
            if gap is not None:
                gaps.append(gap)
                break
            print(
                f"Not a number in (0, {MAX_GAP_MM:.0f}] mm; measure again.",
                file=sys.stderr,
                flush=True,
            )
            segments = []
            question = f"Gripper {k + 1} again, angle {angle:.6f} rad: the gap in mm."
    return gaps


def _run(args, world: dict, phase: list) -> int:
    lines = None
    code, token = 1, None
    try:
        try:
            lines = real_session.Lines().start()
            ready = prepare(args)
            try:
                world["arm"] = rs.connect_arm(
                    ready["interface"], ready["dynamics"], ready["profile"], ready["params"],
                    ready["rt"],
                )  # fmt: skip
            except ValueError as err:
                raise real_session.SessionFailure("profile", str(err)) from err
            phase[0] = "session"
            world["log"] = NullLog()
            world["enabled"] = True
            world["arm"].start()
            gaps = measure(world, ready, lines)
            result = fit_line(
                ready["angles"],
                [g / 1000.0 for g in gaps],
                float(ready["profile"]["joints"][6]["q_hi"]),
            )
            result["angles"], result["gaps_mm"] = ready["angles"], gaps
            rig = dataclasses.replace(ready["rig"], gripper_map=(result["a"], result["b"]))
            try:
                with open(ready["out_path"], "x") as f:
                    f.write(rig.to_json())
            except OSError as err:
                raise real_session.SessionFailure(
                    "write_rig", f"{err}; the fit was {json.dumps(result)}"
                ) from err
            result["rig_out"] = str(ready["out_path"])
            token, code = "GRIPPER_MAP_OK " + json.dumps(result), 0
        finally:
            phase[0] = "closing"
    except real_session.SessionFailure as err:
        token = "GRIPPER_MAP_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token = "GRIPPER_MAP_FAIL " + json.dumps({"reason": "interrupted", "detail": "Ctrl-C"})
        code = 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token = "GRIPPER_MAP_FAIL " + json.dumps({"reason": "exception", "detail": detail})
        code = 1
    if code != 0:
        rs.hold(world)
    print(token, flush=True)
    rs.release(world, args.release, lines if lines is not None else real_session.Lines())
    return code


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    world: dict = {}
    phase = ["setup"]
    previous = signal.signal(signal.SIGINT, rs.make_sigint_handler(world, phase))
    try:
        return _run(args, world, phase)
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    os._exit(main())
