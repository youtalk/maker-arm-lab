"""T2 Thor soak script: the compliant tracking controller inside ``maker_arm_rs``'s real
200 Hz loop against the built-in simulator, for ``--minutes`` (tracking design section 9, T2).

stdlib and ``maker_arm_rs`` only, so it runs unmodified on the Thor from a plain checkout:
no torch, no MuJoCo, no numpy. Every leg is a minimum-jerk walk between the desk and reach
poses, planned at 30 Hz and pushed a tenth of a second ahead of the loop clock. Once a second
it reads ``arm.snapshot()``: any state other than ``"enabled"`` ends the soak as a failure.
At the end it reads ``Tracking.telemetry()`` for the tick health (``max_interval``,
``late_ticks``, ``gaps``) and always disables the arm before returning.

Prints ``SOAK_OK`` or ``SOAK_FAIL`` and a JSON summary on every exit path, argparse errors
included, and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import maker_arm_rs

from maker_arm_lab.core import dynamics_model as dm

ROW_HZ = 30.0
LEG_S = 3.0
PUSH_LEAD_S = 0.1
MAX_INTERVAL_S = 0.010  # the RT bar; only checked when --rt is given
SNAPSHOT_POLL_S = 1.0
FIRST_TICK_TIMEOUT_S = 5.0
# mj.bench_gate.DESK_POSE / REACH_POSE, copied: that module imports mujoco and torch, which
# this Thor script must not.
DESK_POSE = (0.0, -2.03, 1.71, -1.17, 0.0, 0.0)
REACH_POSE = (0.0, -1.2, 0.6, -0.5, 0.0, 0.0)


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("SOAK_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def _rt_pair(text: str) -> tuple[int, int]:
    priority, cpu = text.split(":")
    return int(priority), int(cpu)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument(
        "--rt", type=_rt_pair, default=None, help="PRIORITY:CPU for the loop thread"
    )
    return parser


def min_jerk(u: float) -> float:
    return u**3 * (10.0 - 15.0 * u + 6.0 * u * u)


def leg_rows(qa, qb, grip: float) -> list[list[float]]:
    """A minimum-jerk leg from ``qa`` to ``qb``, one row every 1/ROW_HZ s, LEG_S long."""
    n = round(LEG_S * ROW_HZ)
    rows = []
    for k in range(n + 1):
        s = min_jerk(k / n)
        rows.append([*(a + (b - a) * s for a, b in zip(qa, qb)), grip])
    return rows


def _first_gripper(arm, timeout: float = FIRST_TICK_TIMEOUT_S) -> float:
    """The gripper position off the loop's first published snapshot (None until it ticks)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = arm.snapshot()
        if snap is not None:
            return snap["positions"][-1]
        time.sleep(0.005)
    raise RuntimeError(f"no snapshot within {timeout} s of start_tracking")


def run_soak(minutes: float, rt: tuple[int, int] | None) -> dict:
    arm = maker_arm_rs.Arm.sim()
    try:
        arm.enable()
        arm_dof = len(maker_arm_rs.profile()["joints"]) - 1
        tracking = maker_arm_rs.Tracking(
            dm.dynamics(dm.urdf_bodies()), params={"tau_th": [2.0] * arm_dof}
        )
        arm.start_tracking(tracking, rt=rt)
        grip = _first_gripper(arm)
        start = time.monotonic()
        deadline = start + minutes * 60.0
        waypoints, leg, q_from, enabled = (DESK_POSE, REACH_POSE), 0, DESK_POSE, True
        while time.monotonic() < deadline:
            q_to = waypoints[(leg + 1) % 2]
            tracking.push(tracking.now() + PUSH_LEAD_S, 1.0 / ROW_HZ, leg_rows(q_from, q_to, grip))
            leg_end = min(time.monotonic() + LEG_S, deadline)
            while enabled and time.monotonic() < leg_end:
                time.sleep(max(0.0, min(SNAPSHOT_POLL_S, leg_end - time.monotonic())))
                snap = arm.snapshot()
                if snap is not None and snap["state"] != "enabled":
                    enabled = False
            if not enabled:
                break
            q_from, leg = q_to, leg + 1
        telemetry = tracking.telemetry()
        ok = enabled and (rt is None or telemetry["max_interval"] <= MAX_INTERVAL_S)
        return {
            "pass": ok,
            "enabled": enabled,
            "minutes_requested": minutes,
            "elapsed_s": time.monotonic() - start,
            "rt": list(rt) if rt is not None else None,
            "telemetry": {k: telemetry[k] for k in ("max_interval", "late_ticks", "gaps")},
            "uname": platform.uname()._asdict(),
        }
    finally:
        arm.stop()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_soak(args.minutes, args.rt)
        ok = summary["pass"]
        print(("SOAK_OK " if ok else "SOAK_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("SOAK_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())
