"""T1 Thor timing script: the wall-clock cost of ``maker_arm_rs.Tracking.update`` from pure
Python, no hardware and no plant physics (tracking design section 9, T1).

stdlib and ``maker_arm_rs`` only, so it runs unmodified on the Thor from a plain checkout:
no torch, no MuJoCo, no numpy. Builds one ``Tracking`` over the exact rigid-body model,
pushes a single 30 Hz reference plan (the desk pose plus a 0.05 rad, 0.5 Hz sine on every arm
joint) and calls ``update`` ``--ticks`` times at the loop's 200 Hz tick rate, timing each call
with ``time.perf_counter_ns``. The measured latency includes the Python call boundary, so it
is conservative for the Rust ``update`` itself.

``--params PARAMS.json`` overrides keys of the controller's parameters (``{"tau_th": [2.0] *
6}`` by default), for example ``{"dob": true, "dob_f": 5.0, "tau_th": [0.5] * 6}`` to time the
observer compensation; ``--residual MODEL.json`` adds a learned residual
(``scripts/real_residual_train.py``'s export). Each tick then passes ``--temperature`` degC on
every motor, what a residual reads. The summary records both, and the share of ticks on which
each layer gave a nonzero torque (``dob_active_share``, ``residual_active_share``: a residual
that skips every tick, its input past ``X_BOUND``, times nothing).

Prints ``TIMING_OK`` or ``TIMING_FAIL`` and a JSON summary on every exit path, argparse
errors included, and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

TICK_S = 0.005  # the tracking loop's own rate
ROW_HZ = 30.0
AMPLITUDE_RAD = 0.05
FREQ_HZ = 0.5
# mj.bench_gate.DESK_POSE, copied: that module imports mujoco and torch, which this Thor
# script must not.
DESK_POSE = (0.0, -2.03, 1.71, -1.17, 0.0, 0.0)
GRIP = 0.0  # unused by the arm-only compliance law; any in-range value works


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("TIMING_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=100_000)
    parser.add_argument("--bar_us", type=float, default=100.0)
    parser.add_argument("--params", default="", help="JSON of tracking parameters to override")
    parser.add_argument("--residual", default="", help="a residual model JSON to run")
    parser.add_argument("--temperature", type=float, default=30.0, help="degC on every motor")
    return parser


def _load_json(path: str, what: str):
    try:
        return json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError) as err:
        raise ValueError(f"{what} {path}: {err}") from err


def _plan_rows(duration: float, dt: float) -> list[list[float]]:
    """30 Hz rows covering ``duration``: the desk pose plus the sine, one row every ``dt``."""
    n = math.ceil(duration / dt) + 1
    rows = []
    for k in range(n):
        s = AMPLITUDE_RAD * math.sin(2.0 * math.pi * FREQ_HZ * (k * dt))
        rows.append([*(q0 + s for q0 in DESK_POSE), GRIP])
    return rows


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence."""
    idx = round(p / 100.0 * (len(sorted_values) - 1))
    return sorted_values[idx]


def run(ticks: int, params_path: str = "", residual_path: str = "", temp: float = 30.0) -> dict:
    # Imported here, under main's try, so a missing wheel still prints the FAIL token.
    import maker_arm_rs

    from maker_arm_lab.core import dynamics_model as dm

    arm_dof = len(maker_arm_rs.profile()["joints"]) - 1
    override = _load_json(params_path, "--params") if params_path else {}
    if not isinstance(override, dict):
        raise TypeError(f"--params {params_path}: the file must hold an object")
    params = {"tau_th": [2.0] * arm_dof, **override}
    residual = _load_json(residual_path, "--residual") if residual_path else None
    tracking = maker_arm_rs.Tracking(
        dm.dynamics(dm.urdf_bodies()),
        params=params,
        profile=maker_arm_rs.profile(),
        **({} if residual is None else {"residual": residual}),
    )
    temperature = [float(temp)] * (arm_dof + 1)
    dob_on = res_on = 0
    row_dt = 1.0 / ROW_HZ
    tracking.push(0.0, row_dt, _plan_rows(ticks * TICK_S, row_dt))
    latencies_ns = []
    for k in range(ticks):
        t = k * TICK_S
        phase = 2.0 * math.pi * FREQ_HZ * t
        s, c = math.sin(phase), math.cos(phase)
        q = [q0 + AMPLITUDE_RAD * s for q0 in DESK_POSE]
        dq = [AMPLITUDE_RAD * 2.0 * math.pi * FREQ_HZ * c] * arm_dof
        tau = [0.5 * s] * arm_dof
        start = time.perf_counter_ns()
        tracking.update(t, [*q, GRIP], [*dq, 0.0], [*tau, 0.0], temperature)
        latencies_ns.append(time.perf_counter_ns() - start)
        tel = tracking.telemetry() or {}
        dob_on += any(v != 0.0 for v in tel.get("tau_dob", ()))
        res_on += any(v != 0.0 for v in tel.get("tau_res", ()))
    latencies_ns.sort()
    return {
        "ticks": ticks,
        "p50_us": _percentile(latencies_ns, 50) / 1000.0,
        "p99_us": _percentile(latencies_ns, 99) / 1000.0,
        "max_us": latencies_ns[-1] / 1000.0,
        "params": override,
        "residual": residual_path,
        "temperature": temp,
        "dob_active_share": dob_on / ticks,
        "residual_active_share": res_on / ticks,
        "uname": platform.uname()._asdict(),
        "python": platform.python_version(),
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args.ticks, args.params, args.residual, args.temperature)
        summary["bar_us"] = args.bar_us
        ok = summary["p99_us"] <= args.bar_us
        print(("TIMING_OK " if ok else "TIMING_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("TIMING_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())
