"""The tracking bench's scenarios, metrics, deadband calibration and gate table (tracking
design section 8).

Every definition here is pre-registered in ``docs/ma2-tracking-bench.md`` before the first
gate run: the poses, the plans each scenario pushes, the metric each one reads, the deadband
calibration rule and the bars. ``run_suite`` runs every rung and model on ``mj.bench``'s
plant and returns every number, the gate rows and the verdict. Nothing here is tuned from a
gate result.
"""

from __future__ import annotations

import math
import random
from functools import partial
from itertools import pairwise

import maker_arm_rs
import numpy as np

from maker_arm_lab.core import desk_boxes, layout, oracle
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.core.global_ik import GlobalIK, GraspTarget
from maker_arm_lab.core.trajectory import CONTROL_HZ, TRANSPORT_SPEED, min_jerk, polyline_point
from maker_arm_lab.mj import bench
from maker_arm_lab.robots.frames import GRASP_TILT_RAD
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import LEAN_AZIMUTH, leg_resolution

DESK_POSE = (0.0, -2.03, 1.71, -1.17, 0.0, 0.0)
REACH_POSE = (0.0, -1.2, 0.6, -0.5, 0.0, 0.0)
HOVER_Z = 0.12 + layout.BRICK_SIZE[2] / 2  # the nominal draw's hover height over a 2x4
RELEASE_XY = (sum(desk_boxes.DESK_BOX_BAND_X) / 2, sum(desk_boxes.DESK_BOX_BAND_Y) / 2)
# The level release asks for PICK_TILT_RAD (0.0), and the oracle runtime reads a carry leg's
# tilt as max(goal tilt, GRASP_TILT_RAD): at exactly zero tilt the lean azimuth is undefined
# and the release point has no IK solution. This is the tilt the oracle's release flies.
RELEASE_TILT = max(oracle.PICK_TILT_RAD, GRASP_TILT_RAD)
PLAN_T0 = 0.5
ROW_DT = 1.0 / CONTROL_HZ
STATIC_S, SAG_AT_S = 3.0, 2.0
SETTLE_S, SWAY_S, WALL_PROBE_S = 1.5, 0.5, 0.3
PRESS_ABOVE, PRESS_FRACTION, PRESS_HOLD_S, PRESS_PROBE_S, PRESS_STEP_M = 0.03, 0.7, 1.0, 0.5, 0.003
PRESS_MAX_M, PRESS_TOL_M = 0.05, 1e-5
CHUNK, CHUNK_SIGMA = 50, 0.005  # INITIAL: the policy-like chunk offset
WALKS, CAL_SEED, GATE_SEED = 8, 1, 2
TARGET_BOX = ((0.10, 0.26), (-0.20, 0.20), (0.08, 0.16))
CAL_TAIL_S = 1.0
REST_S, REST_SKIP_S = 2.5, 0.5
OSC_WINDOW_S, OSC_FLOOR = 0.5, 0.02  # rad/s
MASS_SWEEP = (0.7, 1.3)
SEED = 20260925
SMOKE_S = 0.5
# TrackingParams' INITIAL defaults in maker-arm-rs (crates/maker-arm/src/tracking.rs), copied
# for the record because the binding has no getter. A test pins the copy against the build.
TRACKING_INITIAL = {
    "f_r": 10.0,
    "k_o": 50.0,
    "f_y": 2.0,
    "e_max": [0.03] * bench.ARM_DOF,
    "gate_fall": 0.020,
    "gate_rise": 0.100,
    "gate_quiet": 0.050,
}
BARS = (
    ("exact", "static_sag_mm", "abs", 1.0),
    ("exact", "sway_mm", "rel", 0.3),
    ("exact", "overshoot_mm", "rel", 0.3),
    ("exact", "false_yield_mm", "abs", 1.0),
    ("exact", "wall_force_03", "rel", 0.5),
    ("exact", "wall_force_peak", "rel", 1.0),
    ("exact", "press_force", "band", 0.2),
    ("housing", "sway_mm", "rel", 0.5),
    ("housing", "false_yield_mm", "abs", 3.0),
    ("housing", "growing", "none", None),
)


# --- poses and plans ---------------------------------------------------------------------
def _target(xyz, tilt: float) -> GraspTarget:
    return GraspTarget(tuple(float(v) for v in xyz), tilt, 0.0, LEAN_AZIMUTH)


def _pose(ik, xyz, tilt: float, what: str, seed=None) -> list[float]:
    found = ik.solve(_target(xyz, tilt), seed=seed)
    if found is None:
        raise RuntimeError(f"no IK solution for the {what} pose at {tuple(xyz)}")
    return found.as_list()


def _hover(ik) -> list[float]:
    return _pose(ik, (*layout.TRAY_CENTER, HOVER_Z), GRASP_TILT_RAD, "hover")


def _tip(ik, q) -> np.ndarray:
    return np.asarray(ik.fk(list(q))[0], dtype=float)


def poses(ik) -> dict[str, list[float]]:
    release = (*RELEASE_XY, desk_boxes.DESK_BOX_RELEASE_Z)
    return {
        "desk": list(DESK_POSE),
        "reach": list(REACH_POSE),
        "hover": _hover(ik),
        "release": _pose(ik, release, RELEASE_TILT, "release"),
    }


def transport_rows(ik, qa, qb) -> list[list[float]]:
    """A min-jerk joint-space leg at TRANSPORT_SPEED, one 7-wide row per 1/30 s."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    steps = leg_resolution(
        10**6,
        1.0,
        distance=float(np.linalg.norm(_tip(ik, qb) - _tip(ik, qa))),
        joint_change=float(np.max(np.abs(qb - qa))),
        speed_scale=TRANSPORT_SPEED,
    )[1]
    grip = bench.grip_value()
    return [[*(qa + (qb - qa) * min_jerk(k / steps)).tolist(), grip] for k in range(steps + 1)]


def random_targets(ik, seed: int, n: int) -> list[list[float]]:
    """``n`` walk poses drawn in TARGET_BOX, each solved from the one before (the hover pose
    for the first). A draw with no solution is replaced by the generator's next draw."""
    rng = random.Random(seed)
    prev, out = _hover(ik), []
    for _ in range(100 * n):  # a bound, so a broken solver fails instead of hanging
        xyz = [rng.uniform(lo, hi) for lo, hi in TARGET_BOX]
        found = ik.solve(_target(xyz, GRASP_TILT_RAD), seed=prev)
        if found is not None:
            prev = found.as_list()
            out.append(prev)
            if len(out) == n:
                return out
    raise RuntimeError(f"random_targets: {len(out)} of {n} targets solved in {100 * n} draws")


def walk_rows(ik, qs) -> list[list[float]]:
    """Transport legs through ``qs`` in order; each junction row appears once."""
    rows: list[list[float]] = []
    for qa, qb in pairwise(qs):
        leg = transport_rows(ik, qa, qb)
        rows += leg[1:] if rows else leg
    return rows


def _kp() -> np.ndarray:
    joints = _profile.sim_profile()["joints"][: bench.ARM_DOF]
    return np.array([float(j["kp"]) for j in joints])


def press_depth(ik, tau_th) -> float:
    """The first depth below the desk contact at which the PD's press torque reaches
    PRESS_FRACTION of the deadband at some joint, by bisection."""
    q_c = _pose(ik, (*layout.TRAY_CENTER, bench.TIP_RADIUS), GRASP_TILT_RAD, "press contact")
    ratio = _kp() / np.asarray(tau_th, float)

    def reaches(d: float) -> bool:
        xyz = (*layout.TRAY_CENTER, bench.TIP_RADIUS - d)
        q_d = _pose(ik, xyz, GRASP_TILT_RAD, "press depth", seed=q_c)
        return float(np.max(ratio * np.abs(np.subtract(q_d, q_c)))) >= PRESS_FRACTION

    lo, hi = 0.0, PRESS_MAX_M
    if not reaches(hi):
        raise RuntimeError(f"press_depth: no depth up to {PRESS_MAX_M} m reaches the fraction")
    while hi - lo > PRESS_TOL_M:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if reaches(mid) else (mid, hi)
    return hi


def descend_rows(ik, depth: float) -> list[list[float]]:
    """Scenario 5's plan: IK waypoints at most PRESS_STEP_M apart from PRESS_ABOVE over the
    desk contact down to ``depth`` below it, walked at the DESCEND pace."""
    top, total = bench.TIP_RADIUS + PRESS_ABOVE, PRESS_ABOVE + depth
    n = math.ceil(total / PRESS_STEP_M)
    targets = [
        _target((*layout.TRAY_CENTER, top - total * i / n), GRASP_TILT_RAD) for i in range(n + 1)
    ]
    chain = ik.solve_path(targets, seed=_pose(ik, targets[0].position, GRASP_TILT_RAD, "press"))
    if chain is None:
        raise RuntimeError("descend_rows: no continuous IK path down to the press depth")
    waypoints = [c.q for c in chain]
    steps = leg_resolution(10**6, 1.0, distance=total)[1]
    grip = bench.grip_value()
    return [
        [*(float(v) for v in polyline_point(waypoints, min_jerk(k / steps))), grip]
        for k in range(steps + 1)
    ]


# --- metrics -----------------------------------------------------------------------------
def _at(trace, t):
    return int(np.argmin(np.abs(trace.t - t)))


def sag_mm(trace, tip_cmd, at=SAG_AT_S):
    return 1000.0 * float(np.linalg.norm(trace.tip[_at(trace, at)] - tip_cmd))


def sway_mm(trace, t_last, window=SWAY_S):
    sel = (trace.t >= t_last - 1e-12) & (trace.t <= t_last + window + 1e-12)
    tip = trace.tip[sel]
    return 1000.0 * float(np.linalg.norm(tip.max(axis=0) - tip.min(axis=0)))


def overshoot_mm(trace, tip_start, tip_final, t_from):
    u = np.asarray(tip_final) - np.asarray(tip_start)
    u = u / np.linalg.norm(u)
    sel = trace.t >= t_from - 1e-12
    return 1000.0 * max(0.0, float(np.max((trace.tip[sel] - tip_final) @ u)))


def contact(trace):
    touch = np.nonzero(trace.force > 0.0)[0]
    if len(touch) == 0:
        return None
    t_contact = float(trace.t[touch[0]])
    return {
        "t_contact": t_contact,
        "peak": float(trace.force.max()),
        "at_0_3": float(trace.force[_at(trace, t_contact + WALL_PROBE_S)]),
    }


def force_at(trace, t):
    return float(trace.force[_at(trace, t)])


def false_yield_mm(trace, ik):
    """The largest tip displacement the admittance offset puts on the reference."""
    with_offset = np.asarray(ik.fk_batch(trace.q_r)[0])
    without = np.asarray(ik.fk_batch(trace.q_r - trace.offset)[0])
    return 1000.0 * float(np.linalg.norm(with_offset - without, axis=1).max())


def growing(trace, window=OSC_WINDOW_S, floor=OSC_FLOOR):
    end = trace.t[-1]
    late = trace.dq[trace.t > end - window]
    early = trace.dq[(trace.t > end - 2 * window) & (trace.t <= end - window)]
    peak_late, peak_early = float(np.abs(late).max()), float(np.abs(early).max())
    return peak_late > peak_early and peak_late > floor


def gate_rows(results):
    """``results[model][rung][metric]`` -> one row per bar: value, reference, pass."""
    rows = []
    b1 = results["exact"]["B1"]
    for model, metric, kind, bar in BARS:
        value = results[model]["A"][metric]
        ref = b1.get(metric)
        if kind == "abs":
            ok = value <= bar
        elif kind == "rel":
            ok = value <= bar * ref
        elif kind == "band":
            ok = abs(value - ref) <= bar * abs(ref)
        else:
            ok = value is False
        rows.append(
            {
                "model": model,
                "metric": metric,
                "kind": kind,
                "bar": bar,
                "value": value,
                "reference": ref,
                "pass": bool(ok),
            }
        )
    return rows


# --- runs --------------------------------------------------------------------------------
def _go(rung, model, tau_th, exact, seed, plant, q0, pushes, duration):
    """One bench run; a fresh Tracking every time, because a Tracking is single-use."""
    tracking = None
    if rung in ("F", "A"):
        params = {"tau_th": list(tau_th), "contact": rung == "A"}
        tracking = maker_arm_rs.Tracking(model, params=params, profile=_profile.sim_profile())
    return bench.run(plant, rung, q0, pushes, duration, tracking=tracking, exact=exact, seed=seed)


def _plan(rows):
    return [(PLAN_T0, PLAN_T0, ROW_DT, rows)]


def _last(rows):
    return PLAN_T0 + (len(rows) - 1) * ROW_DT


def _hold(q):
    return [(0.0, 0.0, ROW_DT, [[*q, bench.grip_value()]])]


def calibrate(ik, model, seed=CAL_SEED, *, noise_seed=SEED) -> list[float]:
    """tau_th per joint for ``model``: rung A without yield on the calibration walk and a
    rest hold at the desk pose, ``max(1.25 max|r_j|, 3 sigma_r_j)``."""
    go = partial(_go, "A", model, [math.inf] * bench.ARM_DOF, None, noise_seed)
    plant = bench.build_plant()
    hover = _hover(ik)
    rows = walk_rows(ik, [hover, *random_targets(ik, seed, WALKS)])
    r_max = np.abs(go(plant, hover, _plan(rows), _last(rows) + CAL_TAIL_S).r).max(axis=0)
    rest = go(plant, list(DESK_POSE), _hold(DESK_POSE), REST_S)
    sigma = rest.r[rest.t >= REST_SKIP_S - 1e-12].std(axis=0)
    return [float(max(1.25 * a, 3.0 * s)) for a, s in zip(r_max, sigma)]


def _setup(ik, tau_th_exact, seed) -> dict:
    """Everything the scenarios share: plants, poses, tips and the plans."""
    free = bench.build_plant()
    p = poses(ik)
    tips = {name: bench.tip_of(free, q) for name, q in p.items()}
    transport = transport_rows(ik, p["hover"], p["release"])
    rng = random.Random(seed + 3)
    chunks = []
    for start in range(0, len(transport), CHUNK):
        t = PLAN_T0 + start * ROW_DT
        offset = [rng.gauss(0.0, CHUNK_SIGMA) for _ in range(bench.ARM_DOF)]
        rows = [[*np.add(r[:6], offset).tolist(), r[6]] for r in transport[start : start + CHUNK]]
        chunks.append((t, t, ROW_DT, rows))
    normal = tips["release"] - tips["hover"]
    normal[2] = 0.0
    normal /= np.linalg.norm(normal)
    center = bench.tip_of(free, transport[len(transport) // 2][:6])
    depth = press_depth(ik, tau_th_exact)
    walk = walk_rows(ik, [p["hover"], *random_targets(ik, GATE_SEED, WALKS)])
    return {
        "ik": ik,
        "free": free,
        "wall": bench.build_plant(wall=(center, normal)),
        "desk": bench.build_plant(desk=True),
        "poses": p,
        "tips": tips,
        "transport": transport,
        "chunks": chunks,
        "wall_center": center.tolist(),
        "wall_normal": normal.tolist(),
        "press_depth_m": depth,
        "press": descend_rows(ik, depth),
        "walk": walk,
    }


def scenario_1(go, s):
    sags, traces = {}, []
    for name, q in s["poses"].items():
        traces.append(go(s["free"], q, _hold(q), STATIC_S))
        sags[name] = sag_mm(traces[-1], s["tips"][name])
    return {"static_sag_mm": max(sags.values()), "static_sag_by_pose_mm": sags}, traces


def _transport_metrics(trace, s):
    t_last = _last(s["transport"])
    return {
        "sway_mm": sway_mm(trace, t_last),
        "overshoot_mm": overshoot_mm(trace, s["tips"]["hover"], s["tips"]["release"], PLAN_T0),
    }


def scenario_2(go, s):
    rows = s["transport"]
    trace = go(s["free"], s["poses"]["hover"], _plan(rows), _last(rows) + SETTLE_S)
    return _transport_metrics(trace, s), [trace]


def scenario_3(go, s):
    duration = _last(s["transport"]) + SETTLE_S
    trace = go(s["free"], s["poses"]["hover"], s["chunks"], duration)
    return {f"chunk_{k}": v for k, v in _transport_metrics(trace, s).items()}, [trace]


def scenario_4(go, s):
    rows = s["transport"]
    trace = go(s["wall"], s["poses"]["hover"], _plan(rows), _last(rows) + SETTLE_S)
    c = contact(trace)
    if c is None:
        raise RuntimeError("scenario 4: the tip never touched the wall, a bench defect")
    metrics = {"wall_t_contact": c["t_contact"], "wall_force_peak": c["peak"]}
    return {**metrics, "wall_force_03": c["at_0_3"]}, [trace]


def scenario_5(go, s):
    rows = s["press"]
    trace = go(s["desk"], rows[0][:6], _plan(rows), _last(rows) + PRESS_HOLD_S)
    return {"press_force": force_at(trace, _last(rows) + PRESS_PROBE_S)}, [trace]


def scenario_6(go, s):
    rows = s["walk"]
    trace = go(s["free"], rows[0][:6], _plan(rows), _last(rows) + SETTLE_S)
    return {"false_yield_mm": false_yield_mm(trace, s["ik"])}, [trace]


SCENARIOS = {
    "1": scenario_1,
    "2": scenario_2,
    "3": scenario_3,
    "4": scenario_4,
    "5": scenario_5,
    "6": scenario_6,
}


def _metrics(go, s) -> dict:
    out, grow = {}, {}
    for name, scenario in SCENARIOS.items():
        metrics, traces = scenario(go, s)
        out.update(metrics)
        grow[name] = any(growing(t) for t in traces)
    # The growing-oscillation bar reads every scenario, 3 included; only its sway and
    # overshoot are report rows.
    return {**out, "growing_by_scenario": grow, "growing": any(grow.values())}


def run_suite(seed=SEED, smoke=False) -> dict:
    exact = dm.dynamics(dm.urdf_bodies())
    if smoke:
        plant = bench.build_plant()
        tracking = maker_arm_rs.Tracking(exact, profile=_profile.sim_profile())
        trace = bench.run(
            plant, "A", DESK_POSE, _hold(DESK_POSE), SMOKE_S, tracking=tracking, seed=seed
        )
        sag = sag_mm(trace, bench.tip_of(plant, DESK_POSE), at=SMOKE_S)
        return {"smoke": {"sag_mm": sag}, "pass": True}
    ik = GlobalIK()
    models = {"exact": exact, "housing": dm.dynamics(dm.housing_bodies())}
    for f in MASS_SWEEP:
        models[f"exact_x{f}"] = dm.dynamics(dm.scaled(dm.urdf_bodies(), f))
    tau_th = {name: calibrate(ik, model, noise_seed=seed) for name, model in models.items()}
    s = _setup(ik, tau_th["exact"], seed)
    runs = [("exact", rung) for rung in bench.RUNGS]
    runs += [("housing", "F"), ("housing", "A")]
    runs += [(f"exact_x{f}", "A") for f in MASS_SWEEP]
    results: dict = {}
    for name, rung in runs:
        go = partial(_go, rung, models[name], tau_th[name], exact, seed)
        results.setdefault(name, {})[rung] = _metrics(go, s)
    rows = gate_rows(results)
    chunk_keys = ("chunk_sway_mm", "chunk_overshoot_mm")
    report = {
        "B0": results["exact"]["B0"],
        "F": {name: results[name]["F"] for name in ("exact", "housing")},
        "scenario_3": {
            f"{name}/{rung}": {k: m[k] for k in chunk_keys}
            for name, by_rung in results.items()
            for rung, m in by_rung.items()
        },
        "mass_sweep": {str(f): results[f"exact_x{f}"]["A"] for f in MASS_SWEEP},
        "housing_static_sag_mm": results["housing"]["A"]["static_sag_mm"],
    }
    return {
        "initial": {
            "bench": {
                "COULOMB_NM": bench.COULOMB_NM,
                "VISCOUS_NMS": bench.VISCOUS_NMS,
                "TIP_RADIUS": bench.TIP_RADIUS,
            },
            "bench_gate": {"CHUNK_SIGMA": CHUNK_SIGMA},
            "tracking": TRACKING_INITIAL,
        },
        "tau_th": tau_th,
        "press_depth_m": s["press_depth_m"],
        "setup": {
            "poses": s["poses"],
            "rows": {k: len(s[k]) for k in ("transport", "press", "walk")},
            "chunks": len(s["chunks"]),
            "wall": {"center": s["wall_center"], "normal": s["wall_normal"]},
        },
        "results": results,
        "report": report,
        "gate": rows,
        "pass": all(r["pass"] for r in rows),
    }
