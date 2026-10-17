"""The A/B read of the two correction layers (MS5 control plan task 11): the pre-registered
adoption rule (MS5 revision section 8.6, ``docs/ms5-foundations.md`` section 6) applied to
three motion reads of the same fixed trajectory set.

    python3 scripts/real_ab_read.py --arm1 ~/data/maker-arm-ms5/s6_arm1_motion.json \\
        --arm2 ~/data/maker-arm-ms5/s6_arm2_motion.json \\
        --arm3 ~/data/maker-arm-ms5/s8_arm3_motion.json \\
        --targets ~/data/maker-arm-ms5/targets.json --out ~/data/maker-arm-ms5/ab.json

Arm 1 is the tracking controller with neither layer, arm 2 adds the observer compensation
(``dob``), arm 3 adds the learned residual (the MLP) to arm 2. Each ``--armN`` is a
``scripts/real_motion_read.py`` output (schema ``maker-arm-lab/motion-read/3``). The three must
read the same fixed trajectory set: the labels of their ``traj`` segments must match as
multisets, or the read refuses.

``--targets`` is a JSON object with exactly two keys::

    {"targets": {"hold_jitter": 0.05, "stop_sway": 0.3},
     "bars": {"yield_free_max_mm": 0.5, "yield_push_min_mm": 2.0}}

``targets`` is the jitter targets table of session 3 (MS5 revision section 8.5,
``docs/ms5-foundations.md`` section 5): metric -> a number in the unit section 5 gives it
(``TARGET_UNITS``). Every listed metric is lower-is-better, and a target is met when the arm's
target value is at most it. The target value is section 5's side of the metric
(``TARGET_POOL``): a motor metric reads the largest of its six ``joints`` values, so every
joint must meet it (``tracking_rms``, ``stop_sway``, ``hold_jitter``, ``hold_drift`` in
rad, ``free_residual_rms`` in N m), the contact drag the same (``drag_free``, ``drag_push``, N m,
the table's "largest resistance torque"), and a tip or contact metric reads its pooled
``tip`` (``tip_sway`` mm, ``flex_share`` fraction, ``yield_free_mm`` mm). At least one target
is needed; a metric outside ``TARGET_METRICS`` (``yield_push_mm`` is a bar, not a target)
refuses the file. ``bars`` holds the pre-registered yield bars of ``docs/ms5-foundations.md``
section 4, both required, nothing else allowed: the free motion yields at most
``yield_free_max_mm`` and a 5 N push at least ``yield_push_min_mm``.

The compared value of a metric, which the ``no_worse``, ``lower`` and bar checks and
``table`` read, is its pooled ``tip`` value when the metric has one (``TIP_METRICS``), else a
pool of its six ``joints`` values: the RMS over the joints for ``free_residual_rms``, the
largest for ``drag_free`` and ``drag_push``. A null tip, null joints or any null joint is a
missing value.

The rule, for arm 2 against arm 1 (layer ``dob``) and arm 3 against arm 2 (layer ``mlp``),
every check must pass:

- ``target``: per target, the arm's target value is at most it.
- ``free_yield``: the arm's ``yield_free_mm`` is at most ``yield_free_max_mm``. This is the
  free-motion bar of ``docs/ms5-foundations.md`` section 4, an extra criterion: section 6's
  list does not name it.
- ``no_worse``: the arm's ``tracking_rms`` and ``tip_sway`` are at most the arm before it's,
  with no tolerance.
- ``push_yield``: the arm's ``yield_push_mm`` is at least ``yield_push_min_mm`` (it must still
  hold with the compensation on). ``yield_push_mm`` is the motion read's max over the prompts,
  so the bar passes when any one prompt yields that far.
- ``lower`` (``mlp`` only): the arm's ``free_residual_rms`` is below arm 2's.

A missing value on either side fails its check with the reason ``missing``; it never passes
silently. Both layers are judged regardless of the other's verdict and both are reported:
``adopted.mlp`` is as computed even when ``dob`` was rejected (the owner reads it then). A
rejected layer is a result, recorded with the reasons, and still ends in ``AB_OK``. Section
6's timing bar (``update()`` p99 at most 100 us on the Thor) is not judged here: it is read
separately with ``scripts/track_timing.py``.

The output JSON (``--out``, never overwritten, schema ``maker-arm-lab/ab-read/1``):
``inputs`` (the four paths), ``targets``, ``bars``, ``table`` (metric -> ``{"arm1", "arm2",
"arm3"}`` compared values, null when missing), ``target_table`` (the same for the target
values of ``TARGET_METRICS``), ``layers`` (``dob`` and ``mlp``: ``{"arm",
"against", "adopted", "checks": [{"check", "metric", "value", "ref", "pass", "reason"}]}``,
``reason`` one of ``met``, ``failed``, ``missing``) and ``adopted`` (``{"dob", "mlp"}``).

It prints exactly one ``AB_OK {out, table, adopted}`` or ``AB_FAIL {reason, detail}`` line on
every exit path, argparse errors included, and exits through ``os._exit``. Stdlib only.
"""

import argparse
import json
import math
import os
import traceback
from pathlib import Path

IN_SCHEMA = "maker-arm-lab/motion-read/3"
OUT_SCHEMA = "maker-arm-lab/ab-read/1"
METRICS = (
    "tracking_rms",
    "stop_sway",
    "hold_jitter",
    "hold_drift",
    "free_residual_rms",
    "yield_free_mm",
    "yield_push_mm",
    "drag_free",
    "drag_push",
    "tip_sway",
    "flex_share",
)
"""The metrics of a motion read, a copy of ``real.motion.UNITS``' keys (a test pins it)."""
TIP_METRICS = (
    "tracking_rms",
    "stop_sway",
    "hold_jitter",
    "hold_drift",
    "yield_free_mm",
    "yield_push_mm",
    "tip_sway",
    "flex_share",
)
"""The metrics with a pooled tip value (``real.motion.UNITS``' non-null ``tip``, pinned)."""
JOINT_POOL = {"free_residual_rms": "rms", "drag_free": "max", "drag_push": "max"}
"""How a joints-only metric pools its six values into the compared one."""
TARGET_POOL = {
    "tracking_rms": "joints_max",
    "stop_sway": "joints_max",
    "hold_jitter": "joints_max",
    "hold_drift": "joints_max",
    "free_residual_rms": "joints_max",
    "yield_free_mm": "tip",
    "drag_free": "joints_max",
    "drag_push": "joints_max",
    "tip_sway": "tip",
    "flex_share": "tip",
}
"""The value a target compares against: the largest joint, or the pooled tip."""
TARGET_UNITS = {
    "tracking_rms": "rad",
    "stop_sway": "rad",
    "hold_jitter": "rad",
    "hold_drift": "rad",
    "free_residual_rms": "N m",
    "yield_free_mm": "mm",
    "drag_free": "N m",
    "drag_push": "N m",
    "tip_sway": "mm",
    "flex_share": "fraction",
}
"""The unit of a target, section 5's."""
TARGET_METRICS = tuple(TARGET_POOL)
"""The metrics a target may name: every one lower-is-better (all but ``yield_push_mm``)."""
BARS = ("yield_free_max_mm", "yield_push_min_mm")
NO_WORSE = ("tracking_rms", "tip_sway")
ARMS = ("arm1", "arm2", "arm3")
LAYERS = (("dob", "arm2", "arm1"), ("mlp", "arm3", "arm2"))


class Failure(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _token_fail(reason: str, detail: str) -> str:
    return "AB_FAIL " + json.dumps({"reason": reason, "detail": detail})


def _fail_usage(message: str) -> None:
    print(_token_fail("usage", message), flush=True)
    os._exit(2)


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _fail_usage(message)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--arm1", required=True, help="the motion read without either layer")
    parser.add_argument("--arm2", required=True, help="the motion read with dob")
    parser.add_argument("--arm3", required=True, help="the motion read with dob and the MLP")
    parser.add_argument("--targets", required=True, help="the targets and bars JSON")
    parser.add_argument("--out", required=True, help="the A/B read JSON to create")
    return parser


def _load(path: Path, what: str):
    if not path.is_file():
        raise Failure(what, f"no such file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise Failure(what, f"{path}: {err}") from err


def _finite(value) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_targets(path: Path) -> tuple[dict, dict]:
    data = _load(path, "targets")
    if not isinstance(data, dict) or set(data) != {"targets", "bars"}:
        keys = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise Failure("targets", f"{path} needs exactly the keys bars and targets, has {keys}")
    targets, bars = data["targets"], data["bars"]
    if not isinstance(targets, dict) or not targets:
        raise Failure("targets", "targets must be a non-empty object of metric -> number")
    unknown = sorted(set(targets) - set(TARGET_METRICS))
    if unknown:
        raise Failure("targets", f"unknown target metric {unknown}; one of {TARGET_METRICS}")
    if not isinstance(bars, dict):
        raise Failure("targets", "bars must be an object")
    missing = [b for b in BARS if b not in bars]
    if missing:
        raise Failure("targets", f"missing bar {missing}")
    extra = sorted(set(bars) - set(BARS))
    if extra:
        raise Failure("targets", f"unknown bar {extra}; the bars are {BARS}")
    for name, value in [*targets.items(), *bars.items()]:
        if not _finite(value):
            raise Failure("targets", f"{name} must be a finite number, got {value!r}")
    return {k: float(v) for k, v in targets.items()}, {k: float(bars[k]) for k in BARS}


def load_read(path: Path, arm: str) -> dict:
    data = _load(path, arm)
    if not isinstance(data, dict) or data.get("schema") != IN_SCHEMA:
        got = data.get("schema") if isinstance(data, dict) else type(data).__name__
        raise Failure(arm, f"{path} is not a {IN_SCHEMA} read (schema {got!r})")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or set(METRICS) - set(metrics):
        missing = sorted(set(METRICS) - set(metrics or {}))
        raise Failure(arm, f"{path}: the metrics {missing} are missing")
    if not isinstance(data.get("segments"), list):
        raise Failure(arm, f"{path} has no segments")
    return data


def trajectory_set(read: dict) -> list:
    return sorted(str(s.get("label")) for s in read["segments"] if s.get("kind") == "traj")


def value(read: dict, metric: str) -> float | None:
    """The compared value of ``metric`` (see the module docstring), None when missing."""
    entry = read["metrics"][metric]
    if metric in TIP_METRICS:
        tip = entry.get("tip")
        return float(tip) if _finite(tip) else None
    joints = entry.get("joints")
    if not isinstance(joints, list) or not joints or not all(_finite(v) for v in joints):
        return None
    if JOINT_POOL[metric] == "rms":
        return math.sqrt(sum(float(v) ** 2 for v in joints) / len(joints))
    return max(float(v) for v in joints)


def target_value(read: dict, metric: str) -> float | None:
    """The value a target of ``metric`` compares against (``TARGET_POOL``), None when
    missing."""
    entry = read["metrics"][metric]
    if TARGET_POOL[metric] == "tip":
        tip = entry.get("tip")
        return float(tip) if _finite(tip) else None
    joints = entry.get("joints")
    if not isinstance(joints, list) or not joints or not all(_finite(v) for v in joints):
        return None
    return max(float(v) for v in joints)


def _check(check: str, metric: str, val, ref, ok) -> dict:
    if val is None or ref is None:
        passed, reason = False, "missing"
    else:
        passed = bool(ok(val, ref))
        reason = "met" if passed else "failed"
    return {
        "check": check,
        "metric": metric,
        "value": val,
        "ref": ref,
        "pass": passed,
        "reason": reason,
    }


def judge(layer: str, arm: dict, before: dict, targets: dict, bars: dict) -> list[dict]:
    """The checks of one layer, ``arm`` against ``before``."""
    checks = [
        _check("target", m, target_value(arm, m), t, lambda v, r: v <= r)
        for m, t in targets.items()
    ]
    checks.append(
        _check(
            "free_yield",
            "yield_free_mm",
            value(arm, "yield_free_mm"),
            bars["yield_free_max_mm"],
            lambda v, r: v <= r,
        )
    )
    for m in NO_WORSE:
        checks.append(_check("no_worse", m, value(arm, m), value(before, m), lambda v, r: v <= r))
    checks.append(
        _check(
            "push_yield",
            "yield_push_mm",
            value(arm, "yield_push_mm"),
            bars["yield_push_min_mm"],
            lambda v, r: v >= r,
        )
    )
    if layer == "mlp":
        m = "free_residual_rms"
        checks.append(_check("lower", m, value(arm, m), value(before, m), lambda v, r: v < r))
    return checks


def run(args) -> dict:
    out = Path(args.out).expanduser()
    if out.exists():
        raise Failure("exists", f"{out} exists; a read is never overwritten")
    paths = {arm: Path(getattr(args, arm)).expanduser() for arm in ARMS}
    targets_path = Path(args.targets).expanduser()
    targets, bars = load_targets(targets_path)
    reads = {arm: load_read(p, arm) for arm, p in paths.items()}
    sets = {arm: trajectory_set(r) for arm, r in reads.items()}
    if not (sets["arm1"] == sets["arm2"] == sets["arm3"]):
        raise Failure("trajectory_set", f"the reads differ in their traj labels: {sets}")
    table = {m: {arm: value(reads[arm], m) for arm in ARMS} for m in METRICS}
    target_table = {m: {arm: target_value(reads[arm], m) for arm in ARMS} for m in TARGET_METRICS}
    layers, adopted = {}, {}
    for layer, arm, against in LAYERS:
        checks = judge(layer, reads[arm], reads[against], targets, bars)
        adopted[layer] = all(c["pass"] for c in checks)
        layers[layer] = {
            "arm": arm,
            "against": against,
            "adopted": adopted[layer],
            "checks": checks,
        }
    body = {
        "schema": OUT_SCHEMA,
        "inputs": {**{a: str(p) for a, p in paths.items()}, "targets": str(targets_path)},
        "targets": targets,
        "bars": bars,
        "table": table,
        "target_table": target_table,
        "layers": layers,
        "adopted": adopted,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(body, f, indent=2)
    return {"out": str(out), "table": table, "adopted": adopted}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token, code = "AB_OK " + json.dumps(run(args)), 0
    except Failure as err:
        token, code = _token_fail(err.reason, err.detail), 1
    except KeyboardInterrupt:
        token, code = _token_fail("interrupted", ""), 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        token, code = _token_fail("exception", f"{err}\n{traceback.format_exc()}"), 1
    print(token, flush=True)
    return code


if __name__ == "__main__":
    os._exit(main())
