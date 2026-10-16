"""The motion reads of one session log (MS5 control plan task 9): the jitter, sway, yield and
flex numbers the pre-registered adoption rules are judged on.

The input is the session log of ``real.telemetry.TelemetryLog`` (``load_log``) and, when the
overhead camera watched the gripper tag, the tag log of the sensing plan's
``scripts/real_tag_log.py``: one JSON object per frame with ``stamp_ns``,
``arrival_mono_ns``, ``rms_px``, ``cam_pos`` (the tag centre in the camera frame, m), ``pos``
(world from tag, m) and ``quat``; a frame without the tag has null poses and is skipped.

The windows (``segments``): a ``hold`` segment, a transport (a ``traj`` segment, the file's
rows only: an ``approach`` walk is neither a transport nor followed by a stop), the stop
(``STOP_WINDOW_S`` after a transport ends, cut where the next ``traj`` or ``approach``
starts) and a ``prompt`` (the arm holds while the operator acts, the push tests).

The motor side reads the loop's ``q`` and the controller's ``trk/`` rows, joined on
``t == trk/tick_t`` (never on the index: a fault-hold tick has a loop row and no tracking
row). Each motor read returns one value per arm joint (``ARM_DOF``, URDF joint order);
``tip_value`` pools the same statistic at the FK pick point (``real.arm.pick_point``), in mm:
the statistic of each base-frame axis, then their Euclidean norm (for an RMS or a standard
deviation that is the RMS or the spread of the tip's 3D displacement; for a peak to peak an
upper bound of the largest distance between two tip positions).

- ``tracking_rms``: RMS of ``q_r - q`` over a transport (rad; tip: FK(q_r) - FK(q)).
- ``stop_sway``: peak to peak of ``q`` over a stop after a least-squares line is removed.
- ``hold_jitter``: standard deviation of ``q`` over a hold after its first ``SETTLE_S``.
- ``free_residual_rms``: RMS of the observer's ``r`` over the free rows of the whole log
  (N m): tracked, ``gate == 1``, not clamped and outside every prompt, as ``ident.fit``
  reads free motion.
- ``yield_mm``: the largest tip shift of FK(q_r) against FK(q_d), ``q_d = q_r - offset``
  (the admittance offset), over a segment (mm).
- ``drag_resistance``: the largest ``|r|`` per joint over a segment (N m).

The tip side reads the tag on the SDK clock (``align``): a frame's host time is its
``arrival_mono_ns`` less the latency, mapped through the log's ``clock_pairs``.

- ``tip_sway``: the peak to peak of the tag position over a stop, line removed per axis,
  pooled as above (mm).
- ``flex_share``: over a window, the variance of the tag position that FK of the measured q
  (applied to the rig's ``tag_offset`` in the ``grasp_center`` frame, ``points``) does not
  explain: ``sum var(tag - fk) / sum var(tag)``, each series less its mean (a constant
  registration bias is not flex). 0 for a rigid arm; it can exceed 1 when the tag noise is
  larger than the motion. NaN when the window has fewer than ``MIN_TAG_FRAMES`` frames or the
  tag does not move.
- ``hold_pairs``: one (mean q, mean tag ``cam_pos``) pair per hold after its settle, the
  input of the sensing plan's ``register_base``.

Every window is half open, ``start_t <= t < end_t``: the row at a segment's end belongs to
the next one. Every read of an empty window is NaN (a joint vector of NaN), never an error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from maker_arm_lab.robots.frames import PICK_OFFSET_MEASURED

ARM_DOF = 6
STOP_WINDOW_S = 0.5
"""The stop window after each transport (MS5 control plan task 9)."""
SETTLE_S = 1.0
"""What a hold read skips after the hold starts (MS5 control plan task 9)."""
EDGE_S = 1.0
"""How far past the first or last clock pair ``align`` still maps a frame, at unit rate: two
drain periods. A frame further out is dropped."""
MIN_TAG_FRAMES = 3
MOTION_KINDS = ("traj", "approach")
"""The segments that move the arm: a stop window ends where the next one starts."""
TIP_METRICS = ("tracking_rms", "stop_sway", "hold_jitter")


@dataclass(frozen=True)
class Segment:
    kind: str  # "hold", "traj", "stop" or "prompt"
    start_t: float  # s, the SDK clock
    end_t: float
    label: str
    index: int  # the row in the log's ``segments`` (a stop: its transport's row)


@dataclass
class MotionLog:
    """One session log, joined: ``t`` (N,) and ``q`` (N, 6) of the loop, and the tracking
    columns ``q_r``, ``r``, ``offset`` (N, 6) and ``gate`` (N,) on the loop rows, NaN where a
    loop row has no tracking row (``tracked`` False)."""

    t: np.ndarray
    q: np.ndarray
    clamped: np.ndarray
    tracked: np.ndarray
    q_r: np.ndarray
    r: np.ndarray
    offset: np.ndarray
    gate: np.ndarray
    raw_segments: list[tuple[float, float, str, str]]
    clock_pairs: np.ndarray
    path: Path | None = None


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def load_log(path) -> MotionLog:
    """The session log at ``path``, the trk rows joined onto the loop rows on ``t``."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)
        q = np.asarray(f["q"][:, :ARM_DOF], dtype=np.float64)
        clamped = np.asarray(f["clamped"][:], dtype=bool)
        trk = f["trk"]
        tick_t = np.asarray(trk["tick_t"][:], dtype=np.float64) if "tick_t" in trk else None
        cols = {}
        for key, width in (("q_r", ARM_DOF), ("r", ARM_DOF), ("offset", ARM_DOF), ("gate", 0)):
            if tick_t is not None and key in trk:
                cols[key] = np.asarray(trk[key][:], dtype=np.float64)
            else:
                cols[key] = np.zeros((0, width) if width else (0,))
        raw = [
            (float(s["start_t"]), float(s["end_t"]), _decode(s["kind"]), _decode(s["label"]))
            for s in f["segments"][:]
        ]
        pairs = np.asarray(f["clock_pairs"][:], dtype=np.float64).reshape(-1, 2)
    if clamped.ndim > 1:
        clamped = clamped.any(axis=1)
    n = t.shape[0]
    tracked = np.zeros(n, dtype=bool)
    joined = {k: np.full((n, ARM_DOF) if k != "gate" else (n,), np.nan) for k in cols}
    if tick_t is not None and tick_t.shape[0] and n:
        order = np.argsort(tick_t, kind="stable")
        tick_sorted = tick_t[order]
        idx = np.minimum(np.searchsorted(tick_sorted, t), tick_sorted.shape[0] - 1)
        tracked = tick_sorted[idx] == t
        rows = order[idx[tracked]]
        for key, col in cols.items():
            joined[key][tracked] = col[rows]
    return MotionLog(
        t=t,
        q=q,
        clamped=clamped,
        tracked=tracked,
        q_r=joined["q_r"],
        r=joined["r"],
        offset=joined["offset"],
        gate=joined["gate"],
        raw_segments=raw,
        clock_pairs=pairs,
        path=path,
    )


def _as_log(log) -> MotionLog:
    return log if isinstance(log, MotionLog) else load_log(log)


def segments(log) -> list[Segment]:
    """The read windows of ``log`` in time order: its holds, transports and prompts, and a
    stop after every transport. A stop starting with a segment it shares a start with comes
    after it."""
    log = _as_log(log)
    raw = log.raw_segments
    out = [
        Segment(kind, start, end, label, i)
        for i, (start, end, kind, label) in enumerate(raw)
        if kind in ("hold", "traj", "prompt")
    ]
    stops = []
    for i, (_, end, kind, label) in enumerate(raw):
        if kind != "traj":
            continue
        stop_end = end + STOP_WINDOW_S
        for j, (start2, _, kind2, _) in enumerate(raw):
            if j != i and kind2 in MOTION_KINDS and end - 1e-9 <= start2 < stop_end:
                stop_end = max(end, start2)
        stops.append(Segment("stop", end, stop_end, label, i))
    return sorted(out + stops, key=lambda s: s.start_t)


def _window(log: MotionLog, start: float, end: float) -> np.ndarray:
    return (log.t >= start) & (log.t < end)


def _nan6() -> np.ndarray:
    return np.full(ARM_DOF, np.nan)


def _detrend(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``x`` (n, k) less its least-squares line in ``t``, per column."""
    if t.shape[0] < 2:
        return x - x.mean(axis=0)
    a = np.stack([t - t.mean(), np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return x - a @ coef


def _p2p(x: np.ndarray) -> np.ndarray:
    return x.max(axis=0) - x.min(axis=0)


# --- the motor side ---------------------------------------------------------------------


def _tracking_rows(log: MotionLog, seg: Segment):
    use = _window(log, seg.start_t, seg.end_t) & log.tracked
    return log.q_r[use], log.q[use]


def tracking_rms(log, seg: Segment) -> np.ndarray:
    """(6,) RMS of ``q_r - q`` over ``seg``'s tracked rows, rad."""
    log = _as_log(log)
    q_r, q = _tracking_rows(log, seg)
    if q.shape[0] == 0:
        return _nan6()
    return np.sqrt(np.mean((q_r - q) ** 2, axis=0))


def _stop_rows(log: MotionLog, seg: Segment):
    use = _window(log, seg.start_t, seg.end_t)
    return log.t[use], log.q[use]


def stop_sway(log, seg: Segment) -> np.ndarray:
    """(6,) peak to peak of ``q`` over ``seg`` after its least-squares line is removed, rad."""
    log = _as_log(log)
    t, q = _stop_rows(log, seg)
    if t.shape[0] < 2:
        return _nan6()
    return _p2p(_detrend(t, q))


def _hold_rows(log: MotionLog, seg: Segment, settle_s: float):
    use = _window(log, seg.start_t + settle_s, seg.end_t)
    return log.q[use]


def hold_jitter(log, seg: Segment, settle_s: float = SETTLE_S) -> np.ndarray:
    """(6,) standard deviation of ``q`` over ``seg`` after its first ``settle_s``, rad."""
    log = _as_log(log)
    q = _hold_rows(log, seg, settle_s)
    if q.shape[0] < 2:
        return _nan6()
    return q.std(axis=0)


def free_mask(log) -> np.ndarray:
    """(N,) the free-motion rows: tracked, ``gate == 1``, not clamped, outside every prompt
    (its end row included)."""
    log = _as_log(log)
    free = log.tracked & (log.gate == 1.0) & ~log.clamped
    for start, end, kind, _ in log.raw_segments:
        if kind == "prompt":
            free &= ~((log.t >= start) & (log.t <= end))  # closed, as ident.fit drops them
    return free


def free_residual_rms(log) -> np.ndarray:
    """(6,) RMS of the observer's ``r`` over the free rows of the whole log, N m."""
    log = _as_log(log)
    free = free_mask(log)
    if not free.any():
        return _nan6()
    return np.sqrt(np.mean(log.r[free] ** 2, axis=0))


def yield_offset(log, seg: Segment) -> np.ndarray:
    """(6,) the largest ``|offset|`` per joint over ``seg``'s tracked rows, rad."""
    log = _as_log(log)
    use = _window(log, seg.start_t, seg.end_t) & log.tracked
    if not use.any():
        return _nan6()
    return np.abs(log.offset[use]).max(axis=0)


def yield_mm(log, seg: Segment, ik) -> float:
    """The largest tip shift of FK(q_r) against FK(q_r - offset) over ``seg``, mm."""
    log = _as_log(log)
    use = _window(log, seg.start_t, seg.end_t) & log.tracked
    if not use.any():
        return math.nan
    q_r = log.q_r[use]
    shift = points(ik, q_r) - points(ik, q_r - log.offset[use])
    return float(1e3 * np.linalg.norm(shift, axis=1).max())


def drag_resistance(log, seg: Segment) -> np.ndarray:
    """(6,) the largest ``|r|`` per joint over ``seg``'s tracked rows, N m."""
    log = _as_log(log)
    use = _window(log, seg.start_t, seg.end_t) & log.tracked
    if not use.any():
        return _nan6()
    return np.abs(log.r[use]).max(axis=0)


# --- FK and the tip pool ----------------------------------------------------------------


def points(ik, q, offset: Sequence[float] = PICK_OFFSET_MEASURED) -> np.ndarray:
    """(N, 3) ``grasp_center`` plus ``offset`` in its frame at each row of ``q`` (N, 6), the
    arm's base frame, m: ``real.arm.pick_point`` for the default offset, the tag centre for
    the rig's ``tag_offset``."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, ARM_DOF)
    if q.shape[0] == 0:
        return np.zeros((0, 3))
    pos, rot = ik.fk_batch(torch.as_tensor(q))
    return pos.numpy() + rot.numpy() @ np.asarray(offset, dtype=np.float64)


def tip_value(metric: str, log, seg: Segment, ik, settle_s: float = SETTLE_S) -> float:
    """The motor read ``metric`` (one of ``TIP_METRICS``) pooled at the FK pick point, mm."""
    log = _as_log(log)
    if metric == "tracking_rms":
        q_r, q = _tracking_rows(log, seg)
        if q.shape[0] == 0:
            return math.nan
        d = points(ik, q_r) - points(ik, q)
        per_axis = np.sqrt(np.mean(d**2, axis=0))
    elif metric == "stop_sway":
        t, q = _stop_rows(log, seg)
        if t.shape[0] < 2:
            return math.nan
        per_axis = _p2p(_detrend(t, points(ik, q)))
    elif metric == "hold_jitter":
        q = _hold_rows(log, seg, settle_s)
        if q.shape[0] < 2:
            return math.nan
        per_axis = points(ik, q).std(axis=0)
    else:
        raise ValueError(f"no tip pool for {metric!r}; one of {TIP_METRICS}")
    return float(1e3 * np.linalg.norm(per_axis))


# --- the tag side -----------------------------------------------------------------------


def mono_to_sdk(log, mono_ns) -> np.ndarray:
    """Host ``time.monotonic_ns()`` stamps on the SDK clock through the log's
    ``clock_pairs``: linear interpolation between pairs; past the first or last pair the map
    runs on at unit rate from that pair for at most ``EDGE_S``, and NaN further out (and
    everywhere, for a log with no pair). A pair is stamped when the drain runs, after its last
    tick, so the map runs late by up to one drain's own latency (a constant bias of a few ms
    at most, left uncorrected)."""
    log = _as_log(log)
    mono = np.asarray(mono_ns, dtype=np.float64)
    pairs = log.clock_pairs
    if pairs.shape[0] == 0:
        return np.full(mono.shape, np.nan)
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    m, s = pairs[:, 0], pairs[:, 1]
    out = np.interp(mono, m, s)
    before, after = mono < m[0], mono > m[-1]
    out[before] = s[0] + (mono[before] - m[0]) * 1e-9
    out[after] = s[-1] + (mono[after] - m[-1]) * 1e-9
    far = (mono < m[0] - EDGE_S * 1e9) | (mono > m[-1] + EDGE_S * 1e9)
    out[far] = np.nan
    return out


def _tag_rows(tag_lines: Iterable[dict], log: MotionLog, latency_ms: float, key: str):
    lines = [ln for ln in tag_lines if ln.get(key) is not None]
    if not lines:
        return np.zeros(0), np.zeros((0, 3))
    mono = np.array([float(ln["arrival_mono_ns"]) for ln in lines]) - float(latency_ms) * 1e6
    t = mono_to_sdk(log, mono)
    pos = np.array([[float(v) for v in ln[key]] for ln in lines], dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(pos).all(axis=1)
    return t[keep], pos[keep]


def align(tag_lines: Iterable[dict], log, latency_ms: float, key: str = "pos") -> np.ndarray:
    """(M, 4) rows ``(t, x, y, z)``: each tag frame with a ``key`` pose (``pos``, world from
    tag, by default; ``cam_pos`` the camera frame) on the SDK clock, its host time
    ``arrival_mono_ns - latency_ms`` mapped by ``mono_to_sdk``, in time order. Frames with no
    pose, or past the map's range, are dropped."""
    log = _as_log(log)
    t, pos = _tag_rows(tag_lines, log, latency_ms, key)
    order = np.argsort(t, kind="stable")
    return np.concatenate([t[order, None], pos[order]], axis=1).reshape(-1, 4)


def hold_pairs(
    tag_lines: Iterable[dict], log, latency_ms: float, settle_s: float = SETTLE_S
) -> list[dict]:
    """One pair per hold of ``log`` that has loop rows and tag frames after its settle:
    ``{"q": mean q (6), "tag_cam": mean tag cam_pos (3), "start_t", "end_t", "label",
    "n_q", "n_tag"}``. A hold with no tag frame after its settle has no pair."""
    log = _as_log(log)
    tag = align(list(tag_lines), log, latency_ms, key="cam_pos")
    pairs = []
    for seg in segments(log):
        if seg.kind != "hold":
            continue
        start = seg.start_t + settle_s
        q = log.q[_window(log, start, seg.end_t)]
        in_tag = (tag[:, 0] >= start) & (tag[:, 0] < seg.end_t)
        if q.shape[0] == 0 or not in_tag.any():
            continue
        pairs.append(
            {
                "q": q.mean(axis=0).tolist(),
                "tag_cam": tag[in_tag, 1:].mean(axis=0).tolist(),
                "start_t": start,
                "end_t": seg.end_t,
                "label": seg.label,
                "n_q": int(q.shape[0]),
                "n_tag": int(in_tag.sum()),
            }
        )
    return pairs


def _tag_window(tag_pos: np.ndarray, seg: Segment) -> np.ndarray:
    tag_pos = np.asarray(tag_pos, dtype=np.float64).reshape(-1, 4)
    return tag_pos[(tag_pos[:, 0] >= seg.start_t) & (tag_pos[:, 0] < seg.end_t)]


def tip_sway(tag_pos: np.ndarray, seg: Segment) -> float:
    """The tag's peak to peak over ``seg`` (rows of ``align``), line removed per axis, the
    axes pooled by their norm, mm."""
    rows = _tag_window(tag_pos, seg)
    if rows.shape[0] < MIN_TAG_FRAMES:
        return math.nan
    return float(1e3 * np.linalg.norm(_p2p(_detrend(rows[:, 0], rows[:, 1:]))))


def flex_share(tag_pos: np.ndarray, fk_pos: np.ndarray, seg: Segment) -> float:
    """The share of the tag's variance over ``seg`` that FK does not explain. ``tag_pos`` is
    ``align``'s rows; ``fk_pos`` (N, 4) is ``(t, x, y, z)`` of ``points(ik, q, tag_offset)``
    at the loop rows, interpolated to each tag frame's time."""
    rows = _tag_window(tag_pos, seg)
    fk_pos = np.asarray(fk_pos, dtype=np.float64).reshape(-1, 4)
    if rows.shape[0] < MIN_TAG_FRAMES or fk_pos.shape[0] < 2:
        return math.nan
    inside = (rows[:, 0] >= fk_pos[0, 0]) & (rows[:, 0] <= fk_pos[-1, 0])
    rows = rows[inside]
    if rows.shape[0] < MIN_TAG_FRAMES:
        return math.nan
    fk = np.stack([np.interp(rows[:, 0], fk_pos[:, 0], fk_pos[:, 1 + k]) for k in range(3)], 1)
    tag = rows[:, 1:] - rows[:, 1:].mean(axis=0)
    resid = (rows[:, 1:] - fk) - (rows[:, 1:] - fk).mean(axis=0)
    total = float((tag**2).sum())
    if total <= 0.0:
        return math.nan
    return float((resid**2).sum() / total)


# --- the whole read ---------------------------------------------------------------------

SCHEMA = "maker-arm-lab/motion-read/1"
UNITS = {
    "tracking_rms": {"joints": "rad", "tip": "mm"},
    "stop_sway": {"joints": "rad", "tip": "mm"},
    "hold_jitter": {"joints": "rad", "tip": "mm"},
    "free_residual_rms": {"joints": "N m", "tip": None},
    "yield_mm": {"joints": "rad", "tip": "mm"},
    "drag_resistance": {"joints": "N m", "tip": None},
    "tip_sway": {"joints": None, "tip": "mm"},
    "flex_share": {"joints": None, "tip": "fraction"},
}
"""Each metric's two values: ``joints`` per arm joint, ``tip`` pooled at the tip (None: the
metric has no such value). ``yield_mm``'s joints are the largest ``|offset|``."""
POOLING = {
    "tracking_rms": "rms over every tracked transport row",
    "stop_sway": "mean over the stops",
    "hold_jitter": "mean over the holds",
    "free_residual_rms": "rms over the free rows of the log",
    "yield_mm": "max over the holds and prompts",
    "drag_resistance": "max over the holds and prompts",
    "tip_sway": "mean over the stops",
    "flex_share": "mean over the stops",
}
SEGMENT_READS = {
    "traj": ("tracking_rms",),
    "stop": ("stop_sway", "tip_sway", "flex_share"),
    "hold": ("hold_jitter", "yield_mm", "drag_resistance"),
    "prompt": ("yield_mm", "drag_resistance"),
}


def _num(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _vec(v) -> list | None:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64)
    if not np.isfinite(v).any():
        return None
    return [_num(x) for x in v]


def _value(joints=None, tip=None) -> dict:
    return {"joints": _vec(joints), "tip": None if tip is None else _num(tip)}


def read(
    log,
    ik,
    tag_lines: Sequence[dict] | None = None,
    latency_ms: float | None = None,
    tag_offset: Sequence[float] | None = None,
    world_from_cam: tuple[Sequence[float], np.ndarray] | None = None,
) -> dict:
    """The whole motion read of ``log``: the body of ``scripts/real_motion_read.py``'s
    output (schema ``SCHEMA``, documented in that script).

    ``tag_lines`` and ``latency_ms`` add the hold pairs, and with ``tag_offset`` the tip
    side. ``world_from_cam`` = (position (3,), rotation (3, 3)) puts the tag's ``cam_pos`` in
    the arm's base frame (the registered overhead pose); None reads the logged ``pos``."""
    log = _as_log(log)
    segs = segments(log)
    tag_pos = None
    tag_world = None
    reason = None
    if tag_lines is None:
        reason = "no tag log"
    elif tag_offset is None:
        reason = "the rig has no tag_offset"
    else:
        if world_from_cam is None:
            tag_pos, tag_world = align(tag_lines, log, latency_ms, key="pos"), "logged_pos"
        else:
            p, rot = world_from_cam
            cam = align(tag_lines, log, latency_ms, key="cam_pos")
            world = cam[:, 1:] @ np.asarray(rot, dtype=np.float64).T + np.asarray(p, float)
            tag_pos = np.concatenate([cam[:, :1], world], axis=1)
            tag_world = "rig_overhead"

    detail = []
    for seg in segs:
        reads: dict[str, dict] = {}
        rows = 0
        if seg.kind == "traj":
            rows = int((_window(log, seg.start_t, seg.end_t) & log.tracked).sum())
            reads["tracking_rms"] = _value(
                tracking_rms(log, seg), tip_value("tracking_rms", log, seg, ik)
            )
        elif seg.kind == "stop":
            reads["stop_sway"] = _value(stop_sway(log, seg), tip_value("stop_sway", log, seg, ik))
            if tag_pos is not None:
                use = _window(log, seg.start_t - 0.05, seg.end_t + 0.05)
                fk_pos = np.concatenate(
                    [log.t[use, None], points(ik, log.q[use], tag_offset)], axis=1
                )
                reads["tip_sway"] = _value(tip=tip_sway(tag_pos, seg))
                reads["flex_share"] = _value(tip=flex_share(tag_pos, fk_pos, seg))
        if seg.kind == "hold":
            reads["hold_jitter"] = _value(
                hold_jitter(log, seg), tip_value("hold_jitter", log, seg, ik)
            )
        if seg.kind in ("hold", "prompt"):
            reads["yield_mm"] = _value(yield_offset(log, seg), yield_mm(log, seg, ik))
            reads["drag_resistance"] = _value(drag_resistance(log, seg))
        detail.append(
            {
                "index": seg.index,
                "kind": seg.kind,
                "label": seg.label,
                "start_t": seg.start_t,
                "end_t": seg.end_t,
                "rows": rows,
                "reads": reads,
            }
        )

    def gather(name, part):
        vals = []
        for d in detail:
            v = d["reads"].get(name, {}).get(part)
            if v is not None:
                vals.append((d, np.asarray(v, dtype=np.float64)))
        return vals

    metrics: dict[str, dict] = {}
    for name in UNITS:
        out = {"joints": None, "tip": None, "count": 0}
        if name == "free_residual_rms":
            out.update(_value(free_residual_rms(log)), count=int(free_mask(log).sum()))
            out["tip"] = None
        else:
            for part in ("joints", "tip"):
                vals = gather(name, part)
                if not vals:
                    continue
                stack = np.stack([v for _, v in vals])
                if name == "tracking_rms":
                    w = np.array([d["rows"] for d, _ in vals], dtype=np.float64)
                    agg = np.sqrt(np.nansum(w[:, None] * stack.reshape(len(vals), -1) ** 2, 0)
                                  / w.sum())  # fmt: skip
                    agg = agg if part == "joints" else agg[0]
                elif name in ("yield_mm", "drag_resistance"):
                    agg = np.nanmax(stack, axis=0)
                else:
                    agg = np.nanmean(stack, axis=0)
                out[part] = _vec(agg) if part == "joints" else _num(agg)
                out["count"] = max(out["count"], len(vals))
        metrics[name] = out

    pairs = None
    if tag_lines is not None:
        pairs = hold_pairs(tag_lines, log, latency_ms)
    return {
        "schema": SCHEMA,
        "units": UNITS,
        "pooling": POOLING,
        "metrics": metrics,
        "segments": detail,
        "tip_side": {"available": tag_pos is not None, "reason": reason, "tag_world": tag_world},
        "tag_frames": None if tag_pos is None else int(tag_pos.shape[0]),
        "pairs_list": pairs,
    }
