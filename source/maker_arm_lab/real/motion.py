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
- ``hold_jitter``: standard deviation of ``q`` over a hold after its first ``SETTLE_S``. It
  removes no trend: a creep through the hold counts as jitter.
- ``hold_drift``: per joint, the absolute difference between the mean ``q`` over the last
  ``DRIFT_WINDOW_S`` of a hold and the mean over the first ``DRIFT_WINDOW_S`` after its
  ``SETTLE_S`` (rad; tip: the distance between the two mean FK pick points, mm), pooled as the
  largest over the holds: the bring-up's "a 60 s hold drifts under 2 mrad" rule reads it. A
  hold shorter than ``SETTLE_S`` plus two windows reads NaN.
- ``free_residual_rms``: RMS of the observer's ``r`` over the free rows of the whole log
  (N m): tracked, ``gate == 1``, not clamped and outside every prompt, as ``ident.fit``
  reads free motion.
- ``yield_mm``: the largest tip shift of FK(q_r) against FK(q_d), ``q_d = q_r - offset``
  (the admittance offset), over a segment (mm). ``read`` reads it on every segment and pools
  it twice: ``yield_free_mm`` over the transports, stops and holds (free motion, which must
  read 0 mm) and ``yield_push_mm`` over the prompts (the push tests, which must read > 0).
- ``drag_resistance``: the largest ``|r|`` per joint over a segment (N m), pooled the same
  two ways as ``drag_free`` and ``drag_push``.

The tip side reads the tag on the SDK clock (``align``): a frame's host time is its
``arrival_mono_ns`` less the latency, mapped through the log's ``clock_pairs``.

- ``tip_sway``: the peak to peak of the tag position over a stop, line removed per axis,
  pooled as above (mm).
- ``flex_share``: over a window, the variance of the tag position that FK of the measured q
  (applied to the rig's ``tag_offset`` in the ``grasp_center`` frame, ``points``) does not
  explain: ``sum var(tag - fk) / sum var(tag)``, each series less its mean (a constant
  registration bias is not flex). 0 for a rigid arm. A near-still tag sits at the camera's
  noise floor, where the ratio is noise over noise: a window whose tag variance is below
  ``FLEX_TAG_VAR_FLOOR_MM2`` reads NaN. ``read`` pools the stops as ``sum resid_var / sum
  tag_var`` (``flex_variances``), never as a mean of ratios. The read is sensitive to the
  latency: an error ``d`` in ``latency_ms`` shifts the tag against FK by ``d`` in time, so a
  sway at angular frequency ``w`` leaves a residual of about ``(w d)^2`` of its variance
  (3.5 Hz and 10 ms: about 5 %), read as flex.
- ``hold_pairs``: one (mean q, mean tag ``cam_pos``) pair per hold after its settle, the
  input of the sensing plan's ``register_base``.

Every window is half open, ``start_t <= t < end_t``: the row at a segment's end belongs to
the next one. Every read of an empty window is NaN (a joint vector of NaN), never an error.
Ticks the rings dropped leave gaps in ``tick``; ``window_stats`` reports a window's rows,
missing ticks and largest gap, and ``read`` nulls a stop whose largest gap exceeds
``MAX_GAP_S`` (its line fit and its FK interpolation would bridge the gap).
"""

from __future__ import annotations

import json
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
FLEX_TAG_VAR_FLOOR_MM2 = 0.1  # INITIAL: session 2 (the sensing tag read measures tag noise)
"""The tag variance (mm^2, summed over the axes) below which a window's flex share reads NaN:
about the still tag's noise variance."""
MAX_GAP_S = 0.02  # INITIAL: session 2
"""The largest gap between two loop rows (or a row and the window's edge) a stop read accepts:
four ticks at 200 Hz."""
DRIFT_WINDOW_S = 1.0
"""The two windows a hold drift read compares (the bring-up rule's 1 s means)."""
CONTROL_HZ = 200.0
MOTION_KINDS = ("traj", "approach")
"""The segments that move the arm: a stop window ends where the next one starts."""
TIP_METRICS = ("tracking_rms", "stop_sway", "hold_jitter", "hold_drift")


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
    tick: np.ndarray
    q: np.ndarray
    clamped: np.ndarray
    tracked: np.ndarray
    q_r: np.ndarray
    r: np.ndarray
    offset: np.ndarray
    gate: np.ndarray
    raw_segments: list[tuple[float, float, str, str]]
    clock_pairs: np.ndarray
    attrs: dict | None = None
    path: Path | None = None


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _attr_value(value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def load_log(path) -> MotionLog:
    """The session log at ``path``, the trk rows joined onto the loop rows on ``t``."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"][:], dtype=np.float64)
        tick = np.asarray(f["tick"][:], dtype=np.int64)
        attrs = {k: _attr_value(v) for k, v in f.attrs.items()}
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
        tick=tick,
        q=q,
        clamped=clamped,
        tracked=tracked,
        q_r=joined["q_r"],
        r=joined["r"],
        offset=joined["offset"],
        gate=joined["gate"],
        raw_segments=raw,
        clock_pairs=pairs,
        attrs=attrs,
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


def window_stats(log, seg: Segment) -> dict:
    """``rows`` (loop rows in ``seg``), ``tracked_rows``, ``missing_ticks`` (the tick span of
    those rows less the rows: ticks the rings dropped inside the window) and ``max_gap_s``,
    the largest time between two rows or between a row and the window's edges (the whole
    window when it has no row)."""
    log = _as_log(log)
    use = _window(log, seg.start_t, seg.end_t)
    t, tick = log.t[use], log.tick[use]
    rows = int(t.shape[0])
    if rows == 0:
        return {
            "rows": 0,
            "tracked_rows": 0,
            "missing_ticks": 0,
            "max_gap_s": float(seg.end_t - seg.start_t),
        }
    edges = np.concatenate([[seg.start_t], t, [seg.end_t]])
    gaps = np.diff(edges)
    # a row is due every tick: the gap after the last row is one tick short of the edge
    gaps[-1] = max(0.0, gaps[-1] - 1.0 / CONTROL_HZ)
    return {
        "rows": rows,
        "tracked_rows": int(log.tracked[use].sum()),
        "missing_ticks": int(tick.max() - tick.min() + 1 - rows),
        "max_gap_s": float(gaps.max()),
    }


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


def _drift_rows(log: MotionLog, seg: Segment, settle_s: float, window_s: float):
    start = seg.start_t + settle_s
    if seg.end_t - start < 2.0 * window_s:
        return None, None
    first = log.q[_window(log, start, start + window_s)]
    last = log.q[_window(log, seg.end_t - window_s, seg.end_t)]
    if first.shape[0] == 0 or last.shape[0] == 0:
        return None, None
    return first, last


def hold_drift(
    log, seg: Segment, settle_s: float = SETTLE_S, window_s: float = DRIFT_WINDOW_S
) -> np.ndarray:
    """(6,) ``|mean q over the last window_s - mean q over the first window_s after the
    settle|`` of hold ``seg``, rad; NaN for a hold too short for two windows."""
    log = _as_log(log)
    first, last = _drift_rows(log, seg, settle_s, window_s)
    if first is None:
        return _nan6()
    return np.abs(last.mean(axis=0) - first.mean(axis=0))


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
    elif metric == "hold_drift":
        first, last = _drift_rows(log, seg, settle_s, DRIFT_WINDOW_S)
        if first is None:
            return math.nan
        per_axis = np.abs(points(ik, last).mean(axis=0) - points(ik, first).mean(axis=0))
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


def flex_variances(tag_pos: np.ndarray, fk_pos: np.ndarray, seg: Segment) -> tuple:
    """``(resid_var_mm2, tag_var_mm2)`` over ``seg``: the variance of ``tag - fk`` and of the
    tag, each summed over the axes, in mm^2; NaN for a window with too few frames.
    ``tag_pos`` is ``align``'s rows; ``fk_pos`` (N, 4) is ``(t, x, y, z)`` of ``points(ik, q,
    tag_offset)`` at the loop rows, interpolated to each tag frame's time."""
    rows = _tag_window(tag_pos, seg)
    fk_pos = np.asarray(fk_pos, dtype=np.float64).reshape(-1, 4)
    if rows.shape[0] < MIN_TAG_FRAMES or fk_pos.shape[0] < 2:
        return math.nan, math.nan
    inside = (rows[:, 0] >= fk_pos[0, 0]) & (rows[:, 0] <= fk_pos[-1, 0])
    rows = rows[inside]
    if rows.shape[0] < MIN_TAG_FRAMES:
        return math.nan, math.nan
    fk = np.stack([np.interp(rows[:, 0], fk_pos[:, 0], fk_pos[:, 1 + k]) for k in range(3)], 1)
    resid = rows[:, 1:] - fk
    tag_var = float(1e6 * rows[:, 1:].var(axis=0).sum())
    resid_var = float(1e6 * resid.var(axis=0).sum())
    return resid_var, tag_var


def flex_share(
    tag_pos: np.ndarray,
    fk_pos: np.ndarray,
    seg: Segment,
    floor_mm2: float = FLEX_TAG_VAR_FLOOR_MM2,
) -> float:
    """The share of the tag's variance over ``seg`` that FK does not explain
    (``flex_variances``); NaN when the tag variance is below ``floor_mm2``."""
    resid_var, tag_var = flex_variances(tag_pos, fk_pos, seg)
    if not (math.isfinite(tag_var) and tag_var >= floor_mm2 and tag_var > 0.0):
        return math.nan
    return resid_var / tag_var


# --- the whole read ---------------------------------------------------------------------

SCHEMA = "maker-arm-lab/motion-read/3"
"""/3 (2026-10-16) adds ``hold_drift``."""
UNITS = {
    "tracking_rms": {"joints": "rad", "tip": "mm"},
    "stop_sway": {"joints": "rad", "tip": "mm"},
    "hold_jitter": {"joints": "rad", "tip": "mm"},
    "hold_drift": {"joints": "rad", "tip": "mm"},
    "free_residual_rms": {"joints": "N m", "tip": None},
    "yield_free_mm": {"joints": "rad", "tip": "mm"},
    "yield_push_mm": {"joints": "rad", "tip": "mm"},
    "drag_free": {"joints": "N m", "tip": None},
    "drag_push": {"joints": "N m", "tip": None},
    "tip_sway": {"joints": None, "tip": "mm"},
    "flex_share": {"joints": None, "tip": "fraction"},
}
"""Each metric's two values: ``joints`` per arm joint, ``tip`` pooled at the tip (None: the
metric has no such value). The yields' joints are the largest ``|offset|``."""
POOLING = {
    "tracking_rms": "rms over every tracked transport row",
    "stop_sway": "mean over the stops",
    "hold_jitter": "mean over the holds",
    "hold_drift": "max over the holds",
    "free_residual_rms": "rms over the free rows of the log",
    "yield_free_mm": "max over the transports, stops and holds",
    "yield_push_mm": "max over the prompts",
    "drag_free": "max over the transports, stops and holds",
    "drag_push": "max over the prompts",
    "tip_sway": "mean over the stops",
    "flex_share": "sum of the residual variances over the sum of the tag variances, over the "
    "stops at or above the tag variance floor",
}
POOLED_FROM = {
    "tracking_rms": ("tracking_rms", ("traj",)),
    "stop_sway": ("stop_sway", ("stop",)),
    "hold_jitter": ("hold_jitter", ("hold",)),
    "hold_drift": ("hold_drift", ("hold",)),
    "yield_free_mm": ("yield_mm", ("traj", "stop", "hold")),
    "yield_push_mm": ("yield_mm", ("prompt",)),
    "drag_free": ("drag_resistance", ("traj", "stop", "hold")),
    "drag_push": ("drag_resistance", ("prompt",)),
    "tip_sway": ("tip_sway", ("stop",)),
}
"""Pooled metric -> (the per-segment read, the segment kinds it pools)."""
STOP_READS = ("stop_sway", "tip_sway", "flex_share")
"""The reads a stop with a gap past ``MAX_GAP_S`` nulls."""


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


def _pool(name: str, stack: np.ndarray, weights: np.ndarray):
    if name == "tracking_rms":
        return np.sqrt(np.nansum(weights[:, None] * stack**2, axis=0) / weights.sum())
    if name in ("hold_drift", "yield_free_mm", "yield_push_mm", "drag_free", "drag_push"):
        return np.nanmax(stack, axis=0)
    return np.nanmean(stack, axis=0)


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
    the arm's base frame (the registered overhead pose); None reads the logged ``pos``.
    ValueError for ``tag_lines`` without a finite ``latency_ms``."""
    log = _as_log(log)
    if tag_lines is not None:
        if latency_ms is None or not math.isfinite(float(latency_ms)):
            raise ValueError("a tag log needs latency_ms (the sensing ledger's stamp p50)")
        tag_lines = list(tag_lines)
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
        stats = window_stats(log, seg)
        entry = {
            "index": seg.index,
            "kind": seg.kind,
            "label": seg.label,
            "start_t": seg.start_t,
            "end_t": seg.end_t,
            **stats,
            "gap_excluded": False,
            "reads": reads,
        }
        if seg.kind == "traj":
            reads["tracking_rms"] = _value(
                tracking_rms(log, seg), tip_value("tracking_rms", log, seg, ik)
            )
        elif seg.kind == "stop":
            if stats["max_gap_s"] > MAX_GAP_S:
                entry["gap_excluded"] = True
                for name in STOP_READS:
                    reads[name] = _value()
            else:
                reads["stop_sway"] = _value(
                    stop_sway(log, seg), tip_value("stop_sway", log, seg, ik)
                )
                if tag_pos is not None:
                    use = _window(log, seg.start_t - 0.05, seg.end_t + 0.05)
                    fk_pos = np.concatenate(
                        [log.t[use, None], points(ik, log.q[use], tag_offset)], axis=1
                    )
                    resid_var, tag_var = flex_variances(tag_pos, fk_pos, seg)
                    entry["flex"] = {
                        "resid_var_mm2": _num(resid_var),
                        "tag_var_mm2": _num(tag_var),
                    }
                    reads["tip_sway"] = _value(tip=tip_sway(tag_pos, seg))
                    reads["flex_share"] = _value(tip=flex_share(tag_pos, fk_pos, seg))
        elif seg.kind == "hold":
            reads["hold_jitter"] = _value(
                hold_jitter(log, seg), tip_value("hold_jitter", log, seg, ik)
            )
            reads["hold_drift"] = _value(
                hold_drift(log, seg), tip_value("hold_drift", log, seg, ik)
            )
        reads["yield_mm"] = _value(yield_offset(log, seg), yield_mm(log, seg, ik))
        reads["drag_resistance"] = _value(drag_resistance(log, seg))
        detail.append(entry)

    metrics: dict[str, dict] = {}
    for name in UNITS:
        out = {"joints": None, "tip": None, "count": 0, "excluded": 0}
        if name == "free_residual_rms":
            out["joints"] = _vec(free_residual_rms(log))
            out["count"] = int(free_mask(log).sum())
        elif name == "flex_share":
            used = [d for d in detail if d["kind"] == "stop" and "flex" in d]
            ok = [
                d for d in used
                if d["flex"]["tag_var_mm2"] is not None
                and d["flex"]["tag_var_mm2"] >= FLEX_TAG_VAR_FLOOR_MM2
                and d["flex"]["tag_var_mm2"] > 0.0
                and d["flex"]["resid_var_mm2"] is not None
            ]  # fmt: skip
            if ok:
                resid = sum(d["flex"]["resid_var_mm2"] for d in ok)
                total = sum(d["flex"]["tag_var_mm2"] for d in ok)
                out["tip"] = _num(resid / total)
            out["count"] = len(ok)
            out["excluded"] = sum(
                1 for d in detail if d["kind"] == "stop" and d["gap_excluded"]
            ) + (len(used) - len(ok))
        else:
            source, kinds = POOLED_FROM[name]
            pool = [d for d in detail if d["kind"] in kinds]
            out["excluded"] = sum(1 for d in pool if d["gap_excluded"])
            for part in ("joints", "tip"):
                vals = [
                    (d, np.asarray(d["reads"][source][part], dtype=np.float64).reshape(-1))
                    for d in pool
                    if source in d["reads"] and d["reads"][source][part] is not None
                ]
                if not vals:
                    continue
                stack = np.stack([v for _, v in vals])
                weights = np.array([d["tracked_rows"] for d, _ in vals], dtype=np.float64)
                agg = _pool(name, stack, weights)
                out[part] = _vec(agg) if part == "joints" else _num(agg[0])
                out["count"] = max(out["count"], len(vals))
        metrics[name] = out

    pairs = None
    if tag_lines is not None:
        pairs = hold_pairs(tag_lines, log, latency_ms)
    return {
        "schema": SCHEMA,
        "units": UNITS,
        "pooling": POOLING,
        "limits": {
            "max_gap_s": MAX_GAP_S,
            "flex_tag_var_floor_mm2": FLEX_TAG_VAR_FLOOR_MM2,
            "stop_window_s": STOP_WINDOW_S,
            "settle_s": SETTLE_S,
            "drift_window_s": DRIFT_WINDOW_S,
        },
        "metrics": metrics,
        "segments": detail,
        "tip_side": {"available": tag_pos is not None, "reason": reason, "tag_world": tag_world},
        "tag_frames": None if tag_pos is None else int(tag_pos.shape[0]),
        "log_meta": dict(log.attrs or {}),
        "pairs_list": pairs,
    }
