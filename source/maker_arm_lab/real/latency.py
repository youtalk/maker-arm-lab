"""MS5 design P7: latency and clock arithmetic for the Conduit stream. Stdlib only.

Two methods. The stamp method logs, per frame, the Thor's arrival time minus the header stamp:
the true capture-to-arrival latency plus the offset between the Thor's clock and the iPhone's.
The photon method needs no clock sync: the Thor switches a light in the iPhone's view and
times the first arriving frame that shows the new state. The switch lands anywhere inside a
frame's exposure window, so a photon delay overstates the capture-to-arrival latency by a
uniform draw over one frame period, and ``photon_summary`` takes half a period off the mean.
The session's clock offset is the stamp method's mean less the photon method's
capture-to-arrival latency, read on the same frames.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import pairwise

from maker_arm_lab.real.frames import Arrival

DROP_FACTOR = 1.5
"""An arrival interval longer than this many median intervals holds lost frames."""


def percentile(values: Iterable[float], q: float) -> float:
    """The ``q``-th percentile (0 to 100), linear between order statistics; nan when empty."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(values: Iterable[float]) -> dict:
    """``n``, p50, p95, max and mean, rounded to 0.01; None for each with no value."""
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(xs),
        "p50": round(percentile(xs, 50), 2),
        "p95": round(percentile(xs, 95), 2),
        "max": round(max(xs), 2),
        "mean": round(sum(xs) / len(xs), 2),
    }


def stamp_summary(arrivals: Sequence[Arrival]) -> dict:
    """One topic's stamp read: arrival minus stamp (ms), the arrival interval (ms, monotonic)
    and the frames lost, counted from the intervals longer than ``DROP_FACTOR`` medians."""
    ordered = sorted(arrivals, key=lambda a: a.arrival_mono_ns)
    intervals = [(b.arrival_mono_ns - a.arrival_mono_ns) / 1e6 for a, b in pairwise(ordered)]
    median = percentile(intervals, 50)
    dropped = 0
    if intervals and median > 0:
        dropped = sum(round(x / median) - 1 for x in intervals if x > DROP_FACTOR * median)
    return {
        "latency_ms": summary(a.latency_ms for a in ordered),
        "interval_ms": summary(intervals),
        "dropped": dropped,
    }


def transitions(samples: Sequence[tuple[int, float]], threshold: float) -> list[tuple[int, bool]]:
    """``(time_ns, on)`` at every sample on the other side of ``threshold`` from the sample
    before it. The first sample sets the state and is no transition."""
    out: list[tuple[int, bool]] = []
    state = None
    for t, value in samples:
        on = value > threshold
        if state is not None and on != state:
            out.append((t, on))
        state = on
    return out


def pair_switches(
    switches: Sequence[tuple[int, bool]], seen: Sequence[tuple[int, bool]], max_ns: int
) -> list[int]:
    """The delay (ns) from each switch ``(time_ns, on)`` to the first seen transition to the
    same state at or after it, before the next switch and within ``max_ns``. A switch the
    camera never shows is left out."""
    delays = []
    for k, (t, on) in enumerate(switches):
        end = t + max_ns
        if k + 1 < len(switches):
            end = min(end, switches[k + 1][0])
        hit = next((s for s, state in seen if t <= s < end and state == on), None)
        if hit is not None:
            delays.append(hit - t)
    return delays


def photon_summary(delays_ns: Sequence[int], frame_period_ms: float) -> dict:
    """The photon read: the delays (ms) and the capture-to-arrival latency, their mean less
    half a frame period (None with no delay)."""
    ms = [d / 1e6 for d in delays_ns]
    c2a = round(sum(ms) / len(ms) - frame_period_ms / 2.0, 2) if ms else None
    return {
        "photon_ms": summary(ms),
        "capture_to_arrival_ms": c2a,
        "frame_period_ms": round(frame_period_ms, 2),
    }


def clock_offset_ms(stamp_mean_ms: float, capture_to_arrival_ms: float) -> float:
    """The Thor's clock minus the iPhone's, ms."""
    return round(stamp_mean_ms - capture_to_arrival_ms, 2)
