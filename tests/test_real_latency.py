"""MS5 design P7: the stamp and photon arithmetic."""

import math

import pytest

from maker_arm_lab.real import latency
from maker_arm_lab.real.frames import Arrival, relay


def _arrivals(n, period_ms=33.3, latency_ms=80.0, skip=()):
    out = []
    for k in range(n):
        if k in skip:
            continue
        mono = int(k * period_ms * 1e6)
        stamp = 1_000_000_000_000 + mono
        out.append(Arrival(stamp, stamp + int(latency_ms * 1e6), mono))
    return out


def test_an_arrival_reads_its_latency_in_ms():
    assert Arrival(1_000_000, 81_000_000, 0).latency_ms == 80.0


def test_percentile_interpolates_between_order_statistics_and_is_nan_when_empty():
    assert latency.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert latency.percentile([5.0], 95) == 5.0
    assert math.isnan(latency.percentile([], 50))


def test_summary_rounds_and_reports_none_when_empty():
    assert latency.summary([1.0, 2.0, 3.0]) == {
        "n": 3,
        "p50": 2.0,
        "p95": 2.9,
        "max": 3.0,
        "mean": 2.0,
    }
    assert latency.summary([])["p50"] is None


def test_stamp_summary_reads_the_latency_the_interval_and_the_dropped_frames():
    out = latency.stamp_summary(_arrivals(30, skip={10, 20, 21}))
    assert out["latency_ms"]["p50"] == pytest.approx(80.0, abs=0.01)
    assert out["interval_ms"]["p50"] == pytest.approx(33.3, abs=0.01)
    assert out["dropped"] == 3


def test_transitions_mark_each_crossing_after_the_first_sample():
    samples = [(0, 10.0), (1, 12.0), (2, 200.0), (3, 210.0), (4, 5.0)]
    assert latency.transitions(samples, 100.0) == [(2, True), (4, False)]


def test_pair_switches_takes_the_first_matching_transition_before_the_next_switch():
    switches = [(100, True), (1000, False), (2000, True)]
    seen = [(150, True), (160, False), (180, True), (1300, False)]
    assert latency.pair_switches(switches, seen, max_ns=5000) == [50, 300]


def test_photon_summary_takes_half_a_frame_off_the_mean():
    out = latency.photon_summary([100_000_000, 120_000_000], frame_period_ms=33.3)
    assert out["photon_ms"]["mean"] == 110.0
    assert out["capture_to_arrival_ms"] == pytest.approx(93.35, abs=0.01)
    assert latency.photon_summary([], 33.3)["capture_to_arrival_ms"] is None


def test_the_clock_offset_is_the_stamp_mean_less_the_photon_latency():
    assert latency.clock_offset_ms(130.0, 93.35) == 36.65


def test_a_message_that_raises_is_dropped_and_reported_and_the_next_one_still_arrives():
    """MS5 final review M3: a corrupt Conduit message must not end the executor's thread."""
    got, errors = [], []

    def convert(msg):
        if msg == "corrupt":
            raise ValueError("not a JPEG")
        return msg.upper()

    def callback(frame):
        if frame == "REFUSED":
            raise RuntimeError("the source refused it")
        got.append(frame)

    call = relay("jpeg", convert, callback, lambda key, err: errors.append((key, str(err))))
    for msg in ("a", "corrupt", "refused", "b"):
        call(msg)
    assert got == ["A", "B"]
    assert errors == [("jpeg", "not a JPEG"), ("jpeg", "the source refused it")]
