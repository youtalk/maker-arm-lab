"""MS5 control plan task 6: the session log (one HDF5 file per powered session)."""

import json
import re

import h5py
import numpy as np
import pytest

from helpers import ROOT
from maker_arm_lab.real.telemetry import TelemetryLog, sdk_revision

PYPROJECT = ROOT / "pyproject.toml"


def _ticks(t0: float, n: int, dropped: int = 0) -> dict:
    t = [t0 + 0.005 * k for k in range(n)]
    seven = [[0.1 * j + k for j in range(7)] for k in range(n)]
    return {
        "t": t,
        "tick": list(range(n)),
        "clamped": [k % 2 == 0 for k in range(n)],
        "q": seven,
        "dq": seven,
        "tau": seven,
        "temp": seven,
        "fault_bits": [[0] * 7 for _ in range(n)],
        "feedback_age": seven,
        "cmd_pos": seven,
        "cmd_vel": seven,
        "cmd_kp": seven,
        "cmd_kd": seven,
        "cmd_tau": seven,
        "dropped": dropped,
    }


def _trk(t0: float, n: int, dropped: int = 0) -> dict:
    six = [[0.01 * j for j in range(6)] for _ in range(n)]
    return {
        "tick_t": [t0 + 0.005 * k for k in range(n)],
        "r": six,
        "gate": [1.0] * n,
        "offset": six,
        "q_r": six,
        "tau_ff": six,
        "dropped": dropped,
    }


def test_appends_grow_resizable_columns_and_tracking_lives_under_trk(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {"rig": {"rt": None}, "dynamics": "urdf"})
    log.append(_ticks(0.0, 3), _trk(0.0, 2), mono_ns=10)
    log.append(_ticks(0.015, 4), _trk(0.010, 5), mono_ns=20)
    log.close()
    with h5py.File(tmp_path / "s.h5") as f:
        assert f["t"].shape == (7,) and f["q"].shape == (7, 7) and f["cmd_tau"].shape == (7, 7)
        assert f["fault_bits"].dtype == np.uint16 and f["clamped"].dtype == np.bool_
        assert f["tick"].dtype == np.int64
        assert f["trk/tick_t"].shape == (7,) and f["trk/r"].shape == (7, 6)
        assert f["t"].maxshape == (None,) and f["trk/q_r"].maxshape == (None, 6)
        assert json.loads(f.attrs["rig"]) == {"rt": None} and f.attrs["dynamics"] == "urdf"
        assert "dropped" not in f and "dropped" not in f["trk"]
        pairs = f["clock_pairs"][:]
        assert pairs.dtype == np.float64 and pairs.shape == (2, 2)
        assert pairs[1].tolist() == [20.0, pytest.approx(0.015 + 0.005 * 3)]


def test_the_dropped_attrs_are_running_totals_of_both_drains(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    log.append(_ticks(0.0, 2, dropped=3), _trk(0.0, 2, dropped=1))
    log.append(_ticks(0.01, 2, dropped=4), None)
    assert log.dropped == 8
    log.close()
    with h5py.File(tmp_path / "s.h5") as f:
        assert f.attrs["dropped"] == 8
        assert f.attrs["dropped_loop"] == 7 and f.attrs["dropped_trk"] == 1


def test_an_empty_drain_writes_no_rows_and_no_clock_pair(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    empty = {k: [] for k in _ticks(0.0, 0) if k != "dropped"}
    log.append({**empty, "dropped": 0}, {"tick_t": [], "dropped": 0})
    log.append(None, None)
    log.append(_ticks(0.0, 2), None)
    assert log.ticks == 2
    log.close()
    with h5py.File(tmp_path / "s.h5") as f:
        assert f["t"].shape == (2,) and f["q"].shape == (2, 7)
        assert f["clock_pairs"].shape == (1, 2)


def test_segments_and_records_are_kept(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    log.add_segment(0.5, 1.5, "hold", "")
    log.add_segment(1.5, 3.0, "traj", "exc_00.npz")
    log.record("pose_note", "the tag is visible")
    log.close()
    with h5py.File(tmp_path / "s.h5") as f:
        seg = f["segments"][:]
        assert seg.shape == (2,)
        assert seg["start_t"].tolist() == [0.5, 1.5] and seg["end_t"].tolist() == [1.5, 3.0]
        assert [k.decode() for k in seg["kind"]] == ["hold", "traj"]
        assert seg["label"][1].decode() == "exc_00.npz"
        assert json.loads(f.attrs["records"]) == {"pose_note": "the tag is visible"}


def test_a_drain_with_ragged_columns_is_refused(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    bad = _ticks(0.0, 3)
    bad["q"] = bad["q"][:2]
    with pytest.raises(ValueError, match="q"):
        log.append(bad, None)
    log.close()


def test_a_log_never_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "s.h5"
    path.write_bytes(b"earlier session")
    with pytest.raises(FileExistsError):
        TelemetryLog(path, {})
    assert path.read_bytes() == b"earlier session"


def test_the_sdk_revision_is_the_pinned_commit():
    pin = re.search(r'maker-arm-rs = \{[^}]*rev = "([0-9a-f]{40})"', PYPROJECT.read_text())
    assert sdk_revision() == pin.group(1)


def test_an_append_onto_misaligned_columns_is_refused(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    log.append(_ticks(0.0, 3), _trk(0.0, 3))
    log._f["q"].resize(2, axis=0)  # as if an append had been cut between two columns
    with pytest.raises(ValueError, match="misaligned"):
        log.append(_ticks(0.015, 2), None)
    log.close()


def test_the_tick_intervals_are_read_across_drains(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    log.append(_ticks(0.0, 3), None)  # 0.000, 0.005, 0.010
    log.append(_ticks(0.020, 2), None)  # a 10 ms gap across the drains, then 5 ms
    log.append(_ticks(0.030, 1), None)  # 5 ms
    t = [0.035, 0.0425, 0.050]  # 7.5 ms (1.5 x the period: not late) and 7.5 ms
    log.append({**_ticks(0.0, 3), "t": t}, None)
    assert log.max_interval_s == pytest.approx(0.010)
    assert log.late_ticks == 1  # only the 10 ms interval is over 1.5 x 5 ms
    log.close()
    with h5py.File(tmp_path / "s.h5") as f:
        assert f.attrs["max_interval_s"] == pytest.approx(0.010)
        assert f.attrs["late_ticks"] == 1


def test_a_log_with_one_tick_has_no_interval(tmp_path):
    log = TelemetryLog(tmp_path / "s.h5", {})
    assert log.max_interval_s == 0.0 and log.late_ticks == 0
    log.append(_ticks(0.0, 1), None)
    assert log.max_interval_s == 0.0 and log.late_ticks == 0
    log.close()
