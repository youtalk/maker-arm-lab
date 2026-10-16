"""The session log: every 200 Hz tick of one powered session in one HDF5 file (MS5 control
plan task 6).

The loop's columns of ``Arm.drain`` sit at the file's root (``t``, ``tick``, ``clamped``, and
seven wide ``q``, ``dq``, ``tau``, ``temp``, ``fault_bits``, ``feedback_age``, ``cmd_pos``,
``cmd_vel``, ``cmd_kp``, ``cmd_kd``, ``cmd_tau``), the controller's columns of
``Tracking.drain`` under ``trk/`` (``tick_t``, ``gate``, and six wide ``r``, ``offset``,
``q_r``, ``tau_ff``). Every column is a resizable dataset that grows by one drain at a time,
named as the SDK names it, so a column a later SDK adds is logged with no change here. The two
groups do not share an index: join them on ``t`` == ``trk/tick_t`` (a fault-hold tick has a
loop row and no tracking row).

``segments`` holds one (start_t, end_t, kind, label) row per plan segment, on the SDK clock.
``clock_pairs`` (N, 2) float64 holds one (``time.monotonic_ns()`` at the drain, the SDK ``t`` of
that drain's last loop record) pair per drain, to put host-stamped data (camera frames) on the
SDK clock. The attrs carry what the session was run with (JSON for anything but a scalar or a
string), the operator's ``records``, and the running totals ``dropped`` = ``dropped_loop`` +
``dropped_trk`` of ticks the 4000-tick rings lost between drains. The file is flushed after
every write, so a session that dies keeps every drain before it.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Self

import h5py
import numpy as np

TRACKING_GROUP = "trk"
DTYPES = {"tick": np.int64, "clamped": np.bool_, "fault_bits": np.uint16}
"""Columns that are not float64."""
SEGMENT_DTYPE = np.dtype(
    [
        ("start_t", np.float64),
        ("end_t", np.float64),
        ("kind", h5py.string_dtype()),
        ("label", h5py.string_dtype()),
    ]
)


def sdk_revision() -> str:
    """The git commit ``maker_arm_rs`` was built from (its install record), or "unknown"."""
    try:
        raw = importlib.metadata.distribution("maker-arm-rs").read_text("direct_url.json")
        return json.loads(raw or "{}").get("vcs_info", {}).get("commit_id", "unknown")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _attr(value):
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        return value
    return json.dumps(value)


class TelemetryLog:
    """One session's log at ``path``, which must not exist yet: a log is never overwritten."""

    def __init__(self, path: Path, attrs: dict):
        self.path = Path(path)
        self._f = h5py.File(self.path, "x")
        for name, value in attrs.items():
            self._f.attrs[name] = _attr(value)
        self._records: dict[str, str] = {}
        self._columns: dict[str, set[str]] = {}
        self.dropped_loop = 0
        self.dropped_trk = 0
        self.ticks = 0
        self._f.create_group(TRACKING_GROUP)
        self._f.create_dataset("segments", (0,), dtype=SEGMENT_DTYPE, maxshape=(None,))
        self._f.create_dataset("clock_pairs", (0, 2), dtype=np.float64, maxshape=(None, 2))
        self._write_totals()
        self._f.attrs["records"] = json.dumps(self._records)
        self._f.flush()

    @property
    def dropped(self) -> int:
        return self.dropped_loop + self.dropped_trk

    def _write_totals(self) -> None:
        self._f.attrs["dropped"] = self.dropped
        self._f.attrs["dropped_loop"] = self.dropped_loop
        self._f.attrs["dropped_trk"] = self.dropped_trk

    def _append_rows(self, group: h5py.Group, columns: dict) -> int:
        """Append one drain's columns to ``group``; returns the row count."""
        arrays = {
            k: np.asarray(v, dtype=DTYPES.get(k, np.float64))
            for k, v in columns.items()
            if k != "dropped"
        }
        if not arrays:
            return 0
        lengths = {k: a.shape[0] for k, a in arrays.items()}
        n = max(lengths.values())
        short = sorted(k for k, m in lengths.items() if m != n)
        if short:
            raise ValueError(f"{group.name}: columns {short} are not {n} rows long")
        if n == 0:
            return 0
        known = self._columns.setdefault(group.name, set())
        if known and set(arrays) != known:
            raise ValueError(
                f"{group.name}: columns changed from {sorted(known)} to {sorted(arrays)}"
            )
        for key, a in arrays.items():
            if key not in group:
                group.create_dataset(
                    key, (0, *a.shape[1:]), dtype=a.dtype, maxshape=(None, *a.shape[1:])
                )
            d = group[key]
            if d.shape[1:] != a.shape[1:]:
                raise ValueError(f"{group.name}/{key}: rows of {a.shape[1:]}, not {d.shape[1:]}")
            start = d.shape[0]
            d.resize(start + n, axis=0)
            d[start:] = a
        known.update(arrays)
        return n

    def append(self, ticks: dict | None, tracking: dict | None = None, mono_ns=None) -> None:
        """One drain: ``ticks`` from ``Arm.drain`` (None: no loop), ``tracking`` from
        ``Tracking.drain``. ``mono_ns`` is ``time.monotonic_ns()`` at the drain, now if None;
        it pairs with the drain's last ``t``."""
        mono_ns = time.monotonic_ns() if mono_ns is None else mono_ns
        ticks = ticks or {}
        tracking = tracking or {}
        n = self._append_rows(self._f, ticks)
        self._append_rows(self._f[TRACKING_GROUP], tracking)
        self.ticks += n
        self.dropped_loop += int(ticks.get("dropped", 0))
        self.dropped_trk += int(tracking.get("dropped", 0))
        self._write_totals()
        if n:
            pairs = self._f["clock_pairs"]
            pairs.resize(pairs.shape[0] + 1, axis=0)
            pairs[-1] = (float(mono_ns), float(ticks["t"][-1]))
        self._f.flush()

    def add_segment(self, start_t: float, end_t: float, kind: str, label: str = "") -> None:
        seg = self._f["segments"]
        seg.resize(seg.shape[0] + 1, axis=0)
        seg[-1] = (float(start_t), float(end_t), str(kind), str(label))
        self._f.flush()

    def record(self, label: str, text: str) -> None:
        """Keep what the operator typed at a prompt under ``label`` (the ``records`` attr)."""
        self._records[str(label)] = str(text)
        self._f.attrs["records"] = json.dumps(self._records)
        self._f.flush()

    def set_attr(self, name: str, value) -> None:
        self._f.attrs[name] = _attr(value)
        self._f.flush()

    def close(self) -> None:
        if self._f.id.valid:
            self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
