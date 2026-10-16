"""The body of one powered arm session (MS5 control plan task 6): the plan, its checks, and
the run that drains every tick into the session log. ``scripts/real_session.py`` is its thin
client; it owns the token, the hold on failure and the release.

A plan is a JSON object ``{"segments": [...]}`` (or the bare list), run in order:

- ``{"hold": seconds}`` holds the current target for that long.
- ``{"traj": "file.npz"}`` (a path relative to the plan) tracks the file's rows: arrays ``t``
  (N,), ``q`` (N, 6) in URDF joint coordinates and ``grip`` (N,), the gripper motor angle, at
  200 Hz from 0. A minimum-jerk walk (``core.trajectory.min_jerk``) at
  ``APPROACH_SPEED_RAD_S`` first takes the arm from its current target to row 0; it is logged
  as its own ``approach`` segment, so a ``traj`` segment is the file's rows only.
- ``{"prompt": "text", "record": "label"}`` asks the operator, waits for Enter and keeps the
  typed line under ``label`` in the log's ``records``.

Every check that can fail (the plan, each file, ``check_traj``) runs in ``load_plan``, before
the arm is enabled. A ``traj`` needs the tracking controller (``scripts/real_session.py
--controller tracking``); the hold controller runs ``hold`` and ``prompt`` only. A prompt first
drops any line typed before it. The run drains the SDK every ``DRAIN_PERIOD_S`` (the rings hold 4000
ticks, 20 s) and stops with ``SessionFailure`` the moment the loop reports a fault or no longer
runs the controller.
"""

from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from maker_arm_lab.core.trajectory import MIN_JERK_PEAK_VELOCITY, min_jerk

CONTROL_HZ = 200.0
"""The SDK loop's rate (``maker_arm_rs.profile()["control_rate_hz"]``) and a ``traj`` file's."""
DT = 1.0 / CONTROL_HZ
APPROACH_SPEED_RAD_S = 0.3  # INITIAL: session 2
"""The mean speed of the walk to a trajectory's first row, along its largest joint motion."""
APPROACH_MIN_S = 1.0  # INITIAL: session 2
"""The shortest walk: a short leg at the approach speed would accelerate hard."""
TRAJ_MARGIN_RAD = 0.1
"""A trajectory stays this far inside every joint's soft limits."""
DRAIN_PERIOD_S = 0.5
POLL_S = 0.02
PUSH_LEAD_S = 0.05
"""A push is due this far ahead of the loop clock, so its first row is never late."""
SEGMENT_KINDS = ("hold", "traj", "prompt")


class SessionFailure(Exception):
    """A failure with a token reason: ``fault``, ``tracking_lost``, ``end_of_input``."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


def _joint_names(profile: dict) -> list[str]:
    return [str(j["name"]) for j in profile["joints"]]


def check_traj(traj: dict, profile: dict, margin_rad: float = TRAJ_MARGIN_RAD) -> None:
    """Raise ValueError unless ``traj`` (``t``, ``q`` (N, 6), ``grip`` (N,)) runs at 200 Hz
    from 0, stays inside ``[q_lo + margin_rad, q_hi - margin_rad]`` on all seven joints and
    never moves a joint faster than the profile's ``max_velocity``. The message names the
    joint and the time."""
    t = np.asarray(traj["t"], dtype=np.float64)
    q = np.asarray(traj["q"], dtype=np.float64)
    grip = np.asarray(traj["grip"], dtype=np.float64)
    n = t.shape[0]
    if t.ndim != 1 or n == 0 or q.shape != (n, 6) or grip.shape != (n,):
        raise ValueError(
            f"a trajectory needs t (N,), q (N, 6) and grip (N,), got {t.shape}, {q.shape}, "
            f"{grip.shape}"
        )
    rows = np.concatenate([q, grip[:, None]], axis=1)
    if not (np.isfinite(t).all() and np.isfinite(rows).all()):
        raise ValueError("a trajectory must be finite")
    if not np.allclose(t, np.arange(n) * DT, rtol=0.0, atol=1e-6):
        raise ValueError(f"t must run at {CONTROL_HZ:.0f} Hz from 0")
    names = _joint_names(profile)
    for j, joint in enumerate(profile["joints"]):
        lo, hi = float(joint["q_lo"]) + margin_rad, float(joint["q_hi"]) - margin_rad
        outside = np.flatnonzero((rows[:, j] < lo) | (rows[:, j] > hi))
        if outside.size:
            k = int(outside[0])
            raise ValueError(
                f"{names[j]} at t = {t[k]:.3f} s is {rows[k, j]:.4f} rad, outside "
                f"[{lo:.4f}, {hi:.4f}] (the soft limits less {margin_rad} rad)"
            )
    vmax = float(profile["max_velocity"])
    speed = np.abs(np.diff(rows, axis=0)) / DT
    for j in range(rows.shape[1]):
        fast = np.flatnonzero(speed[:, j] > vmax)
        if fast.size:
            k = int(fast[0])
            raise ValueError(
                f"{names[j]} at t = {t[k]:.3f} s moves at {speed[k, j]:.3f} rad/s, past the "
                f"profile's max_velocity {vmax} rad/s"
            )


def load_traj(path: Path) -> dict:
    with np.load(path) as data:
        missing = sorted({"t", "q", "grip"} - set(data.files))
        if missing:
            raise ValueError(f"{path.name} has no arrays {missing}")
        return {k: np.asarray(data[k], dtype=np.float64) for k in ("t", "q", "grip")}


def load_plan(path, profile: dict) -> list[dict]:
    """The plan at ``path`` as checked segments, each a dict with ``kind``. Raises ValueError
    naming the segment for anything the run would refuse."""
    path = Path(path)
    data = json.loads(path.read_text())
    raw = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise ValueError("a plan is a non-empty list of segments, or {'segments': [...]}")
    peak = MIN_JERK_PEAK_VELOCITY * APPROACH_SPEED_RAD_S
    if peak > float(profile["max_velocity"]):
        raise ValueError(
            f"the approach peaks at {peak:.3f} rad/s, past max_velocity {profile['max_velocity']}"
        )
    segments, labels = [], set()
    for i, seg in enumerate(raw):
        kinds = [k for k in SEGMENT_KINDS if isinstance(seg, dict) and k in seg]
        extra = set(seg) - {*SEGMENT_KINDS, "record"} if isinstance(seg, dict) else set()
        if len(kinds) != 1 or extra or ("record" in seg and kinds != ["prompt"]):
            raise ValueError(f"segment {i}: expected one of {SEGMENT_KINDS}, got {seg!r}")
        kind = kinds[0]
        try:
            if kind == "hold":
                seconds = seg["hold"]
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                    raise ValueError(f"hold needs seconds, got {seconds!r}")
                if not (math.isfinite(seconds) and seconds > 0.0):
                    raise ValueError(f"hold needs positive seconds, got {seconds!r}")
                segments.append({"kind": "hold", "seconds": float(seconds)})
            elif kind == "traj":
                file = path.parent / str(seg["traj"])
                traj = load_traj(file)
                check_traj(traj, profile)
                segments.append({"kind": "traj", "label": str(seg["traj"]), "traj": traj})
            else:
                label = seg.get("record")
                if not isinstance(label, str) or not label or label in labels:
                    raise ValueError(f"a prompt needs a new, non-empty record label, got {label!r}")
                labels.add(label)
                segments.append({"kind": "prompt", "text": str(seg["prompt"]), "record": label})
        except (OSError, ValueError, KeyError) as err:
            raise ValueError(f"segment {i} ({kind}): {err}") from err
    return segments


def approach_rows(
    start: Sequence[float], goal: Sequence[float], speed: float = APPROACH_SPEED_RAD_S
) -> list[list[float]]:
    """The minimum-jerk walk from ``start`` to ``goal`` (seven wide), one row per tick from
    ``start`` up to the row before ``goal``: its largest joint motion at mean ``speed``, at
    least ``APPROACH_MIN_S`` long. Empty when the two already agree."""
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(goal, dtype=np.float64)
    length = float(np.abs(b - a).max())
    if length < 1e-9:
        return []
    n = math.ceil(max(length / speed, APPROACH_MIN_S) * CONTROL_HZ)
    return [(a + (b - a) * min_jerk(k / n)).tolist() for k in range(n)]


class Lines:
    """The operator's input, one line at a time, read on a thread so the session keeps
    draining while a prompt waits. ``get`` returns None when no line came within ``timeout``
    and raises EOFError once the input has ended."""

    _EOF = object()

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ended = False

    def _read(self) -> None:
        while True:
            try:
                line = self.stream.readline()
            except (OSError, ValueError):
                line = ""
            if not line:
                self._queue.put(self._EOF)
                return
            self._queue.put(line)

    def start(self) -> Lines:
        """Start reading now, so lines typed ahead land in the queue, where ``clear`` drops
        them, rather than wait in the input for the next question."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._read, name="operator-input", daemon=True)
            self._thread.start()
        return self

    def clear(self) -> None:
        """Drop every line typed before this question (on a terminal also the unread input),
        so an early answer, or an early RELEASE, never answers it. The end of input stays."""
        try:
            if self.stream.isatty():
                import termios

                termios.tcflush(self.stream.fileno(), termios.TCIFLUSH)
        except (AttributeError, OSError, ValueError):
            pass
        ended = False
        while True:
            try:
                ended = self._queue.get_nowait() is self._EOF or ended
            except queue.Empty:
                break
        if ended:
            self._queue.put(self._EOF)

    def get(self, timeout: float) -> str | None:
        if self._ended:
            raise EOFError
        self.start()
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._EOF:
            self._ended = True
            raise EOFError
        return item


def drain_into(sdk, log) -> tuple[int, int]:
    """One drain of ``sdk`` into ``log``; returns the ticks with a fault bit and the clamped
    ticks it held."""
    ticks, tracking = sdk.drain()
    log.append(ticks, tracking, mono_ns=time.monotonic_ns())
    if not ticks or not ticks.get("t"):
        return 0, 0
    faults = int(np.any(np.asarray(ticks["fault_bits"]) != 0, axis=1).sum())
    return faults, int(np.count_nonzero(ticks["clamped"]))


def current_target(sdk) -> list[float]:
    """Seven wide: the controller's target (its reference less the admittance offset) and the
    measured gripper angle; the measured pose before the first telemetry."""
    reading = sdk.read()
    tel = sdk.telemetry()
    if tel is None:
        return [*reading.q, reading.gripper]
    return [float(r) - float(o) for r, o in zip(tel["q_r"], tel["offset"])] + [reading.gripper]


class _Run:
    def __init__(self, sdk, log, lines, drain_s, poll_s, out, interrupted):
        self.sdk, self.log, self.lines = sdk, log, lines
        self.interrupted = interrupted
        self.drain_s, self.poll_s, self.out = drain_s, poll_s, out
        self.faults = self.clamped = self.segments = 0
        self.records: dict[str, str] = {}
        self._last_drain = time.monotonic()

    def say(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def now(self) -> float:
        flag = self.interrupted()
        if flag:
            raise SessionFailure("interrupted", flag if isinstance(flag, str) else "Ctrl-C")
        reading = self.sdk.read()
        if reading is None or not reading.alive or reading.state != "enabled" or reading.fault:
            state = None if reading is None else (reading.state, reading.alive, reading.fault)
            raise SessionFailure("fault", f"the loop reports {state}")
        now = self.sdk.now()
        if now is None:
            raise SessionFailure("tracking_lost", "the loop no longer runs the controller")
        return now

    def tick(self) -> None:
        if time.monotonic() - self._last_drain >= self.drain_s:
            self.drain()

    def drain(self) -> None:
        self._last_drain = time.monotonic()
        faults, clamped = drain_into(self.sdk, self.log)
        self.faults += faults
        self.clamped += clamped

    def wait_until(self, sdk_t: float) -> float:
        while True:
            now = self.now()
            if now >= sdk_t:
                return now
            self.tick()
            time.sleep(min(self.poll_s, max(sdk_t - now, 0.0)))

    def segment(self, start: float, end: float, kind: str, label: str = "") -> None:
        self.log.add_segment(start, end, kind, label)
        self.segments += 1

    def hold(self, seconds: float) -> None:
        start = self.now()
        self.segment(start, self.wait_until(start + seconds), "hold")

    def traj(self, label: str, traj: dict) -> None:
        rows = np.concatenate([traj["q"], traj["grip"][:, None]], axis=1).tolist()
        approach = approach_rows(current_target(self.sdk), rows[0])
        t0 = self.now() + PUSH_LEAD_S
        self.sdk.push_rows(t0, DT, approach + rows)
        t_traj = t0 + len(approach) * DT
        t_end = t_traj + (len(rows) - 1) * DT
        if approach:
            self.segment(t0, t_traj, "approach", label)
        self.wait_until(t_end)
        self.segment(t_traj, t_end, "traj", label)

    def prompt(self, text: str, record: str) -> None:
        start = self.now()
        self.lines.clear()
        self.say(f"{text}\n(type the answer for '{record}', then Enter)")
        while True:
            try:
                line = self.lines.get(self.poll_s)
            except EOFError:
                raise SessionFailure(
                    "end_of_input", f"the input ended at prompt {record!r}"
                ) from None
            if line is not None:
                break
            self.now()
            self.tick()
        self.records[record] = line.rstrip("\r\n")
        self.log.record(record, self.records[record])
        self.segment(start, self.now(), "prompt", record)


def run_session(
    sdk,
    segments: list[dict],
    log,
    lines: Lines,
    *,
    drain_s: float = DRAIN_PERIOD_S,
    poll_s: float = POLL_S,
    out=None,
    interrupted=lambda: False,
) -> dict:
    """Run ``segments`` (``load_plan``) on a started ``sdk`` (``real.arm.SdkArm``'s session
    surface: ``read``, ``now``, ``telemetry``, ``push_rows``, ``drain``), every tick into
    ``log``. Returns the counts the ``SESSION_OK`` token reports; raises ``SessionFailure`` on
    a fault, a lost controller or the end of input at a prompt. It never holds or stops the
    arm: that is the caller's.

    ``interrupted`` is polled every ``poll_s`` (a Ctrl-C, SIGTERM or SIGHUP flag): when it
    turns true the run raises ``SessionFailure("interrupted")`` between two drains, never
    inside one, so a log append is never cut short; a string flag is the failure's detail.
    ``max_interval_s`` and ``late_ticks`` are the log's tick-gap reads (``real.telemetry``).

    Under the SDK's hold controller (``real.arm.SdkHoldArm``) a plan holds only ``hold`` and
    ``prompt`` segments: the scripts refuse a ``traj`` before connect."""
    out = out if out is not None else sys.stderr
    run = _Run(sdk, log, lines, drain_s, poll_s, out, interrupted)
    for i, seg in enumerate(segments):
        kind = seg["kind"]
        run.say(f"segment {i + 1}/{len(segments)}: {kind} {seg.get('label', '')}".rstrip())
        if kind == "hold":
            run.hold(seg["seconds"])
        elif kind == "traj":
            run.traj(seg["label"], seg["traj"])
        else:
            run.prompt(seg["text"], seg["record"])
    run.drain()
    return {
        "segments": run.segments,
        "ticks": log.ticks,
        "dropped": log.dropped,
        "faults": run.faults,
        "clamped": run.clamped,
        "max_interval_s": log.max_interval_s,
        "late_ticks": log.late_ticks,
        "records": run.records,
    }
