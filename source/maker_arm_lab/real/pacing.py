"""The loop's clock on a rig (MS5 design sections 4.3 and 8): pace the steps at the control
rate and classify every tick for the watchdogs and the twin gate's bars.

A tick is one loop step, from the end of the previous step's wait to the end of this one's.
With nothing late it lasts one period. A step whose work overruns the period does not sleep,
and the next deadline restarts one period after it, so a late tick is never followed by a
short one: the arm moves at most one period's worth per step. Ticks carry two tags: whether a
perception call ran (those ticks are counted apart and have their own cap) and the server's
time in the policy call, which tells forward ticks from queue pops.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from maker_arm_lab.real.latency import percentile

CONTROL_HZ = 30.0
TICK_CAP_S = 0.100
"""MS5 design section 8: a tick without a perception call over this holds the arm."""
PERCEPTION_TICK_CAP_S = 1.0
"""MS5 design section 8: a tick with a perception call over this holds the arm."""
FORWARD_MS = 2.0
"""A policy call over this many ms ran a forward; a queue pop takes microseconds."""
LATE_SLACK_S = 0.002
"""A wake-up this far past the deadline is sleep overshoot, not a late tick."""


@dataclass(frozen=True)
class Tick:
    duration_s: float
    perception: bool
    policy_ms: float | None


class Pacer:
    def __init__(
        self, period_s: float, clock: Callable[[], float] = time.monotonic, sleep=time.sleep
    ):
        self.period, self.clock, self.sleep = float(period_s), clock, sleep
        self.mark = clock()
        self.deadline = self.mark + self.period

    def wait(self) -> float:
        """Sleep to the deadline, or not at all when it passed; return the tick's duration."""
        now = self.clock()
        if now < self.deadline:
            self.sleep(self.deadline - now)
            now = self.clock()
        duration = now - self.mark
        on_time = now <= self.deadline + LATE_SLACK_S
        self.deadline = (self.deadline if on_time else now) + self.period
        self.mark = now
        return duration


class TickLog:
    """Every tick of a run, with the tags noted during it."""

    def __init__(self):
        self.ticks: list[Tick] = []
        self._perception = False
        self._policy_ms: float | None = None

    def note_perception(self) -> None:
        self._perception = True

    def note_policy(self, policy_ms: float | None) -> None:
        self._policy_ms = policy_ms

    def close(self, duration_s: float) -> Tick:
        tick = Tick(float(duration_s), self._perception, self._policy_ms)
        self.ticks.append(tick)
        self._perception, self._policy_ms = False, None
        return tick


def over_cap(tick: Tick) -> str | None:
    """None, or why the tick holds the arm."""
    cap = PERCEPTION_TICK_CAP_S if tick.perception else TICK_CAP_S
    if tick.duration_s <= cap:
        return None
    kind = "perception tick" if tick.perception else "tick"
    return f"{kind} of {1e3 * tick.duration_s:.0f} ms over {1e3 * cap:.0f} ms"


def _block(ms: Sequence[float]) -> dict:
    if not ms:
        return {"n": 0, "p50": None, "p99": None, "max": None}
    return {
        "n": len(ms),
        "p50": round(percentile(ms, 50), 1),
        "p99": round(percentile(ms, 99), 1),
        "max": round(max(ms), 1),
    }


def tick_summary(ticks: Sequence[Tick]) -> dict:
    """The twin gate's tick read (MS5 design P5): plain ticks (no perception call), perception
    ticks, and among the plain ones the forward ticks, each p50, p99 and max in ms."""
    plain = [1e3 * t.duration_s for t in ticks if not t.perception]
    seen = [1e3 * t.duration_s for t in ticks if t.perception]
    forward = [
        1e3 * t.duration_s
        for t in ticks
        if not t.perception and t.policy_ms is not None and t.policy_ms > FORWARD_MS
    ]
    return {
        "plain_ms": _block(plain),
        "perception_ms": _block(seen),
        "forward_ms": _block(forward),
        "plain_over_100ms": sum(ms > 1e3 * TICK_CAP_S for ms in plain),
    }


class TimedPolicy:
    """A policy client whose every step tags the tick with the server's policy time. A step
    that raises (a timeout, a server error) holds the arm through ``trip(reason, detail)``.
    Every other attribute is the client's."""

    def __init__(self, policy, ticks: TickLog, trip):
        self._policy, self._ticks, self._trip = policy, ticks, trip

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def step(self, *args, **kwargs):
        try:
            action = self._policy.step(*args, **kwargs)
        except Exception as err:
            self._trip("policy", f"{type(err).__name__}: {err}")
            raise
        self._ticks.note_policy(getattr(self._policy, "last_policy_ms", None))
        return action
