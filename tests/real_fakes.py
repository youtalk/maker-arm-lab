"""Test doubles for the MS5 real env. Each task that needs a new one adds it here."""

from __future__ import annotations


class FakeClock:
    """A monotonic clock that advances only when slept on or told to."""

    def __init__(self, start: float = 100.0):
        self.t = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds
