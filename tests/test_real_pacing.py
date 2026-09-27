"""MS5 design sections 4.3 and 8: the step clock, the tick classes and their caps."""

import pytest

from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.real import pacing
from real_fakes import FakeClock

PERIOD = 1.0 / 30.0


def test_an_on_time_tick_sleeps_to_the_deadline_and_keeps_the_grid():
    clock = FakeClock()
    pacer = pacing.Pacer(PERIOD, clock, clock.sleep)
    clock.advance(0.010)
    assert pacer.wait() == pytest.approx(PERIOD)
    assert clock.sleeps == [pytest.approx(PERIOD - 0.010)]
    assert pacer.wait() == pytest.approx(PERIOD)


def test_a_late_tick_does_not_sleep_and_the_next_one_is_a_full_period():
    clock = FakeClock()
    pacer = pacing.Pacer(PERIOD, clock, clock.sleep)
    clock.advance(0.080)
    assert pacer.wait() == pytest.approx(0.080) and clock.sleeps == []
    assert pacer.wait() == pytest.approx(PERIOD)


def test_the_tick_log_tags_one_tick_and_starts_the_next_clean():
    log = pacing.TickLog()
    log.note_perception()
    log.note_policy(12.5)
    first = log.close(0.04)
    second = log.close(0.03)
    assert (first.perception, first.policy_ms) == (True, 12.5)
    assert (second.perception, second.policy_ms) == (False, None)
    assert log.ticks == [first, second]


def test_the_caps_split_by_perception():
    assert pacing.over_cap(pacing.Tick(0.099, False, None)) is None
    assert "100 ms" in pacing.over_cap(pacing.Tick(0.101, False, None))
    assert pacing.over_cap(pacing.Tick(0.9, True, None)) is None
    assert "perception" in pacing.over_cap(pacing.Tick(1.1, True, None))


def test_the_summary_reads_plain_perception_and_forward_ticks_apart():
    ticks = [pacing.Tick(0.0333, False, 0.01)] * 98 + [
        pacing.Tick(0.045, False, 20.0),
        pacing.Tick(0.4, True, 0.01),
    ]
    out = pacing.tick_summary(ticks)
    assert out["plain_ms"]["n"] == 99 and out["perception_ms"]["n"] == 1
    assert out["forward_ms"]["n"] == 1 and out["forward_ms"]["max"] == 45.0
    assert out["plain_over_100ms"] == 0 and out["plain_ms"]["p50"] == pytest.approx(33.3)


class _Policy:
    context_dim = 16

    def __init__(self, fail=False):
        self.fail, self.last_policy_ms = fail, None

    def step(self, *args, **kwargs):
        if self.fail:
            raise TimeoutError("no reply")
        self.last_policy_ms = 7.5
        return "action"


def test_the_timed_policy_tags_the_tick_and_passes_the_rest_through():
    log = pacing.TickLog()
    policy = pacing.TimedPolicy(_Policy(), log, trip=None)
    assert policy.context_dim == 16 and policy.step("obs") == "action"
    assert log.close(0.03).policy_ms == 7.5


def test_a_failed_policy_step_trips():
    def trip(reason, detail):
        raise EvalFailure(f"watchdog_{reason}", detail)

    with pytest.raises(EvalFailure) as err:
        pacing.TimedPolicy(_Policy(fail=True), pacing.TickLog(), trip).step("obs")
    assert err.value.reason == "watchdog_policy" and "TimeoutError" in err.value.detail
