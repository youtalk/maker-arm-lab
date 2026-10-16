"""MS5 control plan task 9: ``real.motion``, the motor-side and tip-side reads of a session
log, on synthetic logs written with ``real.telemetry.TelemetryLog``."""

import math

import numpy as np
import pytest

from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.real import motion
from maker_arm_lab.real.arm import pick_point
from maker_arm_lab.real.motion import (
    align,
    drag_resistance,
    flex_share,
    free_residual_rms,
    hold_jitter,
    hold_pairs,
    segments,
    stop_sway,
    tip_sway,
    tracking_rms,
    yield_mm,
)
from maker_arm_lab.real.telemetry import TelemetryLog

DT = 0.005
POSE = np.array([0.0, -0.6, 1.2, 0.0, 0.8, 0.0])
"""A pose clear of every joint stop."""
MONO0 = 7_000_000_000
"""``time.monotonic_ns()`` at SDK t = 0 in the synthetic logs."""
CHUNK = 100
"""Rows per drain: one clock pair per 0.5 s, as the session drains."""


@pytest.fixture(scope="module")
def ik():
    return GlobalIK()


def write_log(
    path, t, q, *, q_r=None, r=None, offset=None, gate=None, segments=(), drop=(), lost=()
):
    """A session log of six-wide ``q`` at times ``t``: one drain per ``CHUNK`` rows, its clock
    pair at ``MONO0 + t_last``. The tracking rows default to ``q_r = q``, ``r = 0``,
    ``offset = 0``, ``gate = 1``; the rows in ``drop`` have no tracking row (a fault hold), the
    rows in ``lost`` no row at all (ticks the rings dropped: a gap in ``tick``)."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = t.shape[0]
    q_r = q.copy() if q_r is None else np.asarray(q_r, dtype=float)
    r = np.zeros((n, 6)) if r is None else np.asarray(r, dtype=float)
    offset = np.zeros((n, 6)) if offset is None else np.asarray(offset, dtype=float)
    gate = np.ones(n) if gate is None else np.asarray(gate, dtype=float)
    keep = np.ones(n, dtype=bool)
    keep[list(drop)] = False
    tick = np.arange(n)
    if len(lost):
        alive = np.ones(n, dtype=bool)
        alive[list(lost)] = False
        t, q, q_r, r, offset, gate, keep, tick = (
            a[alive] for a in (t, q, q_r, r, offset, gate, keep, tick)
        )
        n = t.shape[0]

    def seven(a):
        return np.concatenate([a, np.zeros((a.shape[0], 1))], axis=1)

    with TelemetryLog(path, {"lab_revision": "test"}) as log:
        for s in range(0, n, CHUNK):
            sl = slice(s, min(n, s + CHUNK))
            m = sl.stop - sl.start
            ticks = {
                "t": t[sl],
                "tick": tick[sl],
                "clamped": np.zeros(m, dtype=bool),
                "q": seven(q[sl]),
                "dq": np.zeros((m, 7)),
                "tau": np.zeros((m, 7)),
                "temp": np.full((m, 7), 30.0),
                "fault_bits": np.zeros((m, 7), dtype=np.uint16),
                "feedback_age": np.zeros((m, 7)),
                "cmd_pos": seven(q_r[sl]),
                "cmd_vel": np.zeros((m, 7)),
                "cmd_kp": np.zeros((m, 7)),
                "cmd_kd": np.zeros((m, 7)),
                "cmd_tau": np.zeros((m, 7)),
            }
            k = keep[sl]
            tracking = {
                "tick_t": t[sl][k],
                "gate": gate[sl][k],
                "r": r[sl][k],
                "offset": offset[sl][k],
                "q_r": q_r[sl][k],
                "tau_ff": np.zeros((int(k.sum()), 6)),
            }
            log.append(ticks, tracking, mono_ns=MONO0 + round(t[sl][-1] * 1e9))
        for seg in segments:
            log.add_segment(*seg)
    return path


def still(duration, start=0.0):
    t = start + np.arange(round(duration / DT)) * DT
    return t, np.tile(POSE, (t.shape[0], 1))


def tag_line(t_sdk, cam_pos=None, pos=None, latency_ms=91.0):
    """One tag frame seen at SDK time ``t_sdk``: it arrives ``latency_ms`` later."""
    arrival = MONO0 + round(t_sdk * 1e9) + round(latency_ms * 1e6)
    return {
        "stamp_ns": arrival - 5_000_000,
        "arrival_mono_ns": arrival,
        "rms_px": 0.3,
        "cam_pos": None if cam_pos is None else list(map(float, cam_pos)),
        "pos": None if pos is None else list(map(float, pos)),
        "quat": None if pos is None else [1.0, 0.0, 0.0, 0.0],
    }


# --- the segments -----------------------------------------------------------------------


def test_a_transport_is_followed_by_its_stop_window_and_an_approach_is_not(tmp_path):
    t, q = still(6.0)
    segs = [
        (0.0, 1.0, "approach", "a.npz"),
        (1.0, 2.0, "traj", "a.npz"),
        (2.0, 4.0, "hold", ""),
        (4.0, 4.2, "traj", "b.npz"),
        (4.3, 5.0, "approach", "c.npz"),
        (4.5, 4.8, "prompt", "push"),
    ]
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, segments=segs))
    got = [(s.kind, s.start_t, s.end_t) for s in segments(log)]
    assert got == [
        ("traj", 1.0, 2.0),
        ("hold", 2.0, 4.0),
        ("stop", 2.0, 2.5),
        ("traj", 4.0, 4.2),
        ("stop", 4.2, 4.3),  # cut where the next walk starts
        ("prompt", 4.5, 4.8),
    ]


# --- the motor side ---------------------------------------------------------------------


def synthetic_log_with_stop(path, amp_rad, f_hz, zeta, drift=0.003):
    """A transport on joint 1 that ends at t = 1 s in a decaying sway of ``f_hz`` and damping
    ``zeta``, scaled so that its trend-free peak to peak over the stop window is
    ``2 * amp_rad``, on top of a slow linear sag of ``drift`` rad/s."""
    t, q = still(3.0)
    after = t >= 1.0
    s = t[after] - 1.0
    w = 2.0 * math.pi * f_hz
    sway = np.exp(-zeta * w * s) * np.cos(w * math.sqrt(1.0 - zeta**2) * s)
    window = s < motion.STOP_WINDOW_S
    sway *= 2.0 * amp_rad / (sway[window].max() - sway[window].min())
    q[after, 1] += sway + drift * s
    q[~after, 1] += np.linspace(-0.2, 0.0, int((~after).sum()))
    return motion.load_log(write_log(path, t, q, segments=[(0.0, 1.0, "traj", "x.npz")]))


def stop_seg(log):
    return next(s for s in segments(log) if s.kind == "stop")


def test_a_decaying_3_5_hz_sway_reads_its_peak_to_peak(tmp_path):
    log = synthetic_log_with_stop(tmp_path / "s.h5", amp_rad=0.002, f_hz=3.5, zeta=0.4)
    sway = stop_sway(log, stop_seg(log))
    assert sway[1] == pytest.approx(0.004, rel=0.1)
    assert np.allclose(np.delete(sway, 1), 0.0, atol=1e-12)


def test_the_tip_read_pools_the_joints_through_fk(tmp_path, ik):
    log = synthetic_log_with_stop(tmp_path / "s.h5", amp_rad=0.002, f_hz=3.5, zeta=0.4)
    tip = motion.tip_value("stop_sway", log, stop_seg(log), ik)
    # a sway of joint 1 alone moves the tip by about the joint's lever arm times its sway
    jac = np.stack(
        [(pick_point(ik, POSE + e * 1e-6) - pick_point(ik, POSE - e * 1e-6)) / 2e-6
         for e in np.eye(6)], axis=1,
    )  # fmt: skip
    lever_mm = 1e3 * np.linalg.norm(jac[:, 1])
    assert tip == pytest.approx(lever_mm * stop_sway(log, stop_seg(log))[1], rel=0.05)


def test_tracking_rms_reads_the_reference_minus_the_measurement(tmp_path, ik):
    t, q = still(2.0)
    q_r = q.copy()
    q_r[:, 2] += 0.01
    log = motion.load_log(
        write_log(tmp_path / "s.h5", t, q, q_r=q_r, segments=[(0.5, 1.5, "traj", "x")])
    )
    seg = segments(log)[0]
    assert np.allclose(tracking_rms(log, seg), [0, 0, 0.01, 0, 0, 0], atol=1e-12)
    shift_mm = 1e3 * np.linalg.norm(pick_point(ik, q_r[0]) - pick_point(ik, q[0]))
    assert motion.tip_value("tracking_rms", log, seg, ik) == pytest.approx(shift_mm, rel=1e-6)


def test_tracking_joins_on_the_tick_time_not_the_index(tmp_path):
    t, q = still(2.0)
    q_r = q.copy()
    q_r[:, 0] += 0.01
    q_r[300, 0] += 5.0  # the row a fault hold lost: an index join would misread every row after
    log = motion.load_log(
        write_log(tmp_path / "s.h5", t, q, q_r=q_r, drop=[300], segments=[(0.5, 1.9, "traj", "x")])
    )
    assert tracking_rms(log, segments(log)[0])[0] == pytest.approx(0.01, abs=1e-12)


def test_hold_jitter_skips_the_settle(tmp_path):
    t, q = still(4.0)
    rng = np.random.default_rng(1)
    q[:, 3] += rng.normal(0.0, 1e-4, t.shape[0])
    q[t < 1.0, 3] += 0.05 * np.cos(20.0 * t[t < 1.0])  # the settle after the hold starts
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, segments=[(0.0, 4.0, "hold", "")]))
    jitter = hold_jitter(log, segments(log)[0])
    assert jitter[3] == pytest.approx(1e-4, rel=0.1)
    assert np.allclose(np.delete(jitter, 3), 0.0, atol=1e-12)
    short = motion.Segment("hold", 0.0, 0.8, "", 0)
    assert np.isnan(hold_jitter(log, short)).all()


def test_the_free_residual_leaves_out_the_gated_and_the_prompted_rows(tmp_path):
    t, q = still(3.0)
    r = np.full((t.shape[0], 6), 0.3)
    gate = np.ones(t.shape[0])
    gate[t < 0.5] = 0.0
    r[t < 0.5] = 9.0
    r[(t >= 2.0) & (t <= 2.5)] = 7.0
    log = motion.load_log(
        write_log(tmp_path / "s.h5", t, q, r=r, gate=gate, segments=[(2.0, 2.5, "prompt", "p")])
    )
    assert np.allclose(free_residual_rms(log), 0.3, atol=1e-12)


def test_free_motion_yields_zero_and_a_push_yields_its_offset(tmp_path, ik):
    t, q = still(3.0)
    offset = np.zeros((t.shape[0], 6))
    push = (t >= 1.5) & (t < 2.5)
    offset[push, 1] = 0.02 * np.sin(math.pi * (t[push] - 1.5))
    r = np.zeros((t.shape[0], 6))
    r[push, 1] = -1.5 * np.sin(math.pi * (t[push] - 1.5))
    segs = [(0.0, 1.5, "hold", "free"), (1.5, 3.0, "prompt", "push")]
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, offset=offset, r=r, segments=segs))
    free, pushed = (s for s in segments(log) if s.kind in ("hold", "prompt"))
    assert yield_mm(log, free, ik) == pytest.approx(0.0, abs=1e-9)
    q_d = POSE - np.array([0, 0.02, 0, 0, 0, 0])
    expected = 1e3 * np.linalg.norm(pick_point(ik, POSE) - pick_point(ik, q_d))
    assert yield_mm(log, pushed, ik) == pytest.approx(expected, rel=1e-3)
    assert np.allclose(drag_resistance(log, free), 0.0)
    assert drag_resistance(log, pushed)[1] == pytest.approx(1.5, rel=1e-3)


# --- the tag side -----------------------------------------------------------------------


def test_align_puts_a_tag_frame_on_the_sdk_clock(tmp_path):
    t, q = still(3.0)
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q))
    lines = [tag_line(s, pos=(s, 0.0, 0.0)) for s in (0.2, 1.0, 1.73, 2.9)]
    lines.append(tag_line(1.5))  # the tag was not seen: no pose
    lines.append(tag_line(2.995 + motion.EDGE_S + 0.5, pos=(0.0, 0.0, 0.0)))  # past the log
    got = align(lines, log, latency_ms=91.0)
    assert got.shape == (4, 4)
    assert np.allclose(got[:, 0], [0.2, 1.0, 1.73, 2.9], atol=1e-6)
    assert np.allclose(got[:, 1], got[:, 0], atol=1e-6)  # the pose rides with its frame


def test_hold_pairs_average_each_hold_after_its_settle(tmp_path):
    t, q = still(10.0)
    hold_q = [POSE + 0.05 * k for k in range(3)]
    cams = [np.array([0.1 * k, 0.02, 0.6]) for k in range(3)]
    segs, lines = [], []
    for k, start in enumerate((0.0, 3.5, 7.0)):
        span = (t >= start) & (t < start + 3.0)
        q[span] = hold_q[k]
        q[span & (t < start + 0.6)] += 0.03  # the settle
        segs.append((start, start + 3.0, "hold", ""))
        for s in np.arange(start + 0.1, start + 3.0, 1.0 / 30.0):
            settled = s >= start + 0.95  # clear of the read's 1 s edge
            lines.append(tag_line(s, cam_pos=cams[k] + (0.0 if settled else 0.2), pos=cams[k]))
    log_with_three_holds = motion.load_log(write_log(tmp_path / "s.h5", t, q, segments=segs))
    pairs = hold_pairs(lines, log_with_three_holds, latency_ms=91.0)
    assert len(pairs) == 3 and np.allclose(pairs[0]["q"], hold_q[0], atol=1e-9)
    for pair, want_q, cam in zip(pairs, hold_q, cams):
        assert np.allclose(pair["q"], want_q, atol=1e-9)
        assert np.allclose(pair["tag_cam"], cam, atol=1e-9)
        assert pair["n_tag"] > 50


def _tag_and_fk(ik, t, q, offset, extra=None):
    pts = motion.points(ik, q, offset)
    fk_pos = np.concatenate([t[:, None], pts], axis=1)
    tag = pts + np.array([0.004, -0.002, 0.001])  # a registration bias, no motion
    if extra is not None:
        tag = tag + extra
    return np.concatenate([t[:, None], tag], axis=1), fk_pos


def test_a_rigid_tip_has_a_flex_share_of_zero(tmp_path, ik):
    log = synthetic_log_with_stop(tmp_path / "s.h5", amp_rad=0.01, f_hz=3.5, zeta=0.4)
    tag_pos, fk_pos = _tag_and_fk(ik, log.t, log.q, [0.0, 0.01, 0.02])
    seg = stop_seg(log)
    assert flex_share(tag_pos, fk_pos, seg) == pytest.approx(0.0, abs=1e-9)
    assert tip_sway(tag_pos, seg) > 0.5


def test_a_flexing_tip_reads_the_unexplained_share(tmp_path, ik):
    t, q = still(2.0)
    q[:, 0] += 0.005 * np.sin(2.0 * math.pi * 3.0 * t)
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, segments=[(0.0, 1.0, "traj", "")]))
    pts = motion.points(ik, log.q, [0.0, 0.0, 0.0])
    arm = pts - pts.mean(axis=0)
    # flex: a 7 Hz motion, uncorrelated with the arm's 3 Hz one, of the same total variance
    flex = np.zeros_like(arm)
    arm_var = (arm**2).sum(axis=1).mean()
    flex[:, 2] = math.sqrt(2.0 * arm_var) * np.sin(2.0 * math.pi * 7.0 * t)
    tag_pos, fk_pos = _tag_and_fk(ik, log.t, log.q, [0.0, 0.0, 0.0], extra=flex)
    seg = motion.Segment("stop", 0.0, 2.0, "", 0)
    assert flex_share(tag_pos, fk_pos, seg) == pytest.approx(0.5, abs=0.03)


def test_tip_sway_reads_the_tag_peak_to_peak_in_mm():
    t = np.arange(0.0, 1.0, 1.0 / 30.0)
    pos = np.zeros((t.shape[0], 3))
    pos[:, 0] = 0.25 + 0.0015 * np.cos(2.0 * math.pi * 4.0 * t) + 0.01 * t
    tag_pos = np.concatenate([t[:, None], pos], axis=1)
    seg = motion.Segment("stop", 0.0, 1.0, "", 0)
    assert tip_sway(tag_pos, seg) == pytest.approx(3.0, rel=0.05)


def test_points_is_the_fk_pick_point_with_its_offset(ik):
    from maker_arm_lab.robots.frames import PICK_OFFSET_MEASURED

    q = np.stack([POSE, POSE + 0.1])
    got = motion.points(ik, q, PICK_OFFSET_MEASURED)
    assert np.allclose(got, [pick_point(ik, row) for row in q], atol=1e-12)


def test_a_near_still_tag_reads_no_flex_share(tmp_path, ik):
    log = synthetic_log_with_stop(tmp_path / "s.h5", amp_rad=1e-5, f_hz=3.5, zeta=0.4, drift=0.0)
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 1e-4, (log.t.shape[0], 3))  # 0.1 mm: the camera's noise floor
    tag_pos, fk_pos = _tag_and_fk(ik, log.t, log.q, [0.0, 0.0, 0.0], extra=noise)
    seg = stop_seg(log)
    resid, total = motion.flex_variances(tag_pos, fk_pos, seg)
    assert total < motion.FLEX_TAG_VAR_FLOOR_MM2 and resid / total > 0.9
    assert math.isnan(flex_share(tag_pos, fk_pos, seg))


def _two_stop_session(tmp_path, ik, lost=()):
    """Two transports, each ending in a stop: the first sways 0.01 rad on joint 1, the second
    barely moves. Tag frames ride on FK with 0.1 mm noise."""
    t, q = still(4.0)
    w = 2.0 * math.pi * 3.5
    for end, amp in ((1.0, 0.01), (3.0, 1e-5)):
        span = (t >= end) & (t < end + 0.6)
        s = t[span] - end
        q[span, 1] += amp * np.exp(-0.4 * w * s) * np.cos(w * s)
    segs = [(0.0, 1.0, "traj", "a"), (2.0, 3.0, "traj", "b")]
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, segments=segs, lost=lost))
    pts = motion.points(ik, log.q, [0.0, 0.0, 0.0])
    noise = np.random.default_rng(5).normal(0.0, 1e-4, pts.shape)
    lines = [
        tag_line(log.t[k], cam_pos=pts[k] + noise[k], pos=pts[k] + noise[k])
        for k in range(0, log.t.shape[0], 7)
    ]
    return log, lines


def test_the_flex_pool_sums_the_variances_and_skips_a_still_stop(tmp_path, ik):
    log, lines = _two_stop_session(tmp_path, ik)
    body = motion.read(log, ik, lines, 91.0, tag_offset=[0.0, 0.0, 0.0])
    stops = [d for d in body["segments"] if d["kind"] == "stop"]
    moving, still_stop = stops
    assert still_stop["flex"]["tag_var_mm2"] < motion.FLEX_TAG_VAR_FLOOR_MM2
    assert still_stop["reads"]["flex_share"]["tip"] is None
    flex = body["metrics"]["flex_share"]
    assert flex["count"] == 1 and flex["excluded"] == 1
    want = moving["flex"]["resid_var_mm2"] / moving["flex"]["tag_var_mm2"]
    assert flex["tip"] == pytest.approx(want, rel=1e-9) and flex["tip"] < 0.05


def test_a_stop_with_a_gap_reads_null_and_is_counted(tmp_path, ik):
    lost = list(range(240, 250))  # 50 ms the rings dropped
    log, lines = _two_stop_session(tmp_path, ik, lost=lost)
    body = motion.read(log, ik, lines, 91.0, tag_offset=[0.0, 0.0, 0.0])
    gapped, clean = (d for d in body["segments"] if d["kind"] == "stop")
    assert gapped["missing_ticks"] == len(lost) and gapped["max_gap_s"] > motion.MAX_GAP_S
    assert gapped["gap_excluded"] and clean["missing_ticks"] == 0 and not clean["gap_excluded"]
    assert all(gapped["reads"][k] == {"joints": None, "tip": None} for k in motion.STOP_READS)
    assert body["metrics"]["stop_sway"]["count"] == 1
    assert body["metrics"]["stop_sway"]["excluded"] == 1
    assert body["metrics"]["flex_share"]["excluded"] == 2  # the gap and the still stop
    traj = next(d for d in body["segments"] if d["kind"] == "traj")
    assert traj["rows"] == traj["tracked_rows"] == 200 and traj["max_gap_s"] < 0.01


def test_a_yield_during_a_transport_reads_free_not_push(tmp_path, ik):
    t, q = still(4.0)
    offset = np.zeros((t.shape[0], 6))
    offset[(t >= 0.5) & (t < 1.0), 2] = 0.01  # the admittance yields in free motion
    segs = [(0.0, 1.5, "traj", "a"), (2.0, 4.0, "prompt", "push")]
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q, offset=offset, segments=segs))
    body = motion.read(log, ik)
    q_d = POSE - np.array([0, 0, 0.01, 0, 0, 0])
    want = 1e3 * np.linalg.norm(pick_point(ik, POSE) - pick_point(ik, q_d))
    assert body["metrics"]["yield_free_mm"]["tip"] == pytest.approx(want, rel=1e-3)
    assert body["metrics"]["yield_free_mm"]["joints"][2] == pytest.approx(0.01)
    assert body["metrics"]["yield_push_mm"]["tip"] == pytest.approx(0.0, abs=1e-9)
    assert body["metrics"]["drag_push"]["count"] == 1


def test_the_read_needs_a_latency_with_a_tag_log(tmp_path, ik):
    t, q = still(1.0)
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q))
    with pytest.raises(ValueError, match="latency_ms"):
        motion.read(log, ik, [tag_line(0.5, pos=(0, 0, 0))], None, tag_offset=[0, 0, 0])


def test_the_read_carries_the_log_attrs(tmp_path, ik):
    t, q = still(1.0)
    log = motion.load_log(write_log(tmp_path / "s.h5", t, q))
    meta = motion.read(log, ik)["log_meta"]
    assert meta["lab_revision"] == "test" and meta["dropped"] == 0 and meta["records"] == {}
