"""MS5 sensing task 7: ``run_read`` against fake frames and a fake worker, the two scripts'
tokens, and the verdict's pooling."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
import pytest
from PIL import Image

from helpers import ROOT
from maker_arm_lab.core import parts as parts_mod
from maker_arm_lab.real.camera_model import Pose, sim_overhead
from maker_arm_lab.real.layouts import Layout, TruthBox, TruthPart, nominal_screen_pose
from maker_arm_lab.real.perceive import WorkerClient
from maker_arm_lab.real.rig import RigConfig
from real_fakes import FakeClock, FakeWorker, StreamOverhead

READ = ROOT / "scripts" / "real_perception_read.py"
VERDICT = ROOT / "scripts" / "real_perception_verdict.py"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


read = _load(READ)
NAME = "brick_2x4"


def _layout(kind="sparse", dx=0.0):
    part = TruthPart(NAME, "wide", 0.20 + dx, 0.05, 0.01, 0.3)
    return Layout("s0", kind, [part], [TruthBox(0.1, 0.2, 0.0)])


def _detection(x=0.20, y=0.05):
    s = parts_mod.by_name([NAME])[0].size
    quat = [math.cos(0.15), 0.0, 0.0, math.sin(0.15)]
    return {
        "centre": [x, y, 0.01],
        "quat": quat,
        "size": [*sorted(s[:2], reverse=True), s[2]],
        "cls": "wide",
    }


def _payload(tmp_path=None, layout=None, n=3, x=0.20, detect=True):
    worker = FakeWorker(boxes=[[0.1, 0.2, 0.0]], detections=[_detection(x)] if detect else [])
    try:
        return read.run_read(
            StreamOverhead(range(1, n + 1)),
            WorkerClient(worker.endpoint),
            layout or _layout(),
            n,
            tmp_path,
        )
    finally:
        worker.close()


def test_run_read_returns_a_table_from_fake_frames(tmp_path):
    out = _payload(tmp_path)
    assert out["table"]["miss"] == 0.0 and out["table"]["counts"]["frames"] == 3
    assert len(out["frames"]) == 3 and out["coverage"] == 0.9
    assert out["k_drift"] == 0.0 and out["rebuilds"] == 0
    assert out["latency_ms"]["p50"] <= out["latency_ms"]["p95"]
    assert (tmp_path / "view_000.png").exists() and (tmp_path / "raw_000.jpg").exists()
    assert Image.open(tmp_path / "view_000.png").size == (224, 224)
    assert not (tmp_path / "view_001.png").exists()
    json.dumps(out)


def test_run_read_counts_a_repeated_stamp_once():
    worker = FakeWorker(boxes=[[0.1, 0.2, 0.0]], detections=[_detection()])
    try:
        overhead = StreamOverhead([1, 1, 2, 2, 3])
        out = read.run_read(overhead, WorkerClient(worker.endpoint), _layout(), 3, None)
    finally:
        worker.close()
    assert [f["stamp_ns"] for f in out["frames"]] == [1, 2, 3]
    assert worker.kinds().count("parts") == 3


def test_run_read_times_out_without_frames():
    clock = FakeClock()
    worker = FakeWorker()
    try:
        with pytest.raises(TimeoutError, match="1 of 3"):
            read.run_read(
                StreamOverhead([1]),
                WorkerClient(worker.endpoint),
                _layout(),
                3,
                None,
                timeout_s=5.0,
                clock=clock,
                sleep=clock.sleep,
            )
    finally:
        worker.close()


def test_run_read_times_out_when_no_bundle_carries_depth():
    clock = FakeClock()
    worker = FakeWorker()
    try:
        with pytest.raises(TimeoutError, match="0 of 3 frames with depth.*depth/image_raw"):
            read.run_read(
                StreamOverhead([1, 2, 3], depth=False),
                WorkerClient(worker.endpoint),
                _layout(),
                3,
                None,
                timeout_s=5.0,
                clock=clock,
                sleep=clock.sleep,
            )
    finally:
        kinds = worker.kinds()
        worker.close()
    assert kinds == ["hello"]


def test_a_rig_with_a_k_and_no_size_fails_before_the_session(tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text(json.dumps({"overhead_k": [500, 0, 320, 0, 500, 240, 0, 0, 1]}))
    lay = tmp_path / "lay.json"
    lay.write_text(_layout().to_json())
    code, tokens = _run(READ, ["--rig", rig, "--layout", lay, "--out", tmp_path / "o.json"])
    assert code == 1 and len(tokens) == 1
    assert tokens[0].startswith("PERCEPTION_READ_FAIL") and '"rig"' in tokens[0]


def test_a_screen_pose_moves_the_truth_it_reads_against(tmp_path):
    rig = RigConfig()
    nominal = nominal_screen_pose(rig)
    moved = Pose(nominal.rot, nominal.pos + np.array([0.01, 0.0, 0.0]))
    screen = tmp_path / "screen.json"
    screen.write_text(json.dumps({"pos": moved.pos.tolist(), "quat": moved.quat()}))
    lay = _layout()
    out = read.truth_layout(rig, lay, str(screen))
    assert out.frame == "measured" and out.parts[0].x == pytest.approx(lay.parts[0].x + 0.01)
    assert read.truth_layout(rig, lay, None) is lay
    # the shifted truth now misses a detection at the nominal spot by 10 mm
    assert _payload(layout=out)["table"]["pos_bias_mm"][0] == pytest.approx(-10.0, abs=0.1)


def _run(script, args, timeout=300):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AMENT_PREFIX_PATH")}
    out = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    tokens = [x for x in out.stdout.splitlines() if x.startswith(("PERCEPTION_",))]
    return out.returncode, tokens


def test_the_read_usage_error_prints_one_fail_line():
    code, tokens = _run(READ, ["--nonsense"])
    assert code != 0 and len(tokens) == 1 and tokens[0].startswith("PERCEPTION_READ_FAIL")


def _write(tmp_path, name, **kw):
    path = tmp_path / f"{name}.json"
    out = _payload(**kw)
    out["name"] = name
    path.write_text(json.dumps(out))
    return path


def _verdict(*paths, rms="0"):
    code, tokens = _run(VERDICT, [*paths, "--check_rms_mm", rms])
    assert len(tokens) == 1
    return code, tokens[0].split(" ", 1)[0], json.loads(tokens[0].split(" ", 1)[1])


def _pair(tmp_path, **kw):
    """A sparse and a cluster read of the same detections: the least the verdict pools."""
    return (
        _write(tmp_path, "s0", layout=_layout("sparse"), **kw),
        _write(tmp_path, "c0", layout=_layout("cluster"), **kw),
    )


def test_the_verdict_pools_sparse_and_cluster_and_reports_the_heap_and_the_empty_screen(
    tmp_path,
):
    heap = Layout(
        "h0", "heap", [TruthPart(NAME, "wide", math.nan, math.nan, math.nan, 0.0)] * 2, []
    )
    paths = [
        *_pair(tmp_path),
        _write(tmp_path, "h0", layout=heap),
        _write(tmp_path, "e0", layout=Layout("empty", "sparse", [], [])),
    ]
    code, token, out = _verdict(*paths)
    assert token == "PERCEPTION_PASS" and code == 0, out
    assert out["table"]["counts"]["frames"] == 6  # the empty screen is not pooled
    assert out["report_only"]["h0"] == {"detected": 3, "poured": 6}
    assert out["report_only"]["e0"] == {"extra": 3}  # one false part in each of 3 frames
    want = {"sparse": [paths[0]], "cluster": [paths[1]], "heap": [paths[2]], "empty": [paths[3]]}
    assert out["reads"] == {k: [str(p) for p in v] for k, v in want.items()}


@pytest.mark.parametrize("kind", ["sparse", "cluster"])
def test_the_verdict_needs_a_read_of_each_pooled_kind(tmp_path, kind):
    code, token, out = _verdict(_write(tmp_path, "r0", layout=_layout(kind)))
    other = "cluster" if kind == "sparse" else "sparse"
    assert token == "PERCEPTION_FAIL" and code == 1 and other in out["reason"]
    assert out["reads"][kind] and not out["reads"][other]


def test_the_verdict_adds_the_calibration_error_to_the_bias_rows(tmp_path):
    paths = _pair(tmp_path, x=0.2055)  # 5.5 mm bias in x against a 6.5 mm limit
    assert _verdict(*paths, rms="0")[1] == "PERCEPTION_PASS"
    code, token, out = _verdict(*paths, rms="2")
    assert token == "PERCEPTION_FAIL" and code == 1 and out["rows"] == ["pos_bias_checked_mm.x"]
    assert out["table"]["pos_bias_mm"][0] == pytest.approx(5.5, abs=0.1)
    assert out["table"]["pos_bias_checked_mm"][0] == pytest.approx(7.5, abs=0.1)


def test_the_verdict_fails_on_one_row_over_its_limit(tmp_path):
    _, token, out = _verdict(*_pair(tmp_path, x=0.30))
    assert token == "PERCEPTION_FAIL" and "miss" in out["rows"]


def test_a_read_that_measured_nothing_fails(tmp_path):
    _, token, out = _verdict(*_pair(tmp_path, detect=False))
    assert token == "PERCEPTION_FAIL" and "pos_bias_checked_mm.x" in out["rows"]


def test_ctrl_c_ends_the_verdict_with_its_fail_token(tmp_path, monkeypatch, capsys):
    import maker_arm_lab.real.layouts as layouts_mod

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(layouts_mod, "Layout", interrupt)
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps({"layout": {"name": "s", "kind": "sparse", "parts": [], "boxes": []}})
    )
    code = _load(VERDICT).main([str(path), "--check_rms_mm", "0"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("PERCEPTION_")]
    assert code == 1 and lines == ["PERCEPTION_FAIL " + json.dumps({"reason": "interrupted"})]


def test_a_near_square_box_folds_its_yaw_by_a_quarter_turn():
    assert read.box_period(RigConfig(box_interior=(0.0746, 0.0753, 0.075))) == math.pi / 2
    assert read.box_period(RigConfig(box_interior=(0.11, 0.07, 0.04))) == math.pi


def _read_run(tmp_path, monkeypatch, rig):
    """``_run`` against a fake 224 x 224 stream of three frames and a fake worker; no ROS."""
    import maker_arm_lab.real
    from maker_arm_lab.real import sources

    class Sub:
        error = None

        def __init__(self, **kw):
            pass

        def start(self):
            return self

        def close(self):
            pass

    fake = types.ModuleType("maker_arm_lab.real.ros")
    fake.ConduitSubscriber = Sub
    monkeypatch.setitem(sys.modules, "maker_arm_lab.real.ros", fake)
    monkeypatch.setattr(maker_arm_lab.real, "ros", fake, raising=False)

    def overhead(*a, **k):
        o = StreamOverhead([1, 2, 3])
        o.on_jpeg = o.on_info = o.on_depth = o.on_depth_info = o.on_error = None  # Sub ignores
        return o

    monkeypatch.setattr(sources, "ConduitOverhead", overhead)
    (tmp_path / "rig.json").write_text(rig.to_json())
    (tmp_path / "lay.json").write_text(_layout().to_json())
    worker = FakeWorker(boxes=[[0.1, 0.2, 0.0]], detections=[_detection()])
    try:
        args = read.build_parser().parse_args(
            ["--rig", str(tmp_path / "rig.json"), "--layout", str(tmp_path / "lay.json"),
             "--worker", worker.endpoint, "--frames", "3", "--timeout_s", "5",
             "--out", str(tmp_path / "out.json")]
        )  # fmt: skip
        return read._run(args)
    finally:
        worker.close()


def test_a_rig_without_an_overhead_size_fails_before_the_session(tmp_path, monkeypatch):
    token, code = _read_run(tmp_path, monkeypatch, RigConfig())
    assert code == 1 and token.startswith("PERCEPTION_READ_FAIL ") and "calibrate first" in token
    assert not (tmp_path / "out.json").exists()


def test_a_stream_of_another_size_than_the_calibration_fails_the_read(tmp_path, monkeypatch):
    token, code = _read_run(tmp_path, monkeypatch, RigConfig(overhead_size=(1920, 1440)))
    assert code == 1 and token.startswith("PERCEPTION_READ_FAIL ")
    assert "224x224" in token and "1920x1440" in token
    assert not (tmp_path / "out.json").exists()


def test_the_read_records_the_k_and_the_floor_it_read_with(tmp_path, monkeypatch):
    rig = RigConfig(overhead_size=(224, 224), tray_floor_z=0.0061)
    token, code = _read_run(tmp_path, monkeypatch, rig)
    assert code == 0 and token.startswith("PERCEPTION_READ_OK "), token
    out = json.loads((tmp_path / "out.json").read_text())
    k = list(sim_overhead()[0].ros_k())
    assert out["k_first"] == k and out["k_last"] == k and out["tray_floor_z"] == 0.0061
