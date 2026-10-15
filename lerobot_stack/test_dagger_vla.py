"""dagger_vla.py (the MS7h DAgger shards -> V format chain) on tiny synthetic shards."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import convert_to_lerobot as c2l
import dagger_vla
import derive_vla_dataset as dv
from test_convert_to_lerobot import _continuous, _shard

SCRIPT = Path(__file__).resolve().parent / "dagger_vla.py"
CAMERA = [f"overhead_cam_{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw")]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def heap_shard(path):
    """A 16-wide camera_box_target shard: takeover 1 opens closed jaws over 3 frames."""
    _shard(path, actor=[1] * 5 + [2] * 3, bricks=[0] * 5 + [1] * 3, box=True, target=True)
    with h5py.File(path, "a") as f:
        f["data/demo_0/obs/context_camera"].attrs["names"] = json.dumps(CAMERA)
    per_jaw = np.array([0.0, 0.002, 0.005, 0.00725, 0.008, 0.008, 0.008, 0.008])
    return _continuous(path, per_jaw, 0.016)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """Three shards and a corpus derived from the first after a merge, as h/vla/dataset was
    (derive reads the list columns that aggregate_datasets writes)."""
    tmp = tmp_path_factory.mktemp("dag")
    shards = [heap_shard(tmp / f"proc_{p}.hdf5") for p in range(3)]
    kw = {"split_cycles": True, "dagger": True, "gripper": "continuous"}
    c2l.convert(shards[:1], tmp / "act", "t/x", 30, context="camera_box_target", **kw)
    dagger_vla.merge([tmp / "act"], tmp / "merged")
    dv.derive(tmp / "merged", tmp / "corpus")
    return tmp, shards


def chain(*argv):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, argv)], capture_output=True, text=True, check=False
    )
    last = out.stdout.splitlines()[-1]
    assert last.startswith("DAGGER_"), out.stdout + out.stderr
    return last, dict(re.findall(r"(\w+)=(\S+)", last))


def test_two_lanes_reach_the_v_format_with_the_corpus_stats_and_a_checked_tar(rig):
    tmp, shards = rig
    out = tmp / "out" / "dag"
    line, kv = chain("--shards", *shards[1:], "--corpus", tmp / "corpus", "--out", out)
    assert line.startswith("DAGGER_VLA_OK"), line
    assert (kv["episodes"], kv["roots"], kv["reopen_tol"], kv["frames"]) == ("4", "2", "off", "16")
    corpus = tmp / "corpus"
    assert (out / "meta/stats.json").read_bytes() == (corpus / "meta/stats.json").read_bytes()
    info = json.loads((out / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [23]
    parts = sorted(out.parent.glob("dag.tar.part*"))
    whole = b"".join(p.read_bytes() for p in parts)
    assert len(parts) == int(kv["parts"]) and len(whole) == int(kv["bytes"])
    assert hashlib.sha256(whole).hexdigest() == kv["sha256"]
    sha, size, name = (out.parent / "dag.tar.sha256").read_text().split()
    assert (sha, int(size), name) == (kv["sha256"], len(whole), "dag.tar")
    line, _ = chain("--shards", *shards[1:], "--corpus", corpus, "--out", out)
    assert line.startswith("DAGGER_VLA_FAIL FileExistsError"), line


def test_another_host_s_act_root_joins_only_at_the_same_reopen_rule(rig):
    tmp, shards = rig
    act = tmp / "b" / "act"
    line, kv = chain("--shards", shards[1], "--act_only", "--out", act, "--drop_reopen_lead", 0.002)
    assert line.startswith("DAGGER_ACT_OK"), line
    assert (kv["episodes"], kv["reopen_frames"], kv["reopen_takeovers"]) == ("2", "3", "1")
    args = ["--act", act, "--shards", shards[2], "--corpus", tmp / "corpus"]
    line, kv = chain(*args, "--out", tmp / "mix" / "dag", "--drop_reopen_lead", 0.002)
    assert line.startswith("DAGGER_VLA_OK"), line
    assert (kv["episodes"], kv["roots"], kv["reopen_frames"], kv["frames"]) == ("4", "2", "6", "10")
    line, _ = chain(*args, "--out", tmp / "mix2" / "dag")
    assert "reopen_tol 0.002, this run None" in line


def test_a_root_unlike_the_corpus_is_refused(rig):
    tmp, shards = rig
    other = tmp / "corpus15"
    shutil.copytree(tmp / "corpus", other)
    info = json.loads((other / "meta/info.json").read_text())
    (other / "meta/info.json").write_text(json.dumps({**info, "fps": 15}))
    line, _ = chain("--shards", shards[1], "--corpus", other, "--out", tmp / "bad" / "dag")
    assert line.startswith("DAGGER_VLA_FAIL") and "differs from the corpus" in line
    assert "['fps']" in line
    line, _ = chain("--out", tmp / "none")
    assert line == "DAGGER_VLA_FAIL ValueError: give --shards or --act"
