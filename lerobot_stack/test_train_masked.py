"""train_masked.py (lerobot-train with the MS7g frame mask), in the LeRobot venv."""

import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

import train_masked as tm


class FakeDataset:
    """Two episodes of 6 and 4 frames, a 4-frame action chunk, LeRobot's padding at ends."""

    episode_ends = (6, 10)
    num_frames = 10
    delta_timestamps: ClassVar[dict] = {"action": [k / 30 for k in range(4)]}
    meta = SimpleNamespace(fps=30, episodes={"length": [6, 4]})
    root = "fake"

    def __getitem__(self, j):
        end = next(e for e in self.episode_ends if j < e)
        pad = torch.BoolTensor([j + k >= end for k in range(4)])
        return {"index": torch.tensor(j), "action_is_pad": pad}


def test_masked_frames_are_never_samples():
    data = tm.MaskedFrames(FakeDataset(), np.array([2, 3]))
    assert len(data) == data.num_frames == 8
    assert [int(data[i]["index"]) for i in range(len(data))] == [0, 1, 4, 5, 6, 7, 8, 9]


def test_a_chunk_is_padded_from_the_first_masked_frame_it_reaches():
    data = tm.MaskedFrames(FakeDataset(), np.array([2, 3]))
    assert data[0]["action_is_pad"].tolist() == [False, False, True, True]
    assert data[1]["action_is_pad"].tolist() == [False, True, True, True]
    # Frame 4 reaches no masked frame; LeRobot's own end-of-episode padding stays.
    assert data[2]["action_is_pad"].tolist() == [False, False, True, True]


def test_without_masked_frames_nothing_changes():
    plain, data = FakeDataset(), tm.MaskedFrames(FakeDataset(), np.zeros(0, np.int64))
    for j in range(10):
        assert torch.equal(data[j]["action_is_pad"], plain[j]["action_is_pad"])


def test_everything_else_is_the_dataset():
    data = tm.MaskedFrames(FakeDataset(), np.array([1]))
    assert data.meta.fps == 30 and data.root == "fake"


def test_a_mask_of_another_dataset_is_refused(tmp_path):
    np.savez(tmp_path / "m.npz", lengths=np.array([5, 5]), drop=np.array([1]))
    make = tm.masked(lambda cfg: FakeDataset(), str(tmp_path / "m.npz"))
    with pytest.raises(RuntimeError, match="another dataset"):
        make(None)
    np.savez(tmp_path / "m.npz", lengths=np.array([6, 4]), drop=np.array([1]))
    assert len(make(None)) == 9


def _main(monkeypatch, argv):
    from lerobot.scripts import lerobot_train

    seen = {}
    monkeypatch.setattr(lerobot_train, "main", lambda: seen.update(argv=list(sys.argv)))
    original = lerobot_train.make_dataset
    monkeypatch.setattr(lerobot_train, "make_dataset", original)  # restored after the test
    monkeypatch.setattr(sys, "argv", ["train_masked.py"])
    tm.main(argv)
    return seen["argv"][1:], lerobot_train.make_dataset is original


def test_without_the_switch_it_is_lerobot_train(monkeypatch):
    argv, untouched = _main(monkeypatch, ["--steps=5", "--dataset.root=/x"])
    assert argv == ["--steps=5", "--dataset.root=/x"] and untouched


def test_the_switch_wraps_the_dataset_and_leaves_lerobot_the_rest(monkeypatch):
    argv, untouched = _main(monkeypatch, ["--steps=5", "--frame_mask=/m.npz"])
    assert argv == ["--steps=5"] and not untouched


def test_on_a_converted_dataset_the_masked_reopens_leave_every_chunk(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import reopen_mask as rm
    from test_reopen_mask import converted, corpus, fake_pick

    shards = corpus(tmp_path)
    root = converted(tmp_path, shards)
    _, drop, _ = rm.truth_mask(shards, pick_points=fake_pick)
    chunk = {"action": [k / 30 for k in range(30)]}
    data = tm.MaskedFrames(LeRobotDataset(rm.REPO_ID, root=root, delta_timestamps=chunk), drop)
    assert len(data) == data.dataset.num_frames - len(drop)
    for i in range(len(data)):
        item = data[i]
        targets = int(item["index"]) + np.flatnonzero(~item["action_is_pad"].numpy())
        assert not np.isin(targets, drop).any()
