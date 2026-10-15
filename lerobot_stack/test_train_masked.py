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
    num_episodes = 2
    fps = 30
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
    monkeypatch.setattr(lerobot_train, "make_policy", lerobot_train.make_policy)
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


# MS7g D12: the wrapped, normalized context (--context_norm).


class ContextDataset(FakeDataset):
    """FakeDataset with a 4-wide context whose index 3 is the camera roll, like the corpus's."""

    def __init__(self, root="fake"):
        self.root = root
        names = ["overhead_cam_x", "overhead_cam_y", "overhead_cam_z", "overhead_cam_roll"]
        self.meta = SimpleNamespace(
            fps=30,
            episodes={"length": [6, 4]},
            features={tm.ENV: {"names": names}},
            stats={tm.ENV: {"mean": np.zeros(4)}},
        )

    def __getitem__(self, j):
        roll = 3.13 if j % 2 else -3.135
        return {**super().__getitem__(j), tm.ENV: torch.tensor([0.1, 0.2, 0.7, roll])}


def _write_context(root, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "data" / "chunk-000").mkdir(parents=True)
    for k, part in enumerate(np.array_split(rows, 2)):
        table = pa.table({tm.ENV: pa.array(part.tolist(), pa.list_(pa.float32()))})
        pq.write_table(table, root / "data" / "chunk-000" / f"file-{k:03d}.parquet")


def test_the_context_stats_are_over_the_wrapped_roll(tmp_path):
    rows = np.array([[0.1, 0.2, 0.7, 3.13], [0.3, 0.2, 0.7, -3.135], [0.2, 0.5, 0.7, 3.14]])
    _write_context(tmp_path, rows.astype(np.float32))
    stats = tm.context_stats(tmp_path, 4)
    wrapped = np.array([3.13 - np.pi, -3.135 + np.pi, 3.14 - np.pi])
    np.testing.assert_allclose(stats["mean"], [0.2, 0.3, 0.7, wrapped.mean()], atol=1e-6)
    np.testing.assert_allclose(stats["std"][[0, 3]], [rows[:, 0].std(), wrapped.std()], atol=1e-6)
    assert stats["count"].tolist() == [3] and stats["max"][3] < 0.01


def test_the_context_switch_wraps_every_sample_and_replaces_the_stats(tmp_path, monkeypatch):
    rows = np.array([[0.1, 0.2, 0.7, 3.13], [0.1, 0.2, 0.7, -3.135]], np.float32)
    _write_context(tmp_path, rows)
    data = tm.masked(lambda cfg: ContextDataset(tmp_path), None, tm.CONTEXT_NORM)(None)
    assert len(data) == 10 and abs(float(data.meta.stats[tm.ENV]["mean"][3])) < 0.01
    roll = [float(data[j][tm.ENV][3]) for j in (0, 1)]
    np.testing.assert_allclose(roll, [-3.135 + np.pi, 3.13 - np.pi], atol=1e-5)
    assert data[0][tm.ENV][:3].tolist() == pytest.approx([0.1, 0.2, 0.7])
    # Without the switch the context reaches training as recorded.
    plain = tm.masked(lambda cfg: ContextDataset(tmp_path), None)(None)
    assert float(plain[0][tm.ENV][3]) == pytest.approx(-3.135)


def test_a_context_whose_roll_moved_is_refused(tmp_path):
    other = ContextDataset(tmp_path)
    other.meta.features[tm.ENV]["names"][3] = "overhead_cam_pitch"
    with pytest.raises(RuntimeError, match="not the roll"):
        tm.masked(lambda cfg: other, None, tm.CONTEXT_NORM)(None)


@pytest.mark.parametrize("proj", tm.PROJ_MODES)
def test_the_policy_maps_env_and_handles_the_projection(proj):
    from lerobot.configs.types import NormalizationMode

    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 3)
    old = torch.nn.Linear(4, 3)
    old.load_state_dict(layer.state_dict())
    stats = {"mean": np.array([0.1, 0.2, 0.7, 0.0]), "std": np.array([0.01, 0.02, 0.005, 0.02])}
    cfg = SimpleNamespace(normalization_mapping={"STATE": NormalizationMode.MEAN_STD})
    policy = SimpleNamespace(model=SimpleNamespace(encoder_env_state_input_proj=layer))
    meta = SimpleNamespace(stats={tm.ENV: stats})
    tm.normalized(lambda cfg, ds_meta, **kw: policy, proj)(cfg, ds_meta=meta, rename_map={})
    assert cfg.normalization_mapping["ENV"] == NormalizationMode.MEAN_STD
    x = torch.tensor([[0.11, 0.17, 0.705, 0.01]])
    z = (x - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
    same = torch.equal(layer.weight, old.weight)
    if proj == "keep":
        assert same
    elif proj == "reinit":
        assert not same
    else:  # fold: the normalized input reaches the same output as the raw one did
        torch.testing.assert_close(layer(z.float()), old(x), atol=1e-5, rtol=1e-5)


def test_the_context_switches_are_checked_and_leave_lerobot_the_rest(monkeypatch):
    argv, untouched = _main(monkeypatch, ["--steps=5", f"--context_norm={tm.CONTEXT_NORM}"])
    assert argv == ["--steps=5"] and not untouched
    for bad in (
        ["--context_proj=fold"],
        ["--context_norm=x"],
        [f"--context_norm={tm.CONTEXT_NORM}", "--context_proj=zero"],
    ):
        with pytest.raises(SystemExit):
            _main(monkeypatch, bad)


# MS7i ACT-C: the corpus mixed with extra roots (--extra_root, --extra_max_share).


def test_the_extra_roots_keep_their_natural_share_under_the_cap():
    lengths = [np.array([10, 20]), np.array([5])]
    kept = tm.keep_episodes(lengths, corpus_frames=100, max_share=0.3)  # 35 / 135 = 0.26
    assert [k.tolist() for k in kept] == [[0, 1], [0]]


def test_over_the_cap_a_fixed_subset_of_episodes_stays_within_the_share():
    lengths = [np.full(50, 10), np.full(30, 7)]  # 710 frames against a 300-frame budget
    kept = tm.keep_episodes(lengths, corpus_frames=700, max_share=0.3)
    frames = sum(int(n[k].sum()) for n, k in zip(lengths, kept))
    assert 300 - 10 < frames <= 300  # 300 / (700 + 300) = 0.3
    assert all(
        len(k) and np.all(np.diff(k) > 0) and k.max() < len(n) for n, k in zip(lengths, kept)
    )
    again = tm.keep_episodes(lengths, corpus_frames=700, max_share=0.3)
    assert all(np.array_equal(a, b) for a, b in zip(kept, again))  # fixed: no global RNG


def _mix_maker(tmp_path, other_features=False):
    rows = np.array([[0.1, 0.2, 0.7, 3.13], [0.1, 0.2, 0.7, -3.135]], np.float32)
    _write_context(tmp_path, rows)

    def make(cfg):
        data = ContextDataset(cfg.dataset.root)
        if other_features and cfg.dataset.root != tmp_path:
            data.meta.features = {tm.ENV: {"names": ["a", "b", "c", "d"]}}
        return data

    return make, SimpleNamespace(dataset=SimpleNamespace(root=tmp_path))


def test_the_mix_adds_each_extra_frame_once_with_the_corpus_meta_and_the_roll_wrap(tmp_path):
    make, cfg = _mix_maker(tmp_path)
    mix = tm.masked(make, None, tm.CONTEXT_NORM, ("/b1", "/b2"), max_share=0.7)(cfg)
    assert (len(mix), mix.num_frames, mix.num_episodes) == (30, 30, 6)
    assert cfg.dataset.root == tmp_path  # restored
    assert [d.root for d in mix.datasets[1:]] == ["/b1", "/b2"]
    assert mix.meta is mix.datasets[0].meta  # the corpus's stats serve every root
    assert abs(float(mix.meta.stats[tm.ENV]["mean"][3])) < 0.01
    for j in (0, 10, 20):  # a corpus, a b1 and a b2 frame: every root's roll is wrapped
        assert float(mix[j][tm.ENV][3]) == pytest.approx(-3.135 + np.pi, abs=1e-5)
    capped = tm.masked(make, None, tm.CONTEXT_NORM, ("/b1", "/b2"), max_share=0.3)(cfg)
    assert len(capped) - 10 <= 10 * 0.3 / 0.7 and capped.num_episodes < 6


def test_an_extra_root_with_other_features_is_refused(tmp_path):
    make, cfg = _mix_maker(tmp_path, other_features=True)
    with pytest.raises(ValueError, match="other features"):
        tm.masked(make, None, tm.CONTEXT_NORM, ("/b1",))(cfg)
    assert cfg.dataset.root == tmp_path


def test_without_an_extra_root_the_dataset_is_f9n_s(tmp_path):
    make, cfg = _mix_maker(tmp_path)
    data = tm.masked(make, None, tm.CONTEXT_NORM)(cfg)
    assert type(data) is tm.MaskedFrames and data.transform is tm.wrap_item_roll
    assert data.keep.tolist() == list(range(10)) and data.dataset.root == tmp_path


def test_the_mix_switches_are_checked_and_leave_lerobot_the_rest(monkeypatch):
    argv, untouched = _main(
        monkeypatch, ["--steps=5", "--extra_root=/b1", "--extra_root=/b2", "--extra_max_share=0.2"]
    )
    assert argv == ["--steps=5"] and not untouched
    for bad in (["--extra_max_share=0.3"], ["--extra_root=/b", "--extra_max_share=1"]):
        with pytest.raises(SystemExit):
            _main(monkeypatch, bad)


def test_on_converted_datasets_the_mix_keeps_whole_episodes_of_the_extra_root(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import reopen_mask as rm
    from test_reopen_mask import converted, corpus

    root = converted(tmp_path, corpus(tmp_path))

    def make(cfg):
        return LeRobotDataset(rm.REPO_ID, root=cfg.dataset.root)

    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root))
    whole = tm.masked(make, None, extras=(str(root),), max_share=0.6)(cfg)
    n = len(whole.datasets[0])
    assert len(whole) == 2 * n and whole.num_episodes == 2 * whole.datasets[0].num_episodes
    # Episodes of 68, 72 and 72 frames against a budget of 0.45 / 0.55 * 212 = 173 frames.
    capped = tm.masked(make, None, extras=(str(root),), max_share=0.45)(cfg)
    extra = capped.datasets[1]
    lengths = np.asarray(extra.meta.episodes["length"])
    kept = sorted({int(extra[i]["episode_index"]) for i in range(len(extra))})
    assert len(kept) == 2 and len(extra) == lengths[kept].sum() and capped.num_episodes == 5
