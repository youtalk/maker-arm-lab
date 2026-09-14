"""The label map of the held-or-not probe (MS4 c2 design 6.6)."""

import numpy as np

import probe_held


def test_label_frames_maps_phases_to_held_free_or_excluded():
    phase = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], np.int8)
    labels = probe_held.label_frames(phase)
    assert labels.tolist() == [-1, 0, 0, 0, -1, -1, 1, 1, 1, 0, -1, -1]
    assert labels.dtype == np.int8


def test_a_tiny_model_fits_a_separable_batch():
    import torch

    model = probe_held.HeldProbe()
    images = torch.zeros(16, 3, 64, 64)
    images[8:] = 1.0
    state = torch.zeros(16, 7)
    labels = torch.tensor([0] * 8 + [1] * 8)
    acc = probe_held.fit(model, images, state, labels, images, state, labels, epochs=30)
    assert acc == 1.0


class _FakeMeta:
    def __init__(self, total_episodes, total_frames):
        self.total_episodes = total_episodes
        self.total_frames = total_frames


class _FakeDataset:
    """Stands in for LeRobotDataset: item idx carries idx in every element of both fields,
    so a test can decode which global frame index load_lane actually fetched."""

    def __init__(self, repo_id, root, total_episodes, total_frames):
        self.meta = _FakeMeta(total_episodes, total_frames)

    def __getitem__(self, idx):
        import torch

        return {
            "observation.images.wrist": torch.full((3, 224, 224), float(idx)),
            "observation.state": torch.full((7,), float(idx)),
        }


def _write_sidecars(lane, phases_per_episode):
    sidecars = lane / "sidecars"
    sidecars.mkdir(parents=True)
    for ep, phases in enumerate(phases_per_episode):
        np.savez(sidecars / f"{ep}.npz", cycle_phase=np.array(phases, np.int8))


def test_load_lane_samples_only_kept_frames_with_correct_labels(monkeypatch, tmp_path):
    import torch

    phases_per_episode = [[1, 6, 2, 7], [4, 8, 9, 5]]  # 8 frames total, 6 kept (idx 4 excluded)
    lane = tmp_path / "lane_0"
    _write_sidecars(lane, phases_per_episode)
    total_frames = sum(len(p) for p in phases_per_episode)
    expected_labels = probe_held.label_frames(
        np.concatenate([np.array(p, np.int8) for p in phases_per_episode])
    )

    def fake_ctor(repo_id, root):
        return _FakeDataset(repo_id, root, len(phases_per_episode), total_frames)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", fake_ctor)

    images, state, labels = probe_held.load_lane(lane, frames=100, rng=np.random.default_rng(0))

    kept = int((expected_labels >= 0).sum())
    # avg_pool2d(kernel=224 // 64=3) on a 224x224 source lands at 74x74, not exactly 64x64;
    # HeldProbe's AdaptiveAvgPool2d(4) absorbs that, so only shape[:2] is asserted here.
    assert images.shape[:2] == (kept, 3)
    assert state.shape == (kept, 7)
    assert labels.shape == (kept,)

    picked_indices = state[:, 0].long().tolist()
    assert len(set(picked_indices)) == kept  # sampled without replacement
    for row, idx in enumerate(picked_indices):
        assert expected_labels[idx] != -1  # only kept (held/free) frames were sampled
        assert labels[row].item() == expected_labels[idx]
        assert torch.all(state[row] == idx)
        assert torch.all(images[row] == idx)  # avg_pool2d of a constant field is unchanged


def test_load_lane_caps_the_sample_at_frames_per_lane(monkeypatch, tmp_path):
    phases_per_episode = [[1, 6, 2, 7, 3, 9]]  # all 6 frames kept
    lane = tmp_path / "lane_0"
    _write_sidecars(lane, phases_per_episode)
    total_frames = sum(len(p) for p in phases_per_episode)

    def fake_ctor(repo_id, root):
        return _FakeDataset(repo_id, root, len(phases_per_episode), total_frames)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", fake_ctor)

    images, _, _ = probe_held.load_lane(lane, frames=2, rng=np.random.default_rng(0))
    assert images.shape[0] == 2


def test_load_lane_raises_when_sidecar_and_dataset_frame_counts_disagree(monkeypatch, tmp_path):
    phases_per_episode = [[1, 6, 2, 7]]
    lane = tmp_path / "lane_0"
    _write_sidecars(lane, phases_per_episode)

    def fake_ctor(repo_id, root):
        return _FakeDataset(repo_id, root, len(phases_per_episode), total_frames=999)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", fake_ctor)

    try:
        probe_held.load_lane(lane, frames=100, rng=np.random.default_rng(0))
        raise AssertionError("expected RuntimeError on a sidecar/dataset frame count mismatch")
    except RuntimeError as err:
        assert "999" in str(err)
