"""derive_vla_dataset.py (the pi0.5 dataset, MS7h design 7.1), in the LeRobot venv."""

import json
import os
from fractions import Fraction

import numpy as np
import pytest

import derive_vla_dataset as dv
from policy_server import wrap_context_roll

REPO = "youtalk/maker-arm-legosort"
STATE_NAMES = [f"link_00{k}_joint" for k in range(2, 8)] + ["gripper_open"]
ENV_NAMES = [
    "overhead_cam_x",
    "overhead_cam_y",
    "overhead_cam_z",
    "overhead_cam_roll",
    "overhead_cam_pitch",
    "overhead_cam_yaw",
    "box_x",
    "box_y",
    "box_yaw",
    "target_x",
    "target_y",
    "target_z",
    "target_cos",
    "target_sin",
    "target_width",
    "target_half_height",
]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": (16,), "names": ENV_NAMES},
    "action": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
    **{
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        }
        for cam in ("wrist", "overhead")
    },
}


def host_root(path, lengths, rng, labels=False):
    """A v3.0 root of len(lengths) episodes with random state, action and context; the camera
    roll on either side of +-pi, as rpy_from_quat returns it. ``labels``: also the int8 label
    and subtask columns of convert_to_lerobot.py --own_frames labeled, at random."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = dict(FEATURES)
    if labels:
        features |= {n: {"dtype": "int8", "shape": (1,), "names": [n]} for n in dv.LABELS}
    ds = LeRobotDataset.create(REPO, 30, features, root=path, use_videos=True)
    for n in lengths:
        for _ in range(n):
            env = rng.normal(size=16).astype(np.float32)
            env[3] = rng.choice([np.pi - 0.02, -np.pi + 0.03]) + rng.normal(scale=0.005)
            ds.add_frame(
                {
                    "observation.state": rng.normal(size=7).astype(np.float32),
                    "observation.environment_state": env,
                    "action": rng.normal(size=7).astype(np.float32),
                    "observation.images.wrist": rng.integers(0, 255, (64, 64, 3), np.uint8),
                    "observation.images.overhead": rng.integers(0, 255, (64, 64, 3), np.uint8),
                    "task": "sort the lego bricks into the color-matched bins",
                    **(
                        {
                            "label": rng.integers(0, 3, 1).astype(np.int8),
                            "subtask": rng.integers(0, 4, 1).astype(np.int8),
                        }
                        if labels
                        else {}
                    ),
                }
            )
        ds.save_episode()
    ds.finalize()
    return path


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """Two waves merged like merge_roots.sh: w1 = one host of 2 episodes, w2 = two hosts of 1
    and 2 episodes, plus the ledgers that name them."""
    from lerobot.datasets.aggregate import aggregate_datasets

    tmp, rng = tmp_path_factory.mktemp("corpus"), np.random.default_rng(7)
    hosts = [
        host_root(tmp / "w1/dataset_a", [5, 4], rng),
        host_root(tmp / "w2/dataset_b", [6], rng),
        host_root(tmp / "w2/dataset_ca", [3, 5], rng),
    ]
    out = tmp / "w12/dataset"
    aggregate_datasets([REPO] * 3, REPO, roots=hosts, aggr_root=out)
    roots = [str(h) for h in hosts]
    (tmp / "w12/dataset.ledger.json").write_text(json.dumps({"target": 5, "roots": roots}))
    return out


def frames(root):
    table = dv.read_tables(root / "data")
    return {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}


def test_waves_follow_the_ledgers(merged):
    plan = dv.waves(merged)
    assert [(w, list(n)) for w, n in plan] == [("w1", [5, 4]), ("w2", [6, 3, 5])]


def test_keep_mask_keeps_the_first_share_of_each_wave():
    assert dv.keep_mask([2, 3], Fraction(1, 3)).tolist() == [True, False, True, False, False]
    assert dv.keep_mask([2, 3], Fraction(1)).all()


@pytest.fixture(scope="module")
def derived(merged):
    out = merged.parent.parent / "vla/dataset"
    result = dv.derive(merged, out)
    assert result["episodes"] == 5 and result["frames"] == 23 and result["waves"] == "w1:2/2,w2:3/3"
    return out


def test_the_state_is_23_wide_in_order_and_the_context_leaves(merged, derived):
    info = json.loads((derived / "meta/info.json").read_text())
    names = STATE_NAMES + [n + ("_wrapped" if n.endswith("roll") else "") for n in ENV_NAMES]
    assert info["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": [23],
        "names": names,
    }
    assert names[10] == "overhead_cam_roll_wrapped"
    assert "observation.environment_state" not in info["features"]
    src, new = frames(merged), frames(derived)
    assert "observation.environment_state" not in new
    state = np.stack(new["observation.state"])
    np.testing.assert_array_equal(state[:, :7], np.stack(src["observation.state"]))
    context = np.stack(src["observation.environment_state"])
    np.testing.assert_array_equal(state[:, 7:], wrap_context_roll(context))
    assert np.abs(state[:, 10]).max() < 0.1  # the wrapped roll is a small angle
    np.testing.assert_array_equal(np.stack(new["action"]), np.stack(src["action"]))


def test_one_task_and_recomputed_state_and_action_stats(merged, derived):
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO, root=derived)
    assert list(ds.meta.tasks.index) == [dv.TASK]
    assert {ds[i]["task"] for i in range(len(ds))} == {dv.TASK}
    assert ds[0]["observation.state"].shape == (23,)
    assert ds[0]["observation.images.overhead"].shape == (3, 64, 64)
    for name in ("observation.state", "action"):
        column = np.stack(frames(derived)[name]).astype(np.float64)
        want = get_feature_stats(column, axis=0, keepdims=False)
        for key in ("min", "max", "mean", "std", "q01", "q99"):
            np.testing.assert_allclose(ds.meta.stats[name][key], want[key])
    assert (
        ds.meta.stats["observation.state"]["q01"][10]
        < ds.meta.stats["observation.state"]["q99"][10]
    )
    src = LeRobotDataset(REPO, root=merged)
    assert "observation.environment_state" not in ds.meta.stats
    for key in src.meta.stats["observation.images.wrist"]:
        np.testing.assert_array_equal(
            ds.meta.stats["observation.images.wrist"][key],
            src.meta.stats["observation.images.wrist"][key],
        )


def test_videos_are_hard_links(merged, derived):
    videos = sorted(p.relative_to(derived) for p in derived.glob("videos/*/*/*.mp4"))
    assert videos
    for rel in videos:
        assert os.path.samefile(derived / rel, merged / rel)


def test_a_fraction_keeps_the_first_episodes_of_each_wave(merged):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = merged.parent.parent / "vla/third"
    result = dv.derive(merged, out, Fraction(1, 3))
    assert result["waves"] == "w1:1/2,w2:1/3"  # source episodes 0 and 2, 5 and 6 frames
    ds = LeRobotDataset(REPO, root=out, delta_timestamps={"action": [k / 30 for k in range(4)]})
    assert ds.meta.total_episodes == 2 and len(ds) == 11
    assert list(ds.meta.episodes["dataset_from_index"]) == [0, 5]
    src = LeRobotDataset(REPO, root=merged)
    for i, j in [(0, 0), (4, 4), (5, 9), (10, 14)]:  # derived frame -> source frame
        a, b = ds[i], src[j]
        assert int(a["episode_index"]) == (i >= 5) and int(a["index"]) == i
        np.testing.assert_array_equal(a["observation.state"][:7], b["observation.state"])
        np.testing.assert_array_equal(a["action"][0], b["action"])
        np.testing.assert_allclose(a["observation.images.wrist"], b["observation.images.wrist"])


def test_an_existing_out_is_refused(merged, derived):
    with pytest.raises(FileExistsError):
        dv.derive(merged, derived)


def test_a_labeled_root_moves_its_labels_into_the_prompt_codes(tmp_path, derived):
    from lerobot.datasets.aggregate import aggregate_datasets

    from train_pi05 import CODES

    host = host_root(tmp_path / "lane0", [4, 3], np.random.default_rng(3), labels=True)
    src = tmp_path / "w9/own"  # merged as dagger_vla.py merges its lanes
    aggregate_datasets([REPO], REPO, roots=[host], aggr_root=src)
    out = tmp_path / "vla"
    assert dv.derive(src, out)["frames"] == 7
    info = json.loads((out / "meta/info.json").read_text())
    plain = json.loads((derived / "meta/info.json").read_text())
    assert info["features"].keys() == plain["features"].keys()  # the corpus's features
    old, new = frames(src), frames(out)
    assert not set(dv.LABELS) & set(new)
    want = 3 * old["subtask"] + old["label"]
    codes = np.load(out / CODES)
    assert codes.dtype == np.int8 and codes.tolist() == want.tolist()
    assert not set(dv.LABELS) & set(json.loads((out / "meta/stats.json").read_text()))
    episodes = dv.read_tables(out / "meta/episodes").column_names
    assert not [c for c in episodes if c.startswith(("stats/label/", "stats/subtask/"))]
    assert not (derived / CODES).exists()  # an unlabeled root gets no sidecar
