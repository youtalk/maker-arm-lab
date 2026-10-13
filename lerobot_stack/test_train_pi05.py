"""train_pi05.py: the pi0.5 loss ignores padded actions and equals the stock mean without them;
the mix draws each source's share and keeps the corpus metadata; Run A's prompt codes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_pi05 import (
    CODES,
    PHRASES,
    TASK,
    Mixed,
    forward,
    masked_means,
    mixed_maker,
    prompt,
    split_mix,
)


def test_masked_means_skip_padding():
    losses = torch.ones(2, 4, 3)
    losses[1, 2:] = 100.0  # the padded tail of sample 1
    is_pad = torch.zeros(2, 4, dtype=torch.bool)
    is_pad[1, 2:] = True
    loss, per_sample, per_dim = masked_means(losses, is_pad)
    assert loss.item() == 1.0
    assert per_sample.tolist() == [1.0, 1.0]
    assert per_dim.tolist() == [1.0, 1.0, 1.0]


def test_masked_means_without_padding_is_the_plain_mean():
    losses = torch.rand(3, 5, 7)
    loss, per_sample, per_dim = masked_means(losses, torch.zeros(3, 5, dtype=torch.bool))
    assert torch.allclose(loss, losses.mean())
    assert torch.allclose(per_sample, losses.mean((1, 2)))
    assert torch.allclose(per_dim, losses.mean((0, 1)))


def test_forward_masks_the_model_losses():
    # 7 action dims of a 32-wide padded model output; the stub model returns the losses.
    out = torch.ones(2, 4, 32)
    out[0, 3] = 50.0
    policy = SimpleNamespace(
        _preprocess_images=lambda batch: (None, None),
        prepare_action=lambda batch: batch["action"],
        model=SimpleNamespace(forward=lambda *args: out),
        config=SimpleNamespace(output_features={"action": SimpleNamespace(shape=(7,))}),
    )
    batch = {
        "action": torch.zeros(2, 4, 7),
        "action_is_pad": torch.tensor([[False, False, False, True], [False] * 4]),
        "observation.language.tokens": None,
        "observation.language.attention_mask": None,
    }
    loss, loss_dict = forward(policy, batch)
    assert loss.item() == 1.0 and loss_dict["loss"] == 1.0
    per_sample, _ = forward(policy, batch, reduction="none")
    assert per_sample.tolist() == [1.0, 1.0]


class Frames(torch.utils.data.Dataset):
    def __init__(self, name, n, features=("state",), root=None):
        self.name, self.n, self.num_episodes, self.fps, self.root = name, n, 2, 30, root
        self.num_frames = n
        self.meta = SimpleNamespace(features=list(features), stats=name)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"from": self.name, "index": torch.tensor(i), "task": TASK}


def test_three_sources_draw_their_exact_shares_and_keep_the_corpus_meta():
    corpus, own, take = Frames("corpus", 500), Frames("own", 40), Frames("take", 4000)
    mix = Mixed(corpus, [(own, 0.3), (take, 0.2)])
    assert (len(mix), mix.sizes, mix.num_frames, mix.num_episodes) == (
        1000,
        [500, 300, 200],
        1000,
        6,
    )
    assert mix.meta.stats == "corpus"
    assert mix[499]["from"] == "corpus" and int(mix[499]["index"]) == 499
    assert int(mix[500]["index"]) == 0 and int(mix[507]["index"]) == 0  # 40 own frames spread
    assert int(mix[508]["index"]) == 1 and int(mix[799]["index"]) == 39
    assert [int(mix[800 + i]["index"]) for i in (0, 1, 199)] == [0, 20, 3980]  # every 20th
    torch.manual_seed(0)
    draws = [mix[int(i)]["from"] for i in torch.utils.data.RandomSampler(mix)]
    assert [draws.count(s) for s in ("corpus", "own", "take")] == [500, 300, 200]  # one epoch
    one = Mixed(Frames("corpus", 700), [(Frames("dagger", 40), 0.3)])  # HD59's two-source mix
    assert one.sizes == [700, 300] and one[0]["task"] == TASK  # no codes: the dataset's own task
    for bad in ([], [(own, 0.0)], [(own, 0.6), (take, 0.4)], [(own, -0.1)]):
        with pytest.raises(ValueError, match="mix shares"):
            Mixed(corpus, bad)


def test_the_prompt_table_gives_twelve_strings():
    want = "pick up the marked part and place it in its box. close on the part."
    assert prompt(3) == want and prompt(4) == want + " Advantage: positive"
    assert prompt(5) == want + " Advantage: negative" and prompt(5, drop=True) == want
    assert len({prompt(c) for c in range(3 * len(PHRASES))}) == 12
    assert prompt(9 + 1).endswith("release it. Advantage: positive")


def test_a_tagged_frame_drops_its_advantage_sentence_at_the_drop_rate():
    corpus, own = Frames("corpus", 100), Frames("own", 100)
    codes = [np.full(100, 4, np.int8), np.arange(100, dtype=np.int8) % 3]  # own: 0, 1, 2, 0, ...
    mix = Mixed(corpus, [(own, 0.5)], codes)
    torch.manual_seed(0)
    tasks = [mix[i % 100]["task"] for i in range(10_000)]
    assert set(tasks) == {prompt(4), prompt(3, drop=True)}
    dropped = tasks.count(prompt(3, drop=True)) / len(tasks)
    assert abs(dropped - 0.2) < 3 * (0.2 * 0.8 / len(tasks)) ** 0.5  # 3 SE of 10,000 draws
    untagged = {mix[100 + i]["task"] for i in range(0, 100, 3) for _ in range(20)}
    assert untagged == {prompt(0)}  # never changes
    tagged = {mix[100 + i]["task"] for i in range(2, 100, 3) for _ in range(50)}
    assert tagged == {prompt(2), prompt(0)}  # negative, or dropped to the untagged string


def test_the_mix_builds_every_root_and_refuses_other_features_and_bad_codes(tmp_path):
    roots = []

    def make(cfg):
        roots.append(cfg.dataset.root)
        name = "corpus" if cfg.dataset.root == str(tmp_path / "c") else "extra"
        bad = cfg.bad and cfg.dataset.root.endswith("e")
        return Frames(name, 70, ("state", "extra") if bad else ("state",), cfg.dataset.root)

    for root in ("c", "d", "e"):
        (tmp_path / root / "meta").mkdir(parents=True)
        np.save(tmp_path / root / CODES, np.full(70, 4, np.int8))
    mix = [(str(tmp_path / "d"), 0.3), (str(tmp_path / "e"), 0.2)]
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=str(tmp_path / "c")), bad=False)
    out = mixed_maker(make, mix, False)(cfg)
    assert roots == [str(tmp_path / r) for r in "cde"] and cfg.dataset.root == roots[0]
    assert out.sizes == [70, 42, 28] and out.codes is None
    out = mixed_maker(make, mix, True)(cfg)
    assert [c.tolist() for c in out.codes] == [[4] * 70] * 3
    np.save(tmp_path / "e" / CODES, np.full(69, 4, np.int8))  # a frame short
    with pytest.raises(ValueError, match=r"want int8 \(70,\)"):
        mixed_maker(make, mix, True)(cfg)
    np.save(tmp_path / "e" / CODES, np.full(70, 12, np.int8))  # no subtask 4
    with pytest.raises(ValueError, match="codes span"):
        mixed_maker(make, mix, True)(cfg)
    cfg.bad = True
    with pytest.raises(ValueError, match="other features"):
        mixed_maker(make, mix, False)(cfg)
    assert cfg.dataset.root == str(tmp_path / "c")


def test_the_mix_flags_leave_lerobot_s_argv():
    argv = ["--policy.type=pi05", "--mix=/d:0.3", "--steps=8", "--mix=/e:0.2", "--prompt_codes"]
    assert split_mix(argv) == (
        [("/d", 0.3), ("/e", 0.2)],
        True,
        ["--policy.type=pi05", "--steps=8"],
    )
    assert split_mix(argv[:3]) == ([("/d", 0.3)], False, ["--policy.type=pi05", "--steps=8"])
    assert split_mix(["--steps=8"]) == ([], False, ["--steps=8"])  # off: argv unchanged
    with pytest.raises(SystemExit, match="needs --mix"):
        split_mix(["--prompt_codes"])
    with pytest.raises(SystemExit, match="want --mix=ROOT:SHARE"):
        split_mix(["--mix=0.3"])
