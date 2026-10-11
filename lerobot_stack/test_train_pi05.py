"""train_pi05.py: the pi0.5 loss ignores padded actions and equals the stock mean without them;
the DAgger mix draws its share and keeps the corpus metadata."""

from types import SimpleNamespace

import pytest
import torch

from train_pi05 import Mixed, forward, masked_means, mixed_maker, split_mix


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
    def __init__(self, name, n, features=("state",)):
        self.name, self.n, self.num_episodes, self.fps = name, n, 2, 30
        self.meta = SimpleNamespace(features=list(features), stats=name)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"from": self.name, "index": i}


def test_the_mix_draws_the_dagger_share_and_keeps_the_corpus_meta():
    mix = Mixed(Frames("corpus", 700), Frames("dagger", 40), 0.3)
    assert (len(mix), mix.extra, mix.num_frames, mix.num_episodes) == (1000, 300, 1000, 4)
    assert mix.meta.stats == "corpus"
    assert mix[699] == {"from": "corpus", "index": 699}
    assert mix[700] == mix[740] == {"from": "dagger", "index": 0}  # the DAgger frames cycle
    torch.manual_seed(0)
    draws = [mix[int(i)]["from"] for i in torch.utils.data.RandomSampler(mix)]
    assert draws.count("dagger") == 300  # one epoch of the shuffled sampler
    with pytest.raises(ValueError, match="must be in"):
        Mixed(Frames("corpus", 7), Frames("dagger", 4), 1.0)


def test_the_mix_builds_both_roots_and_refuses_other_features():
    roots = []

    def make(cfg):
        roots.append(cfg.dataset.root)
        name = "corpus" if cfg.dataset.root == "/c" else "dagger"
        return Frames(name, 70, ("state", "extra") if cfg.bad and name == "dagger" else ("state",))

    cfg = SimpleNamespace(dataset=SimpleNamespace(root="/c"), bad=False)
    mix = mixed_maker(make, "/d", 0.3)(cfg)
    assert roots == ["/c", "/d"] and cfg.dataset.root == "/c" and mix.extra == 30
    cfg.bad = True
    with pytest.raises(ValueError, match="other features"):
        mixed_maker(make, "/d", 0.3)(cfg)
    assert cfg.dataset.root == "/c"


def test_the_mix_flags_leave_lerobot_s_argv():
    argv = ["--policy.type=pi05", "--dagger_root=/d", "--steps=8", "--dagger_share=0.25"]
    assert split_mix(argv) == ("/d", 0.25, ["--policy.type=pi05", "--steps=8"])
    assert split_mix(argv[:3]) == ("/d", 0.3, ["--policy.type=pi05", "--steps=8"])
    assert split_mix(["--steps=8"]) == (None, 0.3, ["--steps=8"])  # off: argv unchanged
    with pytest.raises(SystemExit, match="needs --dagger_root"):
        split_mix(["--dagger_share=0.3"])
