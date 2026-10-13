"""train_pi05.py: the pi0.5 loss ignores padded actions and equals the stock mean without them."""

from types import SimpleNamespace

import torch

from train_pi05 import forward, masked_means


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
