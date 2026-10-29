"""train_pi05.py: the pi0.5 loss ignores padded actions and equals the stock mean without them;
the mix draws each source's share and keeps the corpus metadata; Run A's prompt codes; the MS7k
levers (a LoRA continuation, close-weighted sampling, the overhead affine) and main without them."""

import copy
import json
import math
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train_pi05 import (
    CODES,
    JAW_FULL_GAP,
    LORA_TARGETS,
    OVERHEAD,
    PHRASES,
    SIGLIP_TARGETS,
    TASK,
    Mixed,
    close_sampling,
    close_weights,
    continued,
    forward,
    load_lora,
    lora_flags,
    main,
    masked_means,
    mixed_maker,
    overhead_aug,
    prompt,
    random_affine,
    siglip_checkpointing,
    split_levers,
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
        model=SimpleNamespace(
            forward=lambda *args: out,
            sample_noise=lambda shape, device: None,
            sample_time=lambda n, device: None,
        ),
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
    own_slot = [int(mix[500 + j]["index"]) for j in range(300)]
    assert own_slot[:41] == [*range(40), 0] and set(own_slot) == set(range(40))  # all, cycled
    take_slot = [int(mix[800 + j]["index"]) for j in range(200)]  # 200 of 4000 frames
    assert take_slot == np.random.default_rng(0).permutation(4000)[:200].tolist()
    assert len(set(take_slot)) == 200 and min(take_slot) < 400 and max(take_slot) > 3600
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


def test_forward_draws_noise_then_time_and_passes_both():
    calls, got = [], {}
    noise, time = torch.zeros(2, 4, 7), torch.ones(2)

    def sample_noise(shape, device):
        calls.append("noise")
        return noise

    def sample_time(n, device):
        calls.append("time")
        return time

    def model_forward(*args):
        calls.append("forward")
        got["noise"], got["time"] = args[-2:]
        return torch.ones(2, 4, 32)

    policy = SimpleNamespace(
        _preprocess_images=lambda batch: (None, None),
        prepare_action=lambda batch: batch["action"],
        model=SimpleNamespace(
            forward=model_forward, sample_noise=sample_noise, sample_time=sample_time
        ),
        config=SimpleNamespace(output_features={"action": SimpleNamespace(shape=(7,))}),
    )
    batch = {
        "action": torch.zeros(2, 4, 7),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "observation.language.tokens": None,
        "observation.language.attention_mask": None,
    }
    forward(policy, batch)
    assert calls == ["noise", "time", "forward"]
    assert got["noise"] is noise and got["time"] is time


def test_the_lora_regex_takes_the_language_model_and_the_expert_only():
    hit = "model.paligemma_with_expert.paligemma.model.language_model.layers.3.mlp.up_proj"
    exp = "model.paligemma_with_expert.gemma_expert.model.layers.17.self_attn.o_proj"
    sig = (
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        ".encoder.layers.0.self_attn.q_proj"
    )
    assert re.fullmatch(LORA_TARGETS, hit) and re.fullmatch(LORA_TARGETS, exp)
    assert not re.fullmatch(LORA_TARGETS, sig)
    assert not re.fullmatch(LORA_TARGETS, "model.action_in_proj")


def test_lora_flags_set_alpha_to_the_rank():
    flags = lora_flags(128)
    assert "--peft.r=128" in flags and "--peft.lora_alpha=128" in flags
    assert "--peft.method_type=LORA" in flags
    assert (
        '--peft.full_training_modules=["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]'
        in flags
    )


def test_mix_and_prompt_codes_is_refused_on_0_6_1(monkeypatch):
    import lerobot

    monkeypatch.setattr(lerobot, "__version__", "0.6.1")
    for flag in ("--mix=/x:0.2", "--prompt_codes"):
        with pytest.raises(SystemExit, match="EpisodeAwareSampler"):
            main([flag, "--steps=8"])


# --- the MS7k levers (spec section 6.2, ledger K11) ---

SIG = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers."


def test_the_siglip_regex_takes_every_siglip_linear_and_keeps_the_old_targets():
    lm = "model.paligemma_with_expert.paligemma.model.language_model.layers.3.mlp.up_proj"
    wide = lora_flags(128, siglip=True)[3].split("=", 1)[1]
    for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"):
        assert re.fullmatch(SIGLIP_TARGETS, f"{SIG}26.{m}") and re.fullmatch(wide, f"{SIG}0.{m}")
    for m in ("mlp.fc1", "mlp.fc2"):
        assert re.fullmatch(wide, f"{SIG}11.{m}")
    assert re.fullmatch(wide, lm) and not re.fullmatch(SIGLIP_TARGETS, lm)
    for other in (
        "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear",
        f"{SIG}0.layer_norm1",
        "model.action_in_proj",
    ):
        assert not re.fullmatch(wide, other)
    assert lora_flags(128, siglip=False) == lora_flags(128)  # off: the B3 flags


class Tiny(torch.nn.Module):
    """lm stands for the language model and the expert, vision for SigLIP, head for the
    modules_to_save of the recipe."""

    def __init__(self):
        super().__init__()
        self.lm, self.vision, self.head = (torch.nn.Linear(4, 4) for _ in range(3))

    def forward(self, x):
        return self.head(self.lm(torch.relu(self.vision(x))))


def test_a_continued_adapter_is_the_saved_model_at_step_0(tmp_path, capsys):
    peft = pytest.importorskip("peft")

    def wrap(base, targets=r"lm", alpha=2, save=("head",)):
        cfg = peft.LoraConfig(
            r=2, lora_alpha=alpha, target_modules=targets, modules_to_save=list(save)
        )
        return peft.get_peft_model(copy.deepcopy(base), cfg)

    torch.manual_seed(0)
    base, x = Tiny(), torch.randn(5, 4)
    saved = wrap(base)
    with torch.no_grad():  # a trained adapter: LoRA A, B and the head copy all moved
        for p in (p for p in saved.parameters() if p.requires_grad):
            p.add_(torch.randn_like(p))
    saved.save_pretrained(tmp_path)
    want = saved(x)
    head = saved.base_model.model.head.modules_to_save["default"].weight
    assert not torch.equal(head, base.head.weight)

    fresh = continued(lambda self, *a, **k: wrap(self), tmp_path, None)(base)  # the run's wrap
    assert torch.equal(fresh(x), want)
    assert torch.equal(fresh.base_model.model.head.modules_to_save["default"].weight, head)
    assert "loaded=4 new=0 trainable=4" in capsys.readouterr().out
    assert not torch.equal(wrap(base)(x), want)  # without the load, another model

    wide = wrap(base, targets=r"lm|vision")  # L2: the new vision LoRA keeps B = 0
    load_lora(wide, tmp_path, new=r"\.vision\.")
    assert torch.equal(wide(x), want)
    assert "loaded=4 new=2 trainable=6" in capsys.readouterr().out
    assert not wide.base_model.model.vision.lora_B["default"].weight.any()

    with pytest.raises(ValueError, match="2 adapter tensors did not load"):
        load_lora(wrap(base, targets=r"lm|vision"), tmp_path)  # new targets need --siglip_lora
    with pytest.raises(ValueError, match="2 saved tensors found no place"):
        load_lora(wrap(base, targets=r"vision"), tmp_path, new=r"\.vision\.")  # no lm LoRA
    with pytest.raises(ValueError, match=r"\(2, 2, \['head'\]\), the run has \(2, 4, "):
        load_lora(wrap(base, alpha=4), tmp_path)
    with pytest.raises(ValueError, match=r"the run has \(2, 2, \[\]\)"):
        load_lora(wrap(base, save=()), tmp_path)  # the head would load into the frozen base


def test_siglip_checkpointing_checkpoints_each_layer_and_keeps_the_lora_gradients():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    cfg = transformers.SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    lora = peft.LoraConfig(r=2, target_modules=SIGLIP_TARGETS, init_lora_weights=False)  # B != 0
    img = torch.randn(3, 3, 8, 8)

    def lora_grads(layered: bool) -> dict:
        torch.manual_seed(0)
        root = tower = transformers.SiglipVisionModel(cfg)
        for name in ("vision_tower", "model", "paligemma", "paligemma_with_expert", "model"):
            parent = torch.nn.Module()  # the real module path, so SIGLIP_TARGETS finds them
            parent.add_module(name, root)
            root = parent
        peft.inject_adapter_in_model(lora, root).train()
        if layered:
            siglip_checkpointing(lambda self: None)(root.model)
            assert all(layer.gradient_checkpointing for layer in tower.vision_model.encoder.layers)
        embed = lambda x: tower(pixel_values=x).last_hidden_state
        out = torch.utils.checkpoint.checkpoint(  # lerobot's checkpoint of embed_image
            embed, img, use_reentrant=False, preserve_rng_state=False
        )
        out.square().sum().backward()
        return {n: p.grad for n, p in root.named_parameters() if p.requires_grad}

    plain, layered = lora_grads(False), lora_grads(True)
    assert len(plain) == 2 * 6 * 2 and plain.keys() == layered.keys()  # A, B of 6 linears a layer
    for name, grad in layered.items():
        assert grad.abs().sum() > 0 and torch.allclose(grad, plain[name], atol=1e-6)


def test_close_weights_mark_the_window_around_each_close_in_its_episode():
    pick, shut, full = 0.024, 0.010, JAW_FULL_GAP
    gap = [full] * 5 + [pick] * 10 + [shut] * 10 + [pick] * 2 + [full] * 3  # episode 0: 30
    gap += [shut] * 10 + [full] * 5 + [pick] * 3 + [shut] * 2  # episode 1 starts closed: 20
    gap += [pick] * 2 + [shut] * 8  # episode 2 closes at its frame 2: 10
    episode = [0] * 30 + [1] * 20 + [2] * 10
    w, closes, none = close_weights(gap, episode, fps=10, weight=3.0, before_s=1.0, after_s=0.5)
    near = np.flatnonzero(w == 3.0).tolist()
    assert closes == 3 and none == 0  # the open-to-pick falls at frames 5 and 45 are no closes
    assert near == [*range(5, 21), *range(38, 50), *range(50, 58)]  # clipped to each episode
    assert set(w.tolist()) == {1.0, 3.0}
    w, closes, none = close_weights(gap[30:50], episode[30:50], 10, 3.0, 1.0, 0.5)
    assert (closes, none) == (1, 0)
    _, closes, none = close_weights([shut] * 5 + [full] * 5, [7] * 10, 10, 3.0, 1.0, 0.5)
    assert (closes, none) == (0, 1)  # an episode that starts closed shows no close


def test_close_sampling_reads_the_corpus_and_draws_near_a_close_w_times_as_often(tmp_path, capsys):
    import pyarrow as pa
    import pyarrow.parquet as pq

    gap = np.tile(np.r_[[JAW_FULL_GAP] * 20, [0.024] * 20, [0.010] * 60], 4)  # 4 episodes
    action = np.zeros((len(gap), 7), np.float32)
    action[:, 6] = gap
    names = [f"j{i}" for i in range(6)] + ["gripper_gap_m"]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": {"action": {"names": names}}}))
    for k, part in enumerate((slice(0, 150), slice(150, 400))):  # two files, in index order
        (tmp_path / f"data/chunk-00{k}").mkdir(parents=True)
        table = pa.table(
            {
                "index": np.arange(400)[part],
                "episode_index": np.repeat(np.arange(4), 100)[part],
                "action": pa.array(list(action[part])),
            }
        )
        pq.write_table(table, tmp_path / f"data/chunk-00{k}/file-000.parquet")

    def corpus(n):  # a LeRobotDataset of n frames at the root
        return type("Corpus", (), {"root": tmp_path, "fps": 10, "__len__": lambda self: n})()

    dataset = corpus(400)
    module = SimpleNamespace(make_train_eval_datasets=lambda cfg: (dataset, None))
    close_sampling(module, 3.0, 1.0, 0.5)
    assert module.make_train_eval_datasets("cfg") == (dataset, None)
    out = capsys.readouterr().out
    assert "CLOSE_WEIGHT closes=4 episodes_without=0 frames=400 window_share=0.1600" in out
    # each close at frame 40 of its episode: frames 30 to 45 weigh 3, 48 of 132 weight units
    assert "weighted_share=0.3636" in out
    draws = list(module.EpisodeAwareSampler([0], [400], shuffle=True, seed=5))
    assert len(draws) == 400 and draws == list(module.EpisodeAwareSampler(seed=5))
    many = np.concatenate([list(module.EpisodeAwareSampler(seed=s)) for s in range(50)])
    share = np.isin(many % 100, np.arange(30, 46)).mean()
    assert abs(share - 48 / 132) < 4 * (0.25 / len(many)) ** 0.5
    module.make_train_eval_datasets = lambda cfg: (corpus(300), None)  # a subset of the root
    close_sampling(module, 3.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="400 weights for 300 frames"):
        module.make_train_eval_datasets("cfg")


def ramps(b, h=40, w=64):
    """(b, 2, h, w): channel 0 holds each pixel's column, channel 1 its row, in pixels."""
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([cols, rows]).to(torch.float64).expand(b, 2, h, w).clone()


def test_random_affine_shifts_and_rolls_within_its_bounds():
    gen = torch.Generator().manual_seed(0)
    img = ramps(4000)
    assert torch.equal(random_affine(img, 0.0, 0.06, 3.0, gen), img)  # P = 0: unchanged
    out = random_affine(img, 0.5, 0.06, 0.0, gen)
    assert out.shape == img.shape
    moved = (out != img).flatten(1).any(1)
    assert abs(moved.double().mean() - 0.5) < 0.03  # half of the samples
    centre = out[:, :, 20, 32] - img[:, :, 20, 32]  # where the centre pixel samples from
    dx, dy = centre[:, 0] / 64, centre[:, 1] / 40  # a share of the frame size
    assert dx.abs().max() <= 0.06 + 1e-6 and dy.abs().max() <= 0.06 + 1e-6
    assert dx[moved].abs().max() > 0.059 and dy[moved].abs().mean() == pytest.approx(0.03, 0.1)
    out = random_affine(img, 1.0, 0.0, 3.0, gen)  # roll only, on a frame that is not square
    centre = out[:, :, 20, 32] - img[:, :, 20, 32]
    assert centre.abs().max() < 0.05  # pixel (20, 32) is 0.7 px from the centre (19.5, 31.5)
    x, y = out[:, 0, 20, 52] - 31.5, out[:, 1, 20, 52] - 19.5  # pixel (20, 52) is at (20.5, 0.5)
    angle = torch.rad2deg(torch.atan2(y, x)) - math.degrees(math.atan2(0.5, 20.5))
    assert angle.abs().max() <= 3.0 + 1e-6 and angle.abs().max() > 2.95
    assert torch.allclose(torch.hypot(x, y), torch.full_like(x, math.hypot(20.5, 0.5)))  # no shear


def test_overhead_aug_moves_only_the_overhead_image_and_only_in_training():
    seen = []

    def stock(self, batch, reduction="mean"):
        seen.append(batch)
        return reduction

    wrist = torch.rand(8, 3, 16, 16)
    batch = {OVERHEAD: ramps(8, 16, 16).float(), "observation.images.wrist": wrist}
    policy = SimpleNamespace(training=True)
    torch.manual_seed(7)
    state = torch.get_rng_state()
    aug = overhead_aug(stock, 1.0, 0.06, 3.0)
    assert aug(policy, batch, "none") == "none"
    assert torch.equal(torch.get_rng_state(), state)  # no draw from the global generator
    assert seen[-1]["observation.images.wrist"] is wrist
    assert not torch.equal(seen[-1][OVERHEAD], batch[OVERHEAD])  # the caller's batch stays
    want = random_affine(batch[OVERHEAD], 1.0, 0.06, 3.0, torch.Generator().manual_seed(7))
    assert torch.equal(seen[-1][OVERHEAD], want)  # its own generator, seeded with --seed
    torch.manual_seed(8)
    overhead_aug(stock, 1.0, 0.06, 3.0)(policy, batch)
    assert not torch.equal(seen[-1][OVERHEAD], want)  # another --seed, another affine
    policy.training = False
    aug(policy, batch)
    assert seen[-1] is batch


def test_the_levers_leave_lerobot_s_argv_and_bad_values_exit(monkeypatch):
    import lerobot

    argv = ["--steps=8", "--lora_from=/a", "--siglip_lora", "--close_weight=3,2,1", "--lora=128"]
    assert split_levers(argv) == (
        {"--lora_from": "/a", "--siglip_lora": "", "--close_weight": "3,2,1"},
        ["--steps=8", "--lora=128"],
    )
    assert split_levers(["--steps=8"]) == ({}, ["--steps=8"])
    monkeypatch.setattr(lerobot, "__version__", "0.6.1")
    for bad, match in (
        (["--close_weight=3,2"], "want three numbers"),
        (["--overhead_aug=1.5,0.06,3"], "want three numbers"),
        (["--siglip_lora=1", "--lora=128"], "takes no value"),
        (["--lora_from="], "needs one"),
        (["--lora_from=/a"], "need --lora=R"),
        (["--siglip_lora"], "need --lora=R"),
        (["--lora_from=/a", "--lora=128", "--policy.use_peft=true"], "no --policy.use_peft"),
        (["--lora_from=/a", "--lora=128", "--resume=true"], "--resume"),
        (["--close_weight=3,2,1", "--resume=true"], "cannot resume"),
    ):
        with pytest.raises(SystemExit, match=match):
            main(bad)


def test_main_installs_a_lever_only_when_its_flag_is_given(monkeypatch):
    import lerobot

    if lerobot.__version__ != "0.6.1":
        pytest.skip("needs the pi venv (lerobot 0.6.1)")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
    from lerobot.scripts import lerobot_train

    names = ("forward", "wrap_with_peft")
    stock = {n: getattr(PI05Policy, n) for n in names}
    for n in names:
        monkeypatch.setattr(PI05Policy, n, stock[n])
    checkpointing = PI05Pytorch.gradient_checkpointing_enable
    monkeypatch.setattr(PI05Pytorch, "gradient_checkpointing_enable", checkpointing)
    for n in ("EpisodeAwareSampler", "make_train_eval_datasets", "main"):
        monkeypatch.setattr(lerobot_train, n, getattr(lerobot_train, n))
    sampler, make = lerobot_train.EpisodeAwareSampler, lerobot_train.make_train_eval_datasets
    runs = []
    monkeypatch.setattr(lerobot_train, "main", lambda: runs.append(list(sys.argv[1:])))
    monkeypatch.setattr(sys, "argv", ["train_pi05.py"])
    plain = ["--steps=8", "--lora=128", "--seed=1001"]
    main(plain)  # K14: the masked forward and the B3 flags, nothing else
    assert PI05Policy.forward is forward and PI05Policy.wrap_with_peft is stock["wrap_with_peft"]
    assert PI05Pytorch.gradient_checkpointing_enable is checkpointing
    assert lerobot_train.EpisodeAwareSampler is sampler
    assert lerobot_train.make_train_eval_datasets is make
    assert runs[-1] == ["--steps=8", *lora_flags(128), "--seed=1001"]
    levers = [
        "--lora_from=/a",
        "--siglip_lora",
        "--close_weight=3,2,1",
        "--overhead_aug=0.5,0.06,3",
    ]
    main([*plain, *levers])
    assert runs[-1] == ["--steps=8", *lora_flags(128, siglip=True), "--seed=1001"]
    assert PI05Policy.forward.__qualname__.startswith("overhead_aug")
    assert PI05Policy.wrap_with_peft.__qualname__.startswith("continued")
    assert PI05Pytorch.gradient_checkpointing_enable.__qualname__.startswith("siglip_checkpoint")
    assert lerobot_train.make_train_eval_datasets.__qualname__.startswith("close_sampling")
