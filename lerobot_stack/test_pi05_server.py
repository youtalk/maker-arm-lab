"""pi05_server.py (the shared pi0.5 server, MS7h design 7.3) with a fake model, CPU only."""

import json
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zmq

import pi05_server as ps
from derive_vla_dataset import derived_state
from policy_client import ServerPolicy

N = 50
RNG = np.random.default_rng(0)


class FakeModel:
    """chunk[r, t] = (state[r, 0], t, 0, ...): the lane marker and the time index."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, obs):
        self.calls.append(obs)
        if self.fail:
            raise RuntimeError("forward failed")
        chunk = np.zeros((len(obs["state"]), N, 7), np.float32)
        chunk[:, :, 0] = obs["state"][:, :1]
        chunk[:, :, 1] = np.arange(N)
        return chunk


def server(model=None, warp=1.0):
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    return ps.SharedServer(model or FakeModel(), "pi05:/ckpt", N, inference, "continuous", warp)


def step(marker, batch=1, context=True, replan=None, rng=RNG, prompt=None):
    state = rng.uniform(0, 1, (batch, 7)).astype(np.float32)
    state[:, 0] = marker
    arrays = {
        "state": state,
        "wrist": rng.integers(0, 255, (batch, 224, 224, 3), np.uint8),
        "overhead": rng.integers(0, 255, (batch, 224, 224, 3), np.uint8),
    }
    if context:
        ctx = rng.normal(size=(batch, 16)).astype(np.float32)
        ctx[:, 3] = np.pi - 0.01  # the overhead camera's roll sits near +-pi
        arrays["context"] = ctx
    if replan is not None:
        arrays["replan"] = np.asarray(replan, dtype=bool)
    if prompt is not None:
        arrays["prompt"] = np.asarray(prompt, dtype=np.int16)
    return {"type": "step", "batch": batch}, arrays


def test_the_state_is_23_wide_with_the_roll_wrapped():
    model = FakeModel()
    header, arrays = step(0.5)
    replies, _ = server(model).process([(b"a", header, arrays)])
    assert replies[0][1]["type"] == "action"
    sent = model.calls[0]
    np.testing.assert_array_equal(sent["state"], derived_state(arrays["state"], arrays["context"]))
    assert sent["state"].shape == (1, 23)
    assert abs(sent["state"][0, 10] - (-0.01)) < 1e-6  # wrap(roll - pi)
    np.testing.assert_array_equal(sent["overhead"], arrays["overhead"])


def test_lanes_with_empty_queues_share_one_forward_and_full_ones_wait_for_none():
    model, s = FakeModel(), server()
    s._predict = model
    replies, _ = s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2]
    assert [r[2]["action"][0, 0] for r in replies] == pytest.approx([0.1, 0.2])
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2]  # both queues still hold actions


def test_each_lane_plays_its_warped_chunk_then_refills():
    s = server(warp=1.5)
    times = [s.process([(b"a", *step(0.3))])[0][0][2]["action"][0, 1] for _ in range(34)]
    assert times[:3] == pytest.approx([0.0, 1.5, 3.0])
    assert len(s.forward_sizes) == 2 and times[33] == 0.0  # 50 actions at 1.5 play 33 steps


def test_replan_refills_only_the_flagged_row():
    s = server()
    s.process([(b"a", *step(0.4, batch=2))])
    reply = s.process([(b"a", *step(0.4, batch=2, replan=[False, True]))])[0][0]
    assert s.forward_sizes == [2, 1]
    assert reply[2]["action"][:, 1].tolist() == [1.0, 0.0]


def test_the_task_of_a_row_is_run_a_s_serving_prompt_of_its_id():
    from train_pi05 import TASK

    assert ps.tasks(None, 2) == [TASK] * 2
    assert ps.tasks(np.array([0, 3], np.int16), 2) == [
        f"{TASK}. reach the marked part. Advantage: positive",
        f"{TASK}. release it. Advantage: positive",
    ]
    assert ps.tasks(np.array([-1, 2], np.int16), 2)[0] == TASK  # a lane without the key


def test_the_prompt_ids_reach_the_forward_only_at_a_refill():
    model = FakeModel()
    s = server(model)
    s.process([(b"a", *step(0.4, batch=2, prompt=[0, 1]))])
    assert (
        model.calls[0]["prompt"].tolist() == [0, 1] and model.calls[0]["prompt"].dtype == np.int16
    )
    s.process([(b"a", *step(0.4, batch=2, prompt=[1, 2]))])  # a new id: no forward
    assert s.forward_sizes == [2]
    reply = s.process([(b"a", *step(0.4, batch=2, replan=[False, True], prompt=[1, 3]))])[0][0]
    assert s.forward_sizes == [2, 1] and model.calls[1]["prompt"].tolist() == [3]
    assert reply[2]["action"][:, 1].tolist() == [2.0, 0.0]  # row 0 plays on, row 1 refilled
    s.process([(b"b", *step(0.2)), (b"c", *step(0.3, prompt=[2]))])  # a lane without the key
    assert model.calls[2]["prompt"].tolist() == [-1, 2]
    s.process([(b"d", *step(0.1))])
    assert "prompt" not in model.calls[3]  # no lane sends it: the obs of before


def test_a_latched_chunk_plays_on_across_an_id_change_and_a_bad_id_is_refused():
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    s = ps.SharedServer(LatchModel(), "pi05:/ckpt", N, inference, "continuous", 1.5, None, 10)
    header, arrays = step(0.5, prompt=[1])
    arrays["state"][:, 6], arrays["context"][:, 14] = 1.0, 0.02  # open jaws, a 2 cm target
    served = []
    for t in range(33):
        arrays["prompt"][:] = 1 if t < 8 else 2  # grasp, then carry in the middle of the chunk
        served.append(s.process([(b"a", header, arrays)])[0][0][2]["action"][0])
    state = derived_state(arrays["state"], arrays["context"])
    latched = ps.latch_jaw(LatchModel()({"state": state}), state, 10)[0]
    np.testing.assert_array_equal(np.stack(served), ps.warp_chunk(latched, 1.5)[0])
    assert s.forward_sizes == [1]
    for bad in ([4], [-1]):
        replies, _ = s.process([(b"a", *step(0.5, prompt=bad))])
        assert replies[0][1]["error"] == "bad_obs" and "prompt ids" in replies[0][1]["message"]


def test_reset_clears_only_its_lane():
    s = server()
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    replies, _ = s.process([(b"a", {"type": "reset"}, {})])
    assert replies[0][1]["type"] == "reset_ack"
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2, 1]


def test_a_bad_request_fails_alone_and_a_failed_forward_reaches_its_lanes():
    model = FakeModel()
    s = server(model)
    replies, _ = s.process([(b"a", *step(0.1, context=False)), (b"b", *step(0.2))])
    assert replies[0][1]["error"] == "bad_obs" and "context width 0" in replies[0][1]["message"]
    assert replies[1][1]["type"] == "action"
    model.fail = True
    replies, keep = s.process([(b"c", *step(0.3))])
    assert replies[0][1]["error"] == "exception" and keep


def still_step(moves, jaws):
    """A 4-row step whose model chunk moves row r's arm moves[r] rad off its state, jaw state
    jaws[r]; the move rides in context column 0 (derived state column 7)."""
    header, arrays = step(0.5, batch=len(moves))
    arrays["state"][:, 6] = jaws
    arrays["context"][:, 0] = moves
    return header, arrays


class StillModel:
    """The first forward holds each row moves[r] off its joints (jaw command 0.05 m, far from
    the jaw state); a resample (``scale`` given) returns 100 + scale everywhere."""

    def __init__(self):
        self.calls = []

    def __call__(self, obs, *scale):
        self.calls.append((obs, *scale))
        if scale:
            return np.full((len(obs["state"]), N, 7), 100 + scale[0], np.float32)
        chunk = np.zeros((len(obs["state"]), N, 7), np.float32)
        chunk[:, :, :6] = obs["state"][:, None, :6] + obs["state"][:, None, 7:8]
        chunk[:, :, 6] = 0.05
        return chunk


def test_the_still_rows_alone_are_resampled_in_one_forward_at_their_noise_scale(capsys):
    model = StillModel()
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    s = ps.SharedServer(model, "pi05:/ckpt", N, inference, "continuous", 1.0, (0.5, 0.15, 0.9))
    # moving and open, still and open, still and closed (a close), still and open
    header, arrays = still_step([0.3, 0.1, 0.1, -0.1], [1.0, 0.95, 0.5, 0.9])
    action = s.process([(b"a", header, arrays)])[0][0][2]["action"]
    (first,), (resample, scale) = model.calls
    assert scale == 0.5 and len(first["state"]) == 4
    for key in first:  # one forward over rows 1 and 3 only
        np.testing.assert_array_equal(resample[key], first[key][[1, 3]])
    assert (action[[1, 3]] == 100.5).all()
    np.testing.assert_allclose(action[[0, 2], :6], arrays["state"][[0, 2], :6] + [[0.3], [0.1]])
    assert re.fullmatch(r"FORWARD rows=4 ms=\d+\.\d still=2", capsys.readouterr().out.strip())
    assert s.ready(0)["inference"] == {
        **inference,
        "action_warp": 1.0,
        "still_noise_scale": 0.5,
        "still_rad": 0.15,
        "still_open": 0.9,
    }
    s.process([(b"b", *still_step([0.3, 0.3, 0.1, 0.2], [1.0, 1.0, 0.0, 1.0]))])
    assert len(model.calls) == 3  # no still row, no resample
    assert capsys.readouterr().out.strip().endswith(" still=0")


def test_the_still_rule_is_off_by_default(capsys):
    s = server()  # FakeModel takes the obs alone, so a resample would fail the forward
    replies, _ = s.process([(b"a", *still_step([0.0], [1.0]))])
    assert replies[0][1]["type"] == "action"
    assert re.fullmatch(r"FORWARD rows=1 ms=\d+\.\d", capsys.readouterr().out.strip())
    assert not [k for k in s.ready(0)["inference"] if k.startswith("still")]


def test_a_retry_row_draws_its_arm_noise_at_the_retry_scale_in_a_forward_of_its_own(capsys):
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    still = (0.5, 0.15, 0.9)
    # moving, still, moving and flagged, still and flagged; every queue is empty
    header, arrays = still_step([0.3, 0.1, 0.3, 0.1], [1.0] * 4)
    flagged = arrays | {"retry": np.array([False, False, True, True])}
    for scale in (0.0, 1.0):
        model, off_model = StillModel(), StillModel()
        on = ps.SharedServer(
            model, "pi05:/ckpt", N, inference, "continuous", 1.0, still, retry_noise_scale=scale
        )
        off = ps.SharedServer(off_model, "pi05:/ckpt", N, inference, "continuous", 1.0, still)
        assert on.ready(0)["inference"]["retry_noise_scale"] == scale
        assert "retry_noise_scale" not in off.ready(0)["inference"]
        served = [s.process([(b"a", header, flagged)])[0][0][2]["action"] for s in (on, off)]
        (retried, *at), (rest, *none), (resample, *again) = model.calls
        full = off_model.calls[0][0]  # off: the old forward over all four rows
        assert (at, none, again) == ([scale], [], [0.5])
        for key in full:
            np.testing.assert_array_equal(retried[key], full[key][[2, 3]])
            np.testing.assert_array_equal(rest[key], full[key][[0, 1]])
        np.testing.assert_array_equal(resample["state"], full["state"][[1]])  # still, unflagged
        assert (served[0][[2, 3]] == 100 + scale).all() and (served[0][1] == 100.5).all()
        np.testing.assert_allclose(served[0][0, :6], arrays["state"][0, :6] + 0.3)
        assert [call[1:] for call in off_model.calls] == [(), (0.5,)]  # off: as before
        assert len(off_model.calls[1][0]["state"]) == 2 and (served[1][[1, 3]] == 100.5).all()
        assert (served[1][[0, 2]] < 100).all()
        logs = capsys.readouterr().out.strip().splitlines()
        assert [line.split(" ", 3)[3] for line in logs] == ["retry=2 still=1", "still=2"]
    # On, a step without the key or with every flag False is the old forward; every row flagged
    # runs no forward at the configured noise.
    for extra in ({}, {"retry": np.zeros(4, bool)}):
        model = StillModel()
        s = ps.SharedServer(model, "pi05:/ckpt", N, inference, "continuous", retry_noise_scale=1.0)
        s.process([(b"a", header, arrays | extra)])
        assert [call[1:] for call in model.calls] == [()] and len(model.calls[0][0]["state"]) == 4
        assert capsys.readouterr().out.strip().endswith(" retry=0")
    model = StillModel()
    s = ps.SharedServer(model, "pi05:/ckpt", N, inference, "continuous", retry_noise_scale=0.0)
    s.process([(b"a", header, arrays | {"retry": np.ones(4, bool)})])
    assert [(len(call[0]["state"]), *call[1:]) for call in model.calls] == [(4, 0.0)]
    assert capsys.readouterr().out.strip().endswith(" retry=4")


def test_hello_ack_and_shutdown():
    replies, keep = server(warp=1.5).process([(b"a", {"type": "hello", "protocol": 4}, {})])
    ack = replies[0][1]
    assert ack["policy_id"] == "pi05:/ckpt" and ack["context_dim"] == 16
    assert ack["inference"]["action_warp"] == 1.5 and ack["gripper"] == "continuous"
    replies, keep = server().process([(b"a", {"type": "shutdown"}, {})])
    assert replies[0][1]["type"] == "shutdown_ack" and not keep


def test_unchanged_clients_share_the_server_over_a_router_socket():
    s = server(warp=1.5)
    socket = zmq.Context.instance().socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    thread = threading.Thread(target=s.serve, args=(socket,), daemon=True)
    thread.start()
    results = {}

    def lane(marker):
        client = ServerPolicy(f"tcp://127.0.0.1:{port}", recv_timeout_ms=10_000)
        assert client.context_dim == 16 and client.policy_id == "pi05:/ckpt"
        actions = []
        for _ in range(40):
            _, arrays = step(marker)
            actions.append(
                client.step(
                    arrays["state"], arrays["wrist"], arrays["overhead"], context=arrays["context"]
                )
            )
        results[marker] = np.concatenate(actions)

    lanes = [threading.Thread(target=lane, args=(m,)) for m in (0.1, 0.2, 0.3)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join(timeout=30)
    for marker, actions in results.items():
        assert actions.shape == (40, 7)
        np.testing.assert_allclose(actions[:, 0], marker, rtol=1e-6)  # never another lane's
        assert actions[:3, 1].tolist() == [0.0, 1.5, 3.0]
    assert len(results) == 3 and sum(s.forward_sizes) == 6  # 40 steps = 2 forwards a lane
    stop = zmq.Context.instance().socket(zmq.REQ)
    stop.connect(f"tcp://127.0.0.1:{port}")
    stop.send_multipart(ps.encode_frames({"type": "shutdown"}))
    stop.recv_multipart()
    thread.join(timeout=10)
    assert not thread.is_alive()


def latch_state(fractions, widths):
    """A derived (n, 23) state with the jaw state's opening fraction and the target width."""
    state = np.zeros((len(fractions), 23), np.float32)
    state[:, 6], state[:, ps.TARGET_WIDTH] = fractions, widths
    return state


def jaw_chunk(*rows):
    """An (n, N, 7) chunk with the arm at its time index and each row's jaw commands."""
    chunk = np.zeros((len(rows), N, 7), np.float32)
    chunk[:, :, :6] = np.arange(N)[:, None]
    chunk[:, :, ps.JAW] = rows
    return chunk


def test_the_latch_holds_the_jaw_after_a_close_and_leaves_a_release_free():
    inside = [0.05] * 10 + [0.01] + [0.05] * 39  # a close at t = 10, then a re-open
    at_start = [0.01] + [0.08] * 49  # a close at t = 0 from an open jaw state
    release = [0.01] * 5 + [0.08] * 45  # from a closed jaw state: no close
    chunk = jaw_chunk(inside, at_start, release, inside)
    state = latch_state([1.0, 1.0, 0.1, 1.0], [0.02, 0.02, 0.02, 0.0])  # row 3: no target
    latched, changed = ps.latch_jaw(chunk, state, 5)
    assert changed == 2 and ps.JAW_FULL_GAP * 0.1 < 0.02
    np.testing.assert_array_equal(latched[:, :, :6], chunk[:, :, :6])  # the arm untouched
    assert latched[0, :, ps.JAW].tolist() == pytest.approx([0.05] * 10 + [0.01] * 6 + [0.05] * 34)
    assert latched[1, :, ps.JAW].tolist() == pytest.approx([0.01] * 6 + [0.08] * 44)
    np.testing.assert_array_equal(latched[2:], chunk[2:])
    assert ps.latch_jaw(chunk, state, 0)[1] == 0


def test_the_full_gap_is_twice_the_converter_travel():
    from convert_to_lerobot import GRIPPER_TRAVEL

    assert ps.JAW_FULL_GAP == 2 * GRIPPER_TRAVEL


class LatchModel:
    """Every row closes at t = 10 and re-opens after it."""

    def __call__(self, obs):
        return jaw_chunk(*[[0.05] * 10 + [0.01] + [0.05] * 39] * len(obs["state"]))


def test_the_server_latches_each_chunk_before_the_warp(capsys):
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    s = ps.SharedServer(LatchModel(), "pi05:/ckpt", N, inference, "continuous", 1.5, None, 10)
    off = ps.SharedServer(LatchModel(), "pi05:/ckpt", N, inference, "continuous", 1.5)
    assert s.ready(0)["inference"]["jaw_latch"] == 10
    assert "jaw_latch" not in off.ready(0)["inference"]
    header, arrays = step(0.5)
    arrays["state"][:, 6], arrays["context"][:, 14] = 1.0, 0.02  # open jaws, a 2 cm target
    served = [s.process([(b"a", header, arrays)])[0][0][2]["action"][0] for _ in range(34)]
    unlatched = [off.process([(b"a", header, arrays)])[0][0][2]["action"][0] for _ in range(34)]
    state = derived_state(arrays["state"], arrays["context"])
    latched = ps.latch_jaw(LatchModel()({"state": state}), state, 10)[0]
    want = ps.warp_chunk(latched, 1.5)[0]
    np.testing.assert_array_equal(np.stack(served[:33]), want)
    np.testing.assert_array_equal(
        np.stack(unlatched[:33]), ps.warp_chunk(LatchModel()({"state": state}), 1.5)[0]
    )
    assert not np.array_equal(np.stack(unlatched[:33]), want)
    forward = capsys.readouterr().out.strip().splitlines()
    assert re.fullmatch(r"FORWARD rows=1 ms=\d+\.\d latched=1", forward[0])
    assert re.fullmatch(r"FORWARD rows=1 ms=\d+\.\d", forward[2])  # off: as before


class StandInPolicy:
    """PI05Policy's surface that load_pi05 uses; records what reaches predict_action_chunk."""

    config = SimpleNamespace(
        image_features={v: None for v in ps.IMAGE_KEYS.values()},
        robot_state_feature=SimpleNamespace(shape=(23,)),
        output_features={"action": SimpleNamespace(shape=(7,))},
        n_action_steps=N,
        chunk_size=N,
        max_action_dim=32,
        num_inference_steps=10,
    )

    def __init__(self):
        self.kwargs = []
        self.model = SimpleNamespace(sample_noise=lambda shape, device: torch.ones(shape))

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_action_chunk(self, batch, **kwargs):
        self.kwargs.append(kwargs)
        self.tasks = batch["task"]
        return torch.zeros(len(batch["task"]), N, 7)


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """load_pi05 on a checkpoint dir with its marker and tokenizer step, lerobot and
    transformers replaced, so the sampling path runs on the CPU in this venv."""
    policy = StandInPolicy()
    modules = {
        "lerobot.policies.pi05.modeling_pi05": SimpleNamespace(
            PI05Policy=SimpleNamespace(from_pretrained=lambda path: policy)
        ),
        "lerobot.policies.factory": SimpleNamespace(
            make_pre_post_processors=lambda **kw: (lambda b: b, lambda c: c)
        ),
        "transformers": SimpleNamespace(
            GemmaTokenizer=SimpleNamespace(from_pretrained=lambda name: None)
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "action_space.json").write_text(
        json.dumps({"action_space": "absolute", "gripper": "continuous"})
    )
    steps = [{"registry_name": ps.TOKENIZER, "config": {"tokenizer_name": "paligemma"}}]
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps({"steps": steps}))
    return policy, str(tmp_path)


def forward(predict_chunk, rows=2):
    _, arrays = step(0.5, batch=rows)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    return predict_chunk({**obs, "wrist": arrays["wrist"], "overhead": arrays["overhead"]})


def test_the_default_sampling_passes_nothing_to_the_model(stand_in):
    policy, ckpt = stand_in
    for num_steps in (None, 10):  # the checkpoint's own steps, named or not
        predict_chunk, n, inference, gripper, sampling = ps.load_pi05(ckpt, "cpu", num_steps)
        assert forward(predict_chunk).shape == (2, N, 7)
        assert policy.kwargs.pop() == {} and sampling == ""
        assert (n, gripper) == (N, "continuous")
        assert inference == {  # the block every pi05 read recorded before the switches
            "n_action_steps": N,
            "temporal_ensemble_coeff": None,
            "chunk_size": N,
            "num_inference_steps": 10,
        }


def test_the_model_reads_each_row_s_task_from_its_id(stand_in):
    policy, ckpt = stand_in
    predict_chunk, *_ = ps.load_pi05(ckpt, "cpu")
    _, arrays = step(0.5, batch=2)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    obs |= {"wrist": arrays["wrist"], "overhead": arrays["overhead"]}
    predict_chunk(obs)
    assert policy.tasks == [ps.TASK] * 2
    predict_chunk({**obs, "prompt": np.array([2, 0], np.int16)})
    assert policy.tasks == ps.tasks(np.array([2, 0]), 2) and "carry it" in policy.tasks[0]


def test_the_sampling_switches_reach_sample_actions_and_name_the_policy(stand_in):
    policy, ckpt = stand_in
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", 50, 0.0)
    forward(predict_chunk, rows=3)
    sent = policy.kwargs.pop()
    assert sent.keys() == {"num_steps", "noise"} and sent["num_steps"] == 50
    assert sent["noise"].shape == (3, N, 32) and not sent["noise"].any()  # zero noise
    assert (inference["num_inference_steps"], inference["noise_scale"]) == (50, 0.0)
    assert sampling == "+steps50+noise0.0"
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", None, 0.5)
    forward(predict_chunk)
    noise = policy.kwargs.pop()["noise"]
    assert noise.dtype == torch.float32 and torch.equal(noise, torch.full((2, N, 32), 0.5))
    assert sampling == "+noise0.5" and inference["num_inference_steps"] == 10
    # A still row's resample: its own scale, and at 1.0 sample_actions' own draw.
    predict_chunk, *_ = ps.load_pi05(ckpt, "cpu", None, 0.0)
    _, arrays = step(0.5, batch=2)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    obs |= {"wrist": arrays["wrist"], "overhead": arrays["overhead"]}
    predict_chunk(obs, 0.7)
    assert torch.equal(policy.kwargs.pop()["noise"], torch.full((2, N, 32), 0.7))
    predict_chunk(obs, 1.0)
    assert policy.kwargs.pop() == {}


def test_the_jaw_noise_scale_scales_only_the_jaw_column(stand_in):
    policy, ckpt = stand_in
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", None, 1.0, 1.0)
    forward(predict_chunk)
    assert policy.kwargs.pop() == {} and sampling == "" and "jaw_noise_scale" not in inference
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", None, 0.5, 0.25)
    assert sampling == "+noise0.5+jawnoise0.25" and inference["jaw_noise_scale"] == 0.25
    _, arrays = step(0.5, batch=2)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    obs |= {"wrist": arrays["wrist"], "overhead": arrays["overhead"]}
    for scale in (None, 0.7, 1.0):  # the forward's own scale, then two still resamples
        predict_chunk(obs) if scale is None else predict_chunk(obs, scale)
        want = torch.full((2, N, 32), 0.5 if scale is None else scale)
        want[..., ps.JAW] *= 0.25
        assert torch.equal(policy.kwargs.pop()["noise"], want)


def test_the_absolute_jaw_noise_keeps_its_std_at_every_scale(stand_in):
    policy, ckpt = stand_in
    generator, draws = torch.Generator().manual_seed(0), []

    def sample_noise(shape, device):
        draws.append(torch.randn(shape, generator=generator))
        return draws[-1]

    policy.model.sample_noise = sample_noise
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", None, 0.0, 1.0, False, 1.0)
    assert sampling == "+noise0.0+jawabs1.0" and inference["jaw_noise_abs"] == 1.0
    _, arrays = step(0.5, batch=2)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    obs |= {"wrist": arrays["wrist"], "overhead": arrays["overhead"]}
    arm = torch.ones(32, dtype=torch.bool)
    arm[ps.JAW] = False
    for scale in (None, 1.0):  # the forward's zero noise, then a still resample at 1.0
        predict_chunk(obs) if scale is None else predict_chunk(obs, scale)
        sent = policy.kwargs.pop()["noise"]  # the explicit noise, even at scale 1.0
        assert torch.equal(sent[..., arm], (scale or 0.0) * draws[-1][..., arm])
        assert torch.equal(sent[..., ps.JAW], draws[-1][..., ps.JAW]) and sent[..., ps.JAW].any()
    predict_chunk, *_, sampling = ps.load_pi05(ckpt, "cpu", None, 1.0, 1.0, False, 0.5)
    predict_chunk(obs)
    sent = policy.kwargs.pop()["noise"]
    assert sampling == "+jawabs0.5" and torch.equal(sent[..., ps.JAW], 0.5 * draws[-1][..., ps.JAW])


def test_the_shared_jaw_draw_holds_one_value_a_row_over_the_chunk(stand_in):
    policy, ckpt = stand_in
    generator, draws = torch.Generator().manual_seed(0), []

    def sample_noise(shape, device):
        draws.append(torch.randn(shape, generator=generator))
        return draws[-1]

    policy.model.sample_noise = sample_noise
    predict_chunk, _, inference, _, sampling = ps.load_pi05(ckpt, "cpu", None, 1.0, 1.0, True)
    forward(predict_chunk)
    assert "noise" in policy.kwargs.pop() and sampling == "+jawshared"
    assert inference["jaw_noise_shared"] is True and "jaw_noise_scale" not in inference
    predict_chunk, *_, sampling = ps.load_pi05(ckpt, "cpu", None, 0.5, 0.25, True)
    assert sampling == "+noise0.5+jawnoise0.25+jawshared"
    _, arrays = step(0.5, batch=3)
    obs = {"state": derived_state(arrays["state"], arrays["context"])}
    obs |= {"wrist": arrays["wrist"], "overhead": arrays["overhead"]}
    for scale in (None, 0.7, 1.0):  # the forward's own scale, then two still resamples
        predict_chunk(obs) if scale is None else predict_chunk(obs, scale)
        s = 0.5 if scale is None else scale
        sent, (full, jaw) = policy.kwargs.pop()["noise"], draws[-2:]
        arm = torch.ones(32, dtype=torch.bool)
        arm[ps.JAW] = False
        assert torch.equal(sent[..., arm], s * full[..., arm])  # the arm's own draw
        assert jaw.shape == (3, 1) and torch.equal(sent[..., ps.JAW], (s * jaw * 0.25).expand(3, N))
        assert len(set(sent[:, 0, ps.JAW].tolist())) == 3  # one value a row, not one a chunk


def test_the_jaw_is_the_last_of_seven_actions(stand_in, monkeypatch):
    policy, ckpt = stand_in
    assert ps.JAW == 6
    monkeypatch.setitem(policy.config.output_features, "action", SimpleNamespace(shape=(8,)))
    with pytest.raises(ValueError, match=r"an action of \(8,\)"):
        ps.load_pi05(ckpt, "cpu")


STILL_ON = ["--still_noise_scale", "0.7", "--still_rad", "0.1", "--still_open", "0.8"]


@pytest.mark.parametrize(
    "still, tail, block",
    [
        ([], "", {}),
        (["--still_rad", "0.1"], "", {}),  # E alone: the rule stays off
        (["--still_noise_scale", "1"], "+still1.0r0.15o0.9", [1.0, 0.15, 0.9]),
        (STILL_ON, "+still0.7r0.1o0.8", [0.7, 0.1, 0.8]),
        (["--jaw_noise_scale", "0", *STILL_ON], "+jawnoise0.0+still0.7r0.1o0.8", [0.7, 0.1, 0.8]),
        (
            ["--jaw_noise_scale", "0", "--jaw_noise_shared", *STILL_ON],
            "+jawnoise0.0+jawshared+still0.7r0.1o0.8",
            [0.7, 0.1, 0.8],
        ),
        (["--jaw_latch", "10", *STILL_ON], "+still0.7r0.1o0.8+latch10", [0.7, 0.1, 0.8]),
        (["--jaw_latch", "0"], "", {}),
        (["--jaw_noise_abs", "1", *STILL_ON], "+jawabs1.0+still0.7r0.1o0.8", [0.7, 0.1, 0.8]),
        (["--retry_noise_scale", "1"], "+retry1.0", {}),
        (
            ["--jaw_noise_abs", "1", "--retry_noise_scale", "1", *STILL_ON, "--jaw_latch", "10"],
            "+jawabs1.0+retry1.0+still0.7r0.1o0.8+latch10",
            [0.7, 0.1, 0.8],
        ),
    ],
)
def test_the_policy_id_names_each_switch_off_its_default(
    stand_in, monkeypatch, capsys, still, tail, block
):
    _, ckpt = stand_in
    monkeypatch.setattr(ps.SharedServer, "serve", lambda self, socket: None)
    with socket.socket() as probe:  # a free port for the bind
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    argv = ["pi05_server.py", "--checkpoint", ckpt, "--device", "cpu", "--port", str(port)]
    argv += ["--action_warp", "1.5", "--num_steps", "50", "--noise_scale", "0", *still]
    monkeypatch.setattr(sys, "argv", argv)
    assert ps.main() == 0
    ready = json.loads(capsys.readouterr().out.split("SERVER_READY ", 1)[1].splitlines()[0])
    assert ready["policy_id"] == f"pi05:{ckpt}+warp1.5+steps50+noise0.0{tail}"
    assert ready["inference"]["num_inference_steps"] == 50
    assert ready["inference"]["noise_scale"] == 0.0
    assert ready["inference"].get("jaw_noise_scale") == (0.0 if "+jawnoise" in tail else None)
    assert ready["inference"].get("jaw_noise_shared") == (True if "+jawshared" in tail else None)
    assert ready["inference"].get("jaw_noise_abs") == (1.0 if "+jawabs" in tail else None)
    assert ready["inference"].get("jaw_latch") == (10 if "+latch" in tail else None)
    assert ready["inference"].get("retry_noise_scale") == (1.0 if "+retry" in tail else None)
    names = ("still_noise_scale", "still_rad", "still_open")
    assert {k: v for k, v in ready["inference"].items() if k in names} == dict(zip(names, block))


@pytest.mark.parametrize(
    "flags",
    [
        ["--num_steps", "0"],
        ["--noise_scale", "-0.1"],
        ["--noise_scale", "nan"],
        ["--jaw_noise_scale", "-0.1"],
        ["--jaw_noise_scale", "inf"],
        ["--jaw_noise_abs", "-1"],
        ["--jaw_noise_abs", "nan"],
        ["--jaw_noise_abs", "1", "--jaw_noise_scale", "0.5"],
        ["--jaw_noise_abs", "1", "--jaw_noise_shared"],
        ["--jaw_latch", "-1"],
        ["--still_noise_scale", "-1"],
        ["--still_rad", "0"],
        ["--still_open", "1.5"],
        ["--retry_noise_scale", "-1"],
        ["--retry_noise_scale", "nan"],
    ],
)
def test_a_bad_sampling_value_fails_with_the_token(flags):
    here = Path(ps.__file__).parent
    result = subprocess.run(
        [sys.executable, "pi05_server.py", "--checkpoint", "/none", *flags],
        cwd=here,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("SERVER_FAIL usage: ") and flags[0] in line
