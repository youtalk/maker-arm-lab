import threading

import numpy as np
import pytest
import zmq

from policy_protocol import (
    ACTION_DTYPE,
    ACTION_SHAPE,
    OBS_SCHEMA,
    PROTOCOL_VERSION,
    STATE_GRIPPER_DIM,
    recv_message,
    send_message,
    validate_action,
    validate_obs,
)


def _obs(batch=2):
    return {
        "state": np.zeros((batch, 7), dtype=np.float32),
        "wrist": np.zeros((batch, 224, 224, 3), dtype=np.uint8),
        "overhead": np.zeros((batch, 224, 224, 3), dtype=np.uint8),
    }


def test_schema_constants_match_the_dataset_features():
    # Version 2, not the original 1: the state narrowed from 8 to 7 for the Maker Arm,
    # a wire-schema change an unbumped version would have let two peers disagree about.
    assert PROTOCOL_VERSION == 2
    assert OBS_SCHEMA["state"] == ((7,), "float32")
    assert OBS_SCHEMA["wrist"] == ((224, 224, 3), "uint8")
    assert ACTION_SHAPE == (7,) and ACTION_DTYPE == "float32"


def test_validate_obs_accepts_a_good_batch():
    validate_obs(_obs(batch=5), batch_size=5)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda o: o.pop("wrist"),
        lambda o: o.update(state=o["state"].astype(np.float64)),
        lambda o: o.update(overhead=o["overhead"][:, :100]),
        lambda o: o.update(extra=np.zeros(1)),
        # The 8-wide state of the openyam-lab original: right dtype, right keys, wrong arm.
        # Nothing else in this file fails if OBS_SCHEMA["state"] silently reverts, because
        # a (7,) action and a (7,) state are the same shape on this robot.
        lambda o: o.update(state=np.zeros((2, 8), dtype=np.float32)),
    ],
)
def test_validate_obs_rejects_and_names_the_problem(corrupt):
    obs = _obs()
    corrupt(obs)
    with pytest.raises(ValueError):
        validate_obs(obs, batch_size=2)


def test_validate_obs_accepts_the_whole_gripper_fraction_range():
    # Guards the discriminating check from being tightened into a false rejection: closed,
    # fully open and mid-travel are all legal states the sim really produces.
    obs = _obs(batch=3)
    obs["state"][:, STATE_GRIPPER_DIM] = [0.0, 1.0, 0.5]

    validate_obs(obs, batch_size=3)


def test_validate_obs_rejects_an_action_array_in_the_state_slot():
    """The swap a shape check cannot catch, now that state and action are both (7,).

    An action array has the same shape AND the same dtype as a state array on this arm, so
    the gripper column is the only thing left that tells them apart: a raw command goes
    negative to close, an opening fraction cannot.
    """
    obs = _obs()
    action = np.zeros((2, 7), dtype=np.float32)
    action[:, STATE_GRIPPER_DIM] = -1.0  # CLOSE, expressible as a command and not a fraction
    obs["state"] = action

    with pytest.raises(ValueError, match="gripper fraction"):
        validate_obs(obs, batch_size=2)


def test_validate_action_contract():
    validate_action(np.zeros((3, 7), dtype=np.float32), batch_size=3)
    with pytest.raises(ValueError):
        validate_action(np.zeros((3, 6), dtype=np.float32), batch_size=3)
    with pytest.raises(ValueError):
        validate_action(np.zeros((2, 7), dtype=np.float32), batch_size=3)


def test_round_trip_over_a_real_socket():
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1")
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://127.0.0.1:{port}")

    def echo():
        header, arrays = recv_message(rep)
        send_message(rep, {"echo": header}, arrays)

    t = threading.Thread(target=echo)
    t.start()
    sent = _obs()
    send_message(req, {"type": "step", "batch": 2}, sent)
    header, arrays = recv_message(req)
    t.join(timeout=5)
    assert header == {"echo": {"type": "step", "batch": 2}}
    for key, value in sent.items():
        np.testing.assert_array_equal(arrays[key], value)
        assert arrays[key].dtype == value.dtype
    req.close(0)
    rep.close(0)


def test_recv_message_rejects_malformed_frame_count():
    """Test that recv_message rejects messages with even frame count."""
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port("tcp://127.0.0.1")
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://127.0.0.1:{port}")

    def send_malformed():
        # Send a message with even frame count: header + dangling name (no npy data)
        req.send_multipart([b'{"type":"bad"}', b"name_only"])

    def recv_and_check():
        with pytest.raises(ValueError, match="malformed message"):
            recv_message(rep)
        # Send a valid reply to unblock the req socket
        rep.send(b'{"ok":true}')

    t = threading.Thread(target=recv_and_check)
    t.start()
    send_malformed()
    req.recv()  # Receive the unblocking reply
    t.join(timeout=5)
    req.close(0)
    rep.close(0)
