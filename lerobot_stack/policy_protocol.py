"""ZMQ wire protocol for the ACT policy server.

numpy + stdlib + pyzmq ONLY: this module is imported by both sides of the
two-venv boundary -- lerobot_stack/policy_server.py (REP) in the LeRobot venv,
and the Isaac-venv eval client (REQ), which appends lerobot_stack/ to sys.path.
That client is a later milestone's; the constraint is recorded now so nothing
adds a lerobot or torch import here in the meantime and breaks it.

A message is a zmq multipart: frame 0 is a UTF-8 JSON header; the remaining
frames alternate (array name, npy bytes), each array encoded via np.save
with allow_pickle=False (np.load never unpickles -> no code execution from
the wire). Schemas are per-sample; a leading batch dimension is implied.

PROTOCOL_VERSION 2 (was 1 in the openyam-lab original): the observation state
narrowed from 8 to the Maker Arm's 7. A version-1 peer's messages still frame
correctly but carry the wrong state width, so the version is what rejects them --
the handshake fails loudly instead of the shape error surfacing inside the policy.
"""

import io
import json

import numpy as np

PROTOCOL_VERSION = 2
OBS_SCHEMA = {
    # (7,) = the dataset's observation.state: 6 arm joint angles + the gripper
    # opening fraction (spec section 4.2), which is what convert_to_lerobot.py writes
    # and therefore what the trained policy expects. tests/test_lerobot_schema_pin.py
    # asserts this exact literal from the other venv, where it cannot import this module.
    "state": ((7,), "float32"),
    "wrist": ((224, 224, 3), "uint8"),
    "overhead": ((224, 224, 3), "uint8"),
}
# Also 7, but a different 7: 6 joint TARGETS + the raw binary gripper command. The two
# widths coincide on this arm and must still be validated against their own constants.
ACTION_SHAPE = (7,)
ACTION_DTYPE = "float32"


def _pack(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _unpack(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def send_message(socket, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    frames = [json.dumps(header).encode()]
    for name, array in (arrays or {}).items():
        frames.append(name.encode())
        frames.append(_pack(np.ascontiguousarray(array)))
    socket.send_multipart(frames)


def recv_message(socket) -> tuple[dict, dict[str, np.ndarray]]:
    frames = socket.recv_multipart()
    header = json.loads(frames[0].decode())
    if len(frames) % 2 != 1:
        raise ValueError(f"malformed message: {len(frames)} frames (want 1 + 2 per array)")
    arrays = {}
    for i in range(1, len(frames), 2):
        arrays[frames[i].decode()] = _unpack(frames[i + 1])
    return header, arrays


def _check(name: str, array: np.ndarray, sample_shape: tuple, dtype: str, batch_size: int) -> None:
    want = (batch_size, *sample_shape)
    if tuple(array.shape) != want:
        raise ValueError(f"{name}: shape {tuple(array.shape)}, want {want}")
    if str(array.dtype) != dtype:
        raise ValueError(f"{name}: dtype {array.dtype}, want {dtype}")


def validate_obs(arrays: dict, batch_size: int) -> None:
    if set(arrays) != set(OBS_SCHEMA):
        raise ValueError(f"obs keys {sorted(arrays)}, want {sorted(OBS_SCHEMA)}")
    for name, (sample_shape, dtype) in OBS_SCHEMA.items():
        _check(name, arrays[name], sample_shape, dtype, batch_size)


def validate_action(array: np.ndarray, batch_size: int) -> None:
    _check("action", array, ACTION_SHAPE, ACTION_DTYPE, batch_size)
