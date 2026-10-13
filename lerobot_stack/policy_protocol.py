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

PROTOCOL_VERSION 3: an optional `context` array (batch, K) float32 rides beside the
three observation arrays. It carries `observation.environment_state` for checkpoints
that take one. a version-2 peer never sends it and a version-3 server with
`context_dim` 0 never expects it.

PROTOCOL_VERSION 4: an optional `anchor` array (batch, 6) float32 rides beside the
observation. It is the client's previous clamped arm command (the measured joints before
step 0), and a `delta_prev` checkpoint's server adds it to the six joint columns of the
action. An absolute checkpoint's server ignores it.

Still PROTOCOL_VERSION 4: an optional `replan` array (batch,) bool rides beside the
observation (MS6 `--replan_on_switch`). A server keeps one action queue per env and refills
env i's queue from a fresh forward when the flag is set, so a new target row reaches the policy
on the step it is written and not 0 to n_action_steps - 1 steps later. Absent means all False
and the queues drain in lockstep, which is the single-queue behaviour. A version-4 server that
predates the key rejects it as bad_obs, so no version bump is needed to keep the mismatch loud.

Still PROTOCOL_VERSION 4: `policy_ms` (the server's time in the policy call, ms) joined the
action reply header without a bump; a client that predates it ignores the key.

Still PROTOCOL_VERSION 4: an optional `retry` array (batch,) bool rides beside the observation
(MS7h `--retry_noise_at`). The heap loop sets env i's flag while its cycle is in retry mode, and a
pi0.5 server with `--retry_noise_scale` forwards that row at its retry noise. Absent means all
False. The ACT server validates it and ignores it.

Still PROTOCOL_VERSION 4: an optional `prompt` array (batch,) int16 rides beside the observation
(MS7i `--pi05_prompt`). It holds env i's subtask id (core.hold.SUBTASKS order), and a pi0.5 server
reads it only when it refills that row's queue, to build Run A's prompt. Absent means the original
task string. A version-4 server that predates the key rejects it as bad_obs, as for replan and
retry. The ACT server validates it and ignores it.
"""

import io
import json

import numpy as np

PROTOCOL_VERSION = 4
OBS_SCHEMA = {
    # (7,) = the dataset's observation.state: 6 arm joint angles + the gripper
    # opening fraction (spec section 4.2), which is what convert_to_lerobot.py writes
    # and therefore what the trained policy expects. tests/test_lerobot_schema_pin.py
    # asserts this exact literal from the other venv, where it cannot import this module.
    "state": ((7,), "float32"),
    "wrist": ((224, 224, 3), "uint8"),
    "overhead": ((224, 224, 3), "uint8"),
}
# Also 7, but a different 7: 6 joint TARGETS + the gripper command. The gripper column is
# the raw command in the checkpoint's contract: a binary sign (raw >= 0 opens) or, under
# `gripper = continuous` in the handshake, the jaw gap in meters. The state column stays an
# opening fraction in both. The two widths coincide on this arm and must still be validated
# against their own constants.
#
# Because they coincide, a shape check can no longer catch an action array handed in where
# an observation belongs -- openyam-lab's 8-vs-7 made that swap self-revealing and this
# schema does not. The VALUES still differ structurally, so validate_obs() checks the one
# thing that does discriminate: the state's gripper column is an opening fraction on [0, 1]
# and the action's is a command that goes negative to close. That check is one-directional.
# A state array passed to validate_action() is NOT detectable -- a fraction on [0, 1] is a
# perfectly legal "open" command -- so an action-side swap fails silently and will surface
# as a baffling eval result rather than an error. Recorded here because it is a real loss
# of observability against the openyam-lab original, not an oversight.
ACTION_SHAPE = (7,)
ACTION_DTYPE = "float32"
# Index of the gripper column in BOTH arrays. policy_server.GRIPPER_DIM is the action-side
# counterpart; the two are deliberately separate constants for two separate contracts that
# happen to share an index, so neither can be "fixed" into the other.
STATE_GRIPPER_DIM = 6
# The fraction is clipped to [0, 1] at conversion, so training data is strictly in range.
# The tolerance absorbs a client that computes the fraction without clipping (a hair over 1
# from float division); anything past it is a swap or genuinely out of distribution.
GRIPPER_FRACTION_TOL = 1e-3

CONTEXT_KEY = "context"
CONTEXT_DTYPE = "float32"

ANCHOR_KEY = "anchor"
ANCHOR_SHAPE = (6,)
ANCHOR_DTYPE = "float32"

REPLAN_KEY = "replan"
REPLAN_DTYPE = "bool"

RETRY_KEY = "retry"
RETRY_DTYPE = "bool"

PROMPT_KEY = "prompt"
PROMPT_DTYPE = "int16"


def _pack(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _unpack(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_frames(header: dict, arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    """The multipart frames of one message. A ROUTER socket sends them after its envelope."""
    frames = [json.dumps(header).encode()]
    for name, array in (arrays or {}).items():
        frames.append(name.encode())
        frames.append(_pack(np.ascontiguousarray(array)))
    return frames


def decode_frames(frames: list[bytes]) -> tuple[dict, dict[str, np.ndarray]]:
    """(header, arrays) of one message's frames, the envelope already stripped."""
    header = json.loads(frames[0].decode())
    if len(frames) % 2 != 1:
        raise ValueError(f"malformed message: {len(frames)} frames (want 1 + 2 per array)")
    arrays = {}
    for i in range(1, len(frames), 2):
        arrays[frames[i].decode()] = _unpack(frames[i + 1])
    return header, arrays


def send_message(socket, header: dict, arrays: dict[str, np.ndarray] | None = None) -> None:
    socket.send_multipart(encode_frames(header, arrays))


def recv_message(socket) -> tuple[dict, dict[str, np.ndarray]]:
    return decode_frames(socket.recv_multipart())


def _check(name: str, array: np.ndarray, sample_shape: tuple, dtype: str, batch_size: int) -> None:
    want = (batch_size, *sample_shape)
    if tuple(array.shape) != want:
        raise ValueError(f"{name}: shape {tuple(array.shape)}, want {want}")
    if str(array.dtype) != dtype:
        raise ValueError(f"{name}: dtype {array.dtype}, want {dtype}")


def _check_gripper_fraction(state: np.ndarray) -> None:
    """Reject a state whose gripper column is not an opening fraction.

    The only check that can tell a (7,) action array from a (7,) state array, now that the
    two are the same shape and dtype. An empty batch passes vacuously (np.all of nothing is
    True), which also keeps the error path away from min() on an empty array.
    """
    column = state[:, STATE_GRIPPER_DIM]
    if not np.all((column >= -GRIPPER_FRACTION_TOL) & (column <= 1.0 + GRIPPER_FRACTION_TOL)):
        raise ValueError(
            f"state: gripper fraction (column {STATE_GRIPPER_DIM}) spans "
            f"[{column.min()}, {column.max()}], want [0, 1] -- an action array has a raw "
            f"gripper COMMAND in that column, not an opening fraction"
        )


def validate_context(array: np.ndarray, batch_size: int) -> None:
    """A (batch, K) float32 array with K >= 1."""
    if array.ndim != 2 or array.shape[0] != batch_size or array.shape[1] < 1:
        raise ValueError(f"context: shape {tuple(array.shape)}, want ({batch_size}, K>=1)")
    if str(array.dtype) != CONTEXT_DTYPE:
        raise ValueError(f"context: dtype {array.dtype}, want {CONTEXT_DTYPE}")


def validate_anchor(array: np.ndarray, batch_size: int) -> None:
    """A (batch, 6) float32 array: the previous clamped arm command."""
    _check(ANCHOR_KEY, array, ANCHOR_SHAPE, ANCHOR_DTYPE, batch_size)


def validate_replan(array: np.ndarray, batch_size: int) -> None:
    """A (batch,) bool array: which envs replan from this observation."""
    _check(REPLAN_KEY, array, (), REPLAN_DTYPE, batch_size)


def validate_retry(array: np.ndarray, batch_size: int) -> None:
    """A (batch,) bool array: which envs are in retry mode."""
    _check(RETRY_KEY, array, (), RETRY_DTYPE, batch_size)


def validate_prompt(array: np.ndarray, batch_size: int) -> None:
    """A (batch,) int16 array: each env's subtask id."""
    _check(PROMPT_KEY, array, (), PROMPT_DTYPE, batch_size)


def validate_obs(arrays: dict, batch_size: int) -> None:
    keys = set(arrays) - {CONTEXT_KEY, ANCHOR_KEY, REPLAN_KEY, RETRY_KEY, PROMPT_KEY}
    if keys != set(OBS_SCHEMA):
        raise ValueError(
            f"obs keys {sorted(arrays)}, want {sorted(OBS_SCHEMA)} "
            "(+ optional context, anchor, replan, retry, prompt)"
        )
    for name, (sample_shape, dtype) in OBS_SCHEMA.items():
        _check(name, arrays[name], sample_shape, dtype, batch_size)
    _check_gripper_fraction(arrays["state"])
    if CONTEXT_KEY in arrays:
        validate_context(arrays[CONTEXT_KEY], batch_size)
    if ANCHOR_KEY in arrays:
        validate_anchor(arrays[ANCHOR_KEY], batch_size)
    if REPLAN_KEY in arrays:
        validate_replan(arrays[REPLAN_KEY], batch_size)
    if RETRY_KEY in arrays:
        validate_retry(arrays[RETRY_KEY], batch_size)
    if PROMPT_KEY in arrays:
        validate_prompt(arrays[PROMPT_KEY], batch_size)


def validate_action(array: np.ndarray, batch_size: int) -> None:
    _check("action", array, ACTION_SHAPE, ACTION_DTYPE, batch_size)
