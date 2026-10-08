"""Shared pi0.5 policy server (MS7h design 7.3): one pi05 copy serves many read lanes.

The lanes are the unchanged REQ clients of the ACT server (policy_client.ServerPolicy, core.loop):
the wire format is policy_protocol.py's, on a ZMQ ROUTER socket that sees one identity a lane.
Each lane keeps its own action queue per env row and its own warp, as the ACT server's
ActionQueues does. The loop drains every request that has arrived, answers the lanes whose queue
still holds an action, and runs one forward over the rows of every lane whose queue is empty (or
whose replan flag is set), so the lanes that wait on a forward share it. The server builds the
23-wide pi05 state from the request's state and context with derive_vla_dataset.derived_state,
the camera roll wrapped exactly as the training data was, and writes derive_vla_dataset.TASK as
the prompt's task, so the clients send what they send to an ACT server with context.

Serving matches B's: the checkpoint's 50 actions a forward, open loop (no temporal ensembling),
its 10 flow steps, and --action_warp (1.5 for the reads). Two sampling switches serve the jaw
test of the MS7h pi0.5 recipe: --num_steps sets the flow steps a forward, and --noise_scale S
starts the flow from S * N(0, I), so 0 gives deterministic chunks. At their defaults (the
checkpoint's steps, 1.0) sample_actions gets no extra argument and draws its own noise. Zero
noise leaves V parked at a cycle start (h_V30kM2k_z0_a, 2026-10-08), so --still_noise_scale S2
samples each still row of a forward once more, from S2 * N(0, I), in one forward over those
rows, and serves that chunk instead (``still_rows``). Zero noise also removes V's re-open after
its own close, so --jaw_noise_scale J multiplies the jaw column of every initial noise (a
resample's too) by J and keeps the arm's noise. The policy id is ``pi05:<checkpoint>`` plus
``+warp<X>``, ``+steps<N>``, ``+noise<S>`` and ``+jawnoise<J>`` for each value off its default,
and ``+still<S2>r<E>o<F>`` with the rule on, so the read cache never mixes it with an ACT read or
another sampling.
2026-10-05 on Machine A's RTX 5090, shared with ACT read servers, on a 50-step dry-run checkpoint:
one forward of 1 row took 172-207 ms, 3 rows 245-291 ms and 6 rows 346 ms, at 9.4 GiB peak.
Every forward prints "FORWARD rows=N ms=T" (plus " still=K", the rows resampled, with the rule
on), so a read's log holds its batch sizes and times.

The checkpoint is a pretrained_model directory of a pi05 fine-tune on the derived dataset; its
action_space.json marker (the same walk as policy_server's) must say "absolute" and name the
gripper contract, because a pi05 run trains on whatever the dataset holds and a missing marker
would serve a continuous jaw gap as a binary sign.

Runs in lerobot_stack/.venv-pi (load_pi05 imports lerobot's pi05, which needs transformers):
    lerobot_stack/.venv-pi/bin/python lerobot_stack/pi05_server.py \\
        --checkpoint <run>/checkpoints/last/pretrained_model --port 30500 --action_warp 1.5
-> SERVER_READY {...} once bound, SERVER_FAIL on every failure.
"""

import argparse
import json
import os
import time
import traceback
from collections import deque

import numpy as np
import zmq

from derive_vla_dataset import TASK, derived_state
from policy_protocol import (
    ACTION_DTYPE,
    ACTION_SHAPE,
    CONTEXT_KEY,
    OBS_SCHEMA,
    PROTOCOL_VERSION,
    REPLAN_KEY,
    decode_frames,
    encode_frames,
    validate_obs,
)
from policy_server import (
    ACTION_WARP_MAX,
    ACTION_WARP_MIN,
    _read_action_space_marker,
    warp_chunk,
)

CONTEXT_DIM = 16
TOKENIZER = "tokenizer_processor"  # the registry name of lerobot's prompt tokenizer step
IMAGE_KEYS = {"wrist": "observation.images.wrist", "overhead": "observation.images.overhead"}
STILL_RAD = 0.15
"""rad: a chunk is still when the max over its served actions (before the warp) and the 6 arm
joints of |action - q_now| is under it, q_now the request's state[:, :6] (both URDF radians).
2026-10-08, that max on recorded chunks: corpus label chunks (derived dataset files 000-001,
1.79M frames, padded past the episode end as LeRobot pads) parked at the desk-box retreat point
with open jaws p50 0.028, p92 0.15 (n 31,497), and the chunks that leave it (> 4 cm) 0.297 at the
least (n 15,468); every other open-jaw frame under 0.15 (n 20,647) sits at retreat height (pick
z >= 0.10). The in-loop V30k trace (default noise, q_now taken as the last command): parked p50
0.087, p88 0.15 (n 647), leaving 0.211 at the least (n 9); the base yaw and the wrist roll make
most of a parked chunk's motion. No zero-noise parked chunk was recorded."""
STILL_OPEN = 0.9
"""A still row also has its jaws open: the request's state[:, 6] (the left jaw's opening
fraction, 1.0 fully open) is at least this, so a close, where the teacher holds the arm still,
is never resampled. 2026-10-08: the parked frames above read 0.9615 at the least (the trace's
measured gap 0.989), the jaw test's 400 frames at a close 0.548 at the most."""
JAW = 6
"""The jaw's column of the 7 actions, which the model samples normalized (per column) and padded
at the end to max_action_dim: the derived dataset's meta/info.json names the 6 URDF joints, then
gripper_gap_m. The checkpoint config names no action column (lerobot 0.5.1's PolicyFeature holds
a type and a shape), so load_pi05 checks only the 7-wide action."""


def still_rows(chunk: np.ndarray, state: np.ndarray, rad: float, open_frac: float) -> np.ndarray:
    """Indices of the rows whose (n, n_action_steps, 7) chunk is still: every action's 6 arm
    joints within ``rad`` of the row's state[:, :6] (the jaw column does not count), with the
    jaws open (state[:, 6] >= ``open_frac``). Pure."""
    moved = np.abs(chunk[:, :, :6] - state[:, None, :6]).max(axis=(1, 2))
    return np.flatnonzero((moved < rad) & (state[:, 6] >= open_frac))


class SharedServer:
    """The protocol and the queues, with the model injected: ``predict_chunk(obs)`` takes
    {"state": (n, 23), "wrist": (n, 224, 224, 3) uint8, "overhead": ...} and returns the
    unnormalized (n, n_action_steps, 7) chunk. ``process`` is socket-free for the tests. With
    ``still`` = (S2, E, F), the still rows of a forward (``still_rows`` at E and F) get
    ``predict_chunk(obs of those rows, S2)``'s chunk."""

    def __init__(
        self,
        predict_chunk,
        policy_id: str,
        n_action_steps: int,
        inference: dict,
        gripper: str,
        action_warp: float = 1.0,
        still: tuple[float, float, float] | None = None,
    ):
        if not ACTION_WARP_MIN <= action_warp <= ACTION_WARP_MAX:
            raise ValueError(
                f"action_warp must be in [{ACTION_WARP_MIN}, {ACTION_WARP_MAX}], got {action_warp}"
            )
        self._predict = predict_chunk
        self._policy_id = policy_id
        self._n = n_action_steps
        self._inference = {**inference, "action_warp": action_warp}
        if still is not None:
            names = ("still_noise_scale", "still_rad", "still_open")
            self._inference |= dict(zip(names, still))
        self._still = still
        self._gripper = gripper
        self._warp = action_warp
        self._queues: dict[bytes, list[deque]] = {}  # lane identity -> one queue an env row
        self.forward_sizes: list[int] = []  # rows of each forward, for the log and the tests

    def ready(self, port: int) -> dict:
        return {
            "port": port,
            "policy_id": self._policy_id,
            "n_action_steps": self._n,
            "protocol": PROTOCOL_VERSION,
            "inference": self._inference,
            "context_dim": CONTEXT_DIM,
            "action_space": "absolute",
            "gripper": self._gripper,
        }

    def _hello_ack(self) -> dict:
        ack = {key: value for key, value in self.ready(0).items() if key != "port"}
        return {
            "type": "hello_ack",
            **ack,
            "obs_schema": {k: [list(shape), dtype] for k, (shape, dtype) in OBS_SCHEMA.items()},
            "action": [list(ACTION_SHAPE), ACTION_DTYPE],
        }

    def _check_step(self, header: dict, arrays: dict) -> str | None:
        """The bad_obs message of a step request, None if it is servable."""
        try:
            validate_obs(arrays, int(header["batch"]))
        except (KeyError, ValueError) as err:
            return str(err)
        width = 0 if CONTEXT_KEY not in arrays else int(arrays[CONTEXT_KEY].shape[1])
        if width != CONTEXT_DIM:
            return f"context width {width}, this policy takes {CONTEXT_DIM}"
        return None

    def process(self, requests: list) -> tuple[list, bool]:
        """[(lane, header, arrays)] -> ([(lane, header, arrays)], keep serving). One reply a
        request; the steps that need a forward share one."""
        replies, steps, keep = [], [], True
        for lane, header, arrays in requests:
            kind = header.get("type")
            if kind == "hello":
                if header.get("protocol") != PROTOCOL_VERSION:
                    message = (
                        f"server speaks {PROTOCOL_VERSION}, client sent {header.get('protocol')}"
                    )
                    replies.append((lane, _error("protocol_mismatch", message), None))
                else:
                    replies.append((lane, self._hello_ack(), None))
            elif kind == "reset":
                self._queues.pop(lane, None)
                replies.append((lane, {"type": "reset_ack"}, None))
            elif kind == "step":
                problem = self._check_step(header, arrays)
                if problem is None:
                    steps.append((lane, arrays))
                else:
                    replies.append((lane, _error("bad_obs", problem), None))
            elif kind == "shutdown":
                replies.append((lane, {"type": "shutdown_ack"}, None))
                keep = False
            else:
                replies.append((lane, _error("exception", f"unknown message type {kind!r}"), None))
        if steps:
            replies.extend(self._step(steps))
        return replies, keep

    def _step(self, steps: list) -> list:
        need = []  # (lane, row) of every env row whose queue is refilled now
        for lane, arrays in steps:
            batch = int(arrays["state"].shape[0])
            if len(self._queues.get(lane, ())) != batch:
                self._queues[lane] = [deque() for _ in range(batch)]
            replan = arrays.get(REPLAN_KEY, np.zeros(batch, dtype=bool))
            need += [(lane, i) for i, q in enumerate(self._queues[lane]) if replan[i] or not q]
        if need:
            rows = {lane: arrays for lane, arrays in steps}
            obs = {
                "state": derived_state(
                    np.stack([rows[lane]["state"][i] for lane, i in need]),
                    np.stack([rows[lane][CONTEXT_KEY][i] for lane, i in need]),
                ),
                **{k: np.stack([rows[lane][k][i] for lane, i in need]) for k in IMAGE_KEYS},
            }
            start = time.perf_counter()
            try:
                chunk = np.asarray(self._predict(obs), dtype=np.float32)
                if chunk.shape != (len(need), self._n, *ACTION_SHAPE):
                    raise ValueError(
                        f"predict_chunk returned {chunk.shape}, want ({len(need)}, {self._n}, 7)"
                    )
                if self._still is not None:  # one more forward over the still rows only
                    still = still_rows(chunk, obs["state"], *self._still[1:])
                    if len(still):
                        resample = {k: v[still] for k, v in obs.items()}
                        chunk[still] = self._predict(resample, self._still[0])
            except Exception:  # noqa: BLE001 -- every lane of the forward gets the error
                error = _error("exception", traceback.format_exc())
                for lane, _ in need:
                    self._queues.pop(lane, None)
                return [(lane, error, None) for lane, _ in steps]
            self.forward_sizes.append(len(need))
            ms = 1000 * (time.perf_counter() - start)
            log = f"FORWARD rows={len(need)} ms={ms:.1f}"  # the read's batch log
            print(log + ("" if self._still is None else f" still={len(still)}"), flush=True)
            for (lane, i), actions in zip(need, warp_chunk(chunk, self._warp)):
                self._queues[lane][i] = deque(actions)
        replies = []
        for lane, arrays in steps:
            action = np.stack([q.popleft() for q in self._queues[lane]])
            if np.isfinite(action).all():
                replies.append((lane, {"type": "action"}, {"action": action}))
            else:
                message = "policy produced non-finite actions"
                replies.append((lane, _error("nan_action", message), None))
        return replies

    def serve(self, socket) -> None:
        """Block for one request, drain the rest that arrived, answer them all; until shutdown."""
        keep = True
        while keep:
            requests = []
            flags = 0
            while True:
                try:
                    frames = socket.recv_multipart(flags)
                except zmq.Again:
                    break
                flags = zmq.NOBLOCK
                lane, body = frames[0], frames[2:]  # ROUTER envelope: identity, empty delimiter
                try:
                    header, arrays = decode_frames(body)
                except Exception as err:  # noqa: BLE001 -- REQ discipline: answer every request
                    socket.send_multipart(
                        [lane, b"", *encode_frames(_error("exception", str(err)))]
                    )
                    continue
                requests.append((lane, header, arrays))
            try:
                replies, keep = self.process(requests)
            except Exception:  # noqa: BLE001 -- REQ discipline: answer every request
                error = _error("exception", traceback.format_exc())
                replies = [(lane, error, None) for lane, _, _ in requests]
            for lane, header, arrays in replies:
                socket.send_multipart([lane, b"", *encode_frames(header, arrays)])


def _error(error: str, message: str) -> dict:
    return {"type": "error", "error": error, "message": message}


def load_pi05(
    checkpoint: str,
    device: str,
    num_steps: int | None = None,
    noise_scale: float = 1.0,
    jaw_noise_scale: float = 1.0,
):
    """(predict_chunk, n_action_steps, inference, gripper, sampling) of a pi05 pretrained_model
    directory. ``sampling`` is the policy id's ``+steps<N>+noise<S>+jawnoise<J>``, each part only
    when its value is off its default (num_steps None means the checkpoint's own).

    The server reads only local files: the checkpoint and the tokenizer in the Hugging Face
    cache (the gated google/paligemma-3b-pt-224 files). transformers 5.3's AutoTokenizer also
    reads the repo's config.json, which an offline host does not cache, so the tokenizer step
    gets the class that the cached tokenizer_config.json names, GemmaTokenizer, loaded here."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before lerobot imports huggingface_hub
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from transformers import GemmaTokenizer

    marker = _read_action_space_marker(checkpoint)
    if marker is None or marker.get("action_space") != "absolute" or "gripper" not in marker:
        raise ValueError(
            f"{checkpoint} needs an action_space.json saying the dataset's action_space "
            f"(absolute) and gripper, found {marker}"
        )
    policy = PI05Policy.from_pretrained(checkpoint)
    config = policy.config
    images = sorted(config.image_features)
    state, action = config.robot_state_feature.shape, config.output_features["action"].shape
    if images != sorted(IMAGE_KEYS.values()) or state != (23,) or action != (7,):
        raise ValueError(
            f"{checkpoint} takes {images} and a state of {state} to an action of {action}, "
            f"not the derived dataset's {sorted(IMAGE_KEYS.values())}, (23,) and (7,)"
        )
    policy.to(device)
    policy.eval()
    with open(os.path.join(checkpoint, "policy_preprocessor.json")) as f:
        steps = json.load(f)["steps"]
    tokenizer = next(
        s["config"]["tokenizer_name"] for s in steps if s["registry_name"] == TOKENIZER
    )
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            TOKENIZER: {"tokenizer": GemmaTokenizer.from_pretrained(tokenizer)},
        },
    )

    kwargs, sampling = {}, ""  # what reaches sample_actions: nothing at the defaults
    if num_steps not in (None, config.num_inference_steps):
        kwargs["num_steps"], sampling = num_steps, f"+steps{num_steps}"
    if noise_scale != 1.0:
        sampling += f"+noise{noise_scale}"
    if jaw_noise_scale != 1.0:
        sampling += f"+jawnoise{jaw_noise_scale}"

    def predict_chunk(obs: dict, scale: float = noise_scale) -> np.ndarray:
        noise = {}
        if scale != 1.0 or jaw_noise_scale != 1.0:  # sample_actions' own draw, scaled
            shape = (len(obs["state"]), config.chunk_size, config.max_action_dim)
            noise["noise"] = scale * policy.model.sample_noise(shape, device)
            noise["noise"][..., JAW] *= jaw_noise_scale
        batch = {
            "observation.state": torch.from_numpy(obs["state"]),
            "task": [TASK] * len(obs["state"]),
        }
        for key, name in IMAGE_KEYS.items():
            batch[name] = torch.from_numpy(obs[key]).permute(0, 3, 1, 2).to(torch.float32) / 255.0
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(pre(batch), **kwargs, **noise)
            chunk = chunk[:, : config.n_action_steps]
            return np.asarray(post(chunk).to("cpu", torch.float32), dtype=np.float32)

    inference = {
        "n_action_steps": int(config.n_action_steps),
        "temporal_ensemble_coeff": None,
        "chunk_size": int(config.chunk_size),
        "num_inference_steps": int(kwargs.get("num_steps", config.num_inference_steps)),
        **({"noise_scale": noise_scale} if noise_scale != 1.0 else {}),
        **({"jaw_noise_scale": jaw_noise_scale} if jaw_noise_scale != 1.0 else {}),
    }
    return predict_chunk, int(config.n_action_steps), inference, marker["gripper"], sampling


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"SERVER_FAIL usage: {message}", flush=True)
        os._exit(2)


def main() -> int:
    parser = TokenArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="a pi05 pretrained_model directory")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action_warp", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, help="flow steps a forward (the checkpoint's)")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="0: deterministic chunks")
    parser.add_argument("--jaw_noise_scale", type=float, default=1.0, help="the jaw column's")
    parser.add_argument("--still_noise_scale", type=float, help="resample still rows at this")
    parser.add_argument("--still_rad", type=float, default=STILL_RAD)
    parser.add_argument("--still_open", type=float, default=STILL_OPEN)
    try:
        args = parser.parse_args()
        if args.num_steps is not None and args.num_steps < 1:
            parser.error(f"--num_steps must be at least 1, got {args.num_steps}")
        if not 0.0 <= args.noise_scale < float("inf"):
            parser.error(f"--noise_scale must be finite and at least 0, got {args.noise_scale}")
        if not 0.0 <= args.jaw_noise_scale < float("inf"):
            parser.error(
                f"--jaw_noise_scale must be finite and at least 0, got {args.jaw_noise_scale}"
            )
        scale, rad, open_frac = args.still_noise_scale, args.still_rad, args.still_open
        if scale is not None and not 0.0 <= scale < float("inf"):
            parser.error(f"--still_noise_scale must be finite and at least 0, got {scale}")
        if not 0.0 < rad < float("inf"):
            parser.error(f"--still_rad must be finite and above 0, got {rad}")
        if not 0.0 <= open_frac <= 1.0:
            parser.error(f"--still_open must be in [0, 1], got {open_frac}")
        predict_chunk, n_action_steps, inference, gripper, sampling = load_pi05(
            args.checkpoint, args.device, args.num_steps, args.noise_scale, args.jaw_noise_scale
        )
        policy_id = f"pi05:{args.checkpoint}" + (
            f"+warp{args.action_warp}" if args.action_warp != 1.0 else ""
        )
        still = None if scale is None else (scale, rad, open_frac)
        policy_id += sampling + ("" if still is None else "+still{}r{}o{}".format(*still))
        server = SharedServer(
            predict_chunk, policy_id, n_action_steps, inference, gripper, args.action_warp, still
        )
        socket = zmq.Context.instance().socket(zmq.ROUTER)
        socket.bind(f"tcp://127.0.0.1:{args.port}")
        print("SERVER_READY " + json.dumps(server.ready(args.port)), flush=True)
        server.serve(socket)  # returns on a shutdown request, as the ACT server does
    except Exception as exc:  # noqa: BLE001 -- any failure must become SERVER_FAIL
        traceback.print_exc()
        print(f"SERVER_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    os._exit(main())
