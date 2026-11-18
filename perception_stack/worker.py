"""The perception worker (MS5 design section 4.2): parts and boxes from one overhead frame.

It serves ``perception_protocol``'s wire on a REP socket. ``parts`` finds the depth blobs above
the tray floor and prompts SAM 2.1, in one batch on a crop of the RGB frame, with one point per
``rules.PROMPT_GRID_M`` square of each blob, so touching parts in one blob each get a prompt. It
keeps the distinct masks (``rules.distinct_masks``), turns each into a detection with ``rules``
and drops a mask larger than any base-class part (a prompt in a gap returns the background). ``boxes`` finds the dark box outlines with ``rules``
alone. ``marker``, ``wrist`` and ``lift`` reply ``unsupported``: no loop runs in MS5.

    perception_stack/.venv/bin/python perception_stack/worker.py --rig rig.json

Prints ``WORKER_READY {worker_id, model}`` once the model is loaded, and ``WORKER_OK {calls}``
after ``shutdown``. Any error prints ``WORKER_FAIL {reason}``. Exits through ``os._exit``.
This file imports torch and ``sam2`` only inside ``Segmenter``, so the lab's CPU suite loads it
by path with a fake segmenter. The rig is the lab's rig JSON; this file never imports the lab.
"""

import argparse
import importlib.util
import io
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def _sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _sibling("rules")
proto = _sibling("perception_protocol")

CROP_MARGIN_PX = 20
BOX_BANDS = ((0.05, 0.21, 0.18, 0.22), (0.05, 0.21, -0.22, -0.18))
"""(x_min, x_max, y_min, y_max) of the two box side bands (MS5 design section 4.1)."""
WORKER_ID = "perception-worker-1"


class Segmenter:
    """Prompted SAM 2.1: one mask per point prompt, one batch, bfloat16 autocast."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda"):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._torch = torch
        self._predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        self._device = device

    def masks(self, rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
        """(N, H, W) bool masks of an (H, W, 3) uint8 image, one per (x, y) point."""
        n = len(points)
        if n == 0:
            return np.zeros((0, *rgb.shape[:2]), bool)
        coords = np.asarray(points, np.float32)[:, None, :]
        labels = np.ones((n, 1), np.int32)
        with (
            self._torch.inference_mode(),
            self._torch.autocast(self._device, dtype=self._torch.bfloat16),
        ):
            self._predictor.set_image(rgb)
            masks, _, _ = self._predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=False
            )
        return np.asarray(masks).reshape(n, *rgb.shape[:2]).astype(bool)


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _camera(block: dict):
    """(k, rot, pos) of a wire camera block."""
    pose = block["pose"]
    return block["k"], _quat_matrix(pose["quat"]), np.array(pose["pos"], float)


def _error(error: str, message: str) -> tuple[dict, dict]:
    return {"type": "error", "error": error, "message": message}, {}


class Worker:
    def __init__(self, segmenter, rig: dict):
        self.segmenter = segmenter
        self.rig = rig
        self.calls = 0

    def handle(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        kind = header.get("type")
        self.calls += kind != "hello"  # the handshake is not a call
        try:
            if kind == "hello":
                if header.get("protocol") != proto.PROTOCOL_VERSION:
                    return _error("protocol", f"worker speaks protocol {proto.PROTOCOL_VERSION}")
                return {
                    "type": "hello_ack",
                    "protocol": proto.PROTOCOL_VERSION,
                    "worker_id": WORKER_ID,
                }, {}
            if kind == "shutdown":
                return {"type": "shutdown_ack"}, {}
            if kind == "parts":
                return self._parts(header, arrays)
            if kind == "boxes":
                return self._boxes(header, arrays)
            return _error("unsupported", f"{kind!r} is not served in MS5")
        except Exception as err:  # noqa: BLE001  a bad request must not stop the worker
            return _error("failed", f"{type(err).__name__}: {err}")

    def _boxes(self, header, arrays):
        rgb = np.asarray(Image.open(io.BytesIO(arrays["rgb_jpeg"].tobytes())).convert("L"))
        k, rot, pos = _camera(header["camera"])
        depth_world = None
        if header.get("depth_camera") is not None and "depth" in arrays:
            dk, drot, dpos = _camera(header["depth_camera"])
            world = rules.deproject(arrays["depth"], dk, drot, dpos)
            # nearest-neighbour resample to the luma grid: depth is only a check
            rows = np.arange(rgb.shape[0]) * world.shape[0] // rgb.shape[0]
            cols = np.arange(rgb.shape[1]) * world.shape[1] // rgb.shape[1]
            depth_world = world[np.ix_(rows, cols)]
        interior = self.rig["box_interior"]
        half_diag = math.hypot(interior[0], interior[1]) / 2
        bands = [
            (x0 - half_diag, x1 + half_diag, y0 - half_diag, y1 + half_diag)
            for x0, x1, y0, y1 in BOX_BANDS
        ]
        poses = rules.box_detections(
            rgb,
            k,
            rot,
            pos,
            self.rig["desk_z"],
            bands,
            tuple(interior),
            depth_world,
            wall_m=self.rig.get("box_wall", rules.BOX_WALL_M),
        )
        return {"type": "boxes", "poses": poses}, {}

    def _parts(self, header, arrays):
        if header.get("depth_camera") is None or "depth" not in arrays:
            return _error("no_depth", "parts needs a depth frame")
        rig = self.rig
        floor_z = rig["tray_floor_z"]
        rgb = np.asarray(Image.open(io.BytesIO(arrays["rgb_jpeg"].tobytes())).convert("RGB"))
        depth = arrays["depth"]
        k, rot, pos = _camera(header["camera"])
        dk, drot, dpos = _camera(header["depth_camera"])
        world = rules.deproject(depth, dk, drot, dpos)
        # ponytail: the tray region is its axis-aligned interior and the floor one plane, so
        # tray_lean_deg is unread. Model the leaned floor if the tray read needs it (task 9).
        cx, cy = rig["tray_centre"]
        hx, hy = rig["tray_interior"][0] / 2, rig["tray_interior"][1] / 2
        region = (cx - hx, cx + hx, cy - hy, cy + hy)
        blobs = rules.blobs_above(world, floor_z, region)
        if not blobs:
            return {"type": "parts", "detections": []}, {}
        prompts = [rules.grid_prompts(world[b[:, 0], b[:, 1]].astype(np.float64)) for b in blobs]
        owner = np.repeat(np.arange(len(blobs)), [len(p) for p in prompts])
        points = rules.project(np.concatenate(prompts), k, rot, pos)
        corners = np.array(
            [(x, y, z) for x in region[:2] for y in region[2:] for z in (floor_z, floor_z + 0.03)]
        )
        px = rules.project(corners, k, rot, pos)
        h, w = rgb.shape[:2]
        x0, y0 = np.maximum(np.floor(px.min(0)) - CROP_MARGIN_PX, 0).astype(int)
        x1 = min(int(np.ceil(px[:, 0].max())) + CROP_MARGIN_PX, w)
        y1 = min(int(np.ceil(px[:, 1].max())) + CROP_MARGIN_PX, h)
        masks = self.segmenter.masks(rgb[y0:y1, x0:x1], points - [x0, y0])
        dh, dw = depth.shape
        cells = []
        for mask in masks:
            rows, cols = np.nonzero(mask)
            pixels = np.stack([rows + y0, cols + x0], axis=-1)
            drows = np.minimum((pixels[:, 0] + 0.5) * dh / h, dh - 1).astype(int)
            dcols = np.minimum((pixels[:, 1] + 0.5) * dw / w, dw - 1).astype(int)
            cells.append((pixels, drows, dcols))
        scores = [np.isfinite(world[r, c, 2]).mean() if len(r) else 0.0 for _, r, c in cells]
        longest = [m + rules.MASK_FOOTPRINT_MARGIN_M for m in rules.PART_MAX_FOOTPRINT_M]
        detections = []
        for i in rules.distinct_masks(masks, scores):
            pixels, drows, dcols = cells[i]
            heights = world[drows, dcols, 2] - floor_z
            blob = blobs[owner[i]]
            top = float(np.percentile(world[blob[:, 0], blob[:, 1], 2] - floor_z, 90))
            xy = rules.mask_xy(pixels, depth[drows, dcols], k, rot, pos, floor_z + top)
            det = rules.part_detection(xy, heights, floor_z)
            if det["size"][0] <= longest[0] and det["size"][1] <= longest[1]:
                detections.append(det)
        detections.sort(key=lambda d: -d["score"])
        return {"type": "parts", "detections": detections}, {}


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("WORKER_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def main(argv=None) -> int:
    import zmq

    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5600")
    parser.add_argument("--rig", required=True, help="the rig JSON")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    args = parser.parse_args(argv)
    try:
        rig = json.loads(Path(args.rig).read_text())
        worker = Worker(Segmenter(args.model), rig)
        socket = zmq.Context.instance().socket(zmq.REP)
        socket.bind(args.endpoint)
        print(
            "WORKER_READY " + json.dumps({"worker_id": WORKER_ID, "model": args.model}), flush=True
        )
        while True:
            header, arrays = proto.recv_message(socket)
            reply, out = worker.handle(header, arrays)
            proto.send_message(socket, reply, out)
            if header.get("type") == "shutdown":
                break
        print("WORKER_OK " + json.dumps({"calls": worker.calls}), flush=True)
        return 0
    except BaseException as err:  # noqa: BLE001  includes Ctrl-C: the token line is the contract
        traceback.print_exc(file=sys.stderr)
        reason = (
            "interrupted" if isinstance(err, KeyboardInterrupt) else f"{type(err).__name__}: {err}"
        )
        print("WORKER_FAIL " + json.dumps({"reason": reason}), flush=True)
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
