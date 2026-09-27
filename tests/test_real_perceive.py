"""MS5 design section 4.3 and P2: the real perception backend on a fake worker."""

import math

import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, loop
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.parts import make_part
from maker_arm_lab.core.perception import PARKED_CENTRE
from maker_arm_lab.real.frames import FrameBundle
from maker_arm_lab.real.pacing import TickLog
from maker_arm_lab.real.perceive import RealPerception, WorkerClient, proto
from maker_arm_lab.real.rig import episode_sheet
from real_fakes import FakeWorker

CAMERA = proto.camera_header(
    (300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0), (640, 480), (0.3, 0.0, 0.7), (0, 1, 0, 0)
)
SIZE = list(make_part("brick", 2, 4).size)


def det(x, y, cls="wide"):
    return {
        "centre": [x, y, SIZE[2] / 2],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": SIZE,
        "cls": cls,
        "score": 0.9,
    }


class _StubEnv:
    def __init__(self):
        self.ticks = TickLog()
        self.holds = 0
        self.marker_points = torch.full((1, 3), math.nan)

    def trip(self, reason, detail):
        self.holds += 1
        raise EvalFailure(f"watchdog_{reason}", detail)

    def overhead_frame(self):
        jpeg = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)
        return FrameBundle({"camera": CAMERA, "depth_camera": None}, {"rgb_jpeg": jpeg})

    def wrist_frame(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def wrist_camera_header(self):
        return CAMERA


class _Rig:
    def pick_point(self, i):
        return torch.tensor([0.2, 0.0, 0.1])


@pytest.fixture()
def rig_parts():
    made = []

    def make(slots=4, timeout_ms=2000, **worker_kwargs):
        worker = FakeWorker(**worker_kwargs)
        env = _StubEnv()
        backend = RealPerception(env, WorkerClient(worker.endpoint, timeout_ms=timeout_ms), slots)
        made.append((worker, backend))
        return worker, env, backend

    yield make
    for worker, backend in made:
        backend.client.close()
        worker.close()


def test_the_backend_has_every_member_the_loop_calls():
    members = {
        "perceiver",
        "marker",
        "class_views",
        "wrist",
        "overhead",
        "wrist_look",
        "lift_class",
    }
    assert members <= set(dir(RealPerception)) and members <= set(dir(loop.SimPerception))


def test_the_client_says_hello(rig_parts):
    _, _, backend = rig_parts()
    assert backend.client.worker_id == "fake"


def test_reset_episode_perceives_and_labels_the_boxes(rig_parts):
    sheet = episode_sheet(20265000, 0)
    turned = [
        [x, y, yaw + math.pi / 2] for x, y, yaw in (sheet["box_poses"][k] for k in (3, 0, 2, 1))
    ]
    worker, env, backend = rig_parts(boxes=turned)
    perceiver = backend.perceiver(1, 0, {})
    perceiver.reset_episode(np.full((4, 3), np.nan))
    assert np.array(perceiver.box_poses) == pytest.approx(np.array(sheet["box_poses"]))
    assert perceiver.box_sizes == desk_boxes.interiors(4)
    assert worker.kinds()[-1] == "boxes" and env.ticks.close(0.1).perception


def test_anything_but_two_boxes_a_side_holds_the_arm(rig_parts):
    _, env, backend = rig_parts(boxes=[[0.1, 0.2, 0.0], [0.2, 0.2, 0.0], [0.1, -0.2, 0.0]])
    with pytest.raises(EvalFailure) as err:
        backend.perceiver(1, 0, {}).reset_episode(None)
    assert err.value.reason == "watchdog_boxes" and env.holds == 1


def test_detections_past_the_slots_are_dropped_in_worker_order(rig_parts):
    _, env, backend = rig_parts(slots=3, detections=[det(0.16 + 0.01 * k, 0.0) for k in range(5)])
    perceiver = backend.perceiver(1, 0, {})
    state = perceiver.perceive(backend.overhead(env, _Rig(), 0))
    assert [d.sim_index for d in state.detections] == [0, 1, 2]
    assert [d.centre[0] for d in state.detections] == pytest.approx([0.16, 0.17, 0.18])
    assert state.box_poses == perceiver.box_poses


def test_no_detection_is_an_empty_scene(rig_parts):
    _, env, backend = rig_parts(slots=3, detections=[])
    state = backend.perceiver(1, 0, {}).perceive(backend.overhead(env, _Rig(), 0))
    scene = state.scene(3, torch.tensor([0.2, 0.0, 0.1]))
    assert state.detections == []
    assert torch.allclose(scene.brick_pos, torch.tensor([PARKED_CENTRE] * 3))


def test_a_malformed_detection_holds_the_arm(rig_parts):
    _, env, backend = rig_parts(detections=[det(0.2, 0.0, cls="blue")])
    with pytest.raises(EvalFailure) as err:
        backend.perceiver(1, 0, {}).perceive(backend.overhead(env, _Rig(), 0))
    assert err.value.reason == "watchdog_perception" and env.holds == 1


def test_the_marker_keeps_the_mean_of_look_minus_fk_and_corrects_the_centres(rig_parts):
    _, env, backend = rig_parts(detections=[det(0.2, 0.0)], marker=[0.21, 0.02, 0.1])
    perceiver = backend.perceiver(1, 0, {})
    marker = backend.marker(perceiver, 1, 0)
    state = perceiver.perceive(backend.overhead(env, _Rig(), 0))
    marker.look([0.2, 0.0, 0.1])
    marker.look([0.2, 0.0, 0.1])
    assert marker.estimate == pytest.approx((0.01, 0.02, 0.0))
    assert marker.correct(state).detections[0].centre == pytest.approx((0.19, -0.02, SIZE[2] / 2))
    assert math.isnan(marker.error_mm())


def test_a_marker_miss_leaves_the_estimate(rig_parts):
    _, env, backend = rig_parts(detections=[det(0.2, 0.0)], marker=None)
    perceiver = backend.perceiver(1, 0, {})
    marker = backend.marker(perceiver, 1, 0)
    perceiver.perceive(backend.overhead(env, _Rig(), 0))
    marker.look([0.2, 0.0, 0.1])
    assert (marker.looks, marker.misses, marker.estimate) == (0, 1, (0.0, 0.0, 0.0))


def test_the_wrist_look_is_found_near_the_frozen_marker(rig_parts):
    worker, env, backend = rig_parts(wrist=[0.203, 0.031, 0.4])
    env.marker_points[0] = torch.tensor([0.2, 0.03, 0.0057])
    x, y, yaw, covered = backend.wrist_look(env, _Rig(), 0, 2)
    assert (x, y) == pytest.approx((0.2, 0.03)) and covered is False and math.isnan(yaw)
    assert backend.wrist(1, 0, {}).look((x, y), yaw, covered) == pytest.approx((0.203, 0.031, 0.4))
    kind, header, arrays = worker.requests[-1]
    assert kind == "wrist" and header["prior"] == pytest.approx([0.2, 0.03])
    assert arrays["wrist_rgb"].shape == (224, 224, 3)


def test_a_wrist_miss_is_none(rig_parts):
    _, _, backend = rig_parts(wrist=None)
    assert backend.wrist(1, 0, {}).look((0.2, 0.03), math.nan, False) is None


def test_the_lift_class_and_the_third_view_come_from_the_wrist_frame(rig_parts):
    worker, env, backend = rig_parts(lift_cls="tall")
    views = backend.class_views(1, 0, {}, 4)
    assert backend.lift_class(env, 0) == ("tall", 0)
    assert views.view(0, "wide") == "tall"
    assert backend.perceiver(1, 0, {}).classify("tall", part=0) == "tall"
    worker.lift_cls = None
    assert backend.lift_class(env, 0) == (None, -1) and views.view(0, "wide") == "wide"


def test_an_unknown_lift_class_holds_the_arm(rig_parts):
    _, env, backend = rig_parts(lift_cls="round")
    with pytest.raises(EvalFailure) as err:
        backend.lift_class(env, 0)
    assert err.value.reason == "watchdog_perception" and env.holds == 1


def test_a_worker_timeout_holds_the_arm(rig_parts):
    _, env, backend = rig_parts(timeout_ms=100, delay_s=0.5)
    with pytest.raises(EvalFailure) as err:
        backend.perceiver(1, 0, {}).reset_episode(None)
    assert err.value.reason == "watchdog_perception" and env.holds == 1
