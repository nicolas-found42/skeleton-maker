# SPDX-License-Identifier: MIT
"""Tracking association and motion rules used by the real environment worker."""

import json

import cv2
import numpy as np

from skeleton_maker import environment
from skeleton_maker.workers import grounded_sam2_da3 as worker
from tests.env_fakes import FakeBackend
from tests.test_environment import _make_video, _run, _run_failing


def _mask(rect, size=(12, 8)):
    x0, y0, x1, y1 = rect
    result = np.zeros((size[1], size[0]), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def test_same_class_detections_match_distinct_propagated_video_masks():
    propagated = {
        3: {"entity": "shot-0/chair-0", "label": "chair", "mask": _mask((0, 0, 4, 8))},
        8: {"entity": "shot-0/chair-1", "label": "chair", "mask": _mask((8, 0, 12, 8))},
    }
    detections = [
        {"status": "matched", "label": "chair", "box": [0, 0, 4, 8]},
        {"status": "matched", "label": "chair", "box": [8, 0, 12, 8]},
    ]

    result = worker._associate_detections(propagated, detections)

    assert [result[index]["entity"] for index in range(2)] == [
        "shot-0/chair-0",
        "shot-0/chair-1",
    ]
    assert all(result[index]["method"] == "sam2_video_mask_coverage" for index in range(2))


def test_competing_propagated_masks_leave_identity_uncertain():
    propagated = {
        3: {"entity": "shot-0/chair-0", "label": "chair", "mask": _mask((0, 0, 7, 8))},
        8: {"entity": "shot-0/chair-1", "label": "chair", "mask": _mask((5, 0, 12, 8))},
    }
    detection = [{"status": "matched", "label": "chair", "box": [3, 0, 9, 8]}]

    result = worker._associate_detections(propagated, detection)

    assert result == {
        0: {
            "entity": None,
            "score": 0.667,
            "method": "sam2_video_mask_coverage",
            "uncertain_candidates": ["shot-0/chair-0", "shot-0/chair-1"],
        }
    }


def test_empty_space_inside_a_propagated_mask_bbox_does_not_support_identity():
    mask = np.zeros((8, 12), dtype=bool)
    mask[:, :2] = True
    mask[:, 10:] = True
    propagated = {3: {"entity": "shot-0/chair-0", "label": "chair", "mask": mask}}
    detection = [{"status": "matched", "label": "chair", "box": [4, 0, 8, 8]}]

    result = worker._associate_detections(propagated, detection)

    assert result[0]["entity"] is None
    assert result[0]["score"] == 0.0


def test_motion_is_based_on_camera_relative_displacement_and_can_stay_unknown():
    stable = [{"center": [10, 10]}, {"center": [15, 10]}]
    moving = [{"center": [10, 10]}, {"center": [35, 10]}]

    assert worker._motion_from_centers(stable, [(5, 0)], diagonal=100) == "static"
    assert worker._motion_from_centers(moving, [(5, 0)], diagonal=100) == "dynamic"
    assert worker._motion_from_centers(stable[:1], [], diagonal=100) == "unknown"
    assert worker._motion_from_centers(stable, [(5, 0, 3)], diagonal=100) == "unknown"


def test_gaps_list_only_missing_sampled_frames_and_mark_reacquisition():
    track = {"status": "active", "last_observed_frame": 10, "gaps": []}

    worker._mark_track_missing(track, 20)
    worker._mark_track_missing(track, 40)
    worker._mark_track_observed(track, 50)

    assert track == {
        "status": "active",
        "last_observed_frame": 50,
        "gaps": [
            {
                "sampled_frame_ids": [20, 40],
                "reacquired_frame": 50,
                "status": "reacquired",
                "reason": "no detector box confirmed the propagated mask; occlusion vs missed detection is unknown",
            }
        ],
    }


def test_public_environment_command_rejects_tracking_gaps_on_unscanned_frames(
    tmp_path, monkeypatch, capsys
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip, frames=30)

    def invalid_gap(response):
        entity = response["entities"][1]
        entity["track"] = {
            "method": "sam2_video_propagation",
            "status": "lost",
            "first_observed_frame": 0,
            "last_observed_frame": 15,
            "gaps": [{"sampled_frame_ids": [1], "reacquired_frame": None, "status": "lost"}],
            "association_uncertainty": [],
        }

    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(mutate=invalid_gap))

    error = _run_failing(
        [str(clip), "--out", str(tmp_path / "environment.json"), "--backend", "fake"], capsys
    )

    assert "track.gaps[0].sampled_frame_ids" in error
    assert "was not a processed frame" in error


def test_public_manifest_keeps_same_class_ids_and_a_reacquired_sampled_gap(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_video(clip, frames=60)
    entities = []
    observations = []

    class TrackingBackend(FakeBackend):
        def run(self, request, assets_dir):
            response = super().run(request, assets_dir)
            frames = [frame["frame_id"] for frame in request["frames"]]
            assert frames == [0, 15, 30, 45]
            (assets_dir / "masks").mkdir(exist_ok=True)
            for index, x0 in enumerate((4, 36)):
                entity_id = f"shot-0/chair-{index}"
                gap = (
                    [
                        {
                            "sampled_frame_ids": [15],
                            "reacquired_frame": 30,
                            "status": "reacquired",
                            "reason": "fixture occlusion interval",
                        }
                    ]
                    if index == 1
                    else []
                )
                entities.append(
                    {
                        "id": entity_id,
                        "shot": "shot-0",
                        "family": "object",
                        "motion": "dynamic" if index else "static",
                        "labels": {
                            "requested": "chair",
                            "native": "chair",
                            "normalized": "chair",
                            "status": "matched",
                        },
                        "track": {
                            "method": "sam2_video_propagation",
                            "status": "active",
                            "first_observed_frame": 0,
                            "last_observed_frame": 45,
                            "gaps": gap,
                            "association_uncertainty": [],
                        },
                    }
                )
                for frame_id in (0, 15, 30, 45):
                    if index == 1 and frame_id == 15:
                        continue
                    mask_name = f"chair-{index}-{frame_id}.png"
                    image = np.zeros((48, 64), np.uint8)
                    image[5:25, x0 : x0 + 8] = 255
                    cv2.imwrite(str(assets_dir / "masks" / mask_name), image)
                    observations.append(
                        {
                            "id": f"obs-{index}-{frame_id}",
                            "entity": entity_id,
                            "frame_id": frame_id,
                            "bbox": [x0, 5, x0 + 8, 25],
                            "mask": {"asset": f"masks/{mask_name}"},
                            "score": 0.8,
                            "score_meaning": "fixture score",
                            "visibility": "visible",
                        }
                    )
            response["entities"] = entities
            response["observations"] = observations
            return response

    backend = TrackingBackend()
    monkeypatch.setitem(environment.BACKENDS, "tracking-fixture", backend)
    out = tmp_path / "environment.json"

    assert (
        _run(
            [
                str(clip),
                "--out",
                str(out),
                "--backend",
                "tracking-fixture",
                "--sample-fps",
                "2",
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )

    manifest = json.loads(out.read_text())
    chairs = [
        entity for entity in manifest["entities"] if entity["labels"]["normalized"] == "chair"
    ]
    assert [entity["id"] for entity in chairs] == ["shot-0/chair-0", "shot-0/chair-1"]
    assert {entity["motion"] for entity in chairs} == {"static", "dynamic"}
    assert {obs["entity"] for obs in manifest["observations"] if obs["frame_id"] == 0} == {
        "shot-0/chair-0",
        "shot-0/chair-1",
    }
    assert chairs[1]["track"]["gaps"] == [
        {
            "sampled_frame_ids": [15],
            "reacquired_frame": 30,
            "status": "reacquired",
            "reason": "fixture occlusion interval",
        }
    ]


def test_public_cli_runs_video_propagation_lifecycle_with_stubbed_inference(tmp_path, monkeypatch):
    """Exercise the worker orchestration through CLI/artifact validation without model weights."""
    clip = tmp_path / "clip.mp4"
    _make_video(clip, frames=90)
    frame_ids = (0, 15, 30, 45, 60, 75)

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class TorchStub:
        class _Context:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        @staticmethod
        def inference_mode():
            return TorchStub._Context()

        @staticmethod
        def no_grad():
            return TorchStub._Context()

    class Predictor:
        def init_state(self, _video_path, **_kwargs):
            return {"obj_ids": [], "boxes": {}}

        @staticmethod
        def _masks(state):
            masks = []
            for object_id in state["obj_ids"]:
                x0, y0, x1, y1 = state["boxes"][object_id]
                mask = np.zeros((48, 64), dtype=np.float32)
                mask[int(y0) : int(y1), int(x0) : int(x1)] = 1
                masks.append(mask[None, ...])
            return Tensor(masks)

        def add_new_points_or_box(self, state, *, frame_idx, obj_id, box):
            if obj_id not in state["boxes"]:
                state["obj_ids"].append(obj_id)
            state["boxes"][obj_id] = [float(value) for value in box]
            return frame_idx, state["obj_ids"], self._masks(state)

        def propagate_in_video(self, state, *, start_frame_idx, max_frame_num_to_track):
            assert max_frame_num_to_track == 1
            yield start_frame_idx + 1, state["obj_ids"], self._masks(state)

    class ImageStub:
        def __init__(self, array):
            self.array = array
            self.size = (array.shape[1], array.shape[0])

        @classmethod
        def fromarray(cls, array):
            return cls(array)

    class Backend:
        name = "worker-lifecycle-fixture"

        def available_devices(self):
            return ["cpu"]

        def supports_geometry(self):
            return False

        def max_frames(self):
            return 20

        def cache_identity(self):
            return None

        def run(self, request, assets_dir):
            request_path = tmp_path / "worker-request.json"
            response_path = tmp_path / "worker-response.json"
            request_path.write_text(json.dumps(request))
            assert worker.run(str(request_path), str(assets_dir), str(response_path)) == 0
            return json.loads(response_path.read_text())

    class ModelStub:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    class ProcessorStub:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    import sys
    import types

    torch_stub = types.ModuleType("torch")
    torch_stub.__dict__.update(
        {"inference_mode": TorchStub.inference_mode, "no_grad": TorchStub.no_grad}
    )
    torchvision_stub = types.ModuleType("torchvision")
    sam2_module = types.ModuleType("sam2")
    sam2_build_module = types.ModuleType("sam2.build_sam")
    sam2_build_module.__dict__["build_sam2_video_predictor"] = lambda *_args, **_kwargs: Predictor()
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.__dict__.update(
        {"AutoModelForZeroShotObjectDetection": ModelStub, "AutoProcessor": ProcessorStub}
    )
    pil_stub = types.ModuleType("PIL")
    pil_stub.__dict__["Image"] = ImageStub
    for name, module in {
        "torch": torch_stub,
        "torchvision": torchvision_stub,
        "sam2": sam2_module,
        "sam2.build_sam": sam2_build_module,
        "transformers": transformers_stub,
        "PIL": pil_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    model_root = tmp_path / "models"
    patched_models = {}
    for key, info in worker.MODELS.items():
        model_dir = model_root / info["directory"]
        model_dir.mkdir(parents=True)
        (model_dir / info["weights"]).write_bytes(b"test-only checkpoint stub")
        patched_models[key] = {**info, "sha256": worker._sha256(model_dir / info["weights"])}
    monkeypatch.setattr(worker, "MODELS", patched_models)
    monkeypatch.setattr(
        worker, "_model_path", lambda key: model_root / patched_models[key]["directory"]
    )
    monkeypatch.setattr(
        worker,
        "_read_frames",
        lambda _video, _ids: iter(
            (frame_id, np.full((48, 64, 3), frame_id, dtype=np.uint8)) for frame_id in frame_ids
        ),
    )
    monkeypatch.setattr(worker, "_nms", lambda _vision, _torch, group: list(range(len(group))))
    detections_by_frame = {
        0: [(5, 5, 15, 15), (30, 5, 40, 15)],
        15: [(5, 5, 15, 15)],
        30: [(5, 5, 15, 15), (30, 5, 40, 15), (50, 5, 60, 15)],
        45: [(5, 5, 15, 15)],
        60: [(5, 5, 15, 15)],
        75: [(5, 5, 15, 15)],
    }

    def detect(_processor, _model, image, labels, _device, _torch):
        if "chair" not in labels:
            return []
        frame_id = int(image.array[0, 0, 0])
        return [
            {
                "status": "matched",
                "label": "chair",
                "phrase": "chair",
                "candidates": [],
                "score": 0.9,
                "box": [float(value) for value in box],
            }
            for box in detections_by_frame[frame_id]
        ]

    monkeypatch.setattr(worker, "_detect", detect)
    monkeypatch.setattr(
        environment.shots,
        "detect_cuts",
        lambda *_args: [{"frame_id": 45, "distance": 1.0, "isolation": 10.0}],
    )
    monkeypatch.setitem(environment.BACKENDS, "worker-lifecycle-fixture", Backend())
    output = tmp_path / "manifest.json"

    assert (
        _run(
            [
                str(clip),
                "--out",
                str(output),
                "--backend",
                "worker-lifecycle-fixture",
                "--classes",
                "chair",
                "--sample-fps",
                "2",
                "--geometry",
                "off",
                "--no-cache",
            ]
        )
        == 0
    )

    manifest = json.loads(output.read_text())
    chairs = manifest["entities"]
    first_shot = [entity for entity in chairs if entity["shot"] == "shot-0"]
    second_shot = [entity for entity in chairs if entity["shot"] == "shot-1"]
    assert [entity["id"] for entity in first_shot] == [
        "shot-0/chair-0",
        "shot-0/chair-1",
        "shot-0/chair-2",
    ]
    assert [entity["id"] for entity in second_shot] == ["shot-1/chair-0"]
    assert first_shot[1]["track"]["gaps"] == [
        {
            "sampled_frame_ids": [15],
            "reacquired_frame": 30,
            "status": "reacquired",
            "reason": "no detector box confirmed the propagated mask; occlusion vs missed detection is unknown",
        }
    ]
    observed = {(item["entity"], item["frame_id"]) for item in manifest["observations"]}
    assert ("shot-0/chair-1", 15) not in observed
    assert ("shot-0/chair-1", 30) in observed
    assert ("shot-0/chair-2", 30) in observed
    assert ("shot-1/chair-0", 45) in observed
