# SPDX-License-Identifier: MIT
"""The public environment command's isolated geometry adapter and calibration contract."""

import copy
import json

import numpy as np
import pytest

from skeleton_maker import cli, environment, envmanifest, envworkers

from .env_fakes import FakeBackend
from .test_environment import _make_video


def _reference(width=64, height=48):
    return {
        "schema": "skeleton-maker.geometry-reference/1",
        "image_size": [width, height],
        "world_frame": {"id": "surveyed-world", "units": "m", "handedness": "right"},
        "camera": {
            "intrinsics": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
            "distortion": {"model": "none", "coefficients": []},
        },
        "anchors": [
            {
                "id": "fit-anchor",
                "frame_id": 0,
                "pixel": [10.0, 12.0],
                "position_m": [0.0, 0.0, 0.0],
                "role": "fit",
                "uncertainty_m": 0.01,
            },
            {
                "id": "check-anchor",
                "frame_id": 0,
                "pixel": [45.0, 12.0],
                "position_m": [2.0, 0.0, 0.0],
                "role": "check",
                "uncertainty_m": 0.01,
            },
        ],
        "measured_dimensions": [
            {
                "id": "known-width",
                "a": {"frame_id": 0, "pixel": [10.0, 12.0]},
                "b": {"frame_id": 0, "pixel": [45.0, 12.0]},
                "length_m": 2.0,
                "role": "check",
                "uncertainty_m": 0.01,
            }
        ],
    }


class FakeGeometryWorker:
    name = "da3-small"

    def __init__(self, frame_limit=600):
        self.requests = []
        self.frame_limit = frame_limit

    def available_devices(self):
        return ["cpu"]

    def supports_geometry(self):
        return True

    def max_frames(self):
        return self.frame_limit

    def cache_identity(self):
        return {"model": "test-da3-small", "checkpoint_sha256": "a" * 64}

    def run(self, request, assets_dir):
        self.requests.append(request)
        target = assets_dir / "geometry" / "depth" / "0.npy"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"test depth asset")
        return {
            "status": "relative-camera-frame",
            "reason": "DA3 depth has no measured metric scale",
            "units": "relative_depth",
            "coordinate_convention": "camera: +x right, +y down, +z forward; extrinsics world-to-camera",
            "scale_provenance": {"kind": "relative_model_prediction", "metric": False},
            "static_fusion_entities": request["geometry_filter"]["static_entity_ids"],
            "excluded_dynamic_entities": request["geometry_filter"]["excluded_entities"],
            "excluded_skeletons": request["geometry_filter"]["excluded_skeletons"],
            "skeleton_exclusion": request["geometry_filter"]["skeleton_exclusion"],
            "frames": [
                {
                    "frame_id": 0,
                    "shot": "shot-0",
                    "depth_asset": "geometry/depth/0.npy",
                    "intrinsics": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]],
                    "extrinsics_w2c": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    "depth_pixel_space": "processed_frame_pixels",
                    "preprocessing": {
                        "process_res": 504,
                        "process_res_method": "upper_bound_resize",
                        "source_size": [64, 48],
                        "processed_size": [56, 42],
                        "undistorted_source_to_processed": [
                            [0.875, 0.0, 0.0],
                            [0.0, 0.875, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "processed_to_undistorted_source": [
                            [1.142857, 0.0, 0.0],
                            [0.0, 1.142857, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "depth_resampling": "none; native DA3 processed depth grid",
                        "intrinsics_pixel_space": "undistorted_source_frame_pixels",
                        "crop": None,
                        "lens_transform": {"model": "none", "applied": False},
                    },
                }
            ],
        }


class KnownCameraGeometryWorker(FakeGeometryWorker):
    """Known synthetic DA3 camera for public CLI calibration integration tests."""

    def run(self, request, assets_dir):
        result = super().run(request, assets_dir)
        depth_path = assets_dir / "geometry" / "depth" / "0.npy"
        np.save(depth_path, np.full((42, 56), 10.0, dtype=np.float32), allow_pickle=False)
        result["frames"][0]["intrinsics"] = request["geometry_reference"]["camera"]["intrinsics"]
        return result


def _known_camera_reference(skew=0.0):
    pixels = [[32.0, 24.0], [40.0, 24.0], [32.0, 32.0], [40.0, 32.0]]
    intrinsics = [[50.0, skew, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
    scene_points = [
        (np.linalg.solve(intrinsics, [pixel[0], pixel[1], 1.0]) * 10.0).tolist() for pixel in pixels
    ]
    return {
        "schema": "skeleton-maker.geometry-reference/1",
        "image_size": [64, 48],
        "world_frame": {"id": "known-world", "units": "m", "handedness": "right"},
        "camera": {
            "intrinsics": intrinsics,
            "distortion": {"model": "none", "coefficients": []},
        },
        "anchors": [
            {
                "id": f"anchor-{i}",
                "frame_id": 0,
                "pixel": pixels[i],
                "position_m": [4.0 + 2.0 * p[0], -1.0 + 2.0 * p[1], 3.0 + 2.0 * p[2]],
                "role": "fit" if i < 3 else "check",
                "uncertainty_m": 0.01,
            }
            for i, p in enumerate(scene_points)
        ],
        "measured_dimensions": [
            {
                "id": "independent-width",
                "a": {"frame_id": 0, "pixel": pixels[0]},
                "b": {"frame_id": 0, "pixel": pixels[1]},
                "length_m": 3.2,
                "role": "check",
                "uncertainty_m": 0.01,
            }
        ],
    }


@pytest.mark.parametrize("skew", [0.0, 2.0])
def test_required_metric_registration_uses_independent_da3_worker_through_public_cli(
    tmp_path, monkeypatch, skew
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(_known_camera_reference(skew)))
    out = tmp_path / "environment.json"
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = KnownCameraGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(out),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
                "--calibration",
                str(reference),
                "--geometry",
                "required",
            ]
        )
        == 0
    )

    manifest = json.loads(out.read_text())
    assert not semantic.supports_geometry()
    assert len(geometry.requests) == 1
    assert manifest["run"]["status"] == "complete"
    assert manifest["geometry"]["status"] == "registered_metric"
    assert manifest["geometry"]["registration"]["target_frame"] == "known-world"
    assert manifest["geometry"]["registration"]["check_anchor_ids"] == ["anchor-3"]
    assert manifest["geometry"]["registration"]["check_dimension_ids"] == ["independent-width"]


@pytest.mark.parametrize("position_offset_m", [0.0, 0.001])
def test_required_metric_registration_preserves_prior_output_when_check_reuses_fit_observation(
    position_offset_m, tmp_path, monkeypatch, capsys
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    reference_data = _known_camera_reference()
    reference_data["anchors"][3] = {
        **copy.deepcopy(reference_data["anchors"][0]),
        "id": "renamed-check-anchor",
        "role": "check",
    }
    reference_data["anchors"][3]["position_m"][0] += position_offset_m
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(reference_data))
    out = tmp_path / "environment.json"
    out.write_text("previous manifest")
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = KnownCameraGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(out),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
                "--calibration",
                str(reference),
                "--geometry",
                "required",
            ]
        )

    assert exc.value.code != 0
    assert "reuses a fit observation" in capsys.readouterr().err
    assert len(geometry.requests) == 1
    assert out.read_text() == "previous manifest"


def test_required_da3_without_calibration_fails_before_semantic_inference(
    tmp_path, monkeypatch, capsys
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment",
                str(clip),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
                "--geometry",
                "required",
            ]
        )

    assert exc.value.code == environment.EXIT_INVALID
    assert "three identified fit anchors" in capsys.readouterr().err
    assert semantic.requests == []
    assert geometry.requests == []


def test_geometry_auto_uses_da3_and_records_calibration_and_relative_scale(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference()))
    manifest_path = tmp_path / "environment.json"
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(manifest_path),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
                "--calibration",
                str(reference_path),
            ]
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    assert len(geometry.requests) == 1
    assert geometry.requests[0]["geometry_reference"] == _reference()
    static_ids = geometry.requests[0]["geometry_filter"]["static_entity_ids"]
    excluded_id = geometry.requests[0]["geometry_filter"]["excluded_entities"][0]["id"]
    assert len(static_ids) == 1
    assert excluded_id != static_ids[0]
    assert geometry.requests[0]["geometry_filter"]["excluded_entities"][0]["motion"] == "unknown"
    assert geometry.requests[0]["geometry_filter"]["excluded_entities"][0]["bbox_frame_ids"] == [
        geometry.requests[0]["geometry_filter"]["excluded_entities"][0]["observed_frame_ids"][0]
    ]
    assert manifest["geometry"]["status"] == "relative-camera-frame"
    assert manifest["geometry"]["static_fusion_entities"] == static_ids
    assert manifest["geometry"]["excluded_dynamic_entities"][0]["id"] == excluded_id
    assert manifest["geometry"]["skeleton_exclusion"]["status"] == "unavailable"
    assert manifest["geometry"]["units"] == "relative_depth"
    assert manifest["geometry"]["scale_provenance"]["metric"] is False
    assert manifest["geometry"]["calibration"]["anchors"] == _reference()["anchors"]
    assert (
        manifest["geometry"]["calibration"]["measured_dimensions"]
        == _reference()["measured_dimensions"]
    )
    assert manifest["geometry"]["frames"][0]["preprocessing"]["lens_transform"]["applied"] is False
    assert "geometry/depth/0.npy" in {asset["path"] for asset in manifest["assets"]}


def test_geometry_filter_excludes_dynamic_and_unknown_and_keeps_static_candidates():
    response = {
        "entities": [
            {"id": "floor", "motion": "static"},
            {"id": "person", "motion": "dynamic", "family": "person"},
            {"id": "uncertain", "motion": "unknown", "family": "object"},
        ],
        "observations": [
            {"entity": "floor", "frame_id": 0, "mask": None},
            {"entity": "person", "frame_id": 1, "mask": {"asset": "masks/person.png"}},
            {
                "entity": "uncertain",
                "frame_id": 2,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "mask": None,
            },
        ],
    }

    geometry_filter = environment._geometry_filter(response)

    assert geometry_filter["static_entity_ids"] == ["floor"]
    assert [entity["id"] for entity in geometry_filter["excluded_entities"]] == [
        "person",
        "uncertain",
    ]
    assert geometry_filter["excluded_entities"][0]["masked_frame_ids"] == [1]
    assert geometry_filter["excluded_entities"][1]["bbox_frame_ids"] == [2]
    assert geometry_filter["excluded_pixel_regions"] == [
        {"frame_id": 1, "asset": "masks/person.png"},
        {"frame_id": 2, "bbox": [0.0, 0.0, 10.0, 10.0]},
    ]


def test_geometry_auto_keeps_semantic_result_when_da3_request_exceeds_preflight_limit(
    tmp_path, monkeypatch
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker(frame_limit=1)
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(tmp_path / "environment.json"),
                "--backend",
                "fake",
                "--sample-fps",
                "2",
                "--no-cache",
            ]
        )
        == 0
    )

    manifest = json.loads((tmp_path / "environment.json").read_text())
    assert len(semantic.requests) == 1
    assert geometry.requests == []
    assert manifest["run"]["status"] == "partial"
    assert manifest["geometry"]["status"] == "unavailable"
    assert "limit is 1 sampled frame" in manifest["geometry"]["reason"]


def test_geometry_required_rejects_over_limit_before_semantic_inference(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    semantic = FakeBackend(geometry=True, identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker(frame_limit=1)
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(tmp_path / "environment.json"),
                "--backend",
                "fake",
                "--geometry",
                "required",
                "--sample-fps",
                "2",
                "--no-cache",
            ]
        )

    assert exc.value.code != 0
    assert semantic.requests == []
    assert geometry.requests == []
    assert not (tmp_path / "environment.json").exists()


def test_public_geometry_request_excludes_supplied_pose_boxes_from_static_depth(
    tmp_path, monkeypatch
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    pose_path = tmp_path / "poses.jsonl"
    pose_path.write_text(
        "".join(
            json.dumps(
                {
                    "frame_id": frame_id,
                    "detections": [
                        {
                            "tracking_id": 7,
                            "bbox": [20.0, 10.0, 20.0, 25.0],
                            "root_pose": {
                                "translation": [0.0, 0.0, 3.0],
                                "rotation": [0.0, 0.0, 0.0, 1.0],
                            },
                        }
                    ],
                }
            )
            + "\n"
            for frame_id in range(30)
        )
    )
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(tmp_path / "environment.json"),
                "--poses",
                str(pose_path),
                "--backend",
                "fake",
                "--sample-fps",
                "1",
                "--no-cache",
            ]
        )
        == 0
    )

    geometry_filter = geometry.requests[0]["geometry_filter"]
    assert geometry_filter["skeleton_exclusion"] == {
        "status": "applied",
        "reason": "supplied pose boxes excluded from static geometry at sampled frames",
    }
    assert geometry_filter["excluded_skeletons"] == [
        {
            "id": "nim-skeleton:7",
            "tracking_id": 7,
            "reason": "supplied NIM skeleton bbox excluded from static geometry",
            "observed_frame_ids": [0],
            "bbox_frame_ids": [0],
            "unmasked_frame_ids": [],
        }
    ]
    assert {
        "frame_id": 0,
        "bbox": [20.0, 10.0, 40.0, 35.0],
        "source": "nim_skeleton",
    } in geometry_filter["excluded_pixel_regions"]


@pytest.mark.parametrize("changed_input", ["calibration", "pose-bbox"])
def test_geometry_cache_key_changes_when_geometry_inputs_change(
    tmp_path, monkeypatch, changed_input
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    out = tmp_path / "environment.json"
    cache_dir = tmp_path / "cache"
    calibration_path = tmp_path / "reference.json"
    calibration = _reference()
    calibration_path.write_text(json.dumps(calibration))
    pose_path = tmp_path / "poses.jsonl"

    def write_pose(x):
        pose_path.write_text(
            "".join(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "detections": [
                            {
                                "tracking_id": 7,
                                "bbox": [x, 10.0, 20.0, 25.0],
                                "root_pose": {
                                    "translation": [0.0, 0.0, 3.0],
                                    "rotation": [0.0, 0.0, 0.0, 1.0],
                                },
                            }
                        ],
                    }
                )
                + "\n"
                for frame_id in range(30)
            )
        )

    write_pose(20.0)
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    geometry = FakeGeometryWorker()
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: geometry)

    def run():
        args = [
            "environment",
            str(clip),
            "--out",
            str(out),
            "--backend",
            "fake",
            "--sample-fps",
            "1",
            "--cache-dir",
            str(cache_dir),
        ]
        if changed_input == "pose-bbox":
            args.extend(["--poses", str(pose_path)])
        else:
            args.extend(["--calibration", str(calibration_path)])
        return cli.main(args)

    assert run() == 0
    if changed_input == "calibration":
        calibration["camera"]["intrinsics"][0][0] = 75.0
        calibration_path.write_text(json.dumps(calibration))
    else:
        write_pose(25.0)
    assert run() == 0

    assert len(geometry.requests) == 2
    if changed_input == "calibration":
        assert geometry.requests[1]["geometry_reference"]["camera"]["intrinsics"][0][0] == 75.0
    else:
        assert {
            "frame_id": 0,
            "bbox": [25.0, 10.0, 45.0, 35.0],
            "source": "nim_skeleton",
        } in geometry.requests[1]["geometry_filter"]["excluded_pixel_regions"]


@pytest.mark.parametrize("missing_key", ["depth_resampling", "intrinsics_pixel_space"])
def test_manifest_rejects_missing_processed_depth_metadata_as_manifest_error(
    tmp_path, monkeypatch, missing_key
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: FakeGeometryWorker())
    manifest_path = tmp_path / "environment.json"
    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(manifest_path),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry"]["frames"][0]["preprocessing"].pop(missing_key)

    with pytest.raises(envmanifest.ManifestError):
        envmanifest.validate_manifest(manifest)


@pytest.mark.parametrize(
    "destination",
    ["manifest", "manifest-assets", "viewer", "viewer-assets"],
)
def test_environment_outputs_cannot_replace_or_contain_calibration(
    tmp_path, monkeypatch, capsys, destination
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    semantic = FakeBackend(identity={"model": "fake-semantic"})
    monkeypatch.setitem(environment.BACKENDS, "fake", semantic)

    out = tmp_path / "environment.json"
    viewer = None
    if destination == "manifest":
        reference_path = out
    elif destination == "manifest-assets":
        reference_path = envmanifest.assets_dir_for(out) / "reference.json"
    elif destination == "viewer":
        reference_path = tmp_path / "environment.html"
        viewer = str(reference_path)
    else:
        viewer = str(tmp_path / "environment.html")
        reference_path = tmp_path / "environment.viewer.assets" / "reference.json"

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    original_reference = json.dumps(_reference())
    reference_path.write_text(original_reference)
    argv = [
        "environment",
        str(clip),
        "--out",
        str(out),
        "--backend",
        "fake",
        "--calibration",
        str(reference_path),
        "--no-cache",
    ]
    if viewer is not None:
        argv.extend(["--viewer", viewer])

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code != 0
    assert "calibration" in capsys.readouterr().err
    assert reference_path.read_text() == original_reference
    assert semantic.requests == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda ref: ref.update(image_size=[1280, 720]), "image_size does not match"),
        (lambda ref: ref["camera"]["intrinsics"][0].__setitem__(0, float("nan")), "finite"),
        (
            lambda ref: ref["camera"].update(
                intrinsics=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "invertible and well-conditioned",
        ),
        (lambda ref: ref["anchors"][0].update(frame_id=900), "outside the source frame range"),
        (lambda ref: ref["anchors"][0].update(pixel=[65.0, 1.0]), "outside the 64x48 source frame"),
    ],
)
def test_invalid_geometry_reference_fails_before_any_inference(
    tmp_path, monkeypatch, capsys, mutate, message
):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    reference = _reference()
    mutate(reference)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    with pytest.raises(SystemExit):
        cli.main(
            [
                "environment",
                str(clip),
                "--backend",
                "fake",
                "--calibration",
                str(reference_path),
            ]
        )

    assert message in capsys.readouterr().err
    assert backend.requests == []


def test_geometry_off_does_not_discover_da3_worker(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    backend = FakeBackend(identity={"model": "fake-semantic"})
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    def unexpected_discovery():
        raise AssertionError("off mode must not preflight the DA3 worker")

    monkeypatch.setattr(envworkers, "discover_geometry", unexpected_discovery)
    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--sample-fps",
                "0.05",
            ]
        )
        == 0
    )


def test_geometry_auto_records_unavailable_when_da3_worker_is_not_installed(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_video(clip)
    manifest_path = tmp_path / "environment.json"
    backend = FakeBackend(identity={"model": "fake-semantic"})
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: None)

    assert (
        cli.main(
            [
                "environment",
                str(clip),
                "--out",
                str(manifest_path),
                "--backend",
                "fake",
                "--sample-fps",
                "0.05",
                "--no-cache",
            ]
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["run"]["status"] == "partial"
    assert manifest["geometry"]["status"] == "unavailable"
    assert "not installed" in manifest["geometry"]["reason"]
