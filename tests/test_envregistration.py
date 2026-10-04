# SPDX-License-Identifier: MIT
"""Known-camera numerical regressions for camera and scene registration."""

import copy
import json

import numpy as np
import pytest

from skeleton_maker import envregistration, poses


def _fixture(tmp_path):
    depth_dir = tmp_path / "assets" / "geometry" / "depth"
    depth_dir.mkdir(parents=True)
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 3.0
    depth[1, 0] = 2.0
    depth[0, 1] = 2.0
    depth[0, 0] = 2.0
    depth_path = depth_dir / "frame.npy"
    np.save(depth_path, depth, allow_pickle=False)
    k = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    t_wc = np.array([0.5, -0.25, 0.5])
    camera_points = np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    scene_points = camera_points - t_wc
    angle = np.pi / 2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    scale = 2.5
    translation = np.array([8.0, -3.0, 4.0])
    world_points = scale * (rotation @ scene_points.T).T + translation
    anchors = []
    for i, (pixel, point) in enumerate(
        zip([[0, 0], [1, 0], [0, 1], [1, 1]], world_points, strict=True)
    ):
        anchors.append(
            {
                "id": f"control-{i}",
                "frame_id": 1 if i == 3 else 0,
                "pixel": pixel,
                "position_m": point.tolist(),
                "role": "fit" if i < 3 else "check",
                "uncertainty_m": 0.001,
            }
        )
    frame = {
        "frame_id": 0,
        "shot": "shot-0",
        "depth_asset": "geometry/depth/frame.npy",
        "intrinsics": k,
        "extrinsics_w2c": [
            [1.0, 0.0, 0.0, t_wc[0]],
            [0.0, 1.0, 0.0, t_wc[1]],
            [0.0, 0.0, 1.0, t_wc[2]],
        ],
        "depth_pixel_space": "processed_frame_pixels",
        "preprocessing": {
            "source_size": [4, 4],
            "processed_size": [4, 4],
            "undistorted_source_to_processed": np.eye(3).tolist(),
            "processed_to_undistorted_source": np.eye(3).tolist(),
            "depth_resampling": "none",
            "intrinsics_pixel_space": "undistorted_source_frame_pixels",
            "crop": None,
            "lens_transform": {"model": "none", "applied": False},
        },
    }
    reference = {
        "world_frame": {"id": "survey-1", "units": "m", "handedness": "right"},
        "camera": {"intrinsics": k, "distortion": {"model": "none", "coefficients": []}},
        "anchors": anchors,
        "measured_dimensions": [
            {
                "id": "withheld-width",
                "a": {"frame_id": 0, "pixel": [0, 0]},
                "b": {"frame_id": 0, "pixel": [1, 0]},
                "length_m": 5.0,
                "uncertainty_m": 0.001,
                "role": "check",
            }
        ],
    }
    second_frame = copy.deepcopy(frame)
    second_frame["frame_id"] = 1
    geometry = {"frames": [frame, second_frame]}
    return geometry, reference, tmp_path / "assets", rotation, scale, translation, t_wc


def test_measured_fit_and_withheld_check_build_explicit_metric_transforms(tmp_path):
    geometry, reference, assets, rotation, scale, translation, t_wc = _fixture(tmp_path)

    result = envregistration.register_geometry(geometry, reference, assets)

    assert result["status"] == "registered_metric"
    assert result["scale_provenance"]["fit_anchor_ids"] == ["control-0", "control-1", "control-2"]
    assert result["scale_provenance"]["check_anchor_ids"] == ["control-3"]
    scene = np.asarray(result["registration"]["scene_transforms"][0]["matrix_4x4"])
    assert scene[:3, :3] == pytest.approx(scale * rotation, abs=1e-6)
    assert scene[:3, 3] == pytest.approx(translation, abs=1e-6)

    camera = result["registration"]["camera_transforms"][0]
    camera_to_world = np.asarray(camera["matrix_4x4"])
    nim_camera_point_m = np.array([1.0, 2.0, 3.0, 1.0])
    converted = camera_to_world @ nim_camera_point_m
    nim_to_da3 = np.diag([1.0, -1.0, -1.0])
    expected = (
        rotation @ nim_to_da3 @ nim_camera_point_m[:3] + translation - scale * rotation @ t_wc
    )
    assert converted[:3] == pytest.approx(expected, abs=1e-6)
    # NIM's serialized joint position is already camera-relative and includes the
    # body root translation. Applying that root again would move it twice.
    nim_root_translation_m = np.array([0.25, -0.5, 0.75])
    with_duplicate_root = expected + rotation @ nim_to_da3 @ nim_root_translation_m
    assert converted[:3] != pytest.approx(with_duplicate_root, abs=1e-6)
    assert converted[:3] != pytest.approx(
        scale * rotation @ nim_to_da3 @ nim_camera_point_m[:3]
        + translation
        - scale * rotation @ t_wc
    )
    assert camera["root_translation_applied"] is False
    assert camera["stage_transform_applied"] is False
    assert result["evaluation"]["frames"][0]["registered"] is True
    assert result["evaluation"]["frames"][0]["control_points"][0]["xy"] == pytest.approx(
        [1.0, 1.0], abs=1e-5
    )
    assert result["evaluation"]["dimensions"][0]["id"] == "withheld-width"
    assert result["evaluation"]["dimensions"][0]["meters"] == pytest.approx(5.0)
    assert result["evaluation"]["dimensions"][0]["units"] == "m"
    assert result["registration"]["check_dimension_ids"] == ["withheld-width"]


def test_check_anchor_cannot_reuse_a_fit_observation_under_a_new_id(tmp_path):
    geometry, reference, assets, *_ = _fixture(tmp_path)
    reused = copy.deepcopy(reference["anchors"][0])
    reused.update(id="renamed-check", role="check")
    reference["anchors"][-1] = reused

    with pytest.raises(envregistration.RegistrationError, match="reuses a fit observation"):
        envregistration.register_geometry(geometry, reference, assets)


def test_same_landmark_observed_in_a_different_frame_remains_independent(tmp_path):
    geometry, reference, assets, *_ = _fixture(tmp_path)
    reference["anchors"][-1].update(
        frame_id=1, pixel=[0, 0], position_m=copy.deepcopy(reference["anchors"][0]["position_m"])
    )

    result = envregistration.register_geometry(geometry, reference, assets)

    assert result["status"] == "registered_metric"


def test_raw_pose_record_uses_emitted_camera_transform_without_rescaling_or_readding_root(
    tmp_path,
):
    geometry, reference, assets, rotation, scale, translation, t_wc = _fixture(tmp_path)
    root = np.array([0.25, -0.5, 0.75])
    rest_joint = np.array([0.75, 2.5, 2.25])
    camera_joint = root + rest_joint
    detection = {
        "tracking_id": 7,
        "keypoints_3d": [camera_joint.tolist()] + [[0.0, 0.0, 1.0]] * 76,
        "root_pose": {"translation": root.tolist(), "rotation": [0.0, 0.0, 0.0, 1.0]},
    }
    pose_path = tmp_path / "pose.jsonl"
    pose_path.write_text(json.dumps({"frame_id": 0, "detections": [detection]}) + "\n")
    raw_record = poses.read_records(str(pose_path))[0]

    result = envregistration.register_geometry(geometry, reference, assets)
    camera_to_world = np.asarray(result["registration"]["camera_transforms"][0]["matrix_4x4"])
    joint = np.asarray(raw_record["detections"][0]["keypoints_3d"][0] + [1.0])
    actual = (camera_to_world @ joint)[:3]
    nim_to_da3 = np.diag([1.0, -1.0, -1.0])
    expected = rotation @ nim_to_da3 @ camera_joint + translation - scale * rotation @ t_wc
    rescaled = scale * rotation @ nim_to_da3 @ camera_joint + translation - scale * rotation @ t_wc
    root_added_again = expected + rotation @ nim_to_da3 @ root

    assert result["status"] == "registered_metric"
    assert actual == pytest.approx(expected, abs=1e-6)
    assert actual != pytest.approx(rescaled, abs=1e-6)
    assert actual != pytest.approx(root_added_again, abs=1e-6)
    assert result["registration"]["camera_transforms"][0]["root_translation_applied"] is False


def test_lens_corrected_camera_intrinsics_are_used_consistently(tmp_path):
    geometry, reference, assets, rotation, scale, translation, t_wc = _fixture(tmp_path)
    skewed_k = [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    reference["camera"]["intrinsics"] = skewed_k
    camera_points = np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [-4.0, 2.0, 2.0], [-3.0, 3.0, 3.0]])
    world_points = scale * (rotation @ (camera_points - t_wc).T).T + translation
    for anchor, point in zip(reference["anchors"], world_points, strict=True):
        anchor["position_m"] = point.tolist()
    reference["camera"]["distortion"] = {
        "model": "opencv-brown-conrady",
        "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    for frame in geometry["frames"]:
        frame["intrinsics"] = skewed_k
        frame["preprocessing"]["lens_transform"] = {
            "model": "opencv-brown-conrady",
            "applied": True,
            "source_intrinsics": reference["camera"]["intrinsics"],
            "source_coefficients": reference["camera"]["distortion"]["coefficients"],
            "output_intrinsics": reference["camera"]["intrinsics"],
        }

    result = envregistration.register_geometry(geometry, reference, assets)

    assert result["status"] == "registered_metric"


def test_metric_registration_backprojects_skewed_intrinsics(tmp_path):
    geometry, reference, assets, rotation, scale, translation, t_wc = _fixture(tmp_path)
    skewed_k = [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    reference["camera"]["intrinsics"] = skewed_k
    for frame in geometry["frames"]:
        frame["intrinsics"] = skewed_k
    camera_points = np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [-4.0, 2.0, 2.0], [-3.0, 3.0, 3.0]])
    world_points = scale * (rotation @ (camera_points - t_wc).T).T + translation
    for anchor, point in zip(reference["anchors"], world_points, strict=True):
        anchor["position_m"] = point.tolist()

    result = envregistration.register_geometry(geometry, reference, assets)

    assert result["status"] == "registered_metric"
    assert result["evaluation"]["frames"][0]["control_points"][0]["xy"] == pytest.approx(
        [1.0, 1.0], abs=1e-5
    )


def test_metric_registration_rejects_singular_intrinsics_before_depth_access(tmp_path):
    geometry, reference, assets, *_ = _fixture(tmp_path)
    singular = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    reference["camera"]["intrinsics"] = singular
    for frame in geometry["frames"]:
        frame["intrinsics"] = singular
    for depth_path in (assets / "geometry" / "depth").glob("*.npy"):
        depth_path.unlink()

    with pytest.raises(envregistration.RegistrationError, match="invertible and well-conditioned"):
        envregistration.register_geometry(geometry, reference, assets)


@pytest.mark.parametrize(
    "mutation",
    [
        "reversed_pixel_map",
        "noninvertible_pixel_map",
        "stale_resize_intrinsics",
        "stage_transform",
        "reversed_extrinsics",
    ],
)
def test_metric_registration_abstains_when_camera_or_stage_contract_is_invalid(tmp_path, mutation):
    geometry, reference, assets, *_ = _fixture(tmp_path)
    frame = geometry["frames"][0]
    if mutation == "reversed_pixel_map":
        frame["preprocessing"]["undistorted_source_to_processed"] = [
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ]
    elif mutation == "noninvertible_pixel_map":
        frame["preprocessing"]["undistorted_source_to_processed"] = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    elif mutation == "stale_resize_intrinsics":
        frame["preprocessing"]["processed_size"] = [2, 4]
        frame["preprocessing"]["undistorted_source_to_processed"] = [
            [0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        frame["preprocessing"]["processed_to_undistorted_source"] = [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        frame["intrinsics"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        frame["intrinsics"][0][0] = 1.5
    elif mutation == "stage_transform":
        frame["stage_floor_applied"] = True
    elif mutation == "reversed_extrinsics":
        geometry["frames"][1]["extrinsics_w2c"] = [
            [1, 0, 0, -0.5],
            [0, 1, 0, 0.25],
            [0, 0, 1, -0.5],
        ]

    with pytest.raises(envregistration.RegistrationError):
        envregistration.register_geometry(geometry, reference, assets)


def test_bad_independent_dimension_check_prevents_metric_claim(tmp_path):
    geometry, reference, assets, *_ = _fixture(tmp_path)
    reference["measured_dimensions"][0]["length_m"] = 2.5

    with pytest.raises(envregistration.RegistrationError, match="check anchors"):
        envregistration.register_geometry(geometry, reference, assets)
