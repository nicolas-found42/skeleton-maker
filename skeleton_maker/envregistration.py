# SPDX-License-Identifier: MIT
"""Independent calibration of DA3 relative geometry into a measured metric frame."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .envcamera import IntrinsicsError, backproject, validate_intrinsics


class RegistrationError(ValueError):
    """Calibration evidence cannot support a validated shared metric frame."""


def _intrinsics(value, where):
    try:
        return validate_intrinsics(value, where)
    except IntrinsicsError as exc:
        raise RegistrationError(str(exc)) from exc


def _matrix(value, shape, where):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise RegistrationError(f"{where} must be a finite {shape[0]}x{shape[1]} matrix")
    return array


def _proper_rotation(matrix: np.ndarray, where: str) -> np.ndarray:
    rotation = _matrix(matrix, (3, 3), where)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3
    ):
        raise RegistrationError(f"{where} must be a proper orthonormal rotation")
    return rotation


def _pixel_map(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = frame.get("preprocessing")
    if not isinstance(prep, dict):
        raise RegistrationError("DA3 preprocessing metadata is required for calibration")
    if prep.get("crop") is not None:
        raise RegistrationError(
            "cropped DA3 preprocessing is not supported for metric registration"
        )
    if frame.get("stage_floor_applied") or frame.get("stage_up_transform_applied"):
        raise RegistrationError(
            "person-derived stage floor/up transforms cannot register scene geometry"
        )
    if prep.get("intrinsics_pixel_space") != "undistorted_source_frame_pixels":
        raise RegistrationError("DA3 intrinsics are not in the supported undistorted source space")
    if frame.get("depth_pixel_space") != "processed_frame_pixels":
        raise RegistrationError("DA3 depth must use processed-frame pixels")
    source_to_processed = _matrix(
        prep.get("undistorted_source_to_processed"), (3, 3), "undistorted_source_to_processed"
    )
    processed_to_source = _matrix(
        prep.get("processed_to_undistorted_source"), (3, 3), "processed_to_undistorted_source"
    )
    if not np.allclose(source_to_processed[2], [0, 0, 1], atol=1e-9) or not np.allclose(
        processed_to_source[2], [0, 0, 1], atol=1e-9
    ):
        raise RegistrationError("DA3 pixel maps must be affine homogeneous transforms")
    if not np.allclose(source_to_processed @ processed_to_source, np.eye(3), atol=2e-6):
        raise RegistrationError("DA3 source-to-processed pixel maps are not mutual inverses")
    if abs(np.linalg.det(source_to_processed[:2, :2])) < 1e-12:
        raise RegistrationError("DA3 source-to-processed pixel map is not invertible")
    source_size = prep.get("source_size")
    processed_size = prep.get("processed_size")
    if (
        not isinstance(source_size, list)
        or len(source_size) != 2
        or not isinstance(processed_size, list)
        or len(processed_size) != 2
        or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0
            for x in source_size + processed_size
        )
    ):
        raise RegistrationError(
            "DA3 source_size and processed_size must be positive [width, height]"
        )
    expected = np.diag(
        [processed_size[0] / source_size[0], processed_size[1] / source_size[1], 1.0]
    )
    if not np.allclose(source_to_processed, expected, atol=2e-6):
        raise RegistrationError(
            "DA3 pixel map does not match the actual source and processed sizes"
        )
    return source_to_processed, processed_to_source, np.asarray(processed_size, dtype=np.int64)


def _source_pixel_to_depth(
    frame: dict, pixel: list[float], camera: dict, source_pixel_map: np.ndarray
) -> tuple[float, float]:
    point = np.asarray(pixel, dtype=np.float64).reshape(2)
    distortion = camera["distortion"]
    lens = frame["preprocessing"].get("lens_transform", {})
    if distortion["model"] == "opencv-brown-conrady":
        if lens.get("applied") is not True:
            raise RegistrationError("calibrated lens distortion was not applied by the DA3 worker")
        source_k = _intrinsics(camera["intrinsics"], "calibration camera intrinsics")
        corrected_k = _intrinsics(lens.get("output_intrinsics"), "DA3 corrected intrinsics")
        distorted_ray = np.linalg.solve(source_k, np.array([point[0], point[1], 1.0]))
        distorted_xy = distorted_ray[:2] / distorted_ray[2]
        undistorted_xy = cv2.undistortPoints(
            distorted_xy.reshape(1, 1, 2),
            np.eye(3, dtype=np.float64),
            np.asarray(distortion["coefficients"], dtype=np.float64),
        ).reshape(2)
        corrected_pixel = corrected_k @ np.array([*undistorted_xy, 1.0])
        point = corrected_pixel[:2] / corrected_pixel[2]
    elif lens.get("applied"):
        raise RegistrationError("DA3 applied a lens transform not described by calibration")
    projected = source_pixel_map @ np.array([point[0], point[1], 1.0])
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _camera_point(frame: dict, pixel: list[float], depth_root: Path, camera: dict) -> np.ndarray:
    source_to_processed, _, processed_size = _pixel_map(frame)
    u, v = _source_pixel_to_depth(frame, pixel, camera, source_to_processed)
    depth_path = (depth_root / frame["depth_asset"]).resolve()
    try:
        depth_path.relative_to(depth_root.resolve())
    except ValueError as exc:
        raise RegistrationError("DA3 depth asset escapes the staged asset directory") from exc
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2 or list(depth.shape[::-1]) != processed_size.tolist():
        raise RegistrationError(
            f"depth asset {frame['depth_asset']!r} does not match processed_size"
        )
    x = int(np.floor(u + 0.5))
    y = int(np.floor(v + 0.5))
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise RegistrationError("calibration point maps outside the DA3 depth grid")
    z = float(depth[y, x])
    if not math.isfinite(z) or z <= 0:
        raise RegistrationError("calibration point has no positive finite DA3 depth")
    k_source = _intrinsics(frame.get("intrinsics"), "DA3 source-space intrinsics")
    lens = frame["preprocessing"].get("lens_transform", {})
    expected_k = (
        _intrinsics(lens.get("output_intrinsics"), "DA3 corrected source intrinsics")
        if lens.get("applied")
        else _intrinsics(camera.get("intrinsics"), "calibration camera intrinsics")
    )
    if not np.allclose(k_source, expected_k, atol=1e-5, rtol=1e-5):
        raise RegistrationError(
            "DA3 source-space intrinsics do not match the calibration/lens-corrected camera"
        )
    k_depth = source_to_processed @ k_source
    camera_xyz = backproject(k_depth, u, v, z)
    extrinsics = _matrix(frame.get("extrinsics_w2c"), (3, 4), "DA3 world-to-camera extrinsics")
    rotation = _proper_rotation(extrinsics[:, :3], "DA3 world-to-camera rotation")
    return rotation.T @ (camera_xyz - extrinsics[:, 3])


def _anchor_observation(anchor: dict) -> tuple[int, tuple[float, ...]]:
    frame_id = anchor.get("frame_id")
    pixel = anchor.get("pixel")
    position = anchor.get("position_m")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise RegistrationError("fit/check anchors require integer frame IDs")
    if (
        not isinstance(pixel, list)
        or len(pixel) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (isinstance(value, int) or math.isfinite(value))
            for value in pixel
        )
    ):
        raise RegistrationError("fit/check anchors require finite source-frame pixel coordinates")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (isinstance(value, int) or math.isfinite(value))
            for value in position
        )
    ):
        raise RegistrationError("fit/check anchors require finite meter-valued positions")
    return frame_id, tuple(map(float, pixel))


def _fit_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RegistrationError("fit anchors must provide matching 3D points")
    total = float(weights.sum())
    if source.shape[0] < 3 or total <= 0:
        raise RegistrationError("at least three independent fit anchors are required")
    weights = weights / total
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    x = source - source_mean
    y = target - target_mean
    if np.linalg.matrix_rank(x, tol=1e-8) < 2:
        raise RegistrationError("fit anchors are collinear or coincident")
    covariance = (y * weights[:, None]).T @ x
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    variance = float(np.sum(weights * np.sum(x * x, axis=1)))
    if variance <= 1e-12:
        raise RegistrationError("fit anchors have no measurable 3D extent")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise RegistrationError("fit anchors imply an invalid metric scale")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _scene_to_world(
    source: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
):
    return (scale * (rotation @ source.T)).T + translation


def _project_world(
    point_world: list[float],
    frame: dict,
    scale: float,
    rotation_fit: np.ndarray,
    translation_fit: np.ndarray,
    camera: dict,
) -> list[float]:
    ext = _matrix(frame["extrinsics_w2c"], (3, 4), "DA3 world-to-camera extrinsics")
    rotation_wc = _proper_rotation(ext[:, :3], "DA3 world-to-camera rotation")
    scene = rotation_fit.T @ (np.asarray(point_world) - translation_fit) / scale
    camera_point = rotation_wc @ scene + ext[:, 3]
    if camera_point[2] <= 0:
        raise RegistrationError("independent check point projects behind the DA3 camera")
    distortion = camera["distortion"]
    if distortion["model"] == "opencv-brown-conrady":
        normalized = camera_point / camera_point[2]
        distorted, _ = cv2.projectPoints(
            normalized.reshape(1, 3),
            np.zeros(3),
            np.zeros(3),
            np.eye(3, dtype=np.float64),
            np.asarray(distortion["coefficients"], dtype=np.float64),
        )
        distorted_h = np.r_[distorted.reshape(2), 1.0]
        projected_h = (
            _intrinsics(camera["intrinsics"], "calibration camera intrinsics") @ distorted_h
        )
        xy = projected_h[:2] / projected_h[2]
    else:
        k = _intrinsics(camera["intrinsics"], "calibration camera intrinsics")
        pixel = k @ camera_point
        xy = pixel[:2] / pixel[2]
    return [float(xy[0]), float(xy[1])]


def _homogeneous(linear: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix.tolist()


def register_geometry(geometry: dict, calibration: dict, depth_root: Path) -> dict:
    """Return metric registration only with independent 3D fit and check anchors.

    Calibration anchors are never inferred from NIM people. A shot is registered only
    when it has three non-collinear fit anchors plus at least one independent check anchor.
    """
    world_frame = calibration.get("world_frame")
    if not isinstance(world_frame, dict) or not isinstance(world_frame.get("id"), str):
        raise RegistrationError("calibration world_frame.id is required for metric registration")
    if world_frame.get("units") != "m" or world_frame.get("handedness") != "right":
        raise RegistrationError("calibration world_frame must declare right-handed meter units")
    camera = calibration.get("camera")
    if not isinstance(camera, dict):
        raise RegistrationError("camera intrinsics are required for metric registration")
    _intrinsics(camera.get("intrinsics"), "calibration camera intrinsics")
    anchors = calibration.get("anchors", [])
    if not isinstance(anchors, list):
        raise RegistrationError("calibration anchors must be a list")
    frames_by_id = {frame["frame_id"]: frame for frame in geometry.get("frames", [])}
    if len(frames_by_id) != len(geometry.get("frames", [])):
        raise RegistrationError("DA3 geometry contains duplicate source frame IDs")
    reference_size = calibration.get("image_size")
    if reference_size is not None:
        for frame in geometry.get("frames", []):
            if frame.get("preprocessing", {}).get("source_size") != reference_size:
                raise RegistrationError(
                    "DA3 preprocessing source_size differs from calibration image_size"
                )
    by_shot: dict[str, list[dict]] = {}
    for frame in geometry.get("frames", []):
        by_shot.setdefault(frame["shot"], []).append(frame)
        _pixel_map(frame)
        _intrinsics(frame.get("intrinsics"), "DA3 source-space intrinsics")
    if not by_shot:
        raise RegistrationError("DA3 returned no camera frames")

    scene_to_world = []
    camera_transforms = []
    evaluation_frames_by_id = {}
    registration_checks = []
    fit_anchor_ids = []
    check_anchor_ids = []
    check_dimension_ids = []
    evaluation_dimensions = []
    for shot, shot_frames in by_shot.items():
        shot_ids = {frame["frame_id"] for frame in shot_frames}
        shot_anchors = [
            a for a in anchors if a["frame_id"] in shot_ids and a["role"] in ("fit", "check")
        ]
        fit_anchors = [a for a in shot_anchors if a["role"] == "fit"]
        check_anchors = [a for a in shot_anchors if a["role"] == "check"]
        if len(fit_anchors) < 3 or not check_anchors:
            raise RegistrationError(
                f"shot {shot!r} needs three fit anchors and an independent check anchor"
            )
        if any(
            not a.get("id")
            or not isinstance(a.get("uncertainty_m"), (int, float))
            or isinstance(a.get("uncertainty_m"), bool)
            or a["uncertainty_m"] <= 0
            for a in shot_anchors
        ):
            raise RegistrationError("fit/check anchors require IDs and measured uncertainty_m")
        fit_observations = {_anchor_observation(anchor) for anchor in fit_anchors}
        if any(_anchor_observation(anchor) in fit_observations for anchor in check_anchors):
            raise RegistrationError("independent check anchor reuses a fit observation")
        source_fit = np.stack(
            [
                _camera_point(frames_by_id[a["frame_id"]], a["pixel"], depth_root, camera)
                for a in fit_anchors
            ]
        )
        target_fit = np.asarray([a["position_m"] for a in fit_anchors], dtype=np.float64)
        weights = np.asarray([1.0 / (a["uncertainty_m"] ** 2) for a in fit_anchors])
        scale, rotation_fit, translation_fit = _fit_similarity(source_fit, target_fit, weights)
        if np.linalg.det(rotation_fit) < 0.999:
            raise RegistrationError("scene fit produced a reflected coordinate transform")
        fit_anchor_ids.extend(a["id"] for a in fit_anchors)
        check_anchor_ids.extend(a["id"] for a in check_anchors)
        fitted_predictions = _scene_to_world(source_fit, scale, rotation_fit, translation_fit)
        for anchor, prediction in zip(fit_anchors, fitted_predictions, strict=True):
            error = float(np.linalg.norm(prediction - anchor["position_m"]))
            registration_checks.append(
                {
                    "id": anchor["id"],
                    "kind": "fit_anchor_residual",
                    "error_m": error,
                    "uncertainty_m": anchor["uncertainty_m"],
                    "passed": error <= max(3 * anchor["uncertainty_m"], 0.02),
                }
            )
        for anchor in check_anchors:
            frame = frames_by_id[anchor["frame_id"]]
            source_check = _camera_point(frame, anchor["pixel"], depth_root, camera)
            predicted_world = _scene_to_world(
                source_check[None, :], scale, rotation_fit, translation_fit
            )[0]
            error = float(np.linalg.norm(predicted_world - anchor["position_m"]))
            passed = error <= max(3 * anchor["uncertainty_m"], 0.02)
            registration_checks.append(
                {
                    "id": anchor["id"],
                    "kind": "withheld_anchor_position",
                    "error_m": error,
                    "uncertainty_m": anchor["uncertainty_m"],
                    "passed": passed,
                }
            )
            evaluation = evaluation_frames_by_id.setdefault(
                anchor["frame_id"],
                {"frame_id": anchor["frame_id"], "registered": True, "control_points": []},
            )
            evaluation["registered"] = evaluation["registered"] and passed
            evaluation["control_points"].append(
                {
                    "id": anchor["id"],
                    "xy": _project_world(
                        anchor["position_m"],
                        frame,
                        scale,
                        rotation_fit,
                        translation_fit,
                        camera,
                    ),
                }
            )
        for dimension in calibration.get("measured_dimensions", []):
            if dimension.get("role") != "check":
                continue
            if (
                dimension["a"]["frame_id"] not in shot_ids
                or dimension["b"]["frame_id"] not in shot_ids
            ):
                continue
            uncertainty = dimension.get("uncertainty_m")
            if (
                not isinstance(uncertainty, (int, float))
                or isinstance(uncertainty, bool)
                or uncertainty <= 0
            ):
                raise RegistrationError(
                    f"check dimension {dimension['id']!r} requires positive uncertainty_m"
                )
            point_a = _camera_point(
                frames_by_id[dimension["a"]["frame_id"]],
                dimension["a"]["pixel"],
                depth_root,
                camera,
            )
            point_b = _camera_point(
                frames_by_id[dimension["b"]["frame_id"]],
                dimension["b"]["pixel"],
                depth_root,
                camera,
            )
            world_a = _scene_to_world(point_a[None, :], scale, rotation_fit, translation_fit)[0]
            world_b = _scene_to_world(point_b[None, :], scale, rotation_fit, translation_fit)[0]
            predicted_m = float(np.linalg.norm(world_b - world_a))
            error_m = abs(predicted_m - dimension["length_m"])
            passed = error_m <= max(3 * uncertainty, 0.1 * dimension["length_m"])
            check_dimension_ids.append(dimension["id"])
            registration_checks.append(
                {
                    "id": dimension["id"],
                    "kind": "withheld_dimension",
                    "error_m": error_m,
                    "uncertainty_m": uncertainty,
                    "passed": passed,
                }
            )
            evaluation_dimensions.append(
                {"id": dimension["id"], "meters": predicted_m, "units": "m"}
            )
        scene_matrix = _homogeneous(scale * rotation_fit, translation_fit)
        scene_to_world.append(
            {
                "shot": shot,
                "source_frame": f"da3-scene:{shot}",
                "target_frame": world_frame["id"],
                "source_to_target": "da3-scene-relative-to-calibration-world",
                "source_units": "relative_depth",
                "target_units": "m",
                "matrix_4x4": scene_matrix,
                "scale": scale,
            }
        )
        # The DA3 OpenCV camera convention differs from NIM's y-up/z-back camera basis
        # by this proper 180-degree rotation. NIM joints are already positioned in meters.
        nim_to_da3_camera = np.diag([1.0, -1.0, -1.0])
        for frame in shot_frames:
            ext = _matrix(frame["extrinsics_w2c"], (3, 4), "DA3 world-to-camera extrinsics")
            rotation_wc = _proper_rotation(ext[:, :3], "DA3 world-to-camera rotation")
            linear = rotation_fit @ rotation_wc.T @ nim_to_da3_camera
            translation = translation_fit - scale * rotation_fit @ rotation_wc.T @ ext[:, 3]
            camera_transforms.append(
                {
                    "frame_id": frame["frame_id"],
                    "shot": shot,
                    "source_frame": f"nim-camera:{frame['frame_id']}",
                    "target_frame": world_frame["id"],
                    "direction": "nim_camera_m_to_metric_world_m",
                    "units": "m",
                    "matrix_4x4": _homogeneous(linear, translation),
                    "root_translation_applied": False,
                    "stage_transform_applied": False,
                }
            )
    missing_dimensions = [
        dimension
        for dimension in calibration.get("measured_dimensions", [])
        if dimension.get("role") == "check" and dimension["id"] not in check_dimension_ids
    ]
    if missing_dimensions:
        detail = "; ".join(
            f"{dimension['id']!r} (frames {dimension['a']['frame_id']}, {dimension['b']['frame_id']})"
            for dimension in missing_dimensions
        )
        raise RegistrationError(f"check dimension {detail}: not processed in a single shot")
    if not all(item["passed"] for item in registration_checks):
        raise RegistrationError("fit or independent check anchors exceed their uncertainty bounds")
    evaluation_frames = [evaluation_frames_by_id[key] for key in sorted(evaluation_frames_by_id)]
    geometry = dict(geometry)
    geometry.update(
        {
            "status": "registered_metric",
            "reason": "independent measured fit anchors and withheld check anchors validated",
            "units": "m",
            "coordinate_frame_id": world_frame["id"],
            "coordinate_convention": "right-handed calibration world; metric meters",
            "scale_provenance": {
                "kind": "independent_fit_anchors",
                "metric": True,
                "fit_anchor_ids": fit_anchor_ids,
                "check_anchor_ids": check_anchor_ids,
                "fit_dimension_ids": [],
                "check_dimension_ids": check_dimension_ids,
            },
            "registration": {
                "schema": "skeleton-maker.geometry-registration/1",
                "source_frame": "da3-relative-scene-frames",
                "target_frame": world_frame["id"],
                "direction": "source_to_target",
                "source_units": "relative_depth",
                "target_units": "m",
                "scene_transforms": scene_to_world,
                "camera_transforms": camera_transforms,
                "fit_anchor_ids": fit_anchor_ids,
                "check_anchor_ids": check_anchor_ids,
                "check_dimension_ids": check_dimension_ids,
                "checks": registration_checks,
            },
            "evaluation": {"frames": evaluation_frames, "dimensions": evaluation_dimensions},
        }
    )
    return geometry
