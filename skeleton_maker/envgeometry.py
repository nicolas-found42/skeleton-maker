# SPDX-License-Identifier: MIT
"""Geometry reference input shared by CLI validation and the isolated DA3 worker."""

import json
import math
from pathlib import Path

from .envcamera import IntrinsicsError, validate_intrinsics

REFERENCE_SCHEMA = "skeleton-maker.geometry-reference/1"


class GeometryReferenceError(ValueError):
    """A camera calibration or measured scene anchor is invalid."""


def _finite_vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise GeometryReferenceError(f"{where}: expected {length} numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in value
    ):
        raise GeometryReferenceError(f"{where}: values must be finite numbers")
    return [float(item) for item in value]


def load_reference(path, *, width: int, height: int, frame_count: int) -> dict | None:
    """Load and validate a versioned camera and optional measured-anchor file."""
    if path is None:
        return None
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda name: (_ for _ in ()).throw(
                GeometryReferenceError(f"non-finite number {name} is not valid")
            ),
        )
    except GeometryReferenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeometryReferenceError(f"cannot read calibration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GeometryReferenceError("calibration must be a JSON object")
    if data.get("schema") != REFERENCE_SCHEMA:
        raise GeometryReferenceError(f"calibration schema must be {REFERENCE_SCHEMA!r}")
    world_frame = data.get("world_frame")
    if world_frame is not None:
        if not isinstance(world_frame, dict):
            raise GeometryReferenceError("world_frame must be an object")
        identifier = world_frame.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise GeometryReferenceError("world_frame.id must be a non-empty string")
        if world_frame.get("units") != "m":
            raise GeometryReferenceError("world_frame.units must be 'm'")
        if world_frame.get("handedness") != "right":
            raise GeometryReferenceError("world_frame.handedness must be 'right'")
    image_size = data.get("image_size")
    if image_size != [width, height]:
        raise GeometryReferenceError(
            f"calibration image_size does not match the source clip ({width}x{height})"
        )

    camera = data.get("camera")
    if camera is not None:
        if not isinstance(camera, dict):
            raise GeometryReferenceError("camera must be an object")
        matrix = camera.get("intrinsics")
        if not isinstance(matrix, list) or len(matrix) != 3:
            raise GeometryReferenceError("camera.intrinsics must be a 3x3 matrix")
        matrix = [_finite_vector(row, 3, f"camera.intrinsics[{i}]") for i, row in enumerate(matrix)]
        try:
            matrix = validate_intrinsics(matrix, "camera.intrinsics").tolist()
        except IntrinsicsError as exc:
            raise GeometryReferenceError(str(exc)) from exc
        distortion = camera.get("distortion", {"model": "none", "coefficients": []})
        if not isinstance(distortion, dict):
            raise GeometryReferenceError("camera.distortion must be an object")
        model = distortion.get("model")
        if model not in ("none", "opencv-brown-conrady"):
            raise GeometryReferenceError(
                "camera.distortion.model must be 'none' or 'opencv-brown-conrady'"
            )
        coefficients = distortion.get("coefficients", [])
        if not isinstance(coefficients, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in coefficients
        ):
            raise GeometryReferenceError("camera.distortion.coefficients must be finite numbers")
        allowed_lengths = (0,) if model == "none" else (4, 5, 8, 12, 14)
        if len(coefficients) not in allowed_lengths:
            raise GeometryReferenceError(
                f"camera.distortion.coefficients length must be one of {allowed_lengths}"
            )
        data["camera"] = {
            "intrinsics": matrix,
            "distortion": {"model": model, "coefficients": [float(v) for v in coefficients]},
        }

    anchors = data.get("anchors", [])
    if not isinstance(anchors, list):
        raise GeometryReferenceError("anchors must be a list")
    parsed_anchors = []
    anchor_ids = set()
    for i, anchor in enumerate(anchors):
        where = f"anchors[{i}]"
        if not isinstance(anchor, dict):
            raise GeometryReferenceError(f"{where} must be an object")
        frame_id = anchor.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise GeometryReferenceError(f"{where}.frame_id must be an integer")
        if not 0 <= frame_id < frame_count:
            raise GeometryReferenceError(f"{where}.frame_id is outside the source frame range")
        pixel = _finite_vector(anchor.get("pixel"), 2, f"{where}.pixel")
        x, y = pixel
        if not 0 <= x < width or not 0 <= y < height:
            raise GeometryReferenceError(
                f"{where}.pixel {pixel} is outside the {width}x{height} source frame"
            )
        position = _finite_vector(anchor.get("position_m"), 3, f"{where}.position_m")
        role = anchor.get("role", "reference")
        if role not in ("fit", "check", "reference"):
            raise GeometryReferenceError(f"{where}.role must be 'fit', 'check', or 'reference'")
        parsed = {"frame_id": frame_id, "pixel": pixel, "position_m": position, "role": role}
        identifier = anchor.get("id")
        if identifier is not None:
            if not isinstance(identifier, str) or not identifier.strip():
                raise GeometryReferenceError(f"{where}.id must be a non-empty string")
            if identifier in anchor_ids:
                raise GeometryReferenceError(f"{where}.id must be unique")
            anchor_ids.add(identifier)
            parsed["id"] = identifier
        uncertainty = anchor.get("uncertainty_m")
        if uncertainty is not None:
            if (
                isinstance(uncertainty, bool)
                or not isinstance(uncertainty, (int, float))
                or not math.isfinite(uncertainty)
                or uncertainty <= 0
            ):
                raise GeometryReferenceError(f"{where}.uncertainty_m must be positive and finite")
            parsed["uncertainty_m"] = float(uncertainty)
        parsed_anchors.append(parsed)
    dimensions = data.get("measured_dimensions", [])
    if not isinstance(dimensions, list):
        raise GeometryReferenceError("measured_dimensions must be a list")
    parsed_dimensions = []
    dimension_ids = set()
    for i, dimension in enumerate(dimensions):
        where = f"measured_dimensions[{i}]"
        if not isinstance(dimension, dict):
            raise GeometryReferenceError(f"{where} must be an object")
        identifier = dimension.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise GeometryReferenceError(f"{where}.id must be a non-empty string")
        if identifier in dimension_ids:
            raise GeometryReferenceError(f"{where}.id must be unique")
        dimension_ids.add(identifier)
        role = dimension.get("role", "reference")
        if role not in ("fit", "check", "reference"):
            raise GeometryReferenceError(f"{where}.role must be 'fit', 'check', or 'reference'")
        endpoints = {}
        for key in ("a", "b"):
            endpoint = dimension.get(key)
            if not isinstance(endpoint, dict):
                raise GeometryReferenceError(f"{where}.{key} must be an object")
            frame_id = endpoint.get("frame_id")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                raise GeometryReferenceError(f"{where}.{key}.frame_id must be an integer")
            if not 0 <= frame_id < frame_count:
                raise GeometryReferenceError(
                    f"{where}.{key}.frame_id is outside the source frame range"
                )
            pixel = _finite_vector(endpoint.get("pixel"), 2, f"{where}.{key}.pixel")
            if not 0 <= pixel[0] < width or not 0 <= pixel[1] < height:
                raise GeometryReferenceError(
                    f"{where}.{key}.pixel {pixel} is outside the {width}x{height} source frame"
                )
            endpoints[key] = {"frame_id": frame_id, "pixel": pixel}
        length = dimension.get("length_m")
        if (
            isinstance(length, bool)
            or not isinstance(length, (int, float))
            or not math.isfinite(length)
            or length <= 0
        ):
            raise GeometryReferenceError(f"{where}.length_m must be a positive finite number")
        parsed = {"id": identifier, **endpoints, "length_m": float(length), "role": role}
        uncertainty = dimension.get("uncertainty_m")
        if uncertainty is not None:
            if (
                isinstance(uncertainty, bool)
                or not isinstance(uncertainty, (int, float))
                or not math.isfinite(uncertainty)
                or uncertainty <= 0
            ):
                raise GeometryReferenceError(f"{where}.uncertainty_m must be positive and finite")
            parsed["uncertainty_m"] = float(uncertainty)
        parsed_dimensions.append(parsed)
    if camera is None and not parsed_anchors and not parsed_dimensions:
        raise GeometryReferenceError(
            "calibration must contain a camera, measured anchor, or measured dimension"
        )
    data["anchors"] = parsed_anchors
    data["measured_dimensions"] = parsed_dimensions
    return data
