# SPDX-License-Identifier: MIT
"""Prepare exact-frame registered depth geometry for the character stage."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import envmanifest
from .envcamera import IntrinsicsError, backproject, validate_intrinsics

MAX_POINTS_PER_FRAME = 1200
MAX_POINT_CLOUD_FRAMES = 300


def prepare(manifest_path: str | Path, stage_meta: dict, *, document: dict | None = None) -> dict:
    """Return only independently registered, source-frame-matched point clouds."""
    manifest_path = Path(manifest_path).resolve()
    doc = document if document is not None else envmanifest.load_manifest(manifest_path)
    geometry = doc["geometry"]
    status = geometry["status"]
    reason = geometry["reason"]
    result = {"status": status, "reason": reason, "frames": {}}
    display_by_source = {
        frame["source_frame"]: frame
        for shot in stage_meta.get("shots", [])
        for frame in shot.get("display_frames", [])
    }
    for source_frame in display_by_source:
        result["frames"][str(source_frame)] = {
            "points": [],
            "reason": reason if status != "registered_metric" else "",
        }
    if status != "registered_metric":
        return result

    registration = geometry.get("registration", {})
    cameras = {frame["frame_id"]: frame for frame in registration.get("camera_transforms", [])}
    scenes = {frame["shot"]: frame for frame in registration.get("scene_transforms", [])}
    geometry_frames = {frame["frame_id"]: frame for frame in geometry.get("frames", [])}
    assets_root = envmanifest.assets_dir_for(manifest_path)
    visible_frames = 0

    for source_frame, display in display_by_source.items():
        output = result["frames"][str(source_frame)]
        camera = cameras.get(source_frame)
        geom_frame = geometry_frames.get(source_frame)
        if camera is None:
            output["reason"] = (
                f"no exact registered camera transform for source frame {source_frame}"
            )
            continue
        if geom_frame is None:
            output["reason"] = f"no depth geometry for source frame {source_frame}"
            continue
        if visible_frames >= MAX_POINT_CLOUD_FRAMES:
            output["reason"] = "point cloud display limit reached"
            continue
        scene = scenes.get(geom_frame["shot"])
        if scene is None:
            output["reason"] = f"no scene-to-world transform for source frame {source_frame}"
            continue

        depth_path = envmanifest.resolve_asset(
            assets_root, geom_frame["depth_asset"], f"geometry.frames[{source_frame}].depth_asset"
        )
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2:
            output["reason"] = f"depth geometry for source frame {source_frame} is not a 2D grid"
            continue
        intrinsics = np.asarray(geom_frame["intrinsics"], dtype=np.float64)
        source_to_processed = np.asarray(
            geom_frame["preprocessing"]["undistorted_source_to_processed"], dtype=np.float64
        )
        processed_intrinsics = source_to_processed @ intrinsics
        try:
            # Validate even before checking for depth samples so malformed manifests
            # cannot silently produce an empty cloud.
            validate_intrinsics(processed_intrinsics, "depth intrinsics")
        except IntrinsicsError:
            output["reason"] = f"invalid depth intrinsics for source frame {source_frame}"
            continue
        valid = np.isfinite(depth) & (depth > 0)
        indices = np.flatnonzero(valid)
        if not len(indices):
            output["reason"] = f"no valid depth samples for source frame {source_frame}"
            continue
        stride = max(1, math.ceil(len(indices) / MAX_POINTS_PER_FRAME))
        indices = indices[::stride][:MAX_POINTS_PER_FRAME]
        rows, columns = np.divmod(indices, depth.shape[1])
        z = depth[rows, columns].astype(np.float64)
        try:
            camera_points = backproject(processed_intrinsics, columns, rows, z)
        except IntrinsicsError:
            output["reason"] = f"invalid depth intrinsics for source frame {source_frame}"
            continue
        da3_camera_points = np.concatenate([camera_points, np.ones((len(z), 1))], axis=1)

        extrinsics = np.eye(4, dtype=np.float64)
        extrinsics[:3, :] = np.asarray(geom_frame["extrinsics_w2c"], dtype=np.float64)
        scene_to_world = np.asarray(scene["matrix_4x4"], dtype=np.float64)
        nim_camera_to_world = np.asarray(camera["matrix_4x4"], dtype=np.float64)
        camera_to_stage = np.asarray(display["camera_to_stage"], dtype=np.float64)
        da3_camera_to_stage = (
            camera_to_stage
            @ np.linalg.inv(nim_camera_to_world)
            @ scene_to_world
            @ np.linalg.inv(extrinsics)
        )
        points = (da3_camera_to_stage @ da3_camera_points.T).T[:, :3]
        finite = np.isfinite(points).all(axis=1)
        output["points"] = np.round(points[finite], 5).tolist()
        output["reason"] = (
            "" if output["points"] else f"no finite geometry for source frame {source_frame}"
        )
        visible_frames += bool(output["points"])
    return result
