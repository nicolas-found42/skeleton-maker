# SPDX-License-Identifier: MIT
"""Render source-frame environment observations together with full-cadence poses."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import artifacts, envmanifest, poses, render
from .envmanifest import ManifestError
from .utils import run_ffmpeg

FAMILY_COLORS = {
    "surface": (50, 180, 130),
    "object": (40, 150, 245),
    "vehicle": (220, 100, 45),
}
MASK_ALPHA = 0.38


class OverlayError(ValueError):
    """The overlay could not be rendered from the supplied validated artifacts."""


def _dashed_rectangle(frame, box, color):
    x0, y0, x1, y1 = box
    dash = 6
    for x in range(x0, x1, dash * 2):
        cv2.line(frame, (x, y0), (min(x + dash, x1), y0), color, 1)
        cv2.line(frame, (x, y1), (min(x + dash, x1), y1), color, 1)
    for y in range(y0, y1, dash * 2):
        cv2.line(frame, (x0, y), (x0, min(y + dash, y1)), color, 1)
        cv2.line(frame, (x1, y), (x1, min(y + dash, y1)), color, 1)


def _draw_observation(frame, observation, entity, assets_root, mask_cache):
    family = entity["family"]
    color = FAMILY_COLORS.get(family, (230, 230, 230))
    mask_info = observation.get("mask")
    if mask_info is not None:
        rel = mask_info["asset"]
        if rel not in mask_cache:
            target = envmanifest.resolve_asset(assets_root, rel, "overlay mask")
            mask = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2 or mask.shape != frame.shape[:2]:
                raise OverlayError(f"overlay mask {rel!r} is missing, unreadable, or wrong-sized")
            mask_cache[rel] = mask > 0
        selected = mask_cache[rel]
        if np.any(selected):
            pixels = frame[selected].astype(np.float32)
            tint = np.asarray(color, dtype=np.float32)
            frame[selected] = (pixels * (1.0 - MASK_ALPHA) + tint * MASK_ALPHA).astype(np.uint8)

    x0 = max(0, math.ceil(observation["bbox"][0]))
    y0 = max(0, math.ceil(observation["bbox"][1]))
    x1 = min(frame.shape[1] - 1, math.ceil(observation["bbox"][2]) - 1)
    y1 = min(frame.shape[0] - 1, math.ceil(observation["bbox"][3]) - 1)
    if x1 < x0 or y1 < y0:
        return
    if observation["visibility"] == "uncertain":
        _dashed_rectangle(frame, (x0, y0, x1, y1), color)
    else:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    label = entity["labels"]["normalized"]
    if observation["visibility"] != "visible":
        label = f"{label} ({observation['visibility']})"
    cv2.putText(
        frame,
        label,
        (x0, max(12, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def render_overlay(
    video_path: str | os.PathLike,
    manifest_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    pose_path: str | os.PathLike | None = None,
    quality: int = 19,
) -> int:
    """Write an MP4 and return its frame count; observations are drawn at exact frame ids."""
    video, manifest_file, output = (
        Path(video_path).resolve(),
        Path(manifest_path).resolve(),
        Path(output_path),
    )
    manifest = envmanifest.load_manifest(manifest_file)
    if artifacts.sha256_file(video) != manifest["source"]["sha256"]:
        raise ManifestError("overlay source video does not match the environment manifest")
    source = manifest["source"]
    observations = defaultdict(list)
    entities = {entity["id"]: entity for entity in manifest["entities"]}
    for observation in manifest["observations"]:
        observations[observation["frame_id"]].append(observation)
    pose_frames = poses.load_poses(str(pose_path)) if pose_path is not None else {}
    assets_root = envmanifest.assets_dir_for(manifest_file)
    mask_cache = {}

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OverlayError(f"cannot open overlay source video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_num, fps_den = source["frame_rate"]
    fps = fps_num / fps_den
    raw = output.with_name(f".{output.name}.raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OverlayError(f"cannot write overlay staging video: {raw}")

    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for observation in observations.get(frame_id, []):
                _draw_observation(
                    frame, observation, entities[observation["entity"]], assets_root, mask_cache
                )
            for detection in pose_frames.get(frame_id, []):
                drawn = dict(detection)
                drawn["_draw_2d"] = True
                drawn["_draw_3d"] = False
                drawn["_focal"] = 0.0
                color = render.TRACK_COLORS[drawn["tracking_id"] % len(render.TRACK_COLORS)]
                render.draw_detection(frame, drawn, color)
            writer.write(frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()

    if frame_id != source["frame_count"]:
        raw.unlink(missing_ok=True)
        raise OverlayError(
            f"overlay decoded {frame_id} frames but manifest describes {source['frame_count']}"
        )
    try:
        run_ffmpeg(
            [
                "-i",
                str(raw),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(quality),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output),
                "-y",
            ],
            f"re-encoding {output}",
        )
    finally:
        raw.unlink(missing_ok=True)
    return frame_id
