# SPDX-License-Identifier: MIT
"""Build a tiny offline viewer fixture with known mask and skeleton pixels."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

from skeleton_maker import artifacts, envmanifest
from skeleton_maker.environment_viewer import export_viewer


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _mask_png(width: int, height: int, bounds=(20, 30, 100, 90)) -> bytes:
    x0, y0, x1, y1 = bounds
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.append(255 if x0 <= x < x1 and y0 <= y < y1 else 0)
        rows.append(row)
    header = struct.pack(">2I5B", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _chunk(b"IEND", b"")
    )


def make(video: Path, output: Path) -> None:
    width, height, frame_count, rate = 640, 360, 150, 30
    poses = output.with_suffix(".poses.jsonl")
    with poses.open("w", encoding="utf-8") as stream:
        for frame_id in range(frame_count):
            points = [[280 + (index % 8) * 10, 90 + (index // 8) * 10] for index in range(77)]
            record = {
                "frame_id": frame_id,
                "detections": [
                    {
                        "tracking_id": 7,
                        "bbox": [270, 80, 90, 210],
                        "keypoints_2d": points,
                        "keypoints_confidence": [1.0] * 77,
                    }
                ],
            }
            stream.write(json.dumps(record) + "\n")
    masks = {
        "floor.png": _mask_png(width, height),
        "chair.png": _mask_png(width, height, (140, 120, 210, 240)),
        "van.png": _mask_png(width, height, (400, 120, 520, 240)),
    }
    env_bundle = envmanifest.assets_dir_for(output)
    (env_bundle / "masks").mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        (env_bundle / "masks" / name).write_bytes(mask)
    frame_ids = list(range(0, frame_count, 15))
    shots = [
        {
            "id": "shot-0",
            "first_frame": 0,
            "last_frame": 74,
            "first_time": [0, 1],
            "last_time": [37, 15],
        },
        {
            "id": "shot-1",
            "first_frame": 75,
            "last_frame": 149,
            "first_time": [5, 2],
            "last_time": [149, 30],
        },
    ]
    entities = [
        {
            "id": "shot-0/floor",
            "shot": "shot-0",
            "family": "surface",
            "labels": {
                "native": "floor",
                "requested": "floor",
                "normalized": "floor",
                "status": "matched",
                "candidates": ["floor"],
            },
            "motion": "static",
        },
        {
            "id": "shot-1/floor",
            "shot": "shot-1",
            "family": "surface",
            "labels": {
                "native": "floor",
                "requested": "floor",
                "normalized": "floor",
                "status": "matched",
                "candidates": ["floor"],
            },
            "motion": "static",
        },
        {
            "id": "shot-0/chair",
            "shot": "shot-0",
            "family": "object",
            "labels": {
                "native": "chair",
                "requested": "chair",
                "normalized": "chair",
                "status": "matched",
                "candidates": ["chair"],
            },
            "motion": "unknown",
        },
        {
            "id": "shot-1/chair",
            "shot": "shot-1",
            "family": "object",
            "labels": {
                "native": "<chair>",
                "requested": "chair",
                "normalized": "chair",
                "status": "matched",
                "candidates": ["chair"],
            },
            "motion": "unknown",
        },
        {
            "id": "shot-0/van",
            "shot": "shot-0",
            "family": "vehicle",
            "labels": {
                "native": "van",
                "requested": "van",
                "normalized": "van",
                "status": "matched",
                "candidates": ["van"],
            },
            "motion": "static",
        },
    ]
    observations = [
        {
            "id": "floor-0",
            "entity": "shot-0/floor",
            "frame_id": 0,
            "bbox": [20, 30, 100, 90],
            "mask": {"asset": "masks/floor.png"},
            "score": 0.95,
            "score_meaning": "fixture confidence",
            "visibility": "visible",
        },
        {
            "id": "floor-1",
            "entity": "shot-1/floor",
            "frame_id": 75,
            "bbox": [20, 30, 100, 90],
            "mask": {"asset": "masks/floor.png"},
            "score": 0.95,
            "score_meaning": "fixture confidence",
            "visibility": "visible",
        },
        {
            "id": "chair-0",
            "entity": "shot-0/chair",
            "frame_id": 0,
            "bbox": [140, 120, 210, 240],
            "mask": {"asset": "masks/chair.png"},
            "score": 0.92,
            "score_meaning": "fixture confidence",
            "visibility": "visible",
        },
        {
            "id": "chair-absent",
            "entity": "shot-0/chair",
            "frame_id": 15,
            "bbox": [140, 120, 210, 240],
            "mask": None,
            "score": 0.0,
            "score_meaning": "fixture confidence",
            "visibility": "absent",
        },
        {
            "id": "chair-low",
            "entity": "shot-1/chair",
            "frame_id": 75,
            "bbox": [140, 120, 210, 240],
            "mask": None,
            "score": 0.2,
            "score_meaning": "fixture confidence",
            "visibility": "uncertain",
        },
        {
            "id": "van-0",
            "entity": "shot-0/van",
            "frame_id": 0,
            "bbox": [400, 120, 520, 240],
            "mask": {"asset": "masks/van.png"},
            "score": 0.88,
            "score_meaning": "fixture confidence",
            "visibility": "visible",
        },
    ]
    assets = [
        {
            "path": f"masks/{name}",
            "sha256": artifacts.sha256_file(env_bundle / "masks" / name),
            "bytes": len(mask),
        }
        for name, mask in masks.items()
    ]
    manifest = {
        "schema": envmanifest.SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "created_at": "fixture",
            "tool": {"name": "skeleton-maker", "version": "0.1.0"},
        },
        "source": {
            "path": str(video.resolve()),
            "sha256": artifacts.sha256_file(video),
            "width": width,
            "height": height,
            "frame_rate": [rate, 1],
            "frame_count": frame_count,
            "frame_count_source": "fixture",
            "duration_s": 5.0,
        },
        "processed_frames": [
            {"frame_id": frame, "time": [frame, rate], "time_s": frame / rate}
            for frame in frame_ids
        ],
        "poses": {
            "path": str(poses.resolve()),
            "association": "user-supplied",
            "frame_count": frame_count,
            "skeletons": [
                {"id": 7, "first_frame": 0, "last_frame": frame_count - 1, "frames": frame_count}
            ],
            "person_free_ranges": [],
        },
        "frame_range": [0, frame_ids[-1]],
        "shots": shots,
        "shot_detection": None,
        "entities": entities,
        "observations": observations,
        "relations": [],
        "geometry": {
            "status": "unavailable",
            "mode": "off",
            "reason": "fixture has no registration",
        },
        "backend": {"name": "fixture", "version": "1", "device": "cpu"},
        "config": {
            "requested_labels": [],
            "label_vocabulary": [
                {"label": "floor", "family": "surface", "source": "preset"},
                {"label": "chair", "family": "object", "source": "preset"},
                {"label": "van", "family": "vehicle", "source": "preset"},
            ],
            "geometry_mode": "off",
            "sample_fps": 2.0,
            "device": "cpu",
        },
        "assets": assets,
    }
    envmanifest.validate_manifest(manifest)
    output.write_text(envmanifest.dumps(manifest), encoding="utf-8")
    export_viewer(output, str(video), str(output.with_name("environment.html")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    make(args.video, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
