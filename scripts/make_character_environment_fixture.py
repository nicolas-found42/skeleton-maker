# SPDX-License-Identifier: MIT
"""Create a deterministic two-shot registered-geometry viewer fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skeleton_maker.character import make_stage_html
from skeleton_maker.stage import Options, build_stage
from tests.test_character_environment import _registered_manifest
from tests.test_stage import body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    pose_path = args.directory / "pose.jsonl"
    frames = []
    for source_frame in [*range(20), *range(90, 110)]:
        detection = body(origin=(0.0, 0.0, -5.0), tilt_deg=12.0 if source_frame >= 90 else 0.0)
        detection["tracking_id"] = 1
        frames.append((source_frame, [detection]))
    pose_path.write_text(
        "\n".join(json.dumps({"frame_id": frame, "detections": items}) for frame, items in frames)
    )

    manifest = _registered_manifest(args.directory, frame_ids=(0, 90), source_frame_count=110)
    manifest_doc = json.loads(manifest.read_text())
    geometry = manifest_doc["geometry"]
    geometry["frames"][0]["shot"] = "source-shot-0"
    geometry["frames"][1]["shot"] = "source-shot-1"
    scene0 = geometry["registration"]["scene_transforms"][0]
    scene0.update(shot="source-shot-0", source_frame="da3-scene:source-shot-0")
    scene0["matrix_4x4"] = [
        [1.0, 0.0, 0.0, -0.1],
        [0.0, 1.0, 0.0, 0.47],
        [0.0, 0.0, 1.0, 2.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
    scene1 = json.loads(json.dumps(scene0))
    scene1.update(shot="source-shot-1", source_frame="da3-scene:source-shot-1")
    scene1["matrix_4x4"] = [
        [1.0, 0.0, 0.0, 0.4],
        [0.0, 1.0, 0.0, 0.27],
        [0.0, 0.0, 1.0, 2.9],
        [0.0, 0.0, 0.0, 1.0],
    ]
    geometry["registration"]["scene_transforms"] = [scene0, scene1]
    for transform in manifest_doc["geometry"]["registration"]["camera_transforms"]:
        transform["matrix_4x4"][0][3] = 0.1
        if transform["frame_id"] == 90:
            transform["shot"] = "source-shot-1"
    manifest_doc["shots"] = [
        {
            "id": "source-shot-0",
            "first_frame": 0,
            "last_frame": 59,
            "first_time": [0, 1],
            "last_time": [59, 30],
        },
        {
            "id": "source-shot-1",
            "first_frame": 60,
            "last_frame": 109,
            "first_time": [2, 1],
            "last_time": [109, 30],
        },
    ]
    manifest.write_text(json.dumps(manifest_doc))
    output = args.directory / "index.html"
    make_stage_html(str(pose_path), str(output), environment_manifest=str(manifest))

    stage = build_stage(frames, Options(), capture_display_transforms=True)
    camera_to_stage = {
        item["source_frame"]: np.asarray(item["camera_to_stage"], dtype=np.float64)
        for shot in stage.meta["shots"]
        for item in shot["display_frames"]
    }
    da3_to_world = np.array([0.2, 0.4, 2.0, 1.0])
    scene_by_frame = {0: scene0["matrix_4x4"], 90: scene1["matrix_4x4"]}
    expected = {}
    for source_frame in (0, 90):
        nim_camera_to_world = np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 1.8],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        point = (
            camera_to_stage[source_frame]
            @ np.linalg.inv(nim_camera_to_world)
            @ np.asarray(scene_by_frame[source_frame], dtype=np.float64)
            @ da3_to_world
        )
        expected[str(source_frame)] = np.round(point[:3], 5).tolist()
    (args.directory / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    main()
