# SPDX-License-Identifier: MIT
"""Create a synthetic viewer fixture without a NIM call; run from the repository root."""

import argparse
from pathlib import Path

from skeleton_maker.character import build_html, builtin_specs
from skeleton_maker.stage import build_stage
from tests.test_stage import body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--video-name", default="video.mp4")
    args = parser.parse_args()
    if Path(args.video_name).name != args.video_name:
        parser.error("--video-name must be a filename in the fixture directory")
    frames = []
    for frame in [*range(30), *range(90, 120)]:
        detections = []
        for track_id in range(1, 15):
            detection = body(origin=(track_id * 0.3, 0, -5.0))
            detection["tracking_id"] = track_id
            detections.append(detection)
        frames.append((frame, detections))
    page = build_html(
        build_stage(frames),
        builtin_specs(),
        {"video": args.video_name, "character": "robot", "title": "browser regression"},
        "browser regression",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)


if __name__ == "__main__":
    main()
