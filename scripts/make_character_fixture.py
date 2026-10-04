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
    args = parser.parse_args()
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
        {"video": "video.mp4", "character": "robot", "title": "browser regression"},
        "browser regression",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)


if __name__ == "__main__":
    main()
