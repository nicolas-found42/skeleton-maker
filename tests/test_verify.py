# SPDX-License-Identifier: MIT
"""End-to-end check of `verify` on synthetic artifacts: no NIM call needed.

These exercise the exact failure the NIM's frame-indexed protocol makes possible
-- poses matched to the wrong frames -- without spending an API call.
"""

import json
import shutil
import subprocess
import sys

import pytest

from skeleton_maker.constants import NUM_JOINTS

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_video(path, frames=30, size="64x48", crf="18"):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={frames / 30:.3f}",
         "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        check=True,
    )


def _pose_record(frame_id, tid=1):
    joints2 = [[float(i), float(i)] for i in range(NUM_JOINTS)]
    joints3 = [[float(i), float(i), 1.0] for i in range(NUM_JOINTS)]
    return {
        "frame_id": frame_id,
        "detections": [{
            "tracking_id": tid,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "keypoints_2d": joints2,
            "keypoints_confidence": [0.5] * NUM_JOINTS,
            "keypoints_3d": joints3,
            "rest_pose": joints3,
            "joint_rotations": [[0.0, 0.0, 0.0, 1.0]] * NUM_JOINTS,
            "root_pose": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
        }],
    }


@pytest.fixture
def run_dir(tmp_path):
    """A clip, a matching annotation, and poses for 10 single-person frames."""
    clip = tmp_path / "clip.mp4"
    overlay = tmp_path / "overlay.mp4"
    _make_video(clip, frames=10)
    _make_video(overlay, frames=10, crf="12")  # differs from the clip

    boxes = tmp_path / "boxes.txt"
    boxes.write_text("1\n" + "".join(f"{f} 1 1.0 2.0 3.0 4.0\n" for f in range(10)))

    poses = tmp_path / "pose.json"
    with poses.open("w") as fh:
        for f in range(10):
            fh.write(json.dumps(_pose_record(f)) + "\n")

    return {"clip": clip, "overlay": overlay, "boxes": boxes, "poses": poses, "tmp": tmp_path}


def test_verify_passes_on_consistent_artifacts(run_dir):
    from skeleton_maker.verify import run

    assert run(str(run_dir["clip"]), str(run_dir["boxes"]), str(run_dir["poses"]),
               str(run_dir["overlay"]), report=False) is True


def test_verify_fails_when_pose_frame_ids_are_shifted(run_dir):
    """The failure mode that matters: an off-by-one frame mapping."""
    from skeleton_maker.verify import run

    with run_dir["poses"].open("w") as fh:
        for f in range(10):
            fh.write(json.dumps(_pose_record(f + 1)) + "\n")  # shifted by one
    assert run(str(run_dir["clip"]), str(run_dir["boxes"]), str(run_dir["poses"]),
               None, report=False) is False


def test_verify_fails_when_a_detection_is_missing(run_dir):
    from skeleton_maker.verify import run

    with run_dir["poses"].open("w") as fh:
        for f in range(10):
            record = _pose_record(f)
            if f == 3:
                record["detections"] = []
            fh.write(json.dumps(record) + "\n")
    assert run(str(run_dir["clip"]), str(run_dir["boxes"]), str(run_dir["poses"]),
               None, report=False) is False


def test_verify_fails_on_a_truncated_joint_array(run_dir):
    from skeleton_maker.verify import run

    with run_dir["poses"].open("w") as fh:
        for f in range(10):
            record = _pose_record(f)
            if f == 5:
                record["detections"][0]["keypoints_2d"] = record["detections"][0]["keypoints_2d"][:10]
            fh.write(json.dumps(record) + "\n")
    assert run(str(run_dir["clip"]), str(run_dir["boxes"]), str(run_dir["poses"]),
               None, report=False) is False


def test_verify_accepts_a_frame_with_no_bodies(run_dir):
    """A frame with no boxes is legal: the file keeps a record with no detections."""
    from skeleton_maker.verify import run

    with run_dir["boxes"].open("w") as fh:
        fh.write("1\n" + "".join(f"{f} 1 1.0 2.0 3.0 4.0\n" for f in range(10) if f != 7))
    with run_dir["poses"].open("w") as fh:
        for f in range(10):
            record = _pose_record(f)
            if f == 7:
                record["detections"] = []
            fh.write(json.dumps(record) + "\n")
    assert run(str(run_dir["clip"]), str(run_dir["boxes"]), str(run_dir["poses"]),
               None, report=False) is True
