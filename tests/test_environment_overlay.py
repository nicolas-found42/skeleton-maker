# SPDX-License-Identifier: MIT
"""Public CLI coverage for synchronized environment and skeleton overlays."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from skeleton_maker import cli, environment, environment_overlay

from .env_fakes import FakeBackend
from .test_environment import _make_video, _snapshot

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _pose_record(frame_id):
    points = [[58.0, 6.0]] * 77
    confidence = [0.0] * 77
    confidence[0] = 1.0
    joints3 = [[0.0, 0.0, 3.0]] * 77
    return {
        "frame_id": frame_id,
        "detections": [
            {
                "tracking_id": 4,
                "bbox": [54.0, 2.0, 8.0, 10.0],
                "keypoints_2d": points,
                "keypoints_confidence": confidence,
                "keypoints_3d": joints3,
                "rest_pose": joints3,
                "joint_rotations": [[0.0, 0.0, 0.0, 1.0]] * 77,
                "root_pose": {
                    "translation": [0.0, 0.0, 3.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }


@pytest.mark.parametrize("visibility", ["visible", "uncertain"])
def test_fractional_bbox_uses_half_open_pixel_edges(visibility):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    observation = {
        "bbox": [10.4, 25.4, 20.4, 35.4],
        "mask": None,
        "visibility": visibility,
    }
    entity = {"family": "object", "labels": {"normalized": "chair"}}

    environment_overlay._draw_observation(frame, observation, entity, Path("."), {})

    assert np.any(frame[30, 11])
    assert np.array_equal(frame[30, 10], np.zeros(3, dtype=np.uint8))


def _write_poses(path, frame_count=30):
    path.write_text("".join(json.dumps(_pose_record(frame)) + "\n" for frame in range(frame_count)))


def _read_frame(video_path, frame_id):
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        assert ok, f"could not read overlay frame {frame_id}"
        return frame
    finally:
        capture.release()


def test_environment_overlay_draws_only_observed_frames_and_skeleton_full_cadence(
    monkeypatch, tmp_path
):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    poses = tmp_path / "pose.json"
    _write_poses(poses)
    manifest = tmp_path / "environment.json"
    overlay = tmp_path / "environment-overlay.mp4"
    viewer = tmp_path / "environment.html"
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    assert (
        cli.main(
            [
                "environment",
                str(video),
                "--poses",
                str(poses),
                "--out",
                str(manifest),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--sample-fps",
                "2",
                "--no-cache",
                "--overlay",
                str(overlay),
                "--viewer",
                str(viewer),
            ]
        )
        == 0
    )

    assert manifest.is_file()
    assert (tmp_path / "environment.assets" / "masks" / "floor-0.png").is_file()
    assert overlay.is_file()
    assert viewer.is_file()
    assert (tmp_path / "environment.viewer.assets" / "viewer.js").is_file()
    assert len(backend.requests) == 1
    assert backend.requests[0]["poses"]["association"] == "user-supplied"

    observed_frame = _read_frame(overlay, 0)
    between_samples = _read_frame(overlay, 1)
    later_pose_frame = _read_frame(overlay, 14)
    later_observation = _read_frame(overlay, 15)
    source_frame = _read_frame(video, 0)
    source_between = _read_frame(video, 1)
    source_pose_frame = _read_frame(video, 14)
    source_later_observation = _read_frame(video, 15)
    assert np.abs(observed_frame[40, 30].astype(int) - source_frame[40, 30]).max() > 25, (
        "frame 0 contains the floor mask tint"
    )
    assert np.abs(between_samples[40, 30].astype(int) - source_between[40, 30]).max() < 25, (
        "frame 1 has no held or interpolated environment mask"
    )
    assert np.abs(later_pose_frame[6, 58].astype(int) - source_pose_frame[6, 58]).max() > 80, (
        "skeletons render at unsampled source frame 14"
    )
    assert (
        np.abs(later_observation[20, 10].astype(int) - source_later_observation[20, 10]).max() > 50
    ), "the frame 15 object observation is rendered at its source frame"


@pytest.mark.parametrize("collision", ["video", "poses", "manifest", "viewer"])
def test_environment_overlay_collision_is_rejected_before_backend(
    monkeypatch, tmp_path, capsys, collision
):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    poses = tmp_path / "pose.json"
    _write_poses(poses)
    manifest = tmp_path / "environment.json"
    viewer = tmp_path / "environment.html"
    overlay = tmp_path / "environment-overlay.mp4"
    if collision == "video":
        overlay = video
    elif collision == "poses":
        overlay = poses
    elif collision == "manifest":
        overlay = manifest
    elif collision == "viewer":
        overlay = viewer
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    before = _snapshot(tmp_path)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment",
                str(video),
                "--backend",
                "fake",
                "--poses",
                str(poses),
                "--out",
                str(manifest),
                "--overlay",
                str(overlay),
                "--viewer",
                str(viewer),
            ]
        )

    assert exc.value.code == 2
    assert "overlay" in capsys.readouterr().err.lower()
    assert not backend.requests
    assert video.read_bytes() == next(iter(before.values()))
