# SPDX-License-Identifier: MIT
"""Tests for the annotation layer: filtering, writing and parsing."""

import pytest

from skeleton_maker.bbox import filter_frames, read_annotation, write_annotation
from skeleton_maker.constants import MAX_BODIES


def make_frames(n_ids=3, n_frames=5, per_frame=None):
    """Build tracker-style frames: id i appears in every frame."""
    frames = []
    for fid in range(n_frames):
        dets = []
        for tid in range(1, n_ids + 1):
            if per_frame and fid not in per_frame.get(tid, []):
                continue
            dets.append({"tracking_id": tid, "bbox": [10.0 * tid, 20.0, 30.0, 40.0], "score": 0.9})
        frames.append({"frame_id": fid, "detections": dets})
    return frames


def test_filter_keeps_the_most_persistent_ids():
    # id 1 appears in all 5 frames, id 2 in frames 0-1, id 3 in frame 0 only.
    frames = make_frames(3, 5, per_frame={1: range(5), 2: range(2), 3: [0]})
    kept = filter_frames(frames, max_bodies=2)
    ids = {d["tracking_id"] for fr in kept for d in fr["detections"]}
    assert ids == {1, 2}
    assert len(kept) == len(frames), "frame count must not change"


def test_filter_is_a_noop_when_under_the_limit():
    frames = make_frames(2, 3)
    assert filter_frames(frames, max_bodies=50) == frames


def test_write_and_read_round_trip(tmp_path):
    frames = make_frames(3, 4)
    path = tmp_path / "boxes.txt"
    summary = write_annotation(frames, str(path))

    assert summary["bodies"] == 3
    assert summary["rows"] == 12
    assert summary["coverage"] == 100.0
    assert summary["empty_frames"] == []
    assert path.read_text().splitlines()[0] == "3"

    parsed = read_annotation(str(path))
    assert sorted(parsed) == [0, 1, 2, 3]
    assert len(parsed[0]) == 3
    tid, x, y, w, h = parsed[2][0]
    assert (tid, x, y, w, h) == (1, 10.0, 20.0, 30.0, 40.0)


def test_write_rejects_more_ids_than_the_nim_accepts(tmp_path):
    frames = make_frames(MAX_BODIES + 1, 1)
    with pytest.raises(ValueError, match="header limit"):
        write_annotation(frames, str(tmp_path / "boxes.txt"))


def test_write_rejects_a_non_positive_box(tmp_path):
    frames = [
        {
            "frame_id": 0,
            "detections": [{"tracking_id": 1, "bbox": [10.0, 10.0, 0.0, 5.0], "score": 0.9}],
        }
    ]
    with pytest.raises(ValueError, match="non-positive"):
        write_annotation(frames, str(tmp_path / "boxes.txt"))


def test_write_rejects_an_empty_annotation(tmp_path):
    frames = [{"frame_id": 0, "detections": []}]
    with pytest.raises(ValueError, match="no detections"):
        write_annotation(frames, str(tmp_path / "boxes.txt"))


def test_read_rejects_a_bad_header(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0\n")
    with pytest.raises(ValueError, match="outside 1..50"):
        read_annotation(str(path))


def test_read_rejects_a_malformed_row(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1\n0 1 10 20 30\n")
    with pytest.raises(ValueError, match="expected 6"):
        read_annotation(str(path))


def test_read_skips_absent_rows(tmp_path):
    path = tmp_path / "boxes.txt"
    path.write_text("1\n0 1 10 20 30 40\n1 1 -1 -1 -1 -1\n")
    parsed = read_annotation(str(path))
    assert sorted(parsed) == [0]
