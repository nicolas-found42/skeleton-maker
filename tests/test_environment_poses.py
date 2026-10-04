# SPDX-License-Identifier: MIT
"""`environment --poses`: reuse finished pose output, no NIM, honest about identity."""

import json
import shutil
import socket

import pytest

from skeleton_maker import environment, nim, poses, render, stage

from .env_fakes import FakeBackend
from .test_environment import _make_video, _run, _run_failing, _snapshot

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _det(tid, bbox=(5.0, 5.0, 30.0, 40.0)):
    return {
        "tracking_id": tid,
        "bbox": list(bbox),
        "root_pose": {"translation": [0.0, 0.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
    }


def _write_poses(path, frame_ids, people=None, extra=None):
    """people: {frame_id: [tracking ids]}; frames not listed hold nobody."""
    people = people if people is not None else {f: [1] for f in frame_ids}
    with path.open("w") as fh:
        for f in frame_ids:
            rec = {"frame_id": f, "detections": [_det(t) for t in people.get(f, [])]}
            rec.update(extra or {})
            fh.write(json.dumps(rec) + "\n")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_video(path, frames=30)
    return path


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    return backend


def _argv(clip, tmp_path, *extra):
    return [str(clip), "--out", str(tmp_path / "environment.json"), "--backend", "fake", *extra]


def _manifest(tmp_path):
    return json.loads((tmp_path / "environment.json").read_text())


@pytest.fixture
def no_nim(monkeypatch):
    """Any attempt to reach the NIM or the network fails the test."""

    def boom(*a, **k):
        raise AssertionError("NIM or network access attempted")

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(nim, "run", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)


def test_poses_are_reused_without_any_nim_access(clip, fake, tmp_path, no_nim):
    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, range(30))

    assert _run(_argv(clip, tmp_path, "--poses", str(pose_file))) == 0

    block = _manifest(tmp_path)["poses"]
    assert block["path"] == str(pose_file.resolve())
    assert block["association"] == "user-supplied"
    assert block["frame_count"] == 30
    assert block["skeletons"] == [{"id": 1, "first_frame": 0, "last_frame": 29, "frames": 30}]
    assert block["person_free_ranges"] == []


def test_without_poses_the_block_is_null(clip, fake, tmp_path):
    assert _run(_argv(clip, tmp_path)) == 0
    assert _manifest(tmp_path)["poses"] is None


def test_person_free_interval_is_recorded_and_still_scanned(clip, fake, tmp_path):
    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, range(30), people={f: [1, 2] for f in range(15)})

    assert _run(_argv(clip, tmp_path, "--poses", str(pose_file))) == 0

    manifest = _manifest(tmp_path)
    assert manifest["poses"]["person_free_ranges"] == [[15, 29]]
    assert manifest["poses"]["skeletons"][1]["id"] == 2
    free_frames = {o["frame_id"] for o in manifest["observations"] if o["frame_id"] >= 15}
    assert free_frames == {15}, "the surroundings are reported while nobody is present"


def test_matching_source_fingerprint_is_verified(clip, fake, tmp_path):
    from skeleton_maker.artifacts import sha256_file

    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, range(30), extra={"source_sha256": sha256_file(clip)})

    assert _run(_argv(clip, tmp_path, "--poses", str(pose_file))) == 0

    assert _manifest(tmp_path)["poses"]["association"] == "hash-verified"


@pytest.mark.parametrize(
    ("frames", "extra", "message"),
    [
        (range(20), None, "20 pose records but the clip has 30 frames"),
        (range(40), None, "40 pose records but the clip has 30 frames"),
        (range(5, 35), None, "frame ids must be 0..29"),
        ([*range(15), *range(16, 31)], None, "frame ids must be 0..29"),
        ([*range(29), 0], None, "duplicate"),
        (range(30), {"source_sha256": "0" * 64}, "different clip"),
    ],
)
def test_poses_that_do_not_fit_the_clip_are_rejected(
    clip, fake, tmp_path, capsys, frames, extra, message
):
    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, list(frames), people={}, extra=extra)
    before = _snapshot(tmp_path)

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert message in err
    assert _snapshot(tmp_path) == before
    assert fake.requests == []


def test_boxes_far_outside_the_clip_are_rejected(clip, fake, tmp_path, capsys):
    pose_file = tmp_path / "pose.json"
    with pose_file.open("w") as fh:
        for f in range(30):
            fh.write(json.dumps({"frame_id": f, "detections": [_det(1, (400, 300, 900, 700))]}))
            fh.write("\n")

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert "outside the 64x48 clip" in err
    assert "frame 0" in err


@pytest.mark.parametrize("content", ["{not json\n", '{"frame_id": 0}\n', "[1, 2]\n"])
def test_unreadable_pose_files_are_rejected(clip, fake, tmp_path, capsys, content):
    pose_file = tmp_path / "pose.json"
    pose_file.write_text(content)

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert "pose file" in err
    assert not (tmp_path / "environment.json").exists()


def test_detection_without_a_tracking_id_is_a_pointed_error(clip, fake, tmp_path, capsys):
    pose_file = tmp_path / "pose.json"
    with pose_file.open("w") as fh:
        for f in range(30):
            fh.write(json.dumps({"frame_id": f, "detections": [{"bbox": [5, 5, 20, 20]}]}) + "\n")

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert "frame 0: a detection has no integer tracking_id" in err
    assert not (tmp_path / "environment.json").exists()


def test_detection_without_a_root_pose_is_a_pointed_error(clip, fake, tmp_path, capsys):
    pose_file = tmp_path / "pose.json"
    detection = {"tracking_id": 1, "bbox": [5, 5, 20, 20]}
    with pose_file.open("w") as fh:
        for f in range(30):
            fh.write(json.dumps({"frame_id": f, "detections": [detection]}) + "\n")

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert "detection 1 has no root_pose.translation" in err


def test_missing_pose_file_is_rejected(clip, fake, tmp_path, capsys):
    err = _run_failing(_argv(clip, tmp_path, "--poses", str(tmp_path / "gone.json")), capsys)
    assert "no such file" in err


def _with_people(*people):
    def mutate(resp):
        for entity, skeleton in people:
            ent = {
                "id": entity,
                "shot": "shot-0",
                "family": "person",
                "motion": "dynamic",
                "labels": {"native": "person", "normalized": "person"},
            }
            if skeleton is not None:
                ent["skeleton_id"] = skeleton
            resp["entities"].append(ent)

    return mutate


def test_person_entities_link_to_existing_skeleton_ids(clip, monkeypatch, tmp_path):
    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, range(30))
    backend = FakeBackend(mutate=_with_people(("shot-0/person-1", 1)))
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    assert _run(_argv(clip, tmp_path, "--poses", str(pose_file))) == 0

    people = [e for e in _manifest(tmp_path)["entities"] if e["family"] == "person"]
    assert [p["skeleton_id"] for p in people] == [1]


@pytest.mark.parametrize(
    ("people", "message"),
    [
        ([("shot-0/p-9", 9)], "skeleton_id 9"),
        ([("shot-0/p-a", 1), ("shot-0/p-b", 1)], "duplicate person"),
        ([("shot-0/p-x", None)], "must name an existing skeleton"),
    ],
)
def test_person_entities_that_do_not_match_the_poses_fail_the_run(
    clip, monkeypatch, tmp_path, capsys, people, message
):
    pose_file = tmp_path / "pose.json"
    _write_poses(pose_file, range(30))
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(mutate=_with_people(*people)))

    err = _run_failing(_argv(clip, tmp_path, "--poses", str(pose_file)), capsys)

    assert message in err
    assert not (tmp_path / "environment.json").exists()


def test_one_shared_loader_serves_render_stage_and_verify(tmp_path):
    pose_file = tmp_path / "pose.json"
    pose_file.write_text(
        json.dumps({"frame_id": 2, "detections": [_det(1)]})
        + "\n\n"
        + json.dumps({"frame_id": 0, "detections": []})
        + "\n"
    )

    assert render.load_poses is poses.load_poses
    assert poses.load_poses(str(pose_file)) == {2: [_det(1)], 0: []}
    assert stage.load_frames(str(pose_file)) == [(0, []), (2, [_det(1)])]
    assert [r["frame_id"] for r in poses.read_records(str(pose_file))] == [2, 0]
    (tmp_path / "bad.json").write_text("nope\n")
    with pytest.raises(poses.PoseFileError, match="line 1"):
        poses.read_records(str(tmp_path / "bad.json"))
