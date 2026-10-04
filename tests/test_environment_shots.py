# SPDX-License-Identifier: MIT
"""Visual shot boundaries through the public `environment` command, on synthetic clips."""

import json
import shutil
from fractions import Fraction

import cv2
import numpy as np
import pytest

from skeleton_maker import cli, environment, envmanifest

from .env_fakes import FakeBackend

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)

W, H, FPS = 64, 48, 30


def _scene(kind):
    """A textured still in one of two very different palettes."""
    rng = np.random.default_rng({"a": 1, "b": 2, "c": 3}[kind])
    noise = rng.integers(0, 40, (H, W, 3))
    base = {
        "a": np.array([180, 90, 40]),  # blue-ish (BGR)
        "b": np.array([40, 170, 210]),  # amber
        "c": np.array([60, 200, 60]),  # green
    }[kind]
    gradient = np.linspace(-30, 30, W)[None, :, None]
    return np.clip(base + noise + gradient, 0, 255).astype(np.uint8)


def _write(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _fade(a, b, steps):
    out = [(a * (1 - i / (steps - 1))).astype(np.uint8) for i in range(steps)]
    return out + [(b * (i / (steps - 1))).astype(np.uint8) for i in range(steps)]


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    return backend


def _scan(tmp_path, frames, *extra):
    clip = tmp_path / "clip.mp4"
    _write(clip, frames)
    out = tmp_path / "environment.json"
    rc = cli.main(["environment", str(clip), "--out", str(out), "--backend", "fake", *extra])
    assert rc == 0
    return json.loads(out.read_text())


def _ranges(manifest):
    return [[s["first_frame"], s["last_frame"]] for s in manifest["shots"]]


def test_a_cut_splits_the_clip_at_the_first_frame_of_the_new_scene(fake, tmp_path):
    frames = [_scene("a")] * 20 + [_scene("b")] * 20

    manifest = _scan(tmp_path, frames)

    assert _ranges(manifest) == [[0, 19], [20, 39]]
    boundary = manifest["shot_detection"]["boundaries"][0]
    assert boundary["frame_id"] == 20
    assert boundary["distance"] > 0.9


def test_shots_carry_times_that_match_the_frame_rate(fake, tmp_path):
    manifest = _scan(tmp_path, [_scene("a")] * 20 + [_scene("b")] * 20)

    second = manifest["shots"][1]
    assert second["first_time"] == [2, 3]  # frame 20 at 30 fps is 2/3 s
    assert second["last_time"] == [13, 10]  # frame 39 is 39/30 = 13/10 s
    assert Fraction(*second["first_time"]) == Fraction(20, FPS)


def test_several_cuts_and_the_backend_sees_the_shots(fake, tmp_path):
    frames = [_scene("a")] * 15 + [_scene("b")] * 15 + [_scene("c")] * 15

    manifest = _scan(tmp_path, frames)

    assert _ranges(manifest) == [[0, 14], [15, 29], [30, 44]]
    assert [s["id"] for s in fake.requests[0]["shots"]] == ["shot-0", "shot-1", "shot-2"]
    assert {e["shot"] for e in manifest["entities"]} <= {"shot-0", "shot-1", "shot-2"}


def test_a_still_clip_is_one_shot(fake, tmp_path):
    manifest = _scan(tmp_path, [_scene("a")] * 30)

    assert _ranges(manifest) == [[0, 29]]
    assert manifest["shot_detection"]["boundaries"] == []


def test_a_fade_through_black_is_not_a_cut(fake, tmp_path):
    frames = [_scene("a")] * 8 + _fade(_scene("a"), _scene("b"), 12) + [_scene("b")] * 8

    manifest = _scan(tmp_path, frames)

    assert _ranges(manifest) == [[0, 39]]


def test_a_gradual_lighting_change_is_not_a_cut(fake, tmp_path):
    base = _scene("a").astype(np.int16)
    frames = [np.clip(base + i * 2, 0, 255).astype(np.uint8) for i in range(45)]

    manifest = _scan(tmp_path, frames)

    assert _ranges(manifest) == [[0, 44]]


def test_a_whip_pan_is_not_a_cut(fake, tmp_path):
    rng = np.random.default_rng(7)
    wide = rng.integers(0, 255, (H, W * 4, 3)).astype(np.uint8)
    frames = [np.roll(wide, -i * 20, axis=1)[:, :W] for i in range(40)]

    manifest = _scan(tmp_path, frames)

    assert _ranges(manifest) == [[0, 39]]


def test_a_flash_right_after_a_cut_does_not_make_a_one_frame_shot(fake, tmp_path):
    frames = [_scene("a")] * 10 + [_scene("b")] + [_scene("c")] * 19

    manifest = _scan(tmp_path, frames)

    assert all(s["last_frame"] - s["first_frame"] + 1 >= 3 for s in manifest["shots"])


def _poses(path, count, people):
    """people: {frame_id: tracking_id}; root positions are fixed so ids alone decide shots."""
    with path.open("w") as fh:
        for f in range(count):
            dets = []
            if f in people:
                dets = [
                    {
                        "tracking_id": people[f],
                        "bbox": [5, 5, 20, 30],
                        "root_pose": {"translation": [0.0, 0.0, 3.0], "rotation": [0, 0, 0, 1]},
                    }
                ]
            fh.write(json.dumps({"frame_id": f, "detections": dets}) + "\n")


def _scan_with_poses(tmp_path, frames, people):
    clip = tmp_path / "clip.mp4"
    _write(clip, frames)
    pose_file = tmp_path / "pose.json"
    _poses(pose_file, len(frames), people)
    out = tmp_path / "environment.json"
    cli.main(
        [
            "environment",
            str(clip),
            "--out",
            str(out),
            "--backend",
            "fake",
            "--poses",
            str(pose_file),
        ]
    )
    return json.loads(out.read_text())


def test_a_cut_inside_a_person_free_interval_is_found_and_agrees_with_the_poses(fake, tmp_path):
    frames = [_scene("a")] * 20 + [_scene("b")] * 20
    people = {**dict.fromkeys(range(0, 10), 1), **dict.fromkeys(range(30, 40), 2)}

    manifest = _scan_with_poses(tmp_path, frames, people)

    assert _ranges(manifest) == [[0, 19], [20, 39]]
    detection = manifest["shot_detection"]
    assert detection["pose_stage_shots"] == [[0, 9], [30, 39]]
    assert detection["boundaries"][0]["agreement"] == "agrees"
    assert detection["pose_only"] == []


def test_a_visual_cut_the_tracker_missed_is_reported_as_visual_only(fake, tmp_path):
    frames = [_scene("a")] * 20 + [_scene("b")] * 20

    manifest = _scan_with_poses(tmp_path, frames, dict.fromkeys(range(40), 1))

    detection = manifest["shot_detection"]
    assert detection["pose_stage_shots"] == [[0, 39]]
    assert detection["boundaries"][0]["agreement"] == "visual_only"


def test_a_tracker_restart_without_a_visual_cut_is_reported_as_pose_only(fake, tmp_path):
    people = {**dict.fromkeys(range(0, 20), 1), **dict.fromkeys(range(20, 40), 2)}

    manifest = _scan_with_poses(tmp_path, [_scene("a")] * 40, people)

    detection = manifest["shot_detection"]
    assert _ranges(manifest) == [[0, 39]], "poses alone never split the visual shots"
    assert detection["boundaries"] == []
    assert detection["pose_only"] == [{"after_frame": 19, "before_frame": 20}]


def test_without_poses_nothing_is_reconciled(fake, tmp_path):
    manifest = _scan(tmp_path, [_scene("a")] * 20 + [_scene("b")] * 20)

    detection = manifest["shot_detection"]
    assert detection["pose_stage_shots"] is None
    assert detection["boundaries"][0]["agreement"] == "no_poses"
    assert detection["pose_only"] == []


def test_entities_and_observations_stay_inside_their_shot(fake, tmp_path):
    manifest = _scan(tmp_path, [_scene("a")] * 20 + [_scene("b")] * 20)
    shots = {s["id"]: (s["first_frame"], s["last_frame"]) for s in manifest["shots"]}
    entities = {e["id"]: e for e in manifest["entities"]}

    for obs in manifest["observations"]:
        first, last = shots[entities[obs["entity"]]["shot"]]
        assert first <= obs["frame_id"] <= last


def test_the_manifest_loader_rejects_inconsistent_shots(fake, tmp_path):
    _scan(tmp_path, [_scene("a")] * 20 + [_scene("b")] * 20)
    path = tmp_path / "environment.json"

    def edit(fn):
        doc = json.loads(path.read_text())
        fn(doc)
        path.write_text(json.dumps(doc))
        with pytest.raises(envmanifest.ManifestError) as exc:
            envmanifest.load_manifest(path)
        return str(exc.value)

    original = path.read_text()
    assert "does not follow the frame rate" in edit(
        lambda d: d["shots"][1].update(first_time=[1, 3])
    )
    path.write_text(original)
    assert "must start at frame 20" in edit(lambda d: d["shots"][1].update(first_frame=21))
    path.write_text(original)
    assert "cover the clip" in edit(lambda d: d["shots"][1].update(last_frame=35))
    path.write_text(original)
    assert "outside its shot" in edit(lambda d: d["observations"][0].update(frame_id=30))
