# SPDX-License-Identifier: MIT
"""Opt-in character-stage rendering of registered environment geometry."""

import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest

from skeleton_maker import cli, stage_environment
from skeleton_maker.nova77 import CANON_INDEX
from tests.env_corpus import Corpus
from tests.test_stage import body, clip


def test_character_cli_reports_unavailable_environment_without_3d_geometry(tmp_path):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    (tmp_path / "corpus").mkdir()
    corpus = Corpus(tmp_path / "corpus")
    environment = corpus.clip("stage", frames=[0], shots=[(0, 0)])
    environment.pred_geometry = {"status": "unavailable", "reason": "no valid shared registration"}
    environment.scanned = list(range(20))
    corpus.write()

    output = tmp_path / "stage.html"
    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(output),
                "--environment",
                str(corpus.predictions / "stage.json"),
            ]
        )
        == 0
    )

    page = output.read_text()
    options = page.split('id="options">')[1].split("</script>")[0]
    assert json.loads(options)["environment"]["status"] == "unavailable"
    assert "no valid shared registration" in page
    assert 'id="environment-status"' in page


def _registered_manifest(
    tmp_path,
    frame_ids=(0,),
    *,
    status="registered_metric",
    source_frame_count=None,
    frame_rate=(30, 1),
):
    root = tmp_path / "corpus"
    root.mkdir()
    corpus = Corpus(root)
    sample = corpus.clip("stage", frames=frame_ids, shots=[(min(frame_ids), max(frame_ids))])
    sample.scanned = list(frame_ids)
    sample.pred_registration(status)
    corpus.write()
    path = corpus.predictions / "stage.json"
    doc = json.loads(path.read_text())
    doc["source"]["frame_count"] = source_frame_count or max(frame_ids) + 1
    doc["source"]["frame_rate"] = list(frame_rate)
    doc["source"]["duration_s"] = (doc["source"]["frame_count"] - 1) * frame_rate[1] / frame_rate[0]
    for frame in doc["processed_frames"]:
        time = Fraction(frame["frame_id"] * frame_rate[1], frame_rate[0])
        frame["time"] = [time.numerator, time.denominator]
        frame["time_s"] = float(time)
    doc["frame_range"] = [0, doc["source"]["frame_count"] - 1]
    doc["shots"][0]["last_frame"] = doc["source"]["frame_count"] - 1
    last_time = Fraction((doc["source"]["frame_count"] - 1) * frame_rate[1], frame_rate[0])
    doc["shots"][0]["last_time"] = [last_time.numerator, last_time.denominator]
    identity = np.eye(4).tolist()
    doc["geometry"]["registration"]["scene_transforms"][0]["matrix_4x4"] = identity
    for frame in doc["geometry"]["frames"]:
        frame["intrinsics"] = [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
        depth_path = corpus.predictions / "stage.assets" / frame["depth_asset"]
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth = np.full((10, 10), np.nan, dtype=np.float32)
        depth[7, 6] = 2.0
        np.save(depth_path, depth, allow_pickle=False)
        doc["assets"].append(
            {
                "path": frame["depth_asset"],
                "sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                "bytes": depth_path.stat().st_size,
            }
        )
    registration = doc["geometry"]["registration"]
    registration["scene_transforms"][0]["matrix_4x4"] = [
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for index, transform in enumerate(registration["camera_transforms"]):
        transform["matrix_4x4"] = [
            [1.0, 0.0, 0.0, 0.1 + index],
            [0.0, 1.0, 0.0, 1.8],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    path.write_text(json.dumps(doc))
    return path


def test_registered_depth_geometry_uses_camera_and_exact_stage_transform(tmp_path):
    manifest = _registered_manifest(tmp_path)
    stage_meta = {
        "shots": [
            {
                "display_frames": [
                    {
                        "source_frame": 0,
                        "camera_to_stage": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, -0.5],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                    {
                        "source_frame": 1,
                        "camera_to_stage": np.eye(4).tolist(),
                    },
                ]
            }
        ]
    }

    result = stage_environment.prepare(manifest, stage_meta)

    assert result["status"] == "registered_metric"
    assert result["frames"]["0"]["points"] == [[0.5, -0.9, -4.7]]
    assert result["frames"]["1"]["points"] == []
    assert "no exact registered camera transform" in result["frames"]["1"]["reason"]


def test_registered_depth_geometry_backprojects_skewed_intrinsics(tmp_path):
    manifest = _registered_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["geometry"]["frames"][0]["intrinsics"][0][1] = 2.0
    document["geometry"]["registration"]["scene_transforms"][0]["matrix_4x4"] = np.eye(4).tolist()
    document["geometry"]["registration"]["camera_transforms"][0]["matrix_4x4"] = np.eye(4).tolist()
    manifest.write_text(json.dumps(document))
    stage_meta = {
        "shots": [{"display_frames": [{"source_frame": 0, "camera_to_stage": np.eye(4).tolist()}]}]
    }

    result = stage_environment.prepare(manifest, stage_meta)

    assert result["frames"]["0"]["points"] == [[0.12, 0.4, 2.0]]


def test_character_cli_rejects_singular_environment_intrinsics_without_replacing_output(
    tmp_path, capsys
):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["geometry"]["frames"][0]["intrinsics"] = [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    manifest.write_text(json.dumps(document))
    output = tmp_path / "stage.html"
    output.write_text("prior stage")

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 2
    )

    assert "invertible and well-conditioned" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_rejects_environment_pose_hash_mismatch_before_output(tmp_path, capsys):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path, source_frame_count=20)
    document = json.loads(manifest.read_text())
    document["poses"] = {
        "path": str(pose),
        "association": "hash-verified",
        "file_sha256": "0" * 64,
        "frame_count": 20,
        "skeletons": [],
        "person_free_ranges": [],
    }
    manifest.write_text(json.dumps(document))
    output = tmp_path / "stage.html"
    output.write_text("prior stage")

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 2
    )

    assert "pose file sha256" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_uses_manifest_clock_and_rejects_explicit_mismatch(
    tmp_path, monkeypatch, capsys
):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(
        tmp_path, frame_ids=(0,), source_frame_count=20, frame_rate=(10, 1)
    )
    output = tmp_path / "stage.html"

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 0
    )
    from skeleton_maker.stage import Stage

    payload = output.read_text().split('id="payload">', 1)[1].split("</script>", 1)[0]
    assert Stage.from_payload(payload).meta["fps"] == 10.0

    output.write_text("prior stage")
    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(output),
                "--environment",
                str(manifest),
                "--fps",
                "30",
            ]
        )
        == 2
    )
    assert "does not match the environment source frame rate 10/1" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_rejects_pose_frames_outside_environment_source(tmp_path, capsys):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path, source_frame_count=10)
    output = tmp_path / "stage.html"
    output.write_text("prior stage")

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 2
    )

    assert "pose frame IDs must be within" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_hash_verifies_attached_video_and_uses_its_manifest_clock(tmp_path, capsys):
    from tests.test_environment import _make_video

    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    video = tmp_path / "source.mp4"
    _make_video(video, frames=60, size="10x10", rate="10")
    manifest = _registered_manifest(
        tmp_path, frame_ids=(0,), source_frame_count=20, frame_rate=(10, 1)
    )
    document = json.loads(manifest.read_text())
    document["source"]["sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    document["source"]["width"] = 10
    document["source"]["height"] = 10
    document["poses"] = {
        "path": str(pose),
        "association": "user-supplied",
        "file_sha256": hashlib.sha256(pose.read_bytes()).hexdigest(),
        "frame_count": 20,
        "skeletons": [],
        "person_free_ranges": [],
    }
    manifest.write_text(json.dumps(document))
    output = tmp_path / "stage.html"

    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(output),
                "--environment",
                str(manifest),
                "--video",
                str(video),
            ]
        )
        == 0
    )

    from skeleton_maker.stage import Stage

    payload = output.read_text().split('id="payload">', 1)[1].split("</script>", 1)[0]
    assert Stage.from_payload(payload).meta["fps"] == 10.0

    output.write_text("prior stage")
    document["source"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(output),
                "--environment",
                str(manifest),
                "--video",
                str(video),
            ]
        )
        == 2
    )
    assert "source video sha256" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_rejects_video_metadata_inconsistent_with_manifest(tmp_path, capsys):
    from tests.test_environment import _make_video

    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    video = tmp_path / "source.mp4"
    _make_video(video, frames=60, size="10x10", rate="10")
    manifest = _registered_manifest(
        tmp_path, frame_ids=(0,), source_frame_count=20, frame_rate=(9, 1)
    )
    document = json.loads(manifest.read_text())
    document["source"]["sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    document["source"]["width"] = 10
    document["source"]["height"] = 10
    manifest.write_text(json.dumps(document))
    output = tmp_path / "stage.html"
    output.write_text("prior stage")

    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(output),
                "--environment",
                str(manifest),
                "--video",
                str(video),
            ]
        )
        == 2
    )

    assert "source video frame rate does not match" in capsys.readouterr().err
    assert output.read_text() == "prior stage"


def test_character_cli_shows_only_exact_registered_source_frames(tmp_path):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path, frame_ids=(0, 2), source_frame_count=20)
    output = tmp_path / "stage.html"

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 0
    )

    page = output.read_text()
    options = json.loads(page.split('id="options">')[1].split("</script>")[0])
    environment = options["environment"]
    assert environment["status"] == "registered_metric"
    assert environment["frames"]["0"]["points"]
    assert environment["frames"]["2"]["points"]
    assert environment["frames"]["1"]["points"] == []
    assert "no exact registered camera transform" in environment["frames"]["1"]["reason"]
    assert 'id="environment-status"' in page


def test_registered_relative_geometry_is_hidden_with_reason(tmp_path):
    manifest = _registered_manifest(tmp_path, status="registered_relative")
    result = stage_environment.prepare(
        manifest,
        {
            "shots": [
                {"display_frames": [{"source_frame": 0, "camera_to_stage": np.eye(4).tolist()}]}
            ]
        },
    )

    assert result["status"] == "registered_relative"
    assert result["frames"]["0"]["points"] == []
    assert result["frames"]["0"]["reason"] == "test"


def test_character_cli_reports_malformed_environment_manifest_as_user_error(tmp_path):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path, frame_ids=(0,), source_frame_count=20)
    document = json.loads(manifest.read_text())
    del document["geometry"]["registration"]
    manifest.write_text(json.dumps(document))

    assert (
        cli.main(
            [
                "character",
                str(pose),
                "--out",
                str(tmp_path / "stage.html"),
                "--environment",
                str(manifest),
            ]
        )
        == 2
    )


def test_stage_cuts_and_source_gaps_never_reuse_registration_transforms(tmp_path):
    frames = []
    for frame_id in [*range(20), *range(90, 110)]:
        detection = clip(1)[0][1][0]
        detection["tracking_id"] = 1
        if frame_id >= 90:
            detection["keypoints_3d"] = (
                np.asarray(detection["keypoints_3d"]) + np.array([5.0, 0.0, 0.0])
            ).tolist()
            detection["root_pose"]["translation"][0] += 5.0
        frames.append({"frame_id": frame_id, "detections": [detection]})
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps(record) for record in frames))
    manifest = _registered_manifest(tmp_path, frame_ids=(0, 90), source_frame_count=110)
    output = tmp_path / "stage.html"

    assert (
        cli.main(["character", str(pose), "--out", str(output), "--environment", str(manifest)])
        == 0
    )

    page = output.read_text()
    options = json.loads(page.split('id="options">')[1].split("</script>")[0])
    environment = options["environment"]
    assert environment["frames"]["0"]["points"]
    assert environment["frames"]["90"]["points"]
    assert environment["frames"]["0"]["points"] != environment["frames"]["90"]["points"]
    for missing_frame in (1, 19, 91):
        assert environment["frames"][str(missing_frame)]["points"] == []
        assert (
            "no exact registered camera transform"
            in environment["frames"][str(missing_frame)]["reason"]
        )


def test_opt_in_stage_transform_maps_raw_pose_into_exact_rendered_coordinates():
    from skeleton_maker.stage import Options, build_stage

    frames = [
        (frame, [body(origin=(0.0, frame * 0.004, -5.0), tilt_deg=12.0)]) for frame in range(120)
    ]
    exact_pose = Options(smooth_sigma=0.0)
    ordinary = build_stage(frames, exact_pose)
    captured = build_stage(frames, exact_pose, capture_display_transforms=True)

    assert "display_frames" not in ordinary.meta["shots"][0]
    shot = captured.meta["shots"][0]
    assert [frame["source_frame"] for frame in shot["display_frames"]] == list(range(120))
    floor_offsets = [frame["camera_to_stage"][1][3] for frame in shot["display_frames"]]
    assert max(floor_offsets) - min(floor_offsets) > 0.01
    for frame_index in (0, 60, 119):
        camera_point = np.asarray(frames[frame_index][1][0]["keypoints_3d"][0] + [1.0])
        camera_to_stage = np.asarray(shot["display_frames"][frame_index]["camera_to_stage"])
        expected = (camera_to_stage @ camera_point)[:3]
        actual = captured.positions(0, 0)[frame_index, CANON_INDEX["Hips"]]
        np.testing.assert_allclose(actual, expected, atol=0.001)


@pytest.mark.parametrize("collision", ["pose", "manifest", "assets"])
def test_environment_character_cli_rejects_output_collisions_before_overwrite(tmp_path, collision):
    pose = tmp_path / "pose.jsonl"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    manifest = _registered_manifest(tmp_path, frame_ids=(0,), source_frame_count=20)
    if collision == "pose":
        output = pose
        protected = pose
    elif collision == "manifest":
        output = manifest
        protected = manifest
    else:
        output = manifest.parent / "stage.assets" / "stage.html"
        manifest_doc = json.loads(manifest.read_text())
        protected = (
            manifest.parent / "stage.assets" / manifest_doc["geometry"]["frames"][0]["depth_asset"]
        )
    original = protected.read_bytes()

    result = cli.main(
        ["character", str(pose), "--out", str(output), "--environment", str(manifest)]
    )

    assert result == 2
    assert protected.read_bytes() == original
