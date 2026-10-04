# SPDX-License-Identifier: MIT
"""The opt-in combined CLI is exercised through its public command boundary."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from skeleton_maker import artifacts, cli, environment

from .env_fakes import FakeBackend
from .test_environment import _make_video, _snapshot
from .test_environment_overlay import _pose_record

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _install_inference_stubs(monkeypatch, *, nim_calls, body=True):
    def track(args):
        args_out = args.out_bbox
        with open(args_out, "w") as stream:
            stream.write("1\n")
            stream.writelines(f"{frame} 4 54 2 8 10\n" for frame in range(30))
        return 0

    def estimate(video, boxes, output, **kwargs):
        nim_calls.append((video, boxes, output))
        fingerprint = artifacts.sha256_file(video)
        records: list[str] = []
        for frame_id in range(30):
            record: dict[str, object] = (
                _pose_record(frame_id) if body else {"frame_id": frame_id, "detections": []}
            )
            record["source_sha256"] = fingerprint
            records.append(json.dumps(record))
        with open(output, "w") as stream:
            stream.write("\n".join(records) + "\n")
        return {
            "frames": 30,
            "frames_with_bodies": 30 if body else 0,
            "seconds": 0.01,
            "focal_length": 0.0,
        }

    monkeypatch.setattr(cli.detection, "run_track", track)
    monkeypatch.setattr(cli.nim, "run", estimate)


def _read_frame(path, frame_id):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        assert ok, f"could not read {path} frame {frame_id}"
        return frame
    finally:
        capture.release()


def _environment_all_args(source, work):
    return [
        "all",
        str(source),
        "--work",
        str(work),
        "--environment",
        "--environment-backend",
        "fake",
        "--environment-geometry",
        "off",
        "--environment-sample-fps",
        "2",
        "--environment-device",
        "cpu",
        "--environment-no-cache",
    ]


def test_all_environment_reuses_one_pose_run_and_publishes_synchronized_outputs(
    monkeypatch, tmp_path
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    nim_calls = []
    _install_inference_stubs(monkeypatch, nim_calls=nim_calls)
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    assert cli.main(_environment_all_args(source, work)) == 0

    expected = [
        "clip.mp4",
        "boxes.txt",
        "pose.json",
        "overlay.mp4",
        "environment.json",
        "environment.assets/masks/floor-0.png",
        "environment-overlay.mp4",
        "environment.html",
        "environment.viewer.assets/viewer.js",
    ]
    assert all((work / path).is_file() for path in expected)
    assert len(nim_calls) == 1
    assert len(backend.requests) == 1
    manifest = json.loads((work / "environment.json").read_text())
    assert manifest["source"]["path"] == str(work / "clip.mp4")
    assert manifest["poses"]["path"] == str(work / "pose.json")
    assert manifest["source"]["sha256"] == artifacts.sha256_file(work / "clip.mp4")
    assert backend.requests[0]["poses"]["frame_count"] == 30
    assert backend.requests[0]["poses"]["association"] == "hash-verified"

    clip_observed = _read_frame(work / "clip.mp4", 0)
    overlay_observed = _read_frame(work / "environment-overlay.mp4", 0)
    clip_unsampled = _read_frame(work / "clip.mp4", 1)
    overlay_unsampled = _read_frame(work / "environment-overlay.mp4", 1)
    clip_pose_frame = _read_frame(work / "clip.mp4", 14)
    overlay_pose_frame = _read_frame(work / "environment-overlay.mp4", 14)
    clip_later_observation = _read_frame(work / "clip.mp4", 15)
    overlay_later_observation = _read_frame(work / "environment-overlay.mp4", 15)
    assert np.abs(overlay_observed[40, 30].astype(int) - clip_observed[40, 30]).max() > 25
    assert np.abs(overlay_unsampled[40, 30].astype(int) - clip_unsampled[40, 30]).max() < 25
    assert np.abs(overlay_pose_frame[6, 58].astype(int) - clip_pose_frame[6, 58]).max() > 80
    assert (
        np.abs(overlay_later_observation[20, 10].astype(int) - clip_later_observation[20, 10]).max()
        > 50
    )


def test_all_environment_publishes_semantic_outputs_when_auto_registration_abstains(
    monkeypatch, tmp_path
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    _install_inference_stubs(monkeypatch, nim_calls=[])
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    monkeypatch.setattr(environment.envworkers, "discover_geometry", lambda: None)
    argv = _environment_all_args(source, work)
    argv[argv.index("off")] = "auto"

    rc = cli.main(argv)

    assert rc == 0
    assert (work / "pose.json").is_file()
    assert (work / "overlay.mp4").is_file()
    assert (work / "environment.json").is_file()
    assert (work / "environment-overlay.mp4").is_file()
    assert (work / "environment.html").is_file()
    manifest = json.loads((work / "environment.json").read_text())
    assert manifest["run"]["status"] == "partial"
    assert manifest["run"]["perception_status"] == "complete"
    assert manifest["geometry"]["status"] == "unavailable"


def test_all_environment_preflights_missing_backend_before_tracking_or_nim(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    before = _snapshot(tmp_path)
    calls = []
    monkeypatch.setattr(environment, "BACKENDS", {})
    monkeypatch.setattr(environment.envworkers, "discover", lambda name: None)
    monkeypatch.setattr(cli.detection, "run_track", lambda args: calls.append("track"))
    monkeypatch.setattr(cli.nim, "run", lambda *args, **kwargs: calls.append("nim"))

    with pytest.raises(SystemExit) as exc:
        cli.main(_environment_all_args(source, work))

    assert exc.value.code == 2
    assert "fake" in capsys.readouterr().err
    assert calls == []
    assert _snapshot(tmp_path) == before


def test_all_environment_failure_preserves_every_old_output(monkeypatch, tmp_path):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    nim_calls = []
    _install_inference_stubs(monkeypatch, nim_calls=nim_calls)
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = _environment_all_args(source, work)
    assert cli.main(argv) == 0
    for path in work.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes() + b"\nold output sentinel")
    before = _snapshot(tmp_path)

    monkeypatch.setattr(
        backend,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(environment.BackendError("injected failure")),
    )
    with pytest.raises(SystemExit):
        cli.main(argv)

    assert _snapshot(tmp_path) == before
    assert len(nim_calls) == 2


def test_all_environment_group_publication_failure_rolls_back_every_old_output(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    nim_calls = []
    _install_inference_stubs(monkeypatch, nim_calls=nim_calls)
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = _environment_all_args(source, work)
    assert cli.main(argv) == 0
    for path in work.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes() + b"\nold output sentinel")
    before = _snapshot(tmp_path)

    real_replace = artifacts.os.replace
    fail_once = True

    def fail_final_environment_overlay(src, dst):
        nonlocal fail_once
        if Path(dst) == work / "environment-overlay.mp4" and fail_once:
            fail_once = False
            raise OSError("injected combined output publication failure")
        return real_replace(src, dst)

    monkeypatch.setattr(artifacts.os, "replace", fail_final_environment_overlay)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 1
    assert "injected combined output publication failure" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before
    assert len(nim_calls) == 2


def test_all_environment_output_collision_fails_before_tracking_or_nim(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    calls = []
    monkeypatch.setattr(cli.detection, "run_track", lambda args: calls.append("track"))
    monkeypatch.setattr(cli.nim, "run", lambda *args, **kwargs: calls.append("nim"))
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = _environment_all_args(source, work)
    argv.extend(["--out", str(work / "environment-overlay.mp4")])

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    assert "conflicts" in capsys.readouterr().err
    assert calls == []
    assert backend.requests == []


def test_all_environment_calibration_collision_fails_before_tracking_or_nim(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    _make_video(source)
    calls = []
    monkeypatch.setattr(cli.detection, "run_track", lambda args: calls.append("track"))
    monkeypatch.setattr(cli.nim, "run", lambda *args, **kwargs: calls.append("nim"))
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = _environment_all_args(source, work)
    argv.extend(["--environment-calibration", str(work / "environment-overlay.mp4")])

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    assert "environment overlay would overwrite the calibration file" in capsys.readouterr().err
    assert calls == []
    assert backend.requests == []


def test_all_environment_validates_calibration_after_conforming_before_tracking_or_nim(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "input.mp4"
    work = tmp_path / "run"
    calibration = tmp_path / "calibration.json"
    _make_video(source)
    calibration.write_text(
        json.dumps(
            {
                "schema": environment.envgeometry.REFERENCE_SCHEMA,
                "image_size": [32, 24],
                "camera": {
                    "intrinsics": [[30, 0, 16], [0, 30, 12], [0, 0, 1]],
                },
            }
        )
    )
    before = _snapshot(tmp_path)
    calls = []
    monkeypatch.setattr(cli.detection, "run_track", lambda args: calls.append("track"))
    monkeypatch.setattr(cli.nim, "run", lambda *args, **kwargs: calls.append("nim"))
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = _environment_all_args(source, work)
    argv.extend(["--environment-calibration", str(calibration)])

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    assert (
        "calibration image_size does not match the source clip (64x48)" in capsys.readouterr().err
    )
    assert calls == []
    assert backend.requests == []
    assert _snapshot(tmp_path) == before
