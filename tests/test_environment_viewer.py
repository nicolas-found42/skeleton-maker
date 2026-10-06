# SPDX-License-Identifier: MIT
"""Offline environment viewer behavior through the public command and page."""

import html
import json
import shutil

import pytest

from skeleton_maker import cli, environment

from .env_fakes import FakeBackend
from .test_environment import _make_video, _snapshot

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _pose_record(frame_id, tracking_id=4):
    points = [[float(i), float(i + 1)] for i in range(77)]
    return {
        "frame_id": frame_id,
        "detections": [
            {
                "tracking_id": tracking_id,
                "bbox": [5.0, 5.0, 30.0, 40.0],
                "keypoints_2d": points,
                "keypoints_confidence": [1.0] * 77,
                "root_pose": {"translation": [0.0, 0.0, 3.0], "rotation": [0, 0, 0, 1]},
            }
        ],
    }


@pytest.mark.parametrize(
    "collision",
    [
        "viewer-over-video",
        "viewer-over-poses",
        "viewer-over-manifest",
        "viewer-bundle-contains-video",
        "viewer-bundle-contains-poses",
        "viewer-under-environment-assets",
        "viewer-bundle-contains-manifest",
        "viewer-bundle-equals-environment-assets",
    ],
)
def test_environment_viewer_rejects_output_collisions_before_inference(
    monkeypatch, tmp_path, capsys, collision
):
    source = tmp_path / "clip.mp4"
    _make_video(source)
    pose_path = tmp_path / "poses.jsonl"
    pose_path.write_text("".join(json.dumps(_pose_record(i)) + "\n" for i in range(30)))
    out = tmp_path / "environment.json"
    viewer = tmp_path / "review.html"
    expected_source = source.read_bytes()
    expected_poses = pose_path.read_bytes()

    if collision == "viewer-over-video":
        viewer = source
    elif collision == "viewer-over-poses":
        viewer = pose_path
    elif collision == "viewer-over-manifest":
        viewer = out
    elif collision == "viewer-bundle-contains-video":
        bundle = tmp_path / "review.viewer.assets"
        bundle.mkdir()
        source = bundle / "source.mp4"
        _make_video(source)
        expected_source = source.read_bytes()
    elif collision == "viewer-bundle-contains-poses":
        bundle = tmp_path / "review.viewer.assets"
        bundle.mkdir()
        pose_path = bundle / "poses.jsonl"
        pose_path.write_text("".join(json.dumps(_pose_record(i)) + "\n" for i in range(30)))
        expected_poses = pose_path.read_bytes()
    elif collision == "viewer-under-environment-assets":
        viewer = tmp_path / "environment.assets" / "preview.html"
    elif collision == "viewer-bundle-contains-manifest":
        out = tmp_path / "review.viewer.assets" / "environment.json"
        out.parent.mkdir()
    elif collision == "viewer-bundle-equals-environment-assets":
        out = tmp_path / "review.viewer.json"

    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = [str(source), "--out", str(out), "--backend", "fake", "--geometry", "off"]
    if collision in ("viewer-over-poses", "viewer-bundle-contains-poses"):
        argv.extend(["--poses", str(pose_path)])
    argv.extend(["--viewer", str(viewer)])

    with pytest.raises(SystemExit) as exc:
        cli.main(["environment", *argv])

    assert exc.value.code == 2
    assert "viewer" in capsys.readouterr().err.lower()
    assert backend.requests == []
    assert source.read_bytes() == expected_source
    if pose_path.exists():
        assert pose_path.read_bytes() == expected_poses


@pytest.mark.parametrize("reserved_name", ["source.mp4", "viewer.js"])
def test_environment_viewer_namespaces_masks_away_from_video_and_runtime(
    monkeypatch, tmp_path, reserved_name
):
    from scripts.make_environment_viewer_fixture import _mask_png

    class ReservedNameMaskBackend(FakeBackend):
        def run(self, request, assets_dir):
            response = super().run(request, assets_dir)
            mask = assets_dir / "masks" / "floor-0.png"
            reserved = assets_dir / reserved_name
            reserved.write_bytes(_mask_png(64, 48, bounds=(0, 20, 63, 47)))
            mask.unlink()
            for observation in response["observations"]:
                if observation["mask"] is not None:
                    observation["mask"]["asset"] = reserved_name
            return response

    video = tmp_path / "clip.mp4"
    _make_video(video)
    expected_video = video.read_bytes()
    monkeypatch.setitem(environment.BACKENDS, "fake", ReservedNameMaskBackend())
    out = tmp_path / "environment.json"
    viewer = tmp_path / "review.html"

    assert (
        cli.main(
            [
                "environment",
                str(video),
                "--out",
                str(out),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--viewer",
                str(viewer),
            ]
        )
        == 0
    )

    bundle = tmp_path / "review.viewer.assets"
    assert (bundle / "source.mp4").read_bytes() == expected_video
    assert b"Offline source-frame viewer" in (bundle / "viewer.js").read_bytes()
    assert (bundle / "environment" / reserved_name).read_bytes() == _mask_png(
        64, 48, bounds=(0, 20, 63, 47)
    )
    payload = json.loads(viewer.read_text().split('id="payload">', 1)[1].split("</script>", 1)[0])
    assert payload["observations"][0]["mask"]["asset"] == f"environment/{reserved_name}"


def test_environment_viewer_exports_local_video_masks_and_pose_data(monkeypatch, tmp_path):
    video = tmp_path / "source #1.mp4"
    _make_video(video)
    poses = tmp_path / "poses.jsonl"
    poses.write_text("".join(json.dumps(_pose_record(i)) + "\n" for i in range(30)))
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend())
    out = tmp_path / "environment.json"
    viewer = tmp_path / "review.html"

    assert (
        cli.main(
            [
                "environment",
                str(video),
                "--out",
                str(out),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--poses",
                str(poses),
                "--viewer",
                str(viewer),
            ]
        )
        == 0
    )

    bundle = tmp_path / "review.viewer.assets"
    assert viewer.is_file()
    assert (bundle / "source.mp4").read_bytes() == video.read_bytes()
    assert (bundle / "environment/masks/floor-0.png").is_file()
    assert (bundle / "viewer.js").is_file()
    page = viewer.read_text()
    assert 'src="review.viewer.assets/source.mp4"' in page
    assert 'src="review.viewer.assets/viewer.js"' in page
    assert "__PAYLOAD__" not in page
    assert "keypoints_2d" in page


def test_environment_viewer_escapes_user_text_and_rejects_unsafe_assets(tmp_path):
    from skeleton_maker.environment_viewer import build_viewer_html
    from skeleton_maker.envmanifest import ManifestError

    payload = {
        "source": {"width": 10, "height": 10, "frame_rate": [30, 1], "frame_count": 1},
        "processed_frames": [{"frame_id": 0, "time_s": 0}],
        "shots": [],
        "entities": [{"id": "<img src=x onerror=alert(1)>", "labels": {"native": "<script>"}}],
        "observations": [{"mask": {"asset": "../outside.png"}}],
        "poses": [],
        "geometry": {"status": "unavailable", "reason": "none"},
    }

    with pytest.raises(ManifestError, match=r"unsafe asset path|escapes the bundle"):
        build_viewer_html(payload, title="</title><script>alert(1)</script>")

    payload["observations"] = []
    page = build_viewer_html(payload, title="</title><script>alert(1)</script>")
    assert "</title><script>alert(1)</script>" not in page
    embedded = page.split('id="payload">', 1)[1].split("</script>", 1)[0]
    restored = html.unescape(embedded.replace("<\\/", "</"))
    decoded = json.loads(restored)
    assert decoded["entities"][0]["id"] == "<img src=x onerror=alert(1)>"


def test_environment_viewer_reports_a_missing_mask_asset(monkeypatch, tmp_path):
    from skeleton_maker.environment_viewer import export_viewer
    from skeleton_maker.envmanifest import ManifestError

    video = tmp_path / "source.mp4"
    _make_video(video)
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend())
    manifest = tmp_path / "environment.json"
    assert (
        cli.main(
            [
                "environment",
                str(video),
                "--out",
                str(manifest),
                "--backend",
                "fake",
                "--geometry",
                "off",
            ]
        )
        == 0
    )
    (tmp_path / "environment.assets/masks/floor-0.png").unlink()

    with pytest.raises(ManifestError, match=r"asset .* is referenced but missing"):
        export_viewer(manifest, str(video), str(tmp_path / "viewer.html"))
    assert not (tmp_path / "viewer.html").exists()


@pytest.mark.parametrize("failure", ["copy", "write", "publish"])
def test_viewer_write_failure_preserves_all_existing_outputs(
    monkeypatch, tmp_path, capsys, failure
):
    from pathlib import Path

    from skeleton_maker import artifacts, environment_viewer

    video = tmp_path / "clip.mp4"
    _make_video(video)
    manifest = tmp_path / "environment.json"
    viewer = tmp_path / "review.html"
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    argv = [
        "environment",
        str(video),
        "--out",
        str(manifest),
        "--backend",
        "fake",
        "--geometry",
        "off",
        "--no-cache",
        "--viewer",
        str(viewer),
    ]

    assert cli.main(argv) == 0
    bundle = tmp_path / "review.viewer.assets"
    assert manifest.is_file()
    assert (tmp_path / "environment.assets").is_dir()
    assert viewer.is_file()
    assert bundle.is_dir()
    with manifest.open("ab") as stream:
        stream.write(b"\nprevious manifest sentinel")
    (tmp_path / "environment.assets" / "previous.bin").write_bytes(b"previous environment asset")
    with viewer.open("ab") as stream:
        stream.write(b"\nprevious viewer sentinel")
    (bundle / "previous.bin").write_bytes(b"previous viewer asset")
    before = _snapshot(tmp_path)

    if failure == "copy":
        real_copyfile = environment_viewer.shutil.copyfile

        def fail_viewer_video_copy(src, dst, *args, **kwargs):
            destination = Path(dst)
            if (
                destination.name == f"source{video.suffix}"
                and destination.parent.name == bundle.name
                and destination.parent.parent.name.startswith(".staging-")
            ):
                raise OSError("injected viewer copy failure")
            return real_copyfile(src, dst, *args, **kwargs)

        monkeypatch.setattr(environment_viewer.shutil, "copyfile", fail_viewer_video_copy)
    elif failure == "write":
        real_write_text = Path.write_text

        def fail_viewer_page_write(path, *args, **kwargs):
            if path.name == viewer.name and path.parent.name.startswith(".staging-"):
                raise OSError("injected viewer write failure")
            return real_write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_viewer_page_write)
    else:
        real_replace = artifacts.os.replace
        fail_once = True

        def fail_viewer_publication(src, dst):
            nonlocal fail_once
            if Path(dst) == viewer and fail_once:
                fail_once = False
                raise OSError("injected viewer publication failure")
            return real_replace(src, dst)

        monkeypatch.setattr(artifacts.os, "replace", fail_viewer_publication)

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 1
    assert "injected viewer" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before
    assert len(backend.requests) == 2


def test_viewer_backup_cleanup_failure_warns_after_committing_new_outputs(
    monkeypatch, tmp_path, capsys
):
    from pathlib import Path

    from skeleton_maker import artifacts

    video = tmp_path / "clip.mp4"
    _make_video(video)
    manifest = tmp_path / "environment.json"
    viewer = tmp_path / "review.html"
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend())
    argv = [
        "environment",
        str(video),
        "--out",
        str(manifest),
        "--backend",
        "fake",
        "--geometry",
        "off",
        "--no-cache",
        "--viewer",
        str(viewer),
    ]
    assert cli.main(argv) == 0
    (tmp_path / "environment.assets" / "previous.bin").write_bytes(b"old environment asset")
    with viewer.open("ab") as stream:
        stream.write(b"\nold viewer page")
    (tmp_path / "review.viewer.assets" / "previous.bin").write_bytes(b"old viewer bundle")

    real_remove = artifacts._remove_artifact

    def fail_backup_cleanup(path):
        if Path(path).name.endswith(".old"):
            raise OSError("injected backup cleanup failure")
        return real_remove(path)

    monkeypatch.setattr(artifacts, "_remove_artifact", fail_backup_cleanup)

    assert cli.main(argv) == 0

    captured = capsys.readouterr()
    assert "could not remove recovery backup" in captured.err
    assert "injected backup cleanup failure" in captured.err
    assert "old viewer page" not in viewer.read_text()
    assert not (tmp_path / "environment.assets" / "previous.bin").exists()
    assert not (tmp_path / "review.viewer.assets" / "previous.bin").exists()
    backups = list(tmp_path.glob(".*.old"))
    assert len(backups) == 4


def test_standalone_viewer_rejects_a_different_readable_video(monkeypatch, tmp_path):
    from skeleton_maker import environment_viewer, envmanifest

    source = tmp_path / "source.mp4"
    other = tmp_path / "other.mp4"
    _make_video(source)
    _make_video(other, size="80x60")
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend())
    manifest = tmp_path / "environment.json"
    assert (
        cli.main(
            [
                "environment",
                str(source),
                "--out",
                str(manifest),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--no-cache",
            ]
        )
        == 0
    )
    output = tmp_path / "viewer.html"
    with pytest.raises(envmanifest.ManifestError, match=r"source.*hash|source.*match"):
        environment_viewer.export_viewer(manifest, other, output)
    assert not output.exists()
    assert not (tmp_path / "viewer.viewer.assets").exists()
