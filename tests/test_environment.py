# SPDX-License-Identifier: MIT
"""`skeleton-maker environment` through the public CLI, with only inference faked."""

import json
import shutil
import subprocess

import pytest

from skeleton_maker import cli, environment, envmanifest, envworkers

from .env_fakes import FakeBackend

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def _make_video(path, frames=30, size="64x48", rate="30"):
    subprocess.run(  # noqa: S603  fixed arguments to build a test fixture
        [  # noqa: S607  ffmpeg resolved from PATH; the test is skipped without it
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={rate}:duration={frames / 30:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_video(path)
    return path


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    return backend


def _run(argv):
    return cli.main(["environment", *argv])


def test_environment_writes_a_complete_manifest(clip, fake, tmp_path, capsys):
    out = tmp_path / "environment.json"

    rc = _run([str(clip), "--out", str(out), "--backend", "fake", "--geometry", "off"])

    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["schema"] == environment.SCHEMA_VERSION
    assert manifest["run"]["status"] == "complete"
    assert manifest["source"]["frame_rate"] == [30, 1]
    assert manifest["source"]["width"] == 64
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["geometry"]["status"] == "not_requested"
    assert [e["family"] for e in manifest["entities"]] == ["surface", "object"]
    assert manifest["backend"]["name"] == "fake"
    summary = capsys.readouterr().out
    assert str(out) in summary
    assert "frames 0-" in summary


def _snapshot(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _run_failing(argv, capsys, code=None):
    with pytest.raises(SystemExit) as exc:
        _run(argv)
    captured = capsys.readouterr()
    err = captured.err
    assert "environment: complete" not in captured.out
    assert exc.value.code not in (0, None)
    if code is not None:
        assert exc.value.code == code
    return err


def test_help_documents_every_option(capsys):
    with pytest.raises(SystemExit) as exc:
        _run(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in (
        "--out",
        "--backend",
        "--geometry",
        "--sample-fps",
        "--device",
        "--calibration",
        "--viewer",
    ):
        assert option in text


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--geometry", "maybe"], "invalid choice"),
        (["--sample-fps", "0"], "--sample-fps must be between"),
        (["--sample-fps", "500"], "--sample-fps must be between"),
        (["--device", "cuda"], "device 'cuda' is not available"),
        (["--backend", "nope"], "backend 'nope' is not installed"),
    ],
)
def test_rejected_options_fail_before_any_output(clip, fake, tmp_path, capsys, extra, message):
    before = _snapshot(tmp_path)
    out = tmp_path / "environment.json"

    err = _run_failing([str(clip), "--out", str(out), "--backend", "fake", *extra], capsys)

    assert message in err
    assert _snapshot(tmp_path) == before
    assert fake.requests == []


def test_sample_rate_above_the_clip_rate_is_rejected(fake, tmp_path, capsys):
    slow = tmp_path / "slow.mp4"
    _make_video(slow, frames=30, rate="10")
    err = _run_failing(
        [str(slow), "--out", str(tmp_path / "e.json"), "--backend", "fake", "--sample-fps", "20"],
        capsys,
    )
    assert "exceeds the clip's frame rate 10" in err


def test_missing_input_and_empty_video_are_rejected(fake, tmp_path, capsys):
    out = tmp_path / "environment.json"
    err = _run_failing([str(tmp_path / "gone.mp4"), "--out", str(out), "--backend", "fake"], capsys)
    assert "no such file" in err

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    err = _run_failing([str(empty), "--out", str(out), "--backend", "fake"], capsys)
    assert "cannot read" in err or "no video stream" in err
    assert not out.exists()


def test_default_backend_is_reported_missing_not_silently_faked(clip, tmp_path, capsys):
    err = _run_failing([str(clip), "--out", str(tmp_path / "e.json")], capsys)
    assert "grounded-sam2-da3" in err
    assert "not installed" in err


def test_auto_device_picks_what_the_backend_reports(clip, monkeypatch, tmp_path):
    backend = FakeBackend(devices=("mps", "cpu"))
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    assert _run([str(clip), "--out", str(tmp_path / "e.json"), "--backend", "fake"]) == 0

    assert backend.requests[0]["device"] == "mps"


def test_sampling_hits_expected_source_frames(clip, fake, tmp_path):
    out = tmp_path / "e.json"
    _run([str(clip), "--out", str(out), "--backend", "fake", "--sample-fps", "10"])
    manifest = json.loads(out.read_text())
    assert [f["frame_id"] for f in manifest["processed_frames"]] == [
        0,
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
    ]
    assert manifest["processed_frames"][3]["time"] == [3, 10]
    assert manifest["frame_range"] == [0, 27]


def test_time_mapping_is_exact_for_ntsc_rates(tmp_path, fake):
    ntsc = tmp_path / "ntsc.mp4"
    _make_video(ntsc, frames=30, rate="30000/1001")
    out = tmp_path / "e.json"

    _run([str(ntsc), "--out", str(out), "--backend", "fake", "--sample-fps", "2"])

    manifest = json.loads(out.read_text())
    assert manifest["source"]["frame_rate"] == [30000, 1001]
    # frame 15 of a 30000/1001 clip is at 15 * 1001 / 30000 s = 1001/2000 s exactly.
    assert manifest["processed_frames"][1]["frame_id"] == 15
    assert manifest["processed_frames"][1]["time"] == [1001, 2000]


def test_geometry_modes_without_a_geometry_backend(clip, fake, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(envworkers, "discover_geometry", lambda: None)
    out = tmp_path / "e.json"
    assert _run([str(clip), "--out", str(out), "--backend", "fake", "--geometry", "auto"]) == 0
    manifest = json.loads(out.read_text())
    geometry = manifest["geometry"]
    assert geometry["status"] == "unavailable"
    assert "DA3 geometry worker is not installed" in geometry["reason"]
    assert manifest["run"]["status"] == "partial"
    assert manifest["run"]["perception_status"] == "complete"
    assert "semantic scan completed, but spatial registration is partial" in capsys.readouterr().err

    out.unlink()
    capsys.readouterr()
    err = _run_failing(
        [str(clip), "--out", str(out), "--backend", "fake", "--geometry", "required"], capsys
    )
    assert "independent DA3 worker" in err
    assert not out.exists()


def test_required_geometry_fails_if_the_backend_cannot_register(
    clip, monkeypatch, tmp_path, capsys
):
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(geometry=True))
    out = tmp_path / "e.json"

    err = _run_failing(
        [str(clip), "--out", str(out), "--backend", "fake", "--geometry", "required"], capsys
    )

    assert "no valid shared metric registration" in err
    assert not out.exists()


def test_partial_backend_result_is_marked_and_exits_nonzero(clip, monkeypatch, tmp_path, capsys):
    def partial(resp):
        resp["status"] = "partial"
        resp["reason"] = "frames 20-29 failed to decode"

    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(mutate=partial))
    out = tmp_path / "e.json"

    rc = _run([str(clip), "--out", str(out), "--backend", "fake"])

    assert rc == environment.EXIT_PARTIAL
    run = json.loads(out.read_text())["run"]
    assert run["status"] == "partial"
    assert "failed to decode" in run["reason"]
    assert "geometry registration unavailable" in run["reason"]
    assert "partial" in capsys.readouterr().err


def _bad_contract(r):
    r["contract"] = "something/else"


def _nan_score(r):
    r["observations"][0]["score"] = float("nan")


def _unknown_entity(r):
    r["observations"][0]["entity"] = "shot-0/ghost"


def _outside_bounds(r):
    r["observations"][0]["bbox"] = [0, 0, 640, 480]


def _unprocessed_frame(r):
    r["observations"][0]["frame_id"] = 1


def _missing_mask(r):
    r["observations"][0]["mask"] = {"asset": "masks/not-written.png"}


def _escaping_mask(r):
    r["observations"][0]["mask"] = {"asset": "../../etc/passwd"}


def _wrong_family(r):
    r["entities"][0]["family"] = "furniture"


def _failed_status(r):
    r["status"] = "failed"


@pytest.mark.parametrize(
    "mutate",
    [
        _bad_contract,
        _nan_score,
        _unknown_entity,
        _outside_bounds,
        _unprocessed_frame,
        _missing_mask,
        _escaping_mask,
        _wrong_family,
        _failed_status,
    ],
)
def test_malformed_backend_output_never_reports_success(
    clip, monkeypatch, tmp_path, capsys, mutate
):
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(mutate=mutate))
    out = tmp_path / "environment.json"
    out.write_text('{"previous": "result"}')
    assets = tmp_path / "environment.assets"
    assets.mkdir()
    (assets / "old.png").write_bytes(b"old")
    before = _snapshot(tmp_path)

    err = _run_failing([str(clip), "--out", str(out), "--backend", "fake"], capsys)

    assert "environment scan failed" in err
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("raises", "code"),
    [(RuntimeError("worker crashed"), 1), (KeyboardInterrupt(), environment.EXIT_INTERRUPTED)],
)
def test_backend_failure_or_interrupt_preserves_prior_output(
    clip, monkeypatch, tmp_path, capsys, raises, code
):
    monkeypatch.setitem(environment.BACKENDS, "fake", FakeBackend(raises=raises))
    out = tmp_path / "environment.json"
    out.write_text('{"previous": "result"}')
    before = _snapshot(tmp_path)

    if code == 1:
        err = _run_failing([str(clip), "--out", str(out), "--backend", "fake"], capsys, code=1)
        assert "worker crashed" in err
    else:
        assert _run([str(clip), "--out", str(out), "--backend", "fake"]) == code
        assert "interrupted" in capsys.readouterr().err

    assert _snapshot(tmp_path) == before


def test_failed_write_restores_the_previous_artifact(clip, fake, monkeypatch, tmp_path):
    out = tmp_path / "environment.json"
    out.write_text('{"previous": "result"}')
    assets = tmp_path / "environment.assets"
    assets.mkdir()
    (assets / "old.png").write_bytes(b"old")
    before = _snapshot(tmp_path)

    real_replace = environment.artifacts.os.replace

    def failing_replace(src, dst):
        if str(dst) == str(out):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(environment.artifacts.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        _run([str(clip), "--out", str(out), "--backend", "fake"])
    monkeypatch.undo()

    assert _snapshot(tmp_path) == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_rerun_replaces_the_artifact_and_stale_assets(clip, fake, tmp_path):
    out = tmp_path / "environment.json"
    _run([str(clip), "--out", str(out), "--backend", "fake"])
    stale = tmp_path / "environment.assets" / "stale.png"
    stale.write_bytes(b"stale")

    _run([str(clip), "--out", str(out), "--backend", "fake"])

    assert not stale.exists()
    assert (tmp_path / "environment.assets" / "masks" / "floor-0.png").exists()
    assert json.loads(out.read_text())["run"]["status"] == "partial"


@pytest.fixture
def written(clip, fake, tmp_path):
    out = tmp_path / "environment.json"
    _run([str(clip), "--out", str(out), "--backend", "fake"])
    return out


def _edit(path, fn):
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def test_manifest_round_trips_through_the_loader(written):
    doc = envmanifest.load_manifest(written)
    assert doc["assets"][0]["path"] == "masks/floor-0.png"
    assert doc["observations"][0]["mask"] == {"asset": "masks/floor-0.png"}


def test_manifest_loader_reports_missing_entity_family_as_manifest_error(written):
    _edit(written, lambda doc: doc["entities"][0].pop("family"))

    with pytest.raises(envmanifest.ManifestError, match=r"entities\[0\]: missing 'family'"):
        envmanifest.load_manifest(written)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(schema="skeleton-maker.environment/99"), "unsupported manifest schema"),
        (lambda d: d.pop("entities"), "missing 'entities'"),
        (lambda d: d["run"].update(status="finished"), "run.status"),
        (lambda d: d["assets"][0].update(path="../outside.png"), "escapes the bundle"),
        (lambda d: d["assets"][0].update(path="/etc/passwd"), "unsafe asset path"),
        (lambda d: d["assets"][0].update(sha256="0" * 64), "does not match"),
        (lambda d: d["assets"].append({"path": "gone.png", "sha256": "0" * 64}), "missing from"),
    ],
)
def test_loader_rejects_bad_manifests(written, edit, message):
    _edit(written, edit)
    with pytest.raises(envmanifest.ManifestError, match=message):
        envmanifest.load_manifest(written)


def test_loader_rejects_nan_and_bad_json(written):
    written.write_text(written.read_text().replace('"score": 0.9', '"score": NaN'))
    with pytest.raises(envmanifest.ManifestError, match="non-finite"):
        envmanifest.load_manifest(written)
    written.write_text("{not json")
    with pytest.raises(envmanifest.ManifestError, match="not valid JSON"):
        envmanifest.load_manifest(written)


def test_loader_rejects_a_symlink_that_leaves_the_bundle(written, tmp_path):
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    asset = tmp_path / "environment.assets" / "masks" / "floor-0.png"
    asset.unlink()
    asset.symlink_to(outside)
    with pytest.raises(envmanifest.ManifestError, match="escapes the bundle"):
        envmanifest.load_manifest(written)
