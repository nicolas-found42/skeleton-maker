# SPDX-License-Identifier: MIT
"""The inference cache: reused only for identical inputs and settings, never trusted blindly."""

import json
import shutil

import pytest

from skeleton_maker import cli, environment

from .env_fakes import FakeBackend
from .test_environment import _make_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)

IDENTITY = {"model": "fake-detector-1", "checkpoint_sha256": "a" * 64, "preprocessing": "p1"}


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_video(path, frames=30)
    return path


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend(devices=("cpu", "mps"), identity=IDENTITY)
    monkeypatch.setitem(environment.BACKENDS, "fake", fake)
    return fake


def _run(clip, tmp_path, *extra, out="environment.json"):
    argv = [
        "environment",
        str(clip),
        "--out",
        str(tmp_path / out),
        "--backend",
        "fake",
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]
    return cli.main(argv)


def test_identical_rerun_reuses_the_cache_without_calling_the_backend(
    clip, backend, tmp_path, capsys
):
    assert _run(clip, tmp_path) == 0
    first = json.loads((tmp_path / "environment.json").read_text())
    assert first["run"]["status"] == "partial"
    assert first["run"]["perception_status"] == "complete"
    assert "cache miss" in capsys.readouterr().out

    assert _run(clip, tmp_path, out="second.json") == 0

    second = json.loads((tmp_path / "second.json").read_text())
    assert second["run"]["status"] == "partial"
    assert second["run"]["perception_status"] == "complete"
    assert len(backend.requests) == 1, "the second run must not reach the backend"
    assert "cache hit" in capsys.readouterr().out
    assert second["observations"] == first["observations"]
    assert second["entities"] == first["entities"]
    assert second["assets"] == first["assets"]
    assert (tmp_path / "second.assets" / "masks" / "floor-0.png").exists()


@pytest.mark.parametrize(
    "change",
    [
        ["--sample-fps", "10"],
        ["--geometry", "off"],
        ["--device", "mps"],
    ],
)
def test_changed_settings_invalidate_the_cache(clip, backend, tmp_path, change):
    _run(clip, tmp_path)

    _run(clip, tmp_path, *change, out="other.json")

    assert len(backend.requests) == 2


def test_a_different_clip_invalidates_the_cache(clip, backend, tmp_path):
    _run(clip, tmp_path)
    other = tmp_path / "other.mp4"
    _make_video(other, frames=30, size="80x60")

    _run(other, tmp_path, out="other.json")

    assert len(backend.requests) == 2


@pytest.mark.parametrize(
    "new_identity",
    [
        {**IDENTITY, "model": "fake-detector-2"},
        {**IDENTITY, "checkpoint_sha256": "b" * 64},
        {**IDENTITY, "preprocessing": "p2"},
        {**IDENTITY, "settings": {"box_threshold": 0.3}},
    ],
)
def test_a_changed_model_checkpoint_or_preprocessing_invalidates_the_cache(
    clip, backend, monkeypatch, tmp_path, new_identity
):
    _run(clip, tmp_path)
    changed = FakeBackend(devices=("cpu", "mps"), identity=new_identity)
    monkeypatch.setitem(environment.BACKENDS, "fake", changed)

    _run(clip, tmp_path, out="other.json")

    assert changed.requests, "a new identity must run the backend again"


def test_requested_labels_are_part_of_the_cache_key():
    request = {
        "source_sha256": "s",
        "frames": [{"frame_id": 0}],
        "shots": [],
        "device": "cpu",
        "geometry": "off",
    }
    a = environment.cache_key({"m": 1}, {**request, "requested_labels": []})
    b = environment.cache_key({"m": 1}, {**request, "requested_labels": ["forklift"]})
    c = environment.cache_key({"m": 1}, {**request, "requested_labels": []})
    assert a != b
    assert a == c


def test_no_cache_always_runs_and_writes_nothing(clip, backend, tmp_path):
    for out in ("a.json", "b.json"):
        assert _run(clip, tmp_path, "--no-cache", out=out) == 0

    assert len(backend.requests) == 2
    assert not (tmp_path / "cache").exists()


def test_a_backend_without_a_cache_identity_is_never_cached(clip, monkeypatch, tmp_path):
    plain = FakeBackend(devices=("cpu",))
    monkeypatch.setitem(environment.BACKENDS, "fake", plain)

    _run(clip, tmp_path)
    _run(clip, tmp_path, out="b.json")

    assert len(plain.requests) == 2
    assert not (tmp_path / "cache").exists()


def _entry(tmp_path):
    (entry,) = list((tmp_path / "cache").iterdir())
    return entry


def test_a_corrupt_cache_entry_is_ignored_and_replaced(clip, backend, tmp_path, capsys):
    _run(clip, tmp_path)
    (_entry(tmp_path) / "entry.json").write_text("{truncated")
    capsys.readouterr()

    assert _run(clip, tmp_path, out="b.json") == 0

    assert len(backend.requests) == 2
    assert "ignoring unreadable cache entry" in capsys.readouterr().out
    json.loads((_entry(tmp_path) / "entry.json").read_text())


def test_a_tampered_cached_asset_is_not_trusted(clip, backend, tmp_path, capsys):
    _run(clip, tmp_path)
    (_entry(tmp_path) / "assets" / "masks" / "floor-0.png").write_bytes(b"tampered")
    capsys.readouterr()

    assert _run(clip, tmp_path, out="b.json") == 0

    assert len(backend.requests) == 2
    assert "ignoring unreadable cache entry" in capsys.readouterr().out


def test_partial_and_failed_runs_are_not_cached(clip, monkeypatch, tmp_path):
    def partial(resp):
        resp["status"] = "partial"
        resp["reason"] = "one frame failed"

    fake = FakeBackend(identity=IDENTITY, mutate=partial)
    monkeypatch.setitem(environment.BACKENDS, "fake", fake)
    assert _run(clip, tmp_path) == environment.EXIT_PARTIAL

    broken = FakeBackend(identity=IDENTITY, raises=RuntimeError("boom"))
    monkeypatch.setitem(environment.BACKENDS, "fake", broken)
    with pytest.raises(SystemExit):
        _run(clip, tmp_path, out="b.json")

    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("damage", ["response", "mask"])
def test_invalid_cached_response_or_mask_recomputes_from_empty_staging(
    clip, backend, tmp_path, monkeypatch, damage
):
    import hashlib

    import cv2
    import numpy as np

    assert _run(clip, tmp_path) == 0
    entry_dir = _entry(tmp_path)
    path = entry_dir / "entry.json"
    entry = json.loads(path.read_text())
    if damage == "response":
        entry["response"]["status"] = "invalid-status"
    else:
        mask = entry_dir / "assets" / "masks" / "floor-0.png"
        cv2.imwrite(str(mask), np.zeros((2, 2), np.uint8))
        asset = next(a for a in entry["assets"] if a["path"] == "masks/floor-0.png")
        asset["sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    sentinel = entry_dir / "assets" / "stale.txt"
    sentinel.write_text("cached-only asset")
    entry["assets"].append(
        {"path": "stale.txt", "sha256": hashlib.sha256(sentinel.read_bytes()).hexdigest()}
    )
    path.write_text(json.dumps(entry))
    original_run = backend.run

    def assert_empty_then_run(request, assets):
        assert not any(assets.iterdir()), "invalid cache must not leak assets into fresh inference"
        return original_run(request, assets)

    monkeypatch.setattr(backend, "run", assert_empty_then_run)
    assert _run(clip, tmp_path, out="recovered.json") == 0
    assert len(backend.requests) == 2
    assert not (tmp_path / "recovered.assets" / "stale.txt").exists()
