# SPDX-License-Identifier: MIT
"""Case aliases on case-insensitive filesystems must not bypass public path guards."""

import shutil

import pytest

from skeleton_maker import environment

from .env_fakes import FakeBackend
from .test_environment import _make_video, _run

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)


def test_environment_handles_case_alias_bundle_containing_source_safely(
    monkeypatch, tmp_path, capsys
):
    assets = tmp_path / "result.assets"
    assets.mkdir()
    output_assets_alias = tmp_path / "Result.assets"
    aliases = output_assets_alias.exists() and output_assets_alias.samefile(assets)

    source = assets / "input.mp4"
    _make_video(source)
    before = source.read_bytes()
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    args = [
        str(source),
        "--backend",
        "fake",
        "--geometry",
        "off",
        "--out",
        str(tmp_path / "Result.json"),
        "--no-cache",
    ]

    if aliases:
        with pytest.raises(SystemExit) as exc:
            _run(args)
        assert exc.value.code == 2
        assert "would contain or replace the source video" in capsys.readouterr().err
        assert backend.requests == []
    else:
        assert _run(args) == 0
        assert len(backend.requests) == 1
        assert (tmp_path / "Result.json").is_file()

    assert source.read_bytes() == before


def test_environment_rejects_case_alias_between_future_outputs_before_inference(
    monkeypatch, tmp_path, capsys
):
    probe = tmp_path / "case-probe"
    probe.mkdir()
    alias = tmp_path / "CASE-PROBE"
    if not alias.exists() or not alias.samefile(probe):
        pytest.skip("temporary filesystem is case-sensitive")

    source = tmp_path / "clip.mp4"
    _make_video(source)
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    with pytest.raises(SystemExit) as exc:
        _run(
            [
                str(source),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--out",
                str(tmp_path / "Result.json"),
                "--overlay",
                str(tmp_path / "result.JSON"),
                "--no-cache",
            ]
        )

    assert exc.value.code == 2
    assert "conflicts with environment overlay" in capsys.readouterr().err
    assert backend.requests == []
    assert not (tmp_path / "Result.json").exists()


def test_environment_rejects_case_alias_with_multiple_future_parent_directories(
    monkeypatch, tmp_path, capsys
):
    probe = tmp_path / "case-probe"
    probe.mkdir()
    alias = tmp_path / "CASE-PROBE"
    if not alias.exists() or not alias.samefile(probe):
        pytest.skip("temporary filesystem is case-sensitive")

    source = tmp_path / "clip.mp4"
    _make_video(source)
    backend = FakeBackend()
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)

    with pytest.raises(SystemExit) as exc:
        _run(
            [
                str(source),
                "--backend",
                "fake",
                "--geometry",
                "off",
                "--out",
                str(tmp_path / "new-parent" / "also-new" / "Result.json"),
                "--overlay",
                str(tmp_path / "new-parent" / "also-new" / "result.JSON"),
                "--no-cache",
            ]
        )

    assert exc.value.code == 2
    assert "conflicts with environment overlay" in capsys.readouterr().err
    assert backend.requests == []
    assert not (tmp_path / "new-parent").exists()
