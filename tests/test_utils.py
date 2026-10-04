# SPDX-License-Identifier: MIT
import pytest

from skeleton_maker import utils


def test_die_prints_an_error_and_exits_with_the_code(capsys):
    with pytest.raises(SystemExit) as exc:
        utils.die("boom", code=3)
    assert exc.value.code == 3
    assert "error: boom" in capsys.readouterr().err


def test_require_tool_returns_the_resolved_path(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert utils.require_tool("ffmpeg") == "/usr/bin/ffmpeg"


def test_require_tool_exits_with_the_hint_when_missing(monkeypatch, capsys):
    monkeypatch.setattr(utils.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        utils.require_tool("yt-dlp", "Install it: uv pip install yt-dlp")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "yt-dlp not found on PATH" in err
    assert "Install it: uv pip install yt-dlp" in err
