# SPDX-License-Identifier: MIT
import subprocess
from types import SimpleNamespace

from skeleton_maker import cli


def test_download_runs_the_resolved_yt_dlp_with_the_resolved_ffmpeg(monkeypatch, tmp_path):
    paths = {"yt-dlp": "/opt/bin/yt-dlp", "ffmpeg": "/opt/bin/ffmpeg"}
    monkeypatch.setattr(cli, "require_tool", lambda name, hint="": paths[name])
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = SimpleNamespace(
        start=5,
        duration=10,
        out=str(tmp_path / "clip.mp4"),
        url="https://example.com/v",
        format="b",
    )

    assert cli.cmd_download(args) == 0

    cmd = seen["cmd"]
    assert cmd[0] == "/opt/bin/yt-dlp"
    assert cmd[cmd.index("--ffmpeg-location") + 1] == "/opt/bin/ffmpeg"
    assert cmd[cmd.index("--download-sections") + 1] == "*5-15"
    assert cmd[-1] == "https://example.com/v"
