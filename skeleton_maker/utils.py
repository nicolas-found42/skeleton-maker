# SPDX-License-Identifier: MIT
"""Small helpers shared by the pipeline stages."""

import os
import shutil
import subprocess
import sys


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def require_tool(name: str, hint: str = "") -> str:
    """Return the path to an external tool or exit with an actionable message."""
    path = shutil.which(name)
    if not path:
        die(f"{name} not found on PATH. {hint}".strip())
    return path


def run_ffmpeg(args, desc: str) -> None:
    """Run ffmpeg with the standard quiet flags and report failures usefully."""
    ffmpeg = require_tool("ffmpeg", "Install it from https://ffmpeg.org/download.html")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "no output"
        die(f"ffmpeg failed while {desc}: {tail}")


def probe_video(path: str) -> dict:
    """Return the video stream's geometry, codec and frame rate."""
    ffprobe = require_tool("ffprobe", "It ships with ffmpeg.")
    out = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout

    def frac(value: str) -> float:
        if "/" in value:
            num, _, den = value.partition("/")
            try:
                return float(num) / float(den) if float(den) else 0.0
            except ValueError:
                return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0

    info = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        info[key] = value
    frames = info.get("nb_frames", "N/A")
    return {
        "width": int(info.get("width", 0) or 0),
        "height": int(info.get("height", 0) or 0),
        "codec": info.get("codec_name", "?"),
        "pix_fmt": info.get("pix_fmt", "?"),
        "r_fps": frac(info.get("r_frame_rate", "0")),
        "avg_fps": frac(info.get("avg_frame_rate", "0")),
        "nb_frames": int(frames) if frames.isdigit() else None,
        "duration": float(info.get("duration", 0) or 0),
        "size": os.path.getsize(path),
    }


def is_streamable_mp4(path: str) -> bool:
    """True when the MP4's moov atom precedes mdat, so the NIM streams it."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    if len(head) < 16 or head[4:8] != b"ftyp":
        return False
    ftyp_size = int.from_bytes(head[0:4], "big")
    return head[ftyp_size + 4: ftyp_size + 8] == b"moov"
