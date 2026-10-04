# SPDX-License-Identifier: MIT
"""Run the offline character browser regressions against a generated fixture."""

from __future__ import annotations

import contextlib
import functools
import http.server
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from scripts.check_dev_environment import diagnose

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the temporary fixture with byte ranges required for video seeking."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self._serve_range(send_body=True):
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._serve_range(send_body=False):
            return
        super().do_HEAD()

    def _serve_range(self, *, send_body: bool) -> bool:
        requested = self.headers.get("Range")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested or "")
        if not match:
            return False

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return False
        size = path.stat().st_size
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        elif last:
            suffix = int(last)
            start, end = max(size - suffix, 0), size - 1
        else:
            return False
        if start >= size or end < start:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        if send_body:
            with path.open("rb") as source:
                source.seek(start)
                self.copyfileobj_limited(source, self.wfile, length)
        return True

    @staticmethod
    def copyfileobj_limited(source, destination, remaining: int) -> None:
        while remaining:
            block = source.read(min(64 * 1024, remaining))
            if not block:
                break
            destination.write(block)
            remaining -= len(block)


@contextlib.contextmanager
def serve_directory(directory: Path):
    """Serve a directory on an operating-system-assigned loopback port."""
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(("127.0.0.1", int(server.server_port)), timeout=5):
            pass
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603


def browser_child_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Copy the caller environment without passing the NIM credential to children."""
    env = os.environ.copy() if source is None else source.copy()
    env.pop("NVIDIA_API_KEY", None)
    return env


def run_browser_scripts(
    base_url: str,
    node: str,
    *,
    run: Callable[..., None] = _run,
    env: dict[str, str] | None = None,
) -> None:
    """Run both regression scripts; propagate failures so the server context closes."""
    for script in ("check_character_browser.cjs", "check_character_edges.cjs"):
        if env is None:
            run([node, str(ROOT / "scripts" / script), f"{base_url}/index.html"])
        else:
            run([node, str(ROOT / "scripts" / script), f"{base_url}/index.html"], env=env)


def main() -> int:
    issues = diagnose(ROOT)
    if issues:
        print("Browser checks cannot start:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    node = shutil.which("node")
    ffmpeg = shutil.which("ffmpeg")
    if node is None or ffmpeg is None:  # Tool availability can change after diagnosis.
        return 1
    child_env = browser_child_environment()
    with tempfile.TemporaryDirectory(prefix="skeleton-maker-browser-") as scratch:
        fixture_dir = Path(scratch)
        html = fixture_dir / "index.html"
        video = fixture_dir / "video.webm"
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x360:r=30:d=5",
                "-an",
                "-c:v",
                "libvpx",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-g",
                "30",
                "-b:v",
                "120k",
                "-y",
                str(video),
            ],
            env=child_env,
        )
        _run(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "scripts.make_character_fixture",
                str(html),
                "--video-name",
                video.name,
            ],
            env=child_env,
        )

        with serve_directory(fixture_dir) as origin:
            # The fixture generator's canonical output basename is index.html.
            run_browser_scripts(origin, node, env=child_env)

    print("PASS: both offline browser regressions completed; temporary fixture and server removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
