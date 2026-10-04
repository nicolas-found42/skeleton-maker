# SPDX-License-Identifier: MIT
"""The subprocess worker adapter, exercised through the public CLI against a stand-in worker."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skeleton_maker import cli, environment, envworkers

from .test_environment import _make_video, _run_failing, _snapshot

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)

WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_video(path, frames=30)
    return path


@pytest.fixture
def worker(monkeypatch):
    backend = envworkers.SubprocessBackend("stand-in", sys.executable, WORKER)
    monkeypatch.setitem(environment.BACKENDS, "stand-in", backend)
    return backend


def _argv(clip, tmp_path, *extra):
    return [
        "environment",
        str(clip),
        "--out",
        str(tmp_path / "environment.json"),
        "--backend",
        "stand-in",
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


def test_a_worker_run_produces_a_manifest_and_streams_progress(clip, worker, tmp_path, capsys):
    assert cli.main(_argv(clip, tmp_path)) == 0

    manifest = json.loads((tmp_path / "environment.json").read_text())
    assert manifest["run"]["status"] == "complete"
    assert manifest["backend"]["name"] == "fake"
    captured = capsys.readouterr()
    assert "progress: 2/2" in captured.err
    assert "on mps" in captured.out, "auto picks the worker's first reported device"


def test_a_partial_worker_result_is_marked_partial_and_exits_3(
    clip, worker, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("FAKE_WORKER_MODE", "partial")

    rc = cli.main(_argv(clip, tmp_path))

    assert rc == environment.EXIT_PARTIAL
    run = json.loads((tmp_path / "environment.json").read_text())["run"]
    assert run["status"] == "partial"
    assert "[15]" in run["reason"]
    assert not (tmp_path / "cache").exists(), "partial results are never cached"


def test_the_worker_identity_feeds_the_cache(clip, worker, tmp_path, capsys):
    cli.main(_argv(clip, tmp_path))
    capsys.readouterr()

    cli.main(_argv(clip, tmp_path))

    assert "cache hit" in capsys.readouterr().out


def test_a_missing_dependency_fails_before_any_inference(
    clip, worker, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("FAKE_WORKER_MODE", "missing_dep")
    before = _snapshot(tmp_path)

    err = _run_failing(_argv(clip, tmp_path)[1:], capsys)

    assert "torch>=2.5" in err
    assert "not installed in the worker environment" in err
    assert _snapshot(tmp_path) == before


def test_a_frame_limit_from_the_worker_is_enforced_before_inference(
    clip, worker, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("FAKE_WORKER_MAX_FRAMES", "5")
    before = _snapshot(tmp_path)

    err = _run_failing(_argv(clip, tmp_path, "--sample-fps", "10")[1:], capsys)

    assert "10 frames would be scanned; the limit for backend 'stand-in' is 5" in err
    assert _snapshot(tmp_path) == before


def test_an_unsupported_device_is_a_capability_error_naming_what_exists(
    clip, worker, tmp_path, capsys
):
    err = _run_failing(_argv(clip, tmp_path, "--device", "cuda")[1:], capsys)

    assert "device 'cuda' is not available" in err
    assert "mps, cpu" in err


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("crash", "exited with status 3"),
        ("crash", "simulated worker crash"),
        ("malformed", "malformed"),
        ("no_response", "wrote no response"),
        ("bad_contract", "contract"),
    ],
)
def test_worker_failures_never_report_success_and_preserve_prior_output(
    clip, worker, monkeypatch, tmp_path, capsys, mode, message
):
    monkeypatch.setenv("FAKE_WORKER_MODE", mode)
    out = tmp_path / "environment.json"
    out.write_text('{"previous": "result"}')
    before = _snapshot(tmp_path)

    err = _run_failing(_argv(clip, tmp_path)[1:], capsys)

    assert message in err
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / "cache").exists()


class _InterruptAfterFirstLine:
    """Stands in for the worker's stderr: one progress line, then the user presses Ctrl-C."""

    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        yield next(iter(self.stream))
        raise KeyboardInterrupt


def test_an_interrupt_stops_the_worker_and_writes_nothing(clip, worker, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_WORKER_MODE", "hang")
    started = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if args[0][2] == "run":  # the worker, not the ffprobe calls
            proc.stderr = _InterruptAfterFirstLine(proc.stderr)  # ty: ignore[invalid-assignment]
            started.append(proc)
        return proc

    monkeypatch.setattr(envworkers.subprocess, "Popen", spy)
    out = tmp_path / "environment.json"

    rc = cli.main(_argv(clip, tmp_path))

    assert rc == environment.EXIT_INTERRUPTED
    assert not out.exists()
    assert started[0].poll() is not None, "the worker process must not outlive an interrupt"
    assert not (tmp_path / "cache").exists()


def test_discovery_finds_an_installed_worker_and_ignores_a_missing_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SKELETON_MAKER_WORKER_HOME", str(tmp_path))
    assert envworkers.discover("grounded-sam2-da3") is None

    python = tmp_path / "grounded-sam2-da3" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)

    found = envworkers.discover("grounded-sam2-da3")

    assert found is not None
    assert found.name == "grounded-sam2-da3"
    assert envworkers.discover("something-else") is None
