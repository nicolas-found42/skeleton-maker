# SPDX-License-Identifier: MIT
"""Tests for browser tooling diagnostics and temporary server cleanup."""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
from pathlib import Path

import pytest

from scripts import check_dev_environment, setup_dev
from scripts.run_browser_checks import (
    browser_child_environment,
    run_browser_scripts,
    serve_directory,
)


def _install_fake_playwright(root: Path, version: str = "1.63.0") -> None:
    (root / "node_modules" / "playwright").mkdir(parents=True)
    (root / "package.json").write_text(json.dumps({"devDependencies": {"playwright": "1.63.0"}}))
    (root / "node_modules" / "playwright" / "package.json").write_text(
        json.dumps({"version": version})
    )


def _all_tools_available(name: str, **_kwargs: object) -> str:
    return f"/usr/bin/{name}"


def test_doctor_reports_missing_tools_and_project_browser(tmp_path: Path) -> None:
    problems = check_dev_environment.diagnose(tmp_path, path="", which=lambda *_a, **_k: None)
    assert "missing just on PATH" in problems
    assert "missing uv on PATH" in problems
    assert "missing node on PATH" in problems
    assert "missing npm on PATH" in problems
    assert "missing ffmpeg on PATH" in problems
    assert any("package.json does not pin" in problem for problem in problems)


def test_doctor_reports_stale_node_install_without_launching(tmp_path: Path) -> None:
    _install_fake_playwright(tmp_path, "1.62.0")
    called = False

    def unexpected_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError("a version mismatch should fail before launching Chromium")

    problems = check_dev_environment.diagnose(
        tmp_path,
        path="/usr/bin",
        which=_all_tools_available,
        run=unexpected_run,
    )
    assert any("pins 1.63.0" in problem and "has 1.62.0" in problem for problem in problems)
    assert not called


def test_doctor_reports_missing_or_broken_chromium(tmp_path: Path) -> None:
    _install_fake_playwright(tmp_path)

    def failed_launch(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Executable doesn't exist"
        )

    problems = check_dev_environment.diagnose(
        tmp_path, path="/usr/bin", which=_all_tools_available, run=failed_launch
    )
    assert any(
        "Chromium is missing, mismatched, or cannot launch" in problem for problem in problems
    )
    assert any("just browser-install" in problem for problem in problems)


def test_browser_children_never_inherit_the_nim_key() -> None:
    assert browser_child_environment({"NVIDIA_API_KEY": "test-secret", "PATH": "/bin"}) == {
        "PATH": "/bin"
    }


def test_setup_bootstraps_just_and_pinned_browser_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(setup_dev.shutil, "which", lambda _name: None)
    monkeypatch.setattr(setup_dev, "run", commands.append)
    monkeypatch.setattr(setup_dev.sys, "platform", "linux")

    assert setup_dev.main() == 0
    assert commands == [
        ["uv", "tool", "install", "rust-just"],
        ["uvx", "prek", "install"],
        ["npm", "ci"],
        [
            "npm",
            "exec",
            "--",
            "playwright",
            "install",
            "--with-deps",
            "--only-shell",
            "chromium",
        ],
    ]


def test_browser_server_uses_ephemeral_port_and_stops_on_script_failure(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("fixture")
    observed_port = 0

    def fail_on_first_script(command: list[str], **_kwargs: object) -> None:
        assert command[-1].endswith("/index.html")
        raise subprocess.CalledProcessError(1, command)

    with serve_directory(tmp_path) as origin:
        observed_port = int(origin.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", observed_port)
        connection.request("GET", "/index.html")
        response = connection.getresponse()
        assert response.read() == b"fixture"
        connection.request("GET", "/index.html", headers={"Range": "bytes=1-3"})
        response = connection.getresponse()
        assert response.status == 206
        assert response.getheader("Content-Range") == "bytes 1-3/7"
        assert response.read() == b"ixt"
        connection.close()
        with pytest.raises(subprocess.CalledProcessError):
            run_browser_scripts(origin, "node", run=fail_on_first_script)

    assert observed_port > 0
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", observed_port)) != 0
