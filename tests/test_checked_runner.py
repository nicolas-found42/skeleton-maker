from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from scripts.run_checked import Action, CheckedCommandError, main, run_checked


def test_failure_stops_sequence_and_publishes_no_artifact_directory(tmp_path: Path) -> None:
    marker = tmp_path / "second-action-ran"
    destination = tmp_path / "published"
    actions = [
        Action("fail", [sys.executable, "-c", "print('first ran'); raise SystemExit(7)"]),
        Action("must-not-run", [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ran')"]),
    ]

    with pytest.raises(CheckedCommandError, match="later actions were not run") as error:
        run_checked(actions, artifact_dir=destination, cwd=tmp_path)

    assert error.value.returncode == 7
    assert not marker.exists()
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*"))


def test_success_publishes_complete_logs_and_manifest_once(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    actions = [
        Action("prepare", [sys.executable, "-c", "print('prepared')"]),
        Action("consume", [sys.executable, "-c", "print('consumed')"]),
    ]

    result = run_checked(actions, artifact_dir=destination, cwd=tmp_path)

    assert result == destination
    assert (destination / "01-prepare.stdout.txt").read_text() == "prepared\n"
    assert (destination / "02-consume.stdout.txt").read_text() == "consumed\n"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [item["name"] for item in manifest["actions"]] == ["prepare", "consume"]
    assert all("argv" not in item for item in manifest["actions"])
    with pytest.raises(FileExistsError):
        run_checked(actions, artifact_dir=destination)


@pytest.mark.parametrize(
    "actions",
    [
        [{"name": "split", "argv": "echo unsafe"}],
        [{"name": 7, "argv": [sys.executable, "-c", "pass"]}],
        ["not-an-action-object"],
    ],
)
def test_cli_rejects_malformed_action_entries_without_running_or_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actions: list[object]
) -> None:
    plan = tmp_path / "plan.json"
    destination = tmp_path / "published"
    plan.write_text(json.dumps({"actions": actions}), encoding="utf-8")

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("malformed plan must be rejected before any process starts")

    monkeypatch.setattr("scripts.run_checked.subprocess.run", unexpected_run)

    assert main(["--plan", str(plan), "--artifacts", str(destination)]) == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*"))


@pytest.mark.parametrize(
    "action",
    [Action("split", "echo unsafe"), Action(cast(str, 7), [sys.executable, "-c", "pass"])],
)
def test_python_api_rejects_string_argv_and_nonstring_names_before_staging(
    tmp_path: Path, action: Action
) -> None:
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="each action needs"):
        run_checked([action], artifact_dir=destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*"))
