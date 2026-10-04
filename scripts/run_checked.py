#!/usr/bin/env python3
"""Run dependent subprocesses in order and publish artifacts only on success.

Programmatic callers should use :func:`run_checked`. The CLI reads a small JSON
plan, making it suitable for repository orchestration without shell chaining.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Action:
    """One named command in a dependent sequence."""

    name: str
    argv: Sequence[str]


class CheckedCommandError(RuntimeError):
    """A subprocess failed; no staged artifacts were published."""

    def __init__(self, action: str, returncode: int) -> None:
        self.action = action
        self.returncode = returncode
        super().__init__(
            f"action {action!r} exited with status {returncode}; later actions were not run"
        )


def run_checked(
    actions: Sequence[Action],
    *,
    artifact_dir: str | Path,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Run every action in order and atomically publish captured logs.

    The destination must not already exist. Each process's output is staged in
    a temporary sibling directory; on any error the stage is removed, the
    failing process output is replayed to this process's stderr/stdout, and a
    :class:`CheckedCommandError` is raised. Later commands never run.
    """
    if not actions:
        raise ValueError("at least one action is required")
    for action in actions:
        if not isinstance(action, Action):
            raise ValueError("each action must be an Action instance")
        if (
            not isinstance(action.name, str)
            or not action.name.strip()
            or not isinstance(action.argv, Sequence)
            or isinstance(action.argv, (str, bytes))
            or len(action.argv) == 0
            or any(not isinstance(arg, str) for arg in action.argv)
        ):
            raise ValueError("each action needs a name and a non-empty string argv")
    destination = Path(artifact_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"artifact destination already exists: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for index, action in enumerate(actions, start=1):
            completed = subprocess.run(  # noqa: S603 -- explicit argv execution is the runner's purpose
                list(action.argv),
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            (stage / f"{index:02d}-{_safe_name(action.name)}.stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (stage / f"{index:02d}-{_safe_name(action.name)}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                if completed.stdout:
                    sys.stdout.write(completed.stdout)
                if completed.stderr:
                    sys.stderr.write(completed.stderr)
                raise CheckedCommandError(action.name, completed.returncode)
        manifest = {
            "actions": [
                {
                    "name": action.name,
                    "stdout_log": f"{index:02d}-{_safe_name(action.name)}.stdout.txt",
                    "stderr_log": f"{index:02d}-{_safe_name(action.name)}.stderr.txt",
                }
                for index, action in enumerate(actions, start=1)
            ],
            "count": len(actions),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    return safe.strip("_")[:80] or "action"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, required=True, help="JSON object with an actions array of {name, argv}"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="new directory for success-only logs and manifest",
    )
    parser.add_argument(
        "--cwd", type=Path, help="working directory for every action (default: current directory)"
    )
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        raw_actions = plan.get("actions") if isinstance(plan, dict) else None
        if not isinstance(raw_actions, list):
            raise ValueError("plan must be an object with an actions list")
        actions = []
        for index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                raise ValueError(f"actions[{index}] must be an object")
            name = item.get("name")
            command = item.get("argv")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"actions[{index}].name must be a non-empty string")
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(argument, str) for argument in command)
            ):
                raise ValueError(f"actions[{index}].argv must be a non-empty list of strings")
            actions.append(Action(name, command))
        published = run_checked(actions, artifact_dir=args.artifacts, cwd=args.cwd)
    except CheckedCommandError as exc:
        print(str(exc), file=sys.stderr)
        return exc.returncode if 0 < exc.returncode < 126 else 1
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"run_checked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "success", "artifacts": str(published)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
