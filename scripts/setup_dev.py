# SPDX-License-Identifier: MIT
"""Bootstrap the project's local Python, task-runner, Node, and browser tools."""

from __future__ import annotations

import shutil
import subprocess
import sys


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)  # noqa: S603


def main() -> int:
    # This command runs via `uv run`, so uv is present. Install just only on a
    # fresh workstation; preserve an existing user-selected tool version.
    if shutil.which("just") is None:
        run(["uv", "tool", "install", "rust-just"])

    run(["uvx", "prek", "install"])
    run(["npm", "ci"])
    if sys.platform.startswith("linux"):
        run(
            [
                "npm",
                "exec",
                "--",
                "playwright",
                "install",
                "--with-deps",
                "--only-shell",
                "chromium",
            ]
        )
    else:
        run(["npm", "exec", "--", "playwright", "install", "chromium"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
