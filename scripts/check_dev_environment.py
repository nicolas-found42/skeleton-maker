# SPDX-License-Identifier: MIT
"""Diagnose tools and the repository-pinned Playwright Chromium installation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def diagnose(
    root: Path = ROOT,
    *,
    path: str | None = None,
    which: Callable[..., str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    """Return concise setup problems; an empty list means the browser can launch."""
    issues: list[str] = []
    path = os.environ.get("PATH", "") if path is None else path
    tools = {name: which(name, path=path) for name in ("just", "uv", "node", "npm", "ffmpeg")}
    for name, location in tools.items():
        if location is None:
            issues.append(f"missing {name} on PATH")

    package_file = root / "package.json"
    module_file = root / "node_modules" / "playwright" / "package.json"
    expected_version = None
    try:
        expected_version = json.loads(package_file.read_text())["devDependencies"]["playwright"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        issues.append("package.json does not pin the Playwright development dependency")

    installed_version = None
    try:
        installed_version = json.loads(module_file.read_text())["version"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        issues.append("project Playwright dependency is missing; run `just browser-install`")

    if expected_version is not None and installed_version is not None:
        if expected_version != installed_version:
            issues.append(
                f"Playwright version mismatch: package.json pins {expected_version}, "
                f"node_modules has {installed_version}; run `just browser-install`"
            )
        elif tools["node"]:
            probe = r"""
const { chromium } = require('./node_modules/playwright');
(async () => {
  const browser = await chromium.launch({headless: true});
  await browser.close();
})().catch((error) => { console.error(error.message); process.exitCode = 1; });
"""
            try:
                probe_env = {**os.environ, "PATH": path}
                probe_env.pop("NVIDIA_API_KEY", None)
                result = run(
                    [tools["node"], "-e", probe],
                    cwd=root,
                    env=probe_env,
                    capture_output=True,
                    text=True,
                    timeout=45,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                issues.append(f"project Playwright Chromium could not be checked: {error}")
            else:
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip().splitlines()
                    summary = detail[-1] if detail else "browser launch failed"
                    issues.append(
                        "project Playwright Chromium is missing, mismatched, or cannot launch; "
                        f"run `just browser-install` ({summary})"
                    )

    return issues


def main() -> int:
    issues = diagnose()
    if issues:
        print("Browser development environment needs attention:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Browser development environment ready (project Playwright Chromium launches).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
