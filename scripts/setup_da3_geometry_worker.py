# SPDX-License-Identifier: MIT
"""Install the pinned DA3-Small geometry worker outside the base environment."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from skeleton_maker.workers import da3_geometry as worker

REQUIREMENTS = Path(worker.__file__).with_name("da3_geometry.requirements.txt")
PYTHON_VERSION = "3.12"
NAME = "da3-geometry"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"error: {path} is missing")
    actual = sha256_of(path)
    if actual != expected:
        raise SystemExit(
            f"error: {path} has sha256 {actual}, pinned {expected}; delete it and retry"
        )


def run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)  # noqa: S603  fixed argument list, no shell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="worker home (default: ~/.cache/skeleton-maker/workers)")
    parser.add_argument(
        "--models", help="model/code cache (default: ~/.cache/skeleton-maker/models)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print steps without running them")
    args = parser.parse_args(argv)
    cache = Path.home() / ".cache" / "skeleton-maker"
    home = Path(args.home or os.environ.get("SKELETON_MAKER_WORKER_HOME") or cache / "workers")
    models = Path(args.models or os.environ.get("SKELETON_MAKER_MODELS") or cache / "models")
    venv = home / NAME / ".venv"
    python = venv / "bin" / "python"
    source = models / "da3-source"
    marker = models / "da3-code-commit.txt"
    checkpoint_dir = models / "da3-small"
    checkpoint = checkpoint_dir / "model.safetensors"

    if not python.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, str(venv)], args.dry_run)
    run(["uv", "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)], args.dry_run)
    if not (source / ".git").exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
                str(source),
            ],
            args.dry_run,
        )
    run(["git", "-C", str(source), "fetch", "origin", worker.CODE_REVISION], args.dry_run)
    run(["git", "-C", str(source), "checkout", "--detach", worker.CODE_REVISION], args.dry_run)
    if not args.dry_run:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(worker.CODE_REVISION + "\n", encoding="utf-8")
    run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(source)],
        args.dry_run,
    )
    if not checkpoint.exists():
        snippet = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download({worker.MODEL_REPOSITORY!r}, "
            f"revision={worker.MODEL_REVISION!r}, "
            f"allow_patterns=['config.json','model.safetensors'], local_dir={str(checkpoint_dir)!r})"
        )
        run([str(python), "-c", snippet], args.dry_run)
    if args.dry_run:
        print("dry run: nothing was changed")
        return 0

    verify(checkpoint, worker.MODEL_SHA256)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise SystemExit("error: git is required to verify the DA3 source revision")
    current = subprocess.run(  # noqa: S603  fixed executable and arguments, no shell
        [git_executable, "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != worker.CODE_REVISION:
        raise SystemExit(f"error: DA3 source is at {current}, expected {worker.CODE_REVISION}")
    env = {**os.environ, "SKELETON_MAKER_MODELS": str(models)}
    result = subprocess.run(  # noqa: S603  fixed interpreter and script paths
        [str(python), worker.__file__, "preflight"],
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr)
    return 0 if '"ok": true' in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
