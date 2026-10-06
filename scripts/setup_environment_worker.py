# SPDX-License-Identifier: MIT
"""Create the Grounded SAM 2 environment worker: virtualenv, pinned libraries, verified models.

Everything lands outside the repository (default ``~/.cache/skeleton-maker``). Re-running skips
what is already in place and re-verifies every model file against its pinned SHA-256.
Downloads: the pinned Python libraries (PyPI), the SAM 2.1 small checkpoint (about 185 MB,
dl.fbaipublicfiles.com) and Grounding DINO tiny at a pinned revision (about 660 MB,
huggingface.co).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from skeleton_maker.workers import grounded_sam2_da3 as worker

REQUIREMENTS = Path(worker.__file__).with_name("grounded_sam2_da3.requirements.txt")
PYTHON_VERSION = "3.12"
NAME = "grounded-sam2-da3"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    """Exit with a pointed message unless ``path`` has the pinned SHA-256."""
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
        subprocess.run(command, check=True)  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--home", help="worker home (default: ~/.cache/skeleton-maker/workers)")
    parser.add_argument(
        "--models", help="model directory (default: ~/.cache/skeleton-maker/models)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the steps without running them"
    )
    args = parser.parse_args(argv)
    cache = Path(os.path.expanduser("~")) / ".cache" / "skeleton-maker"
    home = Path(args.home or os.environ.get("SKELETON_MAKER_WORKER_HOME") or cache / "workers")
    models = Path(args.models or os.environ.get("SKELETON_MAKER_MODELS") or cache / "models")
    venv = home / NAME / ".venv"
    python = venv / "bin" / "python"

    if not python.exists():
        run(["uv", "venv", "--python", PYTHON_VERSION, str(venv)], args.dry_run)
    run(["uv", "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)], args.dry_run)

    sam = worker.MODELS["sam2"]
    sam_file = models / sam["directory"] / sam["weights"]
    if not sam_file.exists():
        print(f"+ download {sam['source']} -> {sam_file}", flush=True)
        if not args.dry_run:
            sam_file.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(sam["source"], sam_file)  # noqa: S310  pinned https URL
    dino = worker.MODELS["grounding-dino"]
    dino_dir = models / dino["directory"]
    if not (dino_dir / dino["weights"]).exists():
        snippet = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download('IDEA-Research/grounding-dino-tiny', revision='{dino['revision']}',"
            f" allow_patterns=['*.json','*.txt','{dino['weights']}'], local_dir={str(dino_dir)!r})"
        )
        run([str(python), "-c", snippet], args.dry_run)

    if args.dry_run:
        print("dry run: nothing was changed")
        return 0
    verify(sam_file, sam["sha256"])
    verify(dino_dir / dino["weights"], dino["sha256"])
    env = {**os.environ, "SKELETON_MAKER_MODELS": str(models)}
    out = subprocess.run(  # noqa: S603  fixed argument list
        [str(python), worker.__file__, "preflight"], capture_output=True, text=True, env=env
    )
    print(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr)
    return 0 if '"ok": true' in out.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
