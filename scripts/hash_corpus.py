# SPDX-License-Identifier: MIT
"""Hash every file of an evaluation corpus so a later result can name exactly what it used.

    python -m scripts.hash_corpus CORPUS > hashes.json

Each file under ``clips/``, ``annotations/``, ``references/`` and ``build-reports/`` gets a
SHA-256. ``tree_sha256`` hashes the sorted ``path:sha256`` lines, so one value pins the corpus.
"""

import hashlib
import json
import sys
from pathlib import Path

PARTS = ("clips", "annotations", "references", "build-reports")


def hash_corpus(root: Path) -> dict:
    files = {}
    for part in PARTS:
        for path in sorted((root / part).rglob("*")):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    lines = "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items()))
    return {
        "files": files,
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(lines.encode()).hexdigest(),
    }


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not (Path(args[0]) / "clips").is_dir():
        print(
            "usage: python -m scripts.hash_corpus CORPUS (a directory with clips/)", file=sys.stderr
        )
        return 2
    print(json.dumps(hash_corpus(Path(args[0])), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
