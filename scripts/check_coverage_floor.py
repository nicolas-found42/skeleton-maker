#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail if the coverage floor (fail_under) in HEAD_PYPROJECT is lower than in BASE_PYPROJECT.

The floor is a ratchet: it may only go up. CI runs this on every pull request against the
pyproject.toml of the base commit. Usage: check_coverage_floor.py BASE_PYPROJECT HEAD_PYPROJECT
"""

import re
import sys

PATTERN = re.compile(r"^fail_under\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$", re.MULTILINE)


def read_floor(path: str) -> float | None:
    """Return the fail_under value in a pyproject.toml, or None when it is not set."""
    with open(path, encoding="utf-8") as handle:
        match = PATTERN.search(handle.read())
    return float(match.group(1)) if match else None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    base, head = read_floor(argv[1]), read_floor(argv[2])
    if base is None:
        print("base has no coverage floor; nothing to compare")
        return 0
    if head is None:
        print(f"error: the coverage floor was removed (base had {base:g})", file=sys.stderr)
        return 1
    if head < base:
        print(f"error: the coverage floor was lowered from {base:g} to {head:g}", file=sys.stderr)
        print("It may only go up. Add tests instead of lowering it.", file=sys.stderr)
        return 1
    print(f"coverage floor ok: {base:g} -> {head:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
