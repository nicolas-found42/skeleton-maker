#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail if the coverage floor ([tool.coverage.report] fail_under) is lower in HEAD than in BASE.

The floor is a ratchet: it may only go up. CI runs this on every pull request against the
pyproject.toml of the base commit. Usage: check_coverage_floor.py BASE_PYPROJECT HEAD_PYPROJECT
"""

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def read_floor(path: str) -> float | None:
    """Return [tool.coverage.report] fail_under from a pyproject.toml, or None when it is not set."""
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    value = data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"fail_under must be a number, got {value!r}")
    return float(value)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        base, head = read_floor(argv[1]), read_floor(argv[2])
    except (OSError, ValueError, tomllib.TOMLDecodeError) as err:
        print(f"error: cannot read the coverage floor: {err}", file=sys.stderr)
        return 1
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
