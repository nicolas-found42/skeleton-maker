# SPDX-License-Identifier: MIT
"""Filesystem-aware comparisons for input protection and output collisions."""

import os
from pathlib import Path


def same_path(first: str | os.PathLike, second: str | os.PathLike) -> bool:
    """Return whether two paths name the same entry, including existing filesystem aliases."""
    left = Path(first).resolve()
    right = Path(second).resolve()
    return _same_components(left, right)


def is_same_or_ancestor(parent: str | os.PathLike, child: str | os.PathLike) -> bool:
    """Return whether ``parent`` is ``child`` or one of its filesystem-aware ancestors."""
    parent_path = Path(parent).resolve()
    child_path = Path(child).resolve()
    if len(parent_path.parts) > len(child_path.parts):
        return False
    return _same_components(parent_path, Path(*child_path.parts[: len(parent_path.parts)]))


def paths_overlap(first: str | os.PathLike, second: str | os.PathLike) -> bool:
    """Return whether either path is equal to or an ancestor of the other."""
    return is_same_or_ancestor(first, second) or is_same_or_ancestor(second, first)


def _same_components(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        if os.path.samefile(first, second):
            return True
    except OSError:
        pass
    if len(first.parts) != len(second.parts):
        return False
    for index, (left_part, right_part) in enumerate(zip(first.parts, second.parts, strict=True)):
        if left_part == right_part:
            continue
        left_prefix = Path(*first.parts[: index + 1])
        right_prefix = Path(*second.parts[: index + 1])
        try:
            if os.path.samefile(left_prefix, right_prefix):
                continue
        except OSError:
            pass
        if left_part.casefold() == right_part.casefold() and _case_insensitive_directory(
            left_prefix.parent
        ):
            continue
        return False
    return True


def _case_insensitive_directory(path: Path) -> bool:
    """Detect volume case behavior by checking alternate spellings of existing directories."""
    try:
        directory = path.resolve()
        while not directory.is_dir():
            if directory == directory.parent:
                return False
            directory = directory.parent
        device = directory.stat().st_dev
    except OSError:
        return False

    while directory != directory.parent:
        name = directory.name
        for index, character in enumerate(name):
            if "a" <= character <= "z":
                alternate_character = character.upper()
            elif "A" <= character <= "Z":
                alternate_character = character.lower()
            else:
                continue
            alternate = directory.with_name(
                f"{name[:index]}{alternate_character}{name[index + 1 :]}"
            )
            try:
                if (
                    alternate.exists()
                    and not alternate.is_symlink()
                    and os.path.samefile(directory, alternate)
                ):
                    return True
            except OSError:
                pass
            break
        try:
            if directory.parent.stat().st_dev != device:
                break
        except OSError:
            break
        directory = directory.parent
    return False
