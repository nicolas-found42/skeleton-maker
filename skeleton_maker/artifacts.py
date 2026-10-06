# SPDX-License-Identifier: MIT
"""Helpers shared by artifact-writing stages: hashing, frame time and atomic writes."""

import hashlib
import os
import shutil
import tempfile
import uuid
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


def sha256_file(path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrameClock:
    """Exact frame-index <-> time mapping for a constant rational frame rate.

    Time is kept as a Fraction of seconds so a 30000/1001 clip never drifts through
    float rounding; floats are only produced at the edge, for display.
    """

    def __init__(self, rate_num: int, rate_den: int = 1):
        if rate_num <= 0 or rate_den <= 0:
            raise ValueError(f"frame rate must be positive, got {rate_num}/{rate_den}")
        self.rate = Fraction(rate_num, rate_den)

    def time(self, frame_id: int) -> Fraction:
        """Presentation time of a source frame, in seconds."""
        return Fraction(frame_id) / self.rate

    def time_pair(self, frame_id: int) -> list[int]:
        """The time as an exact [numerator, denominator] pair."""
        t = self.time(frame_id)
        return [t.numerator, t.denominator]

    def frame_at(self, seconds: Fraction) -> int:
        """The source frame being shown at ``seconds`` (floor)."""
        return int((seconds * self.rate) // 1)

    def sample(self, frame_count: int, sample_fps: float) -> list[int]:
        """Source frame ids at about ``sample_fps``, always starting at frame 0.

        The sampling step is rounded to whole source frames, so the realised rate is the
        nearest one the clip can express; ids are strictly increasing and in range.
        """
        if frame_count <= 0:
            return []
        step = max(1, round(float(self.rate) / sample_fps))
        return list(range(0, frame_count, step))


def write_atomic(path, text: str) -> None:
    """Write ``text`` so readers see the old file or the whole new one, never a fragment."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def staging_dir(near) -> Generator[Path, None, None]:
    """A scratch directory beside ``near`` (same filesystem), removed on exit."""
    parent = Path(near).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def publish(manifest_path, manifest_text: str, assets_src, assets_dst) -> None:
    """Replace a manifest and its asset directory, restoring the old pair on any failure.

    Everything is prepared in a staging area first, so by the time an existing artifact is
    touched the new one is complete. The previous assets directory is parked and only
    deleted once the new manifest is in place.
    """
    manifest_path = Path(manifest_path)
    assets_dst = Path(assets_dst)
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    parked = assets_dst.with_name(f".{assets_dst.name}.{os.getpid()}.old")
    had_old = assets_dst.exists()
    parked_old = installed_new = False
    try:
        tmp_manifest.write_text(manifest_text, encoding="utf-8")
        if had_old:
            os.replace(assets_dst, parked)
            parked_old = True
        if assets_src is not None:
            os.replace(assets_src, assets_dst)
            installed_new = True
        os.replace(tmp_manifest, manifest_path)
    except BaseException:
        if installed_new:
            shutil.rmtree(assets_dst, ignore_errors=True)
        if parked_old:
            os.replace(parked, assets_dst)
        tmp_manifest.unlink(missing_ok=True)
        raise
    shutil.rmtree(parked, ignore_errors=True)


def publish_group(items: Iterable[tuple[Path | None, Path]]) -> list[str]:
    """Install prepared files/directories as a group and restore the old set on failure.

    Each staged source must be on the same filesystem as its destination. A ``None`` source
    removes an existing destination on success. All preparation and cross-filesystem copying
    must finish before this function is called, so installation only renames completed items.
    """
    entries = [(Path(src) if src is not None else None, Path(dst)) for src, dst in items]
    if not entries:
        return []
    destinations = [dst.absolute() for _, dst in entries]
    if len(set(destinations)) != len(destinations):
        raise ValueError("grouped artifact destinations must be unique")
    if any(
        a in b.parents or b in a.parents
        for i, a in enumerate(destinations)
        for b in destinations[i + 1 :]
    ):
        raise ValueError("grouped artifact destinations must not contain one another")
    for src, dst in entries:
        if src is not None and (
            src.absolute() == dst.absolute() or not (src.exists() or src.is_symlink())
        ):
            raise FileNotFoundError(f"staged artifact is missing or is its destination: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)

    states: list[_PublishState] = []
    try:
        for src, dst in entries:
            backup = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.old")
            state = _PublishState(destination=dst, backup=backup)
            states.append(state)
            if dst.exists() or dst.is_symlink():
                os.replace(dst, backup)
                state.parked = True
            if src is not None:
                os.replace(src, dst)
                state.installed = True
    except BaseException as exc:
        rollback_errors = []
        for state in reversed(states):
            try:
                if state.installed:
                    _remove_artifact(state.destination)
                if state.parked:
                    os.replace(state.backup, state.destination)
            except OSError as rollback_exc:
                rollback_errors.append(f"{state.destination}: {rollback_exc}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise OSError(f"artifact group failed and rollback was incomplete: {details}") from exc
        raise
    else:
        cleanup_warnings = []
        for state in states:
            if state.parked:
                try:
                    _remove_artifact(state.backup)
                except OSError as exc:
                    cleanup_warnings.append(
                        f"could not remove recovery backup {state.backup}: {exc}"
                    )
        return cleanup_warnings


@dataclass
class _PublishState:
    destination: Path
    backup: Path
    parked: bool = False
    installed: bool = False


def _remove_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
