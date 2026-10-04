# SPDX-License-Identifier: MIT
"""Helpers shared by artifact-writing stages: hashing, frame time and atomic writes."""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
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
