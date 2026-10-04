# SPDX-License-Identifier: MIT
"""The one reader for ``pose.json`` (JSON Lines), and the checks that tie it to a clip.

The renderer, the stage builder, ``verify`` and ``environment`` all read the same file; they
share this parser so a malformed line is reported the same way everywhere.
"""

import json
import math

from .constants import BBOX_MARGIN


class PoseFileError(ValueError):
    """The pose file is unreadable or does not belong to the clip it was paired with."""


def read_records(path: str) -> list[dict]:
    """Every record of the file, in file order, duplicates preserved."""
    records = []
    with open(path) as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PoseFileError(
                    f"pose file {path}, line {number}: not valid JSON ({exc})"
                ) from exc
            if (
                not isinstance(record, dict)
                or isinstance(record.get("frame_id"), bool)
                or not isinstance(record.get("frame_id"), int)
                or not isinstance(record.get("detections"), list)
            ):
                raise PoseFileError(
                    f"pose file {path}, line {number}: expected an object with an integer "
                    "'frame_id' and a 'detections' list"
                )
            records.append(record)
    return records


def load_poses(path: str) -> dict:
    """Read the pose file into ``{frame_id: [detection, ...]}`` (the last record wins)."""
    return {r["frame_id"]: r["detections"] for r in read_records(path)}


def check_against_clip(
    records: list[dict], *, frame_count: int, width: int, height: int, clip_sha256: str
) -> str:
    """Raise :class:`PoseFileError` unless the poses plausibly belong to the clip.

    The NIM emits one record per decoded frame, so the ids must be exactly ``0..N-1``.
    Returns the association: ``hash-verified`` when every record carries the clip's
    fingerprint, else ``user-supplied`` (matching length is not identity).
    """
    if len(records) != frame_count:
        raise PoseFileError(
            f"{len(records)} pose records but the clip has {frame_count} frames; "
            "the poses were computed for a different clip or frame range"
        )
    ids = [r["frame_id"] for r in records]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise PoseFileError(f"duplicate frame id {duplicates[0]} in the pose file")
    if set(ids) != set(range(frame_count)):
        raise PoseFileError(
            f"frame ids must be 0..{frame_count - 1} to match the clip, "
            f"found {min(ids)}..{max(ids)}"
        )
    _check_bounds(records, width, height)
    return _association(records, clip_sha256)


def _check_bounds(records: list[dict], width: int, height: int) -> None:
    mx, my = BBOX_MARGIN * width, BBOX_MARGIN * height
    for record in sorted(records, key=lambda r: r["frame_id"]):
        for det in record["detections"]:
            tid = det.get("tracking_id") if isinstance(det, dict) else None
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise PoseFileError(
                    f"frame {record['frame_id']}: a detection has no integer tracking_id"
                )
            box = det.get("bbox")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            ):
                raise PoseFileError(
                    f"frame {record['frame_id']}: a detection has no usable [x, y, w, h] bbox"
                )
            x, y, w, h = box
            if w < 0 or h < 0 or x < -mx or y < -my or x + w > width + mx or y + h > height + my:
                raise PoseFileError(
                    f"frame {record['frame_id']}: bbox {box} is outside the {width}x{height} "
                    f"clip (allowed margin {BBOX_MARGIN:.0%}); these poses do not fit this video"
                )


def _association(records: list[dict], clip_sha256: str) -> str:
    seen = {r.get("source_sha256") for r in records}
    if seen == {None}:
        return "user-supplied"
    if len(seen) != 1 or None in seen:
        raise PoseFileError("source_sha256 is present on some pose records but not all or differs")
    (claimed,) = seen
    if claimed != clip_sha256:
        raise PoseFileError(
            f"the pose file was computed for a different clip (its source_sha256 {claimed} "
            f"does not match this clip's {clip_sha256})"
        )
    return "hash-verified"


def summarize(records: list[dict]) -> dict:
    """Skeleton ids with their extent, and the frame ranges where nobody is present."""
    spans: dict[int, list[int]] = {}
    present = set()
    for record in records:
        for det in record["detections"]:
            tid = det["tracking_id"]
            span = spans.setdefault(tid, [record["frame_id"], record["frame_id"], 0])
            span[0] = min(span[0], record["frame_id"])
            span[1] = max(span[1], record["frame_id"])
            span[2] += 1
        if record["detections"]:
            present.add(record["frame_id"])
    ids = sorted(r["frame_id"] for r in records)
    free: list[list[int]] = []
    for frame in ids:
        if frame in present:
            continue
        if free and free[-1][1] == frame - 1:
            free[-1][1] = frame
        else:
            free.append([frame, frame])
    skeletons = [
        {"id": tid, "first_frame": a, "last_frame": b, "frames": n}
        for tid, (a, b, n) in sorted(spans.items())
    ]
    return {"skeletons": skeletons, "person_free_ranges": free}
