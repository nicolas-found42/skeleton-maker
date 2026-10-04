# SPDX-License-Identifier: MIT
"""Turn tracker output into the NIM's tracked-bounding-box annotation.

Format::

    <number_of_tracked_bodies>              # 1..50, client-enforced
    <frame_id> <tracking_id> <x> <y> <w> <h>

Camera cuts restart a tracker's ids, so a 20-second clip can rack up far more
ids than there are people on screen. The header must stay within 1..50, so the
ids that appear in the fewest frames are dropped: they are both the least
useful and the likeliest to be tracker noise.
"""

import collections
import csv

from .constants import ABSENT_BBOX, MAX_BODIES


def filter_frames(frames: list[dict], max_bodies: int = MAX_BODIES) -> list[dict]:
    """Keep the ``max_bodies`` most persistent tracking ids across all frames."""
    if max_bodies < 1 or max_bodies > MAX_BODIES:
        raise ValueError(f"max_bodies must be between 1 and {MAX_BODIES}")
    persistence = collections.Counter(
        det["tracking_id"] for fr in frames for det in fr["detections"]
    )
    if len(persistence) <= max_bodies:
        return frames
    keep = {tid for tid, _ in persistence.most_common(max_bodies)}
    return [
        {**fr, "detections": [d for d in fr["detections"] if d["tracking_id"] in keep]}
        for fr in frames
    ]


def write_annotation(frames: list[dict], path: str) -> dict:
    """Write the annotation file and return summary statistics.

    Raises ``ValueError`` on anything the NIM would reject, so a bad annotation
    fails here rather than after the video has been streamed.
    """
    ids = {d["tracking_id"] for fr in frames for d in fr["detections"]}
    if not ids:
        raise ValueError("no detections to write; every frame was empty")
    if not 1 <= len(ids) <= MAX_BODIES:
        raise ValueError(
            f"{len(ids)} tracked bodies exceeds the NIM's 1..{MAX_BODIES} header limit; "
            "lower --max-bodies"
        )

    rows = 0
    per_frame = collections.Counter()
    with open(path, "w", newline="") as fh:
        fh.write(f"{len(ids)}\n")
        writer = csv.writer(fh, delimiter=" ", lineterminator="\n")
        for fr in frames:
            if len(fr["detections"]) > MAX_BODIES:
                raise ValueError(
                    f"frame {fr['frame_id']} has {len(fr['detections'])} boxes, "
                    f"more than the {MAX_BODIES} per-frame limit"
                )
            for det in fr["detections"]:
                x, y, w, h = det["bbox"]
                if (x, y, w, h) == ABSENT_BBOX:
                    continue
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"frame {fr['frame_id']} has a box with non-positive size: {det['bbox']}"
                    )
                writer.writerow([fr["frame_id"], det["tracking_id"],
                                 f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
                rows += 1
                per_frame[fr["frame_id"]] += 1

    total = sum(len(fr["detections"]) for fr in frames)
    empty = [fr["frame_id"] for fr in frames if fr["frame_id"] not in per_frame]
    return {
        "bodies": len(ids),
        "rows": rows,
        "coverage": 100.0 * rows / total if total else 0.0,
        "frames_covered": len(per_frame),
        "frames_total": len(frames),
        "max_per_frame": max(per_frame.values()) if per_frame else 0,
        "empty_frames": empty,
    }


def read_annotation(path: str) -> dict[int, list[tuple]]:
    """Parse an annotation file into ``{frame_id: [(tracking_id, x, y, w, h)]}``.

    Used by the tests and by ``verify`` to check a file without the NIM client.
    """
    boxes: dict[int, list[tuple]] = collections.defaultdict(list)
    with open(path) as fh:
        header = fh.readline().split()
        if not header:
            raise ValueError(f"{path}: missing the tracked-body count on line 1")
        try:
            n_bodies = int(header[0])
        except ValueError:
            raise ValueError(f"{path}: line 1 is not an integer: {header[0]!r}") from None
        if not 1 <= n_bodies <= MAX_BODIES:
            raise ValueError(f"{path}: header {n_bodies} outside 1..{MAX_BODIES}")
        for lineno, line in enumerate(fh, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 6:
                raise ValueError(
                    f"{path}: line {lineno} has {len(fields)} fields, expected 6"
                )
            frame_id, tracking_id = int(fields[0]), int(fields[1])
            x, y, w, h = (float(v) for v in fields[2:])
            if (x, y, w, h) == ABSENT_BBOX:
                continue
            boxes[frame_id].append((tracking_id, x, y, w, h))
    return dict(boxes)
