# SPDX-License-Identifier: MIT
"""Visual shot boundaries for a clip, independent of any person or model backend.

A cut is a frame whose colour histogram is far from the previous frame's *and* stands out
from its neighbours. Requiring the stand-out is what keeps fades, gradual lighting changes
and whip pans (all of which change many frames a little, or move content without
changing its colours) from being reported as cuts.

Detection is coarse-to-fine: sampled frames find the intervals where the scene changed,
then only those intervals are decoded frame by frame to place the cut on its exact frame.
"""

from itertools import pairwise

import cv2
import numpy as np

from . import stage

METHOD = "color-histogram-isolated-peak"
VERSION = 1

THUMBNAIL = (64, 36)
BINS = 8
#: Sampled frames at least this far apart (0..1) make their interval a candidate.
CANDIDATE_DISTANCE = 0.25
#: A cut frame must be at least this far from the previous one...
CUT_DISTANCE = 0.40
#: ...and this many times farther than every frame pair within NEIGHBOURS frames of it
#: (floored below), so a run of large changes (a fade, a whip pan) is not a cut.
ISOLATION = 3.0
NEIGHBOURS = 2
NEIGHBOUR_FLOOR = 0.05
#: A cut within this many frames of the previous one (or of the clip's end) is a flash.
MIN_SHOT_FRAMES = 3

PARAMETERS = {
    "thumbnail": list(THUMBNAIL),
    "bins_per_channel": BINS,
    "candidate_distance": CANDIDATE_DISTANCE,
    "cut_distance": CUT_DISTANCE,
    "isolation": ISOLATION,
    "neighbours": NEIGHBOURS,
    "neighbour_floor": NEIGHBOUR_FLOOR,
    "min_shot_frames": MIN_SHOT_FRAMES,
}


class ShotDetectionError(RuntimeError):
    """The video could not be decoded for shot detection."""


def _signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, THUMBNAIL, interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None, [BINS] * 3, [0, 256] * 3).ravel()
    return hist / max(float(hist.sum()), 1.0)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _decode(video: str, wanted: set, last: int) -> dict:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ShotDetectionError(f"cannot decode {video}")
    sigs = {}
    try:
        for index in range(last + 1):
            if not cap.grab():
                break
            if index in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    sigs[index] = _signature(frame)
    finally:
        cap.release()
    return sigs


def detect_cuts(video: str, frame_count: int, sampled_ids) -> list[dict]:
    """Cuts as ``{"frame_id", "distance", "isolation"}``; ``frame_id`` starts the new shot."""
    anchors = sorted({0, *sampled_ids, frame_count - 1})
    sigs = _decode(video, set(anchors), anchors[-1])
    gaps = [
        (a, b)
        for a, b in pairwise(anchors)
        if a in sigs and b in sigs and _distance(sigs[a], sigs[b]) >= CANDIDATE_DISTANCE
    ]
    if not gaps:
        return []
    wanted = set()
    for a, b in gaps:
        wanted.update(range(max(0, a - NEIGHBOURS), min(frame_count - 1, b + NEIGHBOURS) + 1))
    dense = _decode(video, wanted, max(wanted))

    found = []
    for a, b in gaps:
        lo, hi = max(0, a - NEIGHBOURS), min(frame_count - 1, b + NEIGHBOURS)
        if any(i not in dense for i in range(lo, hi + 1)):
            continue
        dist = {j: _distance(dense[j - 1], dense[j]) for j in range(lo + 1, hi + 1)}
        for j in range(a + 1, b + 1):
            neighbours = [
                dist[k] for k in range(j - NEIGHBOURS, j + NEIGHBOURS + 1) if k != j and k in dist
            ]
            reference = max([NEIGHBOUR_FLOOR, *neighbours])
            if dist[j] >= CUT_DISTANCE and dist[j] >= ISOLATION * reference:
                found.append({"frame_id": j, "distance": dist[j], "isolation": dist[j] / reference})
    cuts, previous = [], 0
    for cut in sorted(found, key=lambda c: c["frame_id"]):
        c = cut["frame_id"]
        if c - previous >= MIN_SHOT_FRAMES and frame_count - c >= MIN_SHOT_FRAMES:
            cuts.append(cut)
            previous = c
    return cuts


def build_shots(cuts: list[dict], frame_count: int, clock) -> list[dict]:
    """Contiguous shots covering the whole clip, with exact start and end times."""
    starts = [0, *(c["frame_id"] for c in cuts)]
    ends = [*(s - 1 for s in starts[1:]), frame_count - 1]
    return [
        {
            "id": f"shot-{i}",
            "first_frame": a,
            "last_frame": b,
            "first_time": clock.time_pair(a),
            "last_time": clock.time_pair(b),
        }
        for i, (a, b) in enumerate(zip(starts, ends, strict=True))
    ]


def pose_stage_shots(records: list[dict], fps: float) -> list[list[int]]:
    """The shots the character stage derives from the poses, as ``[first, last]`` frame ids."""
    frames = sorted((r["frame_id"], r["detections"]) for r in records)
    return [list(s) for s in stage.detect_shots(frames, stage.Options(fps=fps))]


def reconcile(cuts: list[dict], pose_shots: list[list[int]] | None, clock) -> dict:
    """Describe where visual cuts and pose-stage shots agree and where they do not.

    The pose stage only sees frames with people, so its boundary lies somewhere in the gap
    between one shot's last frame and the next one's first. A visual cut in such a gap
    ``agrees``; one inside a pose shot is ``visual_only`` (the tracker held its ids across it);
    one before the first or after the last pose frame is ``outside_pose_coverage`` (the poses
    are silent there). A gap with no visual cut in it is ``pose_only``.
    """
    gaps = [] if pose_shots is None else [(a[1], b[0]) for a, b in pairwise(pose_shots)]
    boundaries = []
    for cut in cuts:
        c = cut["frame_id"]
        if pose_shots is None:
            agreement = "no_poses"
        elif any(after < c <= before for after, before in gaps):
            agreement = "agrees"
        elif pose_shots and pose_shots[0][0] <= c <= pose_shots[-1][1]:
            agreement = "visual_only"
        else:
            agreement = "outside_pose_coverage"
        boundaries.append(
            {
                "frame_id": c,
                "time": clock.time_pair(c),
                "distance": round(cut["distance"], 4),
                "isolation": round(cut["isolation"], 2),
                "agreement": agreement,
            }
        )
    pose_only = [
        {"after_frame": after, "before_frame": before}
        for after, before in gaps
        if not any(after < c["frame_id"] <= before for c in cuts)
    ]
    return {
        "method": METHOD,
        "version": VERSION,
        "parameters": PARAMETERS,
        "pose_stage_shots": pose_shots,
        "boundaries": boundaries,
        "pose_only": pose_only,
    }
