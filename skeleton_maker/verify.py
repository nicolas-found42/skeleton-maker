# SPDX-License-Identifier: MIT
"""Check a run's artifacts, so "done" rests on a script rather than a log.

Every check compares the pose output back to the annotation that was actually
sent. A mismatch there is the failure mode that matters: the NIM matches poses
to boxes by frame index, so a silent off-by-one would look like success.
"""

import collections
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

from .bbox import read_annotation
from .constants import MAX_BODIES, NUM_JOINTS
from .utils import require_tool


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _probe(path: str) -> dict:
    ffprobe = require_tool("ffprobe", "It ships with ffmpeg.")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,codec_name,pix_fmt,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return {
        "w": int(s["width"]), "h": int(s["height"]), "n": int(s.get("nb_frames", 0) or 0),
        "codec": s.get("codec_name"), "pix": s.get("pix_fmt"),
        "r_fps": s.get("r_frame_rate"),
        "dur": float(j["format"]["duration"]),
    }


def run(clip: str, boxes: str, poses: str, overlay: str | None = None,
        *, report: bool = True) -> bool:
    """Run every check. Returns True when all pass."""
    failures = []

    def record(name, ok, detail=""):
        if not check(name, ok, detail):
            failures.append(name)

    sent = read_annotation(boxes)
    sent_ids = {row[0] for rows in sent.values() for row in rows}
    sent_boxes = sum(len(rows) for rows in sent.values())
    max_per_frame = max((len(rows) for rows in sent.values()), default=0)
    record("annotation header within 1..50", 1 <= len(sent_ids) <= MAX_BODIES,
           f"{len(sent_ids)} bodies")
    record("no frame exceeds the per-frame box limit", max_per_frame <= MAX_BODIES,
           f"max {max_per_frame}")

    records = []
    with open(poses) as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    frame_ids = [r["frame_id"] for r in records]
    dets = [d for r in records for d in r["detections"]]
    dets_per_frame = collections.Counter(r["frame_id"] for r in records
                                         for _ in r["detections"])

    # The NIM emits one record per decoded frame, while the annotation may
    # legitimately skip frames where nobody was tracked. So the ids need not be
    # equal: the file must be contiguous from 0, and every annotated frame must
    # come back with its detections.
    record("pose frame ids are contiguous from 0",
           frame_ids == list(range(len(frame_ids))),
           f"{len(records)} records, ids {frame_ids[0]}..{frame_ids[-1]}"
           if frame_ids else "no records")
    unposed = [f for f in sorted(sent) if dets_per_frame.get(f, 0) == 0]
    record("every annotated frame came back with poses", not unposed,
           f"{len(sent)} annotated frames" + (f", missing {unposed}" if unposed else ""))
    record("every submitted box produced a detection", len(dets) == sent_boxes,
           f"{len(dets)} detections vs {sent_boxes} boxes submitted")

    bad = 0
    for det in dets:
        arrays_ok = (
            len(det["keypoints_2d"]) == NUM_JOINTS
            and len(det["keypoints_3d"]) == NUM_JOINTS
            and len(det["keypoints_confidence"]) == NUM_JOINTS
            and len(det["rest_pose"]) == NUM_JOINTS
            and len(det["joint_rotations"]) == NUM_JOINTS
        )
        if not arrays_ok:
            bad += 1
            continue
        shapes_ok = (
            all(len(p) == 2 for p in det["keypoints_2d"])
            and all(len(p) == 3 for p in det["keypoints_3d"])
            and all(len(p) == 3 for p in det["rest_pose"])
            and all(len(q) == 4 for q in det["joint_rotations"])
            and len(det["bbox"]) == 4
            and len(det["root_pose"]["translation"]) == 3
            and len(det["root_pose"]["rotation"]) == 4
        )
        if not shapes_ok:
            bad += 1
    record(f"all {NUM_JOINTS}-joint arrays are well formed", bad == 0, f"{bad} malformed")

    if dets:
        norms = [math.sqrt(sum(c * c for c in q))
                 for det in dets for q in det["joint_rotations"]]
        record("joint rotations are unit quaternions",
               all(abs(n - 1.0) < 0.05 for n in norms),
               f"norm {min(norms):.3f}..{max(norms):.3f}")

    clip_info = _probe(clip)
    record("clip is H.264 yuv420p at a constant frame rate",
           clip_info["codec"] == "h264" and clip_info["pix"] == "yuv420p"
           and clip_info["r_fps"] == clip_info["r_fps"],
           f"{clip_info['w']}x{clip_info['h']} {clip_info['codec']} "
           f"{clip_info['pix']} {clip_info['r_fps']} {clip_info['dur']:.3f}s")

    if overlay:
        info = _probe(overlay)
        record("overlay matches the clip's geometry and frame count",
               (info["w"], info["h"], info["n"]) == (clip_info["w"], clip_info["h"], clip_info["n"]),
               f"{info['w']}x{info['h']} {info['n']} frames")
        cap_a, cap_b = cv2.VideoCapture(clip), cv2.VideoCapture(overlay)
        diffs, idx = [], 0
        while True:
            ok_a, fa = cap_a.read()
            ok_b, fb = cap_b.read()
            if not ok_a or not ok_b:
                break
            if idx % 30 == 0:
                ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY).astype(np.int16)
                gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY).astype(np.int16)
                diffs.append(float(np.abs(ga - gb).mean()))
            idx += 1
        cap_a.release()
        cap_b.release()
        record("overlay actually differs from the source clip",
               bool(diffs) and all(d > 0.5 for d in diffs),
               f"{len(diffs)} sampled frames, mean abs luma diff "
               f"{sum(diffs) / len(diffs):.2f}" if diffs else "no frames compared")

    if report:
        print()
        if failures:
            print(f"{len(failures)} check(s) failed: {failures}")
        else:
            print("all checks passed")
    return not failures


def cli(args) -> int:
    return 0 if run(args.clip, args.boxes, args.poses, args.overlay) else 1
