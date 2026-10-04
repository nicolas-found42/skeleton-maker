# SPDX-License-Identifier: MIT
"""Draw the Nova-77 skeletons returned by the NIM onto the source video.

The overlay mirrors the AR SDK sample: a box and id label per tracked body,
2D keypoints in green, and the 3D keypoints reprojected through a pinhole camera
in red. The 3D projection is only as good as the focal length, and the server
does not report which focal the model actually ran with, so ``--focus auto``
reuses the focal length the NIM echoed on the stream when there is one.
"""

import json

import cv2
import numpy as np

from .constants import (
    BBOX_MARGIN,
    COLOR_2D,
    COLOR_3D,
    DRAW_KEYPOINTS_CONFIGS,
    NOVA77_SKELETON_LINKS,
    TRACK_COLORS,
)
from .utils import run_ffmpeg


def load_poses(path: str) -> dict:
    """Read the JSON Lines pose file into ``{frame_id: [detection, ...]}``."""
    poses = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            poses[record["frame_id"]] = record["detections"]
    return poses


def _pixel(point) -> tuple:
    return round(float(point[0])), round(float(point[1]))


def _draw_skeleton(frame, points, visible, color, radius: int) -> None:
    for joint in np.flatnonzero(visible):
        cv2.circle(frame, _pixel(points[joint]), radius, color, -1)
    for child, parent in NOVA77_SKELETON_LINKS:
        if visible[child] and visible[parent]:
            cv2.line(frame, _pixel(points[child]), _pixel(points[parent]), color, 2)


def draw_detection(frame, det, color) -> None:
    """Draw one body's box, label and keypoints onto ``frame`` in place."""
    x, y, w, h = det["bbox"]
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (round(x), round(y)), (round(x + w), round(y + h)), color, 2)
    cv2.putText(
        frame,
        f"ID: {det['tracking_id']}",
        (round(x), max(round(y) - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )

    if det.get("_draw_2d"):
        kp2d = np.asarray(det["keypoints_2d"], np.float32).reshape(-1, 2)
        conf = np.asarray(det["keypoints_confidence"], np.float32)
        if len(kp2d):
            mx, my = w * BBOX_MARGIN, h * BBOX_MARGIN
            px, py = kp2d[:, 0], kp2d[:, 1]
            visible = (
                (conf > 0.0)
                & np.isfinite(kp2d).all(axis=1)
                & (px >= x - mx)
                & (px <= x + w + mx)
                & (py >= y - my)
                & (py <= y + h + my)
            )
            _draw_skeleton(frame, kp2d, visible, COLOR_2D, 4)

    if det.get("_draw_3d"):
        kp3d = np.asarray(det["keypoints_3d"], np.float32).reshape(-1, 3)
        focal = det.get("_focal") or float(np.hypot(width, height))
        if len(kp3d):
            z = kp3d[:, 2:3]
            with np.errstate(divide="ignore", invalid="ignore"):
                projected = focal * kp3d[:, :2] / z + (width * 0.5, height * 0.5)
            px, py = projected[:, 0], projected[:, 1]
            visible = (
                (z[:, 0] > 1e-6)
                & np.isfinite(projected).all(axis=1)
                & (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
            )
            _draw_skeleton(frame, projected, visible, COLOR_3D, 3)


def render(
    video: str,
    pose_json: str,
    output: str,
    *,
    draw: str = "2d",
    focal_length: float = 0.0,
    quality: int = 19,
    verbose: bool = True,
) -> int:
    """Render the overlay and return the number of frames written."""
    if draw not in DRAW_KEYPOINTS_CONFIGS:
        raise SystemExit(f"error: --draw must be one of {list(DRAW_KEYPOINTS_CONFIGS)}")
    want_2d, want_3d = DRAW_KEYPOINTS_CONFIGS[draw]

    poses = load_poses(pose_json)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp = output + ".raw.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"error: cannot write {tmp}")

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(
                frame,
                f"Frame: {frame_id}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            for det in poses.get(frame_id, []):
                det["_draw_2d"], det["_draw_3d"] = want_2d, want_3d
                det["_focal"] = focal_length
                color = TRACK_COLORS[det["tracking_id"] % len(TRACK_COLORS)]
                draw_detection(frame, det, color)
            writer.write(frame)
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    # Re-encode: OpenCV's mp4v is not a web-friendly codec.
    run_ffmpeg(
        [
            "-i",
            tmp,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(quality),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output,
            "-y",
        ],
        f"re-encoding {output}",
    )
    import os

    os.remove(tmp)
    if verbose:
        print(f"wrote {output} ({frame_id} frames, {draw} keypoints)")
    return frame_id


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("render", help="draw skeletons onto the video")
    p.add_argument("video", help="the clip the poses were computed for")
    p.add_argument("pose_json", help="pose output from `pose`")
    p.add_argument("--out", required=True, help="overlay video to write (MP4)")
    p.add_argument(
        "--draw",
        choices=list(DRAW_KEYPOINTS_CONFIGS),
        default="2d",
        help="2d | 3d | both (default: 2d; the 3D layer needs a real focal length)",
    )
    p.add_argument(
        "--focal-length",
        type=float,
        default=0.0,
        help="focal length in pixels for the 3D reprojection; 0 uses the frame diagonal",
    )
    p.add_argument(
        "--quality", type=int, default=19, help="x264 CRF (default: 19, lower is better)"
    )


def run_cli(args) -> int:
    render(
        args.video,
        args.pose_json,
        args.out,
        draw=args.draw,
        focal_length=args.focal_length,
        quality=args.quality,
    )
    return 0
