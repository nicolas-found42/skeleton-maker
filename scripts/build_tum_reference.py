# SPDX-License-Identifier: MIT
"""Turn a TUM RGB-D sequence into an environment clip, a geometry reference and an annotation.

TUM RGB-D (CC BY 4.0) pairs Kinect colour and depth video with a camera trajectory measured by
an eight-camera motion-capture system. None of that comes from this project's models, so it
can score registration and metric scale independently. The output of one run is:

- ``<name>.mp4``: the colour frames at a constant rate, frame ``i`` being the ``i``-th ``rgb.txt``
  entry;
- ``<name>.geometry-reference.json``: camera intrinsics plus measured fit and check anchors and
  one check dimension, in the world frame of the motion-capture system (input to
  ``skeleton-maker environment --calibration``);
- ``<name>.annotation.json``: a geometry-scope annotation (no masks) whose control points and
  withheld dimension are the check measurements;
- ``<name>.build-report.json``: how frames were associated, how well the depth and pose agree,
  and every parameter used.

Anchors are Shi-Tomasi corners on flat, valid-depth pixels. A corner's world position is its
depth back-projected through the measured pose, so it is a measurement, not a prediction. The
anchor uncertainty is an engineering bound (``ANCHOR_UNCERTAINTY_M``), checked against the
depth-versus-pose agreement that the report records.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_SCHEMA = "skeleton-maker.environment-annotations/1"
REFERENCE_SCHEMA = "skeleton-maker.geometry-reference/1"
#: TUM publishes this scale for its 16-bit depth PNG files: 5000 equals one metre.
DEPTH_SCALE = 5000.0
#: freiburg1 colour camera, from the TUM file-format page (fx, fy, cx, cy, then d0..d4).
FREIBURG1 = {
    "size": [640, 480],
    "intrinsics": [517.3, 516.5, 318.6, 255.3],
    "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
}
MAX_ASSOCIATION_S = 0.02
DEPTH_RANGE_M = (0.5, 3.5)
#: Per-point bound, above the median 2 cm depth-versus-pose disagreement measured on freiburg1.
ANCHOR_UNCERTAINTY_M = 0.03
FIT_FRAMES, FIT_PER_FRAME = 4, 2
CHECK_FRAMES, CHECK_PER_FRAME = 4, 3
MIN_DIMENSION_M = 0.4
#: A path or turn smaller than these is "not moving" when tagging camera motion.
TRANSLATION_M, ROTATION_DEG = 0.3, 20.0
#: Frames at or above this camera turn rate smear (checked by eye on freiburg1/rpy at 185 deg/s);
#: a clip is tagged ``motion_blur`` when at least ``BLUR_FRACTION`` of its poses turn this fast.
BLUR_DEG_S, BLUR_FRACTION = 100.0, 0.05


class ReferenceError_(RuntimeError):
    """The sequence cannot yield the required measurements."""


def read_table(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(line.split())
    return rows


def quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Sequence:
    """The associated colour, depth and pose records of one extracted TUM sequence."""

    def __init__(self, root: Path, camera: dict = FREIBURG1):
        self.root = root
        fx, fy, cx, cy = camera["intrinsics"]
        self.k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.dist = np.array(camera["distortion"], dtype=np.float64)
        self.size = tuple(camera["size"])
        self.rgb = [(float(r[0]), r[1]) for r in read_table(root / "rgb.txt")]
        depth = [(float(r[0]), r[1]) for r in read_table(root / "depth.txt")]
        poses = np.array([[float(v) for v in r] for r in read_table(root / "groundtruth.txt")])
        if not self.rgb or not depth or not len(poses):
            raise ReferenceError_(f"{root}: rgb, depth or groundtruth is empty")
        depth_times = np.array([d[0] for d in depth])
        pose_times = poses[:, 0]
        self.depth_file: list[str | None] = []
        self.pose: list[np.ndarray | None] = []
        self.association_s: list[tuple[float, float]] = []
        for stamp, _ in self.rgb:
            di = int(np.argmin(np.abs(depth_times - stamp)))
            pi = int(np.argmin(np.abs(pose_times - stamp)))
            dd, dp = abs(depth_times[di] - stamp), abs(pose_times[pi] - stamp)
            self.association_s.append((float(dd), float(dp)))
            if dd > MAX_ASSOCIATION_S or dp > MAX_ASSOCIATION_S:
                self.depth_file.append(None)
                self.pose.append(None)
                continue
            transform = np.eye(4)
            transform[:3, :3] = quaternion_matrix(*poses[pi, 4:8])
            transform[:3, 3] = poses[pi, 1:4]
            self.depth_file.append(depth[di][1])
            self.pose.append(transform)
        self.poses = poses

    def __len__(self) -> int:
        return len(self.rgb)

    def usable(self, frame: int) -> bool:
        return self.pose[frame] is not None

    def transform(self, frame: int) -> np.ndarray:
        pose = self.pose[frame]
        if pose is None:
            raise ReferenceError_(f"frame {frame} has no associated pose")
        return pose

    def depth(self, frame: int) -> np.ndarray:
        image = cv2.imread(str(self.root / str(self.depth_file[frame])), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ReferenceError_(f"cannot read the depth image for frame {frame}")
        return image.astype(np.float32) / DEPTH_SCALE

    def gray(self, frame: int) -> np.ndarray:
        image = cv2.imread(str(self.root / self.rgb[frame][1]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ReferenceError_(f"cannot read the colour image for frame {frame}")
        return image

    def world_point(self, frame: int, pixel: tuple[float, float], depth_m: float) -> np.ndarray:
        point = np.array([[pixel]], dtype=np.float32)
        x, y = cv2.undistortPoints(point, self.k, self.dist)[0, 0]
        camera = np.array([x * depth_m, y * depth_m, depth_m, 1.0])
        return (self.transform(frame) @ camera)[:3]


def _turn_rates(poses: np.ndarray, window_s: float = 1 / 30) -> np.ndarray:
    """Camera turn rate in degrees per second over about one colour-frame interval."""
    times = poses[:, 0]
    rates = []
    for row in poses:
        j = int(np.searchsorted(times, row[0] + window_s))
        if j >= len(poses) or times[j] == row[0]:
            continue
        relative = quaternion_matrix(*row[4:8]).T @ quaternion_matrix(*poses[j, 4:8])
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)))
        rates.append(angle / (times[j] - row[0]))
    return np.array(rates)


def camera_motion(poses: np.ndarray) -> dict:
    """Translation extent, total rotation and turn rates of the measured trajectory."""
    positions = poses[:, 1:4]
    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    first = quaternion_matrix(*poses[0, 4:8])
    angles = []
    for row in poses[:: max(1, len(poses) // 200)]:
        relative = first.T @ quaternion_matrix(*row[4:8])
        angles.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))))
    rates = _turn_rates(poses)
    return {
        "translation_extent_m": extent,
        "max_rotation_deg": float(max(angles)),
        "peak_turn_rate_deg_s": float(rates.max()) if len(rates) else 0.0,
        "fast_turn_fraction": float((rates >= BLUR_DEG_S).mean()) if len(rates) else 0.0,
    }


def motion_tags(motion: dict) -> list[str]:
    moved = motion["translation_extent_m"] >= TRANSLATION_M
    turned = motion["max_rotation_deg"] >= ROTATION_DEG
    tags = []
    if moved:
        tags.append("translating")
    if turned and not moved:
        tags += ["pan", "underconstrained"]
    if not moved and not turned:
        tags += ["tripod", "underconstrained"]
    if motion["fast_turn_fraction"] >= BLUR_FRACTION:
        tags.append("motion_blur")
    return tags


def _flat_corners(seq: Sequence, frame: int) -> list[dict]:
    """Corners on flat, valid-depth pixels, best response first, with measured world position."""
    gray, depth = seq.gray(frame), seq.depth(frame)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.02, minDistance=15)
    if corners is None:
        return []
    width, height = seq.size
    near, far = DEPTH_RANGE_M
    found = []
    for x, y in corners.reshape(-1, 2):
        col, row = round(float(x)), round(float(y))
        if not (20 <= col < width - 20 and 20 <= row < height - 20):
            continue
        window = depth[row - 4 : row + 5, col - 4 : col + 5]
        valid = window[window > 0]
        if valid.size < 0.8 * window.size or float(valid.std()) > 0.012:
            continue
        z = float(np.median(valid))
        if not near <= z <= far:
            continue
        found.append(
            {
                "frame_id": frame,
                "pixel": [float(col), float(row)],
                "position_m": seq.world_point(frame, (float(col), float(row)), z).tolist(),
            }
        )
    return found


def _spread(points: list[dict], count: int) -> list[dict]:
    """Farthest-point selection in world space; deterministic given the candidate order."""
    if len(points) < count:
        raise ReferenceError_(f"only {len(points)} usable corners; need {count}")
    chosen = [points[0]]
    while len(chosen) < count:
        _, best = max(
            (
                min(
                    float(
                        np.linalg.norm(
                            np.array(candidate["position_m"]) - np.array(c["position_m"])
                        )
                    )
                    for c in chosen
                ),
                index,
            )
            for index, candidate in enumerate(points)
            if candidate not in chosen
        )
        chosen.append(points[best])
    return chosen


def _evenly(items: list[int], count: int) -> list[int]:
    if len(items) < count:
        raise ReferenceError_(f"only {len(items)} sampled frames with depth and pose; need {count}")
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in indices]


def build_anchors(seq: Sequence, sample_step: int) -> dict:
    grid = [f for f in range(0, len(seq), sample_step) if seq.usable(f)]
    fit_frames = _evenly(grid[0::2], FIT_FRAMES)
    check_frames = _evenly(grid[1::2], CHECK_FRAMES)
    fit, check = [], []
    for frame in fit_frames:
        for point in _spread(_flat_corners(seq, frame), FIT_PER_FRAME):
            fit.append({**point, "id": f"fit-{len(fit):03d}", "role": "fit"})
    for frame in check_frames:
        for point in _spread(_flat_corners(seq, frame), CHECK_PER_FRAME):
            check.append({**point, "id": f"check-{len(check):03d}", "role": "check"})
    for anchor in (*fit, *check):
        anchor["uncertainty_m"] = ANCHOR_UNCERTAINTY_M
    singular = np.linalg.svd(
        np.array([a["position_m"] for a in fit]) - np.mean([a["position_m"] for a in fit], axis=0),
        compute_uv=False,
    )
    if singular[1] < 0.05 * singular[0]:
        raise ReferenceError_("fit anchors are nearly collinear; the transform is underdetermined")
    return {"fit": fit, "check": check, "fit_frames": fit_frames, "check_frames": check_frames}


def build_dimension(check: list[dict]) -> dict:
    best = None
    for i, a in enumerate(check):
        for b in check[i + 1 :]:
            if a["frame_id"] != b["frame_id"]:
                continue
            length = float(np.linalg.norm(np.array(a["position_m"]) - np.array(b["position_m"])))
            if length >= MIN_DIMENSION_M and (best is None or length > best[0]):
                best = (length, a, b)
    if best is None:
        raise ReferenceError_(f"no pair of check points is at least {MIN_DIMENSION_M} m apart")
    length, a, b = best
    return {
        "id": "check-dimension",
        "a": {"frame_id": a["frame_id"], "pixel": a["pixel"]},
        "b": {"frame_id": b["frame_id"], "pixel": b["pixel"]},
        "length_m": length,
        "uncertainty_m": float(np.hypot(ANCHOR_UNCERTAINTY_M, ANCHOR_UNCERTAINTY_M)),
        "role": "check",
    }


def depth_pose_agreement(seq: Sequence, pairs: list[tuple[int, int]]) -> dict:
    """Median disagreement between one frame's depth and another's, through the measured poses.

    This is the evidence behind ``ANCHOR_UNCERTAINTY_M``: if the poses or the depth scale were
    wrong, points seen in two frames would not land on the second frame's depth.
    """
    errors = []
    width, height = seq.size
    near, far = DEPTH_RANGE_M
    u, v = np.meshgrid(np.arange(0, width, 4), np.arange(0, height, 4))
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, seq.k, seq.dist).reshape(-1, 2)
    for a, b in pairs:
        if not (seq.usable(a) and seq.usable(b)):
            continue
        depth_a, depth_b = seq.depth(a), seq.depth(b)
        z = depth_a[v.ravel(), u.ravel()]
        keep = (z > near) & (z < far)
        camera_a = np.concatenate(
            [normalized[keep] * z[keep, None], z[keep, None], np.ones((keep.sum(), 1))], axis=1
        )
        camera_b = (np.linalg.inv(seq.transform(b)) @ seq.transform(a) @ camera_a.T).T[:, :3]
        in_front = camera_b[camera_b[:, 2] > near]  # points behind the camera have no pixel
        if not len(in_front):
            continue
        projected, _ = cv2.projectPoints(
            in_front.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), seq.k, seq.dist
        )
        cols = np.round(projected[:, 0, 0]).astype(int)
        rows = np.round(projected[:, 0, 1]).astype(int)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        seen = depth_b[rows[inside], cols[inside]]
        valid = seen > near
        errors.extend(np.abs(seen[valid] - in_front[inside][valid][:, 2]).tolist())
    if not errors:
        raise ReferenceError_("no overlapping depth to compare between the frame pairs")
    return {
        "pairs": [list(p) for p in pairs],
        "points": len(errors),
        "median_abs_error_m": float(np.median(errors)),
        "p90_abs_error_m": float(np.percentile(errors, 90)),
    }


def write_clip(seq: Sequence, destination: Path, frame_rate: int) -> str:
    """Encode every colour frame, in order, at a constant frame rate; returns the SHA-256."""
    if shutil.which("ffmpeg") is None:
        raise ReferenceError_("ffmpeg is required to encode the clip")
    with tempfile.TemporaryDirectory() as tmp:
        for index, (_, relative) in enumerate(seq.rgb):
            (Path(tmp) / f"{index:06d}.png").symlink_to((seq.root / relative).resolve())
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(frame_rate),
            "-i",
            str(Path(tmp) / "%06d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        subprocess.run(command, check=True)  # noqa: S603 - fixed argument list, no shell
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def build(
    root: Path,
    out_dir: Path,
    *,
    name: str,
    split: str,
    frame_rate: int = 30,
    sample_step: int = 30,
    camera: dict = FREIBURG1,
    encode: bool = True,
    clip_sha256: str | None = None,
    extra_tags: tuple[str, ...] = ("indoor",),
) -> dict:
    seq = Sequence(root, camera)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = build_anchors(seq, sample_step)
    dimension = build_dimension(anchors["check"])
    motion = camera_motion(seq.poses)
    sha = write_clip(seq, out_dir / f"{name}.mp4", frame_rate) if encode else clip_sha256
    if sha is None:
        raise ReferenceError_("clip_sha256 is required when the clip is not encoded")
    width, height = seq.size
    distortion = {"model": "opencv-brown-conrady", "coefficients": camera["distortion"]}
    reference = {
        "schema": REFERENCE_SCHEMA,
        "world_frame": {"id": "tum-mocap-world", "units": "m", "handedness": "right"},
        "image_size": [width, height],
        "camera": {
            "intrinsics": [
                [camera["intrinsics"][0], 0, camera["intrinsics"][2]],
                [0, camera["intrinsics"][1], camera["intrinsics"][3]],
                [0, 0, 1],
            ],
            "distortion": distortion,
        },
        "anchors": [
            {k: a[k] for k in ("id", "frame_id", "pixel", "position_m", "role", "uncertainty_m")}
            for a in (*anchors["fit"], *anchors["check"])
        ],
        "measured_dimensions": [dimension],
    }
    annotation = {
        "schema": ANNOTATION_SCHEMA,
        "clip": name,
        "source_sha256": sha,
        "split": split,
        "scope": "geometry",
        "width": width,
        "height": height,
        "frame_rate": [frame_rate, 1],
        "tags": sorted({*extra_tags, *motion_tags(motion)}),
        "aliases": {},
        "provenance": {
            "kind": "published_dataset",
            "source": f"TUM RGB-D {root.name}",
            "citation": "Sturm, Engelhard, Endres, Burgard, Cremers, IROS 2012",
            "license": "CC BY 4.0",
            "quality_control": (
                "camera poses from an eight-camera motion-capture system at 100 Hz; Kinect depth "
                f"associated to colour within {MAX_ASSOCIATION_S} s; anchors are measurements, "
                f"each bounded at {ANCHOR_UNCERTAINTY_M} m"
            ),
        },
        "shots": [{"first_frame": 0, "last_frame": len(seq) - 1}],
        "frames": [],
        "geometry": {
            "eligible_frames": anchors["check_frames"],
            "control_points": [
                {"frame_id": a["frame_id"], "id": a["id"], "xy": a["pixel"]}
                for a in anchors["check"]
            ],
            "dimensions": [
                {
                    "id": dimension["id"],
                    "meters": dimension["length_m"],
                    "uncertainty_m": dimension["uncertainty_m"],
                    "withheld": True,
                }
            ],
        },
    }
    report = {
        "sequence": root.name,
        "frames": len(seq),
        "frames_without_association": sum(1 for p in seq.pose if p is None),
        "worst_association_s": {
            "depth": max(a for a, _ in seq.association_s),
            "pose": max(b for _, b in seq.association_s),
        },
        "camera_motion": motion,
        "depth_pose_agreement": depth_pose_agreement(
            seq,
            [(f, g) for f, g in zip(anchors["fit_frames"], anchors["check_frames"], strict=True)],
        ),
        "parameters": {
            "sample_step": sample_step,
            "frame_rate": frame_rate,
            "max_association_s": MAX_ASSOCIATION_S,
            "depth_range_m": DEPTH_RANGE_M,
            "anchor_uncertainty_m": ANCHOR_UNCERTAINTY_M,
            "fit_frames": anchors["fit_frames"],
            "check_frames": anchors["check_frames"],
        },
        "clip_sha256": sha,
    }
    for suffix, document in (
        ("geometry-reference", reference),
        ("annotation", annotation),
        ("build-report", report),
    ):
        (out_dir / f"{name}.{suffix}.json").write_text(json.dumps(document, indent=2) + "\n")
    return {"reference": reference, "annotation": annotation, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("sequence", type=Path, help="an extracted TUM RGB-D sequence directory")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--sample-step", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        result = build(
            args.sequence,
            args.out_dir,
            name=args.name,
            split=args.split,
            sample_step=args.sample_step,
        )
    except ReferenceError_ as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = result["report"]
    print(
        f"{args.name}: {report['frames']} frames, tags {result['annotation']['tags']}, "
        f"depth/pose agreement {report['depth_pose_agreement']['median_abs_error_m']:.3f} m median"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
