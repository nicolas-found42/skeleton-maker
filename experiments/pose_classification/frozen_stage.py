# SPDX-License-Identifier: MIT
"""Turn ``pose.json`` into a *stage*: clean, shot-aware tracks a character can wear.

Why this exists: the NIM returns one skeleton per tracked box per frame, in camera
coordinates, with whatever the tracker did to the ids. A character needs more:

* **Shots.** Camera cuts restart the tracker, and 3D positions are relative to the
  camera, so a cut teleports every body. Poses are grouped into shots; each shot
  gets its own up axis and floor.
* **Clean tracks.** Low-confidence joints are dropped, short gaps interpolated,
  spikes removed, motion smoothed. Long gaps end the track so the character leaves.
* **Up and floor.** Camera coordinates are y-down. Each shot is rotated so the
  people's spine points up, and the floor is the low percentile of foot height.
* **Left and right.** Which mirrored side of the skeleton is the person's left is
  decided per body from where its toes point (see :mod:`skeleton_maker.nova77`).

Everything is positions: the NIM's ``rest_pose`` is a bone-aligned layout whose
bone lengths match the keypoints, not an anatomical pose, so rotations are not
needed to drive a character and are ignored here.
"""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .nova77 import CANON, CANON_INDEX, canon_sources

NUM_JOINTS = 77
MISSING = -32768  # int16 sentinel for "no data" in the packed blob
VERSION = 1

# Joints that must be present for a frame of a body to be usable.
CORE = (0, 3, 6)  # Hips, Chest, Head
FOOT_JOINTS = (70, 71, 75, 76)  # heels and toes, both sides


@dataclass
class Options:
    """Tunables for cleaning. Defaults suit 30 fps footage."""

    fps: float = 30.0
    min_conf: float = 0.0  # joints with confidence <= this are dropped
    max_gap: int = 5  # interpolate gaps up to this many frames
    min_len: int = 6  # drop track segments shorter than this
    smooth_sigma: float = 1.5  # gaussian sigma in frames
    cut_jump_m: float = 1.2  # root jump (m) in one frame that implies a camera cut
    cut_gap_frames: int = 60  # bodies absent this long always start a new shot
    max_people: int = 60


@dataclass
class Stage:
    meta: dict
    blob: bytes = field(repr=False)

    def payload(self) -> str:
        """gzip + base64 of ``[len(meta_json) u32][meta_json][blob]``."""
        head = json.dumps(self.meta, separators=(",", ":")).encode()
        raw = len(head).to_bytes(4, "little") + head + self.blob
        return base64.b64encode(gzip.compress(raw, 6)).decode()

    @staticmethod
    def from_payload(text: str) -> Stage:
        raw = gzip.decompress(base64.b64decode(text))
        n = int.from_bytes(raw[:4], "little")
        return Stage(json.loads(raw[4 : 4 + n]), raw[4 + n :])

    def positions(self, shot: int, track: int) -> np.ndarray:
        """Packed positions of one track as ``(n, len(CANON), 3)`` metres, NaN where missing."""
        t = self.meta["shots"][shot]["tracks"][track]
        n, j = t["n"], len(CANON)
        arr = np.frombuffer(self.blob, dtype="<i2", count=n * j * 3, offset=t["offset"]).reshape(
            n, j, 3
        )
        out = arr.astype(np.float64) / 1000.0
        out[arr == MISSING] = np.nan
        return out + np.asarray(self.meta["shots"][shot]["center"])


# --- loading ---------------------------------------------------------------


def load_frames(path: str) -> list:
    """``[(frame_id, [detection, ...]), ...]`` sorted by frame id."""
    frames = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                frames[rec["frame_id"]] = rec["detections"]
    return sorted(frames.items())


def _root(det) -> np.ndarray:
    return np.asarray(det["root_pose"]["translation"], np.float64)


def detect_shots(frames: list, opts: Options) -> list:
    """Split frames into shots; returns a list of ``(first_frame, last_frame)``.

    A cut is a frame where the set of tracked ids shares nothing with the last
    frame that had bodies, or where shared bodies jump further in one frame than a
    person can move. The second test catches trackers that keep an id over a cut.
    """
    shots, start, prev = [], None, None
    for fid, dets in frames:
        if not dets:
            continue
        if start is None:
            start = fid
        elif prev is not None:
            pf, pd = prev
            ids_now = {d["tracking_id"]: d for d in dets}
            shared = [(ids_now[d["tracking_id"]], d) for d in pd if d["tracking_id"] in ids_now]
            jump = (
                shared
                and np.median([np.linalg.norm(_root(a) - _root(b)) for a, b in shared])
                > opts.cut_jump_m
            )
            if fid - pf > opts.cut_gap_frames or not shared or jump:
                shots.append((start, pf))
                start = fid
        prev = (fid, dets)
    if start is not None and prev is not None:
        shots.append((start, prev[0]))
    return shots


# --- cleaning --------------------------------------------------------------


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def _smooth_segment(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``(n, ...)`` along axis 0 with edge reflection."""
    if sigma <= 0 or len(x) < 3:
        return x
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    pad = (
        np.concatenate([x[1 : r + 1][::-1], x, x[-r - 1 : -1][::-1]], axis=0)
        if len(x) > r + 1
        else np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    )
    flat = pad.reshape(len(pad), -1)
    out = np.stack([np.convolve(flat[:, i], k, mode="valid") for i in range(flat.shape[1])], axis=1)
    return out.reshape(x.shape)


def _median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    stacked = np.stack([x[:-2], x[1:-1], x[2:]])
    out = x.copy()
    out[1:-1] = np.median(stacked, axis=0)
    return out


def _clean_track(kp: np.ndarray, valid: np.ndarray, opts: Options) -> list:
    """Return ``[(start, positions)]`` segments: gaps filled, spikes removed, smoothed.

    ``kp`` is ``(T, 77, 3)`` with NaN for dropped joints; ``valid`` is ``(T,)``.
    """
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    # Split at gaps longer than max_gap.
    runs, run = [], [idx[0]]
    for a, b in pairwise(idx):
        if b - a - 1 > opts.max_gap:
            runs.append(run)
            run = []
        run.append(b)
    runs.append(run)
    segments = []
    for run in runs:
        s, e = run[0], run[-1] + 1
        if e - s < opts.min_len:
            continue
        seg = kp[s:e].copy()
        t = np.arange(e - s)
        for j in range(NUM_JOINTS):
            for c in range(3):
                col = seg[:, j, c]
                ok = np.isfinite(col)
                if ok.all():
                    continue
                if ok.sum() == 0:
                    seg[:, j, c] = np.nan
                else:
                    seg[:, j, c] = np.interp(t, t[ok], col[ok])
        seg = _median3(seg)
        seg = _smooth_segment(seg, opts.smooth_sigma)
        segments.append((s, seg))
    return segments


# --- geometry --------------------------------------------------------------


def _align_up(up: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``up`` to +Y (Rodrigues)."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(np.dot(up, y))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def _ground_alignment(feet: np.ndarray, *, min_spread: float = 0.7, max_tilt_deg: float = 20.0):
    """Rotation that levels the ground plane fitted to foot points, or ``None``.

    ``feet`` is ``(N, 3)`` in an approximately y-up frame. The plane ``y = a + b x + c z``
    is fitted to the lower envelope (iteratively keeping the lowest 40%, so people
    mid-jump or mistracked do not pull it up). A tilt axis is only fitted when the
    feet are spread over more than ``min_spread`` metres along it; otherwise the
    plane is unconstrained there and the caller's up estimate stands.
    """
    feet = feet[np.isfinite(feet).all(axis=1)]
    if len(feet) < 200:
        return None
    if len(feet) > 20000:
        feet = feet[np.random.default_rng(0).choice(len(feet), 20000, replace=False)]
    use_x, use_z = feet[:, 0].std() > min_spread, feet[:, 2].std() > min_spread
    if not (use_x or use_z):
        return None
    cols = [np.ones(len(feet))] + ([feet[:, 0]] if use_x else []) + ([feet[:, 2]] if use_z else [])
    A = np.stack(cols, axis=1)
    keep = np.ones(len(feet), bool)
    coef = np.zeros(A.shape[1])
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A[keep], feet[keep, 1], rcond=None)
        resid = feet[:, 1] - A @ coef
        keep = resid <= np.percentile(resid, 40)
    b = coef[1] if use_x else 0.0
    c = coef[-1] if use_z else 0.0
    if np.degrees(np.arctan(np.hypot(b, c))) > max_tilt_deg:
        return None
    return _align_up(np.array([-b, 1.0, -c]) / np.linalg.norm([-b, 1.0, -c]))


def _resolve_sides(world: np.ndarray) -> tuple:
    """Decide whether side A is the person's left. ``world`` is ``(n, 77, 3)``, y-up.

    The person's forward is where their toes point; with up = +Y their left is
    ``up x forward``. Returns ``(a_is_left, confident)``.
    """
    fwd = np.nanmedian(
        np.concatenate([world[:, 71] - world[:, 69], world[:, 76] - world[:, 74]]), axis=0
    )
    fwd[1] = 0.0
    lat = np.nanmedian(world[:, 12] - world[:, 40], axis=0)
    lat[1] = 0.0
    n = np.linalg.norm(fwd)
    if not np.isfinite(n) or n < 0.03 or not np.isfinite(lat).all():
        return (
            True,
            False,
        )  # fall back to NVIDIA's diagram: A is the viewer's right of a frontal figure
    left = np.cross([0.0, 1.0, 0.0], fwd / n)
    return bool(np.dot(lat, left) > 0), True


def _canon(world: np.ndarray, a_is_left: bool) -> np.ndarray:
    out = np.empty((len(world), len(CANON), 3))
    for i, src in enumerate(canon_sources(a_is_left)):
        out[:, i] = world[:, list(src)].mean(axis=1) if isinstance(src, tuple) else world[:, src]
    return out


def _body_unit(canon: np.ndarray) -> float:
    """Body scale relative to a 1.75 m adult, from leg length (robust to crouching)."""
    legs = []
    for side in ("L", "R"):
        hip, knee, ankle = (canon[:, CANON_INDEX[f"{side}_{j}"]] for j in ("Hip", "Knee", "Ankle"))
        legs.append(np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1))
    leg = np.nanmedian(np.concatenate(legs))
    return round(float(np.clip(leg / 0.80, 0.5, 1.8)), 3) if np.isfinite(leg) else 1.0


# --- build -----------------------------------------------------------------


def build_stage(frames: list, opts: Options | None = None, *, video: dict | None = None) -> Stage:
    """Build a :class:`Stage` from :func:`load_frames` output."""
    opts = opts or Options()
    shots_meta, chunks, offset = [], [], 0
    cam_flip = np.diag(
        [1.0, -1.0, -1.0]
    )  # OpenCV camera (y down, z forward) -> y up, camera looks -z
    for f0, f1 in detect_shots(frames, opts):
        n = f1 - f0 + 1
        by_id: dict = {}
        for fid, dets in frames:
            if fid < f0 or fid > f1:
                continue
            for d in dets:
                kp = np.asarray(d["keypoints_3d"], np.float64).reshape(NUM_JOINTS, 3)
                conf = np.asarray(d["keypoints_confidence"], np.float64)
                kp = np.where(
                    (conf > opts.min_conf)[:, None] & np.isfinite(kp).all(axis=1, keepdims=True),
                    kp,
                    np.nan,
                )
                # depth must be in front of the camera
                kp[kp[:, 2] <= 0.05] = np.nan
                by_id.setdefault(d["tracking_id"], {})[fid - f0] = kp
        tracks = []
        for tid, per in by_id.items():
            kp = np.full((n, NUM_JOINTS, 3), np.nan)
            for t, v in per.items():
                kp[t] = v
            valid = np.isfinite(kp[:, list(CORE)]).all(axis=(1, 2))
            for s, seg in _clean_track(kp, valid, opts):
                tracks.append((tid, s, seg))
        if not tracks:
            continue
        # people sorted by how long they stay, capped
        tracks.sort(key=lambda x: -len(x[2]))
        tracks = tracks[: opts.max_people]

        # y-up world, camera at the origin looking down -z
        world = [(tid, s, seg @ cam_flip.T) for tid, s, seg in tracks]
        spines = np.concatenate([w[:, 3] - w[:, 0] for _, _, w in world])
        spines = spines[np.isfinite(spines).all(axis=1)]
        up = (
            np.median(spines / np.linalg.norm(spines, axis=1, keepdims=True), axis=0)
            if len(spines)
            else np.array([0, 1.0, 0])
        )
        up /= np.linalg.norm(up)
        R = _align_up(up)
        world = [(tid, s, w @ R.T) for tid, s, w in world]
        # Spines lean and crouch; feet standing on one floor are a better "up" when spread out.
        R2 = _ground_alignment(
            np.concatenate([w[:, list(FOOT_JOINTS)].reshape(-1, 3) for _, _, w in world])
        )
        if R2 is not None:
            R = R2 @ R
            world = [(tid, s, w @ R2.T) for tid, s, w in world]

        # floor: low percentile of sole height pooled over people in a +-1.5 s window
        samples = [[] for _ in range(n)]
        for _, s, w in world:
            h = np.nanmin(w[:, list(FOOT_JOINTS), 1], axis=1)
            for t, v in enumerate(h):
                if np.isfinite(v):
                    samples[s + t].append(v)
        win = int(1.5 * opts.fps)
        floor = np.full(n, np.nan)
        for t in range(n):
            pool = [v for k in range(max(0, t - win), min(n, t + win + 1)) for v in samples[k]]
            if pool:
                floor[t] = np.percentile(pool, 8)
        ok = np.isfinite(floor)
        if not ok.any():
            continue
        floor = np.interp(np.arange(n), np.flatnonzero(ok), floor[ok])
        floor = _smooth_segment(floor[:, None], 8.0)[:, 0]

        all_pts = np.concatenate([w[:, 0] for _, _, w in world])
        center = np.nanmedian(all_pts, axis=0)
        center[1] = 0.0

        track_meta = []
        for tid, s, w in world:
            a_is_left, confident = _resolve_sides(w)
            canon = _canon(w, a_is_left)
            canon = canon - np.array([0, 1, 0]) * floor[s : s + len(w), None, None] - center
            q = np.round(canon * 1000.0)
            q = np.clip(q, -32767, 32767)
            q[~np.isfinite(canon)] = MISSING
            data = q.astype("<i2").tobytes()
            head = canon[:, CANON_INDEX["HeadTop"], 1]
            height = float(np.nanmedian(head)) if np.isfinite(head).any() else 1.7
            track_meta.append(
                {
                    "id": int(tid),
                    "start": int(s),
                    "n": len(w),
                    "offset": offset,
                    "height": round(height, 3),
                    "unit": _body_unit(canon),
                    "a_is_left": a_is_left,
                    "sides_confident": confident,
                }
            )
            chunks.append(data)
            offset += len(data)
        mean_floor = float(np.mean(floor))
        shots_meta.append(
            {
                "frame0": int(f0),
                "n": int(n),
                "center": [round(float(c), 3) for c in center],
                "camera": {
                    "position": [0.0, round(-mean_floor, 3), 0.0],
                    "direction": [round(float(c), 4) for c in (R @ np.array([0.0, 0.0, -1.0]))],
                },
                "tracks": track_meta,
            }
        )
    meta = {
        "version": VERSION,
        "fps": opts.fps,
        "joints": list(CANON),
        "video": video or {},
        "shots": shots_meta,
    }
    return Stage(meta, b"".join(chunks))
