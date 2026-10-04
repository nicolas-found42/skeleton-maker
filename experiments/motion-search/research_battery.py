# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Reproducible, bounded Jev experiment harness for skeleton-maker's local poses.

All outputs stay in work/agent-run/experiments/. This is a research harness, not CLI code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from frozen_processing import Options, build_stage, load_frames

from skeleton_maker.nova77 import CANON, CANON_INDEX, canon_sources

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "agent-run" / "experiments"
MODEL = "typesafe/jev-1.13"
BASE_URL = "https://openrouter.ai/api"
FPS = 30.0
WINDOW_FRAMES = 60
QUESTION_BATCH = 40
MAX_WORKERS = 4
SEED = 4172026
PROMPT_VERSION = "real-pose-battery-0.2"
RUN_SUFFIX = "v1"

QUERY_SPECS = [
    {
        "id": "hands_over_head",
        "query": "The person's wrists stay above the top of their head for most of this window.",
        "baseline": "fraction of frames with both wrists above HeadTop exceeds 0.5",
    },
    {
        "id": "arms_raised",
        "query": "The person repeatedly raises and lowers one or both arms.",
        "baseline": "wrist vertical range exceeds 0.7 torso lengths and has at least two large direction reversals",
    },
    {
        "id": "arms_symmetric",
        "query": "Both arms move upward together, with the two wrists rising at similar times.",
        "baseline": "the two wrist vertical trajectories correlate above 0.7 and each range exceeds 0.35 torso lengths",
    },
    {
        "id": "deep_knee_bend",
        "query": "The person holds a deep knee bend or squat for much of the window.",
        "baseline": "fraction with average hip-knee-ankle angle below 120 degrees exceeds 0.45",
    },
    {
        "id": "body_bounce",
        "query": "The person's hips repeatedly move up and down while the torso stays mostly upright.",
        "baseline": "hip vertical trace has at least two large direction reversals and median trunk lean is below 25 degrees",
    },
    {
        "id": "mostly_still",
        "query": "The person's body stays nearly still throughout the window.",
        "baseline": "median root speed is below 0.1 body scales per second and median joint speed below 0.12",
    },
    {
        "id": "turning",
        "query": "The person's torso turns noticeably from side to side during the window.",
        "baseline": "shoulder-axis horizontal orientation spans at least 35 degrees",
    },
    {
        "id": "traveling_step",
        "query": "The person takes a large step to one side relative to their own body.",
        "baseline": "one ankle moves laterally by more than 0.4 torso lengths relative to Hips",
    },
    {
        "id": "clapping_proxy",
        "query": "The person repeatedly brings their hands together and moves them apart.",
        "baseline": "wrist distance crosses from above to below 0.35 torso lengths at least twice",
    },
    {
        "id": "red_cup",
        "query": "The person drinks from a red cup during this window.",
        "baseline": "not observable from 3D joint positions; correct retrieval behavior is no evidence / abstain",
    },
]
PARAPHRASES = {
    "hands_over_head": [
        "Most of the time, both wrists are higher than the person's head.",
        "The person's hands remain raised above the head for the majority of this interval.",
    ],
    "arms_raised": [
        "One or both arms go up and down more than once during this interval.",
        "The wrists show repeated vertical raising and lowering.",
    ],
    "arms_symmetric": [
        "The wrists on both sides rise at roughly the same time.",
        "Both arms lift together with similar timing.",
    ],
    "deep_knee_bend": [
        "The person spends much of the interval in a deep squat-like leg bend.",
        "For much of this window the knees are strongly flexed.",
    ],
    "body_bounce": [
        "The hips bob up and down repeatedly while the torso remains rather upright.",
        "Repeated vertical bouncing is visible in the body's root.",
    ],
    "mostly_still": [
        "There is almost no body movement in this interval.",
        "The person remains close to the same pose and location.",
    ],
    "turning": [
        "The shoulders rotate back and forth by a noticeable amount.",
        "The person's upper body turns from one side toward the other.",
    ],
    "traveling_step": [
        "At least one foot moves a substantial distance sideways relative to the pelvis.",
        "The person makes a large lateral step.",
    ],
    "clapping_proxy": [
        "Both wrists repeatedly approach one another and then separate.",
        "The hands come together and move apart more than once.",
    ],
    "red_cup": [
        "A red drinking vessel is brought to the person's mouth.",
        "The person takes a drink from a red cup.",
    ],
}


def jhash(value: Any) -> str:
    b = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(b).hexdigest()


def safe_float(v: Any) -> float | None:
    x = float(v)
    return x if math.isfinite(x) else None


def vstats(a: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(a, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
            "std": None,
        }
    return {
        "mean": safe_float(np.mean(x)),
        "median": safe_float(np.median(x)),
        "min": safe_float(np.min(x)),
        "max": safe_float(np.max(x)),
        "p10": safe_float(np.percentile(x, 10)),
        "p90": safe_float(np.percentile(x, 90)),
        "std": safe_float(np.std(x)),
    }


def nanmean_axis(a: np.ndarray, axis: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    count = np.isfinite(a).sum(axis=axis)
    total = np.nansum(a, axis=axis)
    out = np.full(np.shape(total), np.nan, dtype=float)
    return np.divide(total, count, out=out, where=count > 0)


def finite_fraction(a: np.ndarray) -> float:
    return (
        float(np.isfinite(a).all(axis=-1).mean())
        if np.asarray(a).ndim > 1
        else float(np.isfinite(a).mean())
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    cos = np.divide(dot, den, out=np.full_like(dot, np.nan, dtype=float), where=den > 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def velocities(p: np.ndarray, source_frames: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Difference only adjacent source frames; never bridge a missing-frame gap."""
    if len(p) < 2:
        return np.empty((0, p.shape[1], 3), dtype=float)
    dtf = np.diff(source_frames)
    delta = np.diff(p, axis=0)
    # Explicit frame-axis mask avoids turning gaps into high-speed observations.
    delta[dtf != 1] = np.nan
    return delta * fps


def unwrap_finite_runs(x: np.ndarray) -> np.ndarray:
    """Unwrap angular observations only within contiguous finite runs."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    ids = np.flatnonzero(np.isfinite(x))
    if not len(ids):
        return out
    for run in np.split(ids, np.where(np.diff(ids) > 1)[0] + 1):
        out[run] = np.unwrap(x[run])
    return out


def periodicity(x: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 20:
        return {"best_lag_frames": None, "best_corr": None, "reversals": 0}
    # Interpolate short missing samples only for this compact autocorrelation feature.
    xi = np.interp(np.arange(len(x)), np.flatnonzero(mask), x[mask])
    xi = xi - np.convolve(xi, np.ones(15) / 15, mode="same")
    xi = xi / (np.std(xi) + 1e-9)
    corr = []
    for lag in range(6, min(31, len(xi) // 2 + 1)):
        corr.append((float(np.mean(xi[:-lag] * xi[lag:])), lag))
    best, lag = max(corr, default=(None, None))
    d = np.diff(xi)
    nz = d[np.abs(d) > 0.08]
    reversals = int(np.sum(nz[1:] * nz[:-1] < 0)) if len(nz) > 1 else 0
    return {
        "best_lag_frames": lag,
        "best_corr": safe_float(best) if best is not None else None,
        "reversals": reversals,
    }


def summarize(
    p: np.ndarray, conf: np.ndarray | None, frames: np.ndarray, coord: str, sides_confident: bool
) -> dict[str, Any]:
    """Summarize a track window; coordinates retain explicit camera-vs-stage conventions."""
    p = np.asarray(p, dtype=float).copy()
    if coord == "raw_camera_y_down":
        p[:, :, 1] *= -1  # expose y-up features, while storing original convention in metadata
    ci = CANON_INDEX
    names = [
        "Hips",
        "HeadTop",
        "Chest",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
        "L_Hip",
        "R_Hip",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
        "L_Heel",
        "R_Heel",
        "L_Toe",
        "R_Toe",
    ]
    at = {n: p[:, ci[n], :] for n in names}
    hip = at["Hips"]
    torso = np.linalg.norm(at["HeadTop"] - at["Hips"], axis=1)
    torso_scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else float("nan")
    scale = max(torso_scale, 0.05) if math.isfinite(torso_scale) else 1.0
    rel = {n: (v - hip) / scale for n, v in at.items()}
    vel = velocities(p, frames)
    speed = np.linalg.norm(vel, axis=-1) / scale
    # Angles at the knee use hip-knee-ankle; missing values remain missing.
    knee_angle = []
    for side in ("L", "R"):
        knee_angle.append(
            angle_deg(
                at[f"{side}_Hip"] - at[f"{side}_Knee"], at[f"{side}_Ankle"] - at[f"{side}_Knee"]
            )
        )
    knee_angle_arr = nanmean_axis(np.stack(knee_angle), axis=0)
    trunk = at["HeadTop"] - at["Hips"]
    trunk_lean = np.degrees(
        np.arctan2(np.linalg.norm(trunk[:, [0, 2]], axis=1), np.abs(trunk[:, 1]))
    )
    wrist_y = np.stack([rel["L_Wrist"][:, 1], rel["R_Wrist"][:, 1]], axis=1)
    head_y = rel["HeadTop"][:, 1]
    np.stack([rel["L_Elbow"][:, 1], rel["R_Elbow"][:, 1]], axis=1)
    above_head = wrist_y > (head_y[:, None] - 0.08)
    wrists_distance = np.linalg.norm(at["L_Wrist"] - at["R_Wrist"], axis=1) / scale
    # Hips relative to the window median; rel[Hips] would be identically zero.
    hip_y = (hip[:, 1] - np.nanmedian(hip[:, 1])) / scale
    shoulder_axis = at["R_Shoulder"] - at["L_Shoulder"]
    shoulder_yaw = np.degrees(np.arctan2(shoulder_axis[:, 2], shoulder_axis[:, 0]))
    # Root-relative foot position features distinguish a step from camera/body translation.
    lateral = np.stack([rel["L_Ankle"][:, 0], rel["R_Ankle"][:, 0]], axis=1)
    # Stability and jumps use normalized velocities. A frame gap contributes no velocity.
    root_speed = (
        np.linalg.norm(vel[:, ci["Hips"], :], axis=-1) / scale if len(vel) else np.array([])
    )
    joint_speed = speed.reshape(-1)
    # Robust within-window bone ratios; large variance can indicate a fit/input failure.
    bone_ratios = []
    for side in ("L", "R"):
        for proximal, distal in (
            ("Hip", "Knee"),
            ("Knee", "Ankle"),
            ("Shoulder", "Elbow"),
            ("Elbow", "Wrist"),
        ):
            # Shoulder/Elbow/Wrist names are available in CANON but omitted from `at` above.
            a, b = p[:, ci[f"{side}_{proximal}"]], p[:, ci[f"{side}_{distal}"]]
            bone_ratios.append(np.linalg.norm(a - b, axis=1) / scale)
    bones = np.stack(bone_ratios, axis=1) if bone_ratios else np.empty((len(p), 0))
    gaps = np.diff(frames)
    valid = np.isfinite(p).all(axis=-1)
    core_ix = [ci["Hips"], ci["Chest"], ci["HeadTop"]]
    core_valid = np.isfinite(p[:, core_ix, :]).all(axis=(1, 2))
    missing_ids = np.flatnonzero(~core_valid)
    max_core_missing_run = 0
    if len(missing_ids):
        runs = np.split(missing_ids, np.where(np.diff(missing_ids) > 1)[0] + 1)
        max_core_missing_run = max(map(len, runs))
    conf_info = None
    if conf is not None:
        conf = np.asarray(conf, dtype=float)
        conf_info = {
            "all_canonical_mean": safe_float(nanmean_axis(conf, axis=None)),
            "core_mean": safe_float(
                nanmean_axis(conf[:, [ci["Hips"], ci["Chest"], ci["HeadTop"]]], axis=None)
            ),
            "arms_mean": safe_float(
                nanmean_axis(
                    conf[:, [ci["L_Wrist"], ci["R_Wrist"], ci["L_Elbow"], ci["R_Elbow"]]], axis=None
                )
            ),
            "shoulders_mean": safe_float(
                nanmean_axis(conf[:, [ci["L_Shoulder"], ci["R_Shoulder"]]], axis=None)
            ),
        }
    return {
        "coordinate_convention": coord,
        "derived_vertical_sign": "raw camera y is negated for y-up relative feature formulas"
        if coord == "raw_camera_y_down"
        else "stage y-up; floor adjusted; shot center restored",
        "body_scale_headtop_to_hips_median_m": safe_float(torso_scale),
        "valid_fraction_by_joint": {n: finite_fraction(p[:, ci[n], :]) for n in names},
        "missing_fraction_all_canonical": float(1 - valid.mean()),
        "core_frame_valid_fraction": float(core_valid.mean()),
        "max_core_missing_run_frames": int(max_core_missing_run),
        "confidence": conf_info,
        "side_resolution_confident": bool(sides_confident),
        "source_frame_gaps": {
            "count": int(np.sum(gaps != 1)),
            "largest": int(gaps.max()) if len(gaps) else 1,
        },
        "wrist_height_above_head_fraction_by_side": {
            "left": safe_float(np.nanmean(above_head[:, 0])),
            "right": safe_float(np.nanmean(above_head[:, 1])),
        },
        "wrist_height_relative_to_hips": {
            "left": vstats(rel["L_Wrist"][:, 1]),
            "right": vstats(rel["R_Wrist"][:, 1]),
        },
        "wrist_to_head_y_fraction": vstats((wrist_y - head_y[:, None]).reshape(-1)),
        "wrist_distance_body_scales": vstats(wrists_distance),
        "hip_knee_ankle_angle_deg": vstats(knee_angle_arr),
        "hip_vertical_body_scale_about_window_median": vstats(hip_y),
        "root_speed_body_scales_s": vstats(root_speed),
        "joint_speed_body_scales_s": vstats(joint_speed),
        "trunk_lean_deg": vstats(trunk_lean),
        "ankle_lateral_relative_body_scale": {
            "left": vstats(lateral[:, 0]),
            "right": vstats(lateral[:, 1]),
        },
        "shoulder_axis_yaw_deg": vstats(
            unwrap_finite_runs(np.radians(shoulder_yaw)) * (180 / np.pi)
        ),
        "left_wrist_vertical_periodicity": periodicity(rel["L_Wrist"][:, 1]),
        "right_wrist_vertical_periodicity": periodicity(rel["R_Wrist"][:, 1]),
        "hip_vertical_periodicity": periodicity(hip_y),
        "bone_length_ratio_to_torso": {
            "all": vstats(bones),
            "per_bone": [vstats(col) for col in bones.T],
        },
    }


def feature_levels(f: dict[str, Any]) -> dict[str, Any]:
    """Digest ablations: D0 pose geometry; D1 positions; D2 dynamics; D3 evidence quality."""
    qkeys = [
        "valid_fraction_by_joint",
        "missing_fraction_all_canonical",
        "core_frame_valid_fraction",
        "max_core_missing_run_frames",
        "confidence",
        "side_resolution_confident",
        "source_frame_gaps",
    ]
    position = [
        "wrist_height_above_head_fraction_by_side",
        "wrist_height_relative_to_hips",
        "wrist_to_head_y_fraction",
        "wrist_distance_body_scales",
        "hip_knee_ankle_angle_deg",
        "hip_vertical_body_scale_about_window_median",
        "trunk_lean_deg",
        "ankle_lateral_relative_body_scale",
        "shoulder_axis_yaw_deg",
        "bone_length_ratio_to_torso",
    ]
    dynamics = [
        "root_speed_body_scales_s",
        "joint_speed_body_scales_s",
        "left_wrist_vertical_periodicity",
        "right_wrist_vertical_periodicity",
        "hip_vertical_periodicity",
    ]
    D0 = {
        k: f[k]
        for k in (
            "body_scale_headtop_to_hips_median_m",
            "wrist_height_above_head_fraction_by_side",
            "hip_knee_ankle_angle_deg",
            "trunk_lean_deg",
            "wrist_distance_body_scales",
            "bone_length_ratio_to_torso",
        )
    }
    D1 = {**D0, **{k: f[k] for k in position}}
    D2 = {**D1, **{k: f[k] for k in dynamics}}
    D3 = {**D2, **{k: f[k] for k in qkeys}}
    return {"D0_geometry": D0, "D1_positions": D1, "D2_dynamics": D2, "D3_quality": D3}


def load_detection_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[int(row["frame_id"])] = row["detections"]
    return out


def raw_track_arrays(
    raw_by_frame: dict[int, list[dict[str, Any]]], track_id: int, frames: list[int], a_is_left: bool
) -> tuple[np.ndarray, np.ndarray]:
    srcs = canon_sources(a_is_left)
    p = np.full((len(frames), len(CANON), 3), np.nan)
    c = np.full((len(frames), len(CANON)), np.nan)
    for i, fid in enumerate(frames):
        ds = raw_by_frame.get(int(fid), [])
        det = next((d for d in ds if int(d["tracking_id"]) == track_id), None)
        if det is None:
            continue
        points = np.asarray(det["keypoints_3d"], dtype=float).reshape(77, 3)
        conf = np.asarray(det.get("keypoints_confidence", np.ones(77)), dtype=float).reshape(77)
        for j, source in enumerate(srcs):
            ids = list(source) if isinstance(source, tuple) else [source]
            vals = points[ids]
            ok = (conf[ids] > 0) & np.isfinite(vals).all(axis=1)
            if ok.any():
                p[i, j] = vals[ok].mean(axis=0)
                c[i, j] = float(np.mean(conf[ids][ok]))
    return p, c


def build_windows(limit_per_clip: int = 12, fps: float = FPS) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for path in sorted((ROOT / "in").glob("*.skeleton/pose.json")):
        frames = load_frames(str(path))
        # Reuse the parsed line records: these files are large and a second JSON parse costs time.
        raw_by_frame = dict(frames)
        stage = build_stage(frames, Options(fps=fps))
        track_rows: list[tuple[int, int, int, int]] = []
        for si, shot in enumerate(stage.meta["shots"]):
            for ti, t in enumerate(shot["tracks"]):
                n = int(t["n"])
                if n >= WINDOW_FRAMES:
                    track_rows.append((si, ti, n, int(t["id"])))
        # Select from longer segments but distribute across people. One pass gives each selected
        # track one observation; further passes add time-separated observations if a clip has
        # fewer distinct people than the per-clip window budget.
        track_rows.sort(key=lambda x: (-x[2], x[0], x[1]))
        groups: dict[tuple[int, int], list[int]] = {}
        max_slots = max(1, math.ceil(limit_per_clip / max(1, len(track_rows))))
        for si, ti, _n, _ in track_rows:
            max_start = n - WINDOW_FRAMES
            slots = min(max_slots, limit_per_clip)
            if slots == 1 and max_start > 0:
                salt = hashlib.sha256(f"{path.parent.name}:{si}:{ti}".encode()).digest()
                starts = [int.from_bytes(salt[:4], "big") % (max_start + 1)]
            else:
                starts = [round(i * max_start / max(1, slots - 1)) for i in range(slots)]
            groups[(si, ti)] = sorted(set(starts))
        selected_indexes: list[tuple[int, int, int]] = []
        depth = 0
        while len(selected_indexes) < limit_per_clip:
            moved = False
            for si, ti, _n, _ in track_rows:
                starts = groups[(si, ti)]
                if depth < len(starts):
                    selected_indexes.append((si, ti, starts[depth]))
                    moved = True
                    if len(selected_indexes) >= limit_per_clip:
                        break
            if not moved:
                break
            depth += 1
        selected: list[dict[str, Any]] = []
        for si, ti, local_start in selected_indexes:
            shot = stage.meta["shots"][si]
            t = shot["tracks"][ti]
            n = int(t["n"])
            take = WINDOW_FRAMES
            frame0 = int(shot["frame0"]) + int(t["start"])
            track_frames = np.arange(frame0 + local_start, frame0 + local_start + take, dtype=int)
            pos_stage = stage.positions(si, ti)[local_start : local_start + take]
            pos_raw, confidence = raw_track_arrays(
                raw_by_frame, int(t["id"]), track_frames, bool(t["a_is_left"])
            )
            raw_features = summarize(
                pos_raw, confidence, track_frames, "raw_camera_y_down", bool(t["sides_confident"])
            )
            stage_features = summarize(
                pos_stage,
                confidence,
                track_frames,
                "stage_world_y_up_floor_adjusted",
                bool(t["sides_confident"]),
            )
            raw_levels, stage_levels = feature_levels(raw_features), feature_levels(stage_features)
            raw_levels["D3_quality"]["raw_headtop_hips_length_m"] = raw_features[
                "body_scale_headtop_to_hips_median_m"
            ]
            stage_levels["D3_quality"]["inferred_body_height_m"] = safe_float(
                t.get("height", float("nan"))
            )
            stage_levels["D3_quality"]["body_unit"] = safe_float(t.get("unit", float("nan")))
            rec = {
                "source_path_relative": str(path.relative_to(ROOT)),
                "source_track_id": int(t["id"]),
                "clip_blind_id": "clip_"
                + hashlib.sha256(path.parent.name.encode()).hexdigest()[:10],
                "track_blind_id": "track_"
                + hashlib.sha256(f"{path.parent.name}:{t['id']}".encode()).hexdigest()[:10],
                "shot_index": si,
                "source_frame_start": int(track_frames[0]),
                "source_frame_end": int(track_frames[-1]),
                "duration_s": (take - 1) / fps,
                "fps": fps,
                "n_frames": take,
                "window_sampling": "60 consecutive source frames, duration-stratified tracks, evenly-spaced starts; no gaps crossed",
                "stage_track_metadata": {
                    "n": n,
                    "start_in_shot": int(t["start"]),
                    "body_unit": safe_float(t.get("unit", float("nan"))),
                    "height_median_m": safe_float(t.get("height", float("nan"))),
                    "sides_confident": bool(t["sides_confident"]),
                },
                "raw_camera_y_down": raw_levels,
                "stage_world_y_up_floor_adjusted": stage_levels,
                "_full_raw_features": raw_features,
                "_full_stage_features": stage_features,
            }
            rec["window_sha256"] = jhash({k: v for k, v in rec.items() if not k.startswith("_")})
            selected.append(rec)
        clips.append(
            {
                "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "clip_name_for_report_only": path.parent.name,
                "raw_frame_count": len(frames),
                "shot_count": len(stage.meta["shots"]),
                "available_track_count": sum(len(s["tracks"]) for s in stage.meta["shots"]),
                "candidate_window_count": len(track_rows),
                "selected_window_count": len(selected),
                "windows": selected,
            }
        )
    return clips


def flatten(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [w for c in clips for w in c["windows"]]


def mc_label(w: dict[str, Any], q: dict[str, Any], view: str, digest_level: str) -> dict[str, Any]:
    """Mechanical reference rules over selected features; explicitly not human ground truth."""
    d = w[view][digest_level]
    refs: dict[str, bool | None] = {}
    both = d.get("wrist_height_above_head_fraction_by_side", {})
    wrist_range = d.get("wrist_height_relative_to_hips", {})
    knee = d.get("hip_knee_ankle_angle_deg", {}).get("median")
    rs = d.get("root_speed_body_scales_s", {}).get("median")
    hp = d.get("hip_vertical_periodicity", {})
    lp = d.get("left_wrist_vertical_periodicity", {})
    rp = d.get("right_wrist_vertical_periodicity", {})
    wrist_max_range = max(
        [
            x.get("max") - x.get("min")
            for x in wrist_range.values()
            if x.get("max") is not None and x.get("min") is not None
        ]
        or [None]
    )
    if q["id"] == "hands_over_head":
        refs[q["id"]] = both.get("left", 0) > 0.5 and both.get("right", 0) > 0.5
    elif q["id"] == "arms_raised":
        refs[q["id"]] = bool(
            wrist_max_range is not None
            and wrist_max_range > 0.7
            and max(lp.get("reversals", 0), rp.get("reversals", 0)) >= 2
        )
    elif q["id"] == "arms_symmetric":
        refs[q["id"]] = None  # phase correlation is computed separately in the detailed digest
    elif q["id"] == "deep_knee_bend":
        refs[q["id"]] = bool(knee is not None and knee < 120)
    elif q["id"] == "body_bounce":
        refs[q["id"]] = bool(hp.get("reversals", 0) >= 2)
    elif q["id"] == "mostly_still":
        refs[q["id"]] = bool(rs is not None and rs < 0.1)
    elif q["id"] == "turning":
        y = d.get("shoulder_axis_yaw_deg", {})
        refs[q["id"]] = bool(y.get("max") is not None and y["max"] - y["min"] >= 35)
    elif q["id"] == "traveling_step":
        left = d.get("ankle_lateral_relative_body_scale", {}).get("left", {})
        r = d.get("ankle_lateral_relative_body_scale", {}).get("right", {})
        refs[q["id"]] = bool(
            max(
                (left.get("max") or 0) - (left.get("min") or 0),
                (r.get("max") or 0) - (r.get("min") or 0),
            )
            > 0.4
        )
    elif q["id"] == "clapping_proxy":
        refs[q["id"]] = (
            None  # zero-crossing count requires per-frame distance trace; not inferable here
        )
    elif q["id"] == "red_cup":
        refs[q["id"]] = None
    return {
        "mechanical_reference": refs[q["id"]],
        "reference_type": "explicit_proxy_rule_not_human_truth",
        "rule": q["baseline"],
    }


def query_questions(view: str, level: str, variant: str) -> dict[str, Any]:
    out = {}
    for q in QUERY_SPECS:
        texts = [q["query"], *PARAPHRASES[q["id"]]]
        for pi, text in enumerate(texts):
            suffix = "original" if pi == 0 else f"paraphrase_{pi}"
            qid = f"{q['id']}__{suffix}__{variant}"
            out[qid] = {
                "type": "noul",
                "instructions": {
                    "question": f"Does this motion description match the candidate window? {text}",
                    "candidate_data": "`candidate.digest`",
                },
                "criteria": {
                    "true": "The pose and motion statistics in that digest support this described behavior during the window.",
                    "false": "The digest does not support this described behavior, or relevant evidence is insufficient.",
                },
            }
    return out


ACTION_CLASSES = {
    "deep_bend": "A sustained deep bend of the knees/hips.",
    "arms_overhead": "Both arms are raised above the head for a substantial portion.",
    "repetitive_upper_body": "Repeated upper-limb or upper-body movement dominates.",
    "large_root_motion": "Large whole-body displacement or repeated vertical root movement dominates.",
    "mostly_still": "Little motion across the available joints and root.",
    "other_or_ambiguous": "None of these labels is clearly supported, or evidence is too ambiguous.",
}


def action_reference(w: dict[str, Any], view: str) -> str:
    f = w[view]["D3_quality"]
    angle = f.get("hip_knee_ankle_angle_deg", {}).get("median")
    above = f.get("wrist_height_above_head_fraction_by_side", {})
    root = f.get("root_speed_body_scales_s", {}).get("median")
    if angle is not None and angle < 120:
        return "deep_bend"
    if above.get("left", 0) > 0.5 and above.get("right", 0) > 0.5:
        return "arms_overhead"
    if root is not None and root < 0.1:
        return "mostly_still"
    return "other_or_ambiguous"


QUALITY_QUESTIONS = {
    "missing_key_data": "Does this window lack enough valid key joints to support motion interpretation?",
    "root_jump": "Does this window contain an implausibly abrupt whole-body/root jump?",
    "bone_scale_instability": "Do the bone-to-torso ratios vary implausibly within this track window?",
    "side_ambiguity": "Is the person-side left/right assignment uncertain in this evidence?",
    "body_scale_outlier": "Is the inferred body scale or height implausible for a person-sized body?",
    "temporal_gap": "Does this window contain frame gaps that make its motion sequence discontinuous?",
}


def ref_quality(w: dict[str, Any], mode: str) -> bool:
    f = w["stage_world_y_up_floor_adjusted"]["D3_quality"]
    if mode == "missing_key_data":
        return f.get("missing_fraction_all_canonical", 0) > 0.3
    if mode == "root_jump":
        # Stage-cleaned root jump is conservative; use a digest's max speed as a proxy.
        return (f.get("root_speed_body_scales_s", {}).get("max") or 0) > 3.0
    if mode == "bone_scale_instability":
        per_bone = f.get("bone_length_ratio_to_torso", {}).get("per_bone", [])
        ratios = [
            ((x.get("std") or 0) / max(x.get("median") or 0, 1e-6))
            for x in per_bone
            if x.get("median") is not None
        ]
        return max(ratios or [0]) > 0.15
    if mode == "side_ambiguity":
        return not bool(w["stage_track_metadata"]["sides_confident"])
    if mode == "body_scale_outlier":
        h = w["stage_track_metadata"].get("height_median_m")
        return h is None or h < 1.0 or h > 2.6
    if mode == "temporal_gap":
        return f.get("source_frame_gaps", {}).get("count", 0) > 0
    raise KeyError(mode)


def request_one(payload: dict[str, Any], key: str, timeout: int = 90) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(  # noqa: S310 -- fixed HTTPS endpoint
        BASE_URL + "/v1/systemone",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed HTTPS endpoint
            raw = resp.read()
            status = resp.status
        result = json.loads(raw)
        return {
            "status": "ok",
            "http_status": status,
            "response": result,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": None,
        }
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")[:2000]
        try:
            body = json.loads(text)
            # Retain diagnostics but omit opaque account identifiers from error records.
            body = {k: v for k, v in body.items() if k != "user_id"}
        except json.JSONDecodeError:
            body = {"message": text}
        return {
            "status": "http_error",
            "http_status": e.code,
            "response": None,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": body,
        }
    except Exception as e:
        return {
            "status": "client_error",
            "http_status": None,
            "response": None,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": f"{type(e).__name__}: {e}",
        }


def call(payload: dict[str, Any], experiment: str, item_id: str) -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {
            "status": "missing_key",
            "http_status": None,
            "response": None,
            "latency_ms": None,
            "error": "OPENROUTER_API_KEY was not set",
        }
    req_hash = jhash(payload)
    result = request_one(payload, key)
    row = {
        "experiment": experiment,
        "item_id": item_id,
        "model_requested": MODEL,
        "endpoint": BASE_URL + "/v1/systemone",
        "prompt_version": PROMPT_VERSION,
        "request_sha256": req_hash,
        "state_sha256": jhash(payload["state"]),
        "questions_sha256": jhash(payload["questions"]),
        **result,
    }
    return row


def write_dataset(clips: list[dict[str, Any]]) -> None:
    # Write the raw measurements and digest ablations. Source filenames/clip labels live in provenance only.
    (
        {k: v for k, v in clips[0].items() if k not in {"windows", "source_file_sha256"}}
        if clips
        else {}
    )
    out_json = {
        "schema_version": PROMPT_VERSION,
        "model": MODEL,
        "sampling_seed": SEED,
        "fps": FPS,
        "window_frames": WINDOW_FRAMES,
        "digest_schema": "D0 geometry -> D1 positions -> D2 dynamics -> D3 quality",
        "clips": clips,
    }
    path = OUT / "windows.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out_json, indent=2, sort_keys=True, allow_nan=False))
    tmp.replace(path)
    provenance = {"schema_version": PROMPT_VERSION, "sources": [], "selected_windows": []}
    for clip in clips:
        provenance["sources"].append(
            {
                "local_path": clip["windows"][0]["source_path_relative"]
                if clip["windows"]
                else None,
                "clip_label_for_human_report_only": clip["clip_name_for_report_only"],
                "source_file_sha256": clip["source_file_sha256"],
                "raw_frame_count": clip["raw_frame_count"],
                "shot_count": clip["shot_count"],
                "eligible_tracks": clip["candidate_window_count"],
                "selected_windows": clip["selected_window_count"],
            }
        )
        for w in clip["windows"]:
            provenance["selected_windows"].append(
                {
                    "local_path": w["source_path_relative"],
                    "clip_label_for_human_report_only": clip["clip_name_for_report_only"],
                    "source_track_id": w["source_track_id"],
                    "track_blind_id": w["track_blind_id"],
                    "shot_index": w["shot_index"],
                    "source_frame_start": w["source_frame_start"],
                    "source_frame_end": w["source_frame_end"],
                    "window_sha256": w["window_sha256"],
                }
            )
    ppath = OUT / "source.json"
    ptmp = ppath.with_suffix(".json.tmp")
    ptmp.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    ptmp.replace(ppath)


def build_cmd(args: argparse.Namespace) -> None:
    clips = build_windows(args.limit_per_clip)
    write_dataset(clips)
    total = sum(len(c["windows"]) for c in clips)
    print(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_report_only": c["clip_name_for_report_only"],
                        "source_file_sha256": c["source_file_sha256"],
                        "raw_frames": c["raw_frame_count"],
                        "shots": c["shot_count"],
                        "tracks": c["available_track_count"],
                        "candidate_windows": c["candidate_window_count"],
                        "selected": c["selected_window_count"],
                    }
                    for c in clips
                ],
                "total_selected_windows": total,
                "output": str(OUT / "windows.json"),
            },
            indent=2,
        )
    )


def load_windows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ds = json.loads((OUT / "windows.json").read_text())
    return ds, [w for c in ds["clips"] for w in c["windows"]]


def model_state(w: dict[str, Any], view: str, level: str) -> dict[str, Any]:
    # Each ablation call contains only one digest, preventing sibling-view leakage.
    return {
        "candidate": {
            "blind_window_id": w["window_sha256"][:14],
            "source_frames": [w["source_frame_start"], w["source_frame_end"]],
            "duration_seconds": w["duration_s"],
            "frame_rate": w["fps"],
            "representation": view,
            "digest_level": level,
            "digest": w[view][level],
        },
        "task": "Judge only from the supplied numeric pose-derived features; do not infer clothing, props, identity, or activity from video or clip names.",
    }


def run_retrieval(windows: list[dict[str, Any]], count: int | None = None) -> list[dict[str, Any]]:
    # Exactly the same six query themes and wording variants across all three isolated views.
    views = [
        ("raw_camera_y_down", "D3_quality", "raw_D3"),
        ("stage_world_y_up_floor_adjusted", "D0_geometry", "stage_D0"),
        ("stage_world_y_up_floor_adjusted", "D3_quality", "stage_D3"),
    ]
    keep = [
        "hands_over_head",
        "arms_raised",
        "deep_knee_bend",
        "body_bounce",
        "mostly_still",
        "red_cup",
    ]
    selected = windows[:count] if count else windows
    jobs = []
    for w in selected:
        for view, level, vname in views:
            qs = query_questions(view, level, vname)
            qs = {
                k: v
                for k, v in qs.items()
                if any(k.startswith(i + "__") for i in keep)
                and ("__original__" in k or "__paraphrase_1__" in k)
            }
            state = model_state(w, view, level)
            payload = {"model": MODEL, "state": state, "questions": qs}
            jobs.append(({**w, "_experiment_item_id": f"{w['window_sha256']}:{vname}"}, payload))
    return run_jobs(jobs, "retrieval_digests")


def run_action_quality(
    windows: list[dict[str, Any]], count: int | None = None
) -> list[dict[str, Any]]:
    selected = windows[:count] if count else windows
    jobs = []
    for w in selected:
        qs = {
            "action_stage": {
                "type": "choice",
                "instructions": "Which one label best describes the movement evident in `candidate.digest`? Use only that evidence; do not infer a sports/dance/cooking label.",
                "criteria": ACTION_CLASSES,
            }
        }
        atomic_actions = {
            "action_deep_bend": "Is the average hip-knee-ankle angle below 120 degrees for a substantial portion of this window?",
            "action_arms_overhead": "Are both wrists above HeadTop for at least half this window?",
            "action_repetitive_upper": "Does at least one wrist show repeated, large up/down motion in this window?",
            "action_large_root_motion": "Does root speed exceed 0.7 body scales per second or show a vertical range above 0.25 body scales?",
            "action_mostly_still": "Is median root speed below 0.1 body scales per second and median joint speed below 0.12?",
        }
        for aid, question in atomic_actions.items():
            qs[aid] = {
                "type": "noul",
                "instructions": {"question": question, "evidence": "candidate.digest"},
                "criteria": {
                    "true": "The numerical evidence meets the stated behavior description.",
                    "false": "The numerical evidence does not meet the stated behavior description, or is insufficient.",
                },
            }
        for mode, question in QUALITY_QUESTIONS.items():
            qs["quality_" + mode] = {
                "type": "noul",
                "instructions": {"question": question, "evidence": "candidate.digest"},
                "criteria": {
                    "true": "The named failure is present or the evidence is insufficient to rule it out.",
                    "false": "The evidence supports a clean observation for this failure mode.",
                },
            }
        jobs.append(
            (
                w,
                {
                    "model": MODEL,
                    "state": model_state(w, "stage_world_y_up_floor_adjusted", "D3_quality"),
                    "questions": qs,
                },
            )
        )
    return run_jobs(jobs, "action_quality")


def run_jobs(
    jobs: list[tuple[dict[str, Any], dict[str, Any]]], experiment: str
) -> list[dict[str, Any]]:
    output = OUT / f"{experiment}_{RUN_SUFFIX}.jsonl"
    done = set()
    attempts: dict[str, int] = {}
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                old = json.loads(line)
                rh = old["request_sha256"]
                attempts[rh] = max(attempts.get(rh, 0), int(old.get("attempt", 1)))
                if old.get("status") == "ok":
                    done.add(rh)
            except (json.JSONDecodeError, KeyError):
                pass
    pending = [(w, p) for w, p in jobs if jhash(p) not in done]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(call, p, experiment, w["window_sha256"]): (w, p) for w, p in pending}
        with output.open("a") as fh:
            for fut in concurrent.futures.as_completed(futures):
                w, p = futures[fut]
                result = fut.result()
                result["window_sha256"] = w["window_sha256"]
                result["experiment_item_id"] = w.get("_experiment_item_id", w["window_sha256"])
                result["attempt"] = attempts.get(jhash(p), 0) + 1
                result["question_count"] = len(p["questions"])
                result["questions"] = p["questions"]
                result["state"] = p["state"]
                fh.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                fh.flush()
                rows.append(result)
                print(
                    f"{experiment} completed={len(rows)}/{len(pending)} status={result['status']} questions={result['question_count']} latency_ms={result['latency_ms']}",
                    flush=True,
                )
    return rows


def inject(w: dict[str, Any], mode: str) -> dict[str, Any]:
    """Create explicit feature-level corrupted copies from real windows; source files stay intact."""
    clone = json.loads(json.dumps(w))
    d = clone["stage_world_y_up_floor_adjusted"]["D3_quality"]
    if mode == "missing_key_data":
        d["missing_fraction_all_canonical"] = 0.75
        d["core_frame_valid_fraction"] = 0.05
        d["max_core_missing_run_frames"] = 56
        for j in ("L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow"):
            d["valid_fraction_by_joint"][j] = 0.0
    elif mode == "root_jump":
        d["root_speed_body_scales_s"]["max"] = 12.0
    elif mode == "bone_scale_instability":
        d["bone_length_ratio_to_torso"]["all"]["std"] = 0.9
        for bone in d["bone_length_ratio_to_torso"]["per_bone"]:
            bone["std"] = max(bone.get("median") or 0.05, 0.05) * 0.8
    elif mode == "side_ambiguity":
        d["side_resolution_confident"] = False
    elif mode == "body_scale_outlier":
        d["inferred_body_height_m"] = 8.0
    elif mode == "temporal_gap":
        d["source_frame_gaps"] = {"count": 5, "largest": 12}
        d["max_core_missing_run_frames"] = 5
    else:
        raise KeyError(mode)
    clone["injected_corruption"] = {
        "mode": mode,
        "label": True,
        "source_window_sha256": w["window_sha256"],
        "note": "Only the research digest copy was altered; original pose data and stage remain unchanged.",
    }
    return clone


def run_corruption(windows: list[dict[str, Any]], n_pairs: int = 12) -> list[dict[str, Any]]:
    # Stratify samples to include each clip when available, then cycle through modes.
    chosen = windows[:: max(1, len(windows) // n_pairs)][:n_pairs]
    jobs = []
    for w in chosen:
        for mode, _q in QUALITY_QUESTIONS.items():
            corrupt = inject(w, mode)
            state = model_state(corrupt, "stage_world_y_up_floor_adjusted", "D3_quality")
            questions = {
                "detector_" + k: {
                    "type": "noul",
                    "instructions": {"question": text, "evidence": "candidate.digest"},
                    "criteria": {
                        "true": "The evidence indicates this failure is present.",
                        "false": "The evidence does not indicate this failure.",
                    },
                }
                for k, text in QUALITY_QUESTIONS.items()
            }
            jobs.append((corrupt, {"model": MODEL, "state": state, "questions": questions}))
    return run_jobs(jobs, "injected_corruption")


SPEC_CARDS = {
    "blocky": "Box-shaped limbs and torso with a small set of colored standard/basic materials; simple geometric block figure.",
    "clay": "Rounded capsule limbs and sphere props with smooth, non-metal materials and colorful toy-like palette.",
    "critter": "Animal-like figure with fur/belly materials, sphere/cone props, and chain parts for appendages.",
    "mannequin": "Wood-tone articulated human-like mannequin with capsule limbs and separate dark joints.",
    "neon": "Glowing thin cylinder limbs, bright basic materials, wireframe halo, and translucent core.",
    "robot": "Hard-surface mechanical figure with box/cylinder limbs, metallic standard materials, and glowing accents.",
}
SPEC_REQUESTS = [
    {
        "id": "blocky",
        "target": "blocky",
        "text": "I want a simple block figure with square box-like arms and legs.",
    },
    {
        "id": "clay",
        "target": "clay",
        "text": "Make the body look like a soft, rounded clay toy with smooth limbs.",
    },
    {
        "id": "critter",
        "target": "critter",
        "text": "I need an animal-like character with ears or a tail and a fur-colored body.",
    },
    {
        "id": "mannequin",
        "target": "mannequin",
        "text": "Show an articulated wooden anatomy mannequin with visible dark joints.",
    },
    {
        "id": "neon",
        "target": "neon",
        "text": "Render a luminous cyan wireframe figure with thin glowing bones.",
    },
    {
        "id": "robot",
        "target": "robot",
        "text": "Use a metallic robot character with segmented mechanical limbs and glowing details.",
    },
    {
        "id": "no_builtin_a",
        "target": "none",
        "text": "Use a four-legged horse body with hooves and a separate custom saddle mesh.",
    },
    {
        "id": "no_builtin_b",
        "target": "none",
        "text": "Create a realistic flowing cloth garment and a detailed face texture from custom meshes.",
    },
]


def run_spec_assist() -> list[dict[str, Any]]:
    # Ground truth is the authored intent-to-catalog mapping, not human user preference.
    jobs = []
    cards = [{"id": k, "description": v} for k, v in SPEC_CARDS.items()] + [
        {
            "id": "none",
            "description": "No existing built-in character fits; request is outside current built-in options.",
        }
    ]
    for request in SPEC_REQUESTS:
        questions = {
            "best_character": {
                "type": "choice",
                "instructions": "Which existing built-in character spec best fits the request? Use none if no catalog option meets its distinctive requirements.",
                "criteria": {c["id"]: c["description"] for c in cards},
            },
            "fits_existing_scope": {
                "type": "noul",
                "instructions": {
                    "question": "Can the request be fulfilled with an existing built-in character and the currently described geometry/material capabilities?",
                    "request": "`request.text`",
                    "catalog": "`existing_specs`",
                },
                "criteria": {
                    "true": "A current built-in spec meets the main visual requirements.",
                    "false": "The request needs unsupported anatomy, geometry, or materials, or no built-in option fits.",
                },
            },
            "requires_new_capability": {
                "type": "noul",
                "instructions": {
                    "question": "Does this request require a capability beyond the listed built-in spec shapes, joints, material types, and part patterns?",
                    "request": "`request.text`",
                    "catalog": "`existing_specs`",
                },
                "criteria": {
                    "true": "At least one requested feature cannot be represented with the current catalog/schema.",
                    "false": "The request is representable with listed features.",
                },
            },
        }
        state = {
            "request": request["text"],
            "existing_specs": cards,
            "task": "Select or reject a built-in option; do not author JSON or invent an unlisted spec.",
        }
        jobs.append(
            (
                {
                    "window_sha256": hashlib.sha256(request["id"].encode()).hexdigest(),
                    "_experiment_item_id": request["id"],
                    "_request_target": request["target"],
                },
                {"model": MODEL, "state": state, "questions": questions},
            )
        )
    return run_jobs(jobs, "spec_assist")


def run_generic_quality(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for w in windows:
        q = {
            "generic_unreliable": {
                "type": "noul",
                "instructions": {
                    "question": "Is this skeleton window unreliable overall for interpreting its motion? Fast or unusual valid movement alone does not make a track unreliable.",
                    "evidence": "candidate.digest",
                },
                "criteria": {
                    "true": "The joint evidence has a meaningful tracking/data integrity problem that makes motion interpretation unsafe.",
                    "false": "The motion may be energetic or unusual, but the joint evidence is sufficiently coherent for interpretation.",
                },
            }
        }
        jobs.append(
            (
                w,
                {
                    "model": MODEL,
                    "state": model_state(w, "stage_world_y_up_floor_adjusted", "D3_quality"),
                    "questions": q,
                },
            )
        )
    return run_jobs(jobs, "quality_generic")


def summarize_cmd(args: argparse.Namespace) -> None:
    file = OUT / f"{args.experiment}_{RUN_SUFFIX}.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()] if file.exists() else []
    ok = [r for r in rows if r.get("status") == "ok"]
    use = [r.get("response", {}).get("usage", {}) for r in ok]
    costs = [u.get("cost") for u in use if isinstance(u.get("cost"), (float, int))]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "requests": len(rows),
                "ok": len(ok),
                "failures": len(rows) - len(ok),
                "judgments_submitted": sum(r.get("question_count", 0) for r in ok),
                "input_tokens": sum(int(u.get("input_tokens", 0) or 0) for u in use),
                "output_tokens": sum(int(u.get("output_tokens", 0) or 0) for u in use),
                "reported_cost_usd": sum(costs) if costs else None,
                "median_latency_ms": statistics.median(latencies) if latencies else None,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
                "models_returned": sorted({r.get("response", {}).get("model") for r in ok}),
                "http_errors": {
                    str(code): sum(r.get("http_status") == code for r in rows)
                    for code in sorted({r.get("http_status") for r in rows if r.get("http_status")})
                },
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit-per-clip", type=int, default=12)
    b.set_defaults(func=build_cmd)
    for name, fn in [("retrieval", run_retrieval), ("action-quality", run_action_quality)]:
        r = sub.add_parser(name)
        r.add_argument("--count", type=int, default=None)
        r.set_defaults(func=lambda a, f=fn: (lambda ds, ws: f(ws, a.count))(*load_windows()))
    c = sub.add_parser("corruption")
    c.add_argument("--pairs", type=int, default=12)
    c.set_defaults(func=lambda a: run_corruption(load_windows()[1], a.pairs))
    sp = sub.add_parser("spec-assist")
    sp.set_defaults(func=lambda a: run_spec_assist())
    g = sub.add_parser("generic-quality")
    g.set_defaults(func=lambda a: run_generic_quality(load_windows()[1]))
    s = sub.add_parser("summary")
    s.add_argument("experiment")
    s.set_defaults(func=summarize_cmd)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
