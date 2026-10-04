# SPDX-License-Identifier: MIT
"""``skeleton-maker environment-score``: grade environment scans against human labels.

Every gate is reported on its own, with the counts behind it. There is no overall number
that a strong gate could use to cover for a failed one; ``all_gates_passed`` is true only
when every gate is. Targets are design targets fixed before any held-out evaluation:
changing one needs a spec change, not a code tweak.
"""

import json
import math
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from . import artifacts, envmanifest
from .envannotations import (
    COUNTABLE_FAMILIES,
    STRUCTURAL,
    AnnotationError,
    canonical,
    load_annotation,
    load_mask,
    read_mask,
)
from .envmanifest import ManifestError
from .path_safety import is_same_or_ancestor, same_path
from .utils import die

SCORER_VERSION = "1"
SCORE_SCHEMA = "skeleton-maker.environment-score/1"

TARGETS: dict[str, Any] = {
    "mask_iou": 0.50,
    "object_precision": 0.80,
    "object_recall": 0.70,
    "subtype_min_instances": 10,
    "subtype_recall": 0.50,
    "structural_iou": 0.65,
    "structural_clip_iou": 0.50,
    "negative_area_fraction": 0.01,
    "negative_false_positive_rate": 0.05,
    "idf1": 0.75,
    "registration_coverage": 0.80,
    "reprojection_median_pct": 0.5,
    "reprojection_p95_pct": 2.0,
    "metric_relative_error": 0.10,
    "ignore_overlap": 0.50,
    "corpus": {
        "clips": 12,
        "development_clips": 6,
        "heldout_clips": 6,
        "heldout_frames_per_clip": 10,
        "tracking_interval_seconds": 2.0,
        "family_positive_heldout_clips": 2,
        "object_subtypes": 6,
        "vehicle_subtypes": 3,
        "translating_registration_clips": 2,
    },
    "required_tags": [
        "indoor",
        "outdoor",
        "vehicle_stationary",
        "vehicle_moving",
        "handled_object",
        "tripod",
        "translating",
        "pan",
        "occlusion",
        "motion_blur",
        "person_free",
        "edited_cut",
        "underconstrained",
    ],
}

EXIT_FAILED_GATES = 1
EXIT_INVALID = 2
HONEST_GEOMETRY = ("not_requested", "unavailable", "relative", "relative-camera-frame")
REGISTERED_GEOMETRY = ("registered_relative", "registered_metric")


# --- loading -------------------------------------------------------------------------------


def _json_files(directory: str, what: str) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        die(f"no such directory: {directory} ({what})", EXIT_INVALID)
    return sorted(p for p in path.glob("*.json") if p.is_file())


def _validate_score_output_path(output: str, predictions: str, annotations: str) -> None:
    """Reject output paths that could replace or enter either scorer input corpus."""
    target = Path(output).resolve()
    roots = [Path(predictions), Path(annotations)]
    for root in roots:
        if is_same_or_ancestor(root, target):
            die(f"--out must be outside scorer input directories: {root}", EXIT_INVALID)
    if not target.exists():
        return
    for root in roots:
        for source in root.rglob("*"):
            try:
                if source.is_file() and same_path(target, source):
                    die(f"--out aliases scorer input {source}", EXIT_INVALID)
            except OSError:
                # A concurrent removal or inaccessible unrelated file cannot turn an
                # otherwise-contained output into a destructive input alias.
                continue


def _load_corpus(predictions: str, annotations: str):
    pred_files = _json_files(predictions, "predictions")
    ann_files = _json_files(annotations, "annotations")
    if not ann_files:
        die(f"no annotation files (*.json) in {annotations}", EXIT_INVALID)
    clips: dict[str, dict] = {}
    by_sha: dict[str, str] = {}
    for path in ann_files:
        ann = load_annotation(path)
        if ann["clip"] in clips:
            raise AnnotationError(f"{path}: duplicate clip name {ann['clip']!r}")
        if ann["source_sha256"] in by_sha:
            raise AnnotationError(
                f"{path}: same source as clip {by_sha[ann['source_sha256']]!r}; a held-out clip "
                "must not share footage with another clip"
            )
        clips[ann["clip"]] = ann
        by_sha[ann["source_sha256"]] = ann["clip"]

    manifests: dict[str, tuple[dict, Path]] = {}
    unmatched = []
    for path in pred_files:
        manifest = envmanifest.load_manifest(path)
        sha = manifest["source"]["sha256"]
        if sha not in by_sha:
            unmatched.append(path.name)
            continue
        if by_sha[sha] in manifests:
            raise ManifestError(f"{path}: second prediction for clip {by_sha[sha]!r}")
        manifests[by_sha[sha]] = (manifest, path)
    return clips, manifests, unmatched


# --- geometry of masks -----------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


class _Frame:
    """One annotated frame: its labels, the clip's predictions there, and the ignore region."""

    def __init__(self, clip, ann, manifest, bundle, frame):
        self.id = frame["frame_id"]
        shape = (ann["height"], ann["width"])
        self.shape = shape
        aliases = ann.get("aliases", {})
        self.scanned = manifest is not None and self.id in clip["scanned"]
        self.ignore = np.zeros(shape, bool)
        for region in frame.get("ignore", []):
            self.ignore |= load_mask(ann, region["mask"], shape)
        self.positive: dict[str, np.ndarray] = {}
        for surface in frame.get("surfaces", []):
            mask = load_mask(ann, surface["mask"], shape)
            cls = surface["class"]
            self.positive[cls] = self.positive.get(cls, np.zeros(shape, bool)) | mask
        self.negative = list(frame.get("negative_classes", []))
        self.instances = []
        for inst in frame.get("instances", []):
            if inst["visibility"] == "absent":
                continue
            self.instances.append(
                {
                    "id": inst["id"],
                    "family": inst["family"],
                    "cls": canonical(inst["class"], aliases),
                    "visibility": inst["visibility"],
                    "mask": load_mask(ann, inst["mask"], shape),
                }
            )
        self.preds: list[dict] = []
        self.pred_surfaces: dict[str, np.ndarray] = {}
        if self.scanned:
            for obs in clip["observations"].get(self.id, []):
                ent = clip["entities"][obs["entity"]]
                if obs["visibility"] != "visible" or obs.get("mask") is None:
                    continue
                mask = read_mask(
                    bundle / obs["mask"]["asset"],
                    shape,
                    f"prediction mask {obs['mask']['asset']!r}",
                )
                cls = canonical(ent["labels"]["normalized"], aliases)
                if ent["family"] == "surface":
                    if cls in STRUCTURAL:
                        prev = self.pred_surfaces.get(cls, np.zeros(shape, bool))
                        self.pred_surfaces[cls] = prev | mask
                elif ent["family"] in COUNTABLE_FAMILIES:
                    self.preds.append(
                        {"eid": ent["id"], "family": ent["family"], "cls": cls, "mask": mask}
                    )

    def usable_preds(self) -> list[dict]:
        """Predictions that are not mostly inside an ignored region, with ignored pixels removed."""
        out = []
        for pred in self.preds:
            area = int(pred["mask"].sum())
            inside = int(np.logical_and(pred["mask"], self.ignore).sum())
            if area == 0 or inside / area >= TARGETS["ignore_overlap"]:
                continue
            out.append({**pred, "mask": np.logical_and(pred["mask"], ~self.ignore)})
        return out

    def scored_instances(self) -> list[dict]:
        return [
            {**inst, "mask": np.logical_and(inst["mask"], ~self.ignore)} for inst in self.instances
        ]


def _match(frame: _Frame, family: str | None = None):
    """Greedy one-to-one matching of predictions to instances, best IoU first.

    Returns ``(matches, unmatched_preds, unmatched_visible_gts, preds)`` where a match is
    ``(pred, gt)``. A pair needs the same family and class and IoU of at least 0.50.
    """
    preds = [p for p in frame.usable_preds() if family in (None, p["family"])]
    gts = [g for g in frame.scored_instances() if family in (None, g["family"])]
    pairs = []
    for p in preds:
        for g in gts:
            if p["family"] == g["family"] and p["cls"] == g["cls"]:
                iou = _iou(p["mask"], g["mask"])
                if iou >= TARGETS["mask_iou"]:
                    pairs.append((-iou, p["eid"], g["id"], p, g))
    pairs.sort(key=lambda t: t[:3])
    used_p, used_g, matches = set(), set(), []
    for _, _, _, p, g in pairs:
        if p["eid"] in used_p or g["id"] in used_g:
            continue
        used_p.add(p["eid"])
        used_g.add(g["id"])
        matches.append((p, g))
    unmatched_preds = [p for p in preds if p["eid"] not in used_p]
    missed = [g for g in gts if g["id"] not in used_g and g["visibility"] == "visible"]
    return matches, unmatched_preds, missed, preds


# --- gates ----------------------------------------------------------------------------------


def _ratio(num: int, den: int):
    return num / den if den else None


def _count_gate(family: str, frames_by_clip: dict) -> dict:
    total = defaultdict(int)
    per_class: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_clip: dict[str, dict] = {}
    for clip, frames in frames_by_clip.items():
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for frame in frames:
            matches, extra, missed, _ = _match(frame, family)
            for _pred, gt in matches:
                if gt["visibility"] == "visible":
                    counts["tp"] += 1
                    per_class[gt["cls"]]["tp"] += 1
            for pred in extra:
                counts["fp"] += 1
                per_class[pred["cls"]]["fp"] += 1
            for gt in missed:
                counts["fn"] += 1
                per_class[gt["cls"]]["fn"] += 1
        per_clip[clip] = counts
        for key, value in counts.items():
            total[key] += value
    counts = {k: total[k] for k in ("tp", "fp", "fn")}
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    classes = {}
    for cls, c in sorted(per_class.items()):
        entry = dict(c)
        entry["instances"] = c["tp"] + c["fn"]
        entry["recall"] = _ratio(c["tp"], c["tp"] + c["fn"])
        classes[cls] = entry
    reasons = []
    if counts["tp"] + counts["fn"] == 0:
        reasons.append(f"no annotated {family} instances: nothing to score")
    if precision is None:
        reasons.append("no predictions: precision is undefined")
    elif precision < TARGETS["object_precision"]:
        reasons.append(f"precision {precision:.3f} is below {TARGETS['object_precision']}")
    if recall is not None and recall < TARGETS["object_recall"]:
        reasons.append(f"recall {recall:.3f} is below {TARGETS['object_recall']}")
    for cls, entry in classes.items():
        if (
            entry["instances"] >= TARGETS["subtype_min_instances"]
            and entry["recall"] < TARGETS["subtype_recall"]
        ):
            reasons.append(
                f"subtype {cls!r} recall {entry['recall']:.3f} is below {TARGETS['subtype_recall']}"
            )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "counts": counts,
        "precision": precision,
        "recall": recall,
        "per_class": classes,
        "per_clip": per_clip,
    }


def _surface_gate(cls: str, frames_by_clip: dict) -> dict:
    inter_total = union_total = 0
    per_clip_sums: dict[str, list[int]] = {}
    neg_frames = neg_fp = 0
    for clip, frames in frames_by_clip.items():
        for frame in frames:
            keep = ~frame.ignore
            pred = frame.pred_surfaces.get(cls, np.zeros(frame.shape, bool)) & keep
            if cls in frame.positive:
                gt = frame.positive[cls] & keep
                sums = per_clip_sums.setdefault(clip, [0, 0])
                inter, union = (
                    int(np.logical_and(pred, gt).sum()),
                    int(np.logical_or(pred, gt).sum()),
                )
                sums[0] += inter
                sums[1] += union
                inter_total += inter
                union_total += union
            elif cls in frame.negative and frame.scanned:
                neg_frames += 1
                area = int(keep.sum())
                if int(pred.sum()) > TARGETS["negative_area_fraction"] * area:
                    neg_fp += 1
    iou = _ratio(inter_total, union_total)
    per_clip = {c: _ratio(i, u) for c, (i, u) in per_clip_sums.items()}
    rate = _ratio(neg_fp, neg_frames)
    reasons = []
    if iou is None:
        reasons.append(f"no {cls} annotated in any scored clip")
    elif iou < TARGETS["structural_iou"]:
        reasons.append(f"IoU {iou:.3f} is below {TARGETS['structural_iou']}")
    for clip, value in sorted(per_clip.items()):
        if value is not None and value < TARGETS["structural_clip_iou"]:
            reasons.append(
                f"clip {clip!r} IoU {value:.3f} is below {TARGETS['structural_clip_iou']}"
            )
    if rate is None:
        reasons.append(f"no {cls}-negative frames: the false-positive rate cannot be measured")
    elif rate > TARGETS["negative_false_positive_rate"]:
        reasons.append(
            f"{neg_fp} of {neg_frames} {cls}-negative frames hold a predicted region "
            f"over {TARGETS['negative_area_fraction']:.0%} of the image"
        )
    return {
        "pass": not reasons,
        "reasons": reasons,
        "iou": iou,
        "intersection_pixels": inter_total,
        "union_pixels": union_total,
        "per_clip": per_clip,
        "negatives": {"frames": neg_frames, "false_positive_frames": neg_fp, "rate": rate},
    }


def _hungarian_max(weights: list[list[int]]) -> int:
    """Total weight of the best one-to-one assignment (rows to columns, either may be short)."""
    if not weights or not weights[0]:
        return 0
    n = max(len(weights), len(weights[0]))
    cost: list[list[float]] = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i, row in enumerate(weights, 1):
        for j, w in enumerate(row, 1):
            cost[i][j] = -w
    u: list[float] = [0.0] * (n + 1)
    v: list[float] = [0.0] * (n + 1)
    match, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    return round(sum(-cost[match[j]][j] for j in range(1, n + 1) if match[j]))


def _identity_gate(family: str, frames_by_clip: dict, intervals_by_clip: dict) -> dict:
    idtp = n_gt = n_pred = switches = fragments = misses = 0
    per_clip: dict[str, dict] = {}
    for clip, frames in frames_by_clip.items():
        c_idtp = c_gt = c_pred = 0
        for first, last in intervals_by_clip.get(clip, []):
            gt_ids: list[str] = []
            pred_ids: list[str] = []
            overlap: dict[tuple, int] = defaultdict(int)
            track: dict[str, list] = defaultdict(list)
            for frame in sorted(frames, key=lambda f: f.id):
                if not first <= frame.id <= last:
                    continue
                matches, _extra, _, preds = _match(frame, family)
                gts = [g for g in frame.scored_instances() if g["family"] == family]
                visible = [g for g in gts if g["visibility"] == "visible"]
                ignored_pred = {p["eid"] for p, g in matches if g["visibility"] != "visible"}
                counted_preds = [p for p in preds if p["eid"] not in ignored_pred]
                n_gt_frame, n_pred_frame = len(visible), len(counted_preds)
                c_gt += n_gt_frame
                c_pred += n_pred_frame
                for p in counted_preds:
                    if p["eid"] not in pred_ids:
                        pred_ids.append(p["eid"])
                for g in visible:
                    if g["id"] not in gt_ids:
                        gt_ids.append(g["id"])
                    best = None
                    for p in counted_preds:
                        if p["cls"] == g["cls"]:
                            iou = _iou(p["mask"], g["mask"])
                            if iou >= TARGETS["mask_iou"]:
                                overlap[(g["id"], p["eid"])] += 1
                                if best is None or iou > best[0]:
                                    best = (iou, p["eid"])
                    track[g["id"]].append(best[1] if best else None)
            weights = [[overlap[(g, p)] for p in pred_ids] for g in gt_ids]
            c_idtp += _hungarian_max(weights)
            for seq in track.values():
                matched = [s for s in seq if s is not None]
                misses += seq.count(None)
                switches += sum(1 for a, b in pairwise(matched) if a != b)
                started = False
                gap = False
                for s in seq:
                    if s is None:
                        gap = gap or started
                    else:
                        if started and gap:
                            fragments += 1
                        started, gap = True, False
        per_clip[clip] = {"idtp": c_idtp, "gt_detections": c_gt, "pred_detections": c_pred}
        idtp += c_idtp
        n_gt += c_gt
        n_pred += c_pred
    idf1 = _ratio(2 * idtp, n_gt + n_pred)
    reasons = []
    if n_gt == 0:
        reasons.append(f"no annotated {family} identities in any tracking interval")
    elif idf1 is not None and idf1 < TARGETS["idf1"]:
        reasons.append(f"IDF1 {idf1:.3f} is below {TARGETS['idf1']}")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "idf1": idf1,
        "idtp": idtp,
        "gt_detections": n_gt,
        "pred_detections": n_pred,
        "switches": switches,
        "fragmentation": fragments,
        "misses": misses,
        "per_clip": per_clip,
    }


def _nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _registration_gate(clips: dict, manifests: dict) -> dict:
    eligible = valid = 0
    errors: list[float] = []
    per_clip = {}
    for name, ann in clips.items():
        geometry = ann.get("geometry")
        if not geometry or not geometry["eligible_frames"]:
            continue
        diag = math.hypot(ann["width"], ann["height"])
        manifest = manifests.get(name, (None,))[0]
        predicted_geometry = (manifest or {}).get("geometry") or {}
        status_is_registered = predicted_geometry.get("status") in REGISTERED_GEOMETRY
        evaluation = predicted_geometry.get("evaluation") or {}
        predicted = {f["frame_id"]: f for f in evaluation.get("frames", [])}
        c_valid = 0
        for frame_id in geometry["eligible_frames"]:
            eligible += 1
            record = predicted.get(frame_id)
            wanted = [cp for cp in geometry.get("control_points", []) if cp["frame_id"] == frame_id]
            points = {p["id"]: p["xy"] for p in (record or {}).get("control_points", [])}
            if not status_is_registered or not record or record.get("registered") is not True:
                continue
            if any(cp["id"] not in points for cp in wanted):
                continue
            valid += 1
            c_valid += 1
            for cp in wanted:
                px, py = points[cp["id"]]
                errors.append(100.0 * math.hypot(px - cp["xy"][0], py - cp["xy"][1]) / diag)
        per_clip[name] = {
            "eligible_frames": len(geometry["eligible_frames"]),
            "registered": c_valid,
        }
    coverage = _ratio(valid, eligible)
    median = _median(errors) if errors else None
    p95 = _nearest_rank(errors, 0.95) if errors else None
    reasons = []
    if coverage is None:
        reasons.append("no eligible frames: registration cannot be scored")
    elif coverage < TARGETS["registration_coverage"]:
        reasons.append(
            f"coverage {coverage:.3f} is below {TARGETS['registration_coverage']}; "
            "abstained frames count against it"
        )
    if coverage is not None and not errors:
        reasons.append("no control-point reprojections to measure")
    if median is not None and median > TARGETS["reprojection_median_pct"]:
        reasons.append(f"median error {median:.3f}% of the diagonal exceeds 0.5%")
    if p95 is not None and p95 > TARGETS["reprojection_p95_pct"]:
        reasons.append(f"95th-percentile error {p95:.3f}% of the diagonal exceeds 2%")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "eligible_frames": eligible,
        "registered_frames": valid,
        "coverage": coverage,
        "median_error_pct": median,
        "p95_error_pct": p95,
        "percentile_method": "nearest rank",
        "per_clip": per_clip,
    }


def _metric_gate(clips: dict, manifests: dict) -> dict:
    rows = []
    reasons = []
    for name, ann in clips.items():
        withheld = [d for d in (ann.get("geometry") or {}).get("dimensions", []) if d["withheld"]]
        manifest = manifests.get(name, (None,))[0]
        geometry = (manifest or {}).get("geometry") or {}
        predicted = {
            d["id"]: d["meters"] for d in (geometry.get("evaluation") or {}).get("dimensions", [])
        }
        for dim in withheld:
            got = predicted.get(dim["id"])
            error = None if got is None else abs(got - dim["meters"]) / dim["meters"]
            ok = (
                error is not None
                and error <= TARGETS["metric_relative_error"]
                and geometry.get("status") == "registered_metric"
            )
            rows.append(
                {
                    "clip": name,
                    "id": dim["id"],
                    "true_m": dim["meters"],
                    "predicted_m": got,
                    "relative_error": error,
                    "uncertainty_m": dim["uncertainty_m"],
                    "status": geometry.get("status"),
                    "pass": ok,
                }
            )
            if not ok:
                reasons.append(
                    f"{name}/{dim['id']}: "
                    + (
                        "no predicted value"
                        if got is None
                        else f"relative error {error:.3f} or geometry status "
                        f"{geometry.get('status')!r} fails the metric-scale requirement"
                    )
                )
    if not rows:
        reasons.append("no withheld dimension annotated: metric scale cannot be scored")
    return {"pass": not reasons, "reasons": reasons, "dimensions": rows}


def _underconstrained_gate(clips: dict, manifests: dict) -> dict:
    verdicts = {}
    for name, ann in clips.items():
        if "underconstrained" not in ann.get("tags", []):
            continue
        manifest = manifests.get(name, (None,))[0]
        status = ((manifest or {}).get("geometry") or {}).get("status")
        verdicts[name] = status in HONEST_GEOMETRY
    reasons = [
        f"clip {n!r} claims registration without adequate evidence"
        for n, ok in verdicts.items()
        if not ok
    ]
    if not verdicts:
        reasons.append("no underconstrained clip: honest reporting cannot be checked")
    return {"pass": not reasons, "reasons": reasons, "clips": verdicts}


def _coverage_gate(all_clips: dict, manifests: dict) -> dict:
    t = TARGETS["corpus"]
    held = {n: a for n, a in all_clips.items() if a["split"] == "heldout"}
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    n_dev = sum(1 for a in all_clips.values() if a["split"] == "development")
    check(
        "clips",
        len(all_clips) >= t["clips"]
        and n_dev >= t["development_clips"]
        and len(held) >= t["heldout_clips"],
        f"{len(all_clips)} clips ({n_dev} development, {len(held)} held-out); "
        f"need {t['clips']} with {t['development_clips']} and {t['heldout_clips']}",
    )
    short = [n for n, a in held.items() if len(a["frames"]) < t["heldout_frames_per_clip"]]
    check(
        "held-out evaluation frames",
        bool(held) and not short,
        f"clips under {t['heldout_frames_per_clip']} annotated frames: {short or 'none'}",
    )
    lacking = []
    for n, a in held.items():
        rate = a["frame_rate"][0] / a["frame_rate"][1]
        if not any(
            (i["last_frame"] - i["first_frame"]) / rate >= t["tracking_interval_seconds"]
            for i in a.get("tracking_intervals", [])
        ):
            lacking.append(n)
    check(
        "tracking interval seconds",
        bool(held) and not lacking,
        f"held-out clips without a {t['tracking_interval_seconds']} s tracking interval: {lacking or 'none'}",
    )
    check(
        "second review",
        bool(held) and all(a.get("review") for a in held.values()),
        "every held-out clip needs a second reviewer with disagreements resolved",
    )
    incomplete = []
    for name in all_clips:
        manifest = manifests.get(name, (None,))[0]
        run = (manifest or {}).get("run") or {}
        geometry_status = ((manifest or {}).get("geometry") or {}).get("status")
        geometry_only_partial = (
            run.get("status") == "partial"
            and run.get("perception_status") == "complete"
            and geometry_status in HONEST_GEOMETRY
        )
        if run.get("status") != "complete" and not geometry_only_partial:
            incomplete.append(name)
    check(
        "complete prediction runs", not incomplete, f"missing or incomplete: {incomplete or 'none'}"
    )

    def positives(a, fam):
        if fam in STRUCTURAL:
            return any(s["class"] == fam for f in a["frames"] for s in f.get("surfaces", []))
        return any(
            i["family"] == fam and i["visibility"] != "absent"
            for f in a["frames"]
            for i in f.get("instances", [])
        )

    for fam in (*COUNTABLE_FAMILIES, *STRUCTURAL):
        n = sum(1 for a in held.values() if positives(a, fam))
        check(
            f"{fam} positive clips",
            n >= t["family_positive_heldout_clips"],
            f"{n} held-out clips; need {t['family_positive_heldout_clips']}",
        )
    for cls in STRUCTURAL:
        n = sum(
            1 for a in held.values() for f in a["frames"] if cls in f.get("negative_classes", [])
        )
        check(
            f"{cls} negative frames", n >= 1, f"{n} class-negative held-out frames; need at least 1"
        )
    for fam, key in (("object", "object_subtypes"), ("vehicle", "vehicle_subtypes")):
        classes = {
            canonical(i["class"], a.get("aliases", {}))
            for a in all_clips.values()
            for f in a["frames"]
            for i in f.get("instances", [])
            if i["family"] == fam
        }
        check(f"{fam} subtypes", len(classes) >= t[key], f"{len(classes)} distinct; need {t[key]}")
    present = {tag for a in all_clips.values() for tag in a.get("tags", [])}
    missing = [tag for tag in TARGETS["required_tags"] if tag not in present]
    check("required scene tags", not missing, f"missing: {missing or 'none'}")
    translating = [
        n
        for n, a in held.items()
        if "translating" in a.get("tags", [])
        and (a.get("geometry") or {}).get("eligible_frames")
        and (a.get("geometry") or {}).get("control_points")
    ]
    check(
        "translating-camera registration clips",
        len(translating) >= t["translating_registration_clips"],
        f"{len(translating)} held-out clips with eligible frames and control points; need {t['translating_registration_clips']}",
    )
    withheld = [
        n
        for n, a in held.items()
        if any(d["withheld"] for d in (a.get("geometry") or {}).get("dimensions", []))
    ]
    check(
        "withheld metric dimension",
        bool(withheld),
        f"{len(withheld)} held-out clips withhold a measured dimension",
    )
    failing = [c["name"] for c in checks if not c["pass"]]
    return {"pass": not failing, "reasons": [f"failed: {n}" for n in failing], "checks": checks}


# --- orchestration --------------------------------------------------------------------------


def score(predictions: str, annotations: str, split: str = "heldout") -> dict:
    all_clips, manifests, unmatched = _load_corpus(predictions, annotations)
    selected = {n: a for n, a in all_clips.items() if split == "all" or a["split"] == split}
    if not selected:
        die(f"no annotated clips in the {split!r} split", EXIT_INVALID)

    inputs: dict[str, dict] = {}
    frames_by_clip: dict[str, list[_Frame]] = {}
    intervals_by_clip: dict[str, list] = {}
    for name, ann in selected.items():
        manifest_path = manifests.get(name)
        manifest = manifest_path[0] if manifest_path else None
        info = {"split": ann["split"], "prediction": "present" if manifest else "missing"}
        clip: dict = {"scanned": set(), "entities": {}, "observations": {}}
        bundle = Path()
        if manifest:
            info["run_status"] = manifest["run"]["status"]
            bundle = envmanifest.assets_dir_for(manifest_path[1])
            clip["scanned"] = {f["frame_id"] for f in manifest["processed_frames"]}
            clip["entities"] = {e["id"]: e for e in manifest["entities"]}
            for obs in manifest["observations"]:
                clip["observations"].setdefault(obs["frame_id"], []).append(obs)
        frames = [_Frame(clip, ann, manifest, bundle, f) for f in ann["frames"]]
        info["frames_unscanned"] = sorted(f.id for f in frames if not f.scanned)
        inputs[name] = info
        frames_by_clip[name] = frames
        intervals_by_clip[name] = [
            (i["first_frame"], i["last_frame"]) for i in ann.get("tracking_intervals", [])
        ]

    gates: dict = {
        "objects": _count_gate("object", frames_by_clip),
        "vehicles": _count_gate("vehicle", frames_by_clip),
    }
    for cls in STRUCTURAL:
        gates[f"{cls}s"] = _surface_gate(cls, frames_by_clip)
    gates["structural_surfaces"] = {
        "pass": all(gates[f"{c}s"]["pass"] for c in STRUCTURAL),
        "classes": {c: gates[f"{c}s"]["pass"] for c in STRUCTURAL},
    }
    gates["identity_objects"] = _identity_gate("object", frames_by_clip, intervals_by_clip)
    gates["identity_vehicles"] = _identity_gate("vehicle", frames_by_clip, intervals_by_clip)
    gates["registration"] = _registration_gate(selected, manifests)
    gates["metric_scale"] = _metric_gate(selected, manifests)
    gates["underconstrained"] = _underconstrained_gate(selected, manifests)
    gates["corpus_coverage"] = _coverage_gate(all_clips, manifests)
    return {
        "schema": SCORE_SCHEMA,
        "scorer_version": SCORER_VERSION,
        "split": split,
        "targets": TARGETS,
        "inputs": {
            "annotations": len(all_clips),
            "predictions": len(manifests),
            "unmatched_predictions": unmatched,
            "clips": inputs,
        },
        "gates": gates,
        "all_gates_passed": all(g["pass"] for g in gates.values()),
    }


def add_cli(subparsers) -> None:
    p = subparsers.add_parser(
        "environment-score",
        help="score environment manifests against human annotations, gate by gate",
    )
    p.add_argument("--predictions", required=True, help="directory of environment manifests")
    p.add_argument("--annotations", required=True, help="directory of annotation files")
    p.add_argument("--out", required=True, help="score report to write (JSON)")
    p.add_argument(
        "--split",
        choices=("heldout", "development", "all"),
        default="heldout",
        help="which clips the gates score (default: heldout); coverage always sees the corpus",
    )
    p.set_defaults(func=run_cli)


def run_cli(args) -> int:
    _validate_score_output_path(args.out, args.predictions, args.annotations)
    try:
        report = score(args.predictions, args.annotations, args.split)
    except ManifestError as exc:
        die(str(exc), EXIT_INVALID)
    artifacts.write_atomic(args.out, json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, gate in report["gates"].items():
        mark = "pass" if gate["pass"] else "FAIL"
        print(f"{mark:4}  {name}")
        for reason in gate.get("reasons", []):
            print(f"        {reason}")
    print(f"scorer {SCORER_VERSION}, split {args.split}: report written to {args.out}")
    if not report["all_gates_passed"]:
        print("error: not every gate passed", file=sys.stderr)
        return EXIT_FAILED_GATES
    return 0
