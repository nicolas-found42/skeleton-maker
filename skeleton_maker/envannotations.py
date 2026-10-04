# SPDX-License-Identifier: MIT
"""Ground-truth annotation files for scoring environment scans.

One JSON file per clip, schema ``skeleton-maker.environment-annotations/1``, plus the mask
PNG images it references (any non-zero pixel is inside; the size must equal the clip's frame).
Annotations are written by people. A model prediction is never ground truth, so nothing in
this module reads a manifest.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .envmanifest import ManifestError, _need, _one_of, resolve_asset

ANNOTATION_SCHEMA = "skeleton-maker.environment-annotations/1"

SPLITS = ("development", "heldout")
STRUCTURAL = ("wall", "floor", "ceiling")
COUNTABLE_FAMILIES = ("object", "vehicle")
#: ``occluded`` instances are real but unscored: neither a detection nor a miss.
GT_VISIBILITIES = ("visible", "occluded", "absent")


class AnnotationError(ManifestError):
    """An annotation file is malformed; the message names the file and the field."""


def load_annotation(path) -> dict:
    """Parse and validate one annotation file. Masks are checked lazily by :func:`load_mask`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        doc = json.loads(text, parse_constant=_nonfinite)
    except OSError as exc:
        raise AnnotationError(f"{path}: cannot read: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{path}: not valid JSON: {exc}") from exc
    try:
        _validate(doc)
    except AnnotationError as exc:
        raise AnnotationError(f"{path}: {exc}") from exc
    except ManifestError as exc:
        raise AnnotationError(f"{path}: {exc}") from exc
    doc["_path"] = str(path)
    return doc


def _nonfinite(name):
    raise AnnotationError(f"non-finite number {name} is not valid")


def _frame_range(obj, where):
    first = _need(obj, "first_frame", int, where)
    last = _need(obj, "last_frame", int, where)
    if first > last:
        raise AnnotationError(f"{where}: first_frame {first} is after last_frame {last}")
    return first, last


def _list_field(obj, key, where, *, default=None):
    if key not in obj:
        if default is not None:
            return default
        raise AnnotationError(f"{where}: missing '{key}'")
    value = obj[key]
    if not isinstance(value, list):
        raise AnnotationError(f"{where}.{key}: expected a list")
    return value


def _validate(doc) -> None:
    if not isinstance(doc, dict):
        raise AnnotationError("annotation must be a JSON object")
    if doc.get("schema") != ANNOTATION_SCHEMA:
        raise AnnotationError(
            f"unsupported annotation schema {doc.get('schema')!r}; expected {ANNOTATION_SCHEMA!r}"
        )
    _need(doc, "clip", str, "annotation")
    source_sha = _need(doc, "source_sha256", str, "annotation")
    if len(source_sha) != 64:
        raise AnnotationError("source_sha256: expected a 64-character SHA-256 hexadecimal digest")
    if any(character not in "0123456789abcdefABCDEF" for character in source_sha):
        raise AnnotationError("source_sha256: expected a hexadecimal digest")
    _one_of(_need(doc, "split", str, "annotation"), SPLITS, "split")
    width = _need(doc, "width", int, "annotation")
    height = _need(doc, "height", int, "annotation")
    if width <= 0 or height <= 0:
        raise AnnotationError("width and height must be positive")
    rate = _need(doc, "frame_rate", list, "annotation")
    if len(rate) != 2 or not all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in rate
    ):
        raise AnnotationError("frame_rate: expected [numerator, denominator], both > 0")
    tags = doc.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise AnnotationError("tags: expected a list of strings")
    aliases = doc.get("aliases", {})
    if not isinstance(aliases, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
    ):
        raise AnnotationError("aliases: expected an object mapping alias -> canonical class")
    review = doc.get("review")
    if review is not None:
        _need(review, "second_reviewer", str, "review")
        if _need(review, "disagreements_resolved", bool, "review") is not True:
            raise AnnotationError("review.disagreements_resolved must be true before scoring")

    shots = []
    for i, shot in enumerate(_need(doc, "shots", list, "annotation")):
        if not isinstance(shot, dict):
            raise AnnotationError(f"shots[{i}]: expected an object")
        shots.append(_frame_range(shot, f"shots[{i}]"))
    seen_frames = set()
    for i, frame in enumerate(_need(doc, "frames", list, "annotation")):
        where = f"frames[{i}]"
        if not isinstance(frame, dict):
            raise AnnotationError(f"{where}: expected an object")
        fid = _need(frame, "frame_id", int, where)
        if fid in seen_frames:
            raise AnnotationError(f"{where}: duplicate frame_id {fid}")
        seen_frames.add(fid)
        negatives = frame.get("negative_classes", [])
        if not isinstance(negatives, list) or any(c not in STRUCTURAL for c in negatives):
            raise AnnotationError(
                f"{where}.negative_classes: only {', '.join(STRUCTURAL)} can be class-negative"
            )
        positives = set()
        for j, surface in enumerate(_list_field(frame, "surfaces", where, default=[])):
            sw = f"{where}.surfaces[{j}]"
            if not isinstance(surface, dict):
                raise AnnotationError(f"{sw}: expected an object")
            cls = _need(surface, "class", str, sw)
            if cls not in STRUCTURAL:
                raise AnnotationError(f"{sw}.class: {cls!r} is not one of {', '.join(STRUCTURAL)}")
            _need(surface, "mask", str, sw)
            positives.add(cls)
        both = positives & set(negatives)
        if both:
            raise AnnotationError(f"{where}: {sorted(both)[0]} is both positive and negative")
        ids = set()
        for j, inst in enumerate(_list_field(frame, "instances", where, default=[])):
            iw = f"{where}.instances[{j}]"
            if not isinstance(inst, dict):
                raise AnnotationError(f"{iw}: expected an object")
            iid = _need(inst, "id", str, iw)
            if iid in ids:
                raise AnnotationError(f"{iw}: duplicate instance id {iid!r} in the frame")
            ids.add(iid)
            _one_of(_need(inst, "family", str, iw), COUNTABLE_FAMILIES, f"{iw}.family")
            _need(inst, "class", str, iw)
            vis = _need(inst, "visibility", str, iw)
            _one_of(vis, GT_VISIBILITIES, f"{iw}.visibility")
            if vis != "absent":
                _need(inst, "mask", str, iw)
        for j, region in enumerate(_list_field(frame, "ignore", where, default=[])):
            if not isinstance(region, dict):
                raise AnnotationError(f"{where}.ignore[{j}]: expected an object")
            _need(region, "mask", str, f"{where}.ignore[{j}]")
    for i, interval in enumerate(_list_field(doc, "tracking_intervals", "annotation", default=[])):
        if not isinstance(interval, dict):
            raise AnnotationError(f"tracking_intervals[{i}]: expected an object")
        first, last = _frame_range(interval, f"tracking_intervals[{i}]")
        if not any(a <= first and last <= b for a, b in shots):
            raise AnnotationError(
                f"tracking_intervals[{i}]: {first}-{last} crosses a shot boundary or lies "
                "outside every shot; identities are never scored across a cut"
            )
    geometry = doc.get("geometry")
    if geometry is not None:
        if not isinstance(geometry, dict):
            raise AnnotationError("geometry: expected an object")
        eligible = _need(geometry, "eligible_frames", list, "geometry")
        if not all(isinstance(f, int) and not isinstance(f, bool) for f in eligible):
            raise AnnotationError("geometry.eligible_frames: expected frame ids")
        seen_eligible = set()
        for frame_id in eligible:
            if frame_id in seen_eligible:
                raise AnnotationError(f"geometry.eligible_frames: duplicate frame id {frame_id}")
            seen_eligible.add(frame_id)
        control_ids: dict[int, set[str]] = {}
        for i, cp in enumerate(_list_field(geometry, "control_points", "geometry", default=[])):
            cw = f"geometry.control_points[{i}]"
            if not isinstance(cp, dict):
                raise AnnotationError(f"{cw}: expected an object")
            frame_id = _need(cp, "frame_id", int, cw)
            if frame_id not in eligible:
                raise AnnotationError(f"{cw}: frame {cp['frame_id']} is not an eligible frame")
            point_id = _need(cp, "id", str, cw)
            frame_control_ids = control_ids.setdefault(frame_id, set())
            if point_id in frame_control_ids:
                raise AnnotationError(f"{cw}: duplicate id {point_id!r} in frame {frame_id}")
            frame_control_ids.add(point_id)
            xy = _need(cp, "xy", list, cw)
            if len(xy) != 2 or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and (not isinstance(v, float) or math.isfinite(v))
                for v in xy
            ):
                raise AnnotationError(f"{cw}.xy: expected finite numbers [x, y]")
        for i, dim in enumerate(_list_field(geometry, "dimensions", "geometry", default=[])):
            dw = f"geometry.dimensions[{i}]"
            if not isinstance(dim, dict):
                raise AnnotationError(f"{dw}: expected an object")
            _need(dim, "id", str, dw)
            if _need(dim, "meters", float, dw) <= 0:
                raise AnnotationError(f"{dw}.meters: must be positive")
            if _need(dim, "uncertainty_m", float, dw) < 0:
                raise AnnotationError(f"{dw}.uncertainty_m: must not be negative")
            _need(dim, "withheld", bool, dw)


def canonical(label: str, aliases: dict) -> str:
    """The canonical class for a label under an annotation's alias table."""
    label = label.strip().lower()
    return aliases.get(label, label)


def load_mask(annotation: dict, rel: str, shape: tuple) -> np.ndarray:
    """A boolean mask of ``shape`` (rows, cols) from a PNG beside the annotation file."""
    base = Path(annotation["_path"]).parent
    target = resolve_asset(base, rel, f"annotation mask {rel!r}")
    return read_mask(target, shape, f"annotation mask {rel!r}")


def read_mask(path: Path, shape: tuple, where: str) -> np.ndarray:
    if not path.is_file():
        raise AnnotationError(f"{where}: file not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AnnotationError(f"{where}: {path} is not a readable image")
    if image.ndim != 2:
        raise AnnotationError(f"{where}: must be a single-channel PNG, got shape {image.shape}")
    if image.shape != shape:
        raise AnnotationError(
            f"{where}: is {image.shape[1]}x{image.shape[0]} but the clip frame is "
            f"{shape[1]}x{shape[0]}"
        )
    return image > 0
