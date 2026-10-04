# SPDX-License-Identifier: MIT
"""The versioned environment manifest: its vocabulary and load-time validation.

The manifest is JSON plus a sibling ``<stem>.assets/`` directory of local files (masks and
the like). Loading never trusts either: the schema version, the structure, every
number and every asset path is checked before a renderer sees the result.
"""

import json
import math
import os
from fractions import Fraction
from pathlib import Path

from .artifacts import sha256_file

SCHEMA_VERSION = "skeleton-maker.environment/1"

RUN_STATUSES = ("complete", "partial", "failed")
FAMILIES = ("surface", "object", "vehicle", "person")
MOTIONS = ("static", "dynamic", "unknown")
VISIBILITIES = ("visible", "occluded", "absent", "uncertain")
#: unavailable: no geometry result; not_requested: --geometry off; relative: camera-frame only;
#: registered_relative / registered_metric: shared frame, without / with a solved metric scale.
GEOMETRY_STATUSES = (
    "not_requested",
    "unavailable",
    "relative",
    "registered_relative",
    "registered_metric",
)


class ManifestError(ValueError):
    """The manifest or one of its assets is invalid; the message names the offending field."""


def assets_dir_for(manifest_path) -> Path:
    """The directory that holds a manifest's local assets: ``<stem>.assets`` beside it."""
    p = Path(manifest_path)
    return p.with_name(f"{p.stem}.assets")


def _reject_constant(name):
    raise ManifestError(f"non-finite number {name} is not valid in a manifest")


def loads(text: str) -> dict:
    try:
        doc = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ManifestError("manifest must be a JSON object")
    return doc


def dumps(doc: dict) -> str:
    """Serialise deterministically; NaN/Infinity can never be written."""
    return json.dumps(doc, indent=2, sort_keys=False, allow_nan=False) + "\n"


def _need(obj, key, kind, where):
    if not isinstance(obj, dict) or key not in obj:
        raise ManifestError(f"{where}: missing '{key}'")
    value = obj[key]
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManifestError(f"{where}.{key}: expected a number")
        if not math.isfinite(value):
            raise ManifestError(f"{where}.{key}: must be finite")
    elif kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ManifestError(f"{where}.{key}: expected an integer")
    elif not isinstance(value, kind):
        raise ManifestError(f"{where}.{key}: expected {kind.__name__}")
    return value


def _one_of(value, allowed, where):
    if value not in allowed:
        raise ManifestError(f"{where}: {value!r} is not one of {', '.join(allowed)}")


def validate_content(doc: dict, *, width: int, height: int, frame_ids) -> None:
    """Check shots, entities, observations and geometry against the source's frame/pixel space.

    Shared by backend-response validation and manifest loading so both enforce one contract.
    """
    frame_set = set(frame_ids)
    shots = _need(doc, "shots", list, "manifest")
    shot_ids = set()
    shot_range = {}
    for i, shot in enumerate(shots):
        where = f"shots[{i}]"
        sid = _need(shot, "id", str, where)
        if sid in shot_ids:
            raise ManifestError(f"{where}: duplicate shot id {sid!r}")
        shot_ids.add(sid)
        first = _need(shot, "first_frame", int, where)
        last = _need(shot, "last_frame", int, where)
        if first > last:
            raise ManifestError(f"{where}: first_frame {first} is after last_frame {last}")
        shot_range[sid] = (first, last)

    entity_ids = set()
    entity_shot: dict = {}
    person_keys: set = set()
    for i, ent in enumerate(_need(doc, "entities", list, "manifest")):
        where = f"entities[{i}]"
        eid = _need(ent, "id", str, where)
        if eid in entity_ids:
            raise ManifestError(f"{where}: duplicate entity id {eid!r}")
        entity_ids.add(eid)
        if _need(ent, "shot", str, where) not in shot_ids:
            raise ManifestError(f"{where}: shot {ent['shot']!r} is not declared in shots")
        entity_shot[eid] = ent["shot"]
        _one_of(_need(ent, "family", str, where), FAMILIES, f"{where}.family")
        _one_of(_need(ent, "motion", str, where), MOTIONS, f"{where}.motion")
        skeleton = ent.get("skeleton_id")
        if skeleton is not None and (isinstance(skeleton, bool) or not isinstance(skeleton, int)):
            raise ManifestError(f"{where}.skeleton_id: expected an integer or null")
        if ent["family"] != "person" and skeleton is not None:
            raise ManifestError(f"{where}.skeleton_id: only person entities link to skeletons")
        if skeleton is not None:
            key = (ent["shot"], skeleton)
            if key in person_keys:
                raise ManifestError(
                    f"{where}: duplicate person entity for skeleton_id {skeleton} in {ent['shot']}"
                )
            person_keys.add(key)
        labels = _need(ent, "labels", dict, where)
        _need(labels, "native", str, f"{where}.labels")
        _need(labels, "normalized", str, f"{where}.labels")

    obs_ids = set()
    for i, obs in enumerate(_need(doc, "observations", list, "manifest")):
        where = f"observations[{i}]"
        oid = _need(obs, "id", str, where)
        if oid in obs_ids:
            raise ManifestError(f"{where}: duplicate observation id {oid!r}")
        obs_ids.add(oid)
        if _need(obs, "entity", str, where) not in entity_ids:
            raise ManifestError(f"{where}: entity {obs['entity']!r} does not exist")
        if _need(obs, "frame_id", int, where) not in frame_set:
            raise ManifestError(f"{where}: frame_id {obs['frame_id']} was not a processed frame")
        first, last = shot_range[entity_shot[obs["entity"]]]
        if not first <= obs["frame_id"] <= last:
            raise ManifestError(
                f"{where}: frame {obs['frame_id']} is outside its shot "
                f"{entity_shot[obs['entity']]!r} ({first}-{last}); identities are shot-local"
            )
        bbox = _need(obs, "bbox", list, where)
        if len(bbox) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in bbox
        ):
            raise ManifestError(f"{where}.bbox: expected four finite numbers [x0, y0, x1, y1]")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
            raise ManifestError(
                f"{where}.bbox: {bbox} is outside the {width}x{height} source frame "
                "or has x0 > x1 / y0 > y1"
            )
        score = _need(obs, "score", float, where)
        if not 0.0 <= score <= 1.0:
            raise ManifestError(f"{where}.score: {score} is outside 0..1")
        _need(obs, "score_meaning", str, where)
        _one_of(_need(obs, "visibility", str, where), VISIBILITIES, f"{where}.visibility")
        mask = obs.get("mask")
        if mask is not None:
            _need(mask, "asset", str, f"{where}.mask")

    geometry = _need(doc, "geometry", dict, "manifest")
    _one_of(_need(geometry, "status", str, "geometry"), GEOMETRY_STATUSES, "geometry.status")
    _need(geometry, "reason", str, "geometry")


def validate_manifest(doc: dict) -> None:
    """Structural and semantic validation of a parsed manifest (assets not touched)."""
    schema = doc.get("schema")
    if schema != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema {schema!r}; this version reads {SCHEMA_VERSION!r}"
        )
    run = _need(doc, "run", dict, "manifest")
    _one_of(_need(run, "status", str, "run"), RUN_STATUSES, "run.status")
    if run["status"] != "complete":
        _need(run, "reason", str, "run")
    source = _need(doc, "source", dict, "manifest")
    for key in ("path", "sha256"):
        _need(source, key, str, "source")
    width = _need(source, "width", int, "source")
    height = _need(source, "height", int, "source")
    rate = _need(source, "frame_rate", list, "source")
    if len(rate) != 2 or not all(isinstance(v, int) and v > 0 for v in rate):
        raise ManifestError("source.frame_rate: expected [numerator, denominator], both > 0")
    _need(source, "frame_count", int, "source")
    frames = _need(doc, "processed_frames", list, "manifest")
    ids = []
    for i, frame in enumerate(frames):
        where = f"processed_frames[{i}]"
        ids.append(_need(frame, "frame_id", int, where))
        t = _need(frame, "time", list, where)
        if len(t) != 2 or not all(isinstance(v, int) for v in t) or t[1] <= 0:
            raise ManifestError(f"{where}.time: expected [numerator, denominator]")
        _need(frame, "time_s", float, where)
    if ids != sorted(set(ids)):
        raise ManifestError("processed_frames: frame ids must be unique and increasing")
    _need(doc, "backend", dict, "manifest")
    config = _need(doc, "config", dict, "manifest")
    _need(config, "requested_labels", list, "config")
    _need(doc, "assets", list, "manifest")
    _validate_shot_times(doc, tuple(rate), source["frame_count"])
    validate_content(doc, width=width, height=height, frame_ids=ids)
    _validate_poses(doc)


def _validate_shot_times(doc: dict, rate: tuple, frame_count: int) -> None:
    """Shots tile the clip exactly and their times are the frame ids at the clip's rate."""
    clock_rate = Fraction(*rate)
    expected = 0
    for i, shot in enumerate(doc["shots"]):
        if shot["first_frame"] != expected:
            raise ManifestError(
                f"shots[{i}]: must start at frame {expected} (shots are contiguous), "
                f"found {shot['first_frame']}"
            )
        expected = shot["last_frame"] + 1
    if expected != frame_count:
        raise ManifestError(
            f"shots must cover the clip: they end at frame {expected - 1} but the clip has "
            f"{frame_count} frames"
        )
    for i, shot in enumerate(doc["shots"]):
        for key, frame in (("first_time", shot["first_frame"]), ("last_time", shot["last_frame"])):
            pair = _need(shot, key, list, f"shots[{i}]")
            if (
                len(pair) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in pair)
                or pair[1] <= 0
                or Fraction(pair[0], pair[1]) != Fraction(frame) / clock_rate
            ):
                raise ManifestError(
                    f"shots[{i}].{key}: {pair} does not follow the frame rate "
                    f"{rate[0]}/{rate[1]} for frame {frame}"
                )
    detection = doc.get("shot_detection")
    if detection is not None:
        starts = [s["first_frame"] for s in doc["shots"][1:]]
        found = [b.get("frame_id") for b in _need(detection, "boundaries", list, "shot_detection")]
        if starts != found:
            raise ManifestError(
                f"shot_detection.boundaries {found} do not match the shot starts {starts}"
            )


POSE_ASSOCIATIONS = ("user-supplied", "hash-verified")


def _validate_poses(doc: dict) -> None:
    """The pose join block, and that person entities name skeletons that exist in it."""
    block = doc.get("poses")
    people = [e for e in doc["entities"] if e["family"] == "person"]
    if block is None:
        for i, ent in enumerate(doc["entities"]):
            if ent.get("skeleton_id") is not None:
                raise ManifestError(
                    f"entities[{i}].skeleton_id: names a skeleton, but no pose file was supplied"
                )
        return
    if not isinstance(block, dict):
        raise ManifestError("poses: expected an object or null")
    _need(block, "path", str, "poses")
    _one_of(_need(block, "association", str, "poses"), POSE_ASSOCIATIONS, "poses.association")
    _need(block, "frame_count", int, "poses")
    ids = set()
    for i, skeleton in enumerate(_need(block, "skeletons", list, "poses")):
        where = f"poses.skeletons[{i}]"
        ids.add(_need(skeleton, "id", int, where))
        for key in ("first_frame", "last_frame", "frames"):
            _need(skeleton, key, int, where)
    for i, span in enumerate(_need(block, "person_free_ranges", list, "poses")):
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(v, int) for v in span)
            and span[0] <= span[1]
        ):
            raise ManifestError(f"poses.person_free_ranges[{i}]: expected [first, last]")
    for ent in people:
        skeleton = ent.get("skeleton_id")
        if skeleton is None:
            raise ManifestError(
                f"person entity {ent['id']!r} must name an existing skeleton_id from the pose file"
            )
        if skeleton not in ids:
            raise ManifestError(
                f"person entity {ent['id']!r}: skeleton_id {skeleton} is not in the pose file"
            )


def resolve_asset(bundle: Path, rel: str, where: str = "asset") -> Path:
    """Resolve a manifest-relative asset path, refusing anything outside ``bundle``."""
    if not rel or os.path.isabs(rel) or "\\" in rel or "\x00" in rel:
        raise ManifestError(f"{where}: unsafe asset path {rel!r}")
    if ".." in Path(rel).parts:
        raise ManifestError(f"{where}: asset path {rel!r} escapes the bundle")
    root = bundle.resolve()
    target = (bundle / rel).resolve()
    if root != target and root not in target.parents:
        raise ManifestError(f"{where}: asset path {rel!r} escapes the bundle")
    return target


def check_assets(doc: dict, bundle: Path) -> None:
    """Every referenced asset exists inside ``bundle`` and matches its recorded hash."""
    declared = {}
    for i, entry in enumerate(doc["assets"]):
        where = f"assets[{i}]"
        rel = _need(entry, "path", str, where)
        _need(entry, "sha256", str, where)
        declared[rel] = entry
    for rel, entry in declared.items():
        target = resolve_asset(bundle, rel, f"asset {rel!r}")
        if not target.is_file():
            raise ManifestError(f"asset {rel!r} is referenced but missing from {bundle}")
        if sha256_file(target) != entry["sha256"]:
            raise ManifestError(f"asset {rel!r} does not match its recorded sha256")
    for i, obs in enumerate(doc["observations"]):
        mask = obs.get("mask")
        if mask is not None and mask["asset"] not in declared:
            raise ManifestError(
                f"observations[{i}].mask: asset {mask['asset']!r} is not listed in assets"
            )


def load_manifest(path) -> dict:
    """Read, parse and fully validate a manifest and its assets."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc.strerror or exc}") from exc
    doc = loads(text)
    validate_manifest(doc)
    check_assets(doc, assets_dir_for(path))
    return doc
