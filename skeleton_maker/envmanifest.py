# SPDX-License-Identifier: MIT
"""The versioned environment manifest: its vocabulary and load-time validation.

The manifest is JSON plus a sibling ``<stem>.assets/`` directory of local files (masks and
the like). Loading never trusts either: the schema version, the structure, every
number and every asset path is checked before a renderer sees the result.
"""

import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path

from .artifacts import sha256_file
from .envcamera import IntrinsicsError, validate_intrinsics

SCHEMA_VERSION = "skeleton-maker.environment/1"

RUN_STATUSES = ("complete", "partial", "failed")
FAMILIES = ("surface", "object", "vehicle", "person")
MOTIONS = ("static", "dynamic", "unknown")
VISIBILITIES = ("visible", "occluded", "absent", "uncertain")
#: unavailable: no geometry result; not_requested: --geometry off; relative/relative-camera-frame:
#: unregistered camera-frame only (relative is a compatibility alias);
#: registered_relative / registered_metric: shared frame, without / with a solved metric scale.
GEOMETRY_STATUSES = (
    "not_requested",
    "unavailable",
    "relative",
    "relative-camera-frame",
    "registered_relative",
    "registered_metric",
)
GEOMETRY_STATUS_ALIASES = {
    "registered-relative": "registered_relative",
    "registered-metric": "registered_metric",
}


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
    _normalise_geometry_status(doc)
    return doc


def _normalise_geometry_status(doc: dict) -> None:
    geometry = doc.get("geometry")
    if isinstance(geometry, dict):
        status = geometry.get("status")
        if isinstance(status, str):
            geometry["status"] = GEOMETRY_STATUS_ALIASES.get(status, status)


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


LABEL_STATUSES = ("matched", "ambiguous", "unknown")


def _validate_labels(ent: dict, where: str, vocabulary: dict | None) -> None:
    labels = _need(ent, "labels", dict, where)
    lw = f"{where}.labels"
    _need(labels, "native", str, lw)
    if "requested" not in labels or not (
        labels["requested"] is None or isinstance(labels["requested"], str)
    ):
        raise ManifestError(f"{lw}: 'requested' must be present, a string or null")
    normalized = _need(labels, "normalized", str, lw)
    status = _need(labels, "status", str, lw)
    _one_of(status, LABEL_STATUSES, f"{lw}.status")
    candidates = labels.get("candidates")
    if candidates is not None and not (
        isinstance(candidates, list) and all(isinstance(c, str) for c in candidates)
    ):
        raise ManifestError(f"{lw}.candidates: expected a list of labels")
    if status != "matched":
        if normalized != "unknown":
            raise ManifestError(
                f"{lw}: a {status} label must have normalized 'unknown', found {normalized!r}"
            )
        return
    if vocabulary is not None and ent["family"] != "person":
        if normalized not in vocabulary:
            raise ManifestError(f"{lw}: {normalized!r} is not in the label vocabulary")
        if vocabulary[normalized] != ent["family"]:
            raise ManifestError(
                f"{where}.family: {ent['family']!r} conflicts with the vocabulary family "
                f"{vocabulary[normalized]!r} of {normalized!r}"
            )


def _validate_track(ent: dict, where: str, frame_set: set, shot_bounds: tuple) -> None:
    """Validate optional shot-local tracking evidence without requiring it from old backends."""
    if "track" not in ent:
        return
    track = _need(ent, "track", dict, where)
    track_where = f"{where}.track"
    _need(track, "method", str, track_where)
    _one_of(_need(track, "status", str, track_where), ("active", "lost"), f"{track_where}.status")
    first = _need(track, "first_observed_frame", int, track_where)
    last = _need(track, "last_observed_frame", int, track_where)
    shot_first, shot_last = shot_bounds
    for name, frame in (("first_observed_frame", first), ("last_observed_frame", last)):
        if frame not in frame_set:
            raise ManifestError(f"{track_where}.{name}: frame {frame} was not a processed frame")
        if not shot_first <= frame <= shot_last:
            raise ManifestError(f"{track_where}.{name}: frame {frame} is outside the entity shot")
    if first > last:
        raise ManifestError(f"{track_where}: first_observed_frame is after last_observed_frame")
    for i, gap in enumerate(_need(track, "gaps", list, track_where)):
        gap_where = f"{track_where}.gaps[{i}]"
        sampled = _need(gap, "sampled_frame_ids", list, gap_where)
        if not sampled:
            raise ManifestError(f"{gap_where}.sampled_frame_ids: must not be empty")
        if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in sampled):
            raise ManifestError(f"{gap_where}.sampled_frame_ids: expected integer frame IDs")
        if sampled != sorted(set(sampled)):
            raise ManifestError(
                f"{gap_where}.sampled_frame_ids: expected unique ascending frame IDs"
            )
        for frame in sampled:
            if frame not in frame_set:
                raise ManifestError(
                    f"{gap_where}.sampled_frame_ids: frame {frame} was not a processed frame"
                )
            if not shot_first <= frame <= shot_last:
                raise ManifestError(
                    f"{gap_where}.sampled_frame_ids: frame {frame} is outside the entity shot"
                )
        reacquired = gap.get("reacquired_frame")
        status = _need(gap, "status", str, gap_where)
        _one_of(status, ("lost", "reacquired"), f"{gap_where}.status")
        _need(gap, "reason", str, gap_where)
        if reacquired is None:
            if status != "lost":
                raise ManifestError(f"{gap_where}: reacquired status requires reacquired_frame")
        else:
            if isinstance(reacquired, bool) or not isinstance(reacquired, int):
                raise ManifestError(f"{gap_where}.reacquired_frame: expected an integer or null")
            if reacquired not in frame_set:
                raise ManifestError(
                    f"{gap_where}.reacquired_frame: frame {reacquired} was not a processed frame"
                )
            if reacquired <= sampled[-1]:
                raise ManifestError(f"{gap_where}.reacquired_frame: must follow the gap samples")
            if status != "reacquired":
                raise ManifestError(f"{gap_where}: reacquired_frame requires reacquired status")
    for i, item in enumerate(_need(track, "association_uncertainty", list, track_where)):
        item_where = f"{track_where}.association_uncertainty[{i}]"
        frame = _need(item, "frame_id", int, item_where)
        if frame not in frame_set:
            raise ManifestError(f"{item_where}.frame_id: frame {frame} was not a processed frame")
        candidates = _need(item, "candidate_entities", list, item_where)
        if any(not isinstance(candidate, str) for candidate in candidates):
            raise ManifestError(f"{item_where}.candidate_entities: expected a list of entity IDs")
        _need(item, "method", str, item_where)
        score = _need(item, "score", float, item_where)
        if not 0.0 <= score <= 1.0:
            raise ManifestError(f"{item_where}.score: {score} is outside 0..1")
        _need(item, "meaning", str, item_where)


def validate_content(
    doc: dict, *, width: int, height: int, frame_ids, vocabulary: dict | None = None
) -> None:
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
    doc_entities: dict = {}
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
        family = _need(ent, "family", str, where)
        doc_entities[eid] = family
        _one_of(family, FAMILIES, f"{where}.family")
        _one_of(_need(ent, "motion", str, where), MOTIONS, f"{where}.motion")
        _validate_track(ent, where, frame_set, shot_range[ent["shot"]])
        motion_evidence = ent.get("motion_evidence")
        if motion_evidence is not None:
            evidence_where = f"{where}.motion_evidence"
            _need(motion_evidence, "method", str, evidence_where)
            _need(motion_evidence, "observations", int, evidence_where)
            _need(motion_evidence, "threshold_frame_diagonals", float, evidence_where)
            _need(
                motion_evidence,
                "camera_flow_dispersion_threshold_frame_diagonals",
                float,
                evidence_where,
            )
            _need(motion_evidence, "calibrated", bool, evidence_where)
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
        _validate_labels(ent, where, vocabulary)

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
        association = obs.get("association")
        if association is not None:
            association_where = f"{where}.association"
            _need(association, "method", str, association_where)
            association_score = _need(association, "score", float, association_where)
            if not 0.0 <= association_score <= 1.0:
                raise ManifestError(
                    f"{association_where}.score: {association_score} is outside 0..1"
                )
            _need(association, "meaning", str, association_where)
        family = doc_entities[obs["entity"]]
        if family != "person" and obs["visibility"] == "visible" and mask is None:
            raise ManifestError(
                f"{where}: a visible observation needs a mask (visible extent only, "
                "never an inferred one)"
            )
        if obs["visibility"] == "absent" and mask is not None:
            raise ManifestError(f"{where}: an absent observation must not carry a mask")

    _normalise_geometry_status(doc)
    geometry = _need(doc, "geometry", dict, "manifest")
    status = _need(geometry, "status", str, "geometry")
    _one_of(status, GEOMETRY_STATUSES, "geometry.status")
    _need(geometry, "reason", str, "geometry")
    _validate_geometry_evaluation(geometry, frame_set, width, height)
    if status in ("registered_relative", "registered_metric"):
        _validate_registration(geometry, frame_set)
    if status in ("relative-camera-frame", "registered_relative", "registered_metric"):
        units = _need(geometry, "units", str, "geometry")
        if status in ("relative-camera-frame", "registered_relative") and units != "relative_depth":
            raise ManifestError("geometry.units: relative geometry must use relative_depth")
        if status == "registered_metric" and units != "m":
            raise ManifestError("geometry.units: registered metric geometry must use meters")
        convention = _need(geometry, "coordinate_convention", str, "geometry")
        if not convention:
            raise ManifestError("geometry.coordinate_convention: must not be empty")
        provenance = _need(geometry, "scale_provenance", dict, "geometry")
        metric = _need(provenance, "metric", bool, "geometry.scale_provenance")
        if status == "registered_metric" and not metric:
            raise ManifestError(
                "geometry.scale_provenance.metric must be true for registered_metric"
            )
        if status != "registered_metric" and metric:
            raise ManifestError(
                "geometry.scale_provenance.metric: relative geometry cannot be metric"
            )
        _need(provenance, "kind", str, "geometry.scale_provenance")
        static_entities = _need(geometry, "static_fusion_entities", list, "geometry")
        excluded_entities = _need(geometry, "excluded_dynamic_entities", list, "geometry")
        known_entities = entity_ids
        if any(
            not isinstance(identifier, str) or identifier not in known_entities
            for identifier in static_entities
        ):
            raise ManifestError("geometry.static_fusion_entities: expected known entity IDs")
        if len(static_entities) != len(set(static_entities)):
            raise ManifestError("geometry.static_fusion_entities: duplicate entity ID")
        excluded_ids = []
        for i, item in enumerate(excluded_entities):
            where = f"geometry.excluded_dynamic_entities[{i}]"
            identifier = _need(item, "id", str, where)
            if identifier not in known_entities:
                raise ManifestError(f"{where}.id: expected a known entity ID")
            reason = _need(item, "reason", str, where)
            if not reason:
                raise ManifestError(f"{where}.reason: must not be empty")
            excluded_ids.append(identifier)
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ManifestError("geometry.excluded_dynamic_entities: duplicate entity ID")
        if set(static_entities) & set(excluded_ids):
            raise ManifestError(
                "geometry.static_fusion_entities and excluded_dynamic_entities must be disjoint"
            )
        skeleton_exclusion = geometry.get("skeleton_exclusion")
        excluded_skeletons = geometry.get("excluded_skeletons", [])
        if not isinstance(excluded_skeletons, list):
            raise ManifestError("geometry.excluded_skeletons: expected a list")
        skeleton_ids = []
        for i, item in enumerate(excluded_skeletons):
            where = f"geometry.excluded_skeletons[{i}]"
            identifier = _need(item, "id", str, where)
            tracking_id = _need(item, "tracking_id", int, where)
            if tracking_id < 0 or identifier != f"nim-skeleton:{tracking_id}":
                raise ManifestError(f"{where}: id must name its non-negative NIM tracking_id")
            if not _need(item, "reason", str, where):
                raise ManifestError(f"{where}.reason: must not be empty")
            observed = _need(item, "observed_frame_ids", list, where)
            masked = _need(item, "bbox_frame_ids", list, where)
            for key, values in (("observed_frame_ids", observed), ("bbox_frame_ids", masked)):
                if any(
                    isinstance(frame, bool) or not isinstance(frame, int) or frame not in frame_set
                    for frame in values
                ):
                    raise ManifestError(f"{where}.{key}: expected processed source frame IDs")
            if not set(masked).issubset(observed):
                raise ManifestError(f"{where}.bbox_frame_ids: must be a subset of observed frames")
            skeleton_ids.append(identifier)
        if len(skeleton_ids) != len(set(skeleton_ids)):
            raise ManifestError("geometry.excluded_skeletons: duplicate skeleton ID")
        if skeleton_exclusion is not None:
            where = "geometry.skeleton_exclusion"
            skeleton_status = _need(skeleton_exclusion, "status", str, where)
            _one_of(skeleton_status, ("applied", "unavailable"), f"{where}.status")

            if not _need(skeleton_exclusion, "reason", str, where):
                raise ManifestError(f"{where}.reason: must not be empty")
            if skeleton_status == "unavailable" and excluded_skeletons:
                raise ManifestError(f"{where}: unavailable exclusion cannot list skeletons")
        geometry_frames = _need(geometry, "frames", list, "geometry")
        if not geometry_frames:
            raise ManifestError(
                "geometry.frames: relative geometry must include at least one frame"
            )
        seen_geometry_frames = set()
        for i, item in enumerate(geometry_frames):
            where = f"geometry.frames[{i}]"
            frame_id = _need(item, "frame_id", int, where)
            if frame_id not in frame_set:
                raise ManifestError(f"{where}.frame_id {frame_id} was not processed")
            if frame_id in seen_geometry_frames:
                raise ManifestError(f"{where}.frame_id {frame_id} is duplicated")
            seen_geometry_frames.add(frame_id)
            _need(item, "depth_asset", str, where)
            _matrix(item, "intrinsics", 3, 3, where)
            intrinsics = item["intrinsics"]
            try:
                validate_intrinsics(intrinsics, f"{where}.intrinsics")
            except IntrinsicsError as exc:
                raise ManifestError(str(exc)) from exc
            _matrix(item, "extrinsics_w2c", 3, 4, where)
            depth_pixel_space = _need(item, "depth_pixel_space", str, where)
            if depth_pixel_space not in ("source_frame_pixels", "processed_frame_pixels"):
                raise ManifestError(
                    f"{where}.depth_pixel_space: expected 'source_frame_pixels' or "
                    "'processed_frame_pixels'"
                )
            preprocessing = _need(item, "preprocessing", dict, where)
            size = _need(preprocessing, "process_res", int, f"{where}.preprocessing")
            if size <= 0:
                raise ManifestError(f"{where}.preprocessing.process_res: must be positive")
            _need(preprocessing, "process_res_method", str, f"{where}.preprocessing")
            for size_key in ("source_size", "processed_size"):
                size_pair = _need(preprocessing, size_key, list, f"{where}.preprocessing")
                if len(size_pair) != 2 or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in size_pair
                ):
                    raise ManifestError(
                        f"{where}.preprocessing.{size_key}: expected two positive integers"
                    )
            source_size = preprocessing["source_size"]
            if source_size != [width, height]:
                raise ManifestError(
                    f"{where}.preprocessing.source_size: must match source {width}x{height}"
                )
            _matrix(
                preprocessing,
                "undistorted_source_to_processed",
                3,
                3,
                f"{where}.preprocessing",
            )
            _matrix(
                preprocessing,
                "processed_to_undistorted_source",
                3,
                3,
                f"{where}.preprocessing",
            )
            source_to_processed = preprocessing["undistorted_source_to_processed"]
            processed_to_source = preprocessing["processed_to_undistorted_source"]
            processed_size = preprocessing["processed_size"]
            depth_resampling = _need(
                preprocessing, "depth_resampling", str, f"{where}.preprocessing"
            )
            intrinsics_pixel_space = _need(
                preprocessing, "intrinsics_pixel_space", str, f"{where}.preprocessing"
            )
            if depth_pixel_space == "processed_frame_pixels":
                expected_scale_x = processed_size[0] / source_size[0]
                expected_scale_y = processed_size[1] / source_size[1]
                if not math.isclose(source_to_processed[0][0], expected_scale_x, rel_tol=1e-6):
                    raise ManifestError(
                        f"{where}.preprocessing.undistorted_source_to_processed: x scale "
                        "does not match source_size and processed_size"
                    )
                if not math.isclose(source_to_processed[1][1], expected_scale_y, rel_tol=1e-6):
                    raise ManifestError(
                        f"{where}.preprocessing.undistorted_source_to_processed: y scale "
                        "does not match source_size and processed_size"
                    )
                if not math.isclose(processed_to_source[0][0], 1 / expected_scale_x, rel_tol=1e-6):
                    raise ManifestError(
                        f"{where}.preprocessing.processed_to_undistorted_source: x scale "
                        "must invert source-to-processed"
                    )
                if not math.isclose(processed_to_source[1][1], 1 / expected_scale_y, rel_tol=1e-6):
                    raise ManifestError(
                        f"{where}.preprocessing.processed_to_undistorted_source: y scale "
                        "must invert source-to-processed"
                    )
                if depth_resampling != "none; native DA3 processed depth grid":
                    raise ManifestError(
                        f"{where}.preprocessing.depth_resampling: processed depth must preserve "
                        "the native DA3 grid"
                    )
                if intrinsics_pixel_space != "undistorted_source_frame_pixels":
                    raise ManifestError(
                        f"{where}.preprocessing.intrinsics_pixel_space: expected "
                        "'undistorted_source_frame_pixels'"
                    )
            if "crop" not in preprocessing:
                raise ManifestError(f"{where}.preprocessing: missing 'crop'")
            _need(preprocessing, "lens_transform", dict, f"{where}.preprocessing")


def _validate_geometry_evaluation(
    geometry: dict, frame_ids: set[int], width: int, height: int
) -> None:
    """Validate scorer-facing geometry outputs before they can affect registration scores."""
    evaluation = geometry.get("evaluation")
    if evaluation is None:
        return
    if not isinstance(evaluation, dict):
        raise ManifestError("geometry.evaluation: expected an object")
    frames = _need(evaluation, "frames", list, "geometry.evaluation")
    seen_frames = set()
    for i, frame in enumerate(frames):
        where = f"geometry.evaluation.frames[{i}]"
        if not isinstance(frame, dict) or "frame_id" not in frame:
            raise ManifestError(f"{where}.frame_id: missing frame ID")
        frame_id = _need(frame, "frame_id", int, where)
        if frame_id not in frame_ids:
            raise ManifestError(f"{where}.frame_id: must reference a processed source frame")
        if frame_id in seen_frames:
            raise ManifestError(f"{where}.frame_id: duplicate evaluation frame")
        seen_frames.add(frame_id)
        registered = _need(frame, "registered", bool, where)
        if registered and geometry.get("status") not in (
            "registered_relative",
            "registered_metric",
        ):
            raise ManifestError(
                f"{where}.registered: true conflicts with geometry.status "
                f"{geometry.get('status')!r}"
            )
        points = frame.get("control_points", [])
        if not isinstance(points, list):
            raise ManifestError(f"{where}.control_points: expected a list")
        seen_points = set()
        for j, point in enumerate(points):
            point_where = f"{where}.control_points[{j}]"
            identifier = _need(point, "id", str, point_where)
            if not identifier:
                raise ManifestError(f"{point_where}.id: must not be empty")
            if identifier in seen_points:
                raise ManifestError(f"{point_where}.id: duplicate control point")
            seen_points.add(identifier)
            xy = _need(point, "xy", list, point_where)
            if len(xy) != 2 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in xy
            ):
                raise ManifestError(f"{point_where}.xy: expected two finite pixel coordinates")
            if not 0 <= xy[0] < width or not 0 <= xy[1] < height:
                raise ManifestError(f"{point_where}.xy: outside the source frame")
    dimensions = evaluation.get("dimensions", [])
    if not isinstance(dimensions, list):
        raise ManifestError("geometry.evaluation.dimensions: expected a list")
    seen_dimensions = set()
    for i, dimension in enumerate(dimensions):
        where = f"geometry.evaluation.dimensions[{i}]"
        identifier = _need(dimension, "id", str, where)
        if not identifier:
            raise ManifestError(f"{where}.id: must not be empty")
        if identifier in seen_dimensions:
            raise ManifestError(f"{where}.id: duplicate measured dimension")
        seen_dimensions.add(identifier)
        if _need(dimension, "meters", float, where) <= 0:
            raise ManifestError(f"{where}.meters: must be positive")
        if _need(dimension, "units", str, where) != "m":
            raise ManifestError(f"{where}.units: expected 'm'")


def _validate_registration(geometry: dict, frame_ids: set[int]) -> None:
    where = "geometry.registration"
    registration = _need(geometry, "registration", dict, "geometry")
    if _need(registration, "schema", str, where) != "skeleton-maker.geometry-registration/1":
        raise ManifestError(f"{where}.schema: unsupported registration schema")
    _need(registration, "source_frame", str, where)
    target_frame = _need(registration, "target_frame", str, where)
    _one_of(
        _need(registration, "direction", str, where), ("source_to_target",), f"{where}.direction"
    )
    source_units = _need(registration, "source_units", str, where)
    target_units = _need(registration, "target_units", str, where)
    if geometry["status"] == "registered_metric":
        if target_units != "m" or geometry.get("units") != "m":
            raise ManifestError(f"{where}: registered_metric transforms must use meters")
        if geometry.get("coordinate_frame_id") != target_frame:
            raise ManifestError(f"geometry.coordinate_frame_id must match {where}.target_frame")
        provenance = _need(geometry, "scale_provenance", dict, "geometry")
        if _need(provenance, "metric", bool, "geometry.scale_provenance") is not True:
            raise ManifestError(
                "geometry.scale_provenance.metric must be true for registered_metric"
            )
    else:
        if target_units != "relative_depth" or geometry.get("units") != "relative_depth":
            raise ManifestError(f"{where}: registered_relative must remain in relative_depth units")
        if target_units == "m":
            raise ManifestError(f"{where}: registered_relative cannot claim metric units")
    scene_transforms = _need(registration, "scene_transforms", list, where)
    if not scene_transforms:
        raise ManifestError(f"{where}.scene_transforms: expected at least one transform")
    seen_shots = set()
    for i, item in enumerate(scene_transforms):
        item_where = f"{where}.scene_transforms[{i}]"
        shot = _need(item, "shot", str, item_where)
        if shot in seen_shots:
            raise ManifestError(f"{item_where}.shot: duplicate scene transform")
        seen_shots.add(shot)
        _need(item, "source_frame", str, item_where)
        if _need(item, "target_frame", str, item_where) != target_frame:
            raise ManifestError(f"{item_where}.target_frame: inconsistent target frame")
        if _need(item, "source_units", str, item_where) != source_units:
            raise ManifestError(f"{item_where}.source_units: inconsistent source units")
        if _need(item, "target_units", str, item_where) != target_units:
            raise ManifestError(f"{item_where}.target_units: inconsistent target units")
        scale = _need(item, "scale", float, item_where)
        if scale <= 0:
            raise ManifestError(f"{item_where}.scale: must be positive")
        matrix = _need(item, "matrix_4x4", list, item_where)
        if len(matrix) != 4 or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in row
            )
            for row in matrix
        ):
            raise ManifestError(f"{item_where}.matrix_4x4: expected a finite 4x4 matrix")
        if not all(
            abs(a - b) <= 1e-8 for a, b in zip(matrix[3], [0.0, 0.0, 0.0, 1.0], strict=True)
        ):
            raise ManifestError(f"{item_where}.matrix_4x4: expected homogeneous bottom row")
        linear = [[matrix[row][column] / scale for column in range(3)] for row in range(3)]
        transpose_product = [
            [sum(linear[k][i] * linear[k][j] for k in range(3)) for j in range(3)] for i in range(3)
        ]
        determinant = (
            linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
            - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
            + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
        )
        if (
            any(
                abs(transpose_product[row][column] - (1.0 if row == column else 0.0)) > 2e-3
                for row in range(3)
                for column in range(3)
            )
            or abs(determinant - 1.0) > 2e-3
        ):
            raise ManifestError(
                f"{item_where}.matrix_4x4: scale-normalized transform must be proper"
            )
    camera_transforms = _need(registration, "camera_transforms", list, where)
    if geometry["status"] == "registered_relative" and camera_transforms:
        raise ManifestError(
            f"{where}.camera_transforms: relative registration cannot map metric joints"
        )
    camera_ids = set()
    for i, item in enumerate(camera_transforms):
        item_where = f"{where}.camera_transforms[{i}]"
        frame_id = _need(item, "frame_id", int, item_where)
        if frame_id not in frame_ids:
            raise ManifestError(f"{item_where}.frame_id: must reference a processed source frame")
        if frame_id in camera_ids:
            raise ManifestError(f"{item_where}.frame_id: duplicate camera transform")
        camera_ids.add(frame_id)
        _need(item, "shot", str, item_where)
        _need(item, "source_frame", str, item_where)
        if _need(item, "target_frame", str, item_where) != target_frame:
            raise ManifestError(f"{item_where}.target_frame: inconsistent target frame")
        if _need(item, "direction", str, item_where) != "nim_camera_m_to_metric_world_m":
            raise ManifestError(f"{item_where}.direction: unsupported camera transform direction")
        if _need(item, "units", str, item_where) != target_units:
            raise ManifestError(f"{item_where}.units: must match registration target units")
        if _need(item, "root_translation_applied", bool, item_where):
            raise ManifestError(f"{item_where}.root_translation_applied must be false")
        if _need(item, "stage_transform_applied", bool, item_where):
            raise ManifestError(f"{item_where}.stage_transform_applied must be false")
        matrix = _need(item, "matrix_4x4", list, item_where)
        if len(matrix) != 4 or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in row
            )
            for row in matrix
        ):
            raise ManifestError(f"{item_where}.matrix_4x4: expected a finite 4x4 matrix")
        if not all(
            abs(a - b) <= 1e-8 for a, b in zip(matrix[3], [0.0, 0.0, 0.0, 1.0], strict=True)
        ):
            raise ManifestError(f"{item_where}.matrix_4x4: expected homogeneous bottom row")
        rotation = [row[:3] for row in matrix[:3]]
        product = [
            [sum(rotation[k][i] * rotation[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if (
            any(
                abs(product[row][column] - (1.0 if row == column else 0.0)) > 2e-3
                for row in range(3)
                for column in range(3)
            )
            or abs(determinant - 1.0) > 2e-3
        ):
            raise ManifestError(f"{item_where}.matrix_4x4: rotation must be proper orthonormal")
    fit_anchor_ids = _need(registration, "fit_anchor_ids", list, where)
    check_anchor_ids = _need(registration, "check_anchor_ids", list, where)
    check_dimension_ids = _need(registration, "check_dimension_ids", list, where)
    for key, values in (
        ("fit_anchor_ids", fit_anchor_ids),
        ("check_anchor_ids", check_anchor_ids),
        ("check_dimension_ids", check_dimension_ids),
    ):
        if any(not isinstance(value, str) or not value for value in values):
            raise ManifestError(f"{where}.{key}: expected non-empty string IDs")
        if len(values) != len(set(values)):
            raise ManifestError(f"{where}.{key}: duplicate ID")
    if set(fit_anchor_ids) & set(check_anchor_ids):
        raise ManifestError(f"{where}: fit and check anchor IDs must be independent")
    evaluation = _need(geometry, "evaluation", dict, "geometry")
    evaluation_frames = _need(evaluation, "frames", list, "geometry.evaluation")
    evaluation_dimensions = _need(evaluation, "dimensions", list, "geometry.evaluation")
    if {item["id"] for item in evaluation_dimensions} != set(check_dimension_ids):
        raise ManifestError(
            f"{where}.check_dimension_ids must match withheld evaluation dimensions"
        )
    for frame in evaluation_frames:
        if frame["registered"] and not {point["id"] for point in frame["control_points"]} <= set(
            check_anchor_ids
        ):
            raise ManifestError(
                "geometry.evaluation control point IDs must be independent registration check anchors"
            )
    checks = _need(registration, "checks", list, where)
    if not checks:
        raise ManifestError(f"{where}.checks: at least one independent check is required")
    check_ids = set()
    for i, item in enumerate(checks):
        item_where = f"{where}.checks[{i}]"
        identifier = _need(item, "id", str, item_where)
        if identifier in check_ids:
            raise ManifestError(f"{item_where}.id: duplicate registration check")
        check_ids.add(identifier)
        _need(item, "kind", str, item_where)
        if _need(item, "error_m", float, item_where) < 0:
            raise ManifestError(f"{item_where}.error_m: must be non-negative")
        if _need(item, "uncertainty_m", float, item_where) <= 0:
            raise ManifestError(f"{item_where}.uncertainty_m: must be positive")
        if _need(item, "passed", bool, item_where) is not True:
            raise ManifestError(f"{item_where}.passed: metric registration requires passing checks")
    if check_ids != set(fit_anchor_ids) | set(check_anchor_ids) | set(check_dimension_ids):
        raise ManifestError(
            f"{where}.checks IDs must match the fit, independent check and dimension IDs"
        )


def _matrix(obj, key: str, rows: int, columns: int, where: str) -> None:
    matrix = _need(obj, key, list, where)
    if len(matrix) != rows:
        raise ManifestError(f"{where}.{key}: expected {rows} rows")
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != columns:
            raise ManifestError(f"{where}.{key}[{i}]: expected {columns} values")
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ManifestError(f"{where}.{key}[{i}]: values must be finite numbers")


def validate_manifest(doc: dict) -> None:
    """Structural and semantic validation of a parsed manifest (assets not touched)."""
    schema = doc.get("schema")
    if schema != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema {schema!r}; this version reads {SCHEMA_VERSION!r}"
        )
    run = _need(doc, "run", dict, "manifest")
    _one_of(_need(run, "status", str, "run"), RUN_STATUSES, "run.status")
    if "perception_status" in run:
        _one_of(
            _need(run, "perception_status", str, "run"),
            ("complete", "partial", "failed"),
            "run.perception_status",
        )
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
    if source["frame_count"] <= 0:
        raise ManifestError("source.frame_count: must be positive")
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
    vocabulary = _validate_vocabulary(config)
    _need(doc, "assets", list, "manifest")
    _validate_shot_times(doc, tuple(rate), source["frame_count"])
    validate_content(doc, width=width, height=height, frame_ids=ids, vocabulary=vocabulary)
    perception_status = run.get("perception_status", run["status"])
    if run["status"] == "complete" and perception_status != "complete":
        raise ManifestError("run.perception_status: complete run requires complete perception")
    if (
        run["status"] == "partial"
        and perception_status == "complete"
        and doc["geometry"]["status"]
        not in ("not_requested", "unavailable", "relative", "relative-camera-frame")
    ):
        raise ManifestError(
            "run.perception_status: complete perception may be partial only when geometry is unregistered"
        )
    _validate_relations(doc)
    _validate_poses(doc)


def _validate_vocabulary(config: dict) -> dict:
    """``{label: family}`` from ``config.label_vocabulary``, which every manifest records."""
    vocabulary = {}
    for i, entry in enumerate(_need(config, "label_vocabulary", list, "config")):
        where = f"config.label_vocabulary[{i}]"
        label = _need(entry, "label", str, where)
        _one_of(
            _need(entry, "family", str, where), ("surface", "object", "vehicle"), f"{where}.family"
        )
        _one_of(_need(entry, "source", str, where), ("preset", "user"), f"{where}.source")
        if label in vocabulary:
            raise ManifestError(f"{where}: duplicate label {label!r}")
        vocabulary[label] = entry["family"]
    return vocabulary


def _validate_relations(doc: dict) -> None:
    ids = {e["id"] for e in doc["entities"]}
    for i, rel in enumerate(_need(doc, "relations", list, "manifest")):
        where = f"relations[{i}]"
        _one_of(_need(rel, "type", str, where), ("contained_in",), f"{where}.type")
        for key in ("child", "parent"):
            if _need(rel, key, str, where) not in ids:
                raise ManifestError(f"{where}.{key}: {rel[key]!r} is not an entity")
        frames = _need(rel, "frames", list, where)
        if not frames or not all(isinstance(f, int) and not isinstance(f, bool) for f in frames):
            raise ManifestError(f"{where}.frames: expected a non-empty list of frame ids")
        _one_of(_need(rel, "evidence", str, where), ("bbox",), f"{where}.evidence")


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
    association = _one_of(
        _need(block, "association", str, "poses"), POSE_ASSOCIATIONS, "poses.association"
    )
    _need(block, "frame_count", int, "poses")
    if "file_sha256" in block and (
        not isinstance(block["file_sha256"], str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", block["file_sha256"]) is None
    ):
        raise ManifestError("poses.file_sha256: expected 64 hexadecimal characters")
    if association == "hash-verified" and "file_sha256" not in block:
        raise ManifestError("poses.file_sha256: required for hash-verified pose association")
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
    geometry = doc.get("geometry") or {}
    if geometry.get("status") == "relative-camera-frame":
        for i, frame in enumerate(geometry.get("frames", [])):
            rel = _need(frame, "depth_asset", str, f"geometry.frames[{i}]")
            if rel not in declared:
                raise ManifestError(
                    f"geometry.frames[{i}].depth_asset: asset {rel!r} is not listed"
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
