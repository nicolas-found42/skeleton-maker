# SPDX-License-Identifier: MIT
"""Tiny hand-computable corpora for the accuracy scorer (tests only).

Everything is 10x10 pixels so every IoU, area and rate in a test can be worked out by
hand. Rectangles are ``(row0, row1, col0, col1)``, half-open, in image coordinates.
"""

import hashlib
import json

import cv2
import numpy as np

from skeleton_maker import envmanifest
from skeleton_maker.envannotations import ANNOTATION_SCHEMA

W = H = 10


def _mask(rect):
    m = np.zeros((H, W), np.uint8)
    r0, r1, c0, c1 = rect
    m[r0:r1, c0:c1] = 255
    return m


class Clip:
    def __init__(self, corpus, name, split, frames, shots, tags, fps):
        self.corpus, self.name, self.split = corpus, name, split
        self.frames = list(frames)
        self.shots = shots
        self.tags = tags
        self.fps = fps
        self.sha = hashlib.sha256(name.encode()).hexdigest()
        self.gt = {
            f: {"surfaces": [], "negative_classes": [], "instances": [], "ignore": []}
            for f in self.frames
        }
        self.intervals = []
        self.aliases = {}
        self.geometry = None
        self.review = {"second_reviewer": "reviewer-b", "disagreements_resolved": True}
        self.scope = None
        self.provenance = None
        self.entities = {}
        self.vocab = {}
        self.observations = []
        self.scanned = list(self.frames)
        self.pred_geometry: dict = {"status": "unavailable", "reason": "test"}
        self.run_status = "complete"
        self.perception_status = "complete"
        self._masks = 0

    # --- ground truth -------------------------------------------------------
    def _gt_mask(self, rect):
        self._masks += 1
        rel = f"masks/{self.name}-{self._masks}.png"
        path = self.corpus.annotations / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), _mask(rect))
        return rel

    def gt_surface(self, frame, cls, rect):
        self.gt[frame]["surfaces"].append({"class": cls, "mask": self._gt_mask(rect)})

    def gt_negative(self, frame, cls):
        self.gt[frame]["negative_classes"].append(cls)

    def gt_instance(self, frame, iid, family, cls, rect, visibility="visible"):
        inst = {"id": iid, "family": family, "class": cls, "visibility": visibility}
        if visibility != "absent":
            inst["mask"] = self._gt_mask(rect)
        self.gt[frame]["instances"].append(inst)

    def gt_ignore(self, frame, rect):
        self.gt[frame]["ignore"].append({"mask": self._gt_mask(rect)})

    def interval(self, first, last):
        self.intervals.append({"first_frame": first, "last_frame": last})

    def gt_geometry(self, eligible, control_points=(), dimensions=()):
        self.geometry = {
            "eligible_frames": list(eligible),
            "control_points": list(control_points),
            "dimensions": list(dimensions),
        }

    # --- predictions --------------------------------------------------------
    def _entity(self, eid, family, label):
        self.vocab[label] = family
        self.entities.setdefault(
            eid,
            {
                "id": eid,
                "shot": "shot-0",
                "family": family,
                "motion": "unknown",
                "labels": {
                    "requested": label,
                    "native": label,
                    "normalized": label,
                    "status": "matched",
                },
            },
        )

    def pred(self, frame, eid, family, label, rect, visibility="visible", score=0.9):
        self._entity(eid, family, label)
        self._masks += 1
        rel = f"masks/{self.name}-p{self._masks}.png"
        path = self.corpus.predictions / f"{self.name}.assets" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), _mask(rect))
        self.observations.append(
            {
                "id": f"obs-{len(self.observations)}",
                "entity": eid,
                "frame_id": frame,
                "bbox": [0.0, 0.0, float(W), float(H)],
                "mask": {"asset": rel},
                "score": score,
                "score_meaning": "test",
                "visibility": visibility,
            }
        )

    def pred_surface(self, frame, cls, rect):
        self.pred(frame, f"surface-{cls}", "surface", cls, rect)

    def pred_registration(self, status, frames=(), dimensions=()):
        geometry = {
            "status": status,
            "reason": "test",
            "evaluation": {
                "frames": list(frames),
                "dimensions": [{**item, "units": item.get("units", "m")} for item in dimensions],
            },
        }
        if status in ("registered_relative", "registered_metric"):
            metric = status == "registered_metric"
            units = "m" if metric else "relative_depth"
            target_frame = "fixture-world"
            identity = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            checked_anchor_ids = sorted(
                {point["id"] for frame in frames for point in frame.get("control_points", [])}
            )
            checked_dimension_ids = sorted(item["id"] for item in dimensions)
            if not checked_anchor_ids and not checked_dimension_ids:
                checked_anchor_ids = ["fixture-check"]
            geometry.update(
                {
                    "units": units,
                    "coordinate_convention": "test camera convention",
                    "scale_provenance": {
                        "kind": "fixture",
                        "metric": metric,
                        "fit_anchor_ids": ["fixture-fit"],
                        "check_anchor_ids": ["fixture-check"],
                        "fit_dimension_ids": [],
                    },
                    "static_fusion_entities": [],
                    "excluded_dynamic_entities": [],
                    "excluded_skeletons": [],
                    "frames": [
                        {
                            "frame_id": frame_id,
                            "shot": "shot-0",
                            "depth_asset": f"geometry/depth/{frame_id}.npy",
                            "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "extrinsics_w2c": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                            ],
                            "depth_pixel_space": "processed_frame_pixels",
                            "preprocessing": {
                                "process_res": 10,
                                "process_res_method": "upper_bound_resize",
                                "source_size": [W, H],
                                "processed_size": [W, H],
                                "undistorted_source_to_processed": [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                "processed_to_undistorted_source": [
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ],
                                "depth_resampling": "none; native DA3 processed depth grid",
                                "intrinsics_pixel_space": "undistorted_source_frame_pixels",
                                "crop": None,
                                "lens_transform": {"model": "none", "applied": False},
                            },
                        }
                        for frame_id in self.scanned
                    ],
                }
            )
            registration = {
                "schema": "skeleton-maker.geometry-registration/1",
                "source_frame": "fixture-scene",
                "target_frame": target_frame,
                "direction": "source_to_target",
                "source_units": "relative_depth",
                "target_units": units,
                "scene_transforms": [
                    {
                        "shot": "shot-0",
                        "source_frame": "fixture-scene",
                        "target_frame": target_frame,
                        "source_to_target": "fixture transform",
                        "source_units": "relative_depth",
                        "target_units": units,
                        "matrix_4x4": identity,
                        "scale": 1.0,
                    }
                ],
                "camera_transforms": [
                    {
                        "frame_id": f["frame_id"],
                        "shot": "shot-0",
                        "source_frame": f"nim-camera:{f['frame_id']}",
                        "target_frame": target_frame,
                        "direction": "nim_camera_m_to_metric_world_m",
                        "units": "m",
                        "matrix_4x4": identity,
                        "root_translation_applied": False,
                        "stage_transform_applied": False,
                    }
                    for f in ({"frame_id": frame_id} for frame_id in self.scanned)
                    if metric
                ],
                "fit_anchor_ids": ["fixture-fit"],
                "check_anchor_ids": checked_anchor_ids,
                "check_dimension_ids": checked_dimension_ids,
                "checks": [
                    {
                        "id": identifier,
                        "kind": (
                            "fit_anchor_residual"
                            if identifier == "fixture-fit"
                            else "withheld_dimension"
                            if identifier in checked_dimension_ids
                            else "withheld_anchor_position"
                        ),
                        "error_m": 0.0,
                        "uncertainty_m": 0.01,
                        "passed": True,
                    }
                    for identifier in ["fixture-fit", *checked_anchor_ids, *checked_dimension_ids]
                ],
            }
            geometry["coordinate_frame_id"] = target_frame
            geometry["registration"] = registration
        self.pred_geometry = geometry

    # --- files --------------------------------------------------------------
    def write_annotation(self):
        doc = {
            "schema": ANNOTATION_SCHEMA,
            "clip": self.name,
            "source_sha256": self.sha,
            "split": self.split,
            "width": W,
            "height": H,
            "frame_rate": [self.fps, 1],
            "tags": self.tags,
            "aliases": self.aliases,
            "shots": [{"first_frame": a, "last_frame": b} for a, b in self.shots],
            "frames": [{"frame_id": f, **self.gt[f]} for f in self.frames],
            "tracking_intervals": self.intervals,
        }
        if self.review:
            doc["review"] = self.review
        if self.scope:
            doc["scope"] = self.scope
        if self.provenance:
            doc["provenance"] = self.provenance
        if self.geometry:
            doc["geometry"] = self.geometry
        (self.corpus.annotations / f"{self.name}.json").write_text(json.dumps(doc))

    def write_prediction(self):
        base = self.corpus.predictions / f"{self.name}.assets"
        base.mkdir(parents=True, exist_ok=True)
        for frame in self.pred_geometry.get("frames", []):
            path = base / frame["depth_asset"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                width, height = frame["preprocessing"]["processed_size"]
                np.save(path, np.ones((height, width), dtype=np.float32), allow_pickle=False)
        assets = []
        for path in sorted(path for path in base.rglob("*") if path.is_file()):
            from skeleton_maker.artifacts import sha256_file

            assets.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        last = max(self.scanned) if self.scanned else 0
        doc = {
            "schema": envmanifest.SCHEMA_VERSION,
            "run": {
                "status": self.run_status,
                "perception_status": self.perception_status,
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            "source": {
                "path": f"/clips/{self.name}.mp4",
                "sha256": self.sha,
                "width": W,
                "height": H,
                "frame_rate": [self.fps, 1],
                "frame_count": last + 1,
                "frame_count_source": "container",
                "duration_s": (last + 1) / self.fps,
            },
            "processed_frames": [
                {"frame_id": f, "time": [f, self.fps], "time_s": f / self.fps}
                for f in sorted(self.scanned)
            ],
            "poses": None,
            "frame_range": [min(self.scanned), last] if self.scanned else [0, 0],
            "shots": [
                {
                    "id": "shot-0",
                    "first_frame": 0,
                    "last_frame": last,
                    "first_time": [0, 1],
                    "last_time": [last, self.fps],
                }
            ],
            "entities": list(self.entities.values()),
            "observations": self.observations,
            "geometry": {"mode": "auto", **self.pred_geometry},
            "backend": {"name": "test", "version": "0", "checkpoints": [], "device": "cpu"},
            "relations": [],
            "config": {
                "requested_labels": [],
                "label_vocabulary": [
                    {"label": label, "family": family, "source": "preset"}
                    for label, family in sorted(self.vocab.items())
                ],
                "geometry_mode": "auto",
                "sample_fps": 2,
                "device": "cpu",
            },
            "assets": assets,
        }
        if self.run_status != "complete":
            doc["run"]["reason"] = "test"
        (self.corpus.predictions / f"{self.name}.json").write_text(envmanifest.dumps(doc))


class Corpus:
    def __init__(self, root):
        self.annotations = root / "labels"
        self.predictions = root / "runs"
        self.annotations.mkdir()
        self.predictions.mkdir()
        self.clips = []

    def clip(self, name="a", split="heldout", frames=range(1), shots=None, tags=(), fps=10):
        frames = list(frames)
        shots = shots or [(min(frames), max(frames))]
        clip = Clip(self, name, split, frames, shots, list(tags), fps)
        self.clips.append(clip)
        return clip

    def write(self):
        for clip in self.clips:
            clip.write_annotation()
            clip.write_prediction()
        return self
