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
        self.entities = {}
        self.observations = []
        self.scanned = list(self.frames)
        self.pred_geometry = {"status": "unavailable", "reason": "test"}
        self.run_status = "complete"
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
        self.entities.setdefault(
            eid,
            {
                "id": eid,
                "shot": "shot-0",
                "family": family,
                "motion": "unknown",
                "labels": {"native": label, "normalized": label},
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
        self.pred_geometry = {
            "status": status,
            "reason": "test",
            "evaluation": {"frames": list(frames), "dimensions": list(dimensions)},
        }

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
        if self.geometry:
            doc["geometry"] = self.geometry
        (self.corpus.annotations / f"{self.name}.json").write_text(json.dumps(doc))

    def write_prediction(self):
        base = self.corpus.predictions / f"{self.name}.assets"
        base.mkdir(parents=True, exist_ok=True)
        assets = []
        for path in sorted(base.rglob("*.png")):
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
            "run": {"status": self.run_status, "created_at": "2026-01-01T00:00:00+00:00"},
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
            "shots": [{"id": "shot-0", "first_frame": min(self.scanned), "last_frame": last}],
            "entities": list(self.entities.values()),
            "observations": self.observations,
            "geometry": {"mode": "auto", **self.pred_geometry},
            "backend": {"name": "test", "version": "0", "checkpoints": [], "device": "cpu"},
            "config": {
                "requested_labels": [],
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
