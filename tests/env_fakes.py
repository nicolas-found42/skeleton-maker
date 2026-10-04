# SPDX-License-Identifier: MIT
"""Deterministic stand-in for the environment inference boundary (tests only).

The real perception and geometry workers are heavyweight and live outside the base
package; CI substitutes this fake through the backend registry and nothing else.
"""

import copy

from skeleton_maker import environment

CONTRACT = environment.BACKEND_CONTRACT


class FakeBackend:
    """Returns a fixed, schema-valid response for whatever frames it is asked about."""

    name = "fake"

    def __init__(
        self,
        *,
        devices=("cpu",),
        geometry=False,
        mutate=None,
        raises=None,
        identity=None,
        frame_limit=None,
    ):
        self.devices = list(devices)
        self.geometry = geometry
        self.mutate = mutate
        self.raises = raises
        self.identity = identity
        self.frame_limit = frame_limit
        self.requests = []

    def available_devices(self):
        return list(self.devices)

    def supports_geometry(self):
        return self.geometry

    def max_frames(self):
        return self.frame_limit

    def cache_identity(self):
        return None if self.identity is None else dict(self.identity)

    def run(self, request, assets_dir):
        self.requests.append(copy.deepcopy(request))
        if self.raises is not None:
            raise self.raises
        frames = request["frames"]
        first, last = frames[0]["frame_id"], frames[-1]["frame_id"]

        def shot_of(frame_id):
            return next(
                sh["id"]
                for sh in request["shots"]
                if sh["first_frame"] <= frame_id <= sh["last_frame"]
            )

        floor_shot, chair_shot = shot_of(first), shot_of(last)
        (assets_dir / "masks").mkdir(parents=True, exist_ok=True)
        (assets_dir / "masks" / "floor-0.png").write_bytes(b"\x89PNG fake floor mask")
        response = {
            "contract": CONTRACT,
            "status": "complete",
            "backend": {
                "name": self.name,
                "version": "0.0-test",
                "checkpoints": [{"name": "fake-weights", "sha256": "0" * 64, "license": "none"}],
            },
            "device": request["device"],
            "entities": [
                {
                    "id": f"{floor_shot}/floor-1",
                    "shot": floor_shot,
                    "family": "surface",
                    "labels": {"native": "floor", "normalized": "floor"},
                    "motion": "static",
                },
                {
                    "id": f"{chair_shot}/chair-1",
                    "shot": chair_shot,
                    "family": "object",
                    "labels": {"native": "chair", "normalized": "chair"},
                    "motion": "unknown",
                },
            ],
            "observations": [
                {
                    "id": "obs-0",
                    "entity": f"{floor_shot}/floor-1",
                    "frame_id": first,
                    "bbox": [0.0, 20.0, 63.0, 47.0],
                    "mask": {"asset": "masks/floor-0.png"},
                    "score": 0.9,
                    "score_meaning": "model confidence, uncalibrated",
                    "visibility": "visible",
                },
                {
                    "id": "obs-1",
                    "entity": f"{chair_shot}/chair-1",
                    "frame_id": last,
                    "bbox": [10.0, 10.0, 20.0, 30.0],
                    "mask": None,
                    "score": 0.6,
                    "score_meaning": "model confidence, uncalibrated",
                    "visibility": "occluded",
                },
            ],
            "geometry": {"status": "unavailable", "reason": "fake backend has no geometry"},
        }
        if self.mutate is not None:
            self.mutate(response)
        return response
