# SPDX-License-Identifier: MIT
"""Label vocabulary, label records, containment and mask honesty, through the public CLI."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from skeleton_maker import cli, environment, labels
from skeleton_maker.workers import grounded_sam2_da3 as worker

from .env_fakes import FakeBackend
from .test_environment import _make_video, _run_failing, _snapshot

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required",
)

IDENTITY = {"model": "fake", "preprocessing": "p1"}


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_video(path, frames=30)
    return path


def _install(monkeypatch, **kwargs):
    backend = FakeBackend(identity=IDENTITY, **kwargs)
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    return backend


def _scan(clip, tmp_path, *extra, out="environment.json"):
    argv = [str(clip), "--out", str(tmp_path / out), "--backend", "fake"]
    argv += ["--cache-dir", str(tmp_path / "cache"), *extra]
    return cli.main(["environment", *argv])


def _manifest(tmp_path, out="environment.json"):
    return json.loads((tmp_path / out).read_text())


def _vocabulary(manifest):
    return {v["label"]: v["family"] for v in manifest["config"]["label_vocabulary"]}


def test_the_default_vocabulary_covers_every_named_category(clip, monkeypatch, tmp_path):
    backend = _install(monkeypatch)

    assert _scan(clip, tmp_path) == 0

    vocabulary = _vocabulary(_manifest(tmp_path))
    for label in ("wall", "floor", "ceiling", "door", "window", "stairs", "ground", "road"):
        assert vocabulary[label] == "surface"
    for label in ("chair", "table", "sofa", "bed", "desk", "shelf", "cabinet"):  # furniture
        assert vocabulary[label] == "object"
    for label in ("hammer", "drill", "ladder", "bucket", "box", "bag"):  # tools and containers
        assert vocabulary[label] == "object"
    for label in ("bottle", "cup", "bowl", "phone", "laptop"):  # handheld
        assert vocabulary[label] == "object"
    for label in ("car", "truck", "bus", "bicycle", "motorcycle"):
        assert vocabulary[label] == "vehicle"
    assert "person" not in vocabulary, "people belong to the skeletons"
    assert (
        backend.requests[0]["label_vocabulary"] == _manifest(tmp_path)["config"]["label_vocabulary"]
    )
    assert {v["source"] for v in _manifest(tmp_path)["config"]["label_vocabulary"]} == {"preset"}


def test_extra_classes_are_recorded_and_extend_the_preset(clip, monkeypatch, tmp_path):
    backend = _install(monkeypatch)

    assert (
        _scan(clip, tmp_path, "--classes", "Fire Extinguisher, whiteboard ,forklift=vehicle,chair")
        == 0
    )

    manifest = _manifest(tmp_path)
    assert manifest["config"]["requested_labels"] == ["fire extinguisher", "whiteboard", "forklift"]
    vocabulary = {v["label"]: v for v in manifest["config"]["label_vocabulary"]}
    assert vocabulary["fire extinguisher"] == {
        "label": "fire extinguisher",
        "family": "object",
        "source": "user",
    }
    assert vocabulary["forklift"]["family"] == "vehicle"
    assert vocabulary["chair"]["source"] == "preset", "a preset label is not duplicated"
    assert len([v for v in manifest["config"]["label_vocabulary"] if v["label"] == "chair"]) == 1
    assert backend.requests[0]["requested_labels"] == [
        "fire extinguisher",
        "whiteboard",
        "forklift",
    ]


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        ("forklift=machine", "family 'machine'"),
        ("a,,b", "empty label"),
        ("person", "skeletons"),
        ("x=object=vehicle", "one family"),
    ],
)
def test_bad_classes_are_rejected_before_inference(
    clip, monkeypatch, tmp_path, capsys, classes, message
):
    backend = _install(monkeypatch)
    before = _snapshot(tmp_path)

    err = _run_failing(
        [
            str(clip),
            "--out",
            str(tmp_path / "e.json"),
            "--backend",
            "fake",
            "--classes",
            classes,
        ],
        capsys,
    )

    assert message in err
    assert backend.requests == []
    assert _snapshot(tmp_path) == before


def test_changing_the_classes_invalidates_the_cache(clip, monkeypatch, tmp_path):
    backend = _install(monkeypatch)
    _scan(clip, tmp_path, "--classes", "whiteboard")
    _scan(clip, tmp_path, "--classes", "whiteboard", out="same.json")
    assert len(backend.requests) == 1

    _scan(clip, tmp_path, "--classes", "whiteboard,forklift=vehicle", out="other.json")
    _scan(clip, tmp_path, "--classes", "whiteboard=surface", out="family.json")

    assert len(backend.requests) == 3


def _entity(eid, label, family="object", status="matched", requested=None, **labels_extra):
    return {
        "id": eid,
        "shot": "shot-0",
        "family": family,
        "motion": "unknown",
        "labels": {
            "requested": label if requested is None else requested,
            "native": label,
            "normalized": label,
            "status": status,
            **labels_extra,
        },
    }


def _obs(oid, eid, frame, bbox, mask=None, visibility="visible"):
    return {
        "id": oid,
        "entity": eid,
        "frame_id": frame,
        "bbox": bbox,
        "mask": mask,
        "score": 0.8,
        "score_meaning": "test",
        "visibility": visibility,
    }


def _mask(response_dir, name, rect, size=(64, 48)):
    path = response_dir / "masks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0]), np.uint8)
    x0, y0, x1, y1 = rect
    image[y0:y1, x0:x1] = 255
    cv2.imwrite(str(path), image)
    return {"asset": f"masks/{name}"}


class _Scene(FakeBackend):
    """A fake whose response is replaced by a scripted scene."""

    def __init__(self, build, **kwargs):
        super().__init__(identity=IDENTITY, **kwargs)
        self.build = build

    def run(self, request, assets_dir):
        response = super().run(request, assets_dir)
        entities, observations = self.build(request, assets_dir)
        response["entities"], response["observations"] = entities, observations
        return response


def _scene(monkeypatch, build):
    backend = _Scene(build)
    monkeypatch.setitem(environment.BACKENDS, "fake", backend)
    return backend


def test_two_chairs_in_one_frame_are_two_instances(clip, monkeypatch, tmp_path):
    def build(request, assets):
        ents = [_entity("shot-0/chair-f0-0", "chair"), _entity("shot-0/chair-f0-1", "chair")]
        obs = [
            _obs(
                "o0", "shot-0/chair-f0-0", 0, [2, 2, 12, 20], _mask(assets, "a.png", (2, 2, 12, 20))
            ),
            _obs(
                "o1",
                "shot-0/chair-f0-1",
                0,
                [30, 2, 44, 20],
                _mask(assets, "b.png", (30, 2, 44, 20)),
            ),
        ]
        return ents, obs

    _scene(monkeypatch, build)

    assert _scan(clip, tmp_path) == 0

    chairs = [e for e in _manifest(tmp_path)["entities"] if e["labels"]["normalized"] == "chair"]
    assert len(chairs) == 2
    assert len({e["id"] for e in chairs}) == 2


def _wall_and_door(door_box, door_rect):
    def build(request, assets):
        ents = [
            _entity("shot-0/surface-wall", "wall", "surface"),
            _entity("shot-0/surface-door", "door", "surface"),
        ]
        obs = [
            _obs(
                "o0",
                "shot-0/surface-wall",
                0,
                [0, 0, 60, 40],
                _mask(assets, "wall.png", (0, 0, 60, 40)),
            ),
            _obs("o1", "shot-0/surface-door", 0, door_box, _mask(assets, "door.png", door_rect)),
        ]
        return ents, obs

    return build


def test_a_door_inside_a_wall_keeps_both_entities_and_records_containment(
    clip, monkeypatch, tmp_path
):
    _scene(monkeypatch, _wall_and_door([20, 10, 30, 40], (20, 10, 30, 40)))

    assert _scan(clip, tmp_path) == 0

    manifest = _manifest(tmp_path)
    assert {e["labels"]["normalized"] for e in manifest["entities"]} >= {"wall", "door"}
    assert manifest["relations"] == [
        {
            "type": "contained_in",
            "child": "shot-0/surface-door",
            "parent": "shot-0/surface-wall",
            "frames": [0],
            "evidence": "bbox",
        }
    ]
    masks = {o["entity"]: o["mask"]["asset"] for o in manifest["observations"]}
    assert masks["shot-0/surface-wall"] != masks["shot-0/surface-door"]


def test_a_door_outside_every_wall_is_not_contained(clip, monkeypatch, tmp_path):
    def build(request, assets):
        ents, obs = _wall_and_door([50, 10, 64, 40], (50, 10, 63, 40))(request, assets)
        return ents, obs

    _scene(monkeypatch, build)

    assert _scan(clip, tmp_path) == 0

    assert _manifest(tmp_path)["relations"] == []


def test_ambiguous_and_unknown_labels_stay_explicit(clip, monkeypatch, tmp_path):
    def build(request, assets):
        amb = _entity(
            "shot-0/thing-f0-0",
            "oven refrigerator",
            status="ambiguous",
            requested=None,
            candidates=["oven", "refrigerator"],
        )
        amb["labels"]["normalized"] = "unknown"
        unk = _entity("shot-0/thing-f0-1", "widget", status="unknown", requested=None)
        unk["labels"]["normalized"] = "unknown"
        obs = [
            _obs("o0", amb["id"], 0, [2, 2, 12, 20], _mask(assets, "a.png", (2, 2, 12, 20))),
            _obs("o1", unk["id"], 0, [30, 2, 44, 20], _mask(assets, "b.png", (30, 2, 44, 20))),
        ]
        return [amb, unk], obs

    _scene(monkeypatch, build)

    assert _scan(clip, tmp_path) == 0

    records = {e["id"]: e["labels"] for e in _manifest(tmp_path)["entities"]}
    assert records["shot-0/thing-f0-0"]["status"] == "ambiguous"
    assert records["shot-0/thing-f0-0"]["candidates"] == ["oven", "refrigerator"]
    assert records["shot-0/thing-f0-1"]["normalized"] == "unknown"


def _bad_entity(mutate):
    def build(request, assets):
        ent = _entity("shot-0/chair-f0-0", "chair")
        mutate(ent)
        obs = [_obs("o0", ent["id"], 0, [2, 2, 12, 20], _mask(assets, "a.png", (2, 2, 12, 20)))]
        return [ent], obs

    return build


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e["labels"].update(normalized="armchair"), "not in the label vocabulary"),
        (lambda e: e.update(family="vehicle"), "family"),
        (lambda e: e["labels"].update(status="maybe"), "labels.status"),
        (lambda e: e["labels"].update(status="ambiguous"), "unknown"),
        (lambda e: e["labels"].pop("requested"), "requested"),
        (lambda e: e["labels"].pop("native"), "native"),
    ],
)
def test_label_records_are_validated_against_the_vocabulary(
    clip, monkeypatch, tmp_path, capsys, mutate, message
):
    _scene(monkeypatch, _bad_entity(mutate))
    before = _snapshot(tmp_path)

    err = _run_failing(
        [
            str(clip),
            "--out",
            str(tmp_path / "e.json"),
            "--backend",
            "fake",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        capsys,
    )

    assert message in err
    assert not (tmp_path / "e.json").exists()
    assert not [p for p in _snapshot(tmp_path) if p not in before]


def _bad_mask(kind):
    def build(request, assets):
        ents = [_entity("shot-0/surface-ceiling", "ceiling", "surface")]
        rect = (0, 0, 64, 10)
        mask = _mask(assets, "c.png", rect)
        bbox = [0, 0, 64, 10]
        visibility = "visible"
        if kind == "missing":
            mask = None
        elif kind == "empty":
            mask = _mask(assets, "c.png", (0, 0, 0, 0))
        elif kind == "wrong_size":
            mask = _mask(assets, "c.png", (0, 0, 4, 4), size=(8, 8))
        elif kind == "outside_bbox":
            bbox = [0, 0, 64, 5]
        elif kind == "one_pixel_outside_bbox":
            rect = (10, 5, 11, 6)
            mask = _mask(assets, "c.png", rect)
            bbox = [0, 0, 10, 10]
        elif kind == "absent_with_mask":
            visibility = "absent"
        return ents, [_obs("o0", ents[0]["id"], 0, bbox, mask, visibility)]

    return build


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "visible observation needs a mask"),
        ("empty", "mask is empty"),
        ("wrong_size", "does not match the 64x48 source frame"),
        ("outside_bbox", "pixels outside its bbox"),
        ("one_pixel_outside_bbox", "pixels outside its bbox"),
        ("absent_with_mask", "absent observation must not carry a mask"),
    ],
)
def test_masks_must_show_only_what_is_visible(clip, monkeypatch, tmp_path, capsys, kind, message):
    _scene(monkeypatch, _bad_mask(kind))

    err = _run_failing(
        [
            str(clip),
            "--out",
            str(tmp_path / "e.json"),
            "--backend",
            "fake",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        capsys,
    )

    assert message in err
    assert not (tmp_path / "e.json").exists()


def test_worker_phrase_resolution_is_explicit_about_ambiguity():
    prompt = ["oven", "refrigerator", "cup"]

    assert worker._resolve_phrase("oven", prompt) == ("matched", "oven", [])
    assert worker._resolve_phrase("couch", ["sofa", "chair"]) == ("matched", "sofa", [])
    assert worker._resolve_phrase("oven refrigerator", prompt) == (
        "ambiguous",
        "unknown",
        ["oven", "refrigerator"],
    )
    assert worker._resolve_phrase("widget", prompt) == ("unknown", "unknown", [])
    assert worker._resolve_phrase("", prompt) == ("unknown", "unknown", [])


def test_the_preset_is_documented_with_its_limits():
    text = (Path(labels.__file__).parent.parent / "docs" / "environment.md").read_text()
    assert "not a promise" in text or "not guaranteed" in text
    for entry in labels.PRESET:
        assert f"`{entry.label}`" in text, f"{entry.label} missing from the documented preset"


def test_the_worker_vocabulary_rules_equal_the_base_package_rules():
    # the worker runs in its own interpreter and cannot import the package, so it keeps copies
    assert worker.ALIASES == labels.ALIASES
    assert worker.PROMPTED_ALONE == labels.PROMPTED_ALONE
